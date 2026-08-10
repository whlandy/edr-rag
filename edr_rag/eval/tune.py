"""P1-6 基于评测集调节 hybrid 检索的 RRF 融合权重 (weight tuning)。

设计:
  - hybrid 策略的本质是「多路召回 + RRF 融合」(见 edr_rag/retrieval.py)。
    RRF 各路的相对权重直接影响融合排序质量。
  - 本模块在评测集上对权重做网格搜索：对每个候选权重配置，把权重注入一个
    新建的 RetrievalService 后跑一遍评测，取目标指标 (默认 recall@k) 最优的
    配置。
  - 优雅降级：可选 `rag` extras 未装（无 embedder）时，hybrid 只融合
    keyword + bm25 两路，权重网格自动退化为 2 维；评测集为空时直接返回
    默认 (None = 等权)，不报错。主链路始终可用。

典型用法:
  >>> from edr_rag.eval.tune import tune_rrf_weights
  >>> best = tune_rrf_weights(workdir, eval_set, metric="recall@k")
  >>> svc.set_fusion_weights(best.weights)  # 应用到检索服务
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from ..retrieval import RetrievalService
from .dataset import EvalSet
from .metrics import aggregate_metrics
from .runner import run_eval

#: 默认候选权重网格。每条为 [keyword, bm25] 权重；有 embedder 时在每条后补
#: vector 权重。值越大该路在 RRF 融合中的贡献越高。
DEFAULT_GRID: list[list[float]] = [
    [1.0, 1.0],
    [1.5, 1.0],
    [1.0, 1.5],
    [2.0, 1.0],
    [1.0, 2.0],
    [1.8, 1.2],
    [1.2, 1.8],
]


#: 缺省 holdout 比例 —— 用于报告「诚实分数」的留出集占比。
DEFAULT_HOLDOUT_RATIO = 0.3

#: 切分随机种子。固定种子保证同一评测集的切分可复现。
DEFAULT_SPLIT_SEED = 20260810


def split_eval_set(
    eval_set: EvalSet,
    holdout_ratio: float = DEFAULT_HOLDOUT_RATIO,
    seed: int = DEFAULT_SPLIT_SEED,
) -> tuple[EvalSet, EvalSet]:
    """把评测集切成 (train, holdout) 两份，**互不重叠** (P4)。

    调参只看 train，报告分数看 holdout —— 这是让「调参过拟合」变得可见的
    最小机制。P1-6 的实现在**同一份**评测集上做网格搜索并报告最优分数，
    那个分数天然是乐观的，无法回答「换一批查询还成立吗」。

    切分按 qid 的稳定哈希分桶，而不是按位置切片 —— 评测集追加新条目时，
    既有条目的归属不会漂移，历史分数仍可比。

    Args:
        eval_set: 待切分的评测集。
        holdout_ratio: 留出比例 (0~1)。<=0 或 >=1 时不切分，两份都返回全集
            （并由调用方在报告里注明分数不可信）。
        seed: 分桶种子。

    Returns:
        (train, holdout)。评测集过小 (< 4 条) 时不切分，返回 (全集, 全集)。
    """
    queries = list(eval_set.queries)
    if holdout_ratio <= 0.0 or holdout_ratio >= 1.0 or len(queries) < 4:
        return (
            EvalSet(queries=queries, name=f"{eval_set.name}:train"),
            EvalSet(queries=queries, name=f"{eval_set.name}:holdout"),
        )

    def _bucket(qid: str) -> float:
        h = hashlib.sha1(f"{seed}:{qid}".encode("utf-8")).hexdigest()
        # 取前 8 位十六进制映射到 [0,1)
        return int(h[:8], 16) / 0xFFFFFFFF

    holdout = [q for q in queries if _bucket(q.qid) < holdout_ratio]
    train = [q for q in queries if _bucket(q.qid) >= holdout_ratio]
    # 退化保护：任一侧为空时不切分，避免报告一个基于 0 条查询的分数。
    if not holdout or not train:
        return (
            EvalSet(queries=queries, name=f"{eval_set.name}:train"),
            EvalSet(queries=queries, name=f"{eval_set.name}:holdout"),
        )
    return (
        EvalSet(queries=train, name=f"{eval_set.name}:train"),
        EvalSet(queries=holdout, name=f"{eval_set.name}:holdout"),
    )


@dataclass
class TuningResult:
    """一次权重调节的结果。"""

    best_weights: list[float] | None
    """train 集上目标指标最优的 RRF 权重 [keyword, bm25, (vector)]；None=等权。"""

    best_score: float
    """最优配置在 **train** 上的指标均值 (0~1)。这个数是乐观的，别拿它汇报。"""

    scores: dict[str, float]
    """每个候选配置 (字符串表示) 在 train 上的指标均值。"""

    metric: str
    """做选择的指标名。"""

    candidates_tried: int
    """实际尝试的候选数 (无 embedder / 空集时可能少于网格)。"""

    holdout_score: float = 0.0
    """最优配置在 **holdout** 上的指标均值 —— 这才是可以对外汇报的数。"""

    baseline_holdout_score: float = 0.0
    """等权 (未调参) 基线在 holdout 上的分数，用于判断调参是否真的有收益。"""

    train_size: int = 0
    holdout_size: int = 0

    split_applied: bool = True
    """False 表示评测集太小无法切分，此时 holdout_score 与 best_score 同源、不可信。"""

    @property
    def overfit_gap(self) -> float:
        """train 与 holdout 的分数差 —— 明显为正说明在过拟合调参集。"""
        return round(self.best_score - self.holdout_score, 4)

    @property
    def tuning_gain(self) -> float:
        """调参相对等权基线在 holdout 上的净收益。<=0 说明调参没意义。"""
        return round(self.holdout_score - self.baseline_holdout_score, 4)


def _score_for_service(
    service: RetrievalService,
    eval_set: EvalSet,
    metric: str,
    corpus_workdir: str | Path,
) -> float:
    """对给定 (已设好权重的) service 跑一次 hybrid 评测，返回目标指标均值。"""
    report = run_eval(
        corpus_workdir, eval_set, strategies=("hybrid",), service=service
    )
    summary = report.summary(keys=(metric,))
    row = summary.get("hybrid", {})
    return float(row.get(metric, 0.0))


def tune_rrf_weights(
    workdir: str | Path,
    eval_set: EvalSet,
    metric: str = "recall@k",
    grid: Optional[Iterable[list[float]]] = None,
    candidates: Optional[Iterable[list[float] | None]] = None,
    holdout_ratio: float = DEFAULT_HOLDOUT_RATIO,
    seed: int = DEFAULT_SPLIT_SEED,
) -> TuningResult:
    """在评测集上网格搜索 hybrid 的 RRF 融合权重（train/holdout 切分）。

    P4 变更：选择在 **train** 上做，报告分数在 **holdout** 上算，并额外算一个
    等权基线的 holdout 分数。这样 ``TuningResult`` 能直接回答两个问题：

      - ``overfit_gap``  —— train 比 holdout 高多少？高很多就是在过拟合调参集。
      - ``tuning_gain``  —— 调参相对等权在 holdout 上净赚多少？<=0 就别调了。

    P1-6 的实现在同一份评测集上搜索并报告最优分数，那个数天然乐观。

    Args:
        workdir: 检索工作目录 (含 actions.jsonl 等)。
        eval_set: 评测集 (EvalSet)。
        metric: 用于挑选最优配置的指标，默认 'recall@k'。
        grid: 候选 [keyword, bm25] 权重网格；缺省用 DEFAULT_GRID。
            有 embedder (vector 可用) 时，每条后自动补上 vector 权重
            (取 keyword 权重的 1.2 倍作为合理起点)。
        candidates: 完全自定义的候选权重列表 (可选)；提供时优先于 grid。
        holdout_ratio: 留出比例；<=0 或评测集过小时不切分 (split_applied=False)。
        seed: 切分种子。

    Returns:
        TuningResult。若 eval_set 无查询，返回默认 (best=None, 各分数 0.0)。
    """
    workdir = Path(workdir)
    grid_src: Iterable[list[float] | None]
    if candidates is not None:
        grid_src = candidates
    else:
        g = list(grid) if grid is not None else DEFAULT_GRID
        base = RetrievalService(workdir)
        has_vector = base._embedder_available()
        if has_vector:
            grid_src = [list(w) + [w[0] * 1.2] for w in g]
        else:
            # 无 embedder 时只融合 keyword+bm25，退化为 2 维权重网格。
            grid_src = g

    if not eval_set.queries:
        return TuningResult(
            best_weights=None, best_score=0.0, scores={}, metric=metric,
            candidates_tried=0, holdout_score=0.0, baseline_holdout_score=0.0,
            train_size=0, holdout_size=0, split_applied=False,
        )

    train, holdout = split_eval_set(eval_set, holdout_ratio=holdout_ratio, seed=seed)
    # 真正切开时两份加起来正好是全集；未切分时两份都是全集 (和为 2N)。
    split_applied = (
        len(train.queries) + len(holdout.queries) == len(eval_set.queries)
    )

    best_weights: list[float] | None = None
    best_score = -1.0
    scores: dict[str, float] = {}
    tried = 0
    for w in grid_src:
        svc = RetrievalService(workdir)
        svc.set_fusion_weights(w)
        score = _score_for_service(svc, train, metric, workdir)
        scores[str(w)] = score
        tried += 1
        if score > best_score:
            best_score = score
            best_weights = w

    # 用选出的权重在 holdout 上重新评一遍 —— 这才是可对外汇报的数。
    best_svc = RetrievalService(workdir)
    best_svc.set_fusion_weights(best_weights)
    holdout_score = _score_for_service(best_svc, holdout, metric, workdir)

    # 等权基线，用于判断调参是否真有收益。
    baseline_svc = RetrievalService(workdir)
    baseline_svc.set_fusion_weights(None)
    baseline_holdout = _score_for_service(baseline_svc, holdout, metric, workdir)

    return TuningResult(
        best_weights=best_weights,
        best_score=max(best_score, 0.0),
        scores=scores,
        metric=metric,
        candidates_tried=tried,
        holdout_score=holdout_score,
        baseline_holdout_score=baseline_holdout,
        train_size=len(train.queries),
        holdout_size=len(holdout.queries),
        split_applied=split_applied,
    )
