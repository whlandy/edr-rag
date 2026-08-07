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


@dataclass
class TuningResult:
    """一次权重调节的结果。"""

    best_weights: list[float] | None
    """评测集上目标指标最优的 RRF 权重 [keyword, bm25, (vector)]；None=等权。"""

    best_score: float
    """最优配置的目标指标均值 (0~1)。"""

    scores: dict[str, float]
    """每个候选配置 (字符串表示) 的指标均值。"""

    metric: str
    """做选择的指标名。"""

    candidates_tried: int
    """实际尝试的候选数 (无 embedder / 空集时可能少于网格)。"""


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
) -> TuningResult:
    """在评测集上网格搜索 hybrid 的 RRF 融合权重。

    Args:
        workdir: 检索工作目录 (含 actions.jsonl 等)。
        eval_set: 评测集 (EvalSet)。
        metric: 用于挑选最优配置的指标，默认 'recall@k'。
        grid: 候选 [keyword, bm25] 权重网格；缺省用 DEFAULT_GRID。
            有 embedder (vector 可用) 时，每条后自动补上 vector 权重
            (取 keyword 权重的 1.2 倍作为合理起点)。
        candidates: 完全自定义的候选权重列表 (可选)；每条即传入
            RetrievalService.set_fusion_weights 的权重列表 (可为 None=等权)。
            提供时优先于 grid。

    Returns:
        TuningResult。若 eval_set 无查询，返回默认 (best=None, best_score=0.0,
        candidates_tried=0)。
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
            # 无 embedder 时只融合 keyword+bm25，退化为 2 维权重重网格。
            grid_src = g

    if not eval_set.queries:
        return TuningResult(best_weights=None, best_score=0.0, scores={}, metric=metric, candidates_tried=0)

    best_weights: list[float] | None = None
    best_score = -1.0
    scores: dict[str, float] = {}
    tried = 0
    for w in grid_src:
        svc = RetrievalService(workdir)
        svc.set_fusion_weights(w)
        score = _score_for_service(svc, eval_set, metric, workdir)
        scores[str(w)] = score
        tried += 1
        if score > best_score:
            best_score = score
            best_weights = w
    return TuningResult(
        best_weights=best_weights,
        best_score=best_score,
        scores=scores,
        metric=metric,
        candidates_tried=tried,
    )
