"""P1-5 评测 runner：对多检索策略跑评测集并汇总指标。

设计:
  - 复用统一检索入口 ``RetrievalService``，因此 keyword / bm25 / vector /
    hybrid 四条策略的评测共用同一条代码路径，也与 CLI / MCP 行为一致。
  - 对比维度就是「策略」：runner 对每条评测查询在每个目标策略下分别检索，
    计算该查询的各项指标，再按策略聚合得到策略级汇总。
  - 可选 `rag` extras 未装时，vector 降级 bm25、hybrid 降 RRF 融合 —
    runner 直接复用 service 的降级结果，主链路始终可用，不会报错。
  - filter 用例: 先不带过滤检索得到 ranked_ids 算 ranking 指标，再用
    service 的 filter 检索得到 filter_returned 算 filter_accuracy。

输出:
  - 逐查询明细 (query_metrics + returned ids)，便于审查。
  - 策略级汇总报告 (dict)，含各指标均值。

P2-6 (检索与生成分开评测):
  runner 在检索之外还驱动生成维度 —— 把检索命中的上下文喂给
  ``synthesize_answer`` 生成结构化答案，再按 generation 维度算：
  引用正确率 / 拒答率+误拒答率 / 高风险识别率。因此一份评测同时产出
  「retrieval 表」(Recall/MRR/nDCG/FilterAcc) 与「generation 表」
  (引用正确率/拒答率/误拒答率/高风识别率)，不再只看最终答案一句话。
  生成侧复用与 CLI / MCP 完全一致的答案链，评测行为等同于真实使用。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..p2.synthesize import SynthesizedAnswer, synthesize_answer
from ..retrieval import RetrievalService
from .dataset import EvalQuery, EvalSet
from .generation import (
    high_risk_identify_rate,
    query_generation_metrics,
    refusal_rate,
)
from .metrics import aggregate_metrics, query_metrics
from .vocab import SLICE_ALL, SLICE_IN_VOCAB, SLICE_OOV, in_slice, is_in_vocabulary

DEFAULT_STRATEGIES = ("keyword", "bm25", "hybrid")


@dataclass
class QueryResult:
    """单条查询在某策略下的评测结果。"""

    qid: str
    query: str
    corpus_type: str
    strategy: str
    ranked_ids: list[str] = field(default_factory=list)
    returned: list[dict] = field(default_factory=list)
    filter_returned: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    # P2-6: 生成维度结果。
    answer: Optional[SynthesizedAnswer] = None
    generation_metrics: dict[str, float] = field(default_factory=dict)
    # P2-6: 生成维度标注（透传自 EvalQuery，供汇总时按策略分组）。
    expected_refusal: Optional[bool] = None
    high_risk: Optional[bool] = None
    # 词表覆盖切片标记：该查询是否命中 query.py 的 SYNONYMS 别名表。
    # 由 eval.vocab 从别名表实时推导，用于把「查表贡献」与「泛化能力」
    # 分开汇报（详见 eval/vocab.py）。
    in_vocabulary: bool = False


@dataclass
class EvalReport:
    """一次评测运行的结果：数据集 + 策略汇总。"""

    eval_set: EvalSet
    rows: list[QueryResult] = field(default_factory=list)

    def by_strategy(self, slice_name: str = SLICE_ALL) -> dict[str, list[QueryResult]]:
        """按策略分组结果行；``slice_name`` 可限定词表覆盖切片。

        切片语义见 eval/vocab.py：``in_vocab`` 的召回含同义词表的查表贡献，
        ``oov`` 才反映泛化能力。缺省 ``all`` 为不切片（向后兼容）。
        """
        out: dict[str, list[QueryResult]] = {}
        for r in self.rows:
            if not in_slice(r.in_vocabulary, slice_name):
                continue
            out.setdefault(r.strategy, []).append(r)
        return out

    def slice_sizes(self) -> dict[str, int]:
        """各词表切片下的**查询变体**数（跨策略去重，按 qid 统计）。"""
        seen: dict[str, set[str]] = {SLICE_IN_VOCAB: set(), SLICE_OOV: set()}
        for r in self.rows:
            seen[SLICE_IN_VOCAB if r.in_vocabulary else SLICE_OOV].add(r.qid)
        return {
            SLICE_ALL: len(seen[SLICE_IN_VOCAB] | seen[SLICE_OOV]),
            SLICE_IN_VOCAB: len(seen[SLICE_IN_VOCAB]),
            SLICE_OOV: len(seen[SLICE_OOV]),
        }

    def summary(
        self,
        keys=("recall@k", "mrr", "ndcg@k", "filter_accuracy"),
        slice_name: str = SLICE_ALL,
    ) -> dict[str, dict[str, float]]:
        """按策略聚合 retrieval 维度指标，返回 {strategy: {metric: mean}}。

        ``slice_name`` 限定词表覆盖切片（``all`` / ``in_vocab`` / ``oov``）。
        用 ``oov`` 判断泛化能力——``in_vocab`` 的数字含同义词表的查表贡献。
        """
        return {
            strategy: aggregate_metrics([r.metrics for r in rows], keys)
            for strategy, rows in self.by_strategy(slice_name).items()
        }

    def generation_summary(self) -> dict[str, dict[str, float]]:
        """按策略聚合 generation 维度指标。

        返回 {strategy: {...}}，其中:
          - citation_accuracy: 引用正确率（在各查询上的均值）。
          - refusal_recall:    拒答率（应拒答中被拒答的比例）。
          - false_refusal:     误拒答率（应作答却被拒答的比例）。
          - high_risk:         高风险识别率（应识别高风险中被识别出的比例）。

        每个指标只在「有对应标注 / 有样本」的策略上计算；无样本的策略
        给出 0.0 占位，保证结构完整。
        """
        out: dict[str, dict[str, float]] = {}
        for strategy, rows in self.by_strategy().items():
            ca = aggregate_metrics(
                [r.generation_metrics for r in rows],
                keys=("citation_accuracy",),
            )
            rr = refusal_rate(
                [(r.answer.answer_status if r.answer else "", r.expected_refusal)
                 for r in rows if r.expected_refusal is not None]
            )
            hr = high_risk_identify_rate(
                [(r.answer, r.high_risk) for r in rows if r.high_risk is not None]
            )
            out[strategy] = {
                "citation_accuracy": ca.get("citation_accuracy", 0.0),
                "refusal_recall": rr["refusal_recall"],
                "false_refusal": rr["false_refusal"],
                "high_risk": hr,
            }
        return out


def _filter_expected(q: EvalQuery) -> Optional[list[str]]:
    return q.filter_expected or None


def _run_query(
    service: RetrievalService,
    q: EvalQuery,
    strategy: str,
) -> QueryResult:
    """对单条查询在单个策略下执行检索并计算指标。

    ranking 指标 (Recall/MRR/NDCG) 用无过滤检索的 top-K 结果；若该用例带
    metadata filter 与 filter_expected，则额外做一次带过滤检索以计算
    filter_accuracy。

    P2-6: 用主检索命中的上下文驱动生成（``synthesize_answer``），计算
    generation 维度的引用正确率 / 拒答判定 / 高风险识别，与 retrieval 指标
    一起落在同一个 QueryResult 里。
    """
    top_k = q.top_k
    # 主检索（无过滤）：用于排序类指标。
    results = service.search(
        q.query,
        corpus_type=q.corpus_type,
        strategy=strategy,
        top_k=top_k,
    )
    ranked_ids = [r.get("id", "") for r in results]

    filter_returned: list[str] = []
    if q.filter is not None and _filter_expected(q) is not None:
        fret = service.search(
            q.query,
            corpus_type=q.corpus_type,
            strategy=strategy,
            filters=q.filter,
            top_k=top_k,
        )
        filter_returned = [r.get("id", "") for r in fret]

    metrics = query_metrics(
        ranked_ids,
        q.relevant,
        k=top_k,
        filter_expected=q.filter_expected if q.filter is not None and q.filter_expected else None,
        filter_returned=filter_returned,
    )

    # P2-6: 生成维度 —— 基于检索命中的上下文生成结构化答案并计算指标。
    # 无论是否有命中都调用 synthesize_answer：空上下文时其内置的
    # 证据不足拒答逻辑会给出 status='refused'，保证「应该拒答」类查询
    # （如隐私越权）能被正确评估，与 CLI/MCP 行为一致。
    answer = None
    generation_metrics: dict[str, float] = {}
    try:
        answer = synthesize_answer(q.query, results)
    except Exception:
        answer = None
    generation_metrics = query_generation_metrics(
        answer,
        q.relevant,
        expected_refusal=q.expected_refusal,
        high_risk=q.high_risk,
    )

    return QueryResult(
        qid=q.qid,
        query=q.query,
        corpus_type=q.corpus_type,
        strategy=strategy,
        ranked_ids=ranked_ids,
        returned=results,
        filter_returned=filter_returned,
        metrics=metrics,
        answer=answer,
        generation_metrics=generation_metrics,
        expected_refusal=q.expected_refusal,
        high_risk=q.high_risk,
        in_vocabulary=is_in_vocabulary(q.query),
    )


def run_eval(
    workdir: str | Path,
    eval_set: EvalSet,
    strategies: tuple[str, ...] = DEFAULT_STRATEGIES,
    use_paraphrases: bool = False,
    service: Optional[RetrievalService] = None,
) -> EvalReport:
    """对评测集跑一组策略，返回 EvalReport。

    Args:
        workdir: 检索工作目录 (含 actions.jsonl / sections.jsonl 等)。
        eval_set: 评测集。
        strategies: 要对比的策略元组。空元组 -> 不执行任何检索。
        use_paraphrases: 是否把每条同义改写也当独立查询评测 (放大规模)。
        service: 可选预建 RetrievalService 复用 (测试/性能用)；缺省按
            workdir 新建。

    Returns:
        EvalReport，含逐查询结果与按策略汇总。
    """
    if not strategies:
        return EvalReport(eval_set=eval_set)
    svc = service or RetrievalService(workdir)
    rows: list[QueryResult] = []
    for q in eval_set.queries:
        for qid, query_text in q.effective_queries(use_paraphrases):
            q_run = EvalQuery(
                qid=qid,
                query=query_text,
                corpus_type=q.corpus_type,
                relevant=q.relevant,
                paraphrases=[],
                top_k=q.top_k,
                filter=q.filter,
                filter_expected=q.filter_expected,
                expected_refusal=q.expected_refusal,
                high_risk=q.high_risk,
            )
            for strategy in strategies:
                rows.append(_run_query(svc, q_run, strategy))
    return EvalReport(eval_set=eval_set, rows=rows)


def _format_vocab_slices(report: EvalReport, metric_labels: dict[str, str]) -> list[str]:
    """渲染 in_vocab / oov 词表切片对比表（两个切片都非空时才输出）。

    只有一个切片有样本时整张表退化成上面那张全集表的复制品，纯噪音，直接
    跳过。切片语义与「为什么要分开看」见 eval/vocab.py。
    """
    sizes = report.slice_sizes()
    if not (sizes[SLICE_IN_VOCAB] and sizes[SLICE_OOV]):
        return []

    lines = ["", "== Retrieval 维度 · 词表覆盖切片 =="]
    lines.append(
        f"in_vocab ({sizes[SLICE_IN_VOCAB]} 条): 命中 query.py 的 SYNONYMS 别名表。"
        f"该表照评测集改写写成，此列含查表贡献，仅用于回归别名表是否生效。"
    )
    lines.append(
        f"oov      ({sizes[SLICE_OOV]} 条): 未命中别名表 —— 判断泛化能力请看这一列。"
    )
    head = f"{'切片':<10}{'策略':<10}" + "".join(
        f"{metric_labels.get(m, m):>14}" for m in metric_labels
    )
    lines.append(head)
    lines.append("-" * len(head))
    for slice_name in (SLICE_IN_VOCAB, SLICE_OOV):
        for strategy, metrics in report.summary(slice_name=slice_name).items():
            cells = "".join(f"{metrics.get(m, 0.0):>14.4f}" for m in metric_labels)
            lines.append(f"{slice_name:<10}{strategy:<10}{cells}")
    return lines


def format_summary(report: EvalReport) -> str:
    """把策略级汇总渲染成可读的多行文本 (中文表头)。

    输出两段：retrieval 维度（Recall/MRR/nDCG/FilterAcc）与 generation 维度
    （引用正确率/拒答率/误拒答率/高风险识别率），分别独立成表（P2-6）。
    """
    lines = [f"评测集: {report.eval_set.name}"
             f"  (主查询 {len(report.eval_set.queries)} 条, "
             f"变体 {report.eval_set.variant_total} 条)"]

    # --- retrieval 维度 ---
    lines.append("")
    lines.append("== Retrieval 维度 ==")
    summary = report.summary()
    metric_labels = {
        "recall@k": "Recall@K",
        "mrr": "MRR",
        "ndcg@k": "nDCG@K",
        "filter_accuracy": "FilterAcc",
    }
    header = f"{'策略':<10}" + "".join(f"{metric_labels.get(m, m):>14}" for m in metric_labels)
    lines.append(header)
    lines.append("-" * len(header))
    for strategy, metrics in summary.items():
        cells = "".join(f"{metrics.get(m, 0.0):>14.4f}" for m in metric_labels)
        lines.append(f"{strategy:<10}{cells}")

    # --- 词表覆盖切片 ---
    lines.extend(_format_vocab_slices(report, metric_labels))

    # --- generation 维度 ---
    lines.append("")
    lines.append("== Generation 维度 ==")
    gen = report.generation_summary()
    gen_labels = {
        "citation_accuracy": "引用正确率",
        "refusal_recall": "拒答率",
        "false_refusal": "误拒答率",
        "high_risk": "高风识别率",
    }
    g_header = f"{'策略':<10}" + "".join(f"{gen_labels.get(m, m):>14}" for m in gen_labels)
    lines.append(g_header)
    lines.append("-" * len(g_header))
    for strategy, metrics in gen.items():
        cells = "".join(f"{metrics.get(m, 0.0):>14.4f}" for m in gen_labels)
        lines.append(f"{strategy:<10}{cells}")
    return "\n".join(lines)
