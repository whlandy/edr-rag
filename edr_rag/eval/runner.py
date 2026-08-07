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
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..retrieval import RetrievalService
from .dataset import EvalQuery, EvalSet
from .metrics import aggregate_metrics, query_metrics

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


@dataclass
class EvalReport:
    """一次评测运行的结果：数据集 + 策略汇总。"""

    eval_set: EvalSet
    rows: list[QueryResult] = field(default_factory=list)

    def by_strategy(self) -> dict[str, list[QueryResult]]:
        out: dict[str, list[QueryResult]] = {}
        for r in self.rows:
            out.setdefault(r.strategy, []).append(r)
        return out

    def summary(self, keys=("recall@k", "mrr", "ndcg@k", "filter_accuracy")) -> dict[str, dict[str, float]]:
        """按策略聚合指标，返回 {strategy: {metric: mean}}。"""
        return {
            strategy: aggregate_metrics([r.metrics for r in rows], keys)
            for strategy, rows in self.by_strategy().items()
        }


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
    return QueryResult(
        qid=q.qid,
        query=q.query,
        corpus_type=q.corpus_type,
        strategy=strategy,
        ranked_ids=ranked_ids,
        returned=results,
        filter_returned=filter_returned,
        metrics=metrics,
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
            )
            for strategy in strategies:
                rows.append(_run_query(svc, q_run, strategy))
    return EvalReport(eval_set=eval_set, rows=rows)


def format_summary(report: EvalReport) -> str:
    """把策略级汇总渲染成可读的多行文本 (中文表头)。"""
    lines = [f"评测集: {report.eval_set.name}"
             f"  (主查询 {len(report.eval_set.queries)} 条, "
             f"变体 {report.eval_set.variant_total} 条)"]
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
    return "\n".join(lines)
