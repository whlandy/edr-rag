"""P1-5 评测指标：Recall@K / MRR / nDCG@K / metadata filter accuracy。

所有指标基于「检索返回的 id 列表」与「标注的相关 id 集合」计算，与具体
检索实现解耦，可独立单测 (纯函数，无 IO / 无外部依赖)。

约定:
  - ``ranked_ids``: 检索返回的有序 id 列表 (按相关度降序，已截断到 K)。
  - ``relevant``:   标注的相关 id 集合 (无序)。
  - ``id`` 匹配直接用字符串相等；动作语料里 result id = action_id，
    generic 语料 result id = 文档主键，二者均可用于标注。
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence


def recall_at_k(ranked_ids: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Recall@K = |top-K ∩ relevant| / |relevant|。

    relevant 为空时为退化情况，定义为 1.0 (无相关标注时视为全命中)，
    避免除零并契合评测集里不含 relevant 的纯元数据用例。
    """
    relevant = set(relevant)
    if not relevant:
        return 1.0
    top_k = set(ranked_ids[:k])
    hits = len(top_k & relevant)
    return hits / len(relevant)


def reciprocal_rank(ranked_ids: Sequence[str], relevant: Iterable[str]) -> float:
    """MRR 单样本项：首个相关命中的位置的倒数；无命中为 0。

    对 top-K 截断后的列表计算 (位置从 1 开始计数)。
    """
    relevant = set(relevant)
    for i, rid in enumerate(ranked_ids, start=1):
        if rid in relevant:
            return 1.0 / i
    return 0.0


def _dcg(relevance: Sequence[float], k: int) -> float:
    rel = list(relevance[:k])
    if not rel:
        return 0.0
    # DCG 用折损但保留第 1 位 (rank1 折损为 1)。
    return rel[0] + sum(r / math.log2(i + 1) for i, r in enumerate(rel[1:], start=2))


def ndcg_at_k(ranked_ids: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """nDCG@K：二元相关度 (命中=1) 的归一化折损累计增益。

    理想排序把所有相关项排在前面，据此计算 IDCG 归一化。
    relevant 为空时返回 1.0 (退化情况，同 recall_at_k)。
    """
    relevant = set(relevant)
    if not relevant:
        return 1.0
    rel = [1.0 if rid in relevant else 0.0 for rid in ranked_ids[:k]]
    dcg = _dcg(rel, k)
    if dcg == 0.0:
        return 0.0
    idcg = _dcg([1.0] * min(k, len(relevant)), k)
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def filter_accuracy(returned_ids: Sequence[str], expected_ids: Iterable[str]) -> float:
    """metadata filter accuracy：过滤器结果是否正确。

    定义为 F1 的对称形式 —— 用 Jaccard 相似度衡量「过滤后应出现的 id 集合」
    与「实际返回的 id 集合」的重合度，兼顾误召回 (extra) 与漏召回 (miss)。

    若 expected 为空 (该用例不要求任何结果)，则只要 returned 也为空即正确
    (准确率 1.0)，否则为 0.0。
    """
    returned = set(returned_ids)
    expected = set(expected_ids)
    if not expected:
        return 1.0 if not returned else 0.0
    union = returned | expected
    if not union:
        return 1.0
    return len(returned & expected) / len(union)


def query_metrics(
    ranked_ids: Sequence[str],
    relevant: Iterable[str],
    k: int,
    filter_expected: Iterable[str] | None = None,
    filter_returned: Sequence[str] | None = None,
) -> dict[str, float]:
    """汇总单条查询的所有指标。

    Args:
        ranked_ids: 检索返回的有序 id 列表 (已截断到 top_k)。
        relevant: 标注的相关 id 集合。
        k: 计算 Recall/NDCG 的截断点 (默认取 len(ranked_ids)，详见调用方)。
        filter_expected: 该用例过滤后应出现的 id 集合 (metadata filter 用例)。
        filter_returned: 实际过滤后返回的有序 id 列表。

    Returns:
        {"recall@k", "mrr", "ndcg@k", "filter_accuracy"}。
        filter_accuracy 仅在 filter_expected 与 filter_returned 均提供时计算，
        否则置为 ``None`` 缺省 (由调用方决定聚合时是否纳入)。
    """
    k = max(int(k), 1)
    out = {
        "recall@k": recall_at_k(ranked_ids, relevant, k),
        "mrr": reciprocal_rank(ranked_ids, relevant),
        "ndcg@k": ndcg_at_k(ranked_ids, relevant, k),
    }
    if filter_expected is not None and filter_returned is not None:
        out["filter_accuracy"] = filter_accuracy(filter_returned, filter_expected)
    return out


def aggregate_metrics(
    rows: Iterable[dict[str, float]],
    keys: Sequence[str] = ("recall@k", "mrr", "ndcg@k", "filter_accuracy"),
) -> dict[str, float]:
    """对多条查询的指标做平均（忽略缺失/None 项），得到策略级汇总。

    Args:
        rows: 每条查询的指标 dict (query_metrics 输出)。
        keys: 需要聚合的指标名。

    Returns:
        {metric: 平均值}，平均在「提供了该指标的查询」上计算。
    """
    sums: dict[str, float] = {k_: 0.0 for k_ in keys}
    counts: dict[str, int] = {k_: 0 for k_ in keys}
    for row in rows:
        for k_ in keys:
            val = row.get(k_)
            if val is None:
                continue
            sums[k_] += float(val)
            counts[k_] += 1
    return {k_: (sums[k_] / counts[k_] if counts[k_] else 0.0) for k_ in keys}
