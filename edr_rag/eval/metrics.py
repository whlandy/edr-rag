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


def success_at_k(ranked_ids: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Success@K = 1 当且仅当 top-K 覆盖全部 relevant（完整召回）。

    与 recall_at_k 的区别：recall 对 |relevant|>1 时给部分分（3 命中 2 得
    0.67），success 只看「top-K 内是否检回全部相关项」——答案可达（任一
    gold 在 top-K）与答案完整（gold 全部在 top-K）是两个维度，多 gold 项
    （如兄弟节标注）需要这个不折损的二值指标区分。
    relevant 为空时为退化情况，定义同 recall_at_k = 1.0。
    """
    relevant = set(relevant)
    if not relevant:
        return 1.0
    return 1.0 if set(ranked_ids[:k]) >= relevant else 0.0


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

    ranked_ids 中重复的 id 只按首次出现计分一次：检索层可能对同一 id
    返回多条候选（语料存在重复 id），重复计入会让 DCG 超过 IDCG 上限。
    """
    relevant = set(relevant)
    if not relevant:
        return 1.0
    seen: set[str] = set()
    rel: list[float] = []
    for rid in ranked_ids[:k]:
        if rid in seen:
            continue
        seen.add(rid)
        rel.append(1.0 if rid in relevant else 0.0)
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


def fact_recall_at_k(
    ranked_ids: Sequence[str], fact_groups: Sequence[Sequence[str]], k: int
) -> float:
    """事实级 Recall@K：每个事实组（等价来源集合）命中任一成员即覆盖。

    与普通 recall_at_k 的区别：组内成员互为「同一事实的不同来源」（如
    同内容跨文件变体），命中任一即算覆盖该事实——不把组内全部成员都当
    独立相关文档（否则同一事实被重复计分，旧共享 ID 口径的虚高形状）；
    组间相互独立，多个独立必需事实仍须分别覆盖（fractional by fact）。

    fact_groups 为空时返回 1.0（无事实标注的退化情况，同 recall_at_k）。
    """
    groups = [set(g) for g in fact_groups if g]
    if not groups:
        return 1.0
    top_k = set(ranked_ids[:k])
    covered = sum(1 for g in groups if top_k & g)
    return covered / len(groups)


def fact_mrr(ranked_ids: Sequence[str], fact_groups: Sequence[Sequence[str]]) -> float:
    """事实级 MRR：首个覆盖任一事实的排名倒数（单事实标注下与 MRR 同义）。

    多事实时取「各事实首次被覆盖位置倒数」的均值——每个独立必需事实
    都应尽早出现，与 fact_recall_at_k 的按事实分母一致。
    """
    groups = [set(g) for g in fact_groups if g]
    if not groups:
        return 0.0
    total = 0.0
    for g in groups:
        for i, rid in enumerate(ranked_ids, start=1):
            if rid in g:
                total += 1.0 / i
                break
    return total / len(groups)


def query_metrics(
    ranked_ids: Sequence[str],
    relevant: Iterable[str],
    k: int,
    filter_expected: Iterable[str] | None = None,
    filter_returned: Sequence[str] | None = None,
    fact_groups: Sequence[Sequence[str]] | None = None,
) -> dict[str, float]:
    """汇总单条查询的所有指标。

    Args:
        ranked_ids: 检索返回的有序 id 列表 (已截断到 top_k)。
        relevant: 标注的相关 id 集合。
        k: 计算 Recall/NDCG 的截断点 (默认取 len(ranked_ids)，详见调用方)。
        filter_expected: 该用例过滤后应出现的 id 集合 (metadata filter 用例)。
        filter_returned: 实际过滤后返回的有序 id 列表。
        fact_groups: 可选事实组标注（组=等价来源集合，命中任一成员覆盖该
            事实）。提供时额外输出 fact_recall@k / fact_mrr 两个键；缺省
            不输出，普通文档级指标语义不受影响（旧 gold/基线不变）。

    Returns:
        {"recall@k", "success@k", "mrr", "ndcg@k", "filter_accuracy"}
        (+ {"fact_recall@k", "fact_mrr"} when fact_groups provided)。
        filter_accuracy 仅在 filter_expected 与 filter_returned 均提供时计算，
        否则置为 ``None`` 缺省 (由调用方决定聚合时是否纳入)。
    """
    k = max(int(k), 1)
    out = {
        "recall@k": recall_at_k(ranked_ids, relevant, k),
        "success@k": success_at_k(ranked_ids, relevant, k),
        "mrr": reciprocal_rank(ranked_ids, relevant),
        "ndcg@k": ndcg_at_k(ranked_ids, relevant, k),
    }
    if fact_groups:
        out["fact_recall@k"] = fact_recall_at_k(ranked_ids, fact_groups, k)
        out["fact_mrr"] = fact_mrr(ranked_ids, fact_groups)
    if filter_expected is not None and filter_returned is not None:
        out["filter_accuracy"] = filter_accuracy(filter_returned, filter_expected)
    return out


def aggregate_metrics(
    rows: Iterable[dict[str, float]],
    keys: Sequence[str] = ("recall@k", "success@k", "mrr", "ndcg@k", "filter_accuracy"),
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
