"""P1-5 评测指标单测：Recall@K / MRR / nDCG@K / filter accuracy。

纯函数验证，不依赖任何检索实现与可选 RAG 依赖。
"""
from __future__ import annotations

import math

import pytest

from edr_rag.eval.metrics import (
    reciprocal_rank,
    recall_at_k,
    success_at_k,
    ndcg_at_k,
    filter_accuracy,
    query_metrics,
    aggregate_metrics,
)


# ---------- recall@k ----------

def test_recall_all_relevant_in_top_k():
    ranked = ["a", "b", "c", "d"]
    assert recall_at_k(ranked, ["a", "b"], k=4) == 1.0


def test_recall_partial():
    ranked = ["a", "b", "c", "d"]
    # relevant 三个里只有两个在 top-k 内
    assert recall_at_k(ranked, ["a", "c", "x"], k=3) == pytest.approx(2 / 3)


def test_recall_k_truncation():
    ranked = ["a", "b", "c", "d"]
    # k=2 只看到 a,b；relevant 有 c
    assert recall_at_k(ranked, ["b", "c"], k=2) == pytest.approx(1 / 2)


def test_recall_empty_relevant_is_degenerate_one():
    assert recall_at_k(["a"], [], k=3) == 1.0


def test_recall_no_hit_is_zero():
    assert recall_at_k(["a", "b"], ["z"], k=5) == 0.0


# ---------- success@k ----------

def test_success_full_recall_is_one():
    assert success_at_k(["a", "b", "c", "d"], ["a", "b"], k=4) == 1.0


def test_success_partial_recall_is_zero_not_fraction():
    # recall@k 会给 2/3 部分分，success 只看「全部检回与否」
    assert success_at_k(["a", "c", "d"], ["a", "c", "x"], k=3) == 0.0


def test_success_k_truncation_drops_below_one():
    # b 在 top-k 内、c 被截掉 → 非完整召回
    assert success_at_k(["a", "b", "c"], ["b", "c"], k=2) == 0.0
    assert success_at_k(["a", "b", "c"], ["b", "c"], k=3) == 1.0


def test_success_single_relevant_matches_hit():
    # |relevant|=1 时语义与「是否命中」一致
    assert success_at_k(["a", "b"], ["b"], k=2) == 1.0
    assert success_at_k(["a", "b"], ["z"], k=2) == 0.0


def test_success_empty_relevant_is_degenerate_one():
    assert success_at_k(["a"], [], k=3) == 1.0


def test_success_duplicates_do_not_fulfill_missing_relevant():
    # 重复 id 占位不能凑齐 relevant 集合
    assert success_at_k(["a", "a", "b"], ["a", "b"], k=2) == 0.0
    assert success_at_k(["a", "a", "b"], ["a", "b"], k=3) == 1.0


# ---------- MRR ----------

def test_mrr_first_is_relevant():
    assert reciprocal_rank(["a", "b", "c"], ["a"]) == 1.0


def test_mrr_third_rank():
    assert reciprocal_rank(["a", "b", "c", "d"], ["c"]) == pytest.approx(1 / 3)


def test_mrr_no_hit_zero():
    assert reciprocal_rank(["a", "b"], ["z"]) == 0.0


def test_mrr_empty_relevant_zero():
    assert reciprocal_rank(["a"], []) == 0.0


# ---------- nDCG@K ----------

def test_ndcg_perfect_ranking_is_one():
    ranked = ["a", "b", "c"]
    assert ndcg_at_k(ranked, ["a", "b", "c"], k=3) == pytest.approx(1.0)


def test_ndcg_relevant_not_first_lower():
    # related = {a, b}; x/y 无关。相关项排前分数 > 相关项排后。
    perfect = ndcg_at_k(["a", "b", "x", "y"], ["a", "b"], k=4)
    degraded = ndcg_at_k(["x", "y", "a", "b"], ["a", "b"], k=4)
    assert degraded < perfect


def test_ndcg_partial_relevant_bounded_by_one():
    val = ndcg_at_k(["x", "a", "y", "b"], ["a", "b"], k=4)
    assert 0.0 <= val <= 1.0


def test_ndcg_no_hit_zero():
    assert ndcg_at_k(["a", "b"], ["z"], k=4) == 0.0


def test_ndcg_empty_relevant_degenerate_one():
    assert ndcg_at_k(["a"], [], k=3) == 1.0


def test_ndcg_duplicate_relevant_counted_once():
    # 同一相关 id 重复出现只计首现一次：修复前 DCG 重复计分导致 nDCG=1.307>1
    val = ndcg_at_k(["a", "a", "b"], {"a", "b"}, k=3)
    assert val == pytest.approx(1.0)
    assert ndcg_at_k(["a", "a", "x"], {"a"}, k=3) == pytest.approx(1.0)


def test_ndcg_duplicate_displaces_second_relevant():
    # 重复占位把 b 挤出 top-2：去重后只命中 a 一次，DCG=1；IDCG=1+1/log2(3)（本实现按 log2(i+1) 折损）
    val = ndcg_at_k(["a", "a", "b"], {"a", "b"}, k=2)
    assert val == pytest.approx(1.0 / (1.0 + 1.0 / math.log2(3)))


# ---------- filter accuracy ----------

def test_filter_exact_match():
    assert filter_accuracy(["a", "b"], ["a", "b"]) == 1.0


def test_filter_extra_recall_penalized():
    # 返回了额外 id "c" —— 误召回，Jaccard 降低
    assert filter_accuracy(["a", "b", "c"], ["a", "b"]) == pytest.approx(2 / 3)


def test_filter_miss_penalized():
    # 漏了期待中的 "b"
    assert filter_accuracy(["a"], ["a", "b"]) == pytest.approx(1 / 2)


def test_filter_no_expected_requires_empty_return():
    assert filter_accuracy([], []) == 1.0
    assert filter_accuracy(["a"], []) == 0.0


# ---------- query_metrics / aggregate ----------

def test_query_metrics_without_filter():
    m = query_metrics(["a", "b", "c"], ["a"], k=3)
    assert m["recall@k"] == pytest.approx(1 / 1)
    assert m["success@k"] == 1.0
    assert m["mrr"] == 1.0
    assert m["ndcg@k"] == pytest.approx(1.0)
    assert "filter_accuracy" not in m


def test_query_metrics_with_filter():
    m = query_metrics(
        ["a", "b"], ["a"], k=3,
        filter_expected=["a"], filter_returned=["a"],
    )
    assert m["filter_accuracy"] == 1.0


def test_aggregate_averages_present_keys_only():
    rows = [
        {"recall@k": 1.0, "mrr": 1.0, "ndcg@k": 1.0},
        {"recall@k": 0.0, "mrr": 0.5, "ndcg@k": 0.5},
    ]
    agg = aggregate_metrics(rows, keys=("recall@k", "mrr", "ndcg@k"))
    assert agg["recall@k"] == pytest.approx(0.5)
    assert agg["mrr"] == pytest.approx(0.75)
    assert agg["ndcg@k"] == pytest.approx(0.75)


def test_aggregate_ignores_none_filter_accuracy():
    rows = [
        {"recall@k": 1.0, "filter_accuracy": 1.0},
        {"recall@k": 1.0},  # 无 filter 指标，应被跳过而不计入分母
    ]
    agg = aggregate_metrics(rows, keys=("recall@k", "filter_accuracy"))
    assert agg["filter_accuracy"] == 1.0
    assert agg["recall@k"] == 1.0


# ---------- 事实级指标（fact_groups，20261112-2003 契约） ----------

from edr_rag.eval.metrics import fact_mrr, fact_recall_at_k  # noqa: E402


class TestFactRecallAtK:
    def test_hit_any_member_covers_fact(self):
        # 3 个等价来源承载同一事实；命中任一即覆盖 → recall=1.0
        groups = [["id.a", "id.b", "id.c"]]
        assert fact_recall_at_k(["x", "id.b"], groups, k=5) == 1.0

    def test_miss_all_members_zero(self):
        groups = [["id.a", "id.b"]]
        assert fact_recall_at_k(["x", "y"], groups, k=5) == 0.0

    def test_multiple_independent_facts_covered_separately(self):
        # 两个独立事实组：命中组1任一成员 + 组2任一成员 = 1.0；只命中组1 = 0.5
        groups = [["a1", "a2"], ["b1", "b2"]]
        assert fact_recall_at_k(["a2", "b1"], groups, k=5) == 1.0
        assert fact_recall_at_k(["a1"], groups, k=5) == 0.5

    def test_not_all_equivalent_docs_counted_separately(self):
        # 与普通 recall 的关键差异：组内多成员命中不重复计分。
        # 普通 recall 会把 3 个成员当 3 个相关文档（3 命中 / 3 = 1.0），
        # 事实级口径下同一事实只算 1 —— 虚高形状被消除。
        groups = [["a", "b", "c"]]
        assert fact_recall_at_k(["a", "b", "c"], groups, k=5) == 1.0
        # 对照：普通 recall 在同一输入下得 1.0 但分母是 3 个文档；
        # 若只命中 1 个成员，普通 recall = 1/3（文档级部分分），
        # 事实级 = 1.0（事实已覆盖）。
        assert recall_at_k(["a"], ["a", "b", "c"], k=5) == pytest.approx(1 / 3)
        assert fact_recall_at_k(["a"], groups, k=5) == 1.0

    def test_empty_groups_degenerate_one(self):
        assert fact_recall_at_k(["x"], [], k=5) == 1.0
        assert fact_recall_at_k(["x"], [[]], k=5) == 1.0

    def test_k_truncation_respected(self):
        groups = [["g1"]]
        assert fact_recall_at_k(["g1"], groups, k=5) == 1.0
        assert fact_recall_at_k(["x", "x", "x", "x", "g1"], groups, k=4) == 0.0


class TestFactMrr:
    def test_first_hit_any_member(self):
        groups = [["a", "b", "c"]]
        assert fact_mrr(["x", "c"], groups) == pytest.approx(0.5)

    def test_multiple_facts_averaged(self):
        groups = [["a1", "a2"], ["b1"]]
        # 事实1 rank2 (0.5)、事实2 rank1 (1.0) → 均值 0.75
        assert fact_mrr(["b1", "a2"], groups) == pytest.approx(0.75)

    def test_uncovered_fact_zero_contribution(self):
        groups = [["a"], ["b"]]
        assert fact_mrr(["a"], groups) == pytest.approx(0.5)

    def test_empty_groups_zero(self):
        assert fact_mrr(["x"], []) == 0.0


class TestQueryMetricsFactGroupsAdditive:
    def test_no_fact_groups_keys_absent(self):
        out = query_metrics(["a"], ["a"], k=5)
        assert "fact_recall@k" not in out
        assert "fact_mrr" not in out

    def test_fact_groups_keys_added_without_touching_standard(self):
        standard = query_metrics(["a"], ["a", "b"], k=5)
        with_facts = query_metrics(["a"], ["a", "b"], k=5, fact_groups=[["a", "b"]])
        for key in ("recall@k", "success@k", "mrr", "ndcg@k"):
            assert with_facts[key] == standard[key]
        assert with_facts["fact_recall@k"] == 1.0

    def test_fact_level_differs_from_doc_level_on_twin_hits(self):
        # 旧共享 ID 虚高形状对照：孪生变体 a/b 同承载事实 f1，独立事实 c。
        # 检回 a+b 而未检回 c：文档级 recall=2/3（把同事实重复计分），
        # 事实级 fact_recall=1/2（每事实一票）——虚高被消除。
        ranked = ["a", "b"]
        relevant = ["a", "b", "c"]
        out = query_metrics(ranked, relevant, k=5, fact_groups=[["a", "b"], ["c"]])
        assert out["recall@k"] == pytest.approx(2 / 3)
        assert out["fact_recall@k"] == pytest.approx(0.5)
