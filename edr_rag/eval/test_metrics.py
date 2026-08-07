"""P1-5 评测指标单测：Recall@K / MRR / nDCG@K / filter accuracy。

纯函数验证，不依赖任何检索实现与可选 RAG 依赖。
"""
from __future__ import annotations

import pytest

from edr_rag.eval.metrics import (
    reciprocal_rank,
    recall_at_k,
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
