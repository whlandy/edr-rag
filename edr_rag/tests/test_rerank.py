"""P1-3 tests: RRF fusion + optional cross-encoder rerank.

Covers:
  - rrf_fuse: correct fusion ordering, dedup, rrf_score, empty input
  - available()/rerank(): graceful degradation when optional deps absent
    (this environment lacks FlagEmbedding/torch/llama_index -> rerank skipped)
  - retrieval.hybrid: strategy="hybrid" runs multi-recall -> RRF -> top K,
    rerank toggle threads through (None/True/False), and never raises.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from edr_rag import rerank
from edr_rag.retrieval import RetrievalService

ACTION_A = {
    "action_id": "network.configure_proxy",
    "product": "EDR",
    "module": "network",
    "title": "配置代理服务器",
    "status": "approved",
    "confidence": 0.9,
    "risk": "medium",
    "requires_confirmation": True,
    "preconditions": ["已安装客户端"],
    "procedure": [{"order": 1, "action": "click", "target": "代理"}],
    "success_criteria": ["代理生效"],
    "recovery": [],
}

ACTION_B = {
    "action_id": "system.reset_system",
    "product": "EDR",
    "module": "system",
    "title": "重置系统",
    "status": "needs_review",
    "confidence": 0.8,
    "risk": "high",
    "requires_confirmation": True,
    "preconditions": ["管理员权限"],
    "procedure": [{"order": 1, "action": "click", "target": "重置"}],
    "success_criteria": [],
    "recovery": ["恢复备份"],
}

ACTION_C = {
    "action_id": "network.set_dns",
    "product": "EDR",
    "module": "network",
    "title": "设置 DNS 服务器",
    "status": "approved",
    "confidence": 0.7,
    "risk": "low",
    "requires_confirmation": False,
    "preconditions": [],
    "procedure": [{"order": 1, "action": "click", "target": "DNS"}],
    "success_criteria": ["DNS 生效"],
    "recovery": [],
}

SECTION = {
    "id": "s1",
    "title": "配置代理服务器",
    "content": "本节介绍如何配置代理，需要网络设置。",
    "section_path": ["网络", "代理"],
    "source": {"file": "net/proxy.html"},
}


def _write_artifacts(root: Path):
    (root / "actions.jsonl").write_text(
        "\n".join(json.dumps(a, ensure_ascii=False) for a in [ACTION_A, ACTION_B, ACTION_C]) + "\n",
        encoding="utf-8",
    )
    (root / "sections.jsonl").write_text(
        json.dumps(SECTION, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    (root / "procedures.jsonl").write_text(
        json.dumps({"id": "p1", "title": "配置代理", "steps": []}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (root / "chunks.jsonl").write_text(
        json.dumps({"id": "c1", "title": "配置代理", "text": "打开网络设置选择代理"},
                   ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


@pytest.fixture
def service(tmp_path: Path) -> RetrievalService:
    _write_artifacts(tmp_path)
    return RetrievalService(tmp_path)


# ---------- rrf_fuse ----------

def test_rrf_fuse_orders_by_recall_coverage():
    a = [{"id": "x1"}, {"id": "x2"}, {"id": "x3"}]
    b = [{"id": "x2"}, {"id": "x1"}]
    c = [{"id": "x4"}, {"id": "x2"}, {"id": "x1"}]
    fused = rerank.rrf_fuse([a, b, c])
    ids = [r["id"] for r in fused]
    # x2 appears in all 3 -> top; x1 in 2 -> second; x3/x4 in 1 each.
    assert ids[0] == "x2"
    assert ids[1] == "x1"
    assert set(ids[2:]) == {"x3", "x4"}
    # scores strictly descending.
    scores = [r["rrf_score"] for r in fused]
    assert scores == sorted(scores, reverse=True)
    assert all("rrf_score" in r for r in fused)


def test_rrf_fuse_dedup_keeps_first_fields():
    a = [{"id": "x1", "v": 1}]
    b = [{"id": "x1", "v": 99}]
    fused = rerank.rrf_fuse([a, b])
    assert len(fused) == 1
    # fields come from the first occurrence.
    assert fused[0]["v"] == 1


def test_rrf_fuse_empty_and_missing_key():
    assert rerank.rrf_fuse([]) == []
    assert rerank.rrf_fuse([[{"bad": 1}]]) == []


def test_rrf_fuse_custom_k():
    a = [{"id": "x1"}]
    b = [{"id": "x1"}]
    fused_default = rerank.rrf_fuse([a, b])
    fused_k1 = rerank.rrf_fuse([a, b], k=1.0)
    assert fused_k1[0]["rrf_score"] > fused_default[0]["rrf_score"]


# ---------- rerank optional degradation ----------

def test_rerank_skips_when_unavailable(service):
    # This environment lacks the optional rag deps -> rerank must be a no-op.
    results = service.search("配置", corpus_type="actions", strategy="hybrid")
    assert results, "hybrid should still return results without a reranker"

    # Direct call with enabled=True but no deps: original order, did_rerank=False.
    items = [{"id": "a", "text": "hello world"}, {"id": "b", "text": "other text"}]
    out, did = rerank.rerank("hello", items, enabled=True)
    assert did is False
    assert [r["id"] for r in out] == ["a", "b"]


def test_rerank_disabled_returns_input_unmodified():
    items = [{"id": "a"}, {"id": "b"}]
    out, did = rerank.rerank("q", items, enabled=False)
    assert did is False
    assert out is items


# ---------- hybrid strategy through RetrievalService ----------

def test_hybrid_actions_returns_rich_results(service):
    results = service.search("配置代理", corpus_type="actions", strategy="hybrid")
    assert results
    r = results[0]
    assert r["id"] == "network.configure_proxy"
    assert r["risk"] == "medium"
    assert "snippet" in r


def test_hybrid_generic_returns_results(service):
    results = service.search("配置代理", corpus_type="sections", strategy="hybrid")
    assert results
    assert results[0]["id"] == "s1"


def test_hybrid_actions_respects_filters(service):
    results = service.search(
        "配置", corpus_type="actions", strategy="hybrid", filters={"risk": "high"}
    )
    assert results
    assert all(r["risk"] == "high" for r in results)


def test_hybrid_top_k(service):
    results = service.search("网络", corpus_type="actions", strategy="hybrid", top_k=2)
    assert len(results) <= 2


def test_hybrid_rerank_flag_none_true_false(service):
    for toggle in (None, True, False):
        res = service.search(
            "配置代理", corpus_type="actions", strategy="hybrid", rerank=toggle
        )
        assert res, f"rerank={toggle} should not break hybrid retrieval"


def test_hybrid_planner_retrieve(service):
    out = service.planner_retrieve("配置代理", strategy="hybrid")
    assert "actions" in out
