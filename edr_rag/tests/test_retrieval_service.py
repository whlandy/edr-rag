"""P0-2 tests: unified RetrievalService.

Verifies the single retrieval entry point that backs the pure API (EdrRagAPI),
the P2 CLI, and the MCP agent (EdrRagAgent):
  - search across each corpus_type (sections / procedures / actions / chunks)
  - strategy keyword / bm25 / vector (vector degrades to bm25 w/o optional deps)
  - action filters (risk / status) and exact-id get-mode
  - get() exact lookup (action_id / procedure id)
  - planner_retrieve preserves action -> procedure -> manual priority
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

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

SECTION = {
    "id": "s1",
    "title": "配置代理服务器",
    "content": "本节介绍如何配置代理，需要网络设置。",
    "section_path": ["网络", "代理"],
    "source": {"file": "net/proxy.html"},
}


def _write_artifacts(root: Path):
    (root / "actions.jsonl").write_text(
        "\n".join(json.dumps(a, ensure_ascii=False) for a in [ACTION_A, ACTION_B]) + "\n",
        encoding="utf-8",
    )
    (root / "sections.jsonl").write_text(
        json.dumps(SECTION, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    proc = {
        "id": "procedure.network.configure_proxy",
        "manual_id": "m1",
        "product": "EDR",
        "operation": "配置代理",
        "title": "配置代理服务器",
        "steps": [{"order": 1, "instruction": "打开网络设置"}, {"order": 2, "instruction": "输入代理地址"}],
    }
    (root / "procedures.jsonl").write_text(
        json.dumps(proc, ensure_ascii=False) + "\n", encoding="utf-8",
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


def test_search_actions_rich_shape(service):
    results = service.search("配置代理", corpus_type="actions")
    assert results, "should find the proxy action"
    r = results[0]
    # Rich action result shape.
    assert r["id"] == "network.configure_proxy"
    assert r["title"] == "配置代理服务器"
    assert r["risk"] == "medium"
    assert r["status"] == "approved"
    assert "score" in r and "snippet" in r


def test_search_actions_top_k(service):
    results = service.search("配置", corpus_type="actions", top_k=1)
    assert len(results) == 1


def test_search_actions_risk_filter(service):
    results = service.search("配置", corpus_type="actions", filters={"risk": "high"})
    assert len(results) == 1
    assert results[0]["id"] == "system.reset_system"


def test_search_actions_status_filter(service):
    results = service.search("配置", corpus_type="actions", filters={"status": "approved"})
    assert all(r["status"] == "approved" for r in results)


def test_search_actions_exact_id(service):
    results = service.search(
        "whatever", corpus_type="actions", filters={"id": "system.reset_system"}
    )
    assert len(results) == 1
    assert results[0]["id"] == "system.reset_system"


def test_search_actions_bm25(service):
    results = service.search("配置代理", corpus_type="actions", strategy="bm25")
    assert results


def test_search_actions_vector_degrades_to_bm25(service):
    # Optional RAG deps absent in CI -> gracefully degrades, no error.
    results = service.search("配置代理", corpus_type="actions", strategy="vector")
    assert isinstance(results, list)


@pytest.mark.parametrize("corpus_type,query", [
    ("sections", "配置代理"),
    ("procedures", "配置代理"),
    ("chunks", "代理"),
])
def test_search_generic_corpora(service, corpus_type, query):
    results = service.search(query, corpus_type=corpus_type)
    assert results, f"expected results for {corpus_type}"
    assert "id" in results[0] and "score" in results[0]


def test_search_missing_corpus(tmp_path):
    svc = RetrievalService(tmp_path)  # no artifacts at all
    assert svc.search("代理", corpus_type="sections") == []
    assert svc.search("代理", corpus_type="actions") == []
    assert svc.get("actions", "x") is None


def test_search_unknown_corpus_raises(service):
    with pytest.raises(ValueError):
        service.search("q", corpus_type="nope")


def test_get_action(service):
    rec = service.get("actions", "procedure.system.reset_system", key_field="action_id")
    assert rec is not None
    assert rec["action_id"] == "system.reset_system"


def test_get_action_missing(service):
    assert service.get("actions", "does.not.exist") is None


def test_get_procedure(service):
    rec = service.get("procedures", "procedure.network.configure_proxy")
    assert rec is not None
    assert rec["id"] == "procedure.network.configure_proxy"


def test_planner_retrieve_priority_action_first(service):
    result = service.planner_retrieve("配置代理")
    assert result["query"] == "配置代理"
    # Actions matched -> priority 1 used, no fallback tiers.
    assert result["actions"]
    assert result["procedures"] == []
    assert result["sections"] == []


def test_planner_retrieve_no_action_falls_back(service):
    # A query only present in sections, not in actions/procedures.
    result = service.planner_retrieve("本节")
    assert not result["actions"]
    assert not result["procedures"]
    assert result["sections"]  # section content contains "本节"


def test_edr_rag_api_delegates_to_service(tmp_path):
    # Regression: pure API (EdrRagAPI) must route through the shared service.
    from edr_rag.p2.api import EdrRagAPI
    _write_artifacts(tmp_path)
    api = EdrRagAPI(tmp_path)
    assert api.action_search("配置代理")
    assert api.get_action("network.configure_proxy")
    planner = api.planner_retrieve("配置代理")
    assert planner["actions"]
