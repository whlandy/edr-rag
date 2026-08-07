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


@pytest.mark.parametrize("strategy", ["keyword", "bm25", "vector", "hybrid"])
def test_planner_retrieve_strategy_threads_through(service, strategy):
    # P0-3: planner_retrieve must accept a strategy and thread it to every tier.
    result = service.planner_retrieve("配置代理", strategy=strategy)
    assert result["query"] == "配置代理"
    assert result["actions"], f"planner_retrieve({strategy}) should recall proxy action"
    titles = [r["title"] for r in result["actions"]]
    assert "配置代理服务器" in titles, f"planner_retrieve({strategy}) missing proxy action"
    assert result["procedures"] == []
    assert result["sections"] == []


def test_planner_retrieve_invalid_strategy_raises(service):
    with pytest.raises(ValueError):
        service.planner_retrieve("配置代理", strategy="fuzzy")


def test_edr_rag_api_delegates_to_service(tmp_path):
    # Regression: pure API (EdrRagAPI) must route through the shared service.
    from edr_rag.p2.api import EdrRagAPI
    _write_artifacts(tmp_path)
    api = EdrRagAPI(tmp_path)
    assert api.action_search("配置代理")
    assert api.get_action("network.configure_proxy")
    planner = api.planner_retrieve("配置代理")
    assert planner["actions"]


def test_edr_rag_api_accepts_strategy(tmp_path):
    # P0-3: every API entry point accepts and forwards a strategy.
    from edr_rag.p2.api import EdrRagAPI
    _write_artifacts(tmp_path)
    api = EdrRagAPI(tmp_path)

    # action_search
    assert api.action_search("配置代理", strategy="bm25")
    # procedure_search / manual_search (strategy + scoring alias)
    assert api.procedure_search("配置代理", strategy="bm25")
    assert api.manual_search("配置代理", strategy="keyword")
    # planner_retrieve
    pl = api.planner_retrieve("配置代理", strategy="hybrid")
    assert pl["actions"]

    # invalid strategy rejected at every level
    with pytest.raises(ValueError):
        api.action_search("配置代理", strategy="fuzzy")
    with pytest.raises(ValueError):
        api.planner_retrieve("配置代理", strategy="fuzzy")


# ---------- P0-4: action metadata filter moved to retrieval stage + recall budget ----------

def _write_budget_fixture(root: Path, n: int = 12, approved_idx: int = -1):
    """写入 n 个动作，全部匹配 query「配置」，仅 approved_idx 一个为 approved。

    用于构造「严格过滤器 + 候选池不足」回归场景：高排名动作全部是
    needs_review，唯一 approved 的动作排在评分末尾。
    """
    actions = []
    for i in range(n):
        actions.append({
            "action_id": f"net.configure_{i}",
            "product": "EDR",
            "module": "net",
            "title": f"配置代理服务器 {i}",
            "status": "approved" if i == approved_idx else "needs_review",
            "confidence": 0.9,
            "risk": "medium" if i == approved_idx else "high",
            "requires_confirmation": True,
            "preconditions": [],
            "procedure": [{"order": 1, "action": "click", "target": "配置"}],
            "success_criteria": [],
            "recovery": [],
        })
    (root / "actions.jsonl").write_text(
        "\n".join(json.dumps(a, ensure_ascii=False) for a in actions) + "\n",
        encoding="utf-8",
    )


@pytest.mark.parametrize("strategy", ["keyword", "bm25", "vector"])
def test_action_strict_filter_recalls_beyond_budget(tmp_path, strategy):
    # 20 actions; only the lowest-ranked one (idx 19) is approved.
    # Initial budget = top_k*recall_budget = 3*4 = 12 < 20, so the LOOP is
    # what expands the candidate pool and recalls the approved action.
    # (Old code capped at top_k, or at a fixed 4x=12 budget -> filtered = 0.)
    _write_budget_fixture(tmp_path, n=20, approved_idx=19)
    svc = RetrievalService(tmp_path)

    # Strict status filter that only the lowest-ranked action satisfies.
    results = svc.search(
        "配置", corpus_type="actions", strategy=strategy,
        filters={"status": "approved"}, top_k=3,
    )
    assert len(results) == 1, f"{strategy}: strict filter must still recall the approved action"
    assert results[0]["id"] == "net.configure_19"
    assert results[0]["status"] == "approved"


def test_action_strict_filter_exhausts_without_infinite_loop(tmp_path):
    # Approved action exists but request top_k larger than matches -> returns
    # whatever matches, no infinite loop, and stops at corpus exhaustion.
    _write_budget_fixture(tmp_path, n=8, approved_idx=5)
    svc = RetrievalService(tmp_path)
    results = svc.search(
        "配置", corpus_type="actions", strategy="keyword",
        filters={"status": "approved"}, top_k=10,
    )
    # only one approved exists even though top_k=10
    assert len(results) == 1
    assert results[0]["id"] == "net.configure_5"


def test_action_no_filter_unchanged_by_recall_budget(tmp_path):
    # Without filters, recall_budget must NOT change the returned count / order.
    _write_budget_fixture(tmp_path, n=12, approved_idx=11)
    svc = RetrievalService(tmp_path)
    plain = svc.search("配置", corpus_type="actions", strategy="keyword", top_k=3)
    assert len(plain) == 3
    # top-3 scored actions, all needs_review -> none approved
    assert all(r["status"] == "needs_review" for r in plain)


def test_action_recall_budget_parameter_is_configurable(tmp_path):
    # recall_budget must be accepted end-to-end via both service and API.
    from edr_rag.p2.api import EdrRagAPI
    _write_budget_fixture(tmp_path, n=20, approved_idx=19)
    svc = RetrievalService(tmp_path)
    # service level
    r1 = svc.search(
        "配置", corpus_type="actions", strategy="keyword",
        filters={"status": "approved"}, top_k=3, recall_budget=1,
    )
    # recall_budget=1 -> initial budget=3, loop expands to 6->12->20
    assert r1[0]["id"] == "net.configure_19"
    # API level threads recall_budget through
    api = EdrRagAPI(tmp_path)
    r2 = api.action_search(
        "配置", limit=3, status_filter="approved", strategy="keyword",
        recall_budget=4,
    )
    assert r2 and r2[0]["id"] == "net.configure_19"
    # recall_budget clamped to >=1, never crashes
    r3 = api.action_search(
        "配置", limit=3, status_filter="approved", strategy="keyword",
        recall_budget=0,
    )
    assert r3 and r3[0]["id"] == "net.configure_19"

