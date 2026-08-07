"""P0-3 tests: MCP search flow with configurable strategy.

Verifies that the P3 MCP search entry points (search_sections / search_procedures /
search_actions / search_chunks) route through the unified RetrievalService and
accept a configurable `strategy` (keyword / bm25 / vector / hybrid):

  - each search method returns results and honours strategy (incl. graceful
    degradation of vector/hybrid to bm25 without optional RAG deps)
  - an invalid strategy raises a clear error
  - the MCP tool dispatch (create_mcp_server handle_call_tool) forwards strategy
    and returns JSON text, when the optional `mcp` SDK is available
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

# Reuse the same artifact fixtures as the P0-2 retrieval-service tests so the
# agent (which wraps RetrievalService) can search them.
from edr_rag.tests.test_retrieval_service import (
    ACTION_A,
    ACTION_B,
    SECTION,
    _write_artifacts,
)
from edr_rag.p3.server import EdrRagAgent

STRATEGIES = ("keyword", "bm25", "vector", "hybrid")


@pytest.fixture
def agent(tmp_path: Path) -> EdrRagAgent:
    _write_artifacts(tmp_path)
    return EdrRagAgent(str(tmp_path))


# ----- Agent search methods route via RetrievalService & accept strategy -----

@pytest.mark.parametrize("strategy", STRATEGIES)
def test_search_sections_strategy(agent: EdrRagAgent, strategy: str):
    results = agent.search_sections("配置代理", strategy=strategy)
    assert isinstance(results, list)
    assert results, f"search_sections({strategy}) should recall the section"
    assert results[0]["id"] == "s1"


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_search_procedures_strategy(agent: EdrRagAgent, strategy: str):
    results = agent.search_procedures("配置代理", strategy=strategy)
    assert isinstance(results, list)
    assert results, f"search_procedures({strategy}) should recall the procedure"
    assert results[0]["id"] == "procedure.network.configure_proxy"


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_search_actions_strategy(agent: EdrRagAgent, strategy: str):
    results = agent.search_actions("配置代理", strategy=strategy)
    assert isinstance(results, list)
    assert results, f"search_actions({strategy}) should recall the proxy action"
    # Every strategy must recall the proxy action (by title).
    titles = [r["title"] for r in results]
    assert "配置代理服务器" in titles, f"search_actions({strategy}) missing proxy action"
    # The rich action id (action_id) is only populated on the keyword path;
    # bm25/vector/hybrid keep the generic retrieval shape (documented P0-2 behaviour).
    if strategy == "keyword":
        assert results[0]["id"] == "network.configure_proxy"
    else:
        assert "network.configure_proxy" not in [r["id"] for r in results]


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_search_chunks_strategy(agent: EdrRagAgent, strategy: str):
    results = agent.search_chunks("代理", strategy=strategy)
    assert isinstance(results, list)
    assert results, f"search_chunks({strategy}) should recall the chunk"
    assert results[0]["id"] == "c1"


def test_search_default_strategy_is_keyword(agent: EdrRagAgent):
    # Default (no strategy arg) matches keyword behaviour.
    default = agent.search_actions("配置")
    explicit = agent.search_actions("配置", strategy="keyword")
    assert [r["id"] for r in default] == [r["id"] for r in explicit]


def test_search_invalid_strategy_raises(agent: EdrRagAgent):
    with pytest.raises(ValueError):
        agent.search_actions("配置代理", strategy="fuzzy")


# ----- MCP search flow returns JSON-text results (agent-level ser round-trip) -----

def test_mcp_search_flow_returns_serializable_results(agent: EdrRagAgent):
    # The MCP transport serializes results to JSON text; ensure what the agent
    # returns for a strategy search is JSON-serializable.
    for method, q, kws in (
        (agent.search_sections, "代理", {}),
        (agent.search_procedures, "代理", {}),
        (agent.search_actions, "代理", {"strategy": "bm25"}),
        (agent.search_chunks, "代理", {"strategy": "vector"}),
    ):
        payload = method(q, limit=3, **kws)
        dumped = json.dumps(payload, ensure_ascii=False)
        assert json.loads(dumped) == payload


# ----- MCP tool dispatch (requires optional `mcp` SDK) -----

mcp = pytest.importorskip("mcp", reason="optional mcp SDK not installed")
from mcp.types import CallToolRequest, ListToolsRequest  # noqa: E402
from edr_rag.p3.server import create_mcp_server, MCP_AVAILABLE  # noqa: E402


def _run(coro):
    import asyncio

    return asyncio.run(coro)


@pytest.mark.skipif(not MCP_AVAILABLE, reason="mcp SDK not available")
def test_mcp_server_lists_strategy_on_search_tools():
    app = create_mcp_server(".")
    res = _run(app.request_handlers[ListToolsRequest](ListToolsRequest()))
    tools = {t.name: t for t in res.root.tools}
    for tool_name in ("search_knowledge", "search_procedures", "search_actions"):
        assert tool_name in tools, f"{tool_name} tool missing"
        props = tools[tool_name].inputSchema["properties"]
        assert "strategy" in props, f"{tool_name} schema missing strategy"
        assert props["strategy"]["default"] == "keyword"
        assert props["strategy"]["enum"] == ["keyword", "bm25", "vector", "hybrid"]


@pytest.mark.skipif(not MCP_AVAILABLE, reason="mcp SDK not available")
def test_mcp_server_search_dispatch_forwards_strategy(agent: EdrRagAgent):
    app = create_mcp_server(str(agent.workdir))
    handler = app.request_handlers[CallToolRequest]
    calls = {
        "search_knowledge": ("代理", {"limit": 3, "strategy": "bm25"}),
        "search_procedures": ("代理", {"limit": 3, "strategy": "keyword"}),
        "search_actions": ("代理", {"limit": 3, "strategy": "hybrid"}),
    }

    def _invoke(name: str, args: dict):
        req = CallToolRequest(params={"name": name, "arguments": args})
        res = _run(handler(req))
        return [c.text for c in res.root.content if hasattr(c, "text")]

    for name, (q, extra) in calls.items():
        args = {"query": q, **extra}
        texts = _invoke(name, args)
        assert texts, f"{name} dispatch returned no content"
        parsed = json.loads(texts[0])
        assert parsed, f"{name} dispatch returned empty JSON results"
