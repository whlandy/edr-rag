""""P2-5 高风险 Action 显式确认门控测试。

覆盖:
  - confirmation 策略纯函数（needs_explicit_confirmation / gate_answer 三态）
  - API ask_answer 的 confirm 门控：required / confirmed / not_required
  - MCP agent ask_answer 透传 confirm
  - MCP ask 工具 schema 暴露 confirm 字段并转发到 agent
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from edr_rag.p2 import confirmation as cmod


# ---------- 纯策略函数 ----------

def _sample_answer(*, requires_confirmation=True, answer_status="answered"):
    return {
        "answer": "请按步骤执行",
        "answer_status": answer_status,
        "requires_confirmation": requires_confirmation,
        "dangerous": requires_confirmation or answer_status == "conflict",
        "steps": [{"order": 1, "action": "click", "target": "重置"}],
        "preconditions": ["管理员权限"],
        "warnings": ["高危"],
        "assertions": [],
        "citations": [
            {"kind": "action", "id": "system.reset_system",
             "title": "重置系统", "source": "reset.html"}
        ],
    }


def test_needs_explicit_confirmation_true_for_requires_confirmation():
    assert cmod.needs_explicit_confirmation(_sample_answer(requires_confirmation=True)) is True


def test_needs_explicit_confirmation_true_for_conflict():
    ans = _sample_answer(requires_confirmation=False, answer_status="conflict")
    assert cmod.needs_explicit_confirmation(ans) is True


def test_needs_explicit_confirmation_false_when_safe():
    ans = _sample_answer(requires_confirmation=False, answer_status="answered")
    assert cmod.needs_explicit_confirmation(ans) is False


def test_gate_answer_not_required_returns_as_is():
    ans = _sample_answer(requires_confirmation=False, answer_status="answered")
    out = cmod.gate_answer(ans, confirm=False)
    assert out["confirmation_status"] == cmod.CONFIRMATION_NOT_REQUIRED
    # 不需要确认时步骤照常放行
    assert out["steps"]


def test_gate_answer_required_without_confirm_withholds_steps():
    ans = _sample_answer(requires_confirmation=True)
    out = cmod.gate_answer(ans, confirm=False)
    assert out["confirmation_status"] == cmod.CONFIRMATION_REQUIRED
    assert out["needs_confirmation"] is True
    # 可执行内容被扣留
    assert out["steps"] == []
    assert out["preconditions"] == []
    assert "confirmation_request" in out
    # 基础语义字段保留，方便消费方决策
    assert out["answer_status"] == "answered"
    assert out["requires_confirmation"] is True
    # 不原地修改入参
    assert ans["steps"]


def test_gate_answer_confirmed_releases_full_steps():
    ans = _sample_answer(requires_confirmation=True)
    out = cmod.gate_answer(ans, confirm=True)
    assert out["confirmation_status"] == cmod.CONFIRMATION_CONFIRMED
    assert out["needs_confirmation"] is True
    assert out["steps"], "确认后应放行完整步骤"


# ---------- 端到端：API + MCP ----------

def _build_workdir() -> Path:
    from edr_rag import normalize, extract as extract_mod
    from edr_rag.p1 import extract_procedures as ep
    from edr_rag.p2 import extract_actions as ea
    from edr_rag.p2.test_smoke import _build_fixture, _patch_extract

    tmp = Path(tempfile.mkdtemp())
    fixture = _build_fixture(tmp)
    _patch_extract(fixture)
    workdir = tmp / "workdir"
    normalize.run_ingest(tmp / "fake.chm", "test-manual", "TestProduct", workdir)
    all_procs, all_chunks = ep.extract_procedures_from_file(
        workdir / "sections.jsonl", product="TestProduct"
    )
    ep.write_p1_artifacts(
        workdir, all_procs, all_chunks,
        [json.loads(l) for l in (workdir / "sections.jsonl").read_text("utf-8")
         .splitlines() if l.strip()],
        manual_id="test-manual", product="TestProduct",
    )
    entries = ea.build_actions_from_file(workdir / "procedures.jsonl", product="TestProduct")
    ea.write_p2_artifacts(workdir, entries)
    return workdir


def test_api_ask_answer_default_legacy_not_gated():
    """confirm=None（默认）保持向后兼容：返回完整带标记答案。"""
    from edr_rag.p2.api import EdrRagAPI

    api = EdrRagAPI(_build_workdir())
    result = api.ask_answer("配置代理", limit=5)
    assert result["steps"], "默认（兼容）模式不应扣留步骤"
    assert result["requires_confirmation"] is True  # 配置属 medium 风险
    assert result["confirmation_status"] == cmod.CONFIRMATION_REQUIRED


def test_api_ask_answer_confirm_false_gates_high_risk():
    from edr_rag.p2.api import EdrRagAPI

    api = EdrRagAPI(_build_workdir())
    result = api.ask_answer("配置代理", limit=5, confirm=False)
    assert result["confirmation_status"] == cmod.CONFIRMATION_REQUIRED
    assert result["steps"] == [], "未确认时应扣留可执行步骤"
    assert "confirmation_request" in result


def test_api_ask_answer_confirm_true_releases():
    from edr_rag.p2.api import EdrRagAPI

    api = EdrRagAPI(_build_workdir())
    unconfirmed = api.ask_answer("配置代理", limit=5, confirm=False)
    confirmed = api.ask_answer("配置代理", limit=5, confirm=True)
    assert confirmed["confirmation_status"] == cmod.CONFIRMATION_CONFIRMED
    assert confirmed["steps"], "确认后应放行完整步骤"
    assert len(confirmed["steps"]) >= len(unconfirmed["steps"])


def test_api_ask_answer_safe_query_not_gated_even_when_confirm_false():
    """无高风险动作时即使 confirm=False 也不门控。"""
    from edr_rag.p2.api import EdrRagAPI

    api = EdrRagAPI(_build_workdir())
    result = api.ask_answer("完全不存在的东西xyz", limit=5, confirm=False)
    assert result["confirmation_status"] == cmod.CONFIRMATION_NOT_REQUIRED
    assert result["dangerous"] is False


def test_mcp_agent_ask_answer_passes_confirm():
    from edr_rag.p3.server import EdrRagAgent

    agent = EdrRagAgent(str(_build_workdir()))
    gated = agent.ask_answer("配置代理", limit=5, confirm=False)
    assert gated["confirmation_status"] == cmod.CONFIRMATION_REQUIRED
    assert gated["steps"] == []
    released = agent.ask_answer("配置代理", limit=5, confirm=True)
    assert released["confirmation_status"] == cmod.CONFIRMATION_CONFIRMED
    assert released["steps"]


def test_cli_answer_confirm_gating_json():
    """CLI answer --confirm false 应输出门控答案 JSON（含 confirmation 元数据）。"""
    import subprocess
    import sys as _sys

    workdir = _build_workdir()
    proc = subprocess.run(
        [_sys.executable, "-m", "edr_rag.p2.cli", "answer",
         "--query", "配置代理", "--workdir", str(workdir), "--json",
         "--confirm", "false"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["confirmation_status"] == cmod.CONFIRMATION_REQUIRED
    assert data["steps"] == []


def test_cli_answer_confirm_text_render_does_not_crash():
    """CLI answer --confirm 走文本渲染路径，不应因 dataclass 未知 kwargs 崩溃。"""
    import subprocess
    import sys as _sys

    workdir = _build_workdir()
    proc = subprocess.run(
        [_sys.executable, "-m", "edr_rag.p2.cli", "answer",
         "--query", "配置代理", "--workdir", str(workdir),
         "--confirm", "true"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "confirmation_status=confirmed" in proc.stdout


# ---------- MCP ask 工具 schema 暴露 confirm 字段 ----------

mcp = pytest.importorskip("mcp", reason="optional mcp SDK not installed")
from mcp.types import CallToolRequest, ListToolsRequest  # noqa: E402
from edr_rag.p3.server import create_mcp_server, MCP_AVAILABLE  # noqa: E402


def _run(coro):
    import asyncio
    return asyncio.run(coro)


@pytest.mark.skipif(not MCP_AVAILABLE, reason="mcp SDK not available")
def test_mcp_ask_tool_schema_has_confirm_field():
    app = create_mcp_server(".")
    res = _run(app.request_handlers[ListToolsRequest](ListToolsRequest()))
    tools = {t.name: t for t in res.root.tools}
    assert "ask" in tools, "ask tool missing"
    props = tools["ask"].inputSchema["properties"]
    assert "confirm" in props, "ask tool schema missing confirm field"
    assert props["confirm"]["type"] == "boolean"
    assert props["confirm"].get("default") is False


@pytest.mark.skipif(not MCP_AVAILABLE, reason="mcp SDK not available")
def test_mcp_ask_tool_dispatch_forwards_confirm():
    app = create_mcp_server(str(_build_workdir()))
    handler = app.request_handlers[CallToolRequest]

    def _invoke(args: dict):
        req = CallToolRequest(params={"name": "ask", "arguments": args})
        res = _run(handler(req))
        return json.loads([c.text for c in res.root.content if hasattr(c, "text")][0])

    gated = _invoke({"query": "配置代理", "limit": 5})
    assert gated["confirmation_status"] == cmod.CONFIRMATION_REQUIRED
    assert gated["steps"] == [], "MCP 默认未确认应门控高风险 Action"

    released = _invoke({"query": "配置代理", "limit": 5, "confirm": True})
    assert released["confirmation_status"] == cmod.CONFIRMATION_CONFIRMED
    assert released["steps"], "confirm=true 应放行完整步骤"
