"""P2-4 AskAnswer 结构化输出打通测试。

覆盖:
  - EdrRagAPI.ask_answer 返回结构化字段 + agent 便捷字段 (dangerous/sources)
  - 无命中时 ask_answer 拒答且 dangerous 反映冲突/确认状态
  - MCP agent (EdrRagAgent.ask_answer) 也能返回同一结构化答案
  - CLI answer --json 输出完整结构化字段（回归）
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path


def _build_workdir() -> Path:
    """构建 P0→P2 完整 artifacts 的工作目录。"""
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


def test_ask_answer_returns_structured_with_agent_fields():
    from edr_rag.p2.api import EdrRagAPI

    workdir = _build_workdir()
    api = EdrRagAPI(workdir)
    result = api.ask_answer("配置代理", limit=5)

    # 结构化核心字段
    assert result["answer_status"] in ("answered", "refused", "conflict")
    assert isinstance(result["steps"], list)
    assert isinstance(result["preconditions"], list)
    assert isinstance(result["warnings"], list)
    # agent 便捷字段
    assert "dangerous" in result
    assert isinstance(result["dangerous"], bool)
    assert "sources" in result
    assert isinstance(result["sources"], list)
    # 曾有动作命中
    assert result["steps"], "应当检索并合成出步骤"


def test_ask_answer_dangerous_reflects_confirmation():
    from edr_rag.p2.api import EdrRagAPI

    workdir = _build_workdir()
    api = EdrRagAPI(workdir)
    result = api.ask_answer("重置系统", limit=5)
    # 重置是高危操作 → requires_confirmation 应为 True
    if result["steps"]:
        assert result["requires_confirmation"] is True
        assert result["dangerous"] is True


def test_ask_answer_no_hit_refuses_not_dangerous():
    from edr_rag.p2.api import EdrRagAPI

    workdir = _build_workdir()
    api = EdrRagAPI(workdir)
    result = api.ask_answer("完全不存在的东西xyz", limit=5)
    # 无命中 → 拒答；require_confirmation=False → 不 dangerous
    assert result["answer_status"] == "refused"
    assert result["steps"] == []
    assert result["dangerous"] is False


def test_mcp_agent_ask_answer_works():
    from edr_rag.p3.server import EdrRagAgent

    workdir = _build_workdir()
    agent = EdrRagAgent(str(workdir))
    result = agent.ask_answer("配置代理", limit=5)
    assert result["answer_status"] in ("answered", "refused", "conflict")
    assert "steps" in result
    assert "sources" in result
    assert "dangerous" in result


def test_cli_answer_json_emits_structured_fields():
    """CLI answer --json 应输出完整结构化字段（回归 P2-1 入口）。"""
    import subprocess
    import sys as _sys

    workdir = _build_workdir()
    proc = subprocess.run(
        [_sys.executable, "-m", "edr_rag.p2.cli", "answer",
         "--query", "配置代理", "--workdir", str(workdir), "--json"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    for key in ("answer", "steps", "warnings", "preconditions",
                "citations", "confidence", "requires_confirmation",
                "answer_status", "generated_by"):
        assert key in data, f"missing {key} in CLI JSON output"
