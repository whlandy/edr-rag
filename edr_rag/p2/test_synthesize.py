"""P2-1 结构化答案合成器测试。

覆盖:
  - 结构化字段提取（steps/warnings/preconditions/citations/confidence/requires_confirmation）
  - 组装模式（无 LLM）answer 生成
  - LLM 模式（注入 llm 回调）answer 生成
  - LLM 调用异常回落到组装模式
  - 输入形状兼容（planner_retrieve dict / list[dict] / 空上下文）
  - to_jsonl / answer_to_text 序列化
"""

from edr_rag.p2.synthesize import (
    SynthesizedAnswer,
    answer_to_text,
    synthesize_answer,
)


def _make_action(**overrides):
    action = {
        "id": "网络.代理.配置代理服务器",
        "action_id": "网络.代理.配置代理服务器",
        "title": "配置代理服务器",
        "risk": "medium",
        "status": "candidate",
        "confidence": 0.85,
        "requires_confirmation": True,
        "preconditions": ["具备网络管理员权限", "已打开网络设置"],
        "procedure": [
            {"order": 1, "action": "open_window", "target": "网络设置", "source_instruction": "打开网络设置"},
            {"order": 2, "action": "select", "target": "代理", "source_instruction": "选择代理"},
            {"order": 3, "action": "input_text", "target": "代理地址", "source_instruction": "输入代理地址"},
        ],
        "recovery": ["如需撤销，可关闭代理开关"],
        "score": 3.0,
    }
    action.update(overrides)
    return action


# ---------- 结构化字段提取 ----------


def test_synthesize_extracts_structured_fields():
    ctx = {"actions": [_make_action()], "procedures": [], "sections": []}
    result = synthesize_answer("如何配置代理", ctx)

    assert len(result.steps) == 3
    assert result.steps[0]["action"] == "open_window"
    assert result.preconditions == ["具备网络管理员权限", "已打开网络设置"]
    assert any("代理" in w or "确认" in w for w in result.warnings)
    assert result.requires_confirmation is True
    assert result.confidence == 0.85
    assert len(result.citations) == 1
    assert result.citations[0]["id"] == "网络.代理.配置代理服务器"


def test_synthesize_assemble_answer_no_llm():
    ctx = {"actions": [_make_action()], "procedures": [], "sections": []}
    result = synthesize_answer("如何配置代理", ctx)
    assert result.generated_by == "assemble"
    assert "配置代理服务器" in result.answer
    assert "操作步骤" in result.answer
    assert "3. 输入代理地址" in result.answer
    assert "注意事项" in result.answer


def test_synthesize_with_llm_uses_generated_answer():
    ctx = {"actions": [_make_action()], "procedures": [], "sections": []}
    result = synthesize_answer("如何配置代理", ctx, llm=lambda q, c: f"LLM回答：{q}")
    assert result.generated_by == "llm"
    assert result.answer == "LLM回答：如何配置代理"
    # 结构化字段仍然保留
    assert len(result.steps) == 3
    assert result.citations


def test_synthesize_llm_exception_falls_back_to_assemble():
    def broken_llm(q, c):
        raise RuntimeError("llm down")

    ctx = {"actions": [_make_action()], "procedures": [], "sections": []}
    result = synthesize_answer("如何配置代理", ctx, llm=broken_llm)
    assert result.generated_by == "assemble"
    assert "配置代理服务器" in result.answer


def test_synthesize_empty_context_no_crash():
    result = synthesize_answer("不存在的东西", {"actions": [], "procedures": [], "sections": []})
    assert result.generated_by == "assemble"
    assert "未检索到" in result.answer
    assert result.steps == []
    assert result.requires_confirmation is False
    assert result.confidence == 0.0


def test_synthesize_accepts_action_list_shape():
    result = synthesize_answer("如何配置代理", [_make_action()])
    assert len(result.steps) == 3
    assert result.citations


def test_synthesize_requires_confirmation_aggregation():
    # 只有一个 risk=low 且不要求确认的动作
    action = _make_action(risk="low", requires_confirmation=False, status="candidate")
    ctx = {"actions": [action], "procedures": [], "sections": []}
    result = synthesize_answer("查询状态", ctx)
    assert result.requires_confirmation is False


def test_synthesize_high_risk_adds_warning():
    action = _make_action(risk="high", requires_confirmation=True)
    ctx = {"actions": [action], "procedures": [], "sections": []}
    result = synthesize_answer("重置系统", ctx)
    assert any("high" in w or "高风险" in w for w in result.warnings)


# ---------- 序列化与展示 ----------


def test_to_jsonl_has_expected_keys():
    ctx = {"actions": [_make_action()], "procedures": [], "sections": []}
    result = synthesize_answer("如何配置代理", ctx)
    d = result.to_jsonl()
    for key in ("answer", "steps", "warnings", "preconditions",
                "citations", "confidence", "requires_confirmation"):
        assert key in d


def test_answer_to_text_renders_citations():
    ctx = {"actions": [_make_action()], "procedures": [], "sections": []}
    result = synthesize_answer("如何配置代理", ctx)
    text = answer_to_text(result)
    assert "引用来源" in text
    assert "配置代理服务器" in text


# ---------- API 集成（真实检索路径 + 全量动作补全） ----------

def test_api_synthesize_extracts_full_steps():
    """通过 EdrRagAPI.synthesize 走真实检索，验证动作全量字段被补全。"""
    import json
    import shutil
    import sys
    import tempfile
    from pathlib import Path

    from edr_rag import normalize, extract as extract_mod
    from edr_rag.p1 import extract_procedures as ep
    from edr_rag.p2 import extract_actions as ea
    from edr_rag.p2.test_smoke import _build_fixture, _patch_extract
    from edr_rag.p2 import api as papi

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
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

        api = papi.EdrRagAPI(workdir)
        result = api.synthesize("配置代理", limit=5)

        assert result["generated_by"] == "assemble"
        assert result["requires_confirmation"] is True
        # 通过补全，应能提取到真实 procedure 步骤（非单步兜底）
        assert len(result["steps"]) >= 3
        assert any(s["action"] == "open_window" for s in result["steps"])
        assert any(s["action"] == "input_text" for s in result["steps"])


def test_api_synthesize_with_llm():
    """API 路径也支持注入 llm 回调。"""
    from edr_rag.p2.synthesize import synthesize_answer

    ctx = {"actions": [_make_action()], "procedures": [], "sections": []}
    result = synthesize_answer("如何配置代理", ctx, llm=lambda q, c: f"LLM：{q}")
    assert result.generated_by == "llm"

