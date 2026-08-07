"""P2-3 证据不足拒答 + 冲突处理测试。

覆盖:
  - 无证据（空上下文）-> answer_status='refused'，refusal_reason='insufficient_evidence'，
    明确拒答（不编造），且 answer 仍含「未检索到」提示。
  - 置信度低于阈值 -> answer_status='refused'，refusal_reason='low_confidence'。
  - 自定义 min_confidence 阈值生效。
  - 互斥操作冲突（同一目标 启用 vs 禁用）-> answer_status='conflict'，
    conflicts 带 type/description/sources，requires_confirmation=True，请求澄清。
  - 步骤级目标冲突检测。
  - 风险等级矛盾检测。
  - 正常可作答 -> answer_status='answered'，无 conflicts。
  - answer_to_text 渲染拒答/冲突标注。
  - to_jsonl 包含新字段。
"""

from edr_rag.p2.synthesize import (
    ANSWER_ANSWERED,
    ANSWER_CONFLICT,
    ANSWER_REFUSED,
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
        "preconditions": ["具备网络管理员权限"],
        "procedure": [
            {"order": 1, "action": "open_window", "target": "网络设置", "source_instruction": "打开网络设置"},
            {"order": 2, "action": "input_text", "target": "代理地址", "source_instruction": "输入代理地址"},
        ],
        "recovery": ["如需撤销，可关闭代理开关"],
        "source": {"manual_id": "m", "procedure_id": "procedure.网络.代理", "file": "network/proxy.html"},
        "score": 3.0,
    }
    action.update(overrides)
    return action


# ---------- 证据不足拒答 ----------


def test_empty_context_refuses():
    result = synthesize_answer("不存在的东西", {"actions": [], "procedures": [], "sections": []})
    assert result.answer_status == ANSWER_REFUSED
    assert result.refusal_reason == "insufficient_evidence"
    # 不编造：明确拒答提示
    assert "未检索到" in result.answer
    assert "证据不足" in result.answer
    assert result.steps == []
    assert result.confidence == 0.0


def test_low_confidence_refuses():
    action = _make_action(confidence=0.2)
    ctx = {"actions": [action], "procedures": [], "sections": []}
    result = synthesize_answer("如何配置代理", ctx)
    assert result.answer_status == ANSWER_REFUSED
    assert result.refusal_reason == "low_confidence"
    assert "置信度过低" in result.answer


def test_custom_min_confidence_threshold():
    # 默认 0.4 阈值下 0.5 是可作答的
    action = _make_action(confidence=0.5)
    ctx = {"actions": [action], "procedures": [], "sections": []}
    result = synthesize_answer("如何配置代理", ctx)
    assert result.answer_status == ANSWER_ANSWERED
    # 调高阈值后 -> 拒答
    result2 = synthesize_answer("如何配置代理", ctx, min_confidence=0.9)
    assert result2.answer_status == ANSWER_REFUSED
    assert result2.refusal_reason == "low_confidence"


def test_min_confidence_zero_disables_refusal():
    # min_confidence <= 0 表示关闭拒答（保留旧行为），即使空上下文也不拒答
    result = synthesize_answer("不存在的东西", {"actions": [], "procedures": [], "sections": []},
                               min_confidence=0.0)
    assert result.answer_status == ANSWER_ANSWERED
    assert "未检索到" in result.answer


def test_high_confidence_answers():
    action = _make_action(confidence=0.9, risk="low", requires_confirmation=False, status="candidate")
    ctx = {"actions": [action], "procedures": [], "sections": []}
    result = synthesize_answer("查询状态", ctx)
    assert result.answer_status == ANSWER_ANSWERED
    assert result.refusal_reason is None
    assert result.conflicts == []
    assert result.requires_confirmation is False


# ---------- 冲突检测 ----------


def _make_security_action(**overrides):
    action = {
        "id": "安全.防护.启用防火墙",
        "action_id": "安全.防护.启用防火墙",
        "title": "启用防火墙",
        "risk": "high",
        "status": "needs_review",
        "confidence": 0.9,
        "requires_confirmation": True,
        "procedure": [
            {"order": 1, "action": "enable", "target": "防火墙", "source_instruction": "启用防火墙"},
        ],
        "source": {"manual_id": "m", "procedure_id": "procedure.安全.启用", "file": "security/fw.html"},
        "score": 5.0,
    }
    action.update(overrides)
    return action


def test_opposite_operations_conflict():
    act1 = _make_security_action()
    act2 = _make_security_action(
        id="安全.防护.禁用防火墙", action_id="安全.防护.禁用防火墙", title="禁用防火墙",
        source={"manual_id": "m", "procedure_id": "procedure.安全.禁用", "file": "security/fw2.html"},
        procedure=[{"order": 1, "action": "disable", "target": "防火墙", "source_instruction": "禁用防火墙"}],
    )
    ctx = {"actions": [act1, act2], "procedures": [], "sections": []}
    result = synthesize_answer("处理防火墙", ctx)
    assert result.answer_status == ANSWER_CONFLICT
    assert result.requires_confirmation is True
    assert any(c["type"] == "opposite_operations" for c in result.conflicts)
    assert any("防火墙" in c["description"] for c in result.conflicts)
    # 请求澄清
    assert "澄清" in result.answer


def test_step_target_conflict():
    act1 = _make_action(procedure=[
        {"order": 1, "action": "click", "target": "设置A", "source_instruction": "点击设置A"},
    ])
    act2 = _make_action(
        id="网络.代理.其他", action_id="网络.代理.其他", title="配置代理其他",
        procedure=[
            {"order": 1, "action": "click", "target": "设置B", "source_instruction": "点击设置B"},
        ],
    )
    ctx = {"actions": [act1, act2], "procedures": [], "sections": []}
    result = synthesize_answer("如何配置代理", ctx)
    assert result.answer_status == ANSWER_CONFLICT
    assert any(c["type"] == "step_conflict" for c in result.conflicts)


def test_risk_contradiction_conflict():
    act1 = _make_security_action(title="防火墙设置", risk="low")
    act2 = _make_security_action(
        id="安全.防护.防火墙2", action_id="安全.防护.防火墙2", title="防火墙设置", risk="high",
        source={"manual_id": "m", "procedure_id": "procedure.安全.2", "file": "security/fw2.html"},
    )
    ctx = {"actions": [act1, act2], "procedures": [], "sections": []}
    result = synthesize_answer("防火墙设置", ctx)
    assert result.answer_status == ANSWER_CONFLICT
    assert any(c["type"] == "risk_contradiction" for c in result.conflicts)


def test_no_conflicts_single_action():
    act = _make_action()
    ctx = {"actions": [act], "procedures": [], "sections": []}
    result = synthesize_answer("如何配置代理", ctx)
    assert result.answer_status == ANSWER_ANSWERED
    assert result.conflicts == []


def test_conflicts_carry_sources():
    act1 = _make_security_action()
    act2 = _make_security_action(
        id="安全.防护.禁用防火墙", action_id="安全.防护.禁用防火墙", title="禁用防火墙",
        source={"manual_id": "m", "procedure_id": "procedure.安全.禁用", "file": "security/fw2.html"},
        procedure=[{"order": 1, "action": "disable", "target": "防火墙", "source_instruction": "禁用防火墙"}],
    )
    ctx = {"actions": [act1, act2], "procedures": [], "sections": []}
    result = synthesize_answer("处理防火墙", ctx)
    conflict = next(c for c in result.conflicts if c["type"] == "opposite_operations")
    # 两个来源都带上
    files = {s.get("file") for s in conflict["sources"]}
    assert "security/fw.html" in files
    assert "security/fw2.html" in files


# ---------- 序列化与展示 ----------


def test_to_jsonl_includes_p23_fields():
    ctx = {"actions": [_make_action()], "procedures": [], "sections": []}
    result = synthesize_answer("如何配置代理", ctx)
    d = result.to_jsonl()
    assert d["answer_status"] == ANSWER_ANSWERED
    assert "refusal_reason" in d
    assert "conflicts" in d


def test_answer_to_text_renders_conflict():
    act1 = _make_security_action()
    act2 = _make_security_action(
        id="安全.防护.禁用防火墙", action_id="安全.防护.禁用防火墙", title="禁用防火墙",
        procedure=[{"order": 1, "action": "disable", "target": "防火墙", "source_instruction": "禁用防火墙"}],
    )
    ctx = {"actions": [act1, act2], "procedures": [], "sections": []}
    result = synthesize_answer("处理防火墙", ctx)
    text = answer_to_text(result)
    assert "[状态：文档冲突" in text
    assert "冲突1" in text


def test_answer_to_text_renders_refusal():
    result = synthesize_answer("不存在的东西", {"actions": [], "procedures": [], "sections": []})
    text = answer_to_text(result)
    assert "[状态：证据不足" in text


def test_reconstruct_synthesized_answer_from_dict():
    """CLI `SynthesizedAnswer(**result)` 需要新字段可构造。"""
    ctx = {"actions": [_make_action()], "procedures": [], "sections": []}
    d = synthesize_answer("如何配置代理", ctx).to_jsonl()
    obj = SynthesizedAnswer(**d)
    assert obj.answer_status == ANSWER_ANSWERED
