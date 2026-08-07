"""P2-2 逐条引用来源测试。

覆盖:
  - 每个 step 携带 source 引用（file/section_id/procedure_id/id）
  - warning_sources 与 warnings 一一对应，且携带来源
  - assertions（成功准则）每条带 source
  - 扁平形状（source_file/section_id/procedure_id）也能解析来源
  - 缺 source 时不报错，source 为空 dict / 空字段
  - answer_to_text 渲染逐条来源
"""

from edr_rag.p2.synthesize import (
    answer_to_text,
    synthesize_answer,
)


def _make_action(**overrides):
    action = {
        "id": "网络.代理.配置代理服务器",
        "action_id": "网络.代理.配置代理服务器",
        "title": "配置代理服务器",
        "risk": "high",
        "status": "needs_review",
        "confidence": 0.85,
        "requires_confirmation": True,
        "preconditions": ["具备网络管理员权限"],
        "procedure": [
            {"order": 1, "action": "open_window", "target": "网络设置", "source_instruction": "打开网络设置"},
            {"order": 2, "action": "input_text", "target": "代理地址", "source_instruction": "输入代理地址"},
        ],
        "recovery": ["如需撤销，可关闭代理开关"],
        "success_criteria": ["代理生效", "网络可访问"],
        "source": {
            "manual_id": "test-manual",
            "procedure_id": "procedure.网络.代理.配置代理服务器",
            "file": "network/proxy.html",
        },
        "score": 3.0,
    }
    action.update(overrides)
    return action


def test_each_step_carries_source():
    ctx = {"actions": [_make_action()], "procedures": [], "sections": []}
    result = synthesize_answer("如何配置代理", ctx)
    assert len(result.steps) == 2
    for s in result.steps:
        assert "source" in s
        assert s["source"]["file"] == "network/proxy.html"
        assert s["source"]["procedure_id"] == "procedure.网络.代理.配置代理服务器"
        assert s["source"]["section_id"] == ""
        assert s["source"]["id"] == "网络.代理.配置代理服务器"


def test_warning_sources_parallel_and_carried():
    ctx = {"actions": [_make_action()], "procedures": [], "sections": []}
    result = synthesize_answer("如何配置代理", ctx)
    assert len(result.warnings) >= 1
    assert len(result.warning_sources) == len(result.warnings)
    for src in result.warning_sources:
        assert src["file"] == "network/proxy.html"
        assert src["id"] == "网络.代理.配置代理服务器"


def test_assertions_carry_source():
    act = _make_action(success_criteria=["代理生效", "网络可访问"])
    ctx = {"actions": [act], "procedures": [], "sections": []}
    result = synthesize_answer("如何配置代理", ctx)
    texts = [a["text"] for a in result.assertions]
    assert "代理生效" in texts
    assert "网络可访问" in texts
    for a in result.assertions:
        assert a["source"]["file"] == "network/proxy.html"
        assert a["source"]["procedure_id"] == "procedure.网络.代理.配置代理服务器"


def test_flat_shape_source_resolution():
    """扁平形状（无嵌套 source）也能解析来源引用。"""
    act = _make_action(
        source=None,
        source_file="network/proxy.html",
        section_id="manual.section.network.proxy",
        procedure_id="procedure.网络.代理.配置代理服务器",
    )
    ctx = {"actions": [act], "procedures": [], "sections": []}
    result = synthesize_answer("配置代理", ctx)
    assert len(result.steps) == 2
    src = result.steps[0]["source"]
    assert src["file"] == "network/proxy.html"
    assert src["section_id"] == "manual.section.network.proxy"
    assert src["procedure_id"] == "procedure.网络.代理.配置代理服务器"


def test_missing_source_no_crash():
    """没有 source 信息时不报错，source 为空字段。"""
    act = _make_action(source=None)
    # 显式移除所有来源相关字段
    act.pop("source_file", None)
    act.pop("file", None)
    act.pop("section_id", None)
    act.pop("procedure_id", None)
    ctx = {"actions": [act], "procedures": [], "sections": []}
    result = synthesize_answer("配置代理", ctx)
    assert len(result.steps) == 2
    for s in result.steps:
        assert s["source"] == {"file": "", "section_id": "", "procedure_id": "", "id": act["id"]}
    # warnings / assertions 也都不报错
    assert len(result.warning_sources) == len(result.warnings)
    assert result.assertions


def test_dedup_keeps_warning_source_alignment():
    """跨动作去重 warning 后，warning_sources 仍与 warnings 一一对应。"""
    act1 = _make_action(recovery=["同一条恢复说明"])
    act2 = _make_action(id="其他.动作", action_id="其他.动作",
                        recovery=["同一条恢复说明"], risk="low",
                        requires_confirmation=False, status="candidate",
                        source={"manual_id": "m", "procedure_id": "procedure.其他",
                                "file": "other.html"})
    ctx = {"actions": [act1, act2], "procedures": [], "sections": []}
    result = synthesize_answer("配置代理", ctx)
    assert len(result.warnings) == len(result.warning_sources)
    # "同一条恢复说明" 只出现一次（去重），其来源应来自第一个动作
    assert result.warnings.count("同一条恢复说明") == 1
    idx = result.warnings.index("同一条恢复说明")
    assert result.warning_sources[idx]["file"] == "network/proxy.html"


def test_answer_to_text_renders_sources():
    ctx = {"actions": [_make_action()], "procedures": [], "sections": []}
    result = synthesize_answer("如何配置代理", ctx)
    text = answer_to_text(result)
    assert "操作步骤（含来源）" in text
    assert "注意事项（含来源）" in text
    assert "断言/成功准则（含来源）" in text
    assert "network/proxy.html" in text


def test_jsonl_includes_new_fields():
    ctx = {"actions": [_make_action()], "procedures": [], "sections": []}
    result = synthesize_answer("如何配置代理", ctx)
    d = result.to_jsonl()
    assert "warning_sources" in d
    assert "assertions" in d
    assert {"file", "section_id", "procedure_id", "id"} <= set(d["steps"][0]["source"].keys())
