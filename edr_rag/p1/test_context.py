"""P1-4 测试：父子节点与相邻步骤上下文补全。

验证目标：
  1. 命中 step 时，自动补全所属 procedure（含 preconditions / warnings /
     recovery / expected_result）与相邻 steps、祖先 section。
  2. 命中 procedure 时，补全自身各字段 + 相邻 procedures + 祖先 section。
  3. preconditions / warnings 等结构化字段从 procedure metadata 正确提取
     （无需回读 JSONL）。
  4. 未知 node_id 返回非抛错的结构化结果（各键为 None / 空）。
  5. render_context_text 按文档顺序、带标签渲染，且无虚构字段。
"""

from __future__ import annotations

from edr_rag.p1.context import (
    EXPECTED,
    PRE,
    RECOVERY,
    WARN,
    assemble_context,
    render_context_text,
)
from edr_rag.p1.nodes import (
    build_structured_nodes,
)

SECTION = {
    "id": "sec-01",
    "manual_id": "M-001",
    "product": "华三交换机",
    "title": "配置远程登录会话",
    "section_path": ["网络配置", "远程管理"],
    "content": "本文介绍如何配置远程登录会话。",
    "source_ref": {"relative_path": "docs/remote.md"},
}

PROCEDURE = {
    "id": "procedure.远程管理.配置远程登录会话",
    "manual_id": "M-001",
    "product": "华三交换机",
    "operation": "配置远程登录会话",
    "title": "配置远程登录会话",
    "section_path": ["网络配置", "远程管理"],
    "version": "V2.1",
    "status": "candidate",
    "preconditions": ["以管理员身份登录设备", "设备已上电"],
    "expected_result": "能够通过远程会话成功管理设备",
    "warnings": ["修改加密配置前请备份"],
    "recovery": ["连接失败请检查端口 22"],
    "steps": [
        {"order": 1, "instruction": "输入代理服务器地址 10.1.2.3",
         "action_hint": "input_text", "target": "代理地址"},
        {"order": 2, "instruction": "启用加密传输协议 TLS",
         "action_hint": "enable", "target": "TLS"},
        {"order": 3, "instruction": "保存配置", "action_hint": "save", "target": ""},
    ],
    "source": {"file": "docs/remote.md", "section_id": "sec-01"},
}


def _nodes():
    return build_structured_nodes([SECTION], [PROCEDURE])


def _step(nodes, order):
    return next(n for n in nodes if n.level == "step" and n.metadata["step"] == order)


def test_step_hit_completes_procedure_and_adjacent_steps():
    """命中中间 step 时，补全 procedure（含 preconditions/warnings）+ 相邻 steps + section。"""
    nodes = _nodes()
    middle = _step(nodes, 2)
    sec_node = next(n for n in nodes if n.level == "section")
    ctx = assemble_context(middle.id, nodes, adjacency=1)

    assert ctx["node_id"] == middle.id
    assert ctx["level"] == "step"

    # 父 procedure 结构化字段
    proc = ctx["procedure"]
    assert proc is not None
    assert proc["procedure_id"] == PROCEDURE["id"]
    assert proc[PRE] == ["以管理员身份登录设备", "设备已上电"]
    assert proc[WARN] == ["修改加密配置前请备份"]
    assert proc[RECOVERY] == ["连接失败请检查端口 22"]
    assert proc[EXPECTED] == "能够通过远程会话成功管理设备"

    # 相邻 steps：前后各 1
    adj = ctx["adjacent_steps"]
    relations = {a["relation"]: a["order"] for a in adj}
    assert relations.get("prev") == 1
    assert relations.get("next") == 3

    # 祖先 section（注意节点 id 带 node.section. 前缀）
    sec = ctx["section"]
    assert sec is not None and sec["id"] == sec_node.id
    assert sec["module"] == "网络配置 / 远程管理"


def test_procedure_hit_returns_self_as_procedure():
    """命中 procedure 自身时，procedure 就是它自己，相邻为相邻 procedure。"""
    nodes = _nodes()
    proc_node = next(n for n in nodes if n.level == "procedure")
    ctx = assemble_context(proc_node.id, nodes, adjacency=1)

    assert ctx["level"] == "procedure"
    assert ctx["procedure"]["procedure_id"] == PROCEDURE["id"]
    assert ctx["procedure"][PRE] == ["以管理员身份登录设备", "设备已上电"]
    # 只有一个 procedure，无相邻 procedure
    assert ctx["procedure"]["id"] == proc_node.id


def test_section_hit_has_no_procedure_but_section():
    """命中 section 时，procedure/相邻 steps 为 None/空（section 是根，无祖先 section）。"""
    nodes = _nodes()
    sec_node = next(n for n in nodes if n.level == "section")
    ctx = assemble_context(sec_node.id, nodes)

    assert ctx["level"] == "section"
    assert ctx["procedure"] is None
    assert ctx["adjacent_steps"] == []
    # section 无父节点，故无祖先 section；但主要通过 hit 自我识别
    assert ctx["section"] is None
    assert ctx["hit"]["id"] == sec_node.id
    assert ctx["hit"]["level"] == "section"


def test_unknown_node_returns_non_raising_shape():
    """未知 node_id 不抛错，返回结构化空结果。"""
    nodes = _nodes()
    ctx = assemble_context("node.step.does.not.exist", nodes)
    assert ctx["hit"] is None
    assert ctx["procedure"] is None
    assert ctx["adjacent_steps"] == []
    assert ctx["section"] is None
    assert ctx["window"] == []


def test_render_context_text_ordered_and_no_fabrication():
    """渲染文本按文档顺序带标签，且无虚构字段。"""
    nodes = _nodes()
    middle = _step(nodes, 2)
    ctx = assemble_context(middle.id, nodes, adjacency=1)
    text = render_context_text(ctx)

    # 标签齐全
    assert "[章节]" in text
    assert "[过程]" in text
    assert "[前置条件]" in text and "以管理员身份登录设备" in text
    assert "[注意]" in text and "修改加密配置前请备份" in text
    assert "[恢复]" in text
    assert "[预期结果]" in text

    # 无虚构：预期结果字段为空时不该出现该标签
    empty_proc = dict(PROCEDURE)
    empty_proc["expected_result"] = ""
    empty_nodes = build_structured_nodes(
        [dict(SECTION)], [empty_proc]
    )
    ectx = assemble_context(_step(empty_nodes, 1).id, empty_nodes)
    etext = render_context_text(ectx)
    assert "[预期结果]" not in etext


def test_window_present_and_in_document_order():
    """window 键携带 P1-2 窗口节点（祖先在前、自身居中、相邻随后）。"""
    nodes = _nodes()
    middle = _step(nodes, 2)
    ctx = assemble_context(middle.id, nodes, adjacency=1)
    ids = [w["id"] for w in ctx["window"]]
    sec = next(n for n in nodes if n.level == "section")
    assert ids[0] == sec.id
    assert middle.id in ids
