"""P1-2 测试：结构化 Node（章节/Procedure/Step/Action 分层）与父子/相邻关系。

验证目标：
  1. 由 ManualSection/Procedure/ProcedureStep 生成带 metadata 的节点
     （section_path、procedure_id、step order、product、module）。
  2. 保留父子关系（section -> procedure -> step/action）与相邻关系（prev/next）。
  3. 上下文补全窗口能沿祖先 + 相邻兄弟拼接上下文。
  4. 文件级 build_structured_nodes_file 写出 nodes.jsonl 且可回读。
  5. 无向量依赖时主链路不报错（纯 JSON 节点，不含 LlamaIndex）。
"""

from __future__ import annotations

import json

from edr_rag.p1.nodes import (
    StructuredNode,
    build_action_node,
    build_context_window,
    build_structured_nodes,
    build_structured_nodes_file,
    load_structured_nodes,
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
    "preconditions": ["以管理员身份登录设备"],
    "expected_result": "能够通过远程会话成功管理设备",
    "steps": [
        {"order": 1, "instruction": "输入代理服务器地址 10.1.2.3",
         "action_hint": "input_text", "target": "代理地址"},
        {"order": 2, "instruction": "启用加密传输协议 TLS",
         "action_hint": "enable", "target": "TLS"},
        {"order": 3, "instruction": "保存配置", "action_hint": "save", "target": ""},
    ],
    "source": {"file": "docs/remote.md", "section_id": "sec-01"},
}

ACTION = {
    "action_id": "远程管理.配置远程登录会话",
    "manual_id": "M-001",
    "product": "华三交换机",
    "module": "网络配置",
    "title": "配置远程登录会话",
    "status": "needs_review",
    "risk": "medium",
    "requires_confirmation": True,
    "procedure": [
        {"order": 1, "action": "input_text", "target": "代理地址"},
        {"order": 2, "action": "enable", "target": "TLS"},
    ],
    "success_criteria": ["能够通过远程会话成功管理设备"],
    "source": {"manual_id": "M-001", "procedure_id": PROCEDURE["id"], "file": "docs/remote.md"},
}


def test_section_node_metadata():
    """章节节点携带 section_path/product/module metadata。"""
    node = build_structured_nodes([SECTION], [])
    assert len(node) == 1
    sec = node[0]
    assert sec.level == "section"
    assert sec.metadata["product"] == "华三交换机"
    assert sec.metadata["section_path"] == ["网络配置", "远程管理"]
    assert sec.metadata["module"] == "网络配置 / 远程管理"
    assert sec.metadata["type"] == "section"


def test_procedure_step_hierarchy():
    """procedure/step 层级与 metadata（procedure_id、step order）正确。"""
    nodes = build_structured_nodes([SECTION], [PROCEDURE])
    sec = next(n for n in nodes if n.level == "section")
    proc = next(n for n in nodes if n.level == "procedure")
    steps = [n for n in nodes if n.level == "step"]
    assert len(steps) == 3

    # 父子关系
    assert proc.parent_id == sec.id
    assert sec.children == [proc.id]
    assert all(s.parent_id == proc.id for s in steps)
    assert proc.children == [s.id for s in steps]

    # metadata
    assert proc.metadata["procedure_id"] == PROCEDURE["id"]
    assert proc.metadata["section_path"] == ["网络配置", "远程管理"]
    assert proc.metadata["module"] == "网络配置 / 远程管理"
    assert [s.metadata["step"] for s in steps] == [1, 2, 3]

    # 相邻关系（step 兄弟按文档顺序串接）
    assert steps[0].next_id == steps[1].id
    assert steps[1].prev_id == steps[0].id
    assert steps[1].next_id == steps[2].id
    assert steps[0].prev_id == ""
    assert steps[2].next_id == ""


def test_action_node_hangs_under_procedure():
    """action 节点作为 procedure 的 child 挂到树上。"""
    nodes = build_structured_nodes([SECTION], [PROCEDURE], [ACTION])
    action = next(n for n in nodes if n.level == "action")
    proc = next(n for n in nodes if n.level == "procedure")
    assert action.parent_id == proc.id
    assert action.id in proc.children
    assert action.metadata["action_id"] == ACTION["action_id"]
    assert action.metadata["procedure_id"] == PROCEDURE["id"]
    assert action.metadata["section_path"] == ["网络配置", "远程管理"]
    assert action.metadata["risk"] == "medium"


def test_oracle_procedure_when_section_content_empty():
    """section 内容为空（跳过章节节点）时，procedure 仍作为孤儿节点出现。"""
    empty_section = dict(SECTION)
    empty_section["content"] = ""
    nodes = build_structured_nodes([empty_section], [PROCEDURE])
    proc = next(n for n in nodes if n.level == "procedure")
    steps = [n for n in nodes if n.level == "step"]
    assert proc.parent_id == ""
    assert len(steps) == 3
    assert all(s.parent_id == proc.id for s in steps)


def test_context_window_includes_ancestor_and_neighbors():
    """上下文补全：含父章节 + 自身 + 相邻 step。"""
    nodes = build_structured_nodes([SECTION], [PROCEDURE])
    steps = sorted([n for n in nodes if n.level == "step"], key=lambda n: n.metadata["step"])
    middle = steps[1]

    window = build_context_window(middle.id, nodes, ancestors=2, adjacency=1)
    ids = [n.id for n in window]
    # 父章节在最前，自身在中间，相邻兄弟出现
    sec = next(n for n in nodes if n.level == "section")
    assert ids[0] == sec.id
    assert middle.id in ids
    assert steps[0].id in ids and steps[2].id in ids


def test_build_and_reload_file(tmp_path):
    """build_structured_nodes_file 写出 nodes.jsonl 且可回读。"""
    workdir = tmp_path / "w"
    workdir.mkdir()
    with (workdir / "sections.jsonl").open("w", encoding="utf-8") as f:
        f.write(json.dumps(SECTION, ensure_ascii=False) + "\n")
    with (workdir / "procedures.jsonl").open("w", encoding="utf-8") as f:
        f.write(json.dumps(PROCEDURE, ensure_ascii=False) + "\n")
    with (workdir / "actions.jsonl").open("w", encoding="utf-8") as f:
        f.write(json.dumps(ACTION, ensure_ascii=False) + "\n")

    nodes, nodes_path = build_structured_nodes_file(workdir)
    assert nodes_path.exists()
    assert (workdir / "nodes_meta.json").exists()

    loaded = load_structured_nodes(nodes_path)
    assert len(loaded) == len(nodes)
    levels = {n.level for n in loaded}
    assert {"section", "procedure", "step", "action"} <= levels

    # 回读后父子/相邻关系一致
    by_id = {n.id: n for n in loaded}
    proc = next(n for n in loaded if n.level == "procedure")
    assert by_id[proc.id].children
    assert all(child in by_id for child in proc.children)


def test_no_sections_returns_empty(tmp_path):
    """无 sections 时返回空列表且不报错（不写污染文件）。"""
    workdir = tmp_path / "w2"
    workdir.mkdir()
    nodes, _ = build_structured_nodes_file(workdir)
    assert nodes == []


def test_to_json_roundtrip():
    """StructuredNode JSON 序列化往返无损。"""
    n = StructuredNode(
        id="node.step.x.s1", level="step", text="打开设置",
        metadata={"step": 1, "section_path": ["网络"]},
        parent_id="node.procedure.x", next_id="node.step.x.s2",
    )
    d = n.to_json()
    n2 = StructuredNode.from_json(d)
    assert n2 == n


def test_level_counts_in_meta(tmp_path):
    """nodes_meta.json 统计各层级节点数。"""
    workdir = tmp_path / "w3"
    workdir.mkdir()
    with (workdir / "sections.jsonl").open("w", encoding="utf-8") as f:
        f.write(json.dumps(SECTION, ensure_ascii=False) + "\n")
    with (workdir / "procedures.jsonl").open("w", encoding="utf-8") as f:
        f.write(json.dumps(PROCEDURE, ensure_ascii=False) + "\n")
    build_structured_nodes_file(workdir)
    meta = json.loads((workdir / "nodes_meta.json").read_text(encoding="utf-8"))
    # 仅 section + procedure + 3 steps（未写 actions.jsonl，故无 action 节点）
    assert meta["node_count"] == 1 + 1 + 3
    assert meta["levels"]["step"] == 3
    assert meta["levels"]["action"] == 0
