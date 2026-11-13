"""P1-2 结构化 Node（章节/Procedure/Step/Action 分层）。

在 P0 ManualSection、P1 Procedure/ProcedureStep、P2 Action 之上，生成带
metadata（section_path、procedure_id、step order、product、module）的
**结构化 node**，并显式保留**父子关系**（parent/child）与**相邻关系**
（previous/next adjacency），供向量索引与上下文补全（context completion）使用。

分层模型（与项目 P1/P2 分层一致）:
  - level="section"     -> 一个 ManualSection（章节）
  - level="procedure"   -> 一个 Procedure（过程步骤集）
  - level="step"        -> 一个 ProcedureStep（单步）
  - level="action"      -> 一个 Action（P2 动作目录条目，挂在 procedure 之下）

关系字段:
  - parent_id: 父 node id（section -> procedure -> step/action）
  - prev_id / next_id: 同父下的相邻兄弟（保持文档顺序）,用于上下文补全时
    取相邻片段拼接。

本模块**不依赖** LlamaIndex/向量库 —— node 是纯 JSON 可序列化的中间表示，
任何下游（向量索引、重排、上下文组装）都可消费。缺向量依赖时主链路照常工作。

"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------- 节点模型 ----------

@dataclass
class StructuredNode:
    """一个结构化节点，携带层级、metadata 与父子/相邻关系。"""

    id: str
    level: str  # section | procedure | step | action
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    parent_id: str = ""
    prev_id: str = ""
    next_id: str = ""
    children: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return asdict(self)

    #: write_jsonl / JSONL 产物约定：含 to_jsonl() 的对象由此序列化。
    to_jsonl = to_json

    @classmethod
    def from_json(cls, d: dict) -> "StructuredNode":
        return cls(
            id=d.get("id", ""),
            level=d.get("level", ""),
            text=d.get("text", ""),
            metadata=d.get("metadata", {}) or {},
            parent_id=d.get("parent_id", ""),
            prev_id=d.get("prev_id", ""),
            next_id=d.get("next_id", ""),
            children=d.get("children", []) or [],
        )


# ---------- 基础 metadata 助手 ----------

def _section_meta(section: dict) -> dict[str, Any]:
    """从 ManualSection dict 提取公共 metadata。"""
    section_path = section.get("section_path") or []
    if isinstance(section_path, str):
        section_path = [section_path]
    src = section.get("source_ref") or {}
    if isinstance(src, dict):
        rel_file = src.get("relative_path", "")
    else:
        rel_file = section.get("source_file", "")
    return {
        "type": "section",
        "manual_id": section.get("manual_id", ""),
        "product": section.get("product", ""),
        "title": section.get("title", ""),
        "module": " / ".join(section_path) if section_path else section.get("product", ""),
        "section_path": list(section_path),
        "source_file": rel_file,
        "version": (
            section.get("version")
            or section.get("metadata", {}).get("version", "")
            or ""
        ),
    }


# ---------- 节点构建 ----------

def build_section_node(section: dict) -> Optional[StructuredNode]:
    """把一个 ManualSection dict 变成 level='section' 的节点。

    若章节无内容则返回 None（无可索引文本）。
    """
    text = (section.get("content") or "").strip()
    if not text:
        return None
    section_id = section.get("id", "")
    if not section_id:
        return None
    return StructuredNode(
        id=f"node.section.{section_id}",
        level="section",
        text=text,
        metadata=_section_meta(section),
    )


def _proc_meta(proc: dict, section_path: list[str]) -> dict[str, Any]:
    """从 Procedure dict 提取 metadata（含 procedure_id / section_path / module）。

    P1-4: 同时携带结构化 preconditions / warnings / recovery / expected_result，
    使下游上下文补全（context completion）无需再回读 procedures.jsonl，
    即可按字段给出前置条件、注意、恢复、预期结果等上下文。
    """
    src = proc.get("source") or {}
    if isinstance(src, dict):
        rel_file = src.get("file", "")
    else:
        rel_file = ""
    return {
        "type": "procedure",
        "manual_id": proc.get("manual_id", ""),
        "product": proc.get("product", ""),
        "title": proc.get("title", ""),
        "operation": proc.get("operation", ""),
        "procedure_id": proc.get("id", ""),
        "status": proc.get("status", ""),
        "module": " / ".join(section_path) if section_path else proc.get("product", ""),
        "section_path": list(section_path),
        "source_file": rel_file,
        "version": proc.get("version", ""),
        # P1-4: 结构化上下文字段（列表若无则保留空列表）
        "preconditions": list(proc.get("preconditions", []) or []),
        "warnings": list(proc.get("warnings", []) or []),
        "recovery": list(proc.get("recovery", []) or []),
        "expected_result": proc.get("expected_result", ""),
    }


def build_procedure_node(
    proc: dict,
    *,
    section_node: Optional[StructuredNode] = None,
    section_path: Optional[list[str]] = None,
) -> tuple[Optional[StructuredNode], list[StructuredNode]]:
    """把一个 Procedure dict 变成 {level='procedure' node, [level='step' nodes]}。

    Args:
        proc: Procedure dict（procedures.jsonl 中的一行）。
        section_node: 父 section 节点（用于 parent_id 关联，可省略）。
        section_path: 该 procedure 所属章节的 section_path；缺省时回退到
            section_node 的 metadata，再回退到 proc.section_path。

    Returns:
        (procedure_node, step_nodes)。若 procedure 无有效文本则 (None, [])。
    """
    if section_path is None:
        if section_node:
            section_path = list(section_node.metadata.get("section_path", []))
        else:
            sp = proc.get("section_path") or []
            section_path = list(sp) if isinstance(sp, list) else []
    section_path = list(section_path)

    proc_id = proc.get("id", "")
    text = _procedure_to_flat_text(proc)
    if not text:
        return None, []

    node_id = f"node.procedure.{proc_id}"
    children_ids: list[str] = []
    step_nodes: list[StructuredNode] = []

    raw_steps = proc.get("steps", []) or []
    for i, s in enumerate(raw_steps):
        if not isinstance(s, dict):
            continue
        order = s.get("order", i + 1)
        inst = (s.get("instruction") or s.get("source_instruction") or "").strip()
        if not inst:
            continue
        step_id = f"node.step.{proc_id}.s{order}"
        children_ids.append(step_id)
        meta = dict(_proc_meta(proc, section_path))
        meta["type"] = "step"
        meta["step"] = order  # step order 显式携带
        meta["action_hint"] = s.get("action_hint", "")
        meta["target"] = s.get("target", "")
        step_nodes.append(
            StructuredNode(
                id=step_id,
                level="step",
                text=inst,
                metadata=meta,
                parent_id=node_id,
            )
        )

    # 相邻关系：同父下按文档顺序串 prev/next
    _link_adjacency(step_nodes)

    proc_node = StructuredNode(
        id=node_id,
        level="procedure",
        text=text,
        metadata=_proc_meta(proc, section_path),
        parent_id=section_node.id if section_node else "",
        children=children_ids,
    )
    return proc_node, step_nodes


def build_action_node(
    action: dict,
    *,
    procedure_node: StructuredNode,
) -> Optional[StructuredNode]:
    """把一个 P2 ActionCatalogEntry dict 变成挂在 procedure 下的 action 节点。

    action 是 procedure 的另一种精化视图（操作意图、风险、成功准则），
    作为 procedure 的 child 挂到同一树上。
    """
    text = _action_to_flat_text(action)
    if not text:
        return None
    action_id = action.get("action_id", "")
    if not action_id:
        return None
    proc_path = procedure_node.metadata.get("section_path", [])
    proc_path = list(proc_path) if isinstance(proc_path, list) else []
    meta = {
        "type": "action",
        "manual_id": action.get("manual_id", ""),
        "product": action.get("product", ""),
        "module": action.get("module", ""),
        "title": action.get("title", ""),
        "action_id": action_id,
        "procedure_id": procedure_node.metadata.get("procedure_id", ""),
        "section_path": proc_path,
        "status": action.get("status", ""),
        "risk": action.get("risk", ""),
        "requires_confirmation": action.get("requires_confirmation", False),
    }
    return StructuredNode(
        id=f"node.action.{action_id}",
        level="action",
        text=text,
        metadata=meta,
        parent_id=procedure_node.id,
    )


# ---------- 组装整棵树 ----------

def build_structured_nodes(
    sections: list[dict],
    procedures: list[dict],
    actions: Optional[list[dict]] = None,
) -> list[StructuredNode]:
    """把章节/过程/步骤/动作组装成一棵带父子与相邻关系的节点树。

    结构:
        section
          └ procedure(s)
              ├ step(s)          (来自 procedure.steps)
              └ action (optional) (来自 P2 actions 中 procedure_id 匹配者)

    Args:
        sections: ManualSection dict 列表（sections.jsonl）。
        procedures: Procedure dict 列表（procedures.jsonl）。
        actions: ActionCatalogEntry dict 列表（actions.jsonl），可省略。

    Returns:
        全量节点列表（扁平，通过 parent_id/children/prev/next 表达层级与相邻）。
    """
    nodes: list[StructuredNode] = []

    # section_path 索引，便于 procedure 关联章节 context
    section_by_id: dict[str, dict] = {}
    for s in sections:
        sid = s.get("id")
        if sid:
            section_by_id[sid] = s

    # procs 按 section_id 分组（source_ref.section_id）
    procs_by_section: dict[str, list[dict]] = {}
    for p in procedures:
        src = p.get("source") or {}
        sec_id = src.get("section_id", "") if isinstance(src, dict) else ""
        procs_by_section.setdefault(sec_id, []).append(p)

    # 为每个 section 构建节点；无内容的 section 跳过（但也允许孤儿 procedure）
    section_nodes: dict[str, StructuredNode] = {}
    for s in sections:
        sn = build_section_node(s)
        if sn is not None:
            section_nodes[s["id"]] = sn
            nodes.append(sn)

    # actions 索引：action 通常不直接带 procedure_id 之外的信息，靠 procedure
    # 关联；但我们也可用 source.procedure_id 反查。
    actions_by_proc: dict[str, list[dict]] = {}
    for a in actions or []:
        src = a.get("source") or {}
        pid = (
            a.get("procedure_id", "")
            if a.get("procedure_id")
            else (src.get("procedure_id", "") if isinstance(src, dict) else "")
        )
        actions_by_proc.setdefault(pid, []).append(a)

    processed: set[str] = set()

    # 处理每个 procedure，挂到其 section 下
    for sec_id, procs in procs_by_section.items():
        parent = section_nodes.get(sec_id)
        proc_stack: list[StructuredNode] = []
        for p in procs:
            pid = p.get("id", "")
            if pid in processed:
                continue
            processed.add(pid)
            pnode, snodes = build_procedure_node(
                p, section_node=parent,
                section_path=(parent.metadata.get("section_path") if parent else None),
            )
            if pnode is None:
                continue
            proc_stack.append(pnode)
            nodes.append(pnode)
            for sn in snodes:
                nodes.append(sn)
            # 挂 action children
            for a in actions_by_proc.get(pid, []):
                anode = build_action_node(a, procedure_node=pnode)
                if anode is not None:
                    pnode.children.append(anode.id)
                    nodes.append(anode)
        # 同 section 下的相邻 procedure 关系
        _link_adjacency(proc_stack)
        if parent is not None:
            parent.children = [n.id for n in proc_stack]

    # 处理孤儿 procedure（未关联到任何 section 节点，如 section 无内容被跳过，
    # 或 source.section_id 缺失/找不到对应 section）
    for p in procedures:
        pid = p.get("id", "")
        if pid in processed:
            continue
        processed.add(pid)
        pnode, snodes = build_procedure_node(p)
        if pnode is None:
            continue
        nodes.append(pnode)
        for sn in snodes:
            nodes.append(sn)
        for a in actions_by_proc.get(pid, []):
            anode = build_action_node(a, procedure_node=pnode)
            if anode is not None:
                pnode.children.append(anode.id)
                nodes.append(anode)

    return nodes


def _link_adjacency(siblings: list[StructuredNode]) -> None:
    """为同一父下的兄弟节点串 prev/next（保持列表顺序）。"""
    for idx, n in enumerate(siblings):
        n.prev_id = siblings[idx - 1].id if idx > 0 else ""
        n.next_id = siblings[idx + 1].id if idx + 1 < len(siblings) else ""


# ---------- 上下文补全 ----------

def build_context_window(
    node_id: str,
    nodes: list[StructuredNode],
    *,
    ancestors: int = 2,
    adjacency: int = 1,
) -> list[StructuredNode]:
    """为给定 node 组装上下文补全窗口（供向量检索后拼接上下文使用)。

    返回的节点按文档顺序排列，包含:
      - 至多 `ancestors` 个祖先（父、祖父 ...），优先最近的；
      - 目标节点自身；
      - 至多 `adjacency` 个前后相邻兄弟。

    顺序保证: 祖先在前、自身居中、相邻兄弟随后（按原始顺序插入）。
    """
    by_id: dict[str, StructuredNode] = {n.id: n for n in nodes}
    target = by_id.get(node_id)
    if target is None:
        return []

    # 祖先链（自身 -> 父 -> 祖父 ...），取最近 ancestors 个
    chain: list[StructuredNode] = [target]
    cur: Optional[StructuredNode] = target
    for _ in range(ancestors):
        if not cur.parent_id:
            break
        parent = by_id.get(cur.parent_id)
        if parent is None or parent in chain:
            break
        chain.append(parent)
        cur = parent
    chain.reverse()  # 祖先在前

    # 相邻兄弟（在有目标节点的兄弟列表中找 prev/next）
    siblings: list[StructuredNode] = []
    for n in nodes:
        if n.parent_id == target.parent_id and n is not target:
            siblings.append(n)
    siblings.sort(key=lambda n: _sibling_index(n))
    if target.parent_id == "":
        siblings = []

    neighbors: list[StructuredNode] = []
    if target.prev_id:
        for n in siblings:
            if n.next_id and n.next_id == target.id:
                neighbors.append(n)
                break
        else:
            for n in siblings:
                if n.id == target.prev_id:
                    neighbors.append(n)
                    break
    if target.next_id:
        for n in siblings:
            if n.id == target.next_id:
                neighbors.append(n)
                break

    # 祖先 + 自身 + 邻居，去重保序
    result: list[StructuredNode] = []
    seen: set[str] = set()
    for n in chain + [target] + neighbors:
        if n.id in seen:
            continue
        seen.add(n.id)
        result.append(n)
    return result


def _sibling_index(n: StructuredNode) -> int:
    """稳定排序键：靠 id 中的 step order / 顺序。"""
    meta = n.metadata
    if "step" in meta and isinstance(meta.get("step"), int):
        return meta["step"]
    return 0


# ---------- 文本渲染 ----------

def _procedure_to_flat_text(proc: dict) -> str:
    """把 Procedure 渲染成扁平可索引文本（与 P1 search 语义对齐）。"""
    lines: list[str] = []
    if proc.get("title"):
        lines.append(proc["title"])
    if proc.get("operation"):
        lines.append("操作: " + proc["operation"])
    if proc.get("version"):
        lines.append("版本: " + proc["version"])
    if proc.get("preconditions"):
        lines.append("前置条件: " + "；".join(proc["preconditions"]))
    for s in proc.get("steps", []) or []:
        if isinstance(s, dict):
            lines.append(f"{s.get('order', 0)}. {s.get('instruction', '')}")
    if proc.get("expected_result"):
        lines.append("预期结果: " + proc["expected_result"])
    if proc.get("warnings"):
        lines.append("注意: " + "；".join(proc["warnings"]))
    if proc.get("recovery"):
        lines.append("恢复: " + "；".join(proc["recovery"]))
    return "\n".join(ln for ln in lines if ln)


def _action_to_flat_text(action: dict) -> str:
    """把 Action 渲染成扁平可索引文本。"""
    lines: list[str] = []
    if action.get("title"):
        lines.append(action["title"])
    for s in action.get("procedure", []) or []:
        if isinstance(s, dict):
            lines.append(f"{s.get('order', 0)}. {s.get('action', '')} {s.get('target', '')}")
    if action.get("success_criteria"):
        lines.append("成功准则: " + "；".join(action["success_criteria"]))
    if action.get("recovery"):
        lines.append("恢复: " + "；".join(action["recovery"]))
    return "\n".join(ln for ln in lines if ln)


# ---------- 文件级编排 ----------

def build_structured_nodes_file(
    workdir: str | Path,
    *,
    write_jsonl: bool = True,
) -> tuple[list[StructuredNode], Path]:
    """从 workdir 的产物构建结构化节点并写出 nodes.jsonl。

    读取:
      - <workdir>/sections.jsonl（必需）
      - <workdir>/procedures.jsonl（可选，缺省时仅章节节点）
      - <workdir>/actions.jsonl（可选，P2 产物）

    写出:
      - <workdir>/nodes.jsonl：扁平节点记录（含 parent_id/prev_id/next_id/children）
      - <workdir>/nodes_meta.json：节点统计

    Returns:
        (nodes, nodes_path)。workdir 中缺少 sections.jsonl 时返回空 list。
    """
    workdir = Path(workdir)
    nodes_path = workdir / "nodes.jsonl"
    meta_path = workdir / "nodes_meta.json"

    from ..atomic_io import atomic_write_text, write_jsonl as _jsonl_writer

    from ..p1.search import JsonlCorpus

    # 缺陷④（构建期）：磁盘冻结 sections/procedures 仍含 DOCTYPE 泄漏
    # （读取侧防御只覆盖运行时语料加载），直接构建会把泄漏写进 nodes.jsonl
    # 的 text/metadata.recovery。构建期同口径剥离，使重建产物磁盘即干净；
    # 运行时防御层保留（对未重建的旧产物仍生效）。
    sections = [
        JsonlCorpus._strip_doctype_leak(r)
        for r in _read_jsonl(workdir / "sections.jsonl")
    ]
    if not sections:
        logger.warning("no sections found in %s; nothing to build", workdir)
        return [], nodes_path

    procedures = [
        JsonlCorpus._strip_doctype_leak(r)
        for r in _read_jsonl(workdir / "procedures.jsonl")
    ]
    actions = [
        JsonlCorpus._strip_doctype_leak(r)
        for r in _read_jsonl(workdir / "actions.jsonl")
    ]

    nodes = build_structured_nodes(sections, procedures, actions)

    if write_jsonl:
        _jsonl_writer(nodes_path, nodes)
        meta = {
            "manual_id": sections[0].get("manual_id", "") if sections else "",
            "product": sections[0].get("product", "") if sections else "",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "section_count": len(sections),
            "procedure_count": len(procedures),
            "action_count": len(actions),
            "node_count": len(nodes),
            "levels": {
                lvl: sum(1 for n in nodes if n.level == lvl)
                for lvl in ("section", "procedure", "step", "action")
            },
        }
        atomic_write_text(meta_path, json.dumps(meta, ensure_ascii=False, indent=2))

    return nodes, nodes_path


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:  # pragma: no cover - defensive
                    logger.warning("skip unparseable line in %s", path)
    return out


def load_structured_nodes(path: str | Path) -> list[StructuredNode]:
    """从 nodes.jsonl 加载结构化节点列表。"""
    path = Path(path)
    out: list[StructuredNode] = []
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(StructuredNode.from_json(json.loads(line)))
    return out
