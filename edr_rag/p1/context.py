"""P1-4 父子节点与相邻步骤上下文补全（context completion）。

命中某个 step / procedure / action 节点时，自动沿 P1-2 建立的**结构化节点关系**
（parent_id / prev_id / next_id / children）补全所属 procedure 的相邻 steps、
preconditions、warnings、recovery、expected_result 以及祖先 section，作为
检索上下文一并返回，供 LLM/上层调度读取，避免只给单个命中片段。

本模块**不依赖** LlamaIndex/向量库 —— 纯基于 P1-2 的 StructuredNode 关系图，
任何检索出口（API / CLI / MCP）都能复用。缺向量依赖（可选 RAG extras）时
主链路照常工作：上下文补全只消费 nodes.jsonl 里的结构化节点，不触发向量模型。

核心 API:
  - assemble_context(node_id, nodes, *, adjacency=1) -> dict
        组装一个**结构化**上下文（hit / procedure / adjacent_steps / section
        / preconditions / warnings / recovery / expected_result）。
  - render_context_text(ctx, *, max_len=None) -> str
        把结构化上下文渲染成带标签的分段纯文本（按文档顺序）。
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from .nodes import StructuredNode, build_context_window

logger = logging.getLogger(__name__)

#: 返回 dict 中预定义的 section 键，便于上游按名取用。
PRE = "preconditions"
WARN = "warnings"
RECOVERY = "recovery"
EXPECTED = "expected_result"


def _node_meta_list(node: StructuredNode, key: str) -> list[str]:
    """从节点 metadata 取列表字段，容忍缺省 / 非列表。"""
    val = node.metadata.get(key, []) if node.metadata else []
    if not isinstance(val, list):
        val = [val] if val else []
    return [str(x) for x in val]


def _node_meta_str(node: StructuredNode, key: str) -> str:
    val = node.metadata.get(key, "") if node.metadata else ""
    return str(val) if val else ""


def _section_dict(node: Optional[StructuredNode]) -> Optional[dict[str, Any]]:
    """把 section 节点提炼成轻量 dict（无 section 时返回 None）。"""
    if node is None:
        return None
    return {
        "id": node.id,
        "title": _node_meta_str(node, "title"),
        "section_path": list(node.metadata.get("section_path", [])) if node.metadata else [],
        "module": _node_meta_str(node, "module"),
        "text": node.text,
    }


def _procedure_dict(node: StructuredNode) -> dict[str, Any]:
    """把 procedure 节点提炼成结构化 dict（含 P1-4 上下文字段）。"""
    return {
        "id": node.id,
        "title": _node_meta_str(node, "title"),
        "operation": _node_meta_str(node, "operation"),
        "procedure_id": _node_meta_str(node, "procedure_id"),
        "section_path": list(node.metadata.get("section_path", [])) if node.metadata else [],
        "module": _node_meta_str(node, "module"),
        "source_file": _node_meta_str(node, "source_file"),
        "version": _node_meta_str(node, "version"),
        PRE: _node_meta_list(node, PRE),
        WARN: _node_meta_list(node, WARN),
        RECOVERY: _node_meta_list(node, RECOVERY),
        EXPECTED: _node_meta_str(node, EXPECTED),
        "text": node.text,
    }


def _hit_dict(node: StructuredNode) -> dict[str, Any]:
    return {
        "id": node.id,
        "level": node.level,
        "text": node.text,
    }


def assemble_context(
    node_id: str,
    nodes: list[StructuredNode],
    *,
    adjacency: int = 1,
    include_section: bool = True,
) -> dict[str, Any]:
    """为命中节点组装结构化上下文。

    基于 P1-2 的父子 / 相邻关系（build_context_window 已给出祖先 + 相邻兄弟），
    再进一步把所属 procedure 的 preconditions / warnings / recovery /
    expected_result 等字段显式提炼出来，便于上游按字段组装 prompt。

    Args:
        node_id: 命中的节点 id（step / procedure / action / section 均可）。
        nodes: 全量结构化节点列表（nodes.jsonl 回读结果）。
        adjacency: 相邻兄弟取几个（每侧），默认 1（即前后各 1）。
        include_section: 是否附带祖先 section 上下文。

    Returns:
        结构化 dict:
          - node_id / level: 命中节点
          - hit: 命中节点轻量信息
          - procedure: 命中节点所属 procedure 的结构化 dict（含 preconditions /
            warnings / recovery / expected_result / text）；若命中本身就是
            procedure，则为它自身；无 procedure 归属（如命中 section）时为 None。
          - adjacent_steps: 命中 step 的相邻兄弟 step 列表
            [{id, order, text, relation: 'prev'|'next'}]；命中为 procedure 时，
            相邻 procedure 列表（relation 同样标记）。
          - section: 祖先 section 轻量 dict（include_section 关闭或无祖先时为 None）。
          - window: 独立的原始节点窗口（P1-2 build_context_window 结果，按文档顺序）。

        某类上下文缺失时对应键为 None / 空列表，不会报错。
    """
    by_id: dict[str, StructuredNode] = {n.id: n for n in nodes}
    target = by_id.get(node_id)
    if target is None:
        logger.warning("assemble_context: unknown node_id=%s", node_id)
        return {
            "node_id": node_id,
            "level": "",
            "hit": None,
            "procedure": None,
            "adjacent_steps": [],
            "section": None,
            "window": [],
        }

    # 用 P1-2 的窗口先把祖先 + 相邻兄弟兜出来（按文档顺序）
    window = build_context_window(node_id, nodes, ancestors=2, adjacency=adjacency)

    # 定位命中节点所属的 procedure 节点：沿 parent 链向上找到 level=='procedure'
    procedure_node: Optional[StructuredNode] = None
    if target.level == "procedure":
        procedure_node = target
    else:
        cur: Optional[StructuredNode] = target
        seen: set[str] = set()
        while cur is not None and cur.parent_id and cur.id not in seen:
            seen.add(cur.id)
            parent = by_id.get(cur.parent_id)
            if parent is None:
                break
            if parent.level == "procedure":
                procedure_node = parent
                break
            cur = parent

    # 祖先 section
    section_node: Optional[StructuredNode] = None
    if include_section:
        cur2: Optional[StructuredNode] = procedure_node if procedure_node else target
        seen2: set[str] = set()
        while cur2 is not None and cur2.parent_id and cur2.id not in seen2:
            seen2.add(cur2.id)
            parent = by_id.get(cur2.parent_id)
            if parent is None:
                break
            if parent.level == "section":
                section_node = parent
                break
            cur2 = parent

    # 相邻兄弟：同一父下、与命中相邻（用 prev_id/next_id 关系链定位）
    adjacent_steps: list[dict[str, Any]] = []
    if target.level in ("step", "procedure", "action"):
        # 收集同父兄弟并按其文档顺序排序（step 按 order，其余保序）
        siblings = [
            n for n in nodes
            if n.parent_id == target.parent_id and n.id != target.id
        ]
        siblings.sort(key=lambda n: (n.metadata.get("step", 0) if isinstance(n.metadata.get("step"), int) else 0, n.id))
        if target.prev_id:
            for n in siblings:
                if n.id == target.prev_id:
                    adjacent_steps.append(_step_ctx(n, "prev"))
                    break
            else:
                for n in siblings:
                    if n.next_id == target.id:
                        adjacent_steps.append(_step_ctx(n, "prev"))
                        break
        if target.next_id:
            for n in siblings:
                if n.id == target.next_id:
                    adjacent_steps.append(_step_ctx(n, "next"))
                    break
                if getattr(n, "prev_id", "") == target.id:
                    adjacent_steps.append(_step_ctx(n, "next"))
                    break

    return {
        "node_id": node_id,
        "level": target.level,
        "hit": _hit_dict(target),
        "procedure": _procedure_dict(procedure_node) if procedure_node else None,
        "adjacent_steps": adjacent_steps,
        "section": _section_dict(section_node),
        "window": [n.to_json() for n in window],
    }


def _step_ctx(node: StructuredNode, relation: str) -> dict[str, Any]:
    """把相邻 step/procedure 节点提炼成含 relation 标记的 dict。"""
    order = node.metadata.get("step", 0) if node.metadata else 0
    return {
        "id": node.id,
        "level": node.level,
        "order": order,
        "text": node.text,
        "relation": relation,
    }


def render_context_text(
    ctx: dict[str, Any],
    *,
    max_len: Optional[int] = None,
) -> str:
    """把 assemble_context 的结构化结果渲染成带标签的分段纯文本。

    顺序（文档方向）:
        [章节] 祖先 section
        [过程] procedure.title
        [前置条件] ...
        [步骤] 相邻前一个 step + 命中 step + 相邻后一个 step
        [注意] ...
        [恢复] ...
        [预期结果] ...

    片段一律一行的形式，宁可缺也不虚构：无上下文字段就跳过该标签。
    """
    lines: list[str] = []

    if ctx.get("section"):
        sec = ctx["section"]
        lines.append(f"[章节] {sec.get('title') or sec.get('module') or sec.get('id')}")

    proc = ctx.get("procedure")
    if proc:
        title = proc.get("title") or proc.get("operation") or proc.get("id")
        lines.append(f"[过程] {title}")
        pre = proc.get(PRE) or []
        if pre:
            lines.append("[前置条件] " + "；".join(pre))
        warn = proc.get(WARN) or []
        if warn:
            lines.append("[注意] " + "；".join(warn))

    hit = ctx.get("hit")
    if hit:
        lines.append(f"[命中] ({hit.get('level')}) {hit.get('text')}")

    for adj in ctx.get("adjacent_steps") or []:
        tag = "上一步骤" if adj.get("relation") == "prev" else "下一步骤"
        lines.append(f"[{tag}] ({adj.get('order')}) {adj.get('text')}")

    if proc and proc.get(RECOVERY):
        lines.append("[恢复] " + "；".join(proc[RECOVERY]))
    if proc and proc.get(EXPECTED):
        lines.append("[预期结果] " + proc[EXPECTED])

    text = "\n".join(ln for ln in lines if ln)
    if max_len and len(text) > max_len:
        text = text[:max_len] + "..."
    return text
