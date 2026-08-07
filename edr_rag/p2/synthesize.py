"""P2-1 结构化答案合成器 — 输入检索上下文，输出结构化答案。

P2-2 逐条引用来源：答案的每个 steps / warnings / assertions 都携带来源引用
（source file / section_id / procedure_id），可追溯到 artifacts
（sections.jsonl / procedures.jsonl / actions.jsonl）。

设计 (§6, Answer Synthesis):
  检索(planner_retrieve / action_search / procedure_search)返回结构化上下文后，
  需要把它组织成最终呈递给用户/规划器的答案。本模块负责「答案生成」：

    synthesize_answer(user_request, context, llm=None) -> SynthesizedAnswer

输出结构化结果（SynthesizedAnswer.to_jsonl()）:
  - answer:               自然语言回答。LLM 可用时由 LLM 生成；否则由结构化
                          字段组装（保证无 LLM 时主链路仍可用且可读）。
  - steps:                list[dict] 操作步骤（order/action/target/instruction/
                          来源引用 source）。每个 step 携带 source（file/
                          section_id/procedure_id/action_id），可逐条溯源。
  - warnings:             list[str] 注意事项文本（含动作级别的 recovery/风险提示）。
                          保留为纯文本以兼容既有调用方与渲染路径。
  - warning_sources:      list[dict] 与 warnings 一一对应的来源引用（P2-2）：
                          warning_sources[i] 是 warnings[i] 的来源(file/section_id/
                          procedure_id/action_id)。
  - assertions:           list[dict] 断言/成功准则（P2-2），派生自动作的
                          success_criteria。每条含 text 与 source 引用。
  - preconditions:        list[str] 前置条件。
  - citations:            list[dict] 引用（来源 action/procedure/section id、
                          title、score），供整体溯源。
  - confidence:           float 综合置信度。
  - requires_confirmation:bool 是否需要人工确认。

LLM 可选（设计要点）:
  - `llm` 是一个可调用对象，签名 `callable(user_request, context_text) -> str`，
    返回生成的自然语言 answer。传入则用其输出填充 answer；不传（默认）则回落到
    `_assemble_answer` 基于结构化字段组装。
  - 本模块不引入任何新的运行时必需依赖：LLM 仅是可选的注入函数。
  - 缺依赖时永不报错——组装路径是纯 stdlib 的确定性逻辑。

输入 context 兼容两种形状（向前兼容现有检索入口）:
  1. planner_retrieve 的 dict:  {query, actions:[...], procedures:[...], sections:[...]}
  2. 动作检索 / 流程检索的 list[dict]（自动归一化为上下文）:
     - actions 走富字段(id/title/risk/status/confidence/preconditions/procedure/source)
     - procedures 走检索结果 dict
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional

#: 允许的 answer 生成模式（供 CLI 展示）。
ASSEMBLE_MODE = "assemble"
LLM_MODE = "llm"


@dataclass
class SynthesizedAnswer:
    """结构化答案（P2-1，P2-2 逐条引用来源）。

    `answer` 为全文；`steps`/`warnings`/`preconditions`/`citations` 为结构化
    字段；`confidence` 与 `requires_confirmation` 汇总给规划器用于决策。
    `generated_by` 记录 answer 是 LLM 生成还是组装（assemble），便于审计。

    P2-2 溯源字段:
      - 每个 `steps[i]` 含 `source`（file/section_id/procedure_id/action_id）。
      - `warning_sources[i]` 对应 `warnings[i]` 的来源。
      - `assertions` 为断言/成功准则，每条含 `text` 与 `source`。
    """

    answer: str
    steps: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    warning_sources: list[dict] = field(default_factory=list)
    assertions: list[dict] = field(default_factory=list)
    preconditions: list[str] = field(default_factory=list)
    citations: list[dict] = field(default_factory=list)
    confidence: float = 0.0
    requires_confirmation: bool = False
    generated_by: str = ASSEMBLE_MODE

    def to_jsonl(self) -> dict:
        return asdict(self)


# ---------- 上下文归一化 ----------


def _iter_action_results(context: Any) -> list[dict]:
    """从 context 中提取动作富字段结果列表。

    context 可能是 planner_retrieve dict 或 list[dict]；两者都归一到
    actions 的 list[dict] 返回（没有则 []）。
    """
    if isinstance(context, dict):
        actions = context.get("actions") or []
        return list(actions)
    if isinstance(context, (list, tuple)):
        return list(context)
    return []


def _normalize_context(context: Any) -> dict:
    """把各种输入形状归一到统一的 {actions, procedures, sections} 上下文。

    便于 synthesize_answer 内部统一读取，不破坏调用方传参方式。
    """
    if isinstance(context, dict) and ("actions" in context or "procedures" in context):
        return {
            "actions": context.get("actions") or [],
            "procedures": context.get("procedures") or [],
            "sections": context.get("sections") or [],
        }
    return {"actions": _iter_action_results(context), "procedures": [], "sections": []}


# ---------- 结构化字段提取 ----------


def _resolve_source(action: dict, *, kind: str = "action") -> dict:
    """从动作/流程/章节记录中解析来源引用（P2-2）。

    兼容多种形状:
      - 动作富字段:     action["source"] = {manual_id, procedure_id, file}
      - 流程记录:       proc["source"]  = {file, section_id, anchor}
      - 扁平形状:       action["file"] / action["section_id"] /
                        action["procedure_id"] / action["source_file"] ...

    Returns:
        {file, section_id, procedure_id, id} 的引用 dict；字段缺失时为空串。
    """
    raw_src = action.get("source")
    src = raw_src if isinstance(raw_src, dict) else {}
    ref = {
        "file": (
            src.get("file")
            or action.get("source_file")
            or action.get("file")
            or ""
        ),
        "section_id": src.get("section_id") or action.get("section_id") or "",
        "procedure_id": (
            src.get("procedure_id")
            or action.get("procedure_id")
            or (action.get("id", "") if str(action.get("id", "")).startswith("procedure.") else "")
        ),
        "id": action.get("id") or action.get("action_id") or "",
    }
    # 归一化 None -> ""
    return {k: (v if v is not None else "") for k, v in ref.items()}


def _extract_steps(action: dict) -> list[dict]:
    """从动作富字段里提取操作步骤（P2-2 逐条带来源引用）。

    优先读富字段的 procedure（ActionStep 字典序）；无 procedure 时尝试
    从裸 action/action_id 兜底为单步。每个 step 附带 `source` 引用
    （file/section_id/procedure_id/action_id），可追溯到 artifacts。
    """
    source = _resolve_source(action)
    steps: list[dict] = []
    for s in action.get("procedure") or []:
        if isinstance(s, dict):
            steps.append({
                "order": s.get("order", 0),
                "action": s.get("action", ""),
                "target": s.get("target", ""),
                "instruction": s.get("source_instruction", ""),
                "source": dict(source),
            })
    if not steps:
        # 兜底：没有 procedure 时，把动作本身作为一步说明。
        title = action.get("title", "")
        if title:
            steps.append({
                "order": 1,
                "action": action.get("action", ""),
                "target": "",
                "instruction": title,
                "source": dict(source),
            })
    return steps


def _extract_warnings(action: dict) -> tuple[list[str], list[dict]]:
    """从动作富字段 + 步骤说明里收集注意事项（P2-2 逐条带来源）。

    读 action 的 recovery（恢复/回滚说明）与 risk 提示，并补充
    requires_confirmation 的确认提示。去重保序，避免重复项。

    Returns:
        (warnings, sources)：warnings 为纯文本列表；sources 与之一一对应，
        每条为 _resolve_source 来源引用。
    """
    source = _resolve_source(action)
    warnings: list[str] = []
    sources: list[dict] = []
    seen: set[str] = set()

    def _add(text: str) -> None:
        text = text.strip()
        if text and text not in seen:
            seen.add(text)
            warnings.append(text)
            sources.append(dict(source))

    for r in action.get("recovery") or []:
        _add(r)

    risk = action.get("risk", "")
    if risk == "high":
        _add("该操作风险等级为 high，可能带来不可逆影响，请谨慎执行。")
    elif risk == "medium":
        _add("该操作风险等级为 medium，属于配置/变更类操作。")

    if action.get("requires_confirmation") or action.get("status") == "needs_review":
        _add("该操作需要人工确认后方可执行（requires_confirmation）。")

    return warnings, sources


def _extract_assertions(action: dict) -> list[dict]:
    """从动作的成功准则（success_criteria）提取断言（P2-2）。

    每条断言含 text 与 source 引用，作为答案的可验证断言，逐条可溯源。
    """
    source = _resolve_source(action)
    assertions: list[dict] = []
    seen: set[str] = set()
    for sc in action.get("success_criteria") or []:
        text = str(sc).strip()
        if text and text not in seen:
            seen.add(text)
            assertions.append({"text": text, "source": dict(source)})
    return assertions


def _extract_preconditions(action: dict) -> list[str]:
    return [p for p in (action.get("preconditions") or []) if isinstance(p, str) and p.strip()]


def _extract_citations(context: dict) -> list[dict]:
    """收集溯源引用（action/procedure/section id + title + score）。"""
    citations: list[dict] = []
    seen: set[str] = set()

    def _add(cite_id: str, title: str, kind: str, score: float) -> None:
        if not cite_id or cite_id in seen:
            return
        seen.add(cite_id)
        citations.append({
            "kind": kind,
            "id": cite_id,
            "title": title or "",
            "score": round(float(score or 0.0), 3),
        })

    for a in context.get("actions") or []:
        _add(
            a.get("id") or a.get("action_id", ""),
            a.get("title", ""),
            "action",
            a.get("score", 0.0),
        )
    for p in context.get("procedures") or []:
        _add(p.get("id", ""), p.get("title", ""), "procedure", p.get("score", 0.0))
    for s in context.get("sections") or []:
        _add(s.get("id", ""), s.get("title", ""), "section", s.get("score", 0.0))
    return citations


def _compute_confidence(context: dict) -> float:
    """基于命中条目置信度 + 数量计算综合置信度。

    取所有命中条目的 confidence 均值；无命中则 0.0。
    """
    values: list[float] = []
    for a in context.get("actions") or []:
        if a.get("confidence") is not None:
            values.append(float(a["confidence"]))
    for p in context.get("procedures") or []:
        if p.get("confidence") is not None:
            values.append(float(p["confidence"]))
    if not values:
        return 0.0
    return round(sum(values) / len(values), 3)


def _compute_requires_confirmation(context: dict) -> bool:
    """任一命中动作需要确认 / 处于 needs_review，则整体需要确认。"""
    for a in context.get("actions") or []:
        if a.get("requires_confirmation") or a.get("status") == "needs_review":
            return True
    return False


# ---------- answer 组装（无 LLM 路径） ----------


def _assemble_answer(user_request: str, context: dict, steps: list[dict],
                     warnings: list[str], preconditions: list[str]) -> str:
    """基于结构化字段组装自然语言 answer（无 LLM 时的确定性路径）。

    组装策略：
      - 有动作/流程命中 -> 标题 + 分步操作步骤（含前置条件与注意事项）。
      - 无命中 -> 明确的「未找到明确操作步骤」提示。
    保证输出可读、可呈递、可审计。
    """
    lines: list[str] = []
    actions = context.get("actions") or []
    procedures = context.get("procedures") or []
    sections = context.get("sections") or []

    primary = actions or procedures
    pre = "、".join(preconditions)

    if primary:
        # 标题：优先动作/流程标题
        title = next((x.get("title", "") for x in primary if x.get("title", "")), "相关操作")
        header = f"针对「{user_request}」，建议执行：{title}"
        if pre:
            header += f"（前置条件：{pre}）"
        lines.append(header)
        if steps:
            lines.append("")
            lines.append("操作步骤：")
            for s in steps:
                inst = s.get("instruction") or f"{s.get('action', '')} {s.get('target', '')}".strip()
                lines.append(f"  {s.get('order', 1)}. {inst or '（无描述）'}")
        if warnings:
            lines.append("")
            lines.append("注意事项：")
            for w in warnings:
                lines.append(f"  - {w}")
        # 补充流程/章节的补充说明（若有）
        extra = [x.get("title", "") for x in (procedures or sections) if x.get("title", "")]
        if extra:
            lines.append("")
            lines.append("相关参考：" + "；".join(extra))
    else:
        lines.append(f"未检索到与「{user_request}」明确匹配的操作步骤。请尝试更换关键词，或查阅完整手册。")

    return "\n".join(lines)


# ---------- 对外主入口 ----------


def synthesize_answer(
    user_request: str,
    context: Any,
    llm: Optional[Callable[[str, str], str]] = None,
) -> SynthesizedAnswer:
    """输入检索上下文，输出结构化答案（P2-1）。

    Args:
        user_request: 用户问题 / 请求原文。
        context: 检索上下文，兼容 planner_retrieve 的 dict 或检索结果的
            list[dict]（见模块 docstring）。
        llm: 可选的可调用对象 `(user_request, context_text) -> str`，
            用于生成自然语言 answer。为 None 时回落为组装模式（保证
            无 LLM 时主链路可用）。调用异常会被捕获并回落到组装模式。

    Returns:
        SynthesizedAnswer（结构化结果）。
    """
    ctx = _normalize_context(context)
    actions = ctx.get("actions") or []
    # 目前结构化主要源自动作富字段；procedure/section 兜底也纳入 counts。
    primary = actions or ctx.get("procedures") or []

    steps: list[dict] = []
    # warnings 与 warning_sources 并行；_extract_warnings 返回 (texts, sources)。
    warnings: list[str] = []
    warning_sources: list[dict] = []
    assertions: list[dict] = []
    preconditions: list[str] = []
    for a in actions:
        steps.extend(_extract_steps(a))
        a_warnings, a_sources = _extract_warnings(a)
        warnings.extend(a_warnings)
        warning_sources.extend(a_sources)
        assertions.extend(_extract_assertions(a))
        preconditions.extend(_extract_preconditions(a))
    # 去重（按内容）；warnings 与 warning_sources 同步去重，保证一一对应
    unique_steps: list[dict] = []
    seen_steps: set[tuple] = set()
    for s in steps:
        key = (s.get("order"), s.get("action"), s.get("target"))
        if key in seen_steps:
            continue
        seen_steps.add(key)
        unique_steps.append(s)
    seen_w: set[str] = set()
    uniq_warnings: list[str] = []
    uniq_warning_sources: list[dict] = []
    for w, src in zip(warnings, warning_sources):
        if w in seen_w:
            continue
        seen_w.add(w)
        uniq_warnings.append(w)
        uniq_warning_sources.append(src)
    seen_p: set[str] = set()
    uniq_pre = [p for p in preconditions if not (p in seen_p or seen_p.add(p))]
    # 断言去重（按 text）
    uniq_assertions: list[dict] = []
    seen_a: set[str] = set()
    for a in assertions:
        if a["text"] in seen_a:
            continue
        seen_a.add(a["text"])
        uniq_assertions.append(a)

    citations = _extract_citations(ctx)
    confidence = _compute_confidence(ctx)
    requires_confirmation = _compute_requires_confirmation(ctx)

    generated_by = ASSEMBLE_MODE
    answer = ""

    if llm is not None:
        context_text = _render_context_text(user_request, primary, steps, uniq_warnings, uniq_pre)
        try:
            generated = llm(user_request, context_text)
            if generated and generated.strip():
                answer = generated.strip()
                generated_by = LLM_MODE
        except Exception:  # LLM 失败不阻塞主链路，回落到组装
            answer = ""

    if not answer:
        answer = _assemble_answer(
            user_request, ctx, unique_steps, uniq_warnings, uniq_pre
        )

    return SynthesizedAnswer(
        answer=answer,
        steps=unique_steps,
        warnings=uniq_warnings,
        warning_sources=uniq_warning_sources,
        assertions=uniq_assertions,
        preconditions=uniq_pre,
        citations=citations,
        confidence=confidence,
        requires_confirmation=requires_confirmation,
        generated_by=generated_by,
    )


def _render_context_text(
    user_request: str,
    primary: list[dict],
    steps: list[dict],
    warnings: list[str],
    preconditions: list[str],
) -> str:
    """把结构化上下文渲染成自然语言文本，供 LLM 生成 answer 时参考。"""
    lines = [f"用户请求：{user_request}"]
    if primary:
        title = primary[0].get("title", "") if isinstance(primary[0], dict) else ""
        if title:
            lines.append(f"建议操作：{title}")
    if preconditions:
        lines.append("前置条件：" + "；".join(preconditions))
    if steps:
        lines.append("操作步骤：")
        for s in steps:
            inst = s.get("instruction") or f"{s.get('action', '')} {s.get('target', '')}".strip()
            lines.append(f"  {s.get('order', 1)}. {inst or '（无描述）'}")
    if warnings:
        lines.append("注意事项：" + "；".join(warnings))
    return "\n".join(lines)


def _render_source_ref(source: Any) -> str:
    """把来源引用 dict 渲染成可读短串（P2-2）。

    优先级: file 本身 > file + section_id > file + procedure_id。全部为空则
    返回空串，调用方决定是否展示。
    """
    if not isinstance(source, dict):
        return ""
    file = source.get("file") or ""
    section_id = source.get("section_id") or ""
    procedure_id = source.get("procedure_id") or ""
    bits = [b for b in (file, section_id, procedure_id) if b]
    return "/".join(bits)


def answer_to_text(answer: SynthesizedAnswer) -> str:
    """把结构化答案渲染成纯文本（CLI/展示用）。

    输出含 answer 正文 + 逐条来源（steps/warnings/assertions）+ 引用列表，
    方便终端呈现与溯源性展示。
    """
    out = [answer.answer]

    if answer.steps:
        out.append("")
        out.append("操作步骤（含来源）：")
        for s in answer.steps:
            ref = _render_source_ref(s.get("source"))
            suffix = f"  [来源: {ref}]" if ref else ""
            out.append(f"  {s.get('order', 1)}. {s.get('instruction', '')}{suffix}")

    if answer.warnings:
        out.append("")
        out.append("注意事项（含来源）：")
        for i, w in enumerate(answer.warnings):
            src = answer.warning_sources[i] if i < len(answer.warning_sources) else {}
            ref = _render_source_ref(src)
            suffix = f"  [来源: {ref}]" if ref else ""
            out.append(f"  - {w}{suffix}")

    if answer.assertions:
        out.append("")
        out.append("断言/成功准则（含来源）：")
        for a in answer.assertions:
            ref = _render_source_ref(a.get("source"))
            suffix = f"  [来源: {ref}]" if ref else ""
            out.append(f"  - {a.get('text', '')}{suffix}")

    if answer.citations:
        out.append("")
        out.append("引用来源：")
        for c in answer.citations:
            out.append(f"  [{c['kind']}] {c.get('id', '')} — {c.get('title', '')} (score={c.get('score', 0)})")
    return "\n".join(out)
