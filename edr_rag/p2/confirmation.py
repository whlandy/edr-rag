"""P2-5 高风险 Action 显式确认机制。

背景:
  答案合成器（P2-1/P2-3/P2-4）已经能在结构化答案里标记
  requires_confirmation=True / dangerous=True，但这些只是「告知」字段，
  消费方（agent/规划器）可以无视它们直接读取 steps 并执行。P2-5 在
  生成流程上叠加一层「强制显式确认」门控：

    - 当答案命中 requires_confirmation=true 的 Action（或 answer_status
      == 'conflict'），生成流程不允许直接释放可执行步骤，必须由调用方
      显式传 confirm=true 才会放行完整步骤。
    - MCP 的 ask 工具 / API 的 ask_answer 都需要 confirm 字段。

设计:
  - 纯 stdlib、确定性逻辑，无外部依赖，缺依赖时主链路不报错。
  - 门控不改变答案的基础语义字段（answer_status 仍为 answered/refused/
    conflict，requires_confirmation / dangerous 保留），而是增加一组
    confirmation 元数据字段来表述门控状态，保持既有调用方兼容。
  - gate_answer(answer, confirm=...) 三态:
      confirmed    — 调用方显式确认，放行完整 answer（原样返回 + 标注）。
      not_required — 该答案不需要确认，直接放行。
      required     — 需要确认但调用方未确认，门控：扣留可执行内容，
                     改为返回「请确认」提示负载。
"""

from __future__ import annotations

from typing import Optional

#: 门控三态（confirmation_status 字段取值）。
CONFIRMATION_NOT_REQUIRED = "not_required"
CONFIRMATION_REQUIRED = "required"
CONFIRMATION_CONFIRMED = "confirmed"

#: 答案状态（与 synthesize 保持一致引用，避免硬编码）。
ANSWER_CONFLICT = "conflict"


def needs_explicit_confirmation(answer: dict) -> bool:
    """该答案是否需要显式人工确认。

    满足任一即需确认（与 P2-4 的 dangerous 口径一致）:
      - requires_confirmation == True（含 high/medium 风险动作）
      - answer_status == 'conflict'（冲突文档需澄清后执行）

    Args:
        answer: SynthesizedAnswer.to_jsonl() / ask_answer 返回的 dict。

    Returns:
        True 表示需要显式确认，False 表示无需。
    """
    return bool(
        answer.get("requires_confirmation")
        or answer.get("answer_status") == ANSWER_CONFLICT
    )


def _confirmation_request_message(answer: dict) -> str:
    """构造「请确认」的自然语言提示，说明哪些动作需要显式确认。

    从 citations / steps 的 source 提取高风险动作的标题与 id，便于消费方
    明确知道自己将要放行什么。
    """
    lines = [
        "该答案包含需要人工显式确认的操作（requires_confirmation），"
        "以下可执行步骤已在生成流程中被扣留，未予释放。",
        "",
    ]

    # 找出触发确认的动作（citations 中 kind == action 的条目）
    actions = [
        c for c in (answer.get("citations") or [])
        if isinstance(c, dict) and c.get("kind") == "action"
    ]
    if actions:
        lines.append("涉及的需要确认操作：")
        for c in actions:
            title = c.get("title") or ""
            aid = c.get("id") or ""
            tag = f"{title}（{aid}）" if title and aid else (title or aid)
            if tag:
                lines.append(f"  - {tag}  [来源: {c.get('id', '')}]")
    elif answer.get("requires_confirmation"):
        lines.append("答案整体被标记为需要确认（requires_confirmation=True）。")

    if answer.get("answer_status") == ANSWER_CONFLICT:
        lines.append("")
        lines.append("说明：检测到相互冲突的文档，需澄清后再执行。")

    lines.append("")
    lines.append("请在确认理解风险后，以 confirm=true 重新发起请求以放行操作步骤。")

    return "\n".join(lines)


def confirmation_request(answer: dict) -> dict:
    """构造一个「确认请求」负载（不修改原 answer）。

    保留风险/来源信息，把可执行内容（steps/preconditions/warnings/
    assertions）清空，answer 文本替换为请确认提示。调用方无 LLM，纯
    stdlib 组装，确定性。

    Args:
        answer: 需要确认的答案 dict。

    Returns:
        门控后的答案 dict（新对象，不原地修改）。
    """
    return _build_gated(answer, reason=CONFIRMATION_REQUIRED)


def _build_gated(answer: dict, *, reason: str) -> dict:
    """构造门控/确认后的答案 dict（不原地修改原 answer）。"""
    out = dict(answer)
    out["confirmation_status"] = reason
    out["needs_confirmation"] = reason in (CONFIRMATION_REQUIRED, CONFIRMATION_CONFIRMED)
    # 可执行内容在门控时被扣留，避免未确认前释放高风险步骤。
    out["steps"] = []
    out["preconditions"] = []
    out["warnings"] = []
    out["assertions"] = []
    out["warning_sources"] = []
    out["answer"] = _confirmation_request_message(out)
    out["confirmation_request"] = out["answer"]
    return out


def gate_answer(answer: dict, *, confirm: bool) -> dict:
    """对结构化答案应用显式确认门控 (P2-5)。

    Args:
        answer: SynthesizedAnswer.to_jsonl() / ask_answer 返回的 dict
            （含 requires_confirmation / answer_status 等字段）。
        confirm: 调用方是否显式确认。True 表示已确认，False 表示未确认。

    Returns:
        处理后的答案 dict:
          - 不需要确认        → 原样 + confirmation_status='not_required'。
          - 需要确认且已确认  → 放行 + confirmation_status='confirmed'。
          - 需要确认且未确认  → 门控 + confirmation_status='required'，
                                扣留可执行步骤，返回请确认提示。
        不会原地修改入参 dict。
    """
    needs = needs_explicit_confirmation(answer)

    out = dict(answer)
    if not needs:
        out["confirmation_status"] = CONFIRMATION_NOT_REQUIRED
        out.setdefault("needs_confirmation", False)
        return out

    if confirm:
        out["confirmation_status"] = CONFIRMATION_CONFIRMED
        out["needs_confirmation"] = True
        out["confirmation_request"] = _confirmation_request_message(out)
        return out

    return _build_gated(out, reason=CONFIRMATION_REQUIRED)
