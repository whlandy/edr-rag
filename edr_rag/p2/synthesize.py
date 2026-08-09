"""P2-1 结构化答案合成器 — 输入检索上下文，输出结构化答案。

P2-2 逐条引用来源：答案的每个 steps / warnings / assertions 都携带来源引用
（source file / section_id / procedure_id），可追溯到 artifacts
（sections.jsonl / procedures.jsonl / actions.jsonl）。

P2-3 证据不足拒答 + 冲突处理：
  - 置信度低于阈值或证据不足时明确拒答（不编造）：
    SynthesizedAnswer.answer_status == 'refused'，refusal_reason 说明原因
    （insufficient_evidence / low_confidence）。
  - 检测到冲突文档时标注冲突并请求澄清：
    answer_status == 'conflict'，conflicts 列出各冲突（类型/描述/来源），
    并置 requires_confirmation=True（需澄清后再执行）。

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
  - answer_status:        str  答案状态（P2-3）：answered / refused / conflict。
  - refusal_reason:       Optional[str] 拒答原因（P2-3）。
  - conflicts:            list[dict] 冲突列表（P2-3），每条含 type/description/sources。

LLM 可选（设计要点）:
  - `llm` 是一个可调用对象，签名 `callable(user_request, context_text) -> str`，
    返回生成的自然语言 answer。传入则用其输出填充 answer；不传（默认）则回落到
    `_assemble_answer` 基于结构化字段组装。
  - 本模块不引入任何新的运行时必需依赖：LLM 仅是可选的注入函数。
  - 缺依赖时永不报错——组装路径是纯 stdlib 的确定性逻辑。
  - 拒答/冲突状态与 answer 文本由确定性逻辑计算，LLM 出现与否都不改变状态判定。

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

#: 答案状态（P2-3）：
#:   answered — 正常作答；refused — 证据不足/置信度过低，明确拒答（不编造）；
#:   conflict — 检测到冲突文档，标注冲突并请求澄清。
ANSWER_ANSWERED = "answered"
ANSWER_REFUSED = "refused"
ANSWER_CONFLICT = "conflict"

#: 证据不足拒答阈值：当没有足够命中条目，或综合置信度低于该值时，拒答而非编造。
CONFIDENCE_REFUSE_THRESHOLD = 0.4

#: 互斥操作对（同一目标出现相反操作 → 冲突）。P4 扩充覆盖面。
_OPPOSITE_OPERATIONS: tuple[tuple[str, str], ...] = (
    ("enable", "disable"),
    ("启用", "禁用"),
    ("start", "stop"),
    ("启动", "停止"),
    ("open", "close"),
    ("打开", "关闭"),
    ("allow", "block"),
    ("放行", "阻断"),
    ("install", "uninstall"),
    ("安装", "卸载"),
    # --- P4 新增：EDR 场景里高频且后果严重的互斥对 ---
    ("add", "remove"),
    ("添加", "移除"),
    ("添加", "删除"),
    ("create", "delete"),
    ("新建", "删除"),
    ("connect", "disconnect"),
    ("连接", "断开"),
    ("mount", "unmount"),
    ("挂载", "卸载"),
    ("上线", "下线"),
    ("注册", "注销"),
    ("加入", "退出"),
    ("恢复", "隔离"),
    ("信任", "阻断"),
    ("白名单", "黑名单"),
    ("grant", "revoke"),
    ("授权", "撤销"),
    ("resume", "suspend"),
    ("恢复", "暂停"),
    ("锁定", "解锁"),
    ("lock", "unlock"),
    ("显示", "隐藏"),
    ("show", "hide"),
)

#: 版本/适用范围字段 —— 同一操作在不同版本下步骤不同是手册里最常见的真实冲突。
_VERSION_KEYS = ("version", "product_version", "applies_to", "edition", "platform")

#: 风险等级序数，用于判断「差异过大」。
_RISK_ORDER = {"low": 0, "medium": 1, "high": 2}


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
    #: P4: 旧口径（抽取阶段静态 confidence 的均值），保留供审计对照。
    extraction_confidence: float = 0.0
    requires_confirmation: bool = False
    generated_by: str = ASSEMBLE_MODE
    # P2-3: 答案状态 / 拒答原因 / 冲突列表
    answer_status: str = ANSWER_ANSWERED
    refusal_reason: Optional[str] = None
    conflicts: list[dict] = field(default_factory=list)
    #: P4: 命中记录里检出的疑似 prompt injection 信号
    #: [{id, reasons}]，供运维审计与告警。空表示未检出。
    injection_flags: list[dict] = field(default_factory=list)

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


#: 综合置信度的分量权重 (P4 Lock 5)。缺失的分量会被剔除并重新归一化 ——
#: 「没有该信号」不等于「该信号很差」。
_CONFIDENCE_WEIGHTS = {
    "extraction": 0.5,   # 抽取阶段写死的静态置信度
    "retrieval": 0.3,    # 本次检索的质量（top-1 是否决定性胜出）
    "evidence": 0.2,     # 证据量（命中条数）
}

#: 证据量饱和点：命中到这个条数就认为证据量充分。
_EVIDENCE_SATURATION = 3


def _extraction_confidence(context: dict) -> Optional[float]:
    """抽取阶段的静态置信度均值；无标注返回 None。"""
    values: list[float] = []
    for key in ("actions", "procedures"):
        for item in context.get(key) or []:
            if item.get("confidence") is not None:
                try:
                    values.append(float(item["confidence"]))
                except (TypeError, ValueError):
                    continue
    if not values:
        return None
    return sum(values) / len(values)


def _retrieval_confidence(context: dict) -> Optional[float]:
    """本次检索的质量信号；结果不带分数时返回 None。

    用 **top-1 相对 top-2 的领先幅度**（margin）而不是分数绝对值 —— 各策略
    的分数量纲完全不同（keyword 是词频计数、bm25 是 idf 加权、rrf_score 在
    0.03 量级、rerank_score 是 logit），绝对值之间没有可比性，margin 是
    scale-free 的。

    语义：
      - 没有正分命中            -> 0.0（检索没找到相关的东西）
      - 只有一条命中            -> 0.75（决定性，但没有对照）
      - top1 远超 top2          -> 接近 1.0
      - top1 与 top2 难分伯仲   -> 接近 0.5（检索本身也不确定选哪个）
    """
    scores: list[float] = []
    for key in ("actions", "procedures", "sections"):
        for item in context.get(key) or []:
            raw = item.get("rerank_score")
            if raw is None:
                raw = item.get("score")
            if raw is None:
                continue
            try:
                scores.append(float(raw))
            except (TypeError, ValueError):
                continue
    if not scores:
        return None
    scores.sort(reverse=True)
    top1 = scores[0]
    if top1 <= 0:
        return 0.0
    if len(scores) == 1:
        return 0.75
    top2 = scores[1]
    margin = (top1 - max(top2, 0.0)) / top1
    margin = min(max(margin, 0.0), 1.0)
    return 0.5 + 0.5 * margin


def _evidence_confidence(context: dict) -> float:
    """证据量分量：命中条数的饱和函数。"""
    count = len(context.get("actions") or []) + len(context.get("procedures") or [])
    return min(1.0, count / _EVIDENCE_SATURATION)


def _compute_confidence(context: dict) -> float:
    """综合置信度 —— 抽取置信度 + 检索质量 + 证据量的加权平均 (P4 Lock 5)。

    P0–P3 的实现只取命中条目 ``confidence`` 静态字段的**算术平均**，与本次
    检索的质量完全无关。而「证据不足拒答」的阈值就建在这个数上 —— 也就是说
    拒答判定其实不看检索好不好，只看抽取阶段写死的字段。两个后果：

      1. 检索到一堆不相关但 confidence 标得很高的动作 -> 照样自信作答。
      2. 一条高置信 + 一条低置信被平均稀释 -> 明明有确凿证据却拒答。

    现在三个分量都参与，缺失的分量剔除后重新归一化（「没有该信号」不等于
    「该信号很差」——这对没有 score 的调用方保持向后兼容）。

    ``SynthesizedAnswer.extraction_confidence`` 保留旧口径供审计对照。
    """
    parts: list[tuple[float, float]] = []

    extraction = _extraction_confidence(context)
    if extraction is not None:
        parts.append((_CONFIDENCE_WEIGHTS["extraction"], extraction))

    retrieval = _retrieval_confidence(context)
    if retrieval is not None:
        parts.append((_CONFIDENCE_WEIGHTS["retrieval"], retrieval))

    parts.append((_CONFIDENCE_WEIGHTS["evidence"], _evidence_confidence(context)))

    total_weight = sum(w for w, _ in parts)
    if total_weight <= 0:  # pragma: no cover - 权重常量恒为正
        return 0.0
    blended = sum(w * v for w, v in parts) / total_weight
    return round(min(max(blended, 0.0), 1.0), 3)


def _compute_requires_confirmation(context: dict) -> bool:
    """任一命中动作需要确认 / 处于 needs_review，则整体需要确认。"""
    for a in context.get("actions") or []:
        if a.get("requires_confirmation") or a.get("status") == "needs_review":
            return True
    return False


# ---------- 证据充分性 & 冲突检测 (P2-3) ----------


def _has_primary_evidence(ctx: dict) -> bool:
    """是否有可作为依据的命中条目（动作/流程）。"""
    return bool(ctx.get("actions") or ctx.get("procedures"))


def _evidence_refusal_reason(
    ctx: dict, confidence: float, threshold: float = CONFIDENCE_REFUSE_THRESHOLD
) -> Optional[str]:
    """判断是否证据不足需要拒答，返回拒答原因；可作答返回 None。

    规则（P2-3）：
      - 无任何 action/procedure 命中 → insufficient_evidence（不编造）。
      - 有命中但综合置信度低于 threshold → low_confidence。
    """
    if not _has_primary_evidence(ctx):
        return "insufficient_evidence"
    if confidence < threshold:
        return "low_confidence"
    return None


def _op_canonical(op: str) -> str:
    """把操作字符串归一化为小写，便于比较。"""
    return op.strip().lower()


def _step_order(order: Any) -> Optional[int]:
    """把步骤序号规整为 int；缺失或非数值返回 None（该步不参与顺序比较）。

    抽取自 artifacts 的 order 可能是字符串甚至中文序数，裸 int() 会在
    答案合成中途抛错——冲突检测是启发式，脏数据只该让它少报一条。
    """
    if order is None or isinstance(order, bool):
        return None
    try:
        return int(order)
    except (TypeError, ValueError):
        return None


def _is_opposite(op1: str, op2: str) -> bool:
    """判断两个操作是否互为反向（互斥）。"""
    op1, op2 = _op_canonical(op1), _op_canonical(op2)
    if not op1 or not op2:
        return False
    for a, b in _OPPOSITE_OPERATIONS:
        if (op1 == a and op2 == b) or (op1 == b and op2 == a):
            return True
    return False


def _procedure_scope(item: dict, src: dict) -> str:
    """返回一条记录的「所属流程」标识，用作步骤级冲突的比较范围。

    步骤级冲突要回答的是「描述**同一个流程**的两份文档是否互相矛盾」。若
    不限定范围而按 (order, action) 全局分组，任意两个无关动作只要第 1 步
    都是「点击」就会被判为冲突——检索 top-k 本来就是不同流程，这会让几乎
    每个答案都退化成 conflict + requires_confirmation。

    取值优先级：
      1. source.procedure_id —— 同一流程的不同文档/变体共享它，是最准的范围键；
      2. 记录自身 id —— 缺 procedure_id 时退化为「每条记录自成一个范围」，
         此时只有单条记录内部的重复步骤才可能判冲突，跨记录一律不比较
         （保守：漏报只是退回普通作答，误报会污染每一个答案）。
    """
    return str(src.get("procedure_id") or "") or f"#{item.get('id') or id(item)}"


def _detect_conflicts(ctx: dict) -> list[dict]:
    """检测检索到的文档/动作之间的冲突（P2-3）。

    启发式（确定性、纯 stdlib，无外部依赖）：
      1. 互斥操作：同一目标(target)同时出现相反的操作（启用/禁用、打开/关闭等）。
         按 target 全局比较——同一个目标既要启用又要禁用，无论来自哪份文档
         都是真冲突。
      2. 步骤级冲突：**同一流程内**（见 _procedure_scope）同一 step order 的
         同一操作动作，目标(target)不一致。
      3. 风险矛盾：同一标题/操作在不同来源里的风险等级差异过大（low vs high）。

    Returns:
        list[dict]，每条含 type / description / sources（涉及来源引用列表）。
        无冲突返回 []。
    """
    conflicts: list[dict] = []
    items: list[dict] = list(ctx.get("actions") or []) + list(ctx.get("procedures") or [])
    if len(items) < 2:
        return conflicts

    # 1) 互斥操作：target -> [(op, source_ref)]
    target_ops: dict[str, list[tuple[str, dict]]] = {}
    # 2) 步骤级冲突：(流程范围, order, op) -> [(target, source_ref)]
    step_targets: dict[tuple[str, int, str], list[tuple[str, dict]]] = {}
    # 3) 风险矛盾：title -> [(risk, source_ref)]
    title_risks: dict[str, list[tuple[str, dict]]] = {}

    for it in items:
        src = _resolve_source(it)
        scope = _procedure_scope(it, src)
        title = str(it.get("title") or "").strip() or str(it.get("id") or "").strip()
        if title:
            title_risks.setdefault(title, []).append((str(it.get("risk") or ""), dict(src)))
        for s in it.get("procedure") or []:
            if not isinstance(s, dict):
                continue
            op = str(s.get("action") or "").strip()
            tgt = str(s.get("target") or "").strip()
            order = _step_order(s.get("order"))
            if tgt and op:
                target_ops.setdefault(tgt, []).append((op, dict(src)))
            if order is not None and op:
                key = (scope, order, _op_canonical(op))
                step_targets.setdefault(key, []).append((tgt, dict(src)))

    # 互斥操作检测：对每个目标，检查是否存在相反操作对
    ops_by_target: dict[str, set[str]] = {
        tgt: {_op_canonical(op) for op, _ in pairs}
        for tgt, pairs in target_ops.items()
    }
    for tgt, ops in ops_by_target.items():
        op_list = list(ops)
        for i in range(len(op_list)):
            for j in range(i + 1, len(op_list)):
                if _is_opposite(op_list[i], op_list[j]):
                    sources = [src for op, src in target_ops[tgt]
                               if _op_canonical(op) in (op_list[i], op_list[j])]
                    conflicts.append({
                        "type": "opposite_operations",
                        "description": f"目标「{tgt}」出现互斥操作：{op_list[i]} 与 {op_list[j]}，操作互相冲突。",
                        "sources": sources,
                    })

    # 步骤级冲突：同一流程内的同一 (order, action) 出现多个不同 target
    for (_scope, order, op), pairs in step_targets.items():
        distinct_targets = {t for t, _ in pairs if t}
        if len(distinct_targets) > 1:
            sources = [src for _, src in pairs]
            conflicts.append({
                "type": "step_conflict",
                "description": (
                    f"第 {order} 步操作「{op}」在不同文档中的目标不一致："
                    + "、".join(sorted(distinct_targets))
                ),
                "sources": sources,
            })

    # 风险矛盾：同一标题的风险等级跨度 >= 2 级（low vs high）
    for title, risks in title_risks.items():
        levels = {
            _RISK_ORDER[r] for r, _ in risks if r in _RISK_ORDER
        }
        if levels and (max(levels) - min(levels)) >= 2:
            sources = [src for _, src in risks]
            conflicts.append({
                "type": "risk_contradiction",
                "description": f"操作「{title}」在不同来源中的风险等级矛盾（low vs high）。",
                "sources": sources,
            })

    # P4 新增：版本/适用范围分歧。手册里最常见的**真实**冲突不是「启用 vs
    # 禁用」这种教科书式互斥，而是「V5 这么做、V6 那么做」——旧实现完全
    # 检测不到，会把两个版本的步骤混在一起给用户。
    conflicts.extend(_detect_version_conflicts(items))

    # P4 新增：同一操作的步骤数差异过大。同名操作一个 3 步一个 9 步，
    # 通常意味着命中了两份不同的文档（不同版本/不同产品线）。
    conflicts.extend(_detect_step_count_conflicts(items))

    return conflicts


def _item_version(it: dict) -> str:
    """读取一条记录声明的版本/适用范围（顶层或 metadata 内嵌）。"""
    for key in _VERSION_KEYS:
        val = it.get(key)
        if val:
            return str(val).strip()
    meta = it.get("metadata")
    if isinstance(meta, dict):
        for key in _VERSION_KEYS:
            val = meta.get(key)
            if val:
                return str(val).strip()
    return ""


def _detect_version_conflicts(items: list[dict]) -> list[dict]:
    """同一操作标题下出现多个不同版本/适用范围 → 需要澄清 (P4)。"""
    by_title: dict[str, list[tuple[str, dict]]] = {}
    for it in items:
        title = str(it.get("title") or "").strip()
        version = _item_version(it)
        if title and version:
            by_title.setdefault(title, []).append((version, _resolve_source(it)))

    conflicts: list[dict] = []
    for title, pairs in by_title.items():
        versions = {v for v, _ in pairs}
        if len(versions) > 1:
            conflicts.append({
                "type": "version_divergence",
                "description": (
                    f"操作「{title}」在不同版本/适用范围下步骤可能不同："
                    + "、".join(sorted(versions))
                    + "。请确认目标环境的版本后再执行。"
                ),
                "sources": [src for _, src in pairs],
            })
    return conflicts


def _detect_step_count_conflicts(items: list[dict]) -> list[dict]:
    """同名操作的步骤数差异过大 → 大概率命中了两份不同文档 (P4)。"""
    by_title: dict[str, list[tuple[int, dict]]] = {}
    for it in items:
        title = str(it.get("title") or "").strip()
        steps = it.get("procedure") or []
        if title and isinstance(steps, list) and steps:
            by_title.setdefault(title, []).append((len(steps), _resolve_source(it)))

    conflicts: list[dict] = []
    for title, pairs in by_title.items():
        counts = [n for n, _ in pairs]
        if len(counts) < 2:
            continue
        lo, hi = min(counts), max(counts)
        # 阈值：多出一倍且至少差 3 步 —— 避免把「多一步确认」误报成冲突。
        if hi >= lo * 2 and (hi - lo) >= 3:
            conflicts.append({
                "type": "step_count_divergence",
                "description": (
                    f"操作「{title}」在不同来源中的步骤数差异过大"
                    f"（{lo} 步 vs {hi} 步），可能来自不同版本或不同产品线的手册。"
                ),
                "sources": [src for _, src in pairs],
            })
    return conflicts


def _assemble_refusal(user_request: str, reason: str) -> str:
    """生成明确的拒答文本（P2-3）——不编造，提示证据不足。"""
    if reason == "low_confidence":
        return (
            f"关于「{user_request}」，检索到的相关资料的置信度过低，"
            f"无法据此给出可靠操作步骤。为避免误导，暂不提供具体步骤。"
            f"请尝试更换更明确的关键词，或查阅完整手册后重试。"
        )
    return (
        f"未检索到与「{user_request}」明确匹配的操作步骤（证据不足）。"
        f"为避免编造，暂不给出具体操作。请更换关键词重试，"
        f"或直接查阅完整手册相关章节。"
    )


def _assemble_conflict(user_request: str, steps: list[dict], conflicts: list[dict]) -> str:
    """生成冲突提示文本（P2-3）——标注冲突并请求澄清。"""
    lines = [f"针对「{user_request}」，检索到相互冲突的文档，无法直接给出唯一操作步骤。"]
    lines.append("")
    lines.append("检测到以下冲突：")
    for i, c in enumerate(conflicts, 1):
        lines.append(f"  {i}. {c['description']}")
    lines.append("")
    lines.append("请在以下选项中选择以澄清：")
    if steps:
        # 列出第一组步骤作为候选
        lines.append("  a. 按检索到的步骤执行（见下方操作步骤）")
    lines.append("  b. 提供更具体的场景/产品版本，以便重新检索")
    return "\n".join(lines)


# ---------- answer 组装（无 LLM 路径） ----------


def _assemble_answer(
    *,
    user_request: str,
    context: dict,
    steps: list[dict],
    warnings: list[str],
    preconditions: list[str],
    conflicts: list[dict],
    answer_status: str,
    refusal_reason: Optional[str],
) -> str:
    """基于结构化字段组装自然语言 answer（无 LLM 时的确定性路径）。

    组装策略（P2-3）：
      - answer_status == 'refused' -> 明确拒答文本（不编造）。
      - answer_status == 'conflict' -> 标注冲突并请求澄清。
      - 有动作/流程命中 -> 标题 + 分步操作步骤（含前置条件与注意事项）。
      - 无命中（证据不足）-> 明确的「未检索到 / 证据不足」拒答提示。
    保证输出可读、可呈递、可审计。
    """
    if answer_status == ANSWER_REFUSED:
        return _assemble_refusal(user_request, refusal_reason or "insufficient_evidence")

    if answer_status == ANSWER_CONFLICT:
        return _assemble_conflict(user_request, steps, conflicts)

    lines: list[str] = []
    primary = context.get("actions") or context.get("procedures") or []

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
        extra = [x.get("title", "") for x in (context.get("procedures") or context.get("sections") or []) if x.get("title", "")]
        if extra:
            lines.append("")
            lines.append("相关参考：" + "；".join(extra))
    else:
        lines.append(f"未检索到与「{user_request}」明确匹配的操作步骤。请尝试更换关键词，或查阅完整手册。")

    return "\n".join(lines)


# ---------- P4: 全路径注入清洗 ----------


def _sanitize_text(text: Any) -> str:
    """对单条文本做注入清洗（P4 严重问题 8）。"""
    from ..security import sanitize_document_text

    return sanitize_document_text(text if isinstance(text, str) else str(text or ""))


def _sanitize_structured(
    steps: list[dict],
    warnings: list[str],
    preconditions: list[str],
    assertions: list[dict],
) -> tuple[list[dict], list[str], list[str], list[dict]]:
    """清洗所有会流向下游的结构化文本字段。

    P3-3 只在「传了 llm 且 sanitize=True」时清洗喂给模型的上下文；**组装
    路径完全不过滤**。但组装出来的答案并不是终点 —— 它被返回给
    agent/规划器，而 agent 本身就是一个 LLM。也就是说被投毒的手册文本
    照样能一路走到某个模型的上下文里，只是绕了一圈。

    现在两条路径都清洗。清洗是幂等的，对正常文本零影响。
    """
    clean_steps = []
    for st in steps:
        s2 = dict(st)
        for key in ("instruction", "action", "target"):
            if key in s2:
                s2[key] = _sanitize_text(s2[key])
        clean_steps.append(s2)
    clean_warnings = [_sanitize_text(w) for w in warnings]
    clean_pre = [_sanitize_text(p) for p in preconditions]
    clean_assertions = []
    for a in assertions:
        a2 = dict(a)
        a2["text"] = _sanitize_text(a2.get("text", ""))
        clean_assertions.append(a2)
    return clean_steps, clean_warnings, clean_pre, clean_assertions


def _collect_injection_flags(ctx: dict) -> list[dict]:
    """扫描命中记录，记录疑似注入信号供审计 (P4)。"""
    from ..security import detect_prompt_injection

    flags: list[dict] = []
    seen: set[str] = set()
    for key in ("actions", "procedures", "sections"):
        for item in ctx.get(key) or []:
            if not isinstance(item, dict):
                continue
            ident = str(item.get("id") or item.get("action_id") or "")
            if ident in seen:
                continue
            blob = " ".join(
                str(item.get(f) or "")
                for f in ("title", "snippet", "text", "content")
            )
            for st in item.get("procedure") or []:
                if isinstance(st, dict):
                    blob += " " + str(st.get("source_instruction") or "")
            verdict = detect_prompt_injection(blob)
            if verdict.flagged:
                seen.add(ident)
                flags.append({"id": ident, "reasons": verdict.reasons})
    return flags


# ---------- 对外主入口 ----------


def synthesize_answer(
    user_request: str,
    context: Any,
    llm: Optional[Callable[[str, str], str]] = None,
    min_confidence: Optional[float] = None,
    sanitize: bool = True,
) -> SynthesizedAnswer:
    """输入检索上下文，输出结构化答案（P2-1）。

    Args:
        user_request: 用户问题 / 请求原文。
        context: 检索上下文，兼容 planner_retrieve 的 dict 或检索结果的
            list[dict]（见模块 docstring）。
        llm: 可选的可调用对象 `(user_request, context_text) -> str`，
            用于生成自然语言 answer。为 None 时回落为组装模式（保证
            无 LLM 时主链路可用）。调用异常会被捕获并回落到组装模式。
        min_confidence: 证据不足拒答的置信度阈值（P2-3）。为 None 时使用
            模块默认 CONFIDENCE_REFUSE_THRESHOLD。低于该阈值或证据不足
            时明确拒答（answer_status='refused'），不编造。
        sanitize: (P3-3) 为 True（默认）时，喂给 LLM 的检索上下文先做
            注入清洗并包裹在「不可信数据围栏」中（system prompt 隔离），
            降低恶意文档 prompt injection 风险。组装模式（无 llm）不受影响。

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

    # P4 (严重问题 8): 无论走 LLM 还是组装路径，都清洗结构化文本字段。
    # 组装出的答案会被 agent/规划器消费，而 agent 本身就是 LLM——不清洗
    # 等于给注入内容留了一条绕行路径。
    if sanitize:
        unique_steps, uniq_warnings, uniq_pre, uniq_assertions = _sanitize_structured(
            unique_steps, uniq_warnings, uniq_pre, uniq_assertions
        )
    injection_flags = _collect_injection_flags(ctx) if sanitize else []

    citations = _extract_citations(ctx)
    confidence = _compute_confidence(ctx)
    extraction_confidence = round(_extraction_confidence(ctx) or 0.0, 3)
    requires_confirmation = _compute_requires_confirmation(ctx)

    # P2-3: 证据充分性 + 冲突检测
    threshold = CONFIDENCE_REFUSE_THRESHOLD if min_confidence is None else float(min_confidence)
    refusal_reason = _evidence_refusal_reason(ctx, confidence, threshold) if threshold > 0 else None
    conflicts = _detect_conflicts(ctx) if refusal_reason is None else []

    if conflicts:
        answer_status = ANSWER_CONFLICT
        # 冲突需要用户澄清，视为需要确认
        requires_confirmation = True
    elif refusal_reason is not None:
        answer_status = ANSWER_REFUSED
    else:
        answer_status = ANSWER_ANSWERED

    generated_by = ASSEMBLE_MODE
    answer = ""

    if llm is not None:
        if sanitize:
            # P3-3: 检索上下文先行注入清洗 + system prompt 隔离，
            # 让文档内容只能待在「不可信数据围栏」里，无法覆盖指令。
            from ..security import render_isolated_context

            _items = list(primary) + [s for s in steps if isinstance(s, dict)]
            context_text = render_isolated_context(_items, user_request)
        else:
            context_text = _render_context_text(
                user_request, primary, steps, uniq_warnings, uniq_pre
            )
        try:
            generated = llm(user_request, context_text)
            if generated and generated.strip():
                answer = generated.strip()
                generated_by = LLM_MODE
        except Exception:  # LLM 失败不阻塞主链路，回落到组装
            answer = ""

    if not answer:
        answer = _assemble_answer(
            user_request=user_request,
            context=ctx,
            steps=unique_steps,
            warnings=uniq_warnings,
            preconditions=uniq_pre,
            conflicts=conflicts,
            answer_status=answer_status,
            refusal_reason=refusal_reason,
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
        extraction_confidence=extraction_confidence,
        requires_confirmation=requires_confirmation,
        generated_by=generated_by,
        answer_status=answer_status,
        refusal_reason=refusal_reason,
        conflicts=conflicts,
        injection_flags=injection_flags,
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
    方便终端呈现与溯源性展示。拒答与冲突状态也会附加标注（P2-3）。
    """
    out = [answer.answer]

    # P2-3: 状态标注（拒答 / 冲突）
    if answer.answer_status == ANSWER_REFUSED:
        out.append("")
        out.append("[状态：证据不足，已拒答（不编造）]")
    elif answer.answer_status == ANSWER_CONFLICT:
        out.append("")
        out.append("[状态：文档冲突，需澄清后再执行]")
        for i, c in enumerate(answer.conflicts, 1):
            refs = [r for r in c.get("sources", []) if _render_source_ref(r)]
            suffix = f"  [来源: {'、'.join(_render_source_ref(r) for r in refs)}]" if refs else ""
            out.append(f"  冲突{i}：{c.get('description', '')}{suffix}")

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
