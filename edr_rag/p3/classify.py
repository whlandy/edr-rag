"""执行失败分类 (P4 Phase H)。

P3 的 ``EdrRagAgent.classify_failure`` 是 4 条 if-elif 字符串匹配：

    if "not found" in msg and "button" in msg: ...
    elif "timeout" in msg: ...
    elif "network" in msg or "connection" in msg: ...
    elif "ui" in msg or "element" in msg: ...

``FAILURE_TYPES`` 里定义了 11 种失败类型，这 4 条覆盖其中 4 种，剩下 7 种
永远分不出来；而且 ``"ui" in msg`` 会把任何含 "build"、"guide"、"require"
的消息误判成 UI 错误 —— 子串匹配没有词边界。

这里改成一张**有优先级的规则表**：正则带词边界、中英文都覆盖、命中时返回
匹配依据便于人工复核。规则表是数据，加新规则不用改逻辑。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from .feedback import FAILURE_TYPES, FeedbackItem


@dataclass(frozen=True)
class ClassificationRule:
    """一条分类规则。

    Attributes:
        failure_type: 命中后判定的失败类型（必须在 FAILURE_TYPES 内）。
        pattern: 匹配正则（大小写不敏感，已带词边界）。
        priority: 数值越大越优先。用于消歧 —— 比如「按钮找不到」比泛化的
            「元素找不到」更具体，应当优先。
        rationale: 人类可读的命中理由，写进分类结果供复核。
    """

    failure_type: str
    pattern: re.Pattern
    priority: int
    rationale: str


def _rule(failure_type: str, regex: str, priority: int, rationale: str) -> ClassificationRule:
    return ClassificationRule(
        failure_type=failure_type,
        pattern=re.compile(regex, re.IGNORECASE),
        priority=priority,
        rationale=rationale,
    )


#: 规则表。优先级高的先判 —— 更具体的失败类型应当压过泛化的。
RULES: tuple[ClassificationRule, ...] = (
    # --- 高优先级：具体到控件/文案 ---
    _rule("button_text_not_found",
          r"\b(button|btn)\b.{0,40}\b(not\s+found|missing|no\s+such)\b"
          r"|\b(not\s+found|missing)\b.{0,40}\b(button|btn)\b"
          r"|按钮.{0,10}(未找到|找不到|不存在)",
          100, "错误消息同时提到按钮与「找不到」"),
    _rule("missing_ui_label",
          r"\b(label|element|control|widget|selector)\b.{0,40}"
          r"\b(not\s+found|missing|no\s+such|could\s+not\s+locate)\b"
          r"|\b(ElementNotFound|NoSuchElement|LookupError)\b"
          r"|(控件|元素|标签).{0,10}(未找到|找不到|不存在)",
          90, "错误消息提到 UI 元素定位失败"),
    _rule("timeout",
          r"\btimed?\s*out\b|\bTimeoutError\b|\btimeout\b|超时",
          85, "错误消息提到超时"),
    _rule("network_error",
          r"\b(connection\s+(refused|reset|aborted)|network\s+(is\s+)?unreachable"
          r"|ConnectionError|DNS|ECONNREFUSED|socket\s+error)\b"
          r"|(网络|连接).{0,8}(失败|错误|中断|不可达)",
          80, "错误消息提到网络/连接故障"),
    _rule("unexpected_state",
          r"\b(unexpected\s+(state|dialog|window)|invalid\s+state"
          r"|state\s+mismatch|wrong\s+(page|screen))\b"
          r"|(意外|非预期).{0,6}(状态|窗口|对话框)|状态不一致",
          70, "错误消息提到状态不符合预期"),
    _rule("ui_error",
          r"\b(pywinauto|playwright|selenium|uiautomation|COMError)\b"
          r"|\bUI\s*(error|exception|failure)\b",
          60, "错误来自 GUI 自动化框架"),

    # --- 中优先级：由 expected/observed 语义判定 ---
    _rule("failed_success_criteria",
          r"\b(success\s+criteri\w*|assertion|verification)\b.{0,30}\b(fail|not\s+met)\b"
          r"|(成功(准则|标准)|校验|断言).{0,8}(失败|未满足|不通过)",
          55, "操作执行了但成功准则未满足"),
    _rule("user_correction",
          r"\b(user\s+(corrected|overrode|modified)|manual\s+correction)\b"
          r"|(人工|用户).{0,6}(修正|更正|改正|干预)",
          50, "用户手工修正了步骤"),

    # --- 低优先级：检索侧 ---
    _rule("action_not_found",
          r"\bno\s+(matching\s+)?action\b|\baction\s+not\s+found\b"
          r"|(未找到|没有).{0,8}(匹配的)?动作",
          45, "动作目录里没有匹配项"),
    _rule("procedure_not_found",
          r"\bno\s+(matching\s+)?procedure\b|\bprocedure\s+not\s+found\b"
          r"|(未找到|没有).{0,8}(匹配的)?流程",
          44, "流程库里没有匹配项"),
    _rule("low_confidence_retrieval",
          r"\blow\s+confidence\b|\bconfidence\s+(below|too\s+low)\b"
          r"|置信度.{0,6}(过低|不足)|证据不足",
          40, "检索置信度过低"),
)


@dataclass
class Classification:
    """一次分类的结果。"""

    failure_type: str
    matched: bool
    rationale: str = ""
    rule_priority: int = 0
    evidence_used: str = ""

    def to_dict(self) -> dict:
        return {
            "failure_type": self.failure_type,
            "matched": self.matched,
            "rationale": self.rationale,
            "rule_priority": self.rule_priority,
        }


def classification_text(item: FeedbackItem) -> str:
    """把用于分类的所有文本拼起来。

    P3 只看 ``evidence.error_message``。但很多失败根本没有异常消息 —— 操作
    "成功"了而结果不对，线索全在 expected / observed 里。这里三者都看。
    """
    parts = [item.expected or "", item.observed or ""]
    if item.evidence:
        parts.extend([
            item.evidence.error_message or "",
            item.evidence.dump_tree or "",
        ])
    return " \n".join(p for p in parts if p)


def classify(item: FeedbackItem) -> Classification:
    """按规则表分类；无规则命中时保留原有 failure_type。

    Returns:
        Classification。``matched=False`` 表示没有规则命中 —— 这时保留
        调用方原本声明的 failure_type（可能是 edr-wd 那边已经判好的），
        而不是硬塞一个错的。
    """
    text = classification_text(item)
    if not text.strip():
        return Classification(
            failure_type=item.failure_type or "",
            matched=False,
            rationale="no evidence text to classify",
        )

    best: Optional[ClassificationRule] = None
    for rule in RULES:
        if rule.pattern.search(text):
            if best is None or rule.priority > best.priority:
                best = rule

    if best is None:
        return Classification(
            failure_type=item.failure_type or "",
            matched=False,
            rationale="no rule matched",
            evidence_used=text[:200],
        )

    return Classification(
        failure_type=best.failure_type,
        matched=True,
        rationale=best.rationale,
        rule_priority=best.priority,
        evidence_used=text[:200],
    )


def validate_rules() -> list[str]:
    """自检规则表：failure_type 必须都在 FAILURE_TYPES 白名单里。

    Returns:
        问题描述列表（空 = 通过）。由测试调用，防止加规则时打错类型名。
    """
    problems = []
    for rule in RULES:
        if rule.failure_type not in FAILURE_TYPES:
            problems.append(
                f"rule for '{rule.failure_type}' is not in FAILURE_TYPES"
            )
    return problems
