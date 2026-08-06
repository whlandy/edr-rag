"""P2 risk classification and confirmation policy (design §6).

Risk levels:
  low:
    read-only operations, navigation, viewing status, viewing logs
  medium:
    configuration changes, export, import, restart component
  high:
    delete, reset, disable protection, wipe data, irreversible state change

Confirmation rules:
  requires_confirmation = true if risk is medium or high
  requires_confirmation = true if action changes security state
  requires_confirmation = true if rollback is unavailable
  requires_confirmation = true if source confidence is low

High-risk actions must never become `published` without review.
"""

from __future__ import annotations

import re
from typing import Optional


# Keywords that indicate security-related actions
_SECURITY_KEYWORDS = {
    "防火墙", "防病毒", "防护", "保护", "安全", "密码", "口令",
    "认证", "授权", "加密", "证书", "审计", "日志",
    "firewall", "antivirus", "protection", "security", "password",
    "authentication", "authorization", "encryption", "certificate", "audit",
}

# Keywords that indicate an irreversible action (high risk)
_IRREVERSIBLE_KEYWORDS = {
    "删除", "重置", "清除", "卸载", "关闭", "禁用",
    "delete", "remove", "reset", "clear", "uninstall", "disable",
    "关机", "重启", "shutdown", "reboot", "restart",
    "格式化", "format", "wipe",
}

# Keywords that indicate configuration changes (medium risk)
_CONFIG_KEYWORDS = {
    "配置", "设置", "修改", "更改", "变更", "导入", "导出",
    "configure", "config", "set", "modify", "change", "import", "export",
    "添加", "注册", "add", "register", "create",
    "更新", "升级", "update", "upgrade",
}

# Keywords that indicate read-only / low risk
_READONLY_KEYWORDS = {
    "查看", "浏览", "查询", "搜索", "显示", "检查", "测试",
    "view", "browse", "query", "search", "show", "display", "check", "test",
    "status", "状态", "日志", "log", "信息", "信息",
}


def classify_risk(
    title: str,
    section_path: list[str],
    operation: str,
    action_hints: list[str],
) -> str:
    """Classify an action's risk level based on semantic signals.

    Returns one of 'low', 'medium', 'high'.

    Heuristics (conservative — when in doubt, return higher risk):
      1. If title/path/operation contains irreversible keywords -> high
      2. If action_hints include destructive operations -> high
      3. If title/path/operation contains security keywords -> medium (at least)
      4. If operation contains config keywords -> medium
      5. Default -> low
    """
    combined = f"{' '.join(section_path)} {title} {operation}".lower()

    # Check irreversible keywords first (highest priority)
    for kw in _IRREVERSIBLE_KEYWORDS:
        if kw.lower() in combined:
            return "high"

    # Check destructive action hints
    destructive_hints = {"delete", "close", "stop", "disable", "restart"}
    for hint in action_hints:
        if hint in destructive_hints:
            return "high"

    # Check security keywords
    for kw in _SECURITY_KEYWORDS:
        if kw.lower() in combined:
            return "medium"

    # Check config keywords
    for kw in _CONFIG_KEYWORDS:
        if kw.lower() in combined:
            return "medium"

    return "low"


def requires_confirmation(
    risk: str,
    has_recovery: bool,
    confidence: float,
    changes_security_state: bool = False,
) -> bool:
    """Determine if an action requires user confirmation.

    Rules (design §6):
      - requires_confirmation = true if risk is medium or high
      - requires_confirmation = true if action changes security state
      - requires_confirmation = true if rollback is unavailable
      - requires_confirmation = true if source confidence is low
    """
    if risk in ("medium", "high"):
        return True
    if changes_security_state:
        return True
    if not has_recovery:
        return True
    if confidence < 0.5:
        return True
    return False


def determine_initial_status(risk: str, confidence: float) -> str:
    """Determine the initial status for an action candidate.

    Rules:
      - high risk -> 'needs_review' (never auto-published)
      - medium risk with low confidence -> 'needs_review'
      - otherwise -> candidate
    """
    if risk == "high":
        return "needs_review"
    if risk == "medium" and confidence < 0.7:
        return "needs_review"
    return "candidate"


def extract_success_criteria(action_hints: list[str], source_instructions: list[str]) -> list[str]:
    """Generate heuristic success criteria from action hints."""
    criteria = []
    for hint, instruction in zip(action_hints, source_instructions):
        if hint == "click":
            # For click actions, success is seeing the target
            criteria.append(f"{instruction} 完成")
        elif hint == "input_text":
            criteria.append("输入生效")
        elif hint == "open_window":
            criteria.append(f"{instruction} 窗口已打开")
        elif hint == "check":
            criteria.append(f"{instruction} 已勾选")
        elif hint == "save":
            criteria.append("保存成功")
        elif hint == "export":
            criteria.append("导出成功")
        elif hint == "import":
            criteria.append("导入成功")
        elif hint == "enable":
            criteria.append("启用成功")
        elif hint == "disable":
            criteria.append("禁用成功")
    if not criteria:
        criteria.append("操作完成")
    return criteria
