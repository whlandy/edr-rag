"""P2 action candidate extraction — procedure → ActionCatalogEntry.

Converts P1 procedures (from procedures.jsonl) into reviewable
action catalog entries with risk classification and confirmation policy.

Key design (design §6):
  - Each procedure produces exactly one action candidate.
  - Action steps are mapped from procedure steps using action_hint.
  - Risk is auto-classified from title/path/operation.
  - High-risk actions are NEVER auto-published (status=needs_review).
  - High-risk actions that don't have recovery are additionally flagged.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from . import ActionCatalogEntry, ActionSource, ActionStep, ValidationMetadata
from .risk import (
    classify_risk,
    determine_initial_status,
    extract_success_criteria,
    requires_confirmation,
)

logger = logging.getLogger(__name__)


def extract_action_from_procedure(
    proc: dict,
    *,
    product: str | None = None,
) -> Optional[ActionCatalogEntry]:
    """Convert one P1 procedure dict into an ActionCatalogEntry.

    Args:
        proc: A procedure dict from procedures.jsonl.
        product: Optional product name override.

    Returns:
        ActionCatalogEntry or None if the procedure is malformed.
    """
    action_id = proc.get("id", "")
    if not action_id:
        return None

    # Strip "procedure." prefix to get a clean action_id
    if action_id.startswith("procedure."):
        action_id = action_id[len("procedure."):]

    manual_id = proc.get("manual_id", "")
    product = product or proc.get("product", "")
    title = proc.get("title", "")
    section_path = proc.get("section_path", [])
    module = section_path[0] if section_path else ""
    confidence = proc.get("confidence", 0.0)
    preconditions = proc.get("preconditions", [])
    recovery = proc.get("recovery", [])
    source = proc.get("source", {})

    # Extract steps
    raw_steps = proc.get("steps", [])
    if not raw_steps:
        logger.info("Procedure %s has no steps — skipping action extraction", action_id)
        return None

    action_hints = []
    source_instructions = []
    action_steps: list[ActionStep] = []
    for s in raw_steps:
        action_hint = s.get("action_hint", "")
        action_hints.append(action_hint)
        source_inst = s.get("source_instruction") or s.get("instruction", "")
        source_instructions.append(source_inst)

        # Map action_hint to action name, fallback to instruction-based
        action_name = _map_action(action_hint, source_inst)
        target = s.get("target", "")

        action_steps.append(ActionStep(
            order=s.get("order", 0),
            action=action_name,
            target=target,
            source_instruction=source_inst,
            value_source="user" if action_name == "input_text" else "",
        ))

    # Classify risk
    risk = classify_risk(title, section_path, action_id, action_hints)

    # Security check
    security_related = _check_security(title, section_path, action_id)

    # Confirmation
    confirm = requires_confirmation(risk, bool(recovery), confidence, security_related)

    # Initial status
    status = determine_initial_status(risk, confidence)

    # Success criteria
    success_criteria = extract_success_criteria(action_hints, source_instructions)

    # Validation metadata
    has_steps = len(action_steps) > 0
    has_risk = bool(risk)
    has_recovery = bool(recovery)
    source_verified = bool(source.get("file"))

    entry = ActionCatalogEntry(
        action_id=action_id,
        product=product,
        module=module,
        title=title,
        status=status,
        confidence=confidence,
        risk=risk,
        requires_confirmation=confirm,
        preconditions=preconditions,
        procedure=action_steps,
        success_criteria=success_criteria,
        recovery=recovery,
        source=ActionSource(
            manual_id=manual_id,
            procedure_id=proc.get("id", ""),
            file=source.get("file", ""),
        ),
        validation=ValidationMetadata(
            has_steps=has_steps,
            has_risk=has_risk,
            has_recovery=has_recovery,
            source_verified=source_verified,
        ),
    )

    return entry


def _map_action(hint: str, instruction: str) -> str:
    """Map a procedure action_hint to a canonical action name.

    Falls back to heuristic extraction from instruction text.
    """
    if hint:
        return hint
    # Heuristic fallback
    for verb, action in [
        ("点击", "click"),
        ("选择", "select"),
        ("输入", "input_text"),
        ("勾选", "check"),
        ("打开", "open_window"),
        ("关闭", "close"),
        ("启动", "start"),
        ("停止", "stop"),
        ("重启", "restart"),
        ("启用", "enable"),
        ("禁用", "disable"),
        ("添加", "add"),
        ("删除", "delete"),
        ("确认", "confirm"),
        ("保存", "save"),
        ("导出", "export"),
        ("导入", "import"),
        ("取消", "cancel"),
        ("进入", "navigate"),
    ]:
        if verb in instruction:
            return action
    return ""


def _check_security(title: str, section_path: list[str], operation: str) -> bool:
    """Check if an action is security-related."""
    security_kw = {
        "防火墙", "防病毒", "防护", "保护", "安全", "密码", "口令",
        "认证", "授权", "加密", "证书", "审计",
        "firewall", "antivirus", "protection", "security", "password",
        "authentication", "authorization", "encryption", "certificate",
    }
    combined = f"{' '.join(section_path)} {title} {operation}".lower()
    for kw in security_kw:
        if kw.lower() in combined:
            return True
    return False


def build_actions_from_file(
    procedures_path: Path,
    *,
    product: str | None = None,
) -> list[ActionCatalogEntry]:
    """Build action candidates from procedures.jsonl.

    Returns a list of ActionCatalogEntry objects (not yet written).
    """
    entries: list[ActionCatalogEntry] = []

    with Path(procedures_path).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            proc = json.loads(line)
            entry = extract_action_from_procedure(proc, product=product)
            if entry:
                entries.append(entry)

    return entries


def write_p2_artifacts(
    workdir: Path,
    entries: list[ActionCatalogEntry],
) -> tuple[Path, Path]:
    """Write actions.jsonl and actions.yaml under workdir.

    Also writes p2_metadata.json.

    Returns (jsonl_path, yaml_path).
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    # JSONL
    jsonl_path = workdir / "actions.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e.to_jsonl(), ensure_ascii=False) + "\n")

    # YAML (manual YAML lines — no external dep)
    yaml_path = workdir / "actions.yaml"
    with yaml_path.open("w", encoding="utf-8") as f:
        for e in entries:
            for line in e.to_yaml_lines():
                f.write(line + "\n")
            f.write("---\n")

    # Metadata
    from datetime import datetime, timezone
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "action_count": len(entries),
        "by_status": {},
        "by_risk": {},
    }
    for e in entries:
        meta["by_status"][e.status] = meta["by_status"].get(e.status, 0) + 1
        meta["by_risk"][e.risk] = meta["by_risk"].get(e.risk, 0) + 1
    meta_path = workdir / "p2_metadata.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return jsonl_path, yaml_path
