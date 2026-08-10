"""动作目录补丁的生成、审核与应用 (P4 Phase H)。

P3 的 ``propose_patch`` 只是把**调用方自己传进来的** patch 原样返回并把状态
改成 patch_proposed：

    if not target or not target.suggested_patch: return None
    self.feedback_store.update_status(feedback_id, "patch_proposed")
    return target.suggested_patch.to_jsonl()

也就是说它既不生成补丁，也不应用补丁 —— 反馈回路在这里断了。捕获的失败
永远变不成对动作目录的改进。

本模块补上两段：
  1. **生成**：由 failure_type + evidence 推导出具体补丁（``propose``）。
  2. **应用**：真正改写 actions.jsonl（``apply``），原子写入，且在改动前
     捕获一个索引版本快照，使 ``edr-rag index rollback`` 能回退。

安全约束（重要）：
  动作目录直接驱动客户终端上的 GUI 自动化。一条错误的补丁 = 在客户机器上
  点错按钮。所以**绝不自动应用** —— 必须经过显式 approve，且高风险动作的
  补丁需要二次确认。这与 P2-5 对高风险 Action 的确认门控是同一套原则。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ..atomic_io import write_jsonl
from .feedback import FeedbackItem, SuggestedPatch

logger = logging.getLogger(__name__)

#: 审核状态机。captured -> classified -> proposed -> approved -> applied
#: 任一步都可以 rejected。
PATCH_PROPOSED = "patch_proposed"
PATCH_APPROVED = "patch_approved"
PATCH_APPLIED = "patched"
PATCH_REJECTED = "patch_rejected"

#: 允许被补丁修改的动作字段白名单。
#:
#: 不在白名单里的字段一律拒绝 —— 反馈来自被自动化驱动的执行环境，如果它能
#: 改任意字段，就等于给了执行侧改写自己权限约束（risk / tenant / device）的
#: 能力。risk 只能**升**不能降，见 _validate_patch。
PATCHABLE_FIELDS = {
    "selector_synonyms",
    "risk",
    "status",
    "title",
    "preconditions",
    "success_criteria",
    "recovery",
}

#: 风险等级序数 —— 补丁只允许提升风险，不允许降低。
_RISK_ORDER = {"low": 0, "medium": 1, "high": 2}


@dataclass
class PatchResult:
    """一次补丁操作的结果。"""

    ok: bool
    action_id: str = ""
    patch_type: str = ""
    field_name: str = ""
    old_value: Any = None
    new_value: Any = None
    message: str = ""
    requires_confirmation: bool = False
    version_id: Optional[int] = None      # 应用前捕获的可回滚版本
    applied_at: str = ""

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "action_id": self.action_id,
            "patch_type": self.patch_type,
            "field": self.field_name,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "message": self.message,
            "requires_confirmation": self.requires_confirmation,
            "version_id": self.version_id,
            "applied_at": self.applied_at,
        }


# ---------- 生成 ----------


def propose(item: FeedbackItem) -> Optional[SuggestedPatch]:
    """由反馈推导出一个具体补丁；无法推导时返回 None。

    调用方已显式给出 suggested_patch 时以它为准（人比启发式规则可靠）；
    否则按 failure_type 生成。
    """
    if item.suggested_patch and item.suggested_patch.patch_type:
        return item.suggested_patch

    ftype = (item.failure_type or "").strip()
    observed = (item.observed or "").strip()
    expected = (item.expected or "").strip()

    # 按钮文案对不上 —— 手册写的是 expected，界面上实际是 observed。
    # 把实际文案加进选择器同义词，下次就能点中。
    if ftype == "button_text_not_found" and observed:
        return SuggestedPatch(
            patch_type="selector_synonyms",
            field="selector_synonyms",
            old_value=expected,
            new_value=observed,
        )

    # UI 元素定位不到：同上，但把握更低，标成 needs_review 让人来看。
    if ftype == "missing_ui_label":
        return SuggestedPatch(
            patch_type="action_status",
            field="status",
            old_value="",
            new_value="needs_review",
        )

    # 成功准则没满足：动作本身可能已经过时，不该继续被自动执行。
    if ftype == "failed_success_criteria":
        return SuggestedPatch(
            patch_type="action_status",
            field="status",
            old_value="",
            new_value="needs_review",
        )

    # 意外状态 —— 说明这个动作的副作用比手册描述的大，提升风险等级。
    if ftype == "unexpected_state":
        return SuggestedPatch(
            patch_type="action_risk",
            field="risk",
            old_value="",
            new_value="high",
        )

    # 用户手工修正了步骤：把修正后的说法记为标题候选，交人工确认。
    if ftype == "user_correction" and observed:
        return SuggestedPatch(
            patch_type="procedure_step",
            field="title",
            old_value=expected,
            new_value=observed,
        )

    # 检索侧失败（找不到动作/流程、置信度低）不是目录的错，
    # 补丁改不了它 —— 那属于要补语料或补同义词，不在本模块职责内。
    return None


# ---------- 校验 ----------


def _validate_patch(patch: SuggestedPatch, action: dict) -> Optional[str]:
    """校验补丁是否可安全应用；返回拒绝原因，None 表示通过。"""
    field_name = (patch.field or "").strip()
    if field_name not in PATCHABLE_FIELDS:
        return (
            f"field '{field_name}' is not patchable "
            f"(allowed: {sorted(PATCHABLE_FIELDS)})"
        )
    if not str(patch.new_value or "").strip():
        return "patch has an empty new_value"

    # 风险只能升不能降。允许降级等于让执行侧自己把高危操作标成低危，
    # 从而绕过 P2-5 的确认门控。
    if field_name == "risk":
        old = _RISK_ORDER.get(str(action.get("risk") or "").lower())
        new = _RISK_ORDER.get(str(patch.new_value).lower())
        if new is None:
            return f"invalid risk level '{patch.new_value}'"
        if old is not None and new < old:
            return (
                f"refusing to lower risk from '{action.get('risk')}' to "
                f"'{patch.new_value}' — risk may only be escalated by feedback"
            )
    return None


def _apply_to_action(action: dict, patch: SuggestedPatch) -> tuple[Any, Any]:
    """把补丁写进动作 dict（就地修改），返回 (旧值, 新值)。"""
    field_name = patch.field
    old = action.get(field_name)

    if field_name in ("selector_synonyms", "preconditions", "success_criteria", "recovery"):
        # 列表类字段：追加而不是替换 —— 反馈是增量信息，不该抹掉手册原文。
        current = list(old or []) if isinstance(old, list) else ([old] if old else [])
        if patch.new_value not in current:
            current.append(patch.new_value)
        action[field_name] = current
        return old, current

    action[field_name] = patch.new_value
    return old, patch.new_value


# ---------- 应用 ----------


def apply(
    workdir: str | Path,
    item: FeedbackItem,
    patch: Optional[SuggestedPatch] = None,
    *,
    confirm: bool = False,
    capture_version: bool = True,
) -> PatchResult:
    """把补丁应用到 ``<workdir>/actions.jsonl``。

    Args:
        item: 反馈条目（提供 action_id）。
        patch: 要应用的补丁；None 时用 ``propose(item)`` 生成。
        confirm: 高风险动作的补丁需要显式 confirm=True 才放行。
        capture_version: 应用前捕获索引版本快照，使改动可回滚。

    Returns:
        PatchResult。``ok=False`` 时 message 说明原因；
        ``requires_confirmation=True`` 表示需要人工确认后重试。
    """
    workdir = Path(workdir)
    actions_path = workdir / "actions.jsonl"
    if not actions_path.is_file():
        return PatchResult(ok=False, message=f"actions.jsonl not found in {workdir}")

    patch = patch or propose(item)
    if patch is None:
        return PatchResult(ok=False, message="no patch could be derived from this feedback")

    action_id = (item.action_id or "").strip()
    if not action_id:
        return PatchResult(ok=False, patch_type=patch.patch_type,
                           message="feedback has no action_id to patch")

    records = _read_actions(actions_path)
    target = None
    for rec in records:
        if rec.get("action_id") == action_id:
            target = rec
            break
    if target is None:
        return PatchResult(ok=False, action_id=action_id,
                           patch_type=patch.patch_type,
                           message=f"action '{action_id}' not in catalog")

    problem = _validate_patch(patch, target)
    if problem:
        return PatchResult(ok=False, action_id=action_id,
                           patch_type=patch.patch_type,
                           field_name=patch.field, message=problem)

    # 高风险动作的目录改动需要人工确认 —— 它直接决定客户终端上点什么。
    if str(target.get("risk") or "").lower() == "high" and not confirm:
        return PatchResult(
            ok=False,
            action_id=action_id,
            patch_type=patch.patch_type,
            field_name=patch.field,
            old_value=target.get(patch.field),
            new_value=patch.new_value,
            requires_confirmation=True,
            message=(
                f"action '{action_id}' is high risk; patching its catalog entry "
                "requires explicit confirmation (confirm=True)"
            ),
        )

    version_id = None
    if capture_version:
        version_id = _capture_version(workdir, records)

    old, new = _apply_to_action(target, patch)
    write_jsonl(actions_path, records)

    logger.info(
        "applied patch to %s: %s %r -> %r (feedback=%s)",
        action_id, patch.field, old, new, item.feedback_id,
    )
    return PatchResult(
        ok=True,
        action_id=action_id,
        patch_type=patch.patch_type,
        field_name=patch.field,
        old_value=old,
        new_value=new,
        version_id=version_id,
        applied_at=datetime.now(timezone.utc).isoformat(),
        message="patch applied",
    )


def _read_actions(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("skipping unparseable line in %s", path)
    return out


def _capture_version(workdir: Path, records: list[dict]) -> Optional[int]:
    """在改动前捕获一个可回滚的索引版本快照。"""
    try:
        from ..index_versioning import capture_version

        version, _ = capture_version(
            workdir, corpus="actions", records=records,
            id_field="action_id", text_field="title",
        )
        return version.version_id
    except Exception:  # pragma: no cover - 版本化失败不该阻断补丁
        logger.warning("failed to capture index version before patching", exc_info=True)
        return None
