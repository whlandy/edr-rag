"""P3 schema: FeedbackItem and related dataclasses.

P3 scope (design §7):
  - Execution feedback capture
  - Failure classification
  - Feedback item generation
  - Selector synonym updates
  - Action patch proposals
  - Re-index trigger

Feedback lifecycle:
  captured -> needs_review -> patch_proposed -> patched -> reindexed
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# ---------- Feedback item schema ----------


@dataclass
class FeedbackEvidence:
    """Evidence collected during execution failure."""

    screenshot: Optional[str] = None
    dump_tree: Optional[str] = None
    error_message: Optional[str] = None

    def to_jsonl(self) -> dict:
        return asdict(self)


@dataclass
class SuggestedPatch:
    """Suggested changes to the action catalog or procedure extraction.

    P3 patch types:
      - selector_synonyms: update UI selector keywords
      - action_risk: update risk classification
      - action_status: update status (e.g., needs_review -> deprecated)
      - procedure_step: update step instruction or target
    """

    patch_type: str = ""
    field: str = ""
    old_value: str = ""
    new_value: str = ""

    def to_jsonl(self) -> dict:
        return asdict(self)


@dataclass
class FeedbackItem:
    """Structured feedback from an execution run."""

    feedback_id: str
    manual_id: str
    action_id: Optional[str] = None
    procedure_id: Optional[str] = None
    failure_type: str = ""
    expected: str = ""
    observed: str = ""
    evidence: Optional[FeedbackEvidence] = None
    suggested_patch: Optional[SuggestedPatch] = None
    status: str = "captured"
    created_at: str = ""
    closed_at: Optional[str] = None
    #: P4: 审核备注（驳回原因等），供人工复核留痕。
    review_note: str = ""

    def to_jsonl(self) -> dict:
        d = asdict(self)
        if self.evidence:
            d["evidence"] = self.evidence.to_jsonl()
        if self.suggested_patch:
            d["suggested_patch"] = self.suggested_patch.to_jsonl()
        return d


# ---------- Failure classification ----------

# Known failure types (design §7)
FAILURE_TYPES = {
    "button_text_not_found",  # expected button text vs observed
    "missing_ui_label",       # UI element not found
    "low_confidence_retrieval",  # RAG returned low-scored results
    "user_correction",        # user manually corrected a step
    "failed_success_criteria", # action succeeded but success check failed
    "timeout",                # action exceeded time limit
    "unexpected_state",       # application in unexpected state
    "action_not_found",       # no matching action in catalog
    "procedure_not_found",    # no matching procedure found
    "network_error",          # network connectivity issue
    "ui_error",               # pywinauto/Playwright error
}

# Status lifecycle
#: P4: 增补 patch_approved / patch_rejected —— 补丁必须经人工审核才能应用，
#: 审核状态机需要一条明确的「驳回」出边，否则被否掉的提议会一直挂在
#: patch_proposed 上，跟「还没审」分不开。
FEEDBACK_STATUSES = [
    "captured",
    "needs_review",
    "patch_proposed",
    "patch_approved",
    "patch_rejected",
    "patched",
    "reindexed",
    "closed",
]


# ---------- Feedback store ----------

class FeedbackStore:
    """Simple JSONL-based storage for feedback items."""

    def __init__(self, feedback_path: str):
        self.path = feedback_path

    def append(self, item: FeedbackItem) -> None:
        """Append a single feedback item to the JSONL log."""
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(item.to_jsonl(), ensure_ascii=False) + "\n")

    def read_all(self) -> list[FeedbackItem]:
        """Read all feedback items from the JSONL log."""
        import os
        if not os.path.exists(self.path):
            return []
        items = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    item = FeedbackItem(**data)
                    if data.get("evidence"):
                        item.evidence = FeedbackEvidence(**data["evidence"])
                    if data.get("suggested_patch"):
                        item.suggested_patch = SuggestedPatch(**data["suggested_patch"])
                    items.append(item)
        return items

    def update_status(self, feedback_id: str, status: str) -> bool:
        """Update the status of a feedback item in place.

        Returns True if found and updated.
        """
        return self.update_item(feedback_id, status=status)

    def update_item(self, feedback_id: str, **changes) -> bool:
        """就地更新一条反馈的任意字段，**单次原子重写** (P4)。

        P3 的 ``classify_failure`` 要改两个字段，于是先调 update_status 重写
        一遍文件、再自己读回来重写第二遍 —— 两次非原子写，中间崩溃会留下
        状态改了但 failure_type 没改的半截记录。这里一次写完。

        Args:
            **changes: 要更新的字段（status / failure_type / suggested_patch /
                review_note 等）。未知字段会被忽略。

        Returns:
            True 表示找到并更新；False 表示没有该 id。
        """
        items = self.read_all()
        found = False
        for item in items:
            if item.feedback_id != feedback_id:
                continue
            for key, value in changes.items():
                if hasattr(item, key):
                    setattr(item, key, value)
            status = changes.get("status")
            if status in ("patched", "closed"):
                item.closed_at = datetime.now(timezone.utc).isoformat()
            found = True
            break
        if found:
            from ..atomic_io import atomic_write_text

            payload = "".join(
                json.dumps(i.to_jsonl(), ensure_ascii=False) + "\n" for i in items
            )
            atomic_write_text(Path(self.path), payload)
        return found
