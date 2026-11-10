"""P5 batch 5: ReviewDecision Repository 持久化 (design.md §4 / §9).

reviews.jsonl：人工审核记录的本地文件存储。关键契约：

- 幂等：同 (reviewer, decision_id, workflow_hash, pack_hash) 且内容完全一致
  → duplicate，不产生重复行（重复提交同一判定无害）；
- 不可静默覆盖：同 identity 但内容不同（含 decided_at / rationale / decision
  任一字段漂移）→ 拒绝并报 conflict，改判定必须用新 decision_id —— 已批准
  记录不会被后续写入悄悄替换；
- 损坏可检出：任何已存储行 from_dict 复析失败即抛 StoreCorruption，
  不静默跳过（§4 索引失败不损坏原件；审核记录是发布门禁的依据，
  坏行等于门禁失明）；
- 门禁查询 effective_approval：只有 decision=approve 且 (workflow_hash,
  pack_hash) 与当前对象精确匹配才有效 —— 对象修改后 hash 漂移，旧批准
  自动不命中（修改后作废，与 verify_review_decision 同口径）；无记录 /
  只有拒绝记录 → approved=False，unknown 不算通过。

写入走 atomic_write_text（临时文件 + rename），追加不改写既有行。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from edr_rag.atomic_io import atomic_write_text
from edr_rag.compiler.review_decision import ReviewDecision
from edr_rag.knowledge.models import ModelError

REVIEWS_FILE = "reviews.jsonl"


class StoreCorruption(ModelError):
    """已存储记录无法复析（存储损坏，不静默忽略）。"""


class ReviewStoreResult:
    def __init__(self, status: str, decision_id: str, path: Path | None = None):
        self.status = status  # created | duplicate
        self.decision_id = decision_id
        self.path = path

    def to_dict(self) -> dict[str, Any]:
        d = {"status": self.status, "decision_id": self.decision_id}
        if self.path is not None:
            d["path"] = str(self.path)
        return d


def review_identity(record: ReviewDecision) -> tuple[str, str, str, str]:
    return (record.reviewer, record.decision_id, record.workflow_hash, record.pack_hash)


def _read_records(path: Path) -> list[ReviewDecision]:
    if not path.exists():
        return []
    out: list[ReviewDecision] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as e:
            raise StoreCorruption(f"{path.name}: unparseable line: {e}") from e
        if not isinstance(raw, dict):
            raise StoreCorruption(f"{path.name}: record is not an object")
        try:
            out.append(ReviewDecision.from_dict(raw))
        except ModelError as e:
            raise StoreCorruption(f"{path.name}: record fails model reparse: {e}") from e
    return out


def save_review_decision(path: Path, record: ReviewDecision) -> ReviewStoreResult:
    existing = _read_records(path)
    ident = review_identity(record)
    payload = record.to_dict()
    for prior in existing:
        if review_identity(prior) == ident:
            if prior.to_dict() == payload:
                return ReviewStoreResult("duplicate", record.decision_id, path)
            raise ModelError(
                f"review conflict: {record.decision_id!r} by {record.reviewer!r} already "
                f"recorded with different content; use a new decision_id (判定记录不可被静默覆盖)"
            )
    text = "".join(json.dumps(r.to_dict(), ensure_ascii=False, sort_keys=True) + "\n" for r in existing)
    text += json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
    atomic_write_text(path, text)
    return ReviewStoreResult("created", record.decision_id, path)


def list_review_decisions(
    path: Path,
    workflow_hash: str | None = None,
    pack_hash: str | None = None,
) -> list[ReviewDecision]:
    found = _read_records(path)
    if workflow_hash is not None:
        found = [r for r in found if r.workflow_hash == workflow_hash]
    if pack_hash is not None:
        found = [r for r in found if r.pack_hash == pack_hash]
    return found


def effective_approval(path: Path, workflow_hash: str, pack_hash: str) -> dict[str, Any]:
    records = list_review_decisions(path, workflow_hash=workflow_hash, pack_hash=pack_hash)
    approvals = [r for r in records if r.decision == "approve"]
    return {
        "ok": True,
        "approved": bool(approvals),
        "workflow_hash": workflow_hash,
        "pack_hash": pack_hash,
        "matched_records": len(records),
        "approving_reviewers": sorted({r.reviewer for r in approvals}),
    }
