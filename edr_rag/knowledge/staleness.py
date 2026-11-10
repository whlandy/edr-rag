"""P2 batch 3: stale-marking chain + tombstone-driven case invalidation.

design.md §4: 源变更新建 revision、标记下游 stale 并要求重编译；§5: 删除/
tombstone 不静默丢数据；§8: stale 是失效标记，证据失效不自动继承原审核。

Chain implemented here:
  document source change -> section revisions bound to the old source ref
  case supersede / tombstone -> tasks derived from that case -> workflows
  derived from those tasks
Stale marks are append-only facts in stale.jsonl. Loading normalizes both the
legacy case-supersede records ({"stale":…, "superseded_by":…} written by
KnowledgeRepository.import_case) and the richer StaleMark records.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from edr_rag.atomic_io import atomic_write_text
from edr_rag.knowledge.models import ModelError, SourceRevision

STALE_MARKS_FILE = "stale.jsonl"
STALE_KINDS = ("section", "case", "task", "workflow")
STALE_REASONS = (
    "superseded_by_new_source",
    "superseded_by_new_case",
    "tombstone_deleted",
    "downstream_of_stale_case",
)


class StalenessError(ModelError):
    """Stale-mark contract violation."""


def _require_tz(value: str, what: str) -> None:
    try:
        dt = datetime.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise StalenessError(f"{what}: not ISO 8601 ({value!r})") from exc
    if dt.tzinfo is None:
        raise StalenessError(f"{what}: timestamp must carry a timezone offset ({value!r})")


@dataclass
class StaleMark:
    """One invalidation fact: entity revision marked stale with a reason (§8)."""

    kind: str  # section | case | task | workflow
    ref: str  # "<entity>@rev<n>" or stable id
    reason: str  # one of STALE_REASONS
    source_ref: str = ""
    recorded_at: str = ""
    detail: str = ""

    def __post_init__(self) -> None:
        if self.kind not in STALE_KINDS:
            raise StalenessError(f"kind must be one of {STALE_KINDS}, got {self.kind!r}")
        if self.reason not in STALE_REASONS:
            raise StalenessError(f"reason must be one of {STALE_REASONS}, got {self.reason!r}")
        if not self.ref:
            raise StalenessError("ref required")
        if self.recorded_at:
            _require_tz(self.recorded_at, "recorded_at")

    @property
    def key(self) -> tuple[str, str]:
        return (self.kind, self.ref)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "StaleMark":
        allowed = {"kind", "ref", "reason", "source_ref", "recorded_at", "detail"}
        unknown = set(d) - allowed
        if unknown:
            raise StalenessError(f"StaleMark unknown fields: {sorted(unknown)}")
        missing = {"kind", "ref", "reason"} - set(d)
        if missing:
            raise StalenessError(f"StaleMark missing fields: {sorted(missing)}")
        return StaleMark(
            kind=d["kind"], ref=d["ref"], reason=d["reason"],
            source_ref=d.get("source_ref", ""), recorded_at=d.get("recorded_at", ""),
            detail=d.get("detail", ""),
        )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def normalize_record(d: dict[str, Any]) -> StaleMark:
    """Rich StaleMark records pass through; legacy case-supersede records
    ({"stale": …, "superseded_by": …}) normalize to kind=case."""
    if "reason" in d:
        return StaleMark.from_dict(d)
    if d.get("stale"):
        return StaleMark(
            kind="case", ref=d["stale"], reason="superseded_by_new_case",
            detail=f"superseded_by {d.get('superseded_by', '?')}",
        )
    raise StalenessError(f"unrecognized stale record: {sorted(d)}")


class StaleMarkRegistry:
    """Append-only stale marks with idempotent dedupe on (kind, ref)."""

    def __init__(self, repo, stale_path: Path | str | None = None, now=None):
        self.repo = repo
        self.path = Path(stale_path) if stale_path is not None else Path(repo.root) / STALE_MARKS_FILE
        self._now = now or _now_iso

    def load(self) -> list[StaleMark]:
        return [normalize_record(d) for d in _read_records(self.path)]

    def is_stale(self, kind: str, ref: str) -> bool:
        return any(m.kind == kind and m.ref == ref for m in self.load())

    def append(self, marks: list[StaleMark]) -> list[StaleMark]:
        """Persist marks; duplicates of an already-recorded (kind, ref) — in the
        store or within the same batch — are skipped. Returns the fresh marks
        written. File untouched when nothing fresh (legacy records stay
        verbatim)."""
        existing = self.load()
        seen = {m.key for m in existing}
        fresh: list[StaleMark] = []
        for m in marks:
            if m.key in seen:
                continue
            seen.add(m.key)
            fresh.append(m)
        if not fresh:
            return []
        lines = [json.dumps(m.to_dict(), ensure_ascii=False, sort_keys=True) for m in existing]
        lines += [json.dumps(m.to_dict(), ensure_ascii=False, sort_keys=True) for m in fresh]
        atomic_write_text(self.path, "".join(line + "\n" for line in lines))
        return fresh

    # ---------- source -> sections ----------

    def propagate_document_source_change(
        self, new_source: SourceRevision, document_version: str
    ) -> list[StaleMark]:
        """A genuinely new source revision (different hash) invalidates section
        revisions bound to prior refs of the same source_id (§4 源变更标记下游
        stale，重绑定要求). Sections of other document versions are not marked
        (跨版本不混检). Same-hash re-ingestion marks nothing."""
        if not document_version:
            raise StalenessError("document_version required (跨版本不混检：不静默推导版本)")
        priors = [
            s for s in self.repo.list_sources(new_source.source_id)
            if s.revision < new_source.revision and s.source_hash != new_source.source_hash
        ]
        marks: list[StaleMark] = []
        for prior in priors:
            for sec in self.repo.list_sections(source_ref=prior.ref, document_version=document_version):
                marks.append(StaleMark(
                    kind="section", ref=f"{sec.section_id}@rev{sec.revision}",
                    reason="superseded_by_new_source", source_ref=prior.ref,
                    recorded_at=self._now(),
                    detail=f"source {prior.source_id} rev{prior.revision} superseded by rev{new_source.revision}",
                ))
        return self.append(marks)

    # ---------- case -> task -> workflow ----------

    def _downstream_of_cases(self, case_pairs: set[tuple[str, str]]) -> list[StaleMark]:
        marks: list[StaleMark] = []
        stale_tasks: set[tuple[str, str]] = set()
        for t in self.repo.list_tasks():
            if (t.source_case_id, t.source_case_ref) in case_pairs:
                stale_tasks.add((t.task_id, t.source_case_ref))
                marks.append(StaleMark(
                    kind="task", ref=f"{t.task_id}@rev{t.revision}",
                    reason="downstream_of_stale_case", source_ref=t.source_case_ref,
                    recorded_at=self._now(),
                    detail=f"case {t.source_case_id} stale (证据失效不自动继承审核)",
                ))
        for w in self.repo.list_workflows():
            if (w.task_id, w.task_source_ref) in stale_tasks:
                marks.append(StaleMark(
                    kind="workflow", ref=f"{w.workflow_id}@rev{w.revision}",
                    reason="downstream_of_stale_case", source_ref=w.task_source_ref,
                    recorded_at=self._now(), detail=f"task {w.task_id} stale",
                ))
        return marks

    def apply_tombstone(self, case_id: str, source_ref: str) -> list[StaleMark]:
        """Tombstone-driven case invalidation (§5 删除/tombstone): every stored
        revision of the deleted case becomes stale with reason
        tombstone_deleted; tasks/workflows derived from it are marked stale —
        they never silently keep their prior validation (§8)."""
        cases = [
            c for c in self.repo.list_cases()
            if c.case_id == case_id and c.source_ref == source_ref
        ]
        marks: list[StaleMark] = []
        pairs: set[tuple[str, str]] = set()
        for c in cases:
            pairs.add((c.case_id, c.source_ref))
            marks.append(StaleMark(
                kind="case", ref=f"{c.case_id}@rev{c.revision}",
                reason="tombstone_deleted", source_ref=c.source_ref,
                recorded_at=self._now(), detail="case deleted upstream (tombstone)",
            ))
        marks.extend(self._downstream_of_cases(pairs))
        return self.append(marks)


def append_case_supersede_downstream(repo, case_id: str, old_source_refs: list[str], stale_path) -> int:
    """Repository hook: after a case revision supersedes priors, mark tasks and
    workflows derived from the old refs stale (证据失效不自动继承审核, §4/§8)."""
    registry = StaleMarkRegistry(repo, stale_path=stale_path)
    pairs = {(case_id, ref) for ref in old_source_refs}
    return len(registry.append(registry._downstream_of_cases(pairs)))
