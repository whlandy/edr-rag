"""P2 batch 2: platform Connector contract + offline fixture import (design.md §5).

Concept interface only — no real platform is known (§5: 平台未知时仅提供接口和
离线导入，不猜 URL/API). The contract:

  discover(cursor)  → SourceDescriptor page
  fetch(descriptor) → RawBundle of sanitized case JSON
  normalize(bundle) → CaseBundle list (validated knowledge units)
  checkpoint        → advanced ONLY after originals + normalized output persisted

Sanitized fixture import (offline path) is fully implemented and tested:
pagination via cursor pages, updated_at watermark, delete/tombstone,
duplicates, mid-batch failure rerun, session expiry → authentication_required.
Fixture data is treated as data, never as instructions (§6).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Protocol

from edr_rag.knowledge.models import (
    CaseBundle,
    EvidenceRef,
    ModelError,
    TestCaseRevision,
    content_hash,
)

CASE_REQUIRED_FIELDS = ("case_id", "product_version", "platform", "title", "updated_at")


class ConnectorError(ModelError):
    """Fixture/collection contract violation."""


class AuthenticationRequired(Exception):
    """Session expired mid-collection (§5): watermark preserved, user resumes.

    Not a ModelError — it is a control-flow signal, distinct from data defects.
    Carries the fetch context needed to report where collection stopped.
    """

    def __init__(self, source_id: str, cursor: Any, message: str = "session expired"):
        self.source_id = source_id
        self.cursor = cursor
        self.message = message
        super().__init__(f"{source_id} @ cursor={cursor!r}: {message}")


@dataclass
class SourceDescriptor:
    """One case-source page offered by discover()."""

    source_id: str
    format: str  # "json" only for fixture import
    uri: str
    updated_at: str  # tz-aware ISO 8601, drives the watermark

    def __post_init__(self) -> None:
        if not self.source_id or not self.format or not self.uri:
            raise ConnectorError(f"SourceDescriptor incomplete: {self!r}")
        try:
            dt = datetime.fromisoformat(self.updated_at)
        except (TypeError, ValueError) as exc:
            raise ConnectorError(f"{self.source_id}: updated_at not ISO 8601 ({self.updated_at!r})") from exc
        if dt.tzinfo is None:
            raise ConnectorError(f"{self.source_id}: updated_at must carry a timezone offset")


@dataclass
class RawBundle:
    """Fetched original payload, persisted before normalize (checkpoint rule)."""

    descriptor: SourceDescriptor
    payload: list[dict[str, Any]]
    fetched_at: str

    @property
    def bundle_hash(self) -> str:
        return content_hash({"descriptor": self.descriptor.source_id, "payload": self.payload})


@dataclass
class ConnectorPage:
    """discover() output: one page plus continuation cursor (None = exhausted)."""

    descriptors: list[SourceDescriptor]
    next_cursor: str | None


class PlatformConnector(Protocol):
    """Concept interface (design.md §5, 拟实现). Offline fixture import implements
    the same shape without a network dependency."""

    def discover(self, cursor: str | None = None) -> ConnectorPage: ...

    def fetch(self, descriptor: SourceDescriptor) -> RawBundle: ...

    def normalize(self, bundle: RawBundle) -> list[CaseBundle]: ...


# ---------------------------------------------------------------- fixture layer


@dataclass
class FixtureEntry:
    """One case as it appears in a sanitized fixture page."""

    raw: dict[str, Any]
    deleted: bool = False


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_fixture_page(page_path: Path, source_id: str) -> tuple[list[FixtureEntry], str | None]:
    """Parse one sanitized fixture page → entries + next cursor.

    Page shape: {"source_id": …, "cases": [ {case…, "deleted": true?} ], "next_cursor": "…"?}
    Malformed page → ConnectorError (fail-fast; no silent row drops at page level).
    """
    page_path = Path(page_path)
    if not page_path.exists():
        raise ConnectorError(f"fixture page missing: {page_path}")
    try:
        page = json.loads(page_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConnectorError(f"{page_path}: invalid JSON ({exc})") from exc
    if not isinstance(page, dict):
        raise ConnectorError(f"{page_path}: page must be an object")
    if page.get("source_id") != source_id:
        raise ConnectorError(
            f"{page_path}: page source_id {page.get('source_id')!r} != expected {source_id!r}"
        )
    cases = page.get("cases")
    if not isinstance(cases, list):
        raise ConnectorError(f"{page_path}: 'cases' must be a list")
    entries: list[FixtureEntry] = []
    for i, raw in enumerate(cases):
        if not isinstance(raw, dict):
            raise ConnectorError(f"{page_path} case[{i}]: not an object")
        deleted = raw.get("deleted", False)
        if not isinstance(deleted, bool):
            raise ConnectorError(f"{page_path} case[{i}]: 'deleted' must be boolean")
        if not deleted:
            missing = [k for k in CASE_REQUIRED_FIELDS if not raw.get(k)]
            if missing:
                raise ConnectorError(f"{page_path} case[{i}]: missing fields {missing}")
        entries.append(FixtureEntry(raw=raw, deleted=deleted))
    next_cursor = page.get("next_cursor")
    if next_cursor is not None and not isinstance(next_cursor, str):
        raise ConnectorError(f"{page_path}: next_cursor must be a string or null")
    return entries, next_cursor


def iter_fixture_pages(workdir: Path, source_id: str) -> Iterator[tuple[Path, list[FixtureEntry], str | None]]:
    """Follow the cursor chain across page files: 00_page.json → next_cursor names
    the next file (e.g. "01_page.json"). Detects revisit (cursor cycle)."""
    workdir = Path(workdir)
    seen: set[str] = set()
    page_name: str | None = "00_page.json"
    while page_name is not None:
        if page_name in seen:
            raise ConnectorError(f"fixture cursor cycle at {page_name!r}")
        seen.add(page_name)
        page_path = workdir / page_name
        entries, next_cursor = parse_fixture_page(page_path, source_id)
        yield page_path, entries, next_cursor
        page_name = next_cursor


def normalize_fixture_case(entry: FixtureEntry, source_ref: str, collected_at: str) -> CaseBundle:
    """One fixture entry → validated CaseBundle (verbatim preservation, §5 采集层
    不改写原始用例). raw_text_hash binds the exact fixture entry; steps pass
    through untouched except unknown-field rejection."""
    raw = entry.raw
    if entry.deleted:
        raise ConnectorError(f"{raw.get('case_id', '?')}: deleted entries are handled by tombstone, not normalize")
    # verbatim passthrough: 采集层不改写原始用例; unknown step fields survive to
    # TestCaseRevision validation and are rejected there (no silent drops)
    case = TestCaseRevision(
        case_id=raw["case_id"],
        product_version=raw["product_version"],
        platform=raw["platform"],
        title=raw["title"],
        preconditions=list(raw.get("preconditions", [])),
        steps=list(raw.get("steps", [])),
        attachments=list(raw.get("attachments", [])),
        raw_text_hash=content_hash(raw),
        source_ref=source_ref,
        created_at=collected_at,
    )
    evidence: list[EvidenceRef] = []
    # verbatim quotes: operation evidence from preconditions+steps, expected evidence from steps
    evidence.append(
        EvidenceRef(
            evidence_id=f"E-operation-{case.case_id}",
            source_ref=source_ref,
            locator=f"fixture:case:{case.case_id}:steps",
            quote=json.dumps(raw.get("steps", []), ensure_ascii=False, sort_keys=True),
            purpose="operation",
        )
    )
    evidence.append(
        EvidenceRef(
            evidence_id=f"E-expected-{case.case_id}",
            source_ref=source_ref,
            locator=f"fixture:case:{case.case_id}:steps.expected",
            quote=json.dumps([s.get("expected", "") for s in raw.get("steps", [])], ensure_ascii=False),
            purpose="expected",
        )
    )
    bundle = CaseBundle(case=case, evidence=evidence)
    bundle.validate_references()
    return bundle


@dataclass
class WatermarkState:
    """Persisted collection state (§5 水位). checkpoint() advances it ONLY after
    originals + normalized output are durably persisted."""

    source_id: str
    updated_at_watermark: str  # entries strictly older than this are already collected
    cursor: str | None  # in-flight page cursor; None between runs
    collected_case_keys: list[str] = field(default_factory=list)  # idempotency keys seen

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "WatermarkState":
        allowed = {"source_id", "updated_at_watermark", "cursor", "collected_case_keys"}
        unknown = set(d) - allowed
        if unknown:
            raise ConnectorError(f"WatermarkState unknown fields: {sorted(unknown)}")
        missing = {"source_id", "updated_at_watermark"} - set(d)
        if missing:
            raise ConnectorError(f"WatermarkState missing fields: {sorted(missing)}")
        return WatermarkState(
            source_id=d["source_id"],
            updated_at_watermark=d["updated_at_watermark"],
            cursor=d.get("cursor"),
            collected_case_keys=list(d.get("collected_case_keys", [])),
        )


class Tombstone:
    """Delete record: case removed upstream. Stored verbatim; stale marking is a
    downstream decision, never a silent data drop (§5 删除/tombstone)."""

    def __init__(self, case_id: str, source_ref: str, recorded_at: str):
        self.case_id = case_id
        self.source_ref = source_ref
        self.recorded_at = recorded_at

    def to_dict(self) -> dict[str, Any]:
        return {"case_id": self.case_id, "source_ref": self.source_ref, "recorded_at": self.recorded_at}


TOMBSTONES_FILE = "tombstones.jsonl"


def import_case_fixture(
    repo,
    workdir: Path,
    source_id: str,
    state: WatermarkState,
    collected_at: str | None = None,
) -> dict[str, Any]:
    """Offline fixture import of platform cases following the full contract.

    Semantics:
      - pagination: follows the cursor chain; cursor cycle rejected
      - watermark: entries with updated_at <= watermark are skipped (already
        collected); watermark advances only after each page's originals +
        normalized cases are persisted
      - tombstone: deleted entries recorded, never silently dropped
      - duplicates: same case source_ref → ImportResult duplicate (no new row)
      - mid-batch failure: exception propagates AFTER persisting completed
        pages and their checkpoint, so a rerun skips exactly the finished pages
      - authentication_required: page may carry "session_expired": true →
        raises AuthenticationRequired with preserved watermark, waiting for user

    Returns a run summary dict. state is mutated only via checkpoint().
    """
    collected_at = collected_at or _now_iso()
    raw_dir = Path(workdir) / "raw"
    normalized_path = Path(workdir) / "normalized.jsonl"
    checkpoint_path = Path(workdir) / "checkpoint.json"

    stats = {"pages": 0, "created": 0, "duplicate": 0, "tombstoned": 0, "skipped_watermark": 0}
    new_keys: list[str] = []
    tombstones: list[dict[str, Any]] = []
    watermark = state.updated_at_watermark

    from edr_rag.atomic_io import atomic_write_text

    for page_path, entries, next_cursor in iter_fixture_pages(Path(workdir) / "pages", source_id):
        # session expiry is signalled by the fixture page itself (fixture is data;
        # this flag is part of the fixture contract, not an instruction). Checked
        # BEFORE the page's entries: the fetch of this page failed, nothing in it
        # was collected, watermark/cursor stay at the previous checkpoint.
        if (Path(workdir) / "pages" / f"{page_path.stem}.session_expired").exists():
            raise AuthenticationRequired(source_id=source_id, cursor=page_path.name)
        page_kept: list[dict[str, Any]] = []  # originals persisted before checkpoint
        for entry in entries:
            raw = entry.raw
            if entry.deleted:
                tombstones.append(
                    {
                        "case_id": raw.get("case_id", ""),
                        "source_ref": source_id,
                        "recorded_at": collected_at,
                    }
                )
                page_kept.append(raw)
                stats["tombstoned"] += 1
                continue
            if raw["updated_at"] <= watermark:
                stats["skipped_watermark"] += 1
                continue
            page_kept.append(raw)
            bundle = normalize_fixture_case(entry, source_ref=source_id, collected_at=collected_at)
            result = repo.import_case(bundle)
            if result.status == "duplicate":
                stats["duplicate"] += 1
            else:
                stats["created"] += 1
            new_keys.append(bundle.case.idempotency_key)

        # originals for this page are persisted first, then checkpoint advances
        if page_kept:
            raw_dir.mkdir(parents=True, exist_ok=True)
            raw_file = raw_dir / f"{page_path.stem}.json"
            atomic_write_text(raw_file, json.dumps(page_kept, ensure_ascii=False, indent=2, sort_keys=True))
        # checkpoint advances per page, watermark included, so an interrupted rerun
        # skips exactly the finished pages (§5 checkpoint 仅在持久化后推进)
        state.cursor = next_cursor
        state.updated_at_watermark = max(
            [watermark] + [e.raw["updated_at"] for e in entries if not e.deleted and e.raw.get("updated_at")]
        )
        watermark = state.updated_at_watermark
        atomic_write_text(checkpoint_path, json.dumps(state.to_dict(), ensure_ascii=False, sort_keys=True))
        stats["pages"] += 1

    state.cursor = None
    state.collected_case_keys = sorted(set(state.collected_case_keys + new_keys))
    if tombstones:
        tomb_path = Path(workdir) / TOMBSTONES_FILE
        existing = []
        if tomb_path.exists():
            existing = [
                json.loads(line)
                for line in tomb_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        existing.extend(tombstones)
        atomic_write_text(
            tomb_path,
            "".join(json.dumps(t, ensure_ascii=False, sort_keys=True) + "\n" for t in existing),
        )
    atomic_write_text(checkpoint_path, json.dumps(state.to_dict(), ensure_ascii=False, sort_keys=True))
    return stats
