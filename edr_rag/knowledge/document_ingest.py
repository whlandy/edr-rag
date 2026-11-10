"""P2 wiring: P0 extractor output → knowledge-layer revisions (design.md §5).

Input is an existing P0 normalize workdir (manifest.json + sections.jsonl),
NOT a re-parse of the CHM. The extractor output is treated as RawBundle:

  manifest.json      → SourceRevision (source_id=manual_id, hash=source_hash,
                       source_updated_at=generated_at, collected_at=now)
  sections.jsonl     → one DocumentSectionRevision per ManualSection, keyed on
                       ManualSection.id (semantic identity, stable across
                       re-ingestion); section_content_hash binds the section
                       content under this source revision.

document_version must be supplied explicitly by the caller: P0 output has no
version field, so silent derivation is forbidden (§4 跨版本不混检 / 未知产品
版本不能静默合并). Re-import of the same (source identity + hash) is
idempotent; a genuinely new source revision marks prior section revisions
stale (rebind required, no silent inheritance of hash binding, §4/§8).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from edr_rag.knowledge.models import (
    DocumentSectionRevision,
    ModelError,
    SourceRevision,
    content_hash,
)

REQUIRED_SECTION_KEYS = ("id", "manual_id", "title", "content", "source_ref")


@dataclass
class SectionImportResult:
    status: str  # created | duplicate
    source_id: str
    section_count: int
    skipped: list[dict[str, str]]  # {"line": n, "reason": …} — never silent


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_sections(path: Path) -> list[tuple[int, dict[str, Any]]]:
    out: list[tuple[int, dict[str, Any]]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ModelError(f"sections.jsonl line {line_no}: invalid JSON ({exc})") from exc
        if not isinstance(obj, dict):
            raise ModelError(f"sections.jsonl line {line_no}: not an object")
        out.append((line_no, obj))
    return out


def import_document_workdir(
    repo,
    workdir: Path,
    document_version: str,
    collected_at: str | None = None,
    source_format: str = "chm",
) -> SourceRevision:
    """Wire one P0 extraction workdir into SourceRevision + section revisions.

    Raises ModelError on: missing manifest/sections, missing keys, missing
    document_version, manifest/sections manual_id mismatch. Idempotent: same
    manifest source_hash re-imported → source not re-saved, sections re-hashed
    but store keyed on (section_id, source_ref) stays duplicate-free.
    source_format labels the SourceRevision.format ("chm" default; the PDF
    backend passes "pdf") so cross-format identity is explicit, not derived.
    """
    if not document_version:
        raise ModelError("document_version required (跨版本不混检：不静默推导版本)")

    workdir = Path(workdir)
    manifest_path = workdir / "manifest.json"
    sections_path = workdir / "sections.jsonl"
    if not manifest_path.exists():
        raise ModelError(f"missing {manifest_path}")
    if not sections_path.exists():
        raise ModelError(f"missing {sections_path}")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ModelError(f"manifest.json: invalid JSON ({exc})") from exc
    for key in ("manual_id", "source_hash", "generated_at"):
        if not manifest.get(key):
            raise ModelError(f"manifest.json: missing {key!r}")

    rows = _load_sections(sections_path)
    if not rows:
        raise ModelError("sections.jsonl: empty (no sections to wire)")

    source_id = manifest["manual_id"]
    for line_no, obj in rows:
        missing = [k for k in REQUIRED_SECTION_KEYS if not obj.get(k)]
        if missing:
            raise ModelError(f"sections.jsonl line {line_no}: missing keys {missing}")
        if obj["manual_id"] != source_id:
            raise ModelError(
                f"sections.jsonl line {line_no}: manual_id {obj['manual_id']!r} != manifest manual_id {source_id!r}"
            )
        sr = obj["source_ref"]
        if not isinstance(sr, dict) or "relative_path" not in sr:
            raise ModelError(f"sections.jsonl line {line_no}: source_ref missing 'relative_path'")

    existing = repo.list_sources(source_id)
    same_hash = [s for s in existing if s.source_hash == manifest["source_hash"]]
    if same_hash:
        source = max(same_hash, key=lambda s: s.revision)
    else:
        source = SourceRevision(
            source_id=source_id,
            format=source_format,
            uri=str(manifest.get("source_chm") or manifest.get("source_pdf", "")),
            source_hash=manifest["source_hash"],
            source_updated_at=manifest["generated_at"],
            collected_at=collected_at or _now_iso(),
        )
        repo.save_source(source)

    for line_no, obj in rows:
        sr = obj["source_ref"]
        section = DocumentSectionRevision(
            section_id=obj["id"],
            document_version=document_version,
            source_ref=source.ref,
            section_content_hash=content_hash(
                {
                    "title": obj["title"],
                    "content": obj["content"],
                    "section_path": obj.get("section_path", []),
                    "relative_path": sr["relative_path"],
                    "anchor": sr.get("anchor"),
                }
            ),
            toc_path=list(obj.get("section_path", [])),
            page_or_chm_path=sr["relative_path"],
            created_at=collected_at or _now_iso(),
        )
        repo.save_section(section)
    return source


def sections_of(repo, source: SourceRevision, document_version: str) -> list[DocumentSectionRevision]:
    """Latest section revisions bound to a given source ref, explicit version filter only."""
    return repo.list_sections(source_ref=source.ref, document_version=document_version)
