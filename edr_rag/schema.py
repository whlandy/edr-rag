"""P0 schemas for CHM → Manual IR.

Defines the data contracts for CHM ingestion output artifacts:
  - ManualSection: one per HTML page/section
  - AssetRef: one per image or resource
  - Manifest: describes the entire ingestion run

Identity model (locked in P0):
  - `id` is the SEMANTIC identity of a section (stable across re-ingestions).
    Example: "manual.section.network.proxy".
    Downstream P1/P2 (embedding, retrieval, action lookup) keys on this.
  - `source_ref` carries the EXTRACTION artifact identity (where the section
    was found in this particular extraction). It may legitimately change
    across re-ingestions if the CHM internals get repacked:
        network/proxy.htm     (v1)
        html/network/proxy.html (v2)
    Consumers must NOT key on `source_ref`.

These are plain dataclasses (not ORM, not pydantic) because P0 artifacts are
written as JSONL and consumed by simple readers. No LLM or embedding logic.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class Link:
    """A hyperlink from a manual section."""
    text: str
    href: str


@dataclass
class ImageRef:
    """A reference to an image within a section."""
    src: str
    alt: str = ""
    asset_id: str = ""


@dataclass
class SourceRef:
    """Extraction artifact identity for a section.

    Carries the location of the HTML file inside the extracted CHM and an
    optional anchor within that page. This is NOT a stable identity —
    re-ingestion may produce different SourceRef values for the same section.
    Key on `ManualSection.id`, not on `source_ref`.
    """
    relative_path: str  # relative path inside extracted_dir (e.g. "network/proxy.html")
    anchor: str | None = None


@dataclass
class ManualSection:
    """One useful HTML page or section from the CHM manual.

    Identity contract (P0):
      - `id` is the SEMANTIC identity of a section.
        Contract:
          * globally unique within `manual_id`
          * stable across re-ingestion (a re-extracted CHM with repacked
            paths MUST yield the same `id` for the same logical section)
          * MUST NOT contain extraction artifact paths
          * MUST NOT depend on HTML filename
          * format is implementation-defined (the generator may use TOC
            hierarchy, page operation slug, numeric ids, etc.)
        Downstream P1/P2 (embedding, retrieval, action lookup) keys on `id`.
      - `source_ref` MUST be set to where the section was found.
      - `section_path` priority: hhc hierarchy > html heading hierarchy > [title] > [].

    Migration note:
      P0 has no pre-SourceRef artifacts, so no backward-compatibility
      adapter is required. If future ingestions read legacy JSONL that
      has top-level `source_file` instead of `source_ref`, an explicit
      migration step is needed — do NOT silently coerce.
    """
    id: str
    manual_id: str
    product: str
    title: str
    section_path: list[str]
    source_ref: SourceRef
    content: str = ""
    headings: list[str] = field(default_factory=list)
    links: list[Link] = field(default_factory=list)
    images: list[ImageRef] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_jsonl(self) -> dict:
        d = asdict(self)
        # Convert nested dataclasses
        d["source_ref"] = {"relative_path": self.source_ref.relative_path, "anchor": self.source_ref.anchor}
        d["links"] = [{"text": l.text, "href": l.href} for l in self.links]
        d["images"] = [
            {"src": i.src, "alt": i.alt, "asset_id": i.asset_id}
            for i in self.images
        ]
        return d

    @staticmethod
    def from_jsonl(d: dict) -> ManualSection:
        d = dict(d)
        sr = d.get("source_ref")
        if sr is None:
            raise ValueError("ManualSection missing required field 'source_ref'")
        d["source_ref"] = SourceRef(**sr)
        d["links"] = [Link(**l) for l in d.get("links", [])]
        d["images"] = [ImageRef(**i) for i in d.get("images", [])]
        return ManualSection(**d)


@dataclass
class AssetRef:
    """An image or resource extracted from the CHM."""
    id: str
    manual_id: str
    type: str  # "image", "css", "js", etc.
    path: str  # relative path from extracted dir
    absolute_path: str  # absolute path on disk
    referenced_by: list[str] = field(default_factory=list)
    description: str | None = None

    def to_jsonl(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_jsonl(d: dict) -> AssetRef:
        return AssetRef(**d)


@dataclass
class Manifest:
    """Describes one ingestion run. Written as manifest.json."""
    manual_id: str
    source_chm: str
    source_hash: str
    extracted_dir: str
    generated_at: str  # ISO 8601
    section_count: int
    asset_count: int
    parser_version: str = "p0"

    def to_json(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_json(d: dict) -> Manifest:
        return Manifest(**d)