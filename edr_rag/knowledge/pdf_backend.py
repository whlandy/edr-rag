"""P2 last batch: PDF optional parsing backend (design.md §5).

PDF 是可选后端：只有在本机存在真实解析依赖时才产出正文，绝不编造内容。

Backend detection (explicit, ordered):
  1. pypdf      (pip package, cross-platform)
  2. Quartz     (macOS PDFKit via pyobjc bindings — available on this host)
  3. neither    → PdfUnavailable (needs_review signal), no pseudo text

Section structure (§5 保留章节、页码):
  1. document outline/bookmarks (real TOC) when present — label + destination
     page from the PDF itself, never invented
  2. fallback: one section per page (deterministic; no heuristic merging that
     could silently fold real chapters together)

Output contract mirrors the existing P0 normalize workdir (manifest.json +
sections.jsonl), so the PDF path feeds the same knowledge wiring:
import_pdf_document() → SourceRevision(format="pdf") + DocumentSectionRevision
rows via import_document_workdir(source_format="pdf").

Design rules enforced here (§5):
  - 缺依赖 → PdfUnavailable，调用方得到 needs_review 信号，不输出伪正文。
  - 扫描件（全部页面无文本层）→ PdfUnavailable；OCR 不在本任务范围（§5 后续
    可选扩展），不静默截断、不输出空壳正文。
  - 部分页面无文本层 → 如实写入 scan_report（needs_review=True），报告持久化，
    不失明；正文不伪造缺层页的内容。
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from edr_rag.atomic_io import atomic_write_text
from edr_rag.knowledge.models import ModelError

BACKEND_PYPDF = "pypdf"
BACKEND_QUARTZ = "quartz"


class PdfUnavailable(ModelError):
    """No usable PDF backend, or document yields no extractable text (§5
    缺依赖/扫描件 → needs_review，不输出伪正文). Subclass of ModelError so
    callers can catch it with knowledge-layer contract errors; the message
    distinguishes "no backend" from "bad data"."""


def detect_pdf_backend() -> str | None:
    """Return the first usable backend id, or None (§5: explicit detection,
    no guessing). Import errors are expected control flow here."""
    try:
        import pypdf  # noqa: F401
        return BACKEND_PYPDF
    except ImportError:
        pass
    try:
        import Quartz  # noqa: F401
        return BACKEND_QUARTZ
    except ImportError:
        return None


@dataclass
class PageExtraction:
    """One page: extracted text or an explicit no-text-layer marker."""

    page_number: int  # 1-based
    text: str
    has_text_layer: bool


@dataclass
class _QuartzDoc:
    """Internal: quartz extraction needs the document object for outline
    lookups after pages are read; kept alongside the page list."""
    document: Any


def extract_pdf_pages(pdf_path: Path, backend: str | None = None) -> tuple[list[PageExtraction], str]:
    """Extract per-page text with the detected/forced backend.

    Raises PdfUnavailable when no backend exists. Raises ModelError when the
    file is unreadable as PDF (bad data is a contract error, not pseudo text).
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise ModelError(f"pdf file missing: {pdf_path}")
    if backend is None:
        backend = detect_pdf_backend()
    if backend is None:
        raise PdfUnavailable(
            f"no PDF backend available (install pypdf or pyobjc/Quartz); "
            f"refusing to emit pseudo text for {pdf_path}"
        )
    if backend == BACKEND_PYPDF:
        return _extract_pypdf(pdf_path), backend
    if backend == BACKEND_QUARTZ:
        return _extract_quartz(pdf_path)[0], backend
    raise ModelError(f"unknown pdf backend: {backend!r}")


def _extract_pypdf(pdf_path: Path) -> list[PageExtraction]:
    from pypdf import PdfReader

    try:
        reader = PdfReader(str(pdf_path))
        count = len(reader.pages)
    except Exception as exc:  # pypdf raises a broad family of parse errors
        raise ModelError(f"{pdf_path}: unreadable as PDF ({exc})") from exc
    pages: list[PageExtraction] = []
    for i in range(count):
        try:
            text = _norm_text(reader.pages[i].extract_text() or "")
        except Exception as exc:
            raise ModelError(f"{pdf_path} page {i + 1}: text extraction failed ({exc})") from exc
        pages.append(PageExtraction(page_number=i + 1, text=text, has_text_layer=bool(text.strip())))
    return pages


def _norm_text(text: str) -> str:
    """NFKC-normalize extracted text. PDFKit glyph maps can emit compatibility
    forms (e.g. Kangxi radical ⽂ U+2F42 instead of 文 U+6587); NFKC folds them
    so content matching is faithful. Same convention as P0 _slugify."""
    return unicodedata.normalize("NFKC", text) if text else text


def _extract_quartz(pdf_path: Path) -> tuple[list[PageExtraction], Any]:
    import Quartz

    data = pdf_path.read_bytes()
    nsdata = Quartz.NSData.dataWithBytes_length_(data, len(data))
    doc = Quartz.PDFDocument.alloc().initWithData_(nsdata)
    if doc is None:
        raise ModelError(f"{pdf_path}: unreadable as PDF (PDFKit returned no document)")
    pages: list[PageExtraction] = []
    for i in range(doc.pageCount()):
        page = doc.pageAtIndex_(i)
        text = _norm_text(page.string() or "")
        pages.append(PageExtraction(page_number=i + 1, text=text, has_text_layer=bool(text.strip())))
    return pages, doc


# ---------------------------------------------------------------- structure

def read_quartz_outline(doc: Any) -> dict[int, str] | None:
    """Read the PDF's own bookmark tree → {1-based page number: outline label}.

    Uses PDFKit PDFOutline (label + destination page) — real document TOC,
    never invented. Top-level entries only; nested children are folded by
    their top-level chapter (§5 保留章节即可，不重建深层级). Returns None when
    the document has no outline. Duplicate labels on the same page keep the
    first occurrence.
    """
    import Quartz

    root = doc.outlineRoot()
    if root is None or root.numberOfChildren() == 0:
        return None
    mapping: dict[int, str] = {}
    for i in range(root.numberOfChildren()):
        child = root.childAtIndex_(i)
        dest = child.destination()
        if dest is None:
            continue
        page = dest.page()
        if page is None:
            continue
        page_number = doc.indexForPage_(page) + 1  # 0-based → 1-based
        label = child.label() or ""
        if label and page_number not in mapping:
            mapping[page_number] = label
    return mapping or None


def group_pages(
    pages: list[PageExtraction],
    outline: dict[int, str] | None = None,
) -> list[dict[str, Any]]:
    """Group pages → section dicts (deterministic).

    With an outline: sections start at outlined pages; the label is the
    section title (from the document). Pages before the first outline entry
    form a preamble section titled by page. Without an outline: one section
    per page. Pages with no text layer are recorded in pages_without_text of
    the section they belong to — never silently dropped.
    """
    sections: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for pg in pages:
        title = outline.get(pg.page_number) if outline else None
        starts_new = title is not None or current is None or outline is None
        if starts_new:
            if title is None:
                title = f"page-{pg.page_number}"  # preamble or per-page fallback
            current = {
                "title": title,
                "section_path": [title],
                "pages": [pg.page_number],
                "content_parts": [pg.text.strip()] if pg.has_text_layer else [],
                "pages_without_text": [] if pg.has_text_layer else [pg.page_number],
            }
            sections.append(current)
        else:
            current["pages"].append(pg.page_number)
            if pg.has_text_layer:
                current["content_parts"].append(pg.text.strip())
            else:
                current["pages_without_text"].append(pg.page_number)
    return [
        {
            "title": s["title"],
            "section_path": s["section_path"],
            "pages": s["pages"],
            "content": "\n\n".join(s["content_parts"]).strip(),
            "pages_without_text": s["pages_without_text"],
        }
        for s in sections
    ]


@dataclass
class PdfScanReport:
    """Honest extraction report; drives the needs_review decision (§5)."""

    backend: str
    page_count: int
    pages_with_text: int
    structure: str = "per_page"  # outline | per_page
    pages_without_text: list[int] = field(default_factory=list)
    needs_review: bool = False
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def _slug_ascii(s: str) -> str:
    """ASCII-safe slug for knowledge-layer ids (_ID_RE is ASCII-only there).

    CJK and other non-ASCII text is preserved verbatim in section title and
    toc_path (the human-facing fields); the id only needs determinism.
    """
    s = unicodedata.normalize("NFKC", s).strip().lower()
    out = []
    for ch in s:
        if ch.isascii() and ch.isalnum():
            out.append(ch)
        elif ch in (" ", "-", "_"):
            out.append("_")
    slug = re.sub(r"_+", "_", "".join(out)).strip("_")
    return slug


def _section_ids(manual_id: str, sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Assign stable section ids: semantic slug when the title yields one,
    page-anchored id otherwise (outline labels are often non-ASCII)."""
    out = []
    used: set[str] = set()
    for i, sec in enumerate(sections, 1):
        slug = _slug_ascii(sec["title"])
        base = f"pdf.section.{slug}.{i}" if slug else f"pdf.section.page-{sec['pages'][0]}.{i}"
        sid = base
        n = 1
        while sid in used:
            n += 1
            sid = f"{base}-{n}"
        used.add(sid)
        sec = dict(sec)
        sec["id"] = sid
        out.append(sec)
    return out


def write_pdf_workdir(
    pdf_path: Path,
    workdir: Path,
    backend: str | None = None,
) -> tuple[Path, PdfScanReport]:
    """Extract PDF → P0-shaped workdir (manifest.json + sections.jsonl).

    Raises PdfUnavailable when no backend exists or the document is a full
    scan (caller maps to needs_review). Raises ModelError on unreadable/empty
    files — never writes pseudo text. A PARTIAL scan writes through with the
    pages_without_text listed honestly in the persisted scan_report.
    """
    pdf_path = Path(pdf_path)
    workdir = Path(workdir)
    if backend is None:
        backend = detect_pdf_backend()
    if backend is None:
        raise PdfUnavailable(
            f"no PDF backend available (install pypdf or pyobjc/Quartz); "
            f"refusing to emit pseudo text for {pdf_path}"
        )

    if backend == BACKEND_QUARTZ:
        pages, doc = _extract_quartz(pdf_path)
        outline = read_quartz_outline(doc)
    elif backend == BACKEND_PYPDF:
        pages = _extract_pypdf(pdf_path)
        outline = None  # pypdf outline support not wired; per_page fallback
    else:
        raise ModelError(f"unknown pdf backend: {backend!r}")
    if not pages:
        raise ModelError(f"{pdf_path}: zero pages — refusing empty output")

    grouped = group_pages(pages, outline)
    no_text = [p.page_number for p in pages if not p.has_text_layer]
    report = PdfScanReport(
        backend=backend,
        page_count=len(pages),
        pages_with_text=len(pages) - len(no_text),
        structure="outline" if outline else "per_page",
        pages_without_text=no_text,
    )
    if not any(p.has_text_layer for p in pages):
        raise PdfUnavailable(
            f"{pdf_path}: {len(no_text)}/{len(pages)} pages have no text layer "
            f"(backend={backend}); scan document, OCR not implemented (§5) — needs_review"
        )
    if no_text:
        report.needs_review = True
        report.reason = "partial_scan_pages_without_text_layer"

    manual_id = _slug_ascii(pdf_path.stem) or "pdf-manual"
    content_sections = [s for s in grouped if s["content"]]  # never pseudo content
    if not content_sections:
        raise PdfUnavailable(f"{pdf_path}: no section contains extractable text (backend={backend})")
    content_sections = _section_ids(manual_id, content_sections)
    rows = []
    for sec in content_sections:
        rows.append(
            {
                "id": sec["id"],
                "manual_id": manual_id,
                "title": sec["title"],
                "section_path": list(sec["section_path"]),
                "source_ref": {"relative_path": f"#page={sec['pages'][0]}", "anchor": None},
                "content": sec["content"],
                "headings": list(sec["section_path"]),
                "links": [],
                "images": [],
                "metadata": {
                    "pages": sec["pages"],
                    "pages_without_text": sec["pages_without_text"],
                },
            }
        )

    manifest = {
        "manual_id": manual_id,
        "source_pdf": str(pdf_path.resolve()),
        "source_hash": hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
        "extracted_dir": str(workdir),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "section_count": len(rows),
        "asset_count": 0,
        "parser_version": f"pdf-{backend}",
        "source_format": "pdf",
        "scan_report": report.to_dict(),
    }
    workdir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(workdir / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    atomic_write_text(
        workdir / "sections.jsonl",
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
    )
    return workdir, report


def import_pdf_document(
    repo,
    pdf_path: Path,
    workdir: Path,
    document_version: str,
    collected_at: str | None = None,
    backend: str | None = None,
):
    """One-call PDF import → SourceRevision(format="pdf") + section revisions.

    Uses the shared import_document_workdir wiring (idempotency, explicit
    version filter, hash rebind) so PDF and CHM documents obey the same
    knowledge-layer contract (§4). document_version must be explicit (跨版本
    不混检); it is NOT derived from the PDF.
    """
    if not document_version:
        raise ModelError("document_version required (跨版本不混检：不静默推导版本)")
    from edr_rag.knowledge.document_ingest import import_document_workdir

    write_pdf_workdir(pdf_path, workdir, backend=backend)
    return import_document_workdir(
        repo, workdir, document_version=document_version,
        collected_at=collected_at, source_format="pdf",
    )
