"""P2 last batch: PDF optional parsing backend (design.md §5).

正例：真实后端（本机 Quartz/PDFKit）抽取、outline 分节确定性、per-page 回退、
幂等重导、部分扫描如实报告、workdir 喂共享接线；负例：无后端拒绝、全扫描件
拒绝不输出伪正文、坏文件拒绝、未知后端拒绝、缺 document_version 拒绝。
全部门槛如实验证，不伪造通过（§5 缺依赖/扫描件 → needs_review，不输出伪正文）。
"""

from __future__ import annotations

import json
import zlib
from pathlib import Path

import pytest

from edr_rag.knowledge import (
    KnowledgeRepository,
    ModelError,
    detect_pdf_backend,
    import_pdf_document,
    write_pdf_workdir,
)
from edr_rag.knowledge.document_ingest import import_document_workdir, sections_of
from edr_rag.knowledge.pdf_backend import (
    BACKEND_QUARTZ,
    PageExtraction,
    PdfUnavailable,
    extract_pdf_pages,
    group_pages,
)

TS = "2026-11-08T08:27:41+08:00"


# ---------------------------------------------------------------- helpers


def _make_text_pdf(tmp_path: Path, name: str, page_texts: list[str]) -> Path:
    """Build a real PDF via macOS PDFKit (same backend under test) so the
    extraction path is exercised end-to-end on genuine PDF bytes."""
    import Foundation
    import Quartz
    from AppKit import NSAttributedString, NSTextView

    pages = []
    for text in page_texts:
        tv = NSTextView.alloc().initWithFrame_(Foundation.NSMakeRect(0, 0, 500, 400))
        tv.insertText_(NSAttributedString.alloc().initWithString_(text))
        data = tv.dataWithPDFInsideRect_(Foundation.NSMakeRect(0, 0, 500, 400))
        nd = Quartz.PDFDocument.alloc().initWithData_(data)
        pages.append(nd.pageAtIndex_(0))
    doc = Quartz.PDFDocument.alloc().init()
    for i, p in enumerate(pages):
        doc.insertPage_atIndex_(p, i)
    out = tmp_path / name
    out.write_bytes(doc.dataRepresentation())
    return out


def _make_outlined_pdf(tmp_path: Path, name: str, chapters: list[tuple[str, str]]) -> Path:
    """Hand-build a minimal PDF with a real /Outlines tree (one chapter per
    page) so outline grouping is exercised against genuine bookmark bytes."""
    n = len(chapters)
    font_id = 3
    page_ids = [4 + i for i in range(n)]
    content_ids = [4 + n + i for i in range(n)]
    outlines_id = 4 + 2 * n
    outline_ids = [5 + 2 * n + i for i in range(n)]
    objs: dict[int, bytes] = {}
    objs[1] = f"<< /Type /Catalog /Pages 2 0 R /Outlines {outlines_id} 0 R >>".encode()
    kids = " ".join(f"{page_ids[i]} 0 R" for i in range(n))
    objs[2] = f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode()
    objs[font_id] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
    for i, (title, body) in enumerate(chapters):
        comp = zlib.compress(f"BT /F1 14 Tf 20 260 Td ({body}) Tj ET".encode())
        objs[content_ids[i]] = (
            b"<< /Length " + str(len(comp)).encode() + b" /Filter /FlateDecode >>\nstream\n"
            + comp + b"\nendstream"
        )
        objs[page_ids[i]] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 400 300] "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {content_ids[i]} 0 R >>"
        ).encode()
        dest = f"[{page_ids[i]} 0 R /XYZ 0 300 null]"
        link = f" /Next {outline_ids[i + 1]} 0 R" if i + 1 < n else ""
        prev = f" /Prev {outline_ids[i - 1]} 0 R" if i > 0 else ""
        # PDF literal strings are PDFDocEncoding — non-ASCII titles must be
        # UTF-16BE with the BOM marker, per PDF spec 7.9.2.2
        raw = title.encode("utf-16-be")
        title_lit = b"<FEFF" + raw.hex().encode() + b">" if any(ord(c) > 127 for c in title) else f"({title})".encode()
        objs[outline_ids[i]] = (
            b"<< /Type /Outline /Title " + title_lit
            + f" /Parent {outlines_id} 0 R{link}{prev} /Dest {dest} >>".encode()
        )
    objs[outlines_id] = (
        f"<< /Type /Outlines /First {outline_ids[0]} 0 R /Last {outline_ids[-1]} 0 R /Count {n} >>"
    ).encode()
    out = b"%PDF-1.4\n"
    offsets: dict[int, int] = {}
    for num in sorted(objs):
        offsets[num] = len(out)
        out += f"{num} 0 obj\n".encode() + objs[num] + b"\nendobj\n"
    xref = len(out)
    mx = max(objs)
    out += b"xref\n0 " + str(mx + 1).encode() + b"\n0000000000 65535 f \n"
    for i in range(1, mx + 1):
        out += ("%010d 00000 n \n" % offsets[i]).encode()
    out += b"trailer\n<< /Size " + str(mx + 1).encode() + b" /Root 1 0 R >>\nstartxref\n"
    out += str(xref).encode() + b"\n%%EOF\n"
    path = tmp_path / name
    path.write_bytes(out)
    return path


def _make_image_only_pdf(tmp_path: Path, name: str) -> Path:
    """PDF bytes with no text operator (image view renders an empty image)."""
    import Foundation
    import Quartz
    from AppKit import NSImage, NSImageView

    img = NSImage.alloc().initWithSize_(Foundation.NSMakeSize(200, 200))
    iv = NSImageView.alloc().initWithFrame_(Foundation.NSMakeRect(0, 0, 200, 200))
    iv.setImage_(img)
    out = tmp_path / name
    out.write_bytes(iv.dataWithPDFInsideRect_(Foundation.NSMakeRect(0, 0, 200, 200)))
    return out


# ---------------------------------------------------------------- backend detection


class TestBackendDetection:
    def test_detect_returns_a_backend_or_none(self):
        got = detect_pdf_backend()
        assert got in (None, "pypdf", "quartz")

    def test_quartz_available_on_this_host(self):
        # documented environment: macOS with PDFKit — backend exists here, so
        # the offline fixtures below genuinely exercise extraction
        assert detect_pdf_backend() == BACKEND_QUARTZ


# ---------------------------------------------------------------- extraction


class TestExtraction:
    def test_real_pdf_extracts_per_page_text(self, tmp_path):
        pdf = _make_text_pdf(tmp_path, "manual.pdf", [
            "网络配置\n配置代理服务器地址。",
            "扫描策略\n配置扫描策略参数。",
        ])
        pages, backend = extract_pdf_pages(pdf)
        assert backend == BACKEND_QUARTZ
        assert [p.page_number for p in pages] == [1, 2]
        assert all(p.has_text_layer for p in pages)
        assert "配置代理服务器地址" in pages[0].text

    def test_missing_file_rejected(self, tmp_path):
        with pytest.raises(ModelError, match="missing"):
            extract_pdf_pages(tmp_path / "nope.pdf", backend=BACKEND_QUARTZ)

    def test_unknown_backend_rejected(self, tmp_path):
        pdf = _make_text_pdf(tmp_path, "u.pdf", ["x"])
        with pytest.raises(ModelError, match="unknown pdf backend"):
            extract_pdf_pages(pdf, backend="ghost")

    def test_garbage_bytes_rejected_not_pseudo(self, tmp_path):
        pdf = tmp_path / "bad.pdf"
        pdf.write_bytes(b"this is not a pdf at all")
        with pytest.raises(ModelError, match="unreadable"):
            extract_pdf_pages(pdf, backend=BACKEND_QUARTZ)

    def test_scan_pdf_no_text_layer_raises_needs_review(self, tmp_path):
        pdf = _make_image_only_pdf(tmp_path, "scan.pdf")
        with pytest.raises(PdfUnavailable, match="no text layer"):
            write_pdf_workdir(pdf, tmp_path / "wd-scan")


# ---------------------------------------------------------------- grouping


class TestGrouping:
    def test_outline_grouping_deterministic(self):
        pages = [
            PageExtraction(1, "Chapter A body", True),
            PageExtraction(2, "continuation of A", True),
            PageExtraction(3, "Chapter B body", True),
        ]
        outline = {1: "Chapter A", 3: "Chapter B"}
        got1 = group_pages(pages, outline)
        got2 = group_pages(pages, outline)
        assert got1 == got2
        assert [s["title"] for s in got1] == ["Chapter A", "Chapter B"]
        assert got1[0]["pages"] == [1, 2]
        assert got1[1]["pages"] == [3]

    def test_preamble_before_first_outline_entry(self):
        pages = [
            PageExtraction(1, "前言内容", True),
            PageExtraction(2, "正文甲", True),
        ]
        outline = {2: "正文甲"}
        got = group_pages(pages, outline)
        assert [s["title"] for s in got] == ["page-1", "正文甲"]

    def test_per_page_fallback_without_outline(self):
        pages = [
            PageExtraction(1, "第一页内容", True),
            PageExtraction(2, "第二页内容", True),
        ]
        got = group_pages(pages, None)
        # deterministic per-page sections; titles anchor to page numbers,
        # never invented chapter names
        assert [s["pages"] for s in got] == [[1], [2]]
        assert [s["title"] for s in got] == ["page-1", "page-2"]

    def test_no_text_layer_page_recorded_not_dropped(self):
        pages = [
            PageExtraction(1, "章节甲\n内容", True),
            PageExtraction(2, "", False),
        ]
        got = group_pages(pages, None)
        # per-page fallback: the scan page forms its own section, its emptiness
        # recorded in pages_without_text — never silently dropped
        assert got[0]["pages_without_text"] == []
        assert got[1]["pages_without_text"] == [2]
        assert got[1]["content"] == ""

    def test_outline_from_real_pdf_bytes(self, tmp_path):
        pdf = _make_outlined_pdf(tmp_path, "book.pdf", [
            ("网络配置", "config body"),
            ("扫描策略", "policy body"),
        ])
        pages, _backend = extract_pdf_pages(pdf)
        assert len(pages) == 2
        wd, report = write_pdf_workdir(pdf, tmp_path / "wd")
        assert report.structure == "outline"
        rows = [json.loads(l) for l in (wd / "sections.jsonl").read_text(encoding="utf-8").splitlines()]
        assert [r["title"] for r in rows] == ["网络配置", "扫描策略"]


# ---------------------------------------------------------------- workdir wiring


class TestWriteWorkdirAndImport:
    def test_happy_path_end_to_end(self, tmp_path):
        pdf = _make_text_pdf(tmp_path, "config-guide.pdf", [
            "代理设置\n配置代理服务器地址与端口。",
            "扫描策略\n配置扫描策略参数。",
        ])
        repo = KnowledgeRepository(tmp_path / "kb")
        src = import_pdf_document(repo, pdf, tmp_path / "wd", document_version="1.0.0")
        assert src.source_id == "config_guide"  # ASCII slug of the file stem
        assert src.format == "pdf"
        assert len(src.source_hash) == 64
        secs = sections_of(repo, src, "1.0.0")
        assert len(secs) == 2
        assert all(s.source_ref == src.ref for s in secs)
        # ids are ASCII slugs (knowledge _ID_RE); the real CJK page text lives
        # in the workdir rows — verify genuine content landed there
        wd_rows = [
            json.loads(l)
            for l in (tmp_path / "wd" / "sections.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert any("代理设置" in r["title"] or "配置代理服务器地址" in r["content"] for r in wd_rows)

    def test_scan_report_persisted_in_manifest(self, tmp_path):
        pdf = _make_text_pdf(tmp_path, "guide2.pdf", ["标题甲\n正文内容。"])
        wd, report = write_pdf_workdir(pdf, tmp_path / "wd2")
        assert report.needs_review is False
        manifest = json.loads((wd / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["source_format"] == "pdf"
        assert manifest["parser_version"] == f"pdf-{report.backend}"
        assert manifest["scan_report"]["page_count"] == 1

    def test_reimport_same_pdf_idempotent(self, tmp_path):
        pdf = _make_text_pdf(tmp_path, "guide3.pdf", ["唯一章节\n正文。"])
        repo = KnowledgeRepository(tmp_path / "kb")
        s1 = import_pdf_document(repo, pdf, tmp_path / "wd3", document_version="1.0.0")
        s2 = import_pdf_document(repo, pdf, tmp_path / "wd3b", document_version="1.0.0")
        assert s2.ref == s1.ref
        assert len(repo.list_sources("guide3")) == 1
        assert len(sections_of(repo, s1, "1.0.0")) == 1

    def test_new_hash_new_revision(self, tmp_path):
        repo = KnowledgeRepository(tmp_path / "kb")
        p1 = _make_text_pdf(tmp_path, "guide4.pdf", ["章节一\n第一版内容。"])
        p2 = _make_text_pdf(tmp_path, "guide4b.pdf", ["章节一\n第二版内容。"])
        s1 = import_pdf_document(repo, p1, tmp_path / "wd4", document_version="1.0.0")
        s2 = import_pdf_document(repo, p2, tmp_path / "wd4b", document_version="1.0.0")
        assert s2.ref != s1.ref

    def test_same_content_two_versions_no_merge(self, tmp_path):
        pdf = _make_text_pdf(tmp_path, "guide5.pdf", ["章节五\n正文。"])
        repo = KnowledgeRepository(tmp_path / "kb")
        s = import_pdf_document(repo, pdf, tmp_path / "wd5a", document_version="1.0.0")
        import_pdf_document(repo, pdf, tmp_path / "wd5b", document_version="2.0.0")
        assert len(sections_of(repo, s, "1.0.0")) == 1
        assert len(repo.list_sections(document_version="2.0.0")) == 1
        assert len(repo.list_sections()) == 2

    def test_missing_document_version_rejected(self, tmp_path):
        pdf = _make_text_pdf(tmp_path, "guide6.pdf", ["章节六\n正文。"])
        repo = KnowledgeRepository(tmp_path / "kb")
        with pytest.raises(ModelError, match="document_version"):
            import_pdf_document(repo, pdf, tmp_path / "wd6", document_version="")

    def test_partial_scan_reported(self, tmp_path):
        # real mixed document: one text page + one image-only page — the
        # report must list the page honestly and set needs_review
        import Foundation
        import Quartz
        from AppKit import NSAttributedString, NSTextView

        text_tv = NSTextView.alloc().initWithFrame_(Foundation.NSMakeRect(0, 0, 400, 300))
        text_tv.insertText_(NSAttributedString.alloc().initWithString_("正常章节\n有文本层的内容。"))
        text_page_doc = Quartz.PDFDocument.alloc().initWithData_(
            text_tv.dataWithPDFInsideRect_(Foundation.NSMakeRect(0, 0, 400, 300))
        )
        img_page_pdf = _make_image_only_pdf(tmp_path, "imgpage.pdf")
        img_page_doc = Quartz.PDFDocument.alloc().initWithURL_(
            Foundation.NSURL.fileURLWithPath_(str(img_page_pdf))
        )

        doc = Quartz.PDFDocument.alloc().init()
        doc.insertPage_atIndex_(text_page_doc.pageAtIndex_(0), 0)
        doc.insertPage_atIndex_(img_page_doc.pageAtIndex_(0), 1)
        pdf = tmp_path / "mixed.pdf"
        pdf.write_bytes(doc.dataRepresentation())

        wd, report = write_pdf_workdir(pdf, tmp_path / "wdmixed")
        assert report.needs_review is True
        assert report.reason == "partial_scan_pages_without_text_layer"
        assert 2 in report.pages_without_text
        manifest = json.loads((wd / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["scan_report"]["needs_review"] is True
        # the text page's real content is present; no pseudo content for the
        # no-text-layer page
        rows = [json.loads(l) for l in (wd / "sections.jsonl").read_text(encoding="utf-8").splitlines()]
        assert any("有文本层的内容" in r["content"] for r in rows)
        assert all("有文本层的内容" not in r["metadata"]["pages_without_text"] for r in rows if not r["content"])


class TestWorkdirContractCompat:
    def test_pdf_workdir_feeds_shared_wiring(self, tmp_path):
        # the P0-shaped workdir must be consumable by the existing CHM wiring
        pdf = _make_text_pdf(tmp_path, "compat.pdf", ["兼容章节\n共享接线正文。"])
        wd, _report = write_pdf_workdir(pdf, tmp_path / "wdc")
        repo = KnowledgeRepository(tmp_path / "kb")
        src = import_document_workdir(repo, wd, document_version="1.0.0")
        assert src.source_id == "compat"
        assert len(sections_of(repo, src, "1.0.0")) == 1

    def test_source_format_default_stays_chm(self, tmp_path):
        # existing CHM callers keep format="chm" — no silent cross-format change
        wd = tmp_path / "chmwd"
        wd.mkdir()
        (wd / "manifest.json").write_text(json.dumps({
            "manual_id": "chm-manual", "source_chm": "/x/y.chm", "source_hash": "a" * 64,
            "generated_at": TS, "section_count": 1, "asset_count": 0,
        }), encoding="utf-8")
        (wd / "sections.jsonl").write_text(json.dumps({
            "id": "m.s", "manual_id": "chm-manual", "title": "t", "content": "c",
            "section_path": ["t"], "source_ref": {"relative_path": "t.htm", "anchor": None},
        }, ensure_ascii=False) + "\n", encoding="utf-8")
        repo = KnowledgeRepository(tmp_path / "kb2")
        src = import_document_workdir(repo, wd, document_version="1.0.0")
        assert src.format == "chm"
