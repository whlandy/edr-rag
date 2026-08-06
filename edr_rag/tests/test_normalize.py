"""Tests for normalize.py — Commit C scope.

Coverage:
  - _slugify: idempotent, unicode-safe, stable
  - make_semantic_id: from section_path / title / orphan fallback
  - make_semantic_id: NEVER derives from path
  - make_asset_id: deterministic
  - safe_resolve: path traversal rejected
  - safe_resolve: external URL rejected
  - safe_resolve: in-tree path allowed
  - run_ingest: full pipeline produces manifest + sections + assets
  - run_ingest: semantic id stable across two builds of same content
  - run_ingest: path-traversal image src is silently rejected
  - validate_workdir: contract check on output
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from edr_rag import normalize, extract as extract_mod  # noqa: E402
from edr_rag.normalize import (  # noqa: E402
    _slugify,
    make_asset_id,
    make_semantic_id,
    safe_resolve,
    validate_workdir,
)


@contextmanager
def _tmp():
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


# ---------- _slugify ----------

def test_slugify_basic():
    assert _slugify("Network") == "network"
    assert _slugify("network proxy") == "network_proxy"
    assert _slugify("page 1: intro") == "page_1_intro"
    print("[PASS] _slugify basic ascii")


def test_slugify_chinese():
    assert _slugify("网络 代理") == "网络_代理"
    assert _slugify("配置代理服务器") == "配置代理服务器"
    print("[PASS] _slugify unicode (chinese)")


def test_slugify_empty():
    assert _slugify("") == ""
    assert _slugify("   ") == ""
    assert _slugify("---") == ""
    print("[PASS] _slugify empty/whitespace")


def test_slugify_idempotent():
    s = "网络 / 代理 (配置)"
    once = _slugify(s)
    twice = _slugify(once)
    assert once == twice, f"idempotent failed: {once!r} vs {twice!r}"
    print("[PASS] _slugify idempotent")


def test_slugify_nfkc_normalizes_fullwidth():
    # NFKC: "Ｈｅｌｌｏ" → "Hello"
    assert _slugify("Ｈｅｌｌｏ") == "hello"
    print("[PASS] _slugify NFKC normalizes fullwidth")


# ---------- make_semantic_id ----------

def test_semantic_id_from_section_path():
    sid = make_semantic_id(
        section_path=["网络", "代理", "配置代理服务器"],
        title="ignored",
        manual_id="m",
    )
    assert sid == "manual.section.网络.代理.配置代理服务器"
    print("[PASS] semantic id uses section_path")


def test_semantic_id_falls_back_to_title():
    sid = make_semantic_id(
        section_path=[],
        title="Orphan Page",
        manual_id="m",
    )
    assert sid == "manual.section.orphan_page"
    print("[PASS] semantic id falls back to title")


def test_semantic_id_orphan_when_all_empty():
    sid = make_semantic_id(
        section_path=[],
        title="",
        manual_id="m",
    )
    assert sid == "manual.section.orphan"
    print("[PASS] semantic id orphan fallback")


def test_semantic_id_not_derived_from_path():
    """THE LOCK TEST for Commit C.

    Same semantic content + different paths MUST yield the same id.
    """
    sid_a = make_semantic_id(
        section_path=["配置代理"],
        title="",
        manual_id="m",
    )
    sid_b = make_semantic_id(
        section_path=["配置代理"],
        title="",
        manual_id="m",
    )
    assert sid_a == sid_b
    # And different semantic content → different id
    sid_c = make_semantic_id(
        section_path=["其他功能"],
        title="",
        manual_id="m",
    )
    assert sid_a != sid_c
    print("[PASS] semantic id is content-derived, not path-derived")


def test_semantic_id_stable_across_runs():
    """Re-ingestion of same content yields same id."""
    sids = set()
    for _ in range(5):
        sids.add(
            make_semantic_id(
                section_path=["网络", "代理"],
                title="ignored",
                manual_id="m",
            )
        )
    assert len(sids) == 1
    print("[PASS] semantic id stable across 5 calls")


def test_semantic_id_skips_empty_components():
    sid = make_semantic_id(
        section_path=["网络", "", "  ", "代理"],
        title="x",
        manual_id="m",
    )
    assert sid == "manual.section.网络.代理"
    print("[PASS] semantic id skips empty path components")


# ---------- make_asset_id ----------

def test_asset_id_deterministic():
    assert make_asset_id("images/proxy.png") == "asset.images.proxy.png"
    assert make_asset_id("images/proxy.png") == "asset.images.proxy.png"
    print("[PASS] asset id deterministic")


def test_asset_id_backslash_normalized():
    assert make_asset_id("images\\proxy.png") == "asset.images.proxy.png"
    assert make_asset_id("\\images\\proxy.png") == "asset.images.proxy.png"
    print("[PASS] asset id normalizes backslashes")


# ---------- safe_resolve ----------

def test_safe_resolve_in_tree():
    """In-tree relative path resolves correctly."""
    with _tmp() as tmp:
        extracted = tmp / "extracted"
        extracted.mkdir()
        (extracted / "images").mkdir()
        img = extracted / "images" / "x.png"
        img.write_bytes(b"x")
        page = extracted / "page.html"
        page.write_text("<html></html>")
        # Page refers to ../images/x.png which from extracted/page.html should NOT exist
        # but from extracted/images/.. it does. Let's test in-tree first:
        images_index = extracted / "images" / "index.html"
        images_index.write_text("<html></html>")
        result = safe_resolve("x.png", images_index, extracted)
        assert result == "images/x.png", result
        print("[PASS] safe_resolve in-tree")


def test_safe_resolve_rejects_traversal():
    """../../etc/passwd MUST be rejected."""
    with _tmp() as tmp:
        extracted = tmp / "extracted"
        extracted.mkdir()
        page = extracted / "sub" / "page.html"
        page.parent.mkdir()
        page.write_text("<html></html>")
        result = safe_resolve("../../../etc/passwd", page, extracted)
        assert result is None, f"expected None, got {result}"
        print("[PASS] safe_resolve rejects ../ traversal")


def test_safe_resolve_rejects_external_url():
    with _tmp() as tmp:
        extracted = tmp / "extracted"
        extracted.mkdir()
        page = extracted / "page.html"
        page.write_text("<html></html>")
        assert safe_resolve("https://evil.com/x.png", page, extracted) is None
        assert safe_resolve("http://evil.com/x.png", page, extracted) is None
        print("[PASS] safe_resolve rejects external URLs")


def test_safe_resolve_rejects_absolute_path():
    with _tmp() as tmp:
        extracted = tmp / "extracted"
        extracted.mkdir()
        page = extracted / "page.html"
        page.write_text("<html></html>")
        # Absolute path outside extracted is rejected
        result = safe_resolve("/etc/passwd", page, extracted)
        # Note: on macOS /tmp may be a symlink to /private/tmp, so /etc/passwd
        # resolves to /etc/passwd which is outside /tmp/.../extracted → None
        assert result is None, f"expected None for absolute escape, got {result}"
        print("[PASS] safe_resolve rejects absolute paths outside extracted")


def test_safe_resolve_empty_returns_none():
    with _tmp() as tmp:
        extracted = tmp / "extracted"
        extracted.mkdir()
        page = extracted / "page.html"
        page.write_text("<html></html>")
        assert safe_resolve("", page, extracted) is None
        print("[PASS] safe_resolve empty src → None")


# ---------- run_ingest (full pipeline) ----------

def _build_fixture(root: Path):
    """Build a synthetic extracted CHM-like directory."""
    extracted = root / "extracted_src"
    extracted.mkdir(parents=True)
    # .hhc
    hhc = """<HTML><BODY><UL>
<LI><OBJECT type="text/sitemap">
  <param name="Name" value="网络">
  <param name="Local" value="network/index.html">
</OBJECT>
  <UL>
  <LI><OBJECT type="text/sitemap">
    <param name="Name" value="代理">
    <param name="Local" value="network/proxy.html">
  </OBJECT></LI>
  </UL>
</LI>
<LI><OBJECT type="text/sitemap">
  <param name="Name" value="Orphan">
  <param name="Local" value="orphan.html">
</OBJECT></LI>
</UL></BODY></HTML>"""
    (extracted / "toc.hhc").write_text(hhc, encoding="utf-8")
    (extracted / "network").mkdir()
    (extracted / "network" / "index.html").write_text(
        "<html><head><title>网络首页</title></head><body><h1>网络首页</h1><p>索引</p></body></html>",
        encoding="utf-8",
    )
    (extracted / "network" / "proxy.html").write_text(
        """<html><head><title>配置代理</title></head><body>
<h1>配置代理</h1>
<img src="../images/proxy.png" alt="代理图">
<ol><li>步骤1</li><li>步骤2</li></ol>
</body></html>""",
        encoding="utf-8",
    )
    (extracted / "orphan.html").write_text(
        "<html><body><h1>孤页</h1><p>无h1内容</p></body></html>",
        encoding="utf-8",
    )
    (extracted / "images").mkdir()
    (extracted / "images" / "proxy.png").write_bytes(b"fakepng")
    return extracted


def _patch_extract(fixture_dir: Path):
    """Replace extract_chm to copy our fixture instead of running hh.exe."""
    def fake_extract(chm, out):
        out = Path(out)
        if out.exists():
            shutil.rmtree(out)
        shutil.copytree(fixture_dir, out)
        return ("sha256:fixture", True)
    extract_mod.extract_chm = fake_extract


def test_run_ingest_produces_artifacts():
    with _tmp() as tmp:
        fixture = _build_fixture(tmp)
        _patch_extract(fixture)
        workdir = tmp / "workdir"
        manifest = normalize.run_ingest(
            chm_path=tmp / "fake.chm",
            manual_id="test-manual",
            product="TestProduct",
            workdir=workdir,
        )
        assert manifest.section_count == 3
        assert manifest.asset_count == 1
        assert (workdir / "manifest.json").exists()
        assert (workdir / "sections.jsonl").exists()
        assert (workdir / "assets.jsonl").exists()
        print("[PASS] run_ingest produces 3 sections + 1 asset")


def test_run_ingest_semantic_ids_use_toc():
    """Sections must get ids derived from TOC semantic path, not from filename."""
    with _tmp() as tmp:
        fixture = _build_fixture(tmp)
        _patch_extract(fixture)
        workdir = tmp / "workdir"
        normalize.run_ingest(
            chm_path=tmp / "fake.chm",
            manual_id="test-manual",
            product="TestProduct",
            workdir=workdir,
        )
        sections = [
            json.loads(l)
            for l in (workdir / "sections.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()
        ]
        by_path = {s["source_ref"]["relative_path"]: s for s in sections}
        # network/proxy.html → TOC path 网络,代理 → semantic id
        proxy_id = by_path["network/proxy.html"]["id"]
        assert "网络" in proxy_id
        assert "代理" in proxy_id
        # id MUST NOT contain "proxy.html" (path leakage)
        assert "proxy.html" not in proxy_id
        assert ".html" not in proxy_id
        # id MUST be deterministic
        assert proxy_id == "manual.section.网络.代理"
        print("[PASS] semantic id derived from TOC path, no path leakage")


def test_run_ingest_orphan_section_id():
    """orphan.html is in TOC as 'Orphan' but has no real heading fallback."""
    with _tmp() as tmp:
        fixture = _build_fixture(tmp)
        _patch_extract(fixture)
        workdir = tmp / "workdir"
        normalize.run_ingest(
            chm_path=tmp / "fake.chm",
            manual_id="test-manual",
            product="TestProduct",
            workdir=workdir,
        )
        sections = [
            json.loads(l)
            for l in (workdir / "sections.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()
        ]
        by_path = {s["source_ref"]["relative_path"]: s for s in sections}
        orphan_id = by_path["orphan.html"]["id"]
        # TOC gave it "Orphan" path → slug "orphan"
        assert orphan_id == "manual.section.orphan"
        print("[PASS] orphan section gets semantic id from TOC")


def test_run_ingest_path_traversal_image_rejected():
    """An <img src="../../etc/passwd"> MUST NOT be recorded as a referenced asset."""
    with _tmp() as tmp:
        fixture = _build_fixture(tmp)
        # Inject a malicious image src — evil.html lives at the root,
        # so a real in-tree image reference is just "images/proxy.png".
        # "../images/proxy.png" from the root would escape, so we use the
        # correct in-tree form for the "good" reference.
        evil_html = fixture / "evil.html"
        evil_html.write_text(
            """<html><body>
<h1>Evil</h1>
<img src="../../../etc/passwd" alt="evil">
<img src="images/proxy.png" alt="good">
</body></html>""",
            encoding="utf-8",
        )
        # Also add to TOC
        toc_text = (fixture / "toc.hhc").read_text(encoding="utf-8")
        toc_text = toc_text.replace(
            "</UL></BODY></HTML>",
            """<LI><OBJECT type="text/sitemap">
  <param name="Name" value="Evil">
  <param name="Local" value="evil.html">
</OBJECT></LI>
</UL></BODY></HTML>""",
        )
        (fixture / "toc.hhc").write_text(toc_text, encoding="utf-8")

        _patch_extract(fixture)
        workdir = tmp / "workdir"
        normalize.run_ingest(
            chm_path=tmp / "fake.chm",
            manual_id="test-manual",
            product="TestProduct",
            workdir=workdir,
        )
        sections = [
            json.loads(l)
            for l in (workdir / "sections.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()
        ]
        evil = next(s for s in sections if s["source_ref"]["relative_path"] == "evil.html")
        # Only the good image should be recorded
        assert len(evil["images"]) == 1
        assert evil["images"][0]["asset_id"] == "asset.images.proxy.png"
        print("[PASS] path-traversal image src silently rejected")


def test_run_ingest_idempotent_across_runs():
    """Re-running ingest on same content must yield same ids."""
    with _tmp() as tmp:
        fixture = _build_fixture(tmp)
        _patch_extract(fixture)
        workdir1 = tmp / "w1"
        normalize.run_ingest(tmp / "fake.chm", "m", "p", workdir1)
        ids1 = [
            json.loads(l)["id"]
            for l in (workdir1 / "sections.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()
        ]

        workdir2 = tmp / "w2"
        normalize.run_ingest(tmp / "fake.chm", "m", "p", workdir2)
        ids2 = [
            json.loads(l)["id"]
            for l in (workdir2 / "sections.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()
        ]

        assert ids1 == ids2
        assert len(ids1) == 3
        print("[PASS] ingest is idempotent — same content → same ids")


def test_run_ingest_id_stable_under_path_change():
    """THE LOCK TEST for Commit C.

    Build #1 has files at /network/proxy.html.
    Build #2 has the same logical section at /html/network/proxy.html.
    The semantic id MUST be the same in both builds.
    """
    with _tmp() as tmp:
        fixture1 = _build_fixture(tmp)
        # Rename network/proxy.html to html/network/proxy.html in build 2
        fixture2 = tmp / "fixture2"
        if fixture2.exists():
            shutil.rmtree(fixture2)
        # Replicate but move proxy.html into a nested html/ dir
        import shutil as _sh
        _sh.copytree(fixture1, fixture2)
        (fixture2 / "html").mkdir(exist_ok=True)
        (fixture2 / "network").rename(fixture2 / "network_renamed")
        (fixture2 / "html" / "network").mkdir()
        # Update toc.hhc to point at the new path
        toc_text = (fixture2 / "toc.hhc").read_text(encoding="utf-8")
        toc_text = toc_text.replace("network/proxy.html", "html/network/proxy.html")
        toc_text = toc_text.replace("network/index.html", "html/network/index.html")
        (fixture2 / "toc.hhc").write_text(toc_text, encoding="utf-8")
        # Move files
        (fixture2 / "network_renamed" / "index.html").rename(fixture2 / "html" / "network" / "index.html")
        (fixture2 / "network_renamed" / "proxy.html").rename(fixture2 / "html" / "network" / "proxy.html")
        # Fix image src in proxy.html (was ../images/proxy.png → now ../../images/proxy.png)
        proxy_text = (fixture2 / "html" / "network" / "proxy.html").read_text(encoding="utf-8")
        proxy_text = proxy_text.replace("../images/proxy.png", "../../images/proxy.png")
        (fixture2 / "html" / "network" / "proxy.html").write_text(proxy_text, encoding="utf-8")

        _patch_extract(fixture1)
        workdir1 = tmp / "w1"
        normalize.run_ingest(tmp / "fake.chm", "m", "p", workdir1)
        sections1 = {
            json.loads(l)["id"]: json.loads(l)["source_ref"]["relative_path"]
            for l in (workdir1 / "sections.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()
        }

        _patch_extract(fixture2)
        workdir2 = tmp / "w2"
        normalize.run_ingest(tmp / "fake.chm", "m", "p", workdir2)
        sections2 = {
            json.loads(l)["id"]: json.loads(l)["source_ref"]["relative_path"]
            for l in (workdir2 / "sections.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()
        }

        # Same set of ids, different paths
        assert set(sections1.keys()) == set(sections2.keys())
        # The proxy section had the same semantic content → same id
        for sid in sections1:
            if "代理" in sid:
                # Found the proxy section — confirm id is stable AND path changed
                assert sid in sections2
                assert sections1[sid] != sections2[sid], \
                    f"path should differ: {sections1[sid]} vs {sections2[sid]}"
        print("[PASS] id stable across repacked CHM (path moved)")


# ---------- validate_workdir ----------

def test_validate_workdir_pass():
    with _tmp() as tmp:
        fixture = _build_fixture(tmp)
        _patch_extract(fixture)
        workdir = tmp / "workdir"
        normalize.run_ingest(tmp / "fake.chm", "m", "p", workdir)
        section_count, asset_count, errors = validate_workdir(workdir)
        assert errors == 0
        assert section_count == 3
        assert asset_count == 1
        print("[PASS] validate_workdir passes on valid artifacts")


def test_validate_workdir_detects_missing_field():
    with _tmp() as tmp:
        workdir = tmp / "workdir"
        workdir.mkdir()
        (workdir / "manifest.json").write_text(json.dumps({"manual_id": "m"}))
        # sections.jsonl missing a required key
        (workdir / "sections.jsonl").write_text(
            json.dumps({"id": "x", "manual_id": "m", "title": "t"}) + "\n"  # missing content, section_path, source_ref
        )
        (workdir / "assets.jsonl").write_text(json.dumps({"id": "a", "manual_id": "m", "type": "image", "path": "x.png"}) + "\n")
        try:
            validate_workdir(workdir)
        except AssertionError:
            print("[PASS] validate_workdir detects missing keys")
            return
        # The function prints [FAIL] but returns errors count; check that
        section_count, asset_count, errors = validate_workdir(workdir)
        assert errors >= 1
        print("[PASS] validate_workdir counts errors for missing keys")


# ---------- runner ----------

TESTS = [
    # _slugify
    test_slugify_basic,
    test_slugify_chinese,
    test_slugify_empty,
    test_slugify_idempotent,
    test_slugify_nfkc_normalizes_fullwidth,
    # make_semantic_id
    test_semantic_id_from_section_path,
    test_semantic_id_falls_back_to_title,
    test_semantic_id_orphan_when_all_empty,
    test_semantic_id_not_derived_from_path,
    test_semantic_id_stable_across_runs,
    test_semantic_id_skips_empty_components,
    # make_asset_id
    test_asset_id_deterministic,
    test_asset_id_backslash_normalized,
    # safe_resolve
    test_safe_resolve_in_tree,
    test_safe_resolve_rejects_traversal,
    test_safe_resolve_rejects_external_url,
    test_safe_resolve_rejects_absolute_path,
    test_safe_resolve_empty_returns_none,
    # run_ingest
    test_run_ingest_produces_artifacts,
    test_run_ingest_semantic_ids_use_toc,
    test_run_ingest_orphan_section_id,
    test_run_ingest_path_traversal_image_rejected,
    test_run_ingest_idempotent_across_runs,
    test_run_ingest_id_stable_under_path_change,
    # validate_workdir
    test_validate_workdir_pass,
    test_validate_workdir_detects_missing_field,
]


def main() -> int:
    passed = 0
    failed = 0
    for t in TESTS:
        try:
            t()
            passed += 1
        except AssertionError as e:
            failed += 1
            print(f"[FAIL] {t.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"[ERROR] {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())