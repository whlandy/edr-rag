"""Tests for parse_toc, parse_html, and section_path priority resolver.

These tests cover Commit B scope:
  - parse_toc.py: .hhc → TocEntry list, file→path index
  - parse_html.py: HTML → ParsedPage (cleaned content, headings, links, images)
  - section_path.py: 4-level priority contract

All tests are pure-function / fixture-based (no real CHM file required).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running directly
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from edr_rag.parse_toc import (  # noqa: E402
    TocEntry,
    build_file_to_path_index,
    parse_hhc,
)
from edr_rag.parse_html import ParsedPage, parse_html_file  # noqa: E402
from edr_rag.section_path import first_h1, resolve_section_path  # noqa: E402


# ---------- parse_toc tests ----------

def _write(tmp: Path, body: str) -> Path:
    p = tmp / "toc.hhc"
    p.write_text(body, encoding="utf-8")
    return p


def test_parse_hhc_single_level(tmp_workdir: Path):
    hhc = _write(
        tmp_workdir,
        """<HTML><BODY><UL>
<LI><OBJECT type="text/sitemap">
  <param name="Name" value="Introduction">
  <param name="Local" value="intro.html">
</OBJECT></LI>
</UL></BODY></HTML>""",
    )
    entries = parse_hhc(hhc)
    assert len(entries) == 1
    assert entries[0].name == "Introduction"
    assert entries[0].file == "intro.html"
    assert entries[0].path == ["Introduction"]
    print("[PASS] parse_hhc single level")


def test_parse_hhc_nested_levels(tmp_workdir: Path):
    hhc = _write(
        tmp_workdir,
        """<HTML><BODY><UL>
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
</UL></BODY></HTML>""",
    )
    entries = parse_hhc(hhc)
    assert len(entries) == 2
    by_file = {e.file: e for e in entries}
    assert by_file["network/index.html"].path == ["网络"]
    assert by_file["network/proxy.html"].path == ["网络", "代理"]
    print("[PASS] parse_hhc nested levels preserve hierarchy")


def test_parse_hhc_no_local_is_group_node(tmp_workdir: Path):
    """TOC entries without Local are folder/group nodes — should NOT emit."""
    hhc = _write(
        tmp_workdir,
        """<HTML><BODY><UL>
<LI><OBJECT type="text/sitemap">
  <param name="Name" value="Group">
</OBJECT>
  <UL>
  <LI><OBJECT type="text/sitemap">
    <param name="Name" value="Leaf">
    <param name="Local" value="leaf.html">
  </OBJECT></LI>
  </UL>
</LI>
</UL></BODY></HTML>""",
    )
    entries = parse_hhc(hhc)
    # Only leaf emitted; group without Local is omitted
    assert len(entries) == 1
    assert entries[0].name == "Leaf"
    assert entries[0].path == ["Group", "Leaf"]
    print("[PASS] parse_hhc group-only nodes are skipped")


def test_parse_hhc_backslash_normalized(tmp_workdir: Path):
    hhc = _write(
        tmp_workdir,
        """<HTML><BODY><UL>
<LI><OBJECT type="text/sitemap">
  <param name="Name" value="X">
  <param name="Local" value="dir\\sub\\page.html">
</OBJECT></LI>
</UL></BODY></HTML>""",
    )
    entries = parse_hhc(hhc)
    index = build_file_to_path_index(entries)
    assert "dir/sub/page.html" in index
    assert index["dir/sub/page.html"] == ["X"]
    print("[PASS] parse_hhc normalizes backslashes to forward slashes")


def test_parse_hhc_missing_returns_empty(tmp_workdir: Path):
    """If file doesn't exist, return empty list (caller falls back)."""
    bogus = tmp_workdir / "nonexistent.hhc"
    try:
        entries = parse_hhc(bogus)
    except FileNotFoundError:
        # Either raising or returning [] is acceptable P0 behavior;
        # the contract is that normalize.py sees an empty result and falls back.
        print("[PASS] parse_hhc missing file raises FileNotFoundError (caller falls back)")
        return
    assert entries == []
    print("[PASS] parse_hhc missing file returns [] (caller falls back)")


# ---------- parse_html tests ----------

def _write_html(tmp: Path, body: str) -> Path:
    p = tmp / "page.html"
    p.write_text(body, encoding="utf-8")
    return p


def test_parse_html_extracts_title_h1_fallback(tmp_workdir: Path):
    html = _write_html(
        tmp_workdir,
        """<html><head><title>代理设置</title></head><body><p>body</p></body></html>""",
    )
    page = parse_html_file(html, extracted_dir=tmp_workdir)
    assert page.title == "代理设置"
    assert "body" in page.content
    print("[PASS] parse_html extracts <title>")


def test_parse_html_title_falls_back_to_h1(tmp_workdir: Path):
    html = _write_html(
        tmp_workdir,
        """<html><head></head><body><h1>使用h1作为title</h1><p>content</p></body></html>""",
    )
    page = parse_html_file(html, extracted_dir=tmp_workdir)
    assert page.title == "使用h1作为title"
    print("[PASS] parse_html title falls back to first h1")


def test_parse_html_extracts_headings_in_order(tmp_workdir: Path):
    html = _write_html(
        tmp_workdir,
        """<html><body>
<h1>一级</h1>
<h2>二级</h2>
<h1>另一个一级</h1>
<h3>三级</h3>
</body></html>""",
    )
    page = parse_html_file(html, extracted_dir=tmp_workdir)
    assert page.headings == ["一级", "二级", "另一个一级", "三级"]
    print("[PASS] parse_html extracts headings in document order")


def test_parse_html_strips_script_style(tmp_workdir: Path):
    html = _write_html(
        tmp_workdir,
        """<html><head>
<style>p { color: red; }</style>
<script>alert('x')</script>
</head><body>
<h1>T</h1>
<p>real content</p>
</body></html>""",
    )
    page = parse_html_file(html, extracted_dir=tmp_workdir)
    assert "alert" not in page.content
    assert "color: red" not in page.content
    assert "real content" in page.content
    print("[PASS] parse_html strips script/style/meta/link")


def test_parse_html_extracts_links(tmp_workdir: Path):
    html = _write_html(
        tmp_workdir,
        """<html><body>
<h1>T</h1>
<a href="next.html">Next</a>
<a href="https://example.com/help">Help</a>
</body></html>""",
    )
    page = parse_html_file(html, extracted_dir=tmp_workdir)
    assert len(page.links) == 2
    hrefs = {l["href"] for l in page.links}
    assert hrefs == {"next.html", "https://example.com/help"}
    assert any(l["text"] == "Next" for l in page.links)
    print("[PASS] parse_html extracts link href+text")


def test_parse_html_extracts_images_with_asset_id(tmp_workdir: Path):
    html = _write_html(
        tmp_workdir,
        """<html><body>
<h1>T</h1>
<img src="../images/proxy.png" alt="代理图">
<img src="icon.gif" alt="">
</body></html>""",
    )
    page = parse_html_file(html, extracted_dir=tmp_workdir)
    assert len(page.images) == 2
    srcs = [i["src"] for i in page.images]
    # Backslashes normalized, leading ../ NOT silently stripped here —
    # path-traversal resolution is Commit C's job. parse_html just records.
    assert any("proxy.png" in s for s in srcs)
    assert all(i["asset_id"].startswith("asset.") for i in page.images)
    print("[PASS] parse_html extracts image src/alt/asset_id")


def test_parse_html_preserves_list_structure(tmp_workdir: Path):
    html = _write_html(
        tmp_workdir,
        """<html><body>
<h1>步骤</h1>
<ol><li>第一步</li><li>第二步</li></ol>
<ul><li>注意a</li><li>注意b</li></ul>
</body></html>""",
    )
    page = parse_html_file(html, extracted_dir=tmp_workdir)
    assert "1. 第一步" in page.content
    assert "2. 第二步" in page.content
    assert "* 注意a" in page.content
    assert "* 注意b" in page.content
    print("[PASS] parse_html preserves ordered/unordered list structure")


# ---------- section_path priority contract ----------

def test_section_path_priority_1_hhc():
    """hhc path wins over html heading and title."""
    hhc_index = {"network/proxy.html": ["网络", "代理", "配置代理"]}
    path = resolve_section_path(
        hhc_index=hhc_index,
        file="network/proxy.html",
        headings=["完全不同的h1"],
        title="完全不同的title",
    )
    assert path == ["网络", "代理", "配置代理"]
    print("[PASS] priority 1: hhc wins over headings and title")


def test_section_path_priority_2_heading():
    """No hhc → fall back to h1 list."""
    path = resolve_section_path(
        hhc_index={},
        file="orphan.html",
        headings=["配置服务器", "前提条件"],
        title="ignored title",
    )
    assert path == ["配置服务器", "前提条件"]
    print("[PASS] priority 2: html heading fallback")


def test_section_path_priority_3_title():
    """No hhc, no h1 → fall back to [title]."""
    path = resolve_section_path(
        hhc_index={},
        file="orphan.html",
        headings=[],
        title="孤页",
    )
    assert path == ["孤页"]
    print("[PASS] priority 3: [title] fallback")


def test_section_path_priority_4_empty():
    """No hhc, no h1, no title → []."""
    path = resolve_section_path(
        hhc_index={},
        file="orphan.html",
        headings=[],
        title="",
    )
    assert path == []
    print("[PASS] priority 4: [] when everything is empty")


def test_section_path_priority_whitespace_handling():
    """Whitespace-only title should be treated as empty."""
    path = resolve_section_path(
        hhc_index={},
        file="orphan.html",
        headings=[],
        title="   ",
    )
    assert path == []
    print("[PASS] whitespace-only title → []")


def test_section_path_priority_hhc_normalizes_keys():
    """Resolve_section_path must normalize file key the same way parse_toc does."""
    hhc_index = {"network/proxy.html": ["网络", "代理"]}  # forward slash
    # Caller passes backslash — should still match
    path = resolve_section_path(
        hhc_index=hhc_index,
        file="network\\proxy.html",
        headings=["x"],
        title="x",
    )
    assert path == ["网络", "代理"]
    print("[PASS] hhc lookup normalizes backslash keys")


def test_section_path_hhc_takes_priority_over_multiple_h1():
    hhc_index = {"x.html": ["A", "B", "C"]}
    path = resolve_section_path(
        hhc_index=hhc_index,
        file="x.html",
        headings=["D", "E", "F", "G"],  # 4 h1s — would win otherwise
        title="Z",
    )
    assert path == ["A", "B", "C"]
    print("[PASS] hhc wins even when many h1s present")


def test_heading_hierarchy_helper_takes_only_h1():
    """first_h1 returns the first non-empty heading (or empty string)."""
    # Empty / whitespace input
    assert first_h1([]) == ""
    assert first_h1(["   "]) == ""
    # First non-empty wins, rest are ignored (this is NOT a hierarchy helper)
    assert first_h1(["h1 text", "h2 text"]) == "h1 text"
    assert first_h1(["", "  ", "actual h1", "actual h2"]) == "actual h1"
    print("[PASS] first_h1 helper handles empty/whitespace")


# ---------- runner ----------

import tempfile
from contextlib import contextmanager


@contextmanager
def _tmp_workdir():
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


TESTS = [
    # parse_toc
    test_parse_hhc_single_level,
    test_parse_hhc_nested_levels,
    test_parse_hhc_no_local_is_group_node,
    test_parse_hhc_backslash_normalized,
    test_parse_hhc_missing_returns_empty,
    # parse_html
    test_parse_html_extracts_title_h1_fallback,
    test_parse_html_title_falls_back_to_h1,
    test_parse_html_extracts_headings_in_order,
    test_parse_html_strips_script_style,
    test_parse_html_extracts_links,
    test_parse_html_extracts_images_with_asset_id,
    test_parse_html_preserves_list_structure,
    # section_path priority
    test_section_path_priority_1_hhc,
    test_section_path_priority_2_heading,
    test_section_path_priority_3_title,
    test_section_path_priority_4_empty,
    test_section_path_priority_whitespace_handling,
    test_section_path_priority_hhc_normalizes_keys,
    test_section_path_hhc_takes_priority_over_multiple_h1,
    test_heading_hierarchy_helper_takes_only_h1,
]


def main() -> int:
    passed = 0
    failed = 0
    for t in TESTS:
        # Each test that needs tmp gets a fresh one
        if "tmp_workdir" in t.__code__.co_varnames:
            with _tmp_workdir() as tmp:
                try:
                    t(tmp)
                    passed += 1
                except AssertionError as e:
                    failed += 1
                    print(f"[FAIL] {t.__name__}: {e}")
                except Exception as e:
                    failed += 1
                    print(f"[ERROR] {t.__name__}: {type(e).__name__}: {e}")
        else:
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