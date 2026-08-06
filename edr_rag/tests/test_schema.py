"""Schema contract tests for tools.edr_rag.schema.

These tests are the P0 schema lock. They verify:
  1. All dataclasses roundtrip cleanly through JSONL serialization.
  2. Required fields are present and stable across serialization.
  3. Identity decoupling: `id` (semantic) survives, `source_ref` (artifact)
     can change without affecting `id`.

These tests should NOT depend on any downstream module (normalize, cli).
They are pure schema contract tests.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow running directly: `python3 tests/test_schema.py` from tools/edr_rag dir
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from edr_rag.schema import (  # noqa: E402
    AssetRef,
    ImageRef,
    Link,
    ManualSection,
    Manifest,
    SourceRef,
)


# ---------- Test helpers ----------

def _make_section() -> ManualSection:
    return ManualSection(
        id="manual.section.network.proxy",
        manual_id="product-manual",
        product="ProductName",
        title="配置代理服务器",
        section_path=["网络", "代理", "配置代理服务器"],
        source_ref=SourceRef(relative_path="network/proxy.html", anchor=None),
        content="步骤:\n1. 打开系统设置\n2. 输入代理地址",
        headings=["配置代理服务器", "操作步骤"],
        links=[Link(text="故障排查", href="troubleshooting.html")],
        images=[ImageRef(src="images/proxy.png", alt="代理设置页面", asset_id="asset.images.proxy_png")],
        metadata={"language": "zh-CN"},
    )


def _make_asset() -> AssetRef:
    return AssetRef(
        id="asset.images.proxy_png",
        manual_id="product-manual",
        type="image",
        path="images/proxy.png",
        absolute_path="/tmp/manual/images/proxy.png",
        referenced_by=["network/proxy.html"],
        description=None,
    )


def _make_manifest() -> Manifest:
    return Manifest(
        manual_id="product-manual",
        source_chm="/tmp/manual.chm",
        source_hash="sha256:abc",
        extracted_dir="/tmp/extracted",
        generated_at="2026-08-05T12:00:00+08:00",
        section_count=1,
        asset_count=1,
    )


# ---------- ManualSection ----------

def test_manual_section_roundtrip():
    s1 = _make_section()
    d = s1.to_jsonl()
    json_str = json.dumps(d, ensure_ascii=False)
    d2 = json.loads(json_str)
    s2 = ManualSection.from_jsonl(d2)
    assert s2.id == s1.id
    assert s2.manual_id == s1.manual_id
    assert s2.title == s1.title
    assert s2.section_path == s1.section_path
    assert s2.source_ref.relative_path == s1.source_ref.relative_path
    assert s2.content == s1.content
    assert len(s2.links) == 1
    assert s2.links[0].href == "troubleshooting.html"
    assert len(s2.images) == 1
    assert s2.images[0].asset_id == "asset.images.proxy_png"
    print("[PASS] ManualSection roundtrip")


def test_manual_section_required_fields_present():
    """Every ManualSection JSONL MUST have these fields."""
    required = {"id", "manual_id", "product", "title", "section_path",
                "source_ref", "content", "headings", "links", "images", "metadata"}
    d = _make_section().to_jsonl()
    missing = required - d.keys()
    assert not missing, f"Missing fields: {missing}"
    print("[PASS] ManualSection required fields present:", sorted(required))


def test_manual_section_identity_decoupled_from_source_ref():
    """`id` MUST survive a change in `source_ref`. This is the P0 lock."""
    s1 = _make_section()
    # Simulate re-ingestion where the same section is now under html/ subdir
    s2 = ManualSection(
        id=s1.id,  # SAME semantic id
        manual_id=s1.manual_id,
        product=s1.product,
        title=s1.title,
        section_path=s1.section_path,
        source_ref=SourceRef(relative_path="html/network/proxy.html", anchor=None),  # DIFFERENT
        content=s1.content,
        headings=s1.headings,
        links=s1.links,
        images=s1.images,
        metadata=s1.metadata,
    )
    assert s1.id == s2.id
    assert s1.source_ref.relative_path != s2.source_ref.relative_path
    # After roundtrip, semantic id is preserved
    d = s2.to_jsonl()
    s3 = ManualSection.from_jsonl(d)
    assert s3.id == s1.id
    assert s3.source_ref.relative_path == "html/network/proxy.html"
    print("[PASS] ManualSection id stable across source_ref change")


def test_manual_section_source_ref_anchor_optional():
    s = ManualSection(
        id="x", manual_id="m", product="P", title="t",
        section_path=["t"],
        source_ref=SourceRef(relative_path="a.html", anchor="step-3"),
    )
    d = s.to_jsonl()
    assert d["source_ref"]["anchor"] == "step-3"
    # Default anchor (None)
    s2 = ManualSection(
        id="x", manual_id="m", product="P", title="t",
        section_path=["t"],
        source_ref=SourceRef(relative_path="a.html"),
    )
    assert s2.source_ref.anchor is None
    print("[PASS] SourceRef anchor is optional")


def test_manual_section_rejects_missing_source_ref():
    """ManualSection.from_jsonl MUST reject a payload missing source_ref."""
    bad = {
        "id": "x", "manual_id": "m", "product": "P", "title": "t",
        "section_path": ["t"],
        # NO source_ref
        "content": "", "headings": [], "links": [], "images": [], "metadata": {},
    }
    try:
        ManualSection.from_jsonl(bad)
    except (ValueError, TypeError):
        print("[PASS] ManualSection rejects missing source_ref")
        return
    raise AssertionError("Expected ManualSection.from_jsonl to reject missing source_ref")


def test_manual_section_section_path_can_be_empty():
    """`section_path` is allowed to be [] per priority contract step 4."""
    s = ManualSection(
        id="orphan", manual_id="m", product="P", title="orphan page",
        section_path=[],  # fallback exhausted
        source_ref=SourceRef(relative_path="orphan.html"),
    )
    d = s.to_jsonl()
    assert d["section_path"] == []
    s2 = ManualSection.from_jsonl(d)
    assert s2.section_path == []
    print("[PASS] ManualSection.section_path can be empty")


# ---------- AssetRef ----------

def test_asset_ref_roundtrip():
    a1 = _make_asset()
    d = a1.to_jsonl()
    a2 = AssetRef.from_jsonl(json.loads(json.dumps(d)))
    assert a1.id == a2.id
    assert a1.path == a2.path
    assert a1.absolute_path == a2.absolute_path
    assert a1.referenced_by == a2.referenced_by
    print("[PASS] AssetRef roundtrip")


def test_asset_ref_required_fields():
    required = {"id", "manual_id", "type", "path", "absolute_path"}
    d = _make_asset().to_jsonl()
    missing = required - d.keys()
    assert not missing, f"Missing fields: {missing}"
    print("[PASS] AssetRef required fields present:", sorted(required))


# ---------- Manifest ----------

def test_manifest_roundtrip():
    m1 = _make_manifest()
    d = m1.to_json()
    m2 = Manifest.from_json(d)
    assert m1.manual_id == m2.manual_id
    assert m1.source_hash == m2.source_hash
    assert m1.section_count == m2.section_count
    assert m1.asset_count == m2.asset_count
    print("[PASS] Manifest roundtrip")


def test_manifest_required_fields():
    required = {"manual_id", "source_chm", "source_hash", "extracted_dir",
                "generated_at", "section_count", "asset_count"}
    d = _make_manifest().to_json()
    missing = required - d.keys()
    assert not missing, f"Missing fields: {missing}"
    print("[PASS] Manifest required fields present:", sorted(required))


def test_manifest_parser_version_default():
    m = Manifest(
        manual_id="m", source_chm="x", source_hash="h", extracted_dir="e",
        generated_at="t", section_count=0, asset_count=0,
    )
    assert m.parser_version == "p0"
    print("[PASS] Manifest.parser_version defaults to 'p0'")


# ---------- Identity decoupling — key test ----------

def test_id_unchanged_when_path_changes():
    """THE P0 LOCK TEST.

    `id` is the semantic identity. Re-ingestion under different extraction
    paths MUST yield the same `id` if the section is the same. Downstream
    retrieval/action lookup depends on this.
    """
    s_v1 = ManualSection(
        id="manual.section.network.proxy",
        manual_id="manual", product="P", title="proxy",
        section_path=["network", "proxy"],
        source_ref=SourceRef(relative_path="network/proxy.htm"),
    )
    s_v2 = ManualSection(
        id="manual.section.network.proxy",
        manual_id="manual", product="P", title="proxy",
        section_path=["network", "proxy"],
        source_ref=SourceRef(relative_path="html/network/proxy.html"),
    )
    assert s_v1.id == s_v2.id
    assert s_v1.source_ref != s_v2.source_ref
    print("[PASS] identity decoupling — same id, different source_ref")


# ---------- JSON key stability ----------

def test_manual_section_jsonl_keys_stable():
    """The JSONL top-level key set MUST be locked.

    Downstream P1/P2 readers depend on this. Field drift breaks the
    contract even if the data roundtrips correctly.
    """
    expected = {
        "id",
        "manual_id",
        "product",
        "title",
        "section_path",
        "source_ref",
        "content",
        "headings",
        "links",
        "images",
        "metadata",
    }
    payload = _make_section().to_jsonl()
    assert set(payload.keys()) == expected, (
        f"key drift: extra={set(payload.keys()) - expected}, "
        f"missing={expected - set(payload.keys())}"
    )
    print("[PASS] ManualSection JSONL keys stable:", sorted(expected))


def test_source_ref_jsonl_keys_stable():
    expected = {"relative_path", "anchor"}
    payload = _make_section().to_jsonl()["source_ref"]
    assert set(payload.keys()) == expected, (
        f"key drift: extra={set(payload.keys()) - expected}, "
        f"missing={expected - set(payload.keys())}"
    )
    print("[PASS] SourceRef JSONL keys stable:", sorted(expected))


def test_asset_ref_jsonl_keys_stable():
    expected = {
        "id",
        "manual_id",
        "type",
        "path",
        "absolute_path",
        "referenced_by",
        "description",
    }
    payload = _make_asset().to_jsonl()
    assert set(payload.keys()) == expected, (
        f"key drift: extra={set(payload.keys()) - expected}, "
        f"missing={expected - set(payload.keys())}"
    )
    print("[PASS] AssetRef JSONL keys stable:", sorted(expected))


def test_manifest_json_keys_stable():
    expected = {
        "manual_id",
        "source_chm",
        "source_hash",
        "extracted_dir",
        "generated_at",
        "section_count",
        "asset_count",
        "parser_version",
    }
    payload = _make_manifest().to_json()
    assert set(payload.keys()) == expected, (
        f"key drift: extra={set(payload.keys()) - expected}, "
        f"missing={expected - set(payload.keys())}"
    )
    print("[PASS] Manifest JSON keys stable:", sorted(expected))


def test_image_ref_jsonl_keys_stable():
    expected = {"src", "alt", "asset_id"}
    payload = _make_section().to_jsonl()["images"][0]
    assert set(payload.keys()) == expected, (
        f"key drift: extra={set(payload.keys()) - expected}, "
        f"missing={expected - set(payload.keys())}"
    )
    print("[PASS] ImageRef JSONL keys stable:", sorted(expected))


def test_link_jsonl_keys_stable():
    expected = {"text", "href"}
    payload = _make_section().to_jsonl()["links"][0]
    assert set(payload.keys()) == expected, (
        f"key drift: extra={set(payload.keys()) - expected}, "
        f"missing={expected - set(payload.keys())}"
    )
    print("[PASS] Link JSONL keys stable:", sorted(expected))


# ---------- Main runner ----------

TESTS = [
    test_manual_section_roundtrip,
    test_manual_section_required_fields_present,
    test_manual_section_identity_decoupled_from_source_ref,
    test_manual_section_source_ref_anchor_optional,
    test_manual_section_rejects_missing_source_ref,
    test_manual_section_section_path_can_be_empty,
    test_asset_ref_roundtrip,
    test_asset_ref_required_fields,
    test_manifest_roundtrip,
    test_manifest_required_fields,
    test_manifest_parser_version_default,
    test_id_unchanged_when_path_changes,
    # JSON key stability
    test_manual_section_jsonl_keys_stable,
    test_source_ref_jsonl_keys_stable,
    test_asset_ref_jsonl_keys_stable,
    test_manifest_json_keys_stable,
    test_image_ref_jsonl_keys_stable,
    test_link_jsonl_keys_stable,
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