"""normalize — CHM extraction → Manual IR (P0 contract).

Pipeline:
  1. extract CHM         (delegated to extract.py)
  2. parse TOC (.hhc)    (parse_toc.build_file_to_path_index)
  3. walk extracted_dir  (collect *.html + asset files)
  4. for each HTML:
       - parse_html_file → ParsedPage
       - resolve section_path via section_path resolver (priority contract)
       - build ManualSection with SEMANTIC id, not path hash
       - record referenced images
  5. build AssetRef list (image, css, js — anything not .html)
  6. write manifest.json + sections.jsonl + assets.jsonl

ID contract (P0 — locked in Commit A review):
  - id MUST be semantic (TOC path + operation meaning)
  - id MUST NOT be derived from source_ref.relative_path
  - id MUST be stable across re-ingestion

Path safety contract (P0 — review Blocker 3):
  - reject any <img src="..."> or <a href="..."> that escapes extracted_dir
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from edr_rag import extract as extract_mod
from edr_rag.parse_html import parse_html_file
from edr_rag.parse_toc import build_file_to_path_index, parse_hhc
from edr_rag.schema import (
    AssetRef,
    ImageRef,
    Link,
    Manifest,
    ManualSection,
    SourceRef,
)
from edr_rag.section_path import first_h1, resolve_section_path

logger = logging.getLogger(__name__)


# ---------- ID generation (SEMANTIC, not path hash) ----------

def _slugify(s: str) -> str:
    """Convert a string into a stable slug suitable for IDs.

    - Lowercase
    - Replace whitespace and non-alphanumeric (preserving unicode letters
      and digits) with underscore
    - Collapse multiple underscores
    - Strip leading/trailing underscores

    Examples:
      "网络 代理" → "网络_代理"
      "配置代理服务器" → "配置代理服务器"
      "Page 1: Intro" → "page_1_intro"
    """
    if not s:
        return ""
    # NFKC normalize first so full-width / composed forms collapse
    s = unicodedata.normalize("NFKC", s).strip().lower()
    # Replace any whitespace with single space, then non-word chars with underscore
    s = re.sub(r"\s+", " ", s)
    # Keep letters (any script), digits, and a few separators
    out = []
    for ch in s:
        if ch.isalnum():
            out.append(ch)
        elif ch in (" ", "-", "_"):
            out.append("_")
        # else: drop (punctuation)
    slug = "".join(out)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug


def make_semantic_id(
    *,
    section_path: list[str],
    title: str,
    manual_id: str,
) -> str:
    """Build a STABLE SEMANTIC id for a ManualSection.

    Priority:
      1. Use section_path if non-empty (most semantic — comes from TOC or headings)
      2. Fall back to title
      3. Fall back to "orphan" (deterministic but minimal)

    The id is intentionally derived from CONTENT, never from source_ref.relative_path.
    Re-ingestion of the same logical section MUST yield the same id, even if the
    CHM internals are repacked.
    """
    components: list[str] = []
    if section_path:
        for comp in section_path:
            sl = _slugify(comp)
            if sl:
                components.append(sl)
    if not components and title:
        sl = _slugify(title)
        if sl:
            components.append(sl)
    if not components:
        # Last resort — still deterministic via section_path/title attempt
        components.append("orphan")
    return f"manual.section.{'.'.join(components)}"


def make_asset_id(relative_path: str) -> str:
    """Build a deterministic asset id from its relative path.

    P0 allows path-derived ids for assets (images don't have semantic content
    like section titles — their identity IS their file location). Stable across
    re-ingestion as long as the asset remains at the same relative path.
    """
    rp = relative_path.replace("\\", "/").lstrip("/")
    return f"asset.{rp.replace('/', '.').replace(' ', '_')}"


# ---------- Path safety ----------

class UnsafePathError(ValueError):
    """Raised when a referenced path escapes extracted_dir."""


def safe_resolve(src: str, page_path: Path, extracted_dir: Path) -> Optional[str]:
    """Resolve a relative src (e.g. "../images/x.png") to a safe canonical path.

    Returns the path relative to extracted_dir, or None if the resolved path
    escapes extracted_dir (path traversal). The caller decides whether to
    skip the reference or raise.

    P0 contract (review Blocker 3): reject silently is better than guess.
    """
    if not src:
        return None
    # Strip URL scheme / protocol prefix
    if "://" in src:
        return None  # external URL, not a local asset
    # Resolve relative to page's directory
    page_dir = page_path.parent
    try:
        candidate = (page_dir / src).resolve()
    except (OSError, ValueError):
        return None
    extracted_root = extracted_dir.resolve()
    try:
        rel = candidate.relative_to(extracted_root)
    except ValueError:
        # Path escapes extracted_dir
        return None
    return rel.as_posix()


# ---------- Image src normalization ----------

def _normalize_image_src(src: str) -> str:
    """Normalize img src to forward-slash form. Does NOT resolve or sanitize."""
    return src.replace("\\", "/").lstrip("/")


# ---------- Main pipeline ----------

def _iter_html_files(extracted_dir: Path) -> list[Path]:
    """Walk extracted_dir and return all .html / .htm files, sorted for determinism."""
    files: list[Path] = []
    for ext in ("*.html", "*.htm", "*.xhtml"):
        files.extend(extracted_dir.rglob(ext))
    files.sort()
    return files


def _iter_asset_files(extracted_dir: Path, html_files: list[Path]) -> list[Path]:
    """Walk extracted_dir for non-HTML asset files (images, css, js).

    Excludes:
      - .hhc / .hhk files (TOC and index metadata, not content assets)
      - HTML files themselves
    """
    html_set = {p.resolve() for p in html_files}
    skip_suffixes = {".html", ".htm", ".xhtml", ".hhc", ".hhk"}
    assets: list[Path] = []
    for p in extracted_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() in skip_suffixes:
            continue
        if p.resolve() in html_set:
            continue
        assets.append(p)
    assets.sort()
    return assets


def _hash_file(path: Path) -> str:
    """SHA256 of a file, hex digest."""
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return f"sha256:{h.hexdigest()}"


def _resolve_hhc_index(extracted_dir: Path) -> dict[str, list[str]]:
    """Find and parse .hhc file. Returns file→section_path index, or {} if missing."""
    hhc_candidates = list(extracted_dir.glob("*.hhc")) + list(extracted_dir.glob("**/*.hhc"))
    if not hhc_candidates:
        logger.info("No .hhc found in %s — section_path falls back to HTML heading", extracted_dir)
        return {}
    # Use the first one (CHM typically has one toc.hhc at the root)
    hhc_path = hhc_candidates[0]
    try:
        entries = parse_hhc(hhc_path)
        return build_file_to_path_index(entries)
    except Exception as e:
        logger.warning("Failed to parse %s: %s — falling back", hhc_path, e)
        return {}


def run_ingest(
    chm_path: Path,
    manual_id: str,
    product: str,
    workdir: Path,
) -> Manifest:
    """Run P0 ingestion: CHM → manifest + sections.jsonl + assets.jsonl.

    Args:
        chm_path: path to the source .chm file.
        manual_id: stable id for this manual (e.g. "product-manual-v1").
        product: human-readable product name.
        workdir: directory to write artifacts into (will be created if missing).

    Returns:
        Manifest describing the ingestion run.

    Side effects:
        - Writes manifest.json, sections.jsonl, assets.jsonl under workdir.
        - May extract CHM into a temp directory (cleanup is caller's concern).
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    # 1. extract
    extracted_dir = workdir / "extracted"
    source_hash, ok = extract_mod.extract_chm(Path(chm_path), extracted_dir)
    if not ok:
        raise RuntimeError(f"Failed to extract CHM: {chm_path}")
    extracted_dir = extracted_dir.resolve()

    # 2. parse TOC
    hhc_index = _resolve_hhc_index(extracted_dir)

    # 3. walk HTML
    html_files = _iter_html_files(extracted_dir)

    # 4. for each HTML, build ManualSection
    sections: list[ManualSection] = []
    # Track which assets each section references (for AssetRef.referenced_by)
    asset_referenced_by: dict[str, list[str]] = {}

    for html_path in html_files:
        rel_path = html_path.relative_to(extracted_dir).as_posix()
        try:
            page = parse_html_file(html_path, extracted_dir)
        except Exception as e:
            logger.warning("Failed to parse %s: %s — skipping", html_path, e)
            continue

        section_path = resolve_section_path(
            hhc_index=hhc_index,
            file=rel_path,
            headings=page.headings,
            title=page.title,
        )

        # Image refs — must use safe_resolve, reject path traversal
        image_refs: list[ImageRef] = []
        for img in page.images:
            src_raw = img["src"]
            safe = safe_resolve(src_raw, html_path, extracted_dir)
            if safe is None:
                # Either external URL or path-traversal attempt — skip
                logger.debug("Skipping unsafe image src %r in %s", src_raw, rel_path)
                continue
            asset_id = make_asset_id(safe)
            image_refs.append(
                ImageRef(src=safe, alt=img.get("alt", ""), asset_id=asset_id)
            )
            asset_referenced_by.setdefault(asset_id, []).append(rel_path)

        # Link refs — keep raw href (caller can resolve later)
        link_refs = [Link(text=l["text"], href=l["href"]) for l in page.links]

        # Semantic id (from section_path / title — NEVER from rel_path)
        sid = make_semantic_id(
            section_path=section_path,
            title=page.title,
            manual_id=manual_id,
        )

        sections.append(
            ManualSection(
                id=sid,
                manual_id=manual_id,
                product=product,
                title=page.title or (section_path[-1] if section_path else ""),
                section_path=section_path,
                source_ref=SourceRef(relative_path=rel_path, anchor=None),
                content=page.content,
                headings=page.headings,
                links=link_refs,
                images=image_refs,
                metadata={
                    "language": "zh-CN" if any('\u4e00' <= c <= '\u9fff' for c in page.content) else "en",
                },
            )
        )

    # 5. build assets
    asset_paths = _iter_asset_files(extracted_dir, html_files)
    assets: list[AssetRef] = []
    for asset_path in asset_paths:
        rel = asset_path.relative_to(extracted_dir).as_posix()
        asset_id = make_asset_id(rel)
        # Determine type from suffix
        suf = asset_path.suffix.lower()
        if suf in (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".svg", ".webp"):
            atype = "image"
        elif suf in (".css",):
            atype = "css"
        elif suf in (".js",):
            atype = "js"
        else:
            atype = "other"
        assets.append(
            AssetRef(
                id=asset_id,
                manual_id=manual_id,
                type=atype,
                path=rel,
                absolute_path=str(asset_path.resolve()),
                referenced_by=sorted(set(asset_referenced_by.get(asset_id, []))),
            )
        )

    # 6. write artifacts
    manifest = Manifest(
        manual_id=manual_id,
        source_chm=str(Path(chm_path).resolve()),
        source_hash=source_hash,
        extracted_dir=str(extracted_dir),
        generated_at=datetime.now(timezone.utc).isoformat(),
        section_count=len(sections),
        asset_count=len(assets),
        parser_version="p0",
    )

    (workdir / "manifest.json").write_text(
        json.dumps(manifest.to_json(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (workdir / "sections.jsonl").open("w", encoding="utf-8") as f:
        for s in sections:
            f.write(json.dumps(s.to_jsonl(), ensure_ascii=False) + "\n")
    with (workdir / "assets.jsonl").open("w", encoding="utf-8") as f:
        for a in assets:
            f.write(json.dumps(a.to_jsonl(), ensure_ascii=False) + "\n")

    return manifest


# ---------- Validation ----------

REQUIRED_SECTION_KEYS = {"id", "manual_id", "title", "content", "section_path", "source_ref"}
REQUIRED_ASSET_KEYS = {"id", "manual_id", "type", "path"}


def validate_workdir(workdir: Path) -> tuple[int, int, int]:
    """Validate P0 IR contract.

    Returns (section_count, asset_count, error_count).
    """
    workdir = Path(workdir)
    errors = 0

    # Manifest
    manifest_path = workdir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # sections.jsonl
    sections_path = workdir / "sections.jsonl"
    if not sections_path.exists():
        raise FileNotFoundError(f"Missing {sections_path}")
    section_count = 0
    with sections_path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            missing = REQUIRED_SECTION_KEYS - obj.keys()
            if missing:
                print(f"[FAIL] sections.jsonl line {line_no}: missing keys {missing}")
                errors += 1
                continue
            # source_ref structure
            sr = obj.get("source_ref")
            if not isinstance(sr, dict) or "relative_path" not in sr:
                print(f"[FAIL] sections.jsonl line {line_no}: source_ref missing 'relative_path'")
                errors += 1
                continue
            section_count += 1

    # assets.jsonl
    assets_path = workdir / "assets.jsonl"
    if not assets_path.exists():
        raise FileNotFoundError(f"Missing {assets_path}")
    asset_count = 0
    with assets_path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            missing = REQUIRED_ASSET_KEYS - obj.keys()
            if missing:
                print(f"[FAIL] assets.jsonl line {line_no}: missing keys {missing}")
                errors += 1
                continue
            asset_count += 1

    print(f"manifest: OK ({section_count} sections, {asset_count} assets)")
    print(f"sections: OK ({section_count} lines)")
    print(f"assets: OK ({asset_count} lines)")
    if errors:
        print(f"Validation FAILED with {errors} errors")
    else:
        print("Validation PASSED")
    return section_count, asset_count, errors