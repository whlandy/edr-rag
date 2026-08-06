"""Smoke test for P0 CHM ingest pipeline.

Builds a synthetic CHM-like directory structure, patches the CHM extractor
to copy it instead of running hh.exe, then runs the full normalize pipeline
and verifies:

  - manifest.json has correct counts
  - sections.jsonl has correct sections with semantic ids (not path hashes)
  - assets.jsonl has correct asset refs with proper referenced_by
  - validate_workdir passes with 0 errors
  - CLI validate --workdir passes
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "tools")

from edr_rag import normalize, extract as extract_mod  # noqa: E402


def _build_fixture(root: Path) -> Path:
    """Build a synthetic CHM-like extracted directory tree."""
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
  <param name="Name" value="安全状态">
  <param name="Local" value="status.html">
</OBJECT></LI>
</UL></BODY></HTML>"""
    (extracted / "toc.hhc").write_text(hhc, encoding="utf-8")
    (extracted / "network").mkdir()
    (extracted / "network" / "index.html").write_text(
        "<html><head><title>网络首页</title></head><body>"
        "<h1>网络首页</h1><p>网络配置索引</p>"
        "</body></html>",
        encoding="utf-8",
    )
    (extracted / "network" / "proxy.html").write_text(
        """<html><head><title>配置代理服务器</title></head><body>
<h1>配置代理服务器</h1>
<p>要配置代理服务器，请按照以下步骤操作：</p>
<ol><li>打开网络设置</li><li>选择代理</li><li>输入代理地址</li></ol>
<img src="../images/proxy.png" alt="代理配置图">
<a href="https://example.com/docs">外部文档</a>
</body></html>""",
        encoding="utf-8",
    )
    (extracted / "status.html").write_text(
        "<html><head><title>查看安全状态</title></head><body>"
        "<h1>查看安全状态</h1>"
        "<p>安全状态页面</p>"
        "</body></html>",
        encoding="utf-8",
    )
    (extracted / "images").mkdir()
    (extracted / "images" / "proxy.png").write_bytes(b"fakepng")
    return extracted


def _patch_extract(fixture_dir: Path) -> None:
    """Replace extract_chm to copy our fixture instead of running hh.exe."""
    def fake_extract(chm: Path, out: Path) -> tuple[str, bool]:
        out = Path(out)
        if out.exists():
            shutil.rmtree(out)
        shutil.copytree(fixture_dir, out)
        return ("sha256:fixture", True)
    extract_mod.extract_chm = fake_extract


def main() -> int:
    passed = 0
    failed = 0

    def check(name, condition, msg=""):
        nonlocal passed, failed
        if condition:
            print(f"[PASS] {name}")
            passed += 1
        else:
            print(f"[FAIL] {name}: {msg}")
            failed += 1

    try:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            fixture = _build_fixture(tmp)
            _patch_extract(fixture)

            workdir = tmp / "workdir"
            manifest = normalize.run_ingest(
                chm_path=tmp / "fake.chm",
                manual_id="test-manual",
                product="TestProduct",
                workdir=workdir,
            )

            # --- Manifest checks ---
            check("manifest section_count", manifest.section_count == 3,
                  f"got {manifest.section_count}")
            check("manifest asset_count", manifest.asset_count == 1,
                  f"got {manifest.asset_count}")
            check("manifest manifest.json exists", (workdir / "manifest.json").exists())
            check("manifest sections.jsonl exists", (workdir / "sections.jsonl").exists())
            check("manifest assets.jsonl exists", (workdir / "assets.jsonl").exists())

            # --- Sections checks ---
            sections_text = (workdir / "sections.jsonl").read_text(encoding="utf-8")
            sections = [
                json.loads(l)
                for l in sections_text.splitlines()
                if l.strip()
            ]
            check("sections count == 3", len(sections) == 3, f"got {len(sections)}")

            # Build lookup by source_ref.relative_path
            by_path = {s["source_ref"]["relative_path"]: s for s in sections}

            # proxy.html: should have TOC section_path ["网络", "代理"],
            # semantic id "manual.section.网络.代理"
            proxy = by_path["network/proxy.html"]
            check("proxy section_path", proxy["section_path"] == ["网络", "代理"],
                  f"got {proxy['section_path']}")
            check("proxy id is semantic (not path)", "proxy.html" not in proxy["id"],
                  f"got {proxy['id']}")
            check("proxy has images", len(proxy["images"]) == 1,
                  f"got {len(proxy['images'])}")
            check("proxy image asset_id",
                  proxy["images"][0]["asset_id"] == "asset.images.proxy.png",
                  f"got {proxy['images'][0].get('asset_id')}")
            check("proxy source_ref has relative_path",
                  "relative_path" in proxy["source_ref"],
                  "source_ref missing relative_path")
            check("proxy has content", len(proxy["content"]) > 0,
                  "empty content")

            # index.html: TOC path ["网络"]
            index = by_path["network/index.html"]
            check("index section_path", index["section_path"] == ["网络"],
                  f"got {index['section_path']}")
            check("index id", index["id"] == "manual.section.网络",
                  f"got {index['id']}")

            # status.html: TOC path ["安全状态"], no nested
            status = by_path["status.html"]
            check("status section_path", status["section_path"] == ["安全状态"],
                  f"got {status['section_path']}")
            check("status id", status["id"] == "manual.section.安全状态",
                  f"got {status['id']}")

            # --- Asset checks ---
            assets_text = (workdir / "assets.jsonl").read_text(encoding="utf-8")
            assets = [
                json.loads(l)
                for l in assets_text.splitlines()
                if l.strip()
            ]
            check("assets count == 1", len(assets) == 1, f"got {len(assets)}")
            check("asset type=image", assets[0]["type"] == "image",
                  f"got {assets[0]['type']}")
            check("asset path has proxy.png", "proxy.png" in assets[0]["path"],
                  f"got {assets[0]['path']}")
            check("asset referenced_by includes proxy.html",
                  "network/proxy.html" in assets[0]["referenced_by"],
                  f"got {assets[0]['referenced_by']}")

            # --- Validate workdir ---
            section_count, asset_count, errors = normalize.validate_workdir(workdir)
            check("validate_workdir errors == 0", errors == 0, f"got {errors}")
            check("validate_workdir section_count matches", section_count == 3,
                  f"got {section_count}")
            check("validate_workdir asset_count matches", asset_count == 1,
                  f"got {asset_count}")

            # --- CLI validate ---
            result = subprocess.run(
                [sys.executable, "-m", "edr_rag.cli", "validate", "--workdir", str(workdir)],
                capture_output=True, text=True,
                cwd="tools",
            )
            check("CLI validate exit code == 0", result.returncode == 0,
                  f"exit {result.returncode}, stderr: {result.stderr.strip()}")
            check("CLI validate output says PASSED",
                  "PASSED" in result.stdout or result.returncode == 0,
                  f"stdout: {result.stdout.strip()}")

    except Exception as e:
        print(f"[ERROR] Unexpected: {type(e).__name__}: {e}")
        failed += 1

    print(f"\n{passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
