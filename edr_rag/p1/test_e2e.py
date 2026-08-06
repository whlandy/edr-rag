"""End-to-end P1 CLI smoke test.

Runs the P0 ingest + P1 extract-procedures + P1 search via CLI,
verifying the full pipeline produces valid results.
"""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "tools")

from edr_rag import normalize, extract as extract_mod


def _fixture(root: Path):
    extracted = root / "extracted_src"
    extracted.mkdir(parents=True)
    hhc = """<HTML><BODY><UL>
<LI><OBJECT type="text/sitemap">
  <param name="Name" value="网络">
  <param name="Local" value="network/proxy.html">
</OBJECT></LI>
</UL></BODY></HTML>"""
    (extracted / "toc.hhc").write_text(hhc, encoding="utf-8")
    (extracted / "network").mkdir()
    (extracted / "network" / "proxy.html").write_text(
        """<html><head><title>配置代理服务器</title></head><body>
<h1>配置代理服务器</h1>
<p>要配置代理服务器，请按照以下步骤操作：</p>
<ol><li>打开网络设置</li><li>选择代理</li><li>输入代理地址</li></ol>
</body></html>""",
        encoding="utf-8",
    )
    (extracted / "images").mkdir()
    (extracted / "images" / "proxy.png").write_bytes(b"x")
    return extracted


def main():
    passed = 0
    failed = 0

    def check(name, cond, msg=""):
        nonlocal passed, failed
        if cond:
            print(f"[PASS] {name}")
            passed += 1
        else:
            print(f"[FAIL] {name}: {msg}")
            failed += 1

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        fixture = _fixture(tmp)

        def fake_extract(chm, out):
            out = Path(out)
            if out.exists():
                shutil.rmtree(out)
            shutil.copytree(fixture, out)
            return ("sha256:fixture", True)
        extract_mod.extract_chm = fake_extract

        workdir = tmp / "workdir"

        # P0 ingest
        normalize.run_ingest(tmp / "fake.chm", "test-manual", "TestProduct", workdir)
        check("P0 ingest produced sections.jsonl",
              (workdir / "sections.jsonl").exists())

        # P1 extract-procedures
        result = subprocess.run(
            [sys.executable, "-m", "edr_rag.p1.cli", "extract-procedures",
             "--workdir", str(workdir)],
            capture_output=True, text=True,
            cwd="tools",
        )
        check("P1 extract-procedures exit 0", result.returncode == 0,
              f"stderr: {result.stderr}")
        check("P1 procedures.jsonl exists",
              (workdir / "procedures.jsonl").exists())
        check("P1 chunks.jsonl exists",
              (workdir / "chunks.jsonl").exists())
        check("P1 p1_metadata.json exists",
              (workdir / "p1_metadata.json").exists())

        # Read output
        procs = [json.loads(l) for l in
                 (workdir / "procedures.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        check("P1 has at least 1 procedure", len(procs) >= 1,
              f"got {len(procs)}")
        if procs:
            check("P1 procedure has steps", len(procs[0].get("steps", [])) >= 3)

        chunks = [json.loads(l) for l in
                  (workdir / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        check("P1 has chunks", len(chunks) >= 4,
              f"got {len(chunks)}")

        # P1 search — keyword
        search_kw = subprocess.run(
            [sys.executable, "-m", "edr_rag.p1.cli", "search",
             "--query", "代理", "--type", "procedure", "--limit", "3",
             "--workdir", str(workdir)],
            capture_output=True, text=True,
            cwd="tools",
        )
        check("P1 search keyword exit 0", search_kw.returncode == 0,
              f"stderr: {search_kw.stderr}")
        check("P1 search keyword finds results",
              "Search results" in search_kw.stdout)

        # P1 search — bm25
        search_bm = subprocess.run(
            [sys.executable, "-m", "edr_rag.p1.cli", "search",
             "--query", "代理", "--type", "procedure", "--scoring", "bm25",
             "--limit", "3", "--workdir", str(workdir)],
            capture_output=True, text=True,
            cwd="tools",
        )
        check("P1 search bm25 exit 0", search_bm.returncode == 0,
              f"stderr: {search_bm.stderr}")

        # P1 search on sections (P0 artifact)
        search_sections = subprocess.run(
            [sys.executable, "-m", "edr_rag.p1.cli", "search",
             "--query", "代理", "--type", "section", "--limit", "3",
             "--workdir", str(workdir)],
            capture_output=True, text=True,
            cwd="tools",
        )
        check("P1 search sections exit 0", search_sections.returncode == 0,
              f"stderr: {search_sections.stderr}")

    print(f"\n{passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
