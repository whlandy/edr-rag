"""P1 smoke test — procedure extraction + local search.

Runs against the P0 smoke fixture and verifies:
  - extract_procedures_from_section returns procedures for proxy section
  - build_chunks returns section + procedure + step chunks
  - keyword_search and bm25_search return relevant results
"""

from pathlib import Path
import json
import shutil
import sys
import tempfile

sys.path.insert(0, "tools")

from edr_rag import normalize, extract as extract_mod  # noqa: E402
from edr_rag.p1 import extract_procedures as ep  # noqa: E402
from edr_rag.p1 import search  # noqa: E402


def _build_fixture(root: Path):
    extracted = root / "extracted_src"
    extracted.mkdir(parents=True)
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


def _patch_extract(fixture_dir: Path):
    def fake_extract(chm, out):
        out = Path(out)
        if out.exists():
            shutil.rmtree(out)
        shutil.copytree(fixture_dir, out)
        return ("sha256:fixture", True)
    extract_mod.extract_chm = fake_extract


def main():
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

            normalize.run_ingest(tmp / "fake.chm", "test-manual", "TestProduct", workdir)

            # Read sections from P0 output
            sections = [
                json.loads(line)
                for line in (workdir / "sections.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            by_path = {s["source_ref"]["relative_path"]: s for s in sections}

            # --- P1: extract procedures ---
            section_proxy = by_path["network/proxy.html"]
            procs = ep.extract_procedures_from_section(section_proxy, product="TestProduct")
            check("proxy section produces at least 1 procedure", len(procs) >= 1,
                  f"got {len(procs)}")
            if procs:
                p = procs[0]
                check("procedure has steps", len(p.steps) >= 3,
                      f"got {len(p.steps)} steps")
                check("procedure id is semantic", p.id.startswith("procedure."),
                      f"got {p.id}")
                # Semantic id should include TOC-derived components (中文 OK,
                # consistent with P0 manual.section.网络.代理)
                check("procedure id includes section_path semantic",
                      "代理" in p.id, f"got {p.id}")
                check("procedure has operation", len(p.operation) > 0,
                      f"got '{p.operation}'")
                check("procedure has status=candidate", p.status == "candidate",
                      f"got {p.status}")
                # Check step order matches HTML
                step_texts = [s.instruction for s in p.steps]
                check("step order: 打开网络设置",
                      step_texts[0].strip().startswith("打开"),
                      f"got {step_texts[0]}")
                check("step order: 选择代理",
                      len(step_texts) > 1 and step_texts[1].strip().startswith("选择"),
                      f"got {step_texts[1] if len(step_texts) > 1 else 'N/A'}")

            # index.html: not a procedure (just index text) — expect 0 procs
            section_index = by_path["network/index.html"]
            procs_index = ep.extract_procedures_from_section(section_index)
            check("index section yields 0 procedures (not a procedure)", len(procs_index) == 0,
                  f"got {len(procs_index)}")

            # --- P1: build chunks ---
            chunks = ep.build_chunks(section_proxy, procs)
            check("proxy section produces chunk count >= 5",
                  len(chunks) >= 5, f"got {len(chunks)}")  # 1 section + 1 procedure + 3 steps
            types = {c.type for c in chunks}
            check("chunk types include section", "section" in types)
            check("chunk types include procedure", "procedure" in types)
            check("chunk types include step", "step" in types)

            # --- P1: search ---
            corpus = search.JsonlCorpus(sections)
            kw_results = corpus.keyword_search("代理", limit=3)
            check("keyword search returns results for '代理'", len(kw_results) >= 1)
            bm25_results = corpus.bm25_search("代理", limit=3)
            check("bm25 search returns results for '代理'", len(bm25_results) >= 1)

            # --- P1: extract from file orchestrator ---
            all_procs, all_chunks = ep.extract_procedures_from_file(
                workdir / "sections.jsonl", product="TestProduct",
            )
            check("file-level extraction returns procedures", len(all_procs) >= 1)
            check("file-level extraction returns chunks", len(all_chunks) >= 1)

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[ERROR] Unexpected: {type(e).__name__}: {e}")
        failed += 1

    print(f"\n{passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
