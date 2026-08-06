"""P2 smoke test — action extraction, search, and planner retrieval.

Requires P0+P1 to be smoketested first (P0 output + P1 procedures).
"""

from pathlib import Path
import json
import shutil
import sys
import tempfile

sys.path.insert(0, "tools")

from edr_rag import normalize, extract as extract_mod  # noqa: E402
from edr_rag.p1 import extract_procedures as ep  # noqa: E402
from edr_rag.p2 import extract_actions as ea  # noqa: E402
from edr_rag.p2 import api as papi  # noqa: E402


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
<LI><OBJECT type="text/sitemap">
  <param name="Name" value="重置">
  <param name="Local" value="reset.html">
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
    (extracted / "reset.html").write_text(
        """<html><head><title>重置系统</title></head><body>
<h1>重置系统</h1>
<p>危险操作！重置将清除所有数据。</p>
<ol><li>确认操作</li><li>输入管理员密码</li><li>等待重置完成</li></ol>
</body></html>""",
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

            # P0 ingestion
            normalize.run_ingest(tmp / "fake.chm", "test-manual", "TestProduct", workdir)

            # P1 extraction
            all_procs, all_chunks = ep.extract_procedures_from_file(
                workdir / "sections.jsonl", product="TestProduct",
            )
            ep.write_p1_artifacts(
                workdir, all_procs, all_chunks,
                [json.loads(l) for l in (workdir / "sections.jsonl").read_text('utf-8').splitlines() if l.strip()],
                manual_id="test-manual", product="TestProduct",
            )

            # --- P2: build actions ---
            entries = ea.build_actions_from_file(
                workdir / "procedures.jsonl", product="TestProduct",
            )
            check("action catalog has entries", len(entries) >= 1, f"got {len(entries)}")

            # Write P2 artifacts
            jsonl_path, yaml_path = ea.write_p2_artifacts(workdir, entries)
            check("actions.jsonl created", jsonl_path.exists())
            check("actions.yaml created", yaml_path.exists())

            # Read back
            with jsonl_path.open() as f:
                written = [json.loads(l) for l in f if l.strip()]
            check("actions.jsonl has correct count", len(written) == len(entries))

            # Check risk classification
            proxy_action = None
            reset_action = None
            for e in entries:
                if "代理" in e.action_id or "配置" in e.action_id or "proxy" in e.action_id:
                    proxy_action = e
                if "重置" in e.action_id or "reset" in e.action_id:
                    reset_action = e

            check("proxy action found", proxy_action is not None)
            if proxy_action:
                # "配置" keyword in title/path → risk=medium
                check("proxy risk is medium (contains 配置)", proxy_action.risk == "medium",
                      f"got {proxy_action.risk}")
                # medium risk → requires_confirmation=True
                check("proxy requires_confirmation is True (medium risk)",
                      proxy_action.requires_confirmation is True,
                      f"got {proxy_action.requires_confirmation}")
                # Proxy has 3 steps
                check("proxy has 3 steps", len(proxy_action.procedure) == 3,
                      f"got {len(proxy_action.procedure)}")
                # status should be candidate (medium risk but confidence >= 0.7 from 3 steps)
                check("proxy status is candidate", proxy_action.status == "candidate",
                      f"got {proxy_action.status}")

            check("reset action found", reset_action is not None)
            if reset_action:
                # "重置" keyword in title → risk=high
                check("reset risk is high (contains 重置)", reset_action.risk == "high",
                      f"got {reset_action.risk}")
                check("reset requires_confirmation is True",
                      reset_action.requires_confirmation is True,
                      f"got {reset_action.requires_confirmation}")
                check("reset has 3 steps", len(reset_action.procedure) == 3,
                      f"got {len(reset_action.procedure)} steps")
                check("status is needs_review (high risk)", reset_action.status == "needs_review",
                      f"got {reset_action.status}")

            # --- P2: API search ---
            api = papi.EdrRagAPI(workdir)
            action_results = api.action_search("配置代理", limit=5)
            check("action search returns results", len(action_results) >= 1,
                  f"got {len(action_results)}")

            # Planner retrieval
            planner_result = api.planner_retrieve("配置代理", limit=5)
            check("planner retrieval has actions", len(planner_result.get("actions", [])) >= 1,
                  f"got {len(planner_result.get('actions', []))}")

            # Get action by (Chinese) action_id — use the actual id from the entry
            if proxy_action:
                actual_id = proxy_action.action_id  # e.g. "网络.代理.配置代理服务器"
                retrieved = api.get_action(actual_id)
                check("get_action returns by action_id", retrieved is not None)
                if retrieved:
                    check("get_action returns correct id",
                          retrieved.get("action_id") == actual_id)

            # Planner retrieve with non-action query (should fall back to procedures)
            no_action_result = api.planner_retrieve("安全状态", limit=5)
            # Proxy action might match too, but at least check structure
            check("planner retrieval structure OK",
                  "actions" in no_action_result and "procedures" in no_action_result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[ERROR] Unexpected: {type(e).__name__}: {e}")
        failed += 1

    print(f"\n{passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
