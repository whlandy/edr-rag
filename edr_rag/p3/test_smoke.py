"""P3 smoke test — feedback + MCP server logic.

Tests the feedback store and EdrRagAgent (without actual MCP transport).
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
from edr_rag.p3.feedback import FeedbackStore, FeedbackItem, FeedbackEvidence, SuggestedPatch  # noqa: E402
from edr_rag.p3.server import EdrRagAgent  # noqa: E402


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

            # Build P0 → P2 artifacts
            normalize.run_ingest(tmp / "fake.chm", "test-manual", "TestProduct", workdir)
            all_procs, all_chunks = ep.extract_procedures_from_file(
                workdir / "sections.jsonl", product="TestProduct",
            )
            sections = [json.loads(l) for l in (workdir / "sections.jsonl").read_text('utf-8').splitlines() if l.strip()]
            ep.write_p1_artifacts(workdir, all_procs, all_chunks, sections, manual_id="test-manual", product="TestProduct")
            entries = ea.build_actions_from_file(workdir / "procedures.jsonl", product="TestProduct")
            ea.write_p2_artifacts(workdir, entries)

            # --- P3: FeedbackStore ---
            store = FeedbackStore(str(workdir / "feedback.jsonl"))

            # Append a feedback item
            item = FeedbackItem(
                feedback_id="fb_001",
                manual_id="test-manual",
                action_id="网络.代理.配置代理服务器",
                failure_type="button_text_not_found",
                expected="保存",
                observed="保存配置",
                evidence=FeedbackEvidence(
                    screenshot="runs/run_001/screenshot.png",
                    dump_tree="runs/run_001/dump_tree.json",
                    error_message="Button '保存' not found, did you mean '保存配置'?",
                ),
                suggested_patch=SuggestedPatch(
                    patch_type="selector_synonyms",
                    field="step_2_target",
                    old_value="保存",
                    new_value="保存配置",
                ),
                created_at="2026-08-05T12:00:00+08:00",
            )
            store.append(item)
            check("feedback store append works", True)

            # Read all
            items = store.read_all()
            check("feedback store read_all returns items", len(items) == 1)
            check("feedback item has action_id",
                  items[0].action_id == "网络.代理.配置代理服务器")
            check("feedback item has evidence screenshot",
                  items[0].evidence is not None and items[0].evidence.screenshot is not None)
            check("feedback item has suggested_patch",
                  items[0].suggested_patch is not None)

            # Update status
            ok = store.update_status("fb_001", "needs_review")
            check("feedback update status returns True", ok)
            items = store.read_all()
            check("feedback status updated", items[0].status == "needs_review")

            # --- P3: Agent ---
            agent = EdrRagAgent(str(workdir))

            # Knowledge search
            section_results = agent.search_sections("代理", limit=3)
            check("agent search_sections returns results", len(section_results) >= 1)

            proc_results = agent.search_procedures("代理", limit=3)
            check("agent search_procedures returns results", len(proc_results) >= 1)

            action_results = agent.search_actions("代理", limit=3)
            check("agent search_actions returns results", len(action_results) >= 1)

            # Get action
            action_data = agent.get_action("网络.代理.配置代理服务器")
            check("agent get_action returns data", action_data is not None)
            if action_data:
                check("agent get_action has risk", "risk" in action_data)

            # Get procedure
            if all_procs:
                proc_id = all_procs[0].id
                proc_data = agent.get_procedure(proc_id)
                check("agent get_procedure returns data", proc_data is not None)

            # Capture feedback
            fb = {
                "manual_id": "test-manual",
                "action_id": "网络.代理.配置代理服务器",
                "failure_type": "timeout",
                "expected": "配置成功",
                "observed": "超时",
                "evidence": {"error_message": "Timeout after 30s waiting for dialog"},
            }
            fb_id = agent.capture_feedback(fb)
            check("agent capture_feedback returns id", len(fb_id) > 0)

            # Classify failure
            classified = agent.classify_failure(fb_id)
            check("agent classify_failure returns True", classified)

            # Propose patch (no patch for new item without suggested_patch)
            patch = agent.propose_patch(fb_id)
            check("agent propose_patch returns None (no patch data)", patch is None)

            # Propose patch for item with patch data
            patch2 = agent.propose_patch("fb_001")
            check("agent propose_patch for fb_001 returns patch", patch2 is not None)
            if patch2:
                check("proposed patch has patch_type", patch2.get("patch_type") == "selector_synonyms")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[ERROR] Unexpected: {type(e).__name__}: {e}")
        failed += 1

    print(f"\n{passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
