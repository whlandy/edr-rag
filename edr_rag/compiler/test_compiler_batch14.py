"""P6：冻结样例 + 正负例汇总（design.md §10 P6）。

replay_frozen_sample 离线复放完整链路（真实 SKILL.md 快照 → 用例导入 →
compile 四阶段 → validate-plan 复验 → render-guide 同源 → 审核包 →
export-bundle + manifest 复核 → 人工 approve → effective_approval 门禁命中），
并输出量化覆盖报告。

正例：
- 全链 ok，各阶段产物 hash 链一致（validate/render/pack 绑定同一
  workflow_hash）；
- 复放确定性：同参数两次 replay 的 chain 关键字段与量化报告一致；
- 量化覆盖：原步骤 3/3 覆盖、证据行全部可解析、无 unsupported；
- 门禁：save_review created + effective_approval=True。

负例（对 replay 产物的拒绝路径复验，unknown 不算通过）：
- 重新编译（不同 workflow_id → hash 漂移）后旧批准对 新hash+同pack 不命中
  （批准绑定对象 hash，不绑定名义 id）；
- review 缺 pack 提供 → verify_review_decision 判 pending_pack，
  effective=False（只凭 workflow hash 不宣布批准有效）；
- SKILL.md 缺失 → replay ok=False failed_stage=capability_snapshot
  （fixture 依赖真实来源快照，不允许伪造）。

全部为离线确定性验证；无实机执行。replay ok ≠ 实机 passed；
effective_approval=True ≠ published（编译器不自授，§8）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from edr_rag.compiler.frozen_replay import (
    FROZEN_REPLAY_SCHEMA_VERSION,
    frozen_case,
    frozen_evidence,
    replay_frozen_sample,
)
from edr_rag.compiler.review_decision import ReviewDecision, verify_review_decision

BROWSER_SKILL = Path("/Users/edr-test/.hermes/skills/browser-automation/SKILL.md")
browser_skill = pytest.mark.skipif(
    not BROWSER_SKILL.is_file(), reason="browser SKILL.md not present"
)

T = "2026-11-08T09:00:00+00:00"


@browser_skill
class TestFrozenReplayPositive:
    def test_full_chain_ok_and_hash_chain_consistent(self, tmp_path: Path):
        report = replay_frozen_sample(tmp_path, exported_at=T)
        assert report["ok"] is True, report["error"]
        assert report["failed_stage"] is None
        chain = report["chain"]
        # 阶段链完整
        assert list(chain) == [
            "capability_snapshot", "case_import", "compile",
            "validate_plan", "render_guide", "audit_pack",
            "export_bundle", "review_gate",
        ]
        wf_hash = chain["compile"]["workflow_hash"]
        assert chain["validate_plan"]["workflow_hash"] == wf_hash
        assert chain["validate_plan"]["hash_stable"] is True
        assert chain["render_guide"]["workflow_hash"] == wf_hash
        # §7/§8 分层：draft + not_run + runtime_ready=False（变量未绑定值）
        assert chain["compile"]["status"] == "draft"
        assert chain["compile"]["execution_status"] == "not_run"
        assert chain["compile"]["runtime_ready"] is False
        assert chain["compile"]["unbound_variables"] == ["server_address"]
        # bundle 导出且 manifest 复核通过；approval 停在 pending_human_review
        assert chain["export_bundle"]["manifest_verified"] is True
        assert chain["export_bundle"]["approval"] == "pending_human_review"
        # 门禁：approve 落库且命中
        assert chain["review_gate"]["save_status"] == "created"
        assert chain["review_gate"]["effective_approval"] is True

    def test_replay_deterministic(self, tmp_path: Path):
        r1 = replay_frozen_sample(tmp_path / "a", exported_at=T)
        r2 = replay_frozen_sample(tmp_path / "b", exported_at=T)
        assert r1["ok"] and r2["ok"]
        # 同参数复放：hash 链与量化报告一致（root 路径除外）
        a, b = r1["chain"], r2["chain"]
        assert a["compile"]["workflow_hash"] == b["compile"]["workflow_hash"]
        assert a["audit_pack"]["pack_hash"] == b["audit_pack"]["pack_hash"]
        assert a["capability_snapshot"]["content_hash"] == b["capability_snapshot"]["content_hash"]
        assert r1["quantitative"] == r2["quantitative"]

    def test_quantitative_coverage(self, tmp_path: Path):
        report = replay_frozen_sample(tmp_path, exported_at=T)
        q = report["quantitative"]
        assert q["original_steps"] == 3
        assert q["steps_covered_by_atomic"] == 3
        assert q["key_expected_covered"] == 3
        assert q["evidence_rows_total"] == q["evidence_rows_resolved"]
        assert q["evidence_rows_resolved"] > 0
        assert q["unsupported_steps"] == 0
        assert q["provider_stub"] is True  # stub 显式标记，不冒充真实模型
        # 量化门槛：关键预期/来源覆盖 100%
        gate = report["publish_candidate_quantitative_gate"]
        assert gate["key_expected_coverage_100pct"] is True
        assert gate["source_coverage_100pct"] is True
        # 审核包 gaps 只含 §7 分层如实呈现（未绑定值），无其他未解决缺口
        assert set(q["unresolved_gaps"]) <= {"binding_state_unknown"}

    def test_case_fixture_sanitized(self):
        case = frozen_case()
        assert case.product_version == "web-testbed-1.0"
        assert case.source_ref == "case-frozen-web-001@rev1:" + "c" * 16
        # 无真实业务值：变量只出现在语义文本中，fixture 不携带可执行参数
        blob = json.dumps([dict(s) for s in case.steps], ensure_ascii=False)
        assert "http" not in blob and "192.168" not in blob
        evs = frozen_evidence()
        assert all(e.source_ref == case.source_ref for e in evs)
        assert {e.purpose for e in evs} == {"operation", "expected"}


@browser_skill
class TestFrozenReplayNegative:
    def test_recompile_new_id_old_approval_not_hit(self, tmp_path: Path):
        from edr_rag.knowledge.connector import CaseBundle
        from edr_rag.knowledge.models import EvidenceRef
        from edr_rag.knowledge.repository import KnowledgeRepository
        from edr_rag.compiler import OfflineStubProvider
        from edr_rag.compiler.capability_snapshot import collect_browser_snapshot
        from edr_rag.compiler.pipeline import compile_workflow
        from edr_rag.compiler.review_store import effective_approval

        report = replay_frozen_sample(tmp_path / "run", exported_at=T)
        assert report["ok"], report["error"]
        old_hash = report["chain"]["compile"]["workflow_hash"]
        old_pack_hash = report["chain"]["audit_pack"]["pack_hash"]

        # 重新编译（不同 workflow_id → 内容/ hash 漂移）
        repo = KnowledgeRepository(tmp_path / "run" / "kb")
        case = frozen_case()
        snap, _ = collect_browser_snapshot(
            product_version=case.product_version,
            snapshot_id="snap-frozen-web-001", observed_at=T,
        )
        out = compile_workflow(
            repo=repo, case=case,
            understanding_provider=OfflineStubProvider(),
            decomposition_provider=OfflineStubProvider(mode="decomposition"),
            snapshot=snap, workflow_id="wf-frozen-web-002",
            compiled_at=T,
        )
        assert out["ok"]
        new_hash = out["workflow_hash"]
        assert new_hash != old_hash
        # 旧批准对新对象不命中（批准绑定 hash，不绑定名义 id）
        eff = effective_approval(
            repo.root / "reviews.jsonl", new_hash, old_pack_hash
        )
        assert eff["approved"] is False

    def test_review_without_pack_is_pending_not_effective(self, tmp_path: Path):
        report = replay_frozen_sample(tmp_path, exported_at=T)
        assert report["ok"]
        wf_hash = report["chain"]["compile"]["workflow_hash"]
        # 包未随记录提供 → pending_pack，effective=False（unknown 不算通过）
        rec = ReviewDecision(
            reviewer="x", method="human_local", decided_at=T,
            workflow_hash=wf_hash, pack_hash="a" * 64,
            decision="approve", rationale="r", decision_id="rd-nopack",
        )
        from edr_rag.knowledge.models import WorkflowRevision
        # 无真实 workflow 对象时 pack 必须为 None；verify 拒绝宣布有效
        class _W:  # 最小桩：verify 只读 content_hash
            content_hash = wf_hash
        verdict = verify_review_decision(rec, _W(), None)
        assert verdict["effective"] is False
        assert any(d["code"] == "pack_not_provided" for d in verdict["diagnostics"])

    def test_missing_skill_md_fails_honestly(self, tmp_path: Path):
        from edr_rag.compiler import frozen_replay as fr

        orig = fr.collect_browser_snapshot

        def boom(**kw):
            from edr_rag.knowledge.models import ModelError
            raise ModelError("browser SKILL.md not found: /nonexistent")

        fr.collect_browser_snapshot = boom
        try:
            report = replay_frozen_sample(tmp_path, exported_at=T)
        finally:
            fr.collect_browser_snapshot = orig
        assert report["ok"] is False
        assert report["failed_stage"] == "capability_snapshot"
        assert "SKILL.md" in report["error"]
