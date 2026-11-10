"""P5 第三批次：ReviewDecision 人工审核记录模型（design.md §4 / §8 / §9）。

正例：
- 完整链路（真实 SKILL.md 快照 → Mapping → Validation → 渲染 → 审核包）→
  人工 approve 记录 → verify 有效（effective=True）；
- reject / needs_changes 记录同样绑定对象 hash，verify 有效但 effective=False。

负例（hash 绑定 / 作废契约）：
- workflow 修改后 → 作废（workflow_hash_drift）；
- 审核包内容篡改 → 作废（pack_hash_mismatch）；
- 记录签发于另一个包 → record_pack_hash_mismatch；
- pack 未提供 → pending_pack（unknown 不算通过，不得宣布批准有效）；
- 构造负例：空 reviewer / 空理由 / 非法决策枚举 / 非法 method / 缺 tz 时间戳
  → ModelError；from_dict 未知字段 / 缺字段 → ModelError。

全部为离线确定性验证；无实机执行。approve ≠ 实机 passed。
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from edr_rag.compiler import MappingStage, RenderingStage, ValidationStage
from edr_rag.compiler.audit_pack import AuditPackBuilder
from edr_rag.compiler.capability_snapshot import collect_browser_snapshot
from edr_rag.compiler.review_decision import (
    REVIEW_SCHEMA_VERSION,
    ReviewDecision,
    verify_review_decision,
)
from edr_rag.knowledge.models import (
    AtomicIR,
    EvidenceRef,
    ModelError,
    ParamValue,
    StepVerification,
    TaskVariable,
    TestCaseRevision,
    WorkflowRevision,
)

BROWSER_SKILL = Path("/Users/edr-test/.hermes/skills/browser-automation/SKILL.md")
browser_skill = pytest.mark.skipif(
    not BROWSER_SKILL.is_file(), reason="browser SKILL.md not present"
)

T0 = "2026-11-08T06:00:00+00:00"
T1 = "2026-11-08T06:00:01+00:00"
T2 = "2026-11-08T06:00:02+00:00"
T3 = "2026-11-08T06:00:03+00:00"

PV = "web-testbed-1.0"
CASE_REF = "case-review-e2e@rev1:dddddddddddddddd"


def review_case() -> TestCaseRevision:
    return TestCaseRevision(
        case_id="case-review-e2e", product_version=PV, platform="web",
        title="配置服务器地址并保存",
        preconditions=["打开网络设置页面"],
        steps=[
            {"action": "打开网络设置页面", "expected": "页面加载完成"},
            {"action": "在服务器地址输入框输入服务器地址", "expected": "输入框显示所输入的地址"},
            {"action": "保存配置", "expected": "出现保存成功提示"},
        ],
        raw_text_hash="sha256:" + "2" * 64,
        source_ref=CASE_REF,
        created_at=T0,
    )


def review_evidence() -> list[EvidenceRef]:
    return [
        EvidenceRef("E-operation-1", CASE_REF, "steps[1].action", "打开网络设置页面", "operation"),
        EvidenceRef("E-operation-2", CASE_REF, "steps[2].action", "在服务器地址输入框输入服务器地址", "operation"),
        EvidenceRef("E-expected-3", CASE_REF, "steps[3].expected", "出现保存成功提示", "expected"),
    ]


def review_steps() -> list[AtomicIR]:
    return [
        AtomicIR(step_id="S1", source_step=1, action="navigate",
                 target={"semantic_name": "网络设置页面", "locator_status": "runtime_resolution"},
                 depends_on=[], evidence_refs=["E-operation-1"],
                 inference={"basis": "用例前置：打开网络设置页面（导航动作，来源用例原文）"}),
        AtomicIR(step_id="S2", source_step=2, action="input",
                 target={"semantic_name": "服务器地址输入框", "locator_status": "runtime_resolution"},
                 parameters={"value": ParamValue(variable="server_address")},
                 depends_on=["S1"], evidence_refs=["E-operation-2"]),
        AtomicIR(step_id="S3", source_step=3, action="assert",
                 target={"semantic_name": "保存成功提示", "locator_status": "runtime_resolution"},
                 depends_on=["S2"], evidence_refs=["E-expected-3"],
                 verification=StepVerification("state_check", "server_address", "E-expected-3")),
    ]


def review_vars() -> list[TaskVariable]:
    return [TaskVariable(name="server_address", var_type="string", required=True,
                         description="目标服务器地址（用例参数）")]


def build_review_pack(workflow_id: str):
    """真实 SKILL.md 快照 → 全链路 → ok 审核包。"""
    snap, prov = collect_browser_snapshot(product_version=PV, snapshot_id=f"rvw-{workflow_id}",
                                          observed_at=T0)
    mapping = MappingStage(snap, T1).run(review_steps(), "web")
    assert mapping["ok"] is True, mapping["diagnostics"]
    validation = ValidationStage(T2).run(
        mapping["bound_steps"],
        workflow_id=workflow_id, task_id=f"task-{workflow_id}", task_source_ref=CASE_REF,
        product_version=PV, platform="web", declared_variables=review_vars(),
        task_variables=review_vars(), capability_snapshot_ref=snap.content_hash,
    )
    assert validation["ok"] is True, validation["diagnostics"]
    wf: WorkflowRevision = validation["workflow"]
    rendered = RenderingStage(T2).run(wf)
    assert rendered["ok"] is True, rendered["diagnostics"]
    pack = AuditPackBuilder(T2).run(
        wf, rendered, case=review_case(), evidence=review_evidence(),
        validation=validation, snapshot=snap,
    )
    assert pack["ok"] is True, pack["diagnostics"]
    return wf, rendered, pack


def approve_record(wf: WorkflowRevision, pack: dict) -> ReviewDecision:
    return ReviewDecision(
        reviewer="张工（QA 负责人）", method="human_local", decided_at=T3,
        workflow_hash=wf.content_hash, pack_hash=pack["pack_hash"],
        decision="approve", rationale="逐步对照原用例，来源/预期覆盖齐全，静态契约通过",
        decision_id="rd-001",
    )


@browser_skill
class TestReviewDecisionPositive:
    def test_approve_binds_current_hashes(self):
        wf, _rendered, pack = build_review_pack("wf-review-pos")
        record = approve_record(wf, pack)
        verdict = verify_review_decision(record, wf, pack)
        assert verdict["valid"] is True and verdict["ok"] is True
        assert verdict["decision"] == "approve"
        assert verdict["effective"] is True
        d = record.to_dict()
        assert d["schema_version"] == REVIEW_SCHEMA_VERSION
        assert d["workflow_hash"] == wf.content_hash
        assert d["pack_hash"] == pack["pack_hash"]
        # roundtrip
        rt = ReviewDecision.from_dict(d)
        assert rt.to_dict() == d

    def test_reject_and_needs_changes_bind_hashes(self):
        wf, _rendered, pack = build_review_pack("wf-review-rej")
        rej = ReviewDecision(
            reviewer="李工", method="human_remote", decided_at=T3,
            workflow_hash=wf.content_hash, pack_hash=pack["pack_hash"],
            decision="reject", rationale="步骤2 预期与用例原文不符", decision_id="rd-002",
        )
        verdict = verify_review_decision(rej, wf, pack)
        assert verdict["valid"] is True
        assert verdict["effective"] is False  # 拒绝记录有效但不是批准

        nc = ReviewDecision(
            reviewer="李工", method="human_remote", decided_at=T3,
            workflow_hash=wf.content_hash, pack_hash=pack["pack_hash"],
            decision="needs_changes", rationale="需要补充清理步骤", decision_id="rd-003",
        )
        assert verify_review_decision(nc, wf, pack)["effective"] is False


@browser_skill
class TestReviewDecisionInvalidation:
    def test_workflow_change_invalidates(self):
        """workflow 修改后 → 批准作废（对象修改后作废）。"""
        wf, _rendered, pack = build_review_pack("wf-review-wd")
        record = approve_record(wf, pack)
        wf2 = WorkflowRevision(
            workflow_id="wf-review-wd2", product_version=PV, platform="web",
            task_id="task-wf-review-wd", task_source_ref=CASE_REF,
            steps=wf.steps, declared_variables=wf.declared_variables,
            status="validated", created_at=T1,
        )
        assert wf2.content_hash != wf.content_hash
        verdict = verify_review_decision(record, wf2, pack)
        assert verdict["valid"] is False and verdict["effective"] is False
        codes = {d["code"] for d in verdict["diagnostics"]}
        assert "workflow_hash_drift" in codes

    def test_pack_tamper_invalidates(self):
        """审核包内容篡改（如谎称 runtime_ready）→ 批准作废。"""
        wf, _rendered, pack = build_review_pack("wf-review-pt")
        record = approve_record(wf, pack)
        tampered = copy.deepcopy(pack)
        assert tampered["payload"]["runtime_ready"] is not True
        tampered["payload"]["runtime_ready"] = True  # 掩盖未绑定状态
        verdict = verify_review_decision(record, wf, tampered)
        assert verdict["valid"] is False and verdict["effective"] is False
        codes = {d["code"] for d in verdict["diagnostics"]}
        assert "pack_hash_mismatch" in codes

    def test_record_for_other_pack_rejected(self):
        """记录签发于另一个包 → record_pack_hash_mismatch。"""
        wf, _rendered, pack = build_review_pack("wf-review-oth")
        record = approve_record(wf, pack)
        _wf2, _r2, other = build_review_pack("wf-review-oth2")
        assert other["pack_hash"] != pack["pack_hash"]
        verdict = verify_review_decision(record, wf, other)
        assert verdict["valid"] is False
        codes = {d["code"] for d in verdict["diagnostics"]}
        assert "record_pack_hash_mismatch" in codes

    def test_missing_pack_is_not_approval(self):
        """包未随记录提供 → pending_pack，不得宣布批准有效（unknown 不算通过）。"""
        wf, _rendered, pack = build_review_pack("wf-review-np")
        record = approve_record(wf, pack)
        verdict = verify_review_decision(record, wf, None)
        assert verdict["valid"] is False and verdict["effective"] is False
        codes = {d["code"] for d in verdict["diagnostics"]}
        assert "pack_not_provided" in codes

    def test_malformed_pack_rejected(self):
        wf, _rendered, pack = build_review_pack("wf-review-mp")
        record = approve_record(wf, pack)
        bad = {"ok": True, "pack_hash": "deadbeef", "payload": None}
        verdict = verify_review_decision(record, wf, bad)
        assert verdict["valid"] is False
        codes = {d["code"] for d in verdict["diagnostics"]}
        assert "malformed_pack" in codes


class TestReviewDecisionConstruction:
    """构造负例：编译器不自授 —— 记录必须由完整人工输入构造，缺一即拒。"""

    def make_base(self):
        return dict(
            reviewer="张工", method="human_local", decided_at=T3,
            workflow_hash="a" * 64, pack_hash="b" * 64,
            decision="approve", rationale="逐项核对通过", decision_id="rd-x",
        )

    def test_empty_reviewer_rejected(self):
        base = self.make_base()
        base["reviewer"] = "  "
        with pytest.raises(ModelError):
            ReviewDecision(**base)

    def test_empty_rationale_rejected(self):
        base = self.make_base()
        base["rationale"] = ""
        with pytest.raises(ModelError):
            ReviewDecision(**base)

    def test_bad_decision_enum_rejected(self):
        base = self.make_base()
        base["decision"] = "auto_approve"
        with pytest.raises(ModelError):
            ReviewDecision(**base)

    def test_bad_method_rejected(self):
        base = self.make_base()
        base["method"] = "compiler_self_approval"
        with pytest.raises(ModelError):
            ReviewDecision(**base)

    def test_naive_timestamp_rejected(self):
        base = self.make_base()
        base["decided_at"] = "2026-11-08T06:00:00"  # 无时区
        with pytest.raises(ModelError):
            ReviewDecision(**base)

    def test_missing_hash_rejected(self):
        base = self.make_base()
        base["workflow_hash"] = ""
        with pytest.raises(ModelError):
            ReviewDecision(**base)

    def test_from_dict_unknown_and_missing_fields(self):
        base = self.make_base()
        bad = dict(base)
        bad["approved_automatically"] = True  # 未知字段（试图塞自动批准）
        with pytest.raises(ModelError):
            ReviewDecision.from_dict(bad)
        missing = {k: v for k, v in base.items() if k != "pack_hash"}
        with pytest.raises(ModelError):
            ReviewDecision.from_dict(missing)
