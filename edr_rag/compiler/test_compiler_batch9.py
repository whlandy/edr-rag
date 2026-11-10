"""P5 第一批次：同源 Markdown 审核包（design.md §9 / §10 P5）。

正例：web 全链路（真实 SKILL.md 快照 → Mapping → Validation → 渲染）→
AuditPackBuilder 生成审核包 —— 原用例引用、Plan/Guide 逐步对照、来源跳转、
参数缺口、能力适配、静态/实机分层。

负例（§9 审核 hash 绑定 / §10 P5 同源要求）：
- 渲染产物 hash 失步 → 拒绝生成审核包（不为失步产物背书）；
- Guide 与 Plan 参数/预期被篡改 → 拒绝（指南不新增事实）；
- 用例跨版本/跨平台错绑 → 拒绝（§4 跨版本不混检）；
- 证据缺失/来源错绑 → 拒绝；
- workflow 构造后修改 → verify_audit_pack 作废；
- 审核包内容被篡改 → verify_audit_pack 作废；
- 缺 ValidationStage 结论/快照 → gaps 如实列出，不宣称 unknown 为通过。

全部为离线确定性验证；无实机执行，不能证明实机通过。
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from edr_rag.compiler import MappingStage, RenderingStage, ValidationStage
from edr_rag.compiler.audit_pack import AuditPackBuilder, verify_audit_pack
from edr_rag.compiler.capability_snapshot import collect_browser_snapshot
from edr_rag.knowledge.models import (
    AtomicIR,
    CapabilitySnapshot,
    EvidenceRef,
    ParamValue,
    StepVerification,
    TaskVariable,
    TestCaseRevision,
    ToolCapability,
    WorkflowRevision,
)

BROWSER_SKILL = Path("/Users/edr-test/.hermes/skills/browser-automation/SKILL.md")
browser_skill = pytest.mark.skipif(
    not BROWSER_SKILL.is_file(), reason="browser SKILL.md not present"
)

T0 = "2026-11-08T04:00:00+00:00"
T1 = "2026-11-08T04:00:01+00:00"
T2 = "2026-11-08T04:00:02+00:00"

PV = "web-testbed-1.0"

CASE_REF = "case-audit-e2e@rev1:aaaaaaaaaaaaaaaa"


def audit_case() -> TestCaseRevision:
    return TestCaseRevision(
        case_id="case-audit-e2e", product_version=PV, platform="web",
        title="配置服务器地址并保存",
        preconditions=["打开网络设置页面"],
        steps=[
            {"action": "打开网络设置页面", "expected": "页面加载完成"},
            {"action": "在服务器地址输入框输入服务器地址", "expected": "输入框显示所输入的地址"},
            {"action": "保存配置", "expected": "出现保存成功提示"},
        ],
        raw_text_hash="sha256:" + "0" * 64,
        source_ref=CASE_REF,
        created_at=T0,
    )


def audit_evidence() -> list[EvidenceRef]:
    return [
        EvidenceRef("E-operation-1", CASE_REF, "steps[1].action", "打开网络设置页面", "operation"),
        EvidenceRef("E-operation-2", CASE_REF, "steps[2].action", "在服务器地址输入框输入服务器地址", "operation"),
        EvidenceRef("E-expected-3", CASE_REF, "steps[3].expected", "出现保存成功提示", "expected"),
    ]


def workflow_steps() -> list[AtomicIR]:
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


def declare_vars() -> list[TaskVariable]:
    return [TaskVariable(name="server_address", var_type="string", required=True,
                         description="目标服务器地址（用例参数）")]


def build_chain(snap: CapabilitySnapshot, *, workflow_id: str = "wf-audit-e2e"):
    mapping = MappingStage(snap, T1).run(workflow_steps(), "web")
    assert mapping["ok"] is True and mapping["status"] == "bound", mapping["diagnostics"]
    validation = ValidationStage(T2).run(
        mapping["bound_steps"],
        workflow_id=workflow_id, task_id="task-audit-e2e", task_source_ref=CASE_REF,
        product_version=PV, platform="web", declared_variables=declare_vars(),
        task_variables=declare_vars(), capability_snapshot_ref=snap.content_hash,
    )
    assert validation["ok"] is True, validation["diagnostics"]
    wf: WorkflowRevision = validation["workflow"]
    rendered = RenderingStage(T2).run(wf)
    assert rendered["ok"] is True, rendered["diagnostics"]
    return wf, rendered, validation


@browser_skill
class TestAuditPackPositive:
    def test_full_chain_audit_pack(self, tmp_path):
        snap, prov = collect_browser_snapshot(product_version=PV, snapshot_id="audit-snap",
                                              observed_at=T0)
        assert prov["file_sha256"].startswith("sha256:")
        wf, rendered, validation = build_chain(snap)

        pack = AuditPackBuilder(T2).run(
            wf, rendered, case=audit_case(), evidence=audit_evidence(),
            validation=validation, snapshot=snap,
        )
        assert pack["ok"] is True, pack["diagnostics"]
        assert pack["workflow_hash"] == wf.content_hash
        assert pack["complete"] is True and pack["gaps"] == []

        payload = pack["payload"]
        # 静态/实机分层（§9）
        assert payload["static_status"] == wf.status
        assert payload["execution_status"] == "not_run"
        assert payload["runtime_ready"] is False  # 必填变量未绑定值
        assert payload["approval"] == "pending_human_review"
        # hash 绑定
        assert payload["workflow_hash"] == wf.content_hash
        assert payload["capability_snapshot_hash"] == snap.content_hash

        # Plan/Guide 逐步对照行与冻结 workflow 一一对应（§10 P5）
        rows = payload["plan_guide_rows"]
        assert [r["step_id"] for r in rows] == [s.step_id for s in wf.steps]
        s2 = next(r for r in rows if r["step_id"] == "S2")
        assert "var:server_address" in s2["params"]
        # 来源跳转可解析
        ev_rows = payload["evidence_rows"]
        assert all(r["resolved"] for r in ev_rows)
        assert {r["evidence_id"] for r in ev_rows} == {"E-operation-1", "E-operation-2", "E-expected-3"}

        md = pack["markdown"]
        assert f"`{wf.content_hash}`" in md
        assert "静态 validated ≠ 实机 passed" in md
        assert "| S1 | 1 | navigate |" in md
        assert "pending_human_review" in md or "无 —— 编译器不自授" in md

    def test_verify_roundtrip(self, tmp_path):
        snap, _ = collect_browser_snapshot(product_version=PV, snapshot_id="audit-vrf",
                                           observed_at=T0)
        wf, rendered, validation = build_chain(snap, workflow_id="wf-audit-vrf")
        pack = AuditPackBuilder(T2).run(
            wf, rendered, case=audit_case(), evidence=audit_evidence(),
            validation=validation, snapshot=snap,
        )
        verdict = verify_audit_pack(pack, wf)
        assert verdict["valid"] is True and verdict["ok"] is True

    def test_unknown_gaps_listed_not_claimed(self, tmp_path):
        """缺 ValidationStage 结论与快照 → gaps 如实列出，不宣称已核对。"""
        snap, _ = collect_browser_snapshot(product_version=PV, snapshot_id="audit-unk",
                                           observed_at=T0)
        wf, rendered, _ = build_chain(snap, workflow_id="wf-audit-unk")
        pack = AuditPackBuilder(T2).run(
            wf, rendered, case=audit_case(), evidence=audit_evidence(),
        )
        assert pack["ok"] is True
        assert pack["complete"] is False
        codes = {g["code"] for g in pack["gaps"]}
        assert "binding_state_unknown" in codes
        assert "capability_snapshot_not_provided" in codes
        assert pack["payload"]["runtime_ready"] is None


@browser_skill
class TestAuditPackNegative:
    def test_rendered_hash_mismatch_rejected(self, tmp_path):
        """渲染产物不来自该冻结 workflow → 拒绝生成审核包。"""
        snap, _ = collect_browser_snapshot(product_version=PV, snapshot_id="audit-neg-h",
                                           observed_at=T0)
        wf, rendered, _ = build_chain(snap, workflow_id="wf-audit-neg-h")
        wf2 = WorkflowRevision(
            workflow_id="wf-audit-neg-h2", product_version=PV, platform="web",
            task_id="task-audit-e2e", task_source_ref=CASE_REF,
            steps=wf.steps, declared_variables=wf.declared_variables,
            capability_snapshot_ref=snap.content_hash, status="validated",
            created_at=T1,
        )
        assert wf2.content_hash != wf.content_hash
        result = AuditPackBuilder(T2).run(
            wf2, rendered, case=audit_case(), evidence=audit_evidence(),
        )
        assert result["ok"] is False
        codes = {d["code"] for d in result["diagnostics"]}
        assert "rendered_hash_mismatch" in codes

    def test_guide_tamper_rejected(self, tmp_path):
        """Guide 参数/预期被篡改 → 同源被破坏，拒绝。"""
        snap, _ = collect_browser_snapshot(product_version=PV, snapshot_id="audit-neg-t",
                                           observed_at=T0)
        wf, rendered, _ = build_chain(snap, workflow_id="wf-audit-neg-t")
        tampered = copy.deepcopy(rendered)
        tampered["guide"]["steps"][1]["expected"] = "预期被改写"
        result = AuditPackBuilder(T2).run(
            wf, tampered, case=audit_case(), evidence=audit_evidence(),
        )
        assert result["ok"] is False
        codes = {d["code"] for d in result["diagnostics"]}
        assert "guide_divergence" in codes

    def test_cross_version_case_rejected(self, tmp_path):
        """用例与 workflow 跨版本 → 拒绝（§4 跨版本不混检）。"""
        snap, _ = collect_browser_snapshot(product_version=PV, snapshot_id="audit-neg-cv",
                                           observed_at=T0)
        wf, rendered, _ = build_chain(snap, workflow_id="wf-audit-neg-cv")
        other_case = audit_case()
        other_case.product_version = "9.9.9"
        result = AuditPackBuilder(T2).run(
            wf, rendered, case=other_case, evidence=audit_evidence(),
        )
        assert result["ok"] is False
        codes = {d["code"] for d in result["diagnostics"]}
        assert "product_version_mismatch" in codes

    def test_missing_evidence_rejected(self, tmp_path):
        """审核包无原用例证据 → 拒绝（缺证据不算通过）。"""
        snap, _ = collect_browser_snapshot(product_version=PV, snapshot_id="audit-neg-ev",
                                           observed_at=T0)
        wf, rendered, _ = build_chain(snap, workflow_id="wf-audit-neg-ev")
        result = AuditPackBuilder(T2).run(
            wf, rendered, case=audit_case(), evidence=[],
        )
        assert result["ok"] is False
        codes = {d["code"] for d in result["diagnostics"]}
        assert "evidence_missing" in codes

    def test_evidence_source_mismatch_rejected(self, tmp_path):
        """证据绑定到其他来源 → 拒绝。"""
        snap, _ = collect_browser_snapshot(product_version=PV, snapshot_id="audit-neg-es",
                                           observed_at=T0)
        wf, rendered, _ = build_chain(snap, workflow_id="wf-audit-neg-es")
        foreign = [EvidenceRef("E-operation-1", "other-case@rev1:bbbbbbbbbbbbbbbb",
                               "steps[1].action", "打开网络设置页面", "operation")]
        result = AuditPackBuilder(T2).run(
            wf, rendered, case=audit_case(), evidence=foreign,
        )
        assert result["ok"] is False
        codes = {d["code"] for d in result["diagnostics"]}
        assert "evidence_source_mismatch" in codes

    def test_unresolved_evidence_becomes_gap(self, tmp_path):
        """workflow 引用了不在用例证据集中的 evidence id → 审核包生成但缺口如实列出。"""
        snap, _ = collect_browser_snapshot(product_version=PV, snapshot_id="audit-neg-ur",
                                           observed_at=T0)
        wf, rendered, _ = build_chain(snap, workflow_id="wf-audit-neg-ur")
        result = AuditPackBuilder(T2).run(
            wf, rendered, case=audit_case(), evidence=audit_evidence()[:2],
        )
        assert result["ok"] is True
        assert result["complete"] is False
        assert any(g["code"] == "unresolved_evidence_ref" for g in result["gaps"])
        unresolved = [r for r in result["payload"]["evidence_rows"] if not r["resolved"]]
        assert {r["evidence_id"] for r in unresolved} == {"E-expected-3"}
        md = result["markdown"]
        assert "来源不可解析" in md

    def test_workflow_modification_invalidates_pack(self, tmp_path):
        """workflow 构造后修改 → 审核 hash 校验作废（§9）。"""
        snap, _ = collect_browser_snapshot(product_version=PV, snapshot_id="audit-neg-wm",
                                           observed_at=T0)
        wf, rendered, validation = build_chain(snap, workflow_id="wf-audit-neg-wm")
        pack = AuditPackBuilder(T2).run(
            wf, rendered, case=audit_case(), evidence=audit_evidence(),
            validation=validation, snapshot=snap,
        )
        wf.steps[0].target["semantic_name"] = "被篡改的目标"
        verdict = verify_audit_pack(pack, wf)
        assert verdict["valid"] is False
        codes = {d["code"] for d in verdict["diagnostics"]}
        assert "pack_workflow_hash_mismatch" in codes

    def test_pack_tamper_invalidates(self, tmp_path):
        """审核包内容被篡改 → pack hash 校验作废。"""
        snap, _ = collect_browser_snapshot(product_version=PV, snapshot_id="audit-neg-pt",
                                           observed_at=T0)
        wf, rendered, validation = build_chain(snap, workflow_id="wf-audit-neg-pt")
        pack = AuditPackBuilder(T2).run(
            wf, rendered, case=audit_case(), evidence=audit_evidence(),
            validation=validation, snapshot=snap,
        )
        tampered = copy.deepcopy(pack)
        tampered["payload"]["approval"] = "approved"
        verdict = verify_audit_pack(tampered, wf)
        assert verdict["valid"] is False
        codes = {d["code"] for d in verdict["diagnostics"]}
        assert "pack_hash_mismatch" in codes

    def test_capability_drift_surfaces_as_gap(self, tmp_path):
        """绑定 snapshot_hash 与当前快照不一致 → gaps 记 capability_drift。"""
        snap, _ = collect_browser_snapshot(product_version=PV, snapshot_id="audit-neg-cd",
                                           observed_at=T0)
        wf, rendered, validation = build_chain(snap, workflow_id="wf-audit-neg-cd")
        drifted = CapabilitySnapshot(
            snapshot_id="audit-neg-cd", platform="web", product_version=PV,
            capabilities=[ToolCapability(
                tool=snap.capabilities[0].tool, tool_version="SKILL.md@drifted0000",
                tool_schema_hash="sha256:drifted0000", platform="web",
                actions=snap.capabilities[0].actions, observed_at=T1,
            )],
            observed_at=T1,
        )
        pack = AuditPackBuilder(T2).run(
            wf, rendered, case=audit_case(), evidence=audit_evidence(),
            validation=validation, snapshot=drifted,
        )
        assert pack["ok"] is True
        codes = {g["code"] for g in pack["gaps"]}
        assert "capability_drift" in codes
