"""P5 第二批次：export-bundle 文件导出（design.md §9 / §10 P5）。

正例：web 全链路（真实 SKILL.md 快照 → Mapping → Validation → 渲染 →
AuditPackBuilder）→ BundleExporter 导出到本地目录 —— workflow.json /
plan.json / guide.json / audit-pack.md / manifest.json，manifest 含
workflow_hash、pack_hash、各文件 sha256。

负例（§9 审核 hash 绑定 / bundle 完整性）：
- pack 被篡改（pack_hash 失配）→ 拒绝导出；
- workflow 导出后修改 → verify_export_bundle 失效；
- bundle 文件内容修改 → 消费者复核失效；
- bundle 文件缺失 → 消费者复核失效；
- manifest 缺失（不完整 bundle）→ 消费者必须拒绝；
- 目标目录已存在 → 拒绝导出（不覆盖既有 bundle）；
- pack 非 ok（builder 失败产物）→ 拒绝导出。

全部为离线确定性验证；无实机执行，不能证明实机通过。
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from edr_rag.compiler import MappingStage, RenderingStage, ValidationStage
from edr_rag.compiler.audit_pack import AuditPackBuilder
from edr_rag.compiler.capability_snapshot import collect_browser_snapshot
from edr_rag.compiler.export_bundle import BundleExporter, verify_export_bundle
from edr_rag.knowledge.models import (
    AtomicIR,
    EvidenceRef,
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

T0 = "2026-11-08T05:00:00+00:00"
T1 = "2026-11-08T05:00:01+00:00"
T2 = "2026-11-08T05:00:02+00:00"
T3 = "2026-11-08T05:00:03+00:00"

PV = "web-testbed-1.0"
CASE_REF = "case-export-e2e@rev1:cccccccccccccccc"


def export_case() -> TestCaseRevision:
    return TestCaseRevision(
        case_id="case-export-e2e", product_version=PV, platform="web",
        title="配置服务器地址并保存",
        preconditions=["打开网络设置页面"],
        steps=[
            {"action": "打开网络设置页面", "expected": "页面加载完成"},
            {"action": "在服务器地址输入框输入服务器地址", "expected": "输入框显示所输入的地址"},
            {"action": "保存配置", "expected": "出现保存成功提示"},
        ],
        raw_text_hash="sha256:" + "1" * 64,
        source_ref=CASE_REF,
        created_at=T0,
    )


def export_evidence() -> list[EvidenceRef]:
    return [
        EvidenceRef("E-operation-1", CASE_REF, "steps[1].action", "打开网络设置页面", "operation"),
        EvidenceRef("E-operation-2", CASE_REF, "steps[2].action", "在服务器地址输入框输入服务器地址", "operation"),
        EvidenceRef("E-expected-3", CASE_REF, "steps[3].expected", "出现保存成功提示", "expected"),
    ]


def export_steps() -> list[AtomicIR]:
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


def export_vars() -> list[TaskVariable]:
    return [TaskVariable(name="server_address", var_type="string", required=True,
                         description="目标服务器地址（用例参数）")]


def build_full_pack(workflow_id: str):
    """真实 SKILL.md 快照 → 全链路 → ok 审核包（同源：wf + rendered + pack）。"""
    snap, prov = collect_browser_snapshot(product_version=PV, snapshot_id=f"exp-{workflow_id}",
                                          observed_at=T0)
    mapping = MappingStage(snap, T1).run(export_steps(), "web")
    assert mapping["ok"] is True, mapping["diagnostics"]
    validation = ValidationStage(T2).run(
        mapping["bound_steps"],
        workflow_id=workflow_id, task_id=f"task-{workflow_id}", task_source_ref=CASE_REF,
        product_version=PV, platform="web", declared_variables=export_vars(),
        task_variables=export_vars(), capability_snapshot_ref=snap.content_hash,
    )
    assert validation["ok"] is True, validation["diagnostics"]
    wf: WorkflowRevision = validation["workflow"]
    rendered = RenderingStage(T2).run(wf)
    assert rendered["ok"] is True, rendered["diagnostics"]
    pack = AuditPackBuilder(T2).run(
        wf, rendered, case=export_case(), evidence=export_evidence(),
        validation=validation, snapshot=snap,
    )
    assert pack["ok"] is True, pack["diagnostics"]
    return wf, rendered, pack


@browser_skill
class TestExportBundlePositive:
    def test_export_full_bundle(self, tmp_path):
        wf, rendered, pack = build_full_pack("wf-export-e2e")
        out = tmp_path / "bundle"
        result = BundleExporter(T3).export(pack, wf, rendered, out)
        assert result["ok"] is True, result["diagnostics"]
        assert result["workflow_hash"] == wf.content_hash
        assert result["pack_hash"] == pack["pack_hash"]
        assert result["approval"] == "pending_human_review"

        # 五个文件全部落盘
        for name in ("workflow.json", "plan.json", "guide.json", "audit-pack.md", "manifest.json"):
            assert (out / name).is_file(), name

        wf_d = json.loads((out / "workflow.json").read_text(encoding="utf-8"))
        assert wf_d["workflow_id"] == "wf-export-e2e"
        assert wf_d["status"] == wf.status  # 冻结状态原样导出，不被晋级
        assert wf_d["execution_status"] == "not_run"
        plan_d = json.loads((out / "plan.json").read_text(encoding="utf-8"))
        guide_d = json.loads((out / "guide.json").read_text(encoding="utf-8"))
        assert plan_d["workflow_hash"] == wf.content_hash
        assert guide_d["workflow_hash"] == wf.content_hash
        assert [s["step_id"] for s in plan_d["steps"]] == [s.step_id for s in wf.steps]
        assert [s["step_id"] for s in guide_d["steps"]] == [s.step_id for s in wf.steps]
        # 参数以引用出现，不含绑定值
        s2 = next(s for s in plan_d["steps"] if s["step_id"] == "S2")
        assert s2["parameters"]["value"] == {"kind": "variable", "name": "server_address"}

        md = (out / "audit-pack.md").read_text(encoding="utf-8")
        assert f"`{wf.content_hash}`" in md
        assert "pending_human_review" in md or "编译器不自授" in md

        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["kind"] == "workflow-audit-bundle"
        assert manifest["workflow_hash"] == wf.content_hash
        assert manifest["pack_hash"] == pack["pack_hash"]
        assert manifest["approval"] == "pending_human_review"
        assert set(manifest["files"]) == {
            "workflow.json", "plan.json", "guide.json", "audit-pack.md"}
        for name, h in manifest["files"].items():
            assert h.startswith("sha256:"), name

    def test_consumer_verify_roundtrip(self, tmp_path):
        wf, rendered, pack = build_full_pack("wf-export-vrf")
        out = tmp_path / "bundle"
        assert BundleExporter(T3).export(pack, wf, rendered, out)["ok"] is True
        verdict = verify_export_bundle(out)
        assert verdict["valid"] is True and verdict["ok"] is True
        assert verdict["manifest"]["workflow_hash"] == wf.content_hash
        assert verdict["manifest"]["pack_hash"] == pack["pack_hash"]

    def test_export_rejects_existing_dir(self, tmp_path):
        """目标目录已存在 → 拒绝导出（不覆盖既有 bundle）。"""
        wf, rendered, pack = build_full_pack("wf-export-ex")
        out = tmp_path / "bundle"
        assert BundleExporter(T3).export(pack, wf, rendered, out)["ok"] is True
        first_md = (out / "audit-pack.md").read_text(encoding="utf-8")
        result = BundleExporter(T3).export(pack, wf, rendered, out)
        assert result["ok"] is False
        codes = {d["code"] for d in result["diagnostics"]}
        assert "out_dir_exists" in codes
        # 原内容未被覆盖
        assert (out / "audit-pack.md").read_text(encoding="utf-8") == first_md


@browser_skill
class TestExportBundleNegative:
    def test_tampered_pack_rejected(self, tmp_path):
        """审核包被篡改（pack_hash 失配）→ 拒绝导出。"""
        wf, rendered, pack = build_full_pack("wf-export-tamper")
        tampered = copy.deepcopy(pack)
        tampered["payload"]["approval"] = "approved"  # 谎称已批准
        result = BundleExporter(T3).export(tampered, wf, rendered, tmp_path / "b1")
        assert result["ok"] is False
        codes = {d["code"] for d in result["diagnostics"]}
        assert "pack_hash_mismatch" in codes

    def test_workflow_hash_mismatch_rejected(self, tmp_path):
        """workflow 与 pack 不同源（hash 不一致）→ 拒绝导出。"""
        wf, rendered, pack = build_full_pack("wf-export-whm")
        wf2 = WorkflowRevision(
            workflow_id="wf-export-whm2", product_version=PV, platform="web",
            task_id=f"task-wf-export-whm", task_source_ref=CASE_REF,
            steps=wf.steps, declared_variables=wf.declared_variables,
            status="validated", created_at=T1,
        )
        assert wf2.content_hash != wf.content_hash
        result = BundleExporter(T3).export(pack, wf2, rendered, tmp_path / "b2")
        assert result["ok"] is False
        codes = {d["code"] for d in result["diagnostics"]}
        assert "pack_workflow_hash_mismatch" in codes

    def test_rendered_mismatch_rejected(self, tmp_path):
        """渲染产物不来自该冻结 workflow → 拒绝导出。"""
        wf, rendered, pack = build_full_pack("wf-export-rm")
        wf2 = WorkflowRevision(
            workflow_id="wf-export-rm2", product_version=PV, platform="web",
            task_id="task-wf-export-rm", task_source_ref=CASE_REF,
            steps=wf.steps, declared_variables=wf.declared_variables,
            status="validated", created_at=T1,
        )
        result = BundleExporter(T3).export(pack, wf, rendered, tmp_path / "b3")
        assert result["ok"] is True  # 同源导出仍成功
        wf2_rendered = copy.deepcopy(rendered)
        wf2_rendered["workflow_hash"] = wf2.content_hash  # 谎报来源
        result2 = BundleExporter(T3).export(pack, wf2, wf2_rendered, tmp_path / "b4")
        assert result2["ok"] is False
        codes = {d["code"] for d in result2["diagnostics"]}
        assert "pack_workflow_hash_mismatch" in codes  # pack 仍绑定 wf 而非 wf2

    def test_failed_pack_rejected(self, tmp_path):
        """builder 失败产物（ok=False）→ 拒绝导出。"""
        wf, rendered, pack = build_full_pack("wf-export-fp")
        failed = {"ok": False, "pack_hash": "whatever", "payload": {}}
        result = BundleExporter(T3).export(failed, wf, rendered, tmp_path / "b5")
        assert result["ok"] is False
        codes = {d["code"] for d in result["diagnostics"]}
        assert "pack_not_ok" in codes

    def test_consumer_detects_file_tamper(self, tmp_path):
        """bundle 文件内容导出后被修改 → 消费者复核失效。"""
        wf, rendered, pack = build_full_pack("wf-export-ct")
        out = tmp_path / "bundle"
        assert BundleExporter(T3).export(pack, wf, rendered, out)["ok"] is True
        guide = json.loads((out / "guide.json").read_text(encoding="utf-8"))
        guide["steps"][0]["expected"] = "被篡改的预期"
        (out / "guide.json").write_text(
            json.dumps(guide, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        verdict = verify_export_bundle(out)
        assert verdict["valid"] is False
        codes = {d["code"] for d in verdict["diagnostics"]}
        assert "bundle_file_hash_mismatch" in codes

    def test_consumer_detects_missing_file(self, tmp_path):
        """bundle 文件被删除 → 消费者复核失效。"""
        wf, rendered, pack = build_full_pack("wf-export-missing")
        out = tmp_path / "bundle"
        assert BundleExporter(T3).export(pack, wf, rendered, out)["ok"] is True
        (out / "plan.json").unlink()
        verdict = verify_export_bundle(out)
        assert verdict["valid"] is False
        codes = {d["code"] for d in verdict["diagnostics"]}
        assert "bundle_file_missing" in codes

    def test_consumer_rejects_manifest_missing(self, tmp_path):
        """manifest 缺失（不完整 bundle）→ 消费者必须拒绝。"""
        wf, rendered, pack = build_full_pack("wf-export-nomanifest")
        out = tmp_path / "bundle"
        assert BundleExporter(T3).export(pack, wf, rendered, out)["ok"] is True
        (out / "manifest.json").unlink()
        verdict = verify_export_bundle(out)
        assert verdict["valid"] is False and verdict["ok"] is False
        codes = {d["code"] for d in verdict["diagnostics"]}
        assert "manifest_missing" in codes

    def test_consumer_rejects_non_bundle_dir(self, tmp_path):
        """非 bundle 目录（无 manifest）→ 拒绝。"""
        empty = tmp_path / "not-a-bundle"
        empty.mkdir()
        verdict = verify_export_bundle(empty)
        assert verdict["valid"] is False
        codes = {d["code"] for d in verdict["diagnostics"]}
        assert "manifest_missing" in codes
