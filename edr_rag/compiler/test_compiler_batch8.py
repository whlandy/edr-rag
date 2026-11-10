"""P4 第三批次：web 平台端到端链路（design.md §8 / §10 P4「两平台计划」）。

真实 browser-automation SKILL.md 快照（只读来源）→ web AtomicIR（navigate/
input/assert，含变量与断言）→ MappingStage 确定性绑定 → ValidationStage 静态
校验 → WorkflowRevision → ExecutionGate 放行/拒绝。不实机执行、不访问浏览器；
web 平台能力来源只有 SKILL.md 工具表。

负例（§10 P4）：未支持动作（check 在 browser 工具表无声明）、平台错配、
SKILL.md 漂移后旧绑定被门禁三重拒绝、未绑定必填变量拒绝下发（§7）、
渲染同源（P5 前置：Plan/Guide 引用同一冻结 hash）。

全部为离线确定性验证；无真实平台样例，不能证明实机通过。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from edr_rag.compiler import ExecutionGate, MappingStage, RenderingStage, ValidationStage
from edr_rag.knowledge.models import (
    AtomicIR,
    CapabilitySnapshot,
    ParamValue,
    StepBinding,
    StepVerification,
    TaskVariable,
    ToolCapability,
    WorkflowRevision,
)
from edr_rag.knowledge.repository import KnowledgeRepository
from edr_rag.compiler.capability_snapshot import collect_browser_snapshot

BROWSER_SKILL = Path("/Users/edr-test/.hermes/skills/browser-automation/SKILL.md")
browser_skill = pytest.mark.skipif(
    not BROWSER_SKILL.is_file(), reason="browser SKILL.md not present"
)

T0 = "2026-11-08T03:00:00+00:00"
T1 = "2026-11-08T03:00:01+00:00"
T2 = "2026-11-08T03:00:02+00:00"

PV = "web-testbed-1.0"


def web_steps() -> list[AtomicIR]:
    """导航→输入→断言：来自用例原文的 web 语义动作（占位目标，runtime_resolution）。"""
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


def compile_to_workflow(snap: CapabilitySnapshot, *, workflow_id: str,
                        task_source_ref: str = "case-web-e2e@rev1"):
    mapping = MappingStage(snap, T1).run(web_steps(), "web")
    assert mapping["ok"] is True and mapping["status"] == "bound", mapping["diagnostics"]
    validation = ValidationStage(T2).run(
        mapping["bound_steps"],
        workflow_id=workflow_id, task_id="task-web-e2e", task_source_ref=task_source_ref,
        product_version=PV, platform="web", declared_variables=declare_vars(),
        task_variables=declare_vars(),
        capability_snapshot_ref=snap.content_hash,
    )
    return mapping, validation


class TestWebEndToEnd:
    """真实 SKILL.md 快照 → web 全链路 → 门禁放行（§10 P4 web 侧验收）。"""

    @browser_skill
    def test_web_chain_gate_ready(self, tmp_path):
        snap, prov = collect_browser_snapshot(product_version=PV, snapshot_id="web-e2e",
                                              observed_at=T0)
        assert snap.platform == "web"
        actions = snap.capabilities[0].actions
        assert {"navigate", "input", "assert"} <= set(actions)
        assert prov["file_sha256"].startswith("sha256:")

        _, validation = compile_to_workflow(snap, workflow_id="wf-web-e2e")
        assert validation["ok"] is True, validation["diagnostics"]
        assert validation["runtime_ready"] is False  # 必填变量未绑定值 → 仅静态 validated
        wf: WorkflowRevision = validation["workflow"]
        assert wf.platform == "web"
        assert all(s.binding is not None and s.binding.tool.startswith("browser-use:")
                   for s in wf.steps)

        repo = KnowledgeRepository(tmp_path)
        repo.save_snapshot(snap)
        current = repo.get_snapshot("web-e2e")
        assert current.content_hash == snap.content_hash
        result = ExecutionGate(current, T2).run(wf, workflow_hash=wf.content_hash)
        assert result["execution_ready"] is False  # 未绑定必填变量不得下发
        codes = {d["code"] for d in result["diagnostics"]}
        assert "unbound_required_variable" in codes

        # §7 执行前再校验：变量绑定后门禁放行（绑定集合由调用方声明）
        result2 = ExecutionGate(current, T2).run(wf, workflow_hash=wf.content_hash,
                                                 required_bound=["server_address"])
        assert result2["execution_ready"] is True
        assert result2["ok"] is True

    @browser_skill
    def test_rendering_same_source_after_gate(self, tmp_path):
        """门禁放行后同源渲染：Plan/Guide 引用同一冻结 WorkflowRevision hash。"""
        snap, _ = collect_browser_snapshot(product_version=PV, snapshot_id="web-render",
                                           observed_at=T0)
        _, validation = compile_to_workflow(snap, workflow_id="wf-web-render")
        wf = validation["workflow"]
        rendered = RenderingStage(T2).run(wf)
        assert rendered["ok"] is True, rendered.get("diagnostics")
        assert rendered["workflow_hash"] == wf.content_hash
        plan_steps = rendered["plan"]["steps"]
        guide_steps = rendered["guide"]["steps"]
        assert [s["step_id"] for s in plan_steps] == [s.step_id for s in wf.steps]
        assert [s["step_id"] for s in guide_steps] == [s.step_id for s in wf.steps]
        # 同源：参数引用保持变量名，不出现绑定值
        s2_plan = next(s for s in plan_steps if s["step_id"] == "S2")
        assert s2_plan["parameters"]["value"] == {"kind": "variable", "name": "server_address"}


class TestWebNegative:
    """§10 P4 web 侧负例：不支持动作、平台错配、快照漂移。"""

    @browser_skill
    def test_check_action_unsupported_on_web(self):
        """check 在 browser 工具表无对应声明 → unsupported，不静默绑定。"""
        assert "check" not in collect_browser_snapshot(
            product_version=PV, snapshot_id="web-neg-u", observed_at=T0
        )[0].capabilities[0].actions
        snap, _ = collect_browser_snapshot(product_version=PV, snapshot_id="web-neg-u2",
                                           observed_at=T0)
        steps = [AtomicIR(step_id="S1", source_step=1, action="check",
                          target={"semantic_name": "复选框", "locator_status": "runtime_resolution"},
                          depends_on=[], evidence_refs=["E1"])]
        result = MappingStage(snap, T1).run(steps, "web")
        assert result["status"] == "unsupported"
        assert result["unsupported_steps"] == ["S1"]
        assert result["ok"] is False

    @browser_skill
    def test_platform_mismatch_rejected(self):
        snap, _ = collect_browser_snapshot(product_version=PV, snapshot_id="web-neg-pm",
                                           observed_at=T0)
        result = MappingStage(snap, T1).run(web_steps(), "windows")
        assert result["ok"] is False
        assert any(d["code"] == "platform_mismatch" for d in result["diagnostics"])

    @browser_skill
    def test_drifted_skill_md_blocks_old_binding(self, tmp_path):
        """SKILL.md 漂移（快照 hash 变化）→ 旧绑定被门禁拒绝（§8）。"""
        snap, _ = collect_browser_snapshot(product_version=PV, snapshot_id="web-drift",
                                           observed_at=T0)
        _, validation = compile_to_workflow(snap, workflow_id="wf-web-drift")
        wf = validation["workflow"]
        drifted = CapabilitySnapshot(
            snapshot_id="web-drift", platform="web", product_version=PV,
            capabilities=[ToolCapability(
                tool=snap.capabilities[0].tool, tool_version="SKILL.md@drifted0000",
                tool_schema_hash="sha256:drifted0000", platform="web",
                actions=snap.capabilities[0].actions, observed_at=T1,
            )],
            observed_at=T1,
        )
        repo = KnowledgeRepository(tmp_path)
        repo.save_snapshot(snap)
        repo.save_snapshot(drifted)
        current = repo.get_snapshot("web-drift")
        assert current.content_hash == drifted.content_hash != snap.content_hash
        result = ExecutionGate(current, T2).run(wf, workflow_hash=wf.content_hash,
                                                required_bound=["server_address"])
        assert result["execution_ready"] is False
        codes = {d["code"] for d in result["diagnostics"]}
        assert {"snapshot_drift", "snapshot_ref_mismatch", "tool_schema_drift"} <= codes

    @browser_skill
    def test_gate_rejects_windows_workflow_on_web_snapshot(self):
        """web 快照收到 windows workflow → platform_mismatch（跨平台不混检）。"""
        snap, _ = collect_browser_snapshot(product_version=PV, snapshot_id="web-x",
                                           observed_at=T0)
        wf = WorkflowRevision(
            workflow_id="wf-xplat", product_version="5.3.0", platform="windows",
            task_id="task-x", task_source_ref="case-x@rev1", steps=[],
            capability_snapshot_ref=snap.content_hash, created_at=T1,
        )
        result = ExecutionGate(snap, T2).run(wf, workflow_hash=wf.content_hash)
        assert result["execution_ready"] is False
        assert result["diagnostics"][0]["code"] == "platform_mismatch"
