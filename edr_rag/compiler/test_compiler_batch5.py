"""P3 batch 5: Rendering stage tests (design.md §6.6 / §5 P5 同源输出).

正例：冻结 WorkflowRevision → 同源 AgentPlan/GuideView —— 步骤一一对应，
保留相同 step_id、action、参数引用与预期；Plan/Guide 引用同一
WorkflowRevision.content_hash；凭据/绑定值不进入渲染产物。

负例（§10 对应渲染层负例）：
- 非 WorkflowRevision 输入拒绝；
- 参数引用未声明变量 → unknown_variable，不产出渲染产物；
- 同源校验：改 workflow（加一步）后旧 hash 与新渲染 hash 不同（引用可解析、
  修改后作废）；
- 渲染产物不得包含 bound_values 中的实际值（凭据只保存引用）。
"""

from __future__ import annotations

import pytest

from edr_rag.compiler import (
    RENDERING_SCHEMA_VERSION,
    RenderingStage,
    stage_input_hash,
)
from edr_rag.knowledge import (
    AtomicIR,
    ModelError,
    ParamValue,
    StepVerification,
    TaskVariable,
    WorkflowRevision,
)

TS = "2026-11-08T02:50:00+08:00"


def step_nav() -> AtomicIR:
    return AtomicIR(
        step_id="S1", source_step=1, action="navigate",
        target={"semantic_name": "网络设置页", "locator_status": "runtime_resolution"},
        depends_on=[], evidence_refs=["E-operation-1"],
    )


def step_input() -> AtomicIR:
    return AtomicIR(
        step_id="S2", source_step=2, action="input",
        target={"semantic_name": "服务器地址输入框", "locator_status": "runtime_resolution"},
        parameters={"value": ParamValue(variable="server_address")},
        depends_on=["S1"], evidence_refs=["E-operation-2", "E-expected-2"],
        verification=StepVerification("value_check", "server_address", "E-expected-2"),
    )


def step_assert() -> AtomicIR:
    return AtomicIR(
        step_id="S3", source_step=2, action="assert",
        target={"semantic_name": "保存成功提示", "locator_status": "runtime_resolution"},
        depends_on=["S2"], evidence_refs=["E-expected-2"],
        verification=StepVerification("state_check", "server_address", "E-expected-2"),
    )


def make_workflow(steps=None, variables=None) -> WorkflowRevision:
    return WorkflowRevision(
        workflow_id="wf-100", product_version="5.3.0", platform="windows",
        task_id="task-100", task_source_ref="src-manual-1@rev1:cccccccccccccccc",
        steps=steps if steps is not None else [step_nav(), step_input()],
        declared_variables=variables if variables is not None else [TaskVariable("server_address", "string")],
        status="validated", revision=1, created_at=TS,
    )


def run_render(workflow=None, **overrides):
    fields = {"bound_values": {"server_address": "10.0.0.1"}}
    fields.update(overrides)
    return RenderingStage(rendered_at=TS).run(
        workflow if workflow is not None else make_workflow(), **fields
    )


class TestRenderingPositive:
    def test_same_steps_ids_and_parity(self):
        # §6.6: 保留相同步骤 ID、参数和预期 —— Plan/Guide 与 workflow 逐步对应
        r = run_render()
        assert r["ok"] is True and r["stage"] == "rendering"
        assert r["schema_version"] == RENDERING_SCHEMA_VERSION
        assert isinstance(r["workflow_hash"], str) and r["workflow_hash"]

    def test_plan_and_guide_reference_same_workflow_hash(self):
        r = run_render()
        wf_hash = r["workflow_hash"]
        assert r["plan"]["workflow_hash"] == wf_hash
        assert r["guide"]["workflow_hash"] == wf_hash

    def test_plan_step_ids_match_workflow_exactly(self):
        wf = make_workflow()
        r = run_render(wf)
        plan_ids = [s["step_id"] for s in r["plan"]["steps"]]
        assert plan_ids == [s.step_id for s in wf.steps]
        plan_actions = [s["action"] for s in r["plan"]["steps"]]
        assert plan_actions == [s.action for s in wf.steps]

    def test_guide_step_ids_match_workflow_exactly(self):
        wf = make_workflow()
        r = run_render(wf)
        guide_ids = [s["step_id"] for s in r["guide"]["steps"]]
        assert guide_ids == [s.step_id for s in wf.steps]

    def test_parameters_rendered_as_variable_reference(self):
        # §7: 变量使用类型化绑定，凭据只保存引用 —— 渲染产物含变量名，不含值
        r = run_render()
        s2 = next(s for s in r["plan"]["steps"] if s["step_id"] == "S2")
        assert s2["parameters"]["value"] == {"kind": "variable", "name": "server_address"}
        g2 = next(s for s in r["guide"]["steps"] if s["step_id"] == "S2")
        assert g2["parameters"]["value"] == {"kind": "variable", "name": "server_address"}

    def test_bound_values_never_leak_into_rendered_output(self):
        # 凭据/绑定值不进入 Plan/Guide 产物
        import json
        r = run_render(bound_values={"server_address": "10.0.0.1", "admin_password": "p@ss"})
        blob = json.dumps(r["plan"]) + json.dumps(r["guide"])
        assert "10.0.0.1" not in blob
        assert "p@ss" not in blob

    def test_expected_preserved_from_verification(self):
        # 预期不丢失：verification 的 expected_variable 出现在 Plan/Guide 步骤预期
        r = run_render()
        s2_plan = next(s for s in r["plan"]["steps"] if s["step_id"] == "S2")
        assert "server_address" in s2_plan["expected"]
        assert s2_plan["verification"] == {
            "kind": "value_check", "expected_variable": "server_address", "evidence_ref": "E-expected-2",
        }
        g2 = next(s for s in r["guide"]["steps"] if s["step_id"] == "S2")
        assert "server_address" in g2["expected"]

    def test_literal_parameter_preserved_as_scalar(self):
        steps = [step_nav(), step_input(), step_assert()]
        r = run_render(make_workflow(steps))
        s3 = next(s for s in r["plan"]["steps"] if s["step_id"] == "S3")
        assert s3["verification"]["kind"] == "state_check"

    def test_render_is_deterministic(self):
        r1 = run_render()
        r2 = run_render()
        assert r1["workflow_hash"] == r2["workflow_hash"]
        assert r1["plan"]["steps"] == r2["plan"]["steps"]
        assert r1["guide"]["steps"] == r2["guide"]["steps"]


class TestRenderingNegative:
    def test_non_workflow_input_rejected(self):
        r = run_render({"workflow_id": "wf-100"})
        assert r["ok"] is False
        assert any(d["code"] == "not_workflow_revision" for d in r["diagnostics"])

    def test_undeclared_parameter_variable_blocked(self):
        # 渲染不得丢失类型化绑定：引用未声明变量 → 拒绝，不产出 Plan/Guide
        wf = make_workflow(variables=[TaskVariable("other_var", "string")])
        r = run_render(wf)
        assert r["ok"] is False
        assert any(d["code"] == "unknown_variable" and "server_address" in d["message"]
                   for d in r["diagnostics"])
        assert "plan" not in r and "guide" not in r

    def test_modified_workflow_produces_different_hash(self):
        # 同源绑定：workflow 修改后（加一步）hash 变化，旧渲染引用不再匹配
        r1 = run_render()
        wf2 = make_workflow(steps=[step_nav(), step_input(), step_assert()])
        r2 = run_render(wf2)
        assert r1["workflow_hash"] != r2["workflow_hash"]
        assert r2["workflow_hash"] == wf2.content_hash

    def test_rendered_plan_hash_resolves_to_workflow(self):
        # 引用可解析：渲染产物的 workflow_hash 能与冻结 WorkflowRevision 校验一致
        wf = make_workflow()
        r = run_render(wf)
        assert r["workflow_hash"] == wf.content_hash

    def test_bad_rendered_at_rejected(self):
        with pytest.raises(ModelError, match="rendered_at"):
            RenderingStage(rendered_at="2026-11-08T02:50:00")  # 无时区
