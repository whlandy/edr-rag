"""P3 batch 4: Validation stage tests (design.md §6.5 / §7 / §8).

正例：bound AtomicIR → WorkflowRevision，静态 validated 可产出，未绑定
required 变量时 runtime_ready=False 但不阻断静态编译（§7）。
负例（§10 关键负例）：
- 未声明变量 / verification 指向未声明 expected_variable → unknown_variable；
- 断言证据缺失（assert 无 verification、evidence_ref 悬空）→ 不放行；
- unsupported 步骤不晋级，步骤保留（禁止静默删步骤）；
- 依赖环在 validation 层拒绝；
- 声明了 Task 未声明的变量 → variable_not_in_task；
- 空 cleanup 条目拒绝；
- 失败时输出带字段路径 diagnostics 与原步骤。
"""

from __future__ import annotations

import pytest

from edr_rag.compiler import (
    VALIDATION_SCHEMA_VERSION,
    ValidationStage,
    stage_input_hash,
)
from edr_rag.knowledge import (
    AtomicIR,
    ModelError,
    ParamValue,
    StepVerification,
    TaskVariable,
)

TS = "2026-11-08T02:30:00+08:00"


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


def make_steps() -> list[AtomicIR]:
    return [step_nav(), step_input()]


def run_validation(steps, **overrides):
    fields = dict(
        workflow_id="wf-100", task_id="task-100",
        task_source_ref="src-manual-1@rev1:cccccccccccccccc",
        product_version="5.3.0", platform="windows",
        declared_variables=[TaskVariable("server_address", "string")],
        task_variables=[TaskVariable("server_address", "string")],
        bound_values={"server_address": "10.0.0.1"},
        cleanup=["恢复原服务器地址"],
    )
    fields.update(overrides)
    return ValidationStage(validated_at=TS).run(steps, **fields)


class TestValidationPositive:
    def test_valid_workflow_produced(self):
        r = run_validation(make_steps())
        assert r["ok"] is True and r["stage"] == "validation"
        assert r["runtime_ready"] is True and r["unbound_variables"] == []
        wf = r["workflow"]
        assert wf.workflow_id == "wf-100" and len(wf.steps) == 2
        assert r["workflow_hash"] == wf.content_hash
        assert r["schema_version"] == VALIDATION_SCHEMA_VERSION

    def test_input_hash_binds_steps_and_variables(self):
        r1 = run_validation(make_steps())
        r2 = run_validation(make_steps())
        assert r1["input_hash"] == r2["input_hash"]
        r3 = run_validation([step_nav()], task_source_ref="x@rev1:y")
        assert r3["input_hash"] != r1["input_hash"]

    def test_unbound_required_variable_static_ok_runtime_not_ready(self):
        # §7: 缺少必填变量值时可静态 validated，但 runtime_ready=false
        r = run_validation(make_steps(), bound_values={})
        assert r["ok"] is True and r["runtime_ready"] is False
        assert r["unbound_variables"] == ["server_address"]
        assert any(d["code"] == "unbound_required_variable" for d in r["diagnostics"])

    def test_optional_variable_unbound_stays_ready(self):
        r = run_validation(
            make_steps(), bound_values={},
            declared_variables=[TaskVariable("server_address", "string", required=False)],
            task_variables=[TaskVariable("server_address", "string", required=False)],
        )
        assert r["ok"] is True and r["runtime_ready"] is True


class TestValidationNegative:
    def test_unknown_parameter_variable_rejected(self):
        s2 = step_input()
        s2.parameters = {"value": ParamValue(variable="admin_password")}
        r = run_validation(make_steps()[:1] + [s2])
        assert r["ok"] is False
        assert any(d["code"] == "unknown_variable" and "admin_password" in d["message"] for d in r["diagnostics"])

    def test_verification_unknown_expected_variable_rejected(self):
        s2 = step_input()
        s2.verification = StepVerification("value_check", "ghost_var", "E-expected-2")
        r = run_validation(make_steps()[:1] + [s2])
        assert r["ok"] is False
        assert any(d["code"] == "unknown_variable" and "ghost_var" in d["message"] for d in r["diagnostics"])

    def test_assert_without_verification_rejected(self):
        s3 = step_assert()
        s3.verification = None
        r = run_validation(make_steps() + [s3])
        assert r["ok"] is False
        assert any(d["code"] == "missing_verification" and d["path"] == "steps[S3].verification"
                   for d in r["diagnostics"])

    def test_dangling_verification_evidence_rejected(self):
        s2 = step_input()
        s2.verification = StepVerification("value_check", "server_address", "E-ghost")
        r = run_validation(make_steps()[:1] + [s2])
        assert r["ok"] is False
        assert any(d["code"] == "dangling_evidence_ref" for d in r["diagnostics"])

    def test_unsupported_step_not_promoted_and_kept(self):
        # 未声明能力不晋级；步骤原样返回，不静默删（§6.5）
        steps = make_steps()
        r = run_validation(steps, unsupported_steps=["S2"])
        assert r["ok"] is False
        assert any(d["code"] == "unsupported" and d["path"] == "steps[S2]" for d in r["diagnostics"])
        assert [s.step_id for s in r["steps"]] == ["S1", "S2"]
        assert "workflow" not in r

    def test_dependency_cycle_rejected(self):
        a = AtomicIR(step_id="A", source_step=1, action="click",
                     target={"semantic_name": "a", "locator_status": "runtime_resolution"},
                     depends_on=["B"], evidence_refs=["E-1"])
        b = AtomicIR(step_id="B", source_step=2, action="click",
                     target={"semantic_name": "b", "locator_status": "runtime_resolution"},
                     depends_on=["A"], evidence_refs=["E-1"])
        r = run_validation([a, b])
        assert r["ok"] is False
        assert any(d["code"] == "dependency_cycle" for d in r["diagnostics"])

    def test_variable_not_declared_by_task_rejected(self):
        r = run_validation(make_steps(), task_variables=[])
        assert r["ok"] is False
        assert any(d["code"] == "variable_not_in_task" for d in r["diagnostics"])

    def test_empty_cleanup_entry_rejected(self):
        r = run_validation(make_steps(), cleanup=["  "])
        assert r["ok"] is False
        assert any(d["code"] == "empty_cleanup_entry" and d["path"] == "cleanup[0]" for d in r["diagnostics"])

    def test_missing_platform_rejected(self):
        r = run_validation(make_steps(), platform="")
        assert r["ok"] is False
        assert any(d["code"] == "missing_platform" for d in r["diagnostics"])

    def test_missing_source_refs_rejected(self):
        r = run_validation(make_steps(), task_source_ref="")
        assert r["ok"] is False
        assert any(d["code"] == "missing_source" for d in r["diagnostics"])

    def test_non_atomic_ir_input_rejected_with_steps_kept(self):
        r = run_validation([step_nav(), {"step_id": "S2"}])  # type: ignore[list-item]
        assert r["ok"] is False
        assert any(d["code"] == "not_atomic_ir" and d["path"] == "steps[1]" for d in r["diagnostics"])

    def test_diagnostics_carry_field_paths(self):
        r = run_validation(make_steps(), task_variables=[])
        assert all({"path", "code", "message"} <= set(d) for d in r["diagnostics"])

    def test_bad_validated_at_rejected(self):
        with pytest.raises(ModelError, match="validated_at"):
            ValidationStage(validated_at="2026-11-08T02:30:00")  # 无时区
