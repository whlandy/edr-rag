"""P3 batch 3: Mapping stage tests (design.md §6.4 / §8).

正例：CapabilitySnapshot 确定性绑定工具/版本/schema hash/snapshot_hash、
bound AtomicIR 可 roundtrip。负例（P3 验收 §10）：
- 未声明能力 → unsupported，不静默删步骤；
- 平台不匹配 / 平台缺失拒绝；
- 定位器固化门禁：fixed 无真实观测证据 → 降级 runtime_resolution（不猜）；
- 有观测证据的 fixed 放行且 locator 进 binding；
- 裸 dict binding、伪造 schema hash 在模型层拒绝；
- 输入 hash 绑定快照与步骤；
- 依赖环在映射层拒绝。
"""

from __future__ import annotations

import pytest

from edr_rag.compiler import (
    MAPPING_SCHEMA_VERSION,
    MappingStage,
    stage_input_hash,
)
from edr_rag.knowledge import (
    ATOMIC_ACTIONS,
    AtomicIR,
    CapabilitySnapshot,
    ModelError,
    StepBinding,
    ToolCapability,
    content_hash,
)

TS = "2026-11-08T02:00:00+08:00"


def make_snapshot(platform: str = "windows", actions: tuple[str, ...] = ATOMIC_ACTIONS) -> CapabilitySnapshot:
    cap = ToolCapability(
        tool="edr-wd-gui", tool_version="1.4.2", tool_schema_hash=content_hash({"tool": "edr-wd-gui"}),
        platform=platform, actions=actions, observed_at=TS,
    )
    return CapabilitySnapshot(
        snapshot_id="snap-windows-53", platform=platform, product_version="5.3.0",
        capabilities=[cap], observed_at=TS,
    )


def make_step(step_id: str = "S1", action: str = "click", locator_status: str = "runtime_resolution",
              depends_on: list[str] | None = None) -> AtomicIR:
    return AtomicIR(
        step_id=step_id, source_step=1, action=action,
        target={"semantic_name": "保存按钮", "locator_status": locator_status},
        evidence_refs=["E-operation-1"], depends_on=list(depends_on or []),
    )


class TestCapabilitySnapshotModel:
    def test_content_hash_stable_and_binding_anchor(self):
        s = make_snapshot()
        assert len(s.content_hash) == 64
        assert CapabilitySnapshot.from_dict(s.to_dict()).content_hash == s.content_hash

    def test_duplicate_tool_rejected(self):
        cap = ToolCapability(tool="dup", tool_version="1", tool_schema_hash="h" * 64,
                             platform="windows", actions=("click",), observed_at=TS)
        with pytest.raises(ModelError, match="duplicate tool"):
            CapabilitySnapshot(snapshot_id="s", platform="windows", product_version="5.3.0",
                               capabilities=[cap, cap], observed_at=TS)

    def test_empty_actions_rejected(self):
        with pytest.raises(ModelError, match="actions"):
            ToolCapability(tool="t", tool_version="1", tool_schema_hash="h" * 64,
                           platform="windows", actions=(), observed_at=TS)

    def test_unknown_action_rejected(self):
        with pytest.raises(ModelError, match="ATOMIC_ACTIONS"):
            ToolCapability(tool="t", tool_version="1", tool_schema_hash="h" * 64,
                           platform="windows", actions=("regedit",), observed_at=TS)

    def test_capability_for_platform_action_match(self):
        s = make_snapshot()
        assert s.capability_for("click", "windows") is not None
        assert s.capability_for("click", "web") is None  # 平台不匹配
        s2 = make_snapshot(actions=("click", "input"))
        assert s2.capability_for("navigate", "windows") is None  # 未声明动作


class TestStepBindingModel:
    def test_full_binding_roundtrip(self):
        b = StepBinding(tool="edr-wd-gui", tool_version="1.4.2", tool_schema_hash="h" * 64,
                        snapshot_hash="s" * 64, bound_at=TS,
                        locator={"kind": "automation_id", "value": "btn-save",
                                 "observed_at": TS, "environment_version": "5.3.0"})
        assert StepBinding.from_dict(b.to_dict()) == b

    def test_bare_dict_binding_rejected(self):
        with pytest.raises(ModelError, match="StepBinding"):
            AtomicIR(step_id="S1", source_step=1, action="click",
                     target={"semantic_name": "保存", "locator_status": "runtime_resolution"},
                     evidence_refs=["E-1"], binding={"tool": "edr_wd.click"})

    def test_binding_missing_schema_hash_rejected(self):
        with pytest.raises(ModelError, match="tool_schema_hash"):
            StepBinding(tool="t", tool_version="1", tool_schema_hash="",
                        snapshot_hash="s" * 64, bound_at=TS)

    def test_fixed_without_locator_rejected_at_model(self):
        with pytest.raises(ModelError, match="fixed"):
            AtomicIR(step_id="S1", source_step=1, action="click",
                     target={"semantic_name": "保存", "locator_status": "fixed"},
                     evidence_refs=["E-1"],
                     binding=StepBinding(tool="t", tool_version="1", tool_schema_hash="h" * 64,
                                         snapshot_hash="s" * 64, bound_at=TS, locator=None))

    def test_locator_without_environment_version_rejected(self):
        with pytest.raises(ModelError, match="environment_version"):
            StepBinding(tool="t", tool_version="1", tool_schema_hash="h" * 64,
                        snapshot_hash="s" * 64, bound_at=TS,
                        locator={"kind": "css", "value": "#ok", "observed_at": TS,
                                 "environment_version": ""})


class TestMappingHappyPath:
    def test_deterministic_binding_from_snapshot(self):
        snap = make_snapshot()
        out = MappingStage(snap, TS).run([make_step(), make_step("S2", "input")], platform="windows")
        assert out["ok"] is True and out["status"] == "bound"
        assert out["schema_version"] == MAPPING_SCHEMA_VERSION
        assert out["snapshot_hash"] == snap.content_hash
        bound = out["bound_steps"]
        assert all(st.binding is not None for st in bound)
        assert {st.binding.tool for st in bound} == {"edr-wd-gui"}
        assert {st.binding.tool_version for st in bound} == {"1.4.2"}
        assert all(st.binding.tool_schema_hash == snap.capabilities[0].tool_schema_hash for st in bound)
        assert all(st.binding.snapshot_hash == snap.content_hash for st in bound)
        assert all(st.binding.locator is None for st in bound)  # 无观测不固化
        # 原步骤语义保持：不动 target.semantic_name / evidence / depends_on
        assert bound[0].target["semantic_name"] == "保存按钮"
        assert bound[0].evidence_refs == ["E-operation-1"]

    def test_bound_step_roundtrip(self):
        out = MappingStage(make_snapshot(), TS).run([make_step()], platform="windows")
        d = out["bound_steps"][0].to_dict()
        assert AtomicIR.from_dict(d) == out["bound_steps"][0]

    def test_input_hash_binds_snapshot_and_steps(self):
        snap = make_snapshot()
        steps = [make_step()]
        out = MappingStage(snap, TS).run(steps, platform="windows")
        expected = stage_input_hash({
            "schema_version": MAPPING_SCHEMA_VERSION, "snapshot_hash": snap.content_hash,
            "platform": "windows", "steps": [s.to_dict() for s in steps],
        })
        assert out["input_hash"] == expected
        # 快照变化 → input_hash 变化（执行前复核 hash 的锚）
        snap2 = make_snapshot()
        out2 = MappingStage(snap2, TS).run(steps, platform="windows")
        assert out2["input_hash"] == out["input_hash"]  # 相同快照 hash
        snap3 = CapabilitySnapshot(snapshot_id="snap-b", platform="windows", product_version="5.3.0",
                                   capabilities=snap.capabilities, observed_at=TS)
        out3 = MappingStage(snap3, TS).run(steps, platform="windows")
        assert out3["input_hash"] != out["input_hash"]  # snapshot_id 变 → hash 变


class TestMappingNegative:
    def test_undeclared_action_unsupported_not_deleted(self):
        snap = make_snapshot(actions=("click", "input"))  # navigate 未声明
        out = MappingStage(snap, TS).run(
            [make_step("S1", "navigate"), make_step("S2", "click", depends_on=["S1"])],
            platform="windows")
        assert out["ok"] is False and out["status"] == "unsupported"
        assert out["unsupported_steps"] == ["S1"]
        assert any(d["code"] == "unsupported" and "navigate" in d["message"] for d in out["diagnostics"])
        # 不静默删：其余步骤仍绑定，未声明步骤不出现在 bound_steps 但有诊断
        assert [st.step_id for st in out["bound_steps"]] == ["S2"]

    def test_platform_mismatch_rejected(self):
        out = MappingStage(make_snapshot(platform="web"), TS).run([make_step()], platform="windows")
        assert out["ok"] is False
        assert any(d["code"] == "platform_mismatch" for d in out["diagnostics"])

    def test_missing_platform_rejected(self):
        out = MappingStage(make_snapshot(), TS).run([make_step()], platform="")
        assert out["ok"] is False
        assert any(d["code"] == "missing_platform" for d in out["diagnostics"])

    def test_fixed_locator_without_evidence_demoted(self):
        """§6.4: 定位器无真实观测不得固化 → 降级 runtime_resolution。"""
        out = MappingStage(make_snapshot(), TS).run(
            [make_step("S1", "click", locator_status="fixed")], platform="windows")
        assert out["status"] == "bound"
        step = out["bound_steps"][0]
        assert step.target["locator_status"] == "runtime_resolution"
        assert step.binding.locator is None
        assert any(d["code"] == "locator_demoted" for d in out["diagnostics"])

    def test_fixed_locator_with_observed_evidence_allowed(self):
        observed = StepBinding(tool="edr-wd-gui", tool_version="1.4.2", tool_schema_hash="h" * 64,
                               snapshot_hash="s" * 64, bound_at=TS,
                               locator={"kind": "automation_id", "value": "btn-save",
                                        "observed_at": TS, "environment_version": "5.3.0"})
        raw = make_step("S1", "click", locator_status="fixed")
        raw.binding = observed
        out = MappingStage(make_snapshot(), TS).run([raw], platform="windows")
        assert out["status"] == "bound"
        step = out["bound_steps"][0]
        assert step.target["locator_status"] == "fixed"
        assert step.binding.locator == observed.locator
        assert any(d["code"] == "locator_fixed" for d in out["diagnostics"])

    def test_dependency_cycle_rejected_at_mapping(self):
        out = MappingStage(make_snapshot(), TS).run(
            [make_step("S1", depends_on=["S2"]), make_step("S2", depends_on=["S1"])],
            platform="windows")
        assert out["ok"] is False
        assert any(d["code"] == "dependency_cycle" for d in out["diagnostics"])

    def test_non_atomic_input_rejected(self):
        out = MappingStage(make_snapshot(), TS).run([{"step_id": "S1"}], platform="windows")
        assert out["ok"] is False
        assert any(d["code"] == "not_atomic_ir" for d in out["diagnostics"])

    def test_non_snapshot_rejected(self):
        with pytest.raises(ModelError, match="CapabilitySnapshot"):
            MappingStage({"snapshot_id": "fake"}, TS)

    def test_invalid_bound_at_rejected(self):
        with pytest.raises(ModelError, match="bound_at"):
            MappingStage(make_snapshot(), "2026-11-08 02:00:00")
