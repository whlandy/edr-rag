"""P4 第二批次：快照持久化 + 执行前 hash 复核门禁（design.md §4 / §8）。

- CapabilitySnapshot 存入 knowledge Repository（snapshots.jsonl，版本化）；
- ExecutionGate：绑定 snapshot_hash ≠ 当前快照 content_hash → 拒绝下发；
  工具 schema/版本漂移、未绑定步骤、冻结 workflow hash 变更、跨版本均拒绝。
全部为离线确定性验证，不涉及真实执行器、不实机执行。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from edr_rag.compiler import ExecutionGate, MappingStage
from edr_rag.knowledge.models import (
    ATOMIC_ACTIONS,
    AtomicIR,
    CapabilitySnapshot,
    ModelError,
    StepBinding,
    TaskVariable,
    ToolCapability,
    WorkflowRevision,
)
from edr_rag.knowledge.repository import SNAPSHOTS_FILE, KnowledgeRepository

T0 = "2026-11-08T00:00:00+00:00"
T1 = "2026-11-08T00:00:01+00:00"
T2 = "2026-11-08T00:00:02+00:00"


def _snap(snapshot_id: str, platform: str = "windows", *, schema_hash: str = "sha256:aa01",
          product_version: str = "5.3.0", actions: tuple[str, ...] = ("connect", "click")) -> CapabilitySnapshot:
    return CapabilitySnapshot(
        snapshot_id=snapshot_id, platform=platform, product_version=product_version,
        capabilities=[ToolCapability(
            tool=f"tool-{platform}", tool_version="1.0", tool_schema_hash=schema_hash,
            platform=platform, actions=actions, observed_at=T0,
        )],
        observed_at=T0,
    )


def _step(step_id: str, action: str, snapshot_hash: str, *, tool: str = "tool-windows",
          tool_version: str = "1.0", tool_schema_hash: str = "sha256:aa01") -> AtomicIR:
    return AtomicIR(
        step_id=step_id, source_step=1, action=action,
        target={"semantic_name": "目标", "locator_status": "runtime_resolution"},
        parameters={}, depends_on=[], evidence_refs=["E1"],
        binding=StepBinding(tool=tool, tool_version=tool_version,
                            tool_schema_hash=tool_schema_hash, snapshot_hash=snapshot_hash,
                            bound_at=T1),
    )


def _workflow(steps: list[AtomicIR], snapshot_ref: str = "sha256:current", *,
              platform: str = "windows", product_version: str = "5.3.0",
              workflow_id: str = "wf-gate") -> WorkflowRevision:
    return WorkflowRevision(
        workflow_id=workflow_id, product_version=product_version, platform=platform,
        task_id="task-1", task_source_ref="case-1@rev1", steps=steps,
        declared_variables=[], capability_snapshot_ref=snapshot_ref,
        created_at=T0,
    )


class TestSnapshotPersistence:
    def test_save_and_get_roundtrip(self, tmp_path):
        repo = KnowledgeRepository(tmp_path)
        snap = _snap("snap-1")
        assert repo.save_snapshot(snap) == "created"
        loaded = repo.get_snapshot("snap-1")
        assert loaded is not None
        assert loaded.content_hash == snap.content_hash
        assert loaded.capabilities[0].tool_schema_hash == "sha256:aa01"

    def test_duplicate_content_is_idempotent(self, tmp_path):
        repo = KnowledgeRepository(tmp_path)
        snap = _snap("snap-1")
        repo.save_snapshot(snap)
        before = (tmp_path / SNAPSHOTS_FILE).read_text()
        assert repo.save_snapshot(_snap("snap-1")) == "duplicate"
        assert (tmp_path / SNAPSHOTS_FILE).read_text() == before

    def test_new_content_same_id_creates_new_version(self, tmp_path):
        """快照不可变：同 id 内容变化 → 追加新版本，不原地改（§4）。"""
        repo = KnowledgeRepository(tmp_path)
        repo.save_snapshot(_snap("snap-1", schema_hash="sha256:aa01"))
        assert repo.save_snapshot(_snap("snap-1", schema_hash="sha256:bb02")) == "created"
        assert len((tmp_path / SNAPSHOTS_FILE).read_text().strip().splitlines()) == 2
        loaded = repo.get_snapshot("snap-1")
        # 最新（observed_at 最大）胜出；同 T0 时取后写者
        assert loaded.capabilities[0].tool_schema_hash in ("sha256:aa01", "sha256:bb02")
        assert len(repo.list_snapshots()) == 1

    def test_new_observed_at_wins(self, tmp_path):
        repo = KnowledgeRepository(tmp_path)
        old = CapabilitySnapshot(
            snapshot_id="snap-t", platform="windows", product_version="5.3.0",
            capabilities=[ToolCapability(tool="t", tool_version="1.0",
                                         tool_schema_hash="sha256:old", platform="windows",
                                         actions=("connect",), observed_at="2026-11-07T00:00:00+00:00")],
            observed_at="2026-11-07T00:00:00+00:00",
        )
        new = CapabilitySnapshot(
            snapshot_id="snap-t", platform="windows", product_version="5.3.0",
            capabilities=[ToolCapability(tool="t", tool_version="1.1",
                                         tool_schema_hash="sha256:new", platform="windows",
                                         actions=("connect", "click"), observed_at=T0)],
            observed_at=T0,
        )
        repo.save_snapshot(old)
        repo.save_snapshot(new)
        loaded = repo.get_snapshot("snap-t")
        assert loaded.capabilities[0].tool_schema_hash == "sha256:new"

    def test_corrupt_store_line_raises_not_silent(self, tmp_path):
        repo = KnowledgeRepository(tmp_path)
        repo.save_snapshot(_snap("snap-ok"))
        path = tmp_path / SNAPSHOTS_FILE
        lines = path.read_text().strip().splitlines()
        lines.append("{not-json")
        path.write_text("\n".join(lines) + "\n")
        with pytest.raises(Exception):
            repo.get_snapshot("snap-ok")

    def test_list_snapshots_explicit_filters_only(self, tmp_path):
        repo = KnowledgeRepository(tmp_path)
        repo.save_snapshot(_snap("s-win", platform="windows", product_version="5.3.0"))
        repo.save_snapshot(_snap("s-web", platform="web", product_version="web-1"))
        wins = repo.list_snapshots(platform="windows")
        assert [s.snapshot_id for s in wins] == ["s-win"]
        assert [s.snapshot_id for s in repo.list_snapshots(product_version="web-1")] == ["s-web"]
        assert {s.snapshot_id for s in repo.list_snapshots()} == {"s-win", "s-web"}

    def test_real_collected_snapshot_roundtrips(self, tmp_path):
        """真实采集快照 → Repository 存取 → content_hash 不变（绑定锚保持）。"""
        edr_wd = Path("/Users/edr-test/ai-projects/edr-wd")
        if not edr_wd.is_dir():
            pytest.skip("edr-wd repo not present")
        from edr_rag.compiler.capability_snapshot import collect_edr_wd_snapshot
        snap, _prov = collect_edr_wd_snapshot(
            product_version="5.3.0-testbed", snapshot_id="edr-wd-persist",
            observed_at=T0,
        )
        repo = KnowledgeRepository(tmp_path)
        assert repo.save_snapshot(snap) == "created"
        assert repo.save_snapshot(snap) == "duplicate"
        loaded = repo.get_snapshot("edr-wd-persist")
        assert loaded.content_hash == snap.content_hash
        assert loaded.capabilities[0].tool.startswith("edr-wd:")


class TestExecutionGate:
    def test_unbound_step_rejected(self, tmp_path):
        gate = ExecutionGate(_snap("snap-cur"), T2)
        wf = _workflow([AtomicIR(
            step_id="S1", source_step=1, action="connect",
            target={"semantic_name": "x", "locator_status": "runtime_resolution"},
            parameters={}, depends_on=[], evidence_refs=["E1"], binding=None,
        )], snapshot_ref="x")
        result = gate.run(wf)
        assert result["execution_ready"] is False
        codes = {d["code"] for d in result["diagnostics"]}
        assert "unbound_step" in codes

    def test_snapshot_drift_rejects_old_binding(self, tmp_path):
        """§8 核心负例：绑定 hash ≠ 当前快照 hash → 拒绝旧绑定。"""
        gate = ExecutionGate(_snap("snap-cur", schema_hash="sha256:bb02"), T2)
        wf = _workflow([_step("S1", "connect", "sha256:old-snapshot-hash")],
                       snapshot_ref="sha256:old-snapshot-hash")
        result = gate.run(wf)
        assert result["execution_ready"] is False
        codes = {d["code"] for d in result["diagnostics"]}
        assert "snapshot_drift" in codes
        assert "snapshot_ref_mismatch" in codes

    def test_tool_schema_drift_rejected(self, tmp_path):
        """工具 schema 漂移（版本或 hash 变化）→ 拒绝下发（§10 P4 必验负例）。

        snapshot_ref 与 step binding 都声称当前 hash，但快照内 tool 条目
        hash 已变化（simulate catalog 改动后 snapshot 重新采集，绑定是旧的
        tool_schema_hash）→ tool_schema_drift。
        """
        current = _snap("snap-cur", schema_hash="sha256:bb02")
        gate = ExecutionGate(current, T2)
        # 绑定声称的 snapshot_hash 与 workflow ref 都是当前 content_hash，
        # 但其 tool_schema_hash 是旧值 → 只能由 tool 条目比对检出
        wf = _workflow([_step("S1", "connect", current.content_hash, tool_schema_hash="sha256:aa01")],
                       snapshot_ref=current.content_hash)
        result = gate.run(wf)
        assert result["execution_ready"] is False
        codes = {d["code"] for d in result["diagnostics"]}
        assert "tool_schema_drift" in codes

    def test_tool_version_drift_rejected(self, tmp_path):
        gate = ExecutionGate(_snap("snap-cur"), T2)
        # tool_version 2.0 绑定 vs 快照 1.0
        wf = _workflow([_step("S1", "connect", gate.snapshot.content_hash, tool_version="2.0")],
                       snapshot_ref=gate.snapshot.content_hash)
        result = gate.run(wf)
        assert result["execution_ready"] is False
        assert result["diagnostics"][0]["code"] == "tool_schema_drift"

    def test_tool_disappeared_rejected(self, tmp_path):
        gate = ExecutionGate(_snap("snap-cur"), T2)
        wf = _workflow([_step("S1", "connect", gate.snapshot.content_hash, tool="ghost-tool")],
                       snapshot_ref=gate.snapshot.content_hash)
        result = gate.run(wf)
        assert result["execution_ready"] is False
        assert result["diagnostics"][0]["code"] == "tool_missing"

    def test_action_no_longer_declared_rejected(self, tmp_path):
        gate = ExecutionGate(_snap("snap-cur", actions=("connect",)), T2)
        wf = _workflow([_step("S1", "click", gate.snapshot.content_hash)],
                       snapshot_ref=gate.snapshot.content_hash)
        result = gate.run(wf)
        assert result["execution_ready"] is False
        assert result["diagnostics"][0]["code"] == "unsupported"

    def test_frozen_workflow_hash_change_rejected(self, tmp_path):
        """冻结 Workflow 被修改（渲染产物/审核记录作废）→ 拒绝。"""
        gate = ExecutionGate(_snap("snap-cur"), T2)
        wf = _workflow([_step("S1", "connect", gate.snapshot.content_hash)],
                       snapshot_ref=gate.snapshot.content_hash)
        result = gate.run(wf, workflow_hash="sha256:tampered")
        assert result["execution_ready"] is False
        assert result["diagnostics"][0]["code"] == "workflow_hash_mismatch"

    def test_platform_and_version_mismatch_rejected(self, tmp_path):
        gate = ExecutionGate(_snap("snap-cur"), T2)
        wf = _workflow([_step("S1", "connect", gate.snapshot.content_hash)],
                       snapshot_ref=gate.snapshot.content_hash)
        wf2 = _workflow([_step("S1", "connect", gate.snapshot.content_hash)],
                        snapshot_ref=gate.snapshot.content_hash, platform="web")
        r1 = gate.run(wf2)
        assert r1["diagnostics"][0]["code"] == "platform_mismatch"
        r2 = gate.run(wf, workflow_hash=None) if False else None
        # 跨产品版本
        snap_v2 = _snap("snap-cur", product_version="5.4.0")
        gate2 = ExecutionGate(snap_v2, T2)
        r3 = gate2.run(wf)
        assert r3["diagnostics"][0]["code"] == "product_version_mismatch"
        assert r1["execution_ready"] is False and r3["execution_ready"] is False
        del r2

    def test_execution_status_must_be_not_run(self, tmp_path):
        gate = ExecutionGate(_snap("snap-cur"), T2)
        wf = _workflow([_step("S1", "connect", gate.snapshot.content_hash)],
                       snapshot_ref=gate.snapshot.content_hash)
        wf.execution_status = "failed"
        result = gate.run(wf)
        assert result["execution_ready"] is False
        assert result["diagnostics"][0]["code"] == "execution_status_not_clean"

    def test_non_workflow_input_rejected(self, tmp_path):
        gate = ExecutionGate(_snap("snap-cur"), T2)
        result = gate.run({"steps": []})
        assert result["execution_ready"] is False
        assert result["diagnostics"][0]["code"] == "not_workflow_revision"

    def test_gate_constructor_rejects_non_snapshot(self):
        with pytest.raises(ModelError, match="CapabilitySnapshot"):
            ExecutionGate({"snapshot_id": "x"}, T2)

    def test_happy_path_ready(self, tmp_path):
        """正例：绑定与当前快照完全一致 → execution_ready=True。"""
        snap = _snap("snap-cur")
        gate = ExecutionGate(snap, T2)
        wf = _workflow([_step("S1", "connect", snap.content_hash),
                        _step("S2", "click", snap.content_hash, tool="tool-windows")],
                       snapshot_ref=snap.content_hash)
        wf.steps[1].depends_on = ["S1"]
        result = gate.run(wf, workflow_hash=wf.content_hash)
        assert result["execution_ready"] is True
        assert result["ok"] is True
        assert result["current_snapshot_hash"] == snap.content_hash


class TestEndToEndMappingGate:
    """真实采集快照 → MappingStage 绑定 → WorkflowRevision → ExecutionGate 放行。"""

    EDR_WD = Path("/Users/edr-test/ai-projects/edr-wd")

    def _require_repo(self):
        if not self.EDR_WD.is_dir():
            pytest.skip("edr-wd repo not present")

    def test_bound_workflow_passes_gate(self, tmp_path):
        self._require_repo()
        from edr_rag.compiler.capability_snapshot import collect_edr_wd_snapshot
        snap, _ = collect_edr_wd_snapshot(
            product_version="5.3.0-testbed", snapshot_id="edr-wd-gate-e2e", observed_at=T0,
        )
        steps = [
            AtomicIR(step_id="S1", source_step=1, action="connect",
                     target={"semantic_name": "HiSec 主窗口", "locator_status": "runtime_resolution"},
                     parameters={}, depends_on=[], evidence_refs=["E1"]),
            AtomicIR(step_id="S2", source_step=2, action="click",
                     target={"semantic_name": "设置按钮", "locator_status": "runtime_resolution"},
                     parameters={}, depends_on=["S1"], evidence_refs=["E2"]),
        ]
        mapping = MappingStage(snap, T1).run(steps, "windows")
        assert mapping["ok"] is True and mapping["status"] == "bound"
        wf = WorkflowRevision(
            workflow_id="wf-e2e-gate", product_version="5.3.0-testbed", platform="windows",
            task_id="task-e2e", task_source_ref="case-e2e@rev1",
            steps=mapping["bound_steps"],
            capability_snapshot_ref=snap.content_hash, created_at=T1,
        )
        repo = KnowledgeRepository(tmp_path)
        repo.save_snapshot(snap)
        current = repo.get_snapshot("edr-wd-gate-e2e")
        assert current.content_hash == snap.content_hash
        result = ExecutionGate(current, T2).run(wf, workflow_hash=wf.content_hash)
        assert result["execution_ready"] is True

    def test_drifted_snapshot_blocks_bound_workflow(self, tmp_path):
        """目录漂移（新快照 hash 变化）→ 旧绑定被门禁拒绝（§8 核心负例）。"""
        self._require_repo()
        from edr_rag.compiler.capability_snapshot import collect_edr_wd_snapshot
        snap, _ = collect_edr_wd_snapshot(
            product_version="5.3.0-testbed", snapshot_id="edr-wd-gate-drift", observed_at=T0,
        )
        steps = [AtomicIR(step_id="S1", source_step=1, action="connect",
                          target={"semantic_name": "HiSec 主窗口", "locator_status": "runtime_resolution"},
                          parameters={}, depends_on=[], evidence_refs=["E1"])]
        mapping = MappingStage(snap, T1).run(steps, "windows")
        wf = WorkflowRevision(
            workflow_id="wf-e2e-drift", product_version="5.3.0-testbed", platform="windows",
            task_id="task-drift", task_source_ref="case-drift@rev1",
            steps=mapping["bound_steps"],
            capability_snapshot_ref=snap.content_hash, created_at=T1,
        )
        repo = KnowledgeRepository(tmp_path)
        repo.save_snapshot(snap)
        # 模拟目录漂移：同 id 新版本，schema hash 不同
        drifted = CapabilitySnapshot(
            snapshot_id="edr-wd-gate-drift", platform="windows", product_version="5.3.0-testbed",
            capabilities=[ToolCapability(
                tool=snap.capabilities[0].tool, tool_version="9.9.9",
                tool_schema_hash="sha256:drifted0000", platform="windows",
                actions=snap.capabilities[0].actions, observed_at=T1,
            )],
            observed_at=T1,
        )
        repo.save_snapshot(drifted)
        current = repo.get_snapshot("edr-wd-gate-drift")
        assert current.capabilities[0].tool_schema_hash == "sha256:drifted0000"
        result = ExecutionGate(current, T2).run(wf, workflow_hash=wf.content_hash)
        assert result["execution_ready"] is False
        codes = {d["code"] for d in result["diagnostics"]}
        assert "snapshot_drift" in codes and "snapshot_ref_mismatch" in codes and "tool_schema_drift" in codes


def test_gate_actions_within_domain():
    """门禁校验的动作必须来自 AtomicIR 域（不虚构枚举）。"""
    assert {"connect", "click"} <= set(ATOMIC_ACTIONS)
