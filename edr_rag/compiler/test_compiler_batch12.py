"""P5 第四批次：§9 CLI/库入口接线核对 + compile/validate-plan/render-guide（design.md §9 / §10 P5）。

入口核对结论（离线静态核对，均已存在）：
- import-source → knowledge.document_ingest.import_document_workdir
- import-case   → knowledge.connector.import_case_fixture
- export-bundle → compiler.export_bundle.BundleExporter.export
本轮新增缺失的三个库入口（compiler/pipeline.py）：
- compile       → compile_workflow（阶段2–5 串联，不新增门禁）
- validate-plan → validate_plan（WorkflowRevision 结构复验）
- render-guide  → render_guide（RenderingStage 透传）

正例：
- 真实 browser SKILL.md 快照 → compile_workflow 全链路 ok，Workflow 静态
  validated、runtime_ready=False（必填变量未绑定值，§7）、hash 可回放；
- validate_plan 对 ok Workflow 无诊断、对被篡改 workflow 诊断非空 ok=False；
- render_guide 产物与 plan 同 workflow_hash、步骤一一对应。

负例：
- compile：证据缺失（用例未导入 repo）拒绝；ungrounded provider 拒绝
  （failed_stage=case_understanding）；unsupported 动作拒绝
  （failed_stage=mapping）；依赖环拒绝（failed_stage=decomposition）；
- validate_plan：非 Workflow/非法 dict → schema_violation，不算通过；
- render_guide：非法 workflow / 缺 tz 时间戳 → schema_violation。

全部为离线确定性验证；无实机执行。compile ok ≠ 实机 passed，
validate_plan ok ≠ published（编译器不自授发布，§8）。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from edr_rag.compiler.capability_snapshot import collect_browser_snapshot
from edr_rag.compiler.pipeline import compile_workflow, render_guide, validate_plan
from edr_rag.knowledge.connector import import_case_fixture
from edr_rag.knowledge.models import (
    CaseBundle,
    EvidenceRef,
    ModelError,
    TestCaseRevision,
    WorkflowRevision,
)
from edr_rag.compiler import OfflineStubProvider
from edr_rag.knowledge.connector import WatermarkState
from edr_rag.knowledge.repository import KnowledgeRepository

BROWSER_SKILL = Path("/Users/edr-test/.hermes/skills/browser-automation/SKILL.md")
browser_skill = pytest.mark.skipif(
    not BROWSER_SKILL.is_file(), reason="browser SKILL.md not present"
)

T0 = "2026-11-08T07:00:00+00:00"
PV = "web-testbed-1.0"
CASE_REF = "case-pipe-e2e@rev1:eeeeeeeeeeeeeeee"


def pipe_case() -> TestCaseRevision:
    return TestCaseRevision(
        case_id="case-pipe-e2e", product_version=PV, platform="web",
        title="配置服务器地址并保存",
        preconditions=["打开网络设置页面"],
        steps=[
            {"action": "打开网络设置页面", "expected": "页面加载完成"},
            {"action": "输入服务器地址并保存", "expected": "出现保存成功提示"},
        ],
        raw_text_hash="sha256:" + "3" * 64,
        source_ref=CASE_REF,
        created_at=T0,
    )


def pipe_evidence() -> list[EvidenceRef]:
    return [
        EvidenceRef("E-operation-1", CASE_REF, "steps[1].action", "打开网络设置页面", "operation"),
        EvidenceRef("E-operation-2", CASE_REF, "steps[2].action", "输入服务器地址并保存", "operation"),
        EvidenceRef("E-expected-2", CASE_REF, "steps[2].expected", "出现保存成功提示", "expected"),
    ]


def seeded_repo(tmp_path: Path) -> KnowledgeRepository:
    repo = KnowledgeRepository(tmp_path / "kb")
    repo.import_case(CaseBundle(case=pipe_case(), evidence=pipe_evidence()))
    return repo


def make_snapshot(snapshot_id: str = "pipe-snap-1"):
    snap, _prov = collect_browser_snapshot(product_version=PV, snapshot_id=snapshot_id, observed_at=T0)
    return snap


@browser_skill
class TestCompileWorkflow:
    def test_end_to_end_ok_with_unbound_runtime(self, tmp_path: Path):
        repo = seeded_repo(tmp_path)
        snap = make_snapshot()
        out = compile_workflow(
            repo=repo, case=pipe_case(),
            understanding_provider=OfflineStubProvider(),
            decomposition_provider=OfflineStubProvider(mode="decomposition"),
            snapshot=snap, workflow_id="wf-pipe-e2e", compiled_at=T0,
        )
        assert out["ok"] is True, out["diagnostics"]
        wf: WorkflowRevision = out["workflow"]
        # compile 产出初始治理态 draft；validated/published 是后续显式门禁/审核（§8）
        assert wf.status == "draft" and wf.execution_status == "not_run"
        assert wf.capability_snapshot_ref == snap.content_hash
        assert out["workflow_hash"] == wf.content_hash
        assert out["task"].source_case_id == "case-pipe-e2e"
        # 必填变量未绑定值：静态 validated 但 runtime_ready=False（§7）
        assert out["runtime_ready"] is False
        assert "server_address" in out["unbound_variables"]
        # 阶段链完整且逐步 ok
        assert list(out["stages"]) == ["case_understanding", "decomposition", "mapping", "validation"]
        assert all(s["ok"] for s in out["stages"].values())
        # Workflow 未绑定值不得冒充可下发：execution_status 独立 not_run
        assert wf.content_hash == WorkflowRevision.from_dict(wf.to_dict()).content_hash

    def test_recompile_same_input_same_hash(self, tmp_path: Path):
        repo = seeded_repo(tmp_path)
        snap = make_snapshot()
        kw = dict(
            repo=repo, case=pipe_case(),
            understanding_provider=OfflineStubProvider(),
            decomposition_provider=OfflineStubProvider(mode="decomposition"),
            snapshot=snap, workflow_id="wf-pipe-replay", compiled_at=T0,
        )
        out1 = compile_workflow(**kw)
        out2 = compile_workflow(**kw)
        assert out1["ok"] and out2["ok"]
        assert out1["workflow_hash"] == out2["workflow_hash"]

    def test_missing_evidence_rejected(self, tmp_path: Path):
        # 用例未导入 repo：无证据 → 断言/操作必须有证据，拒绝
        repo = KnowledgeRepository(tmp_path / "kb-empty")
        snap = make_snapshot()
        out = compile_workflow(
            repo=repo, case=pipe_case(),
            understanding_provider=OfflineStubProvider(),
            decomposition_provider=OfflineStubProvider(mode="decomposition"),
            snapshot=snap, workflow_id="wf-pipe-noev", compiled_at=T0,
        )
        assert out["ok"] is False
        assert out["workflow"] is None and out["workflow_hash"] is None
        assert out["failed_stage"] == "case_understanding"
        assert out["error"]  # evidence_of 的 ModelError 信息被捕获为 schema violation

    def test_ungrounded_understanding_rejected(self, tmp_path: Path):
        repo = seeded_repo(tmp_path)
        snap = make_snapshot()
        out = compile_workflow(
            repo=repo, case=pipe_case(),
            understanding_provider=OfflineStubProvider(tamper=True),
            decomposition_provider=OfflineStubProvider(mode="decomposition"),
            snapshot=snap, workflow_id="wf-pipe-tamper", compiled_at=T0,
        )
        assert out["ok"] is False
        assert out["failed_stage"] == "case_understanding"
        assert any(d["code"] == "ungrounded_step_range" for d in out["diagnostics"])
        assert out["workflow"] is None

    def test_unsupported_action_rejected_at_mapping(self, tmp_path: Path):
        # 用例动作词「勾选…」→ stub 产出 check；web 快照未声明 check 动作
        case = TestCaseRevision(
            case_id="case-pipe-click", product_version=PV, platform="web",
            title="勾选启用日志选项",
            preconditions=["已打开网络设置页面"],
            steps=[
                {"action": "勾选启用日志选项", "expected": "选项状态变为已启用"},
            ],
            raw_text_hash="sha256:" + "4" * 64,
            source_ref="case-pipe-click@rev1:ffffffffffffffff",
            created_at=T0,
        )
        evidence = [
            EvidenceRef("E-operation-1", case.source_ref, "steps[1].action", "勾选启用日志选项", "operation"),
            EvidenceRef("E-expected-1", case.source_ref, "steps[1].expected", "选项状态变为已启用", "expected"),
        ]
        repo = KnowledgeRepository(tmp_path / "kb-click")
        repo.import_case(CaseBundle(case=case, evidence=evidence))
        snap = make_snapshot("pipe-snap-click")
        out = compile_workflow(
            repo=repo, case=case,
            understanding_provider=OfflineStubProvider(),
            decomposition_provider=OfflineStubProvider(mode="decomposition"),
            snapshot=snap, workflow_id="wf-pipe-click", compiled_at=T0,
        )
        assert out["ok"] is False
        assert out["failed_stage"] == "mapping"
        assert any(d["code"] == "unsupported" for d in out["diagnostics"])
        assert out["workflow"] is None

    def test_dependency_cycle_rejected(self, tmp_path: Path):
        repo = seeded_repo(tmp_path)
        snap = make_snapshot()

        class CycleDecomp(OfflineStubProvider):
            def _complete_decomposition(self, user_content: str):
                resp = super()._complete_decomposition(user_content)
                import json as _json
                payload = _json.loads(resp.text)
                # S1 → S2 → S1 成环（阶段3 图校验在编译入口前拒绝）
                payload["steps"][0]["depends_on"] = ["S2"]
                return type(resp)(text=_json.dumps(payload, ensure_ascii=False),
                                  model=resp.model, finish_reason="stop")

        out = compile_workflow(
            repo=repo, case=pipe_case(),
            understanding_provider=OfflineStubProvider(),
            decomposition_provider=CycleDecomp(mode="decomposition"),
            snapshot=snap, workflow_id="wf-pipe-cycle", compiled_at=T0,
        )
        assert out["ok"] is False
        assert out["failed_stage"] == "decomposition"
        assert any(d["code"] == "dependency_cycle" for d in out["diagnostics"])

    def test_provider_failure_reported_not_raised(self, tmp_path: Path):
        repo = seeded_repo(tmp_path)
        snap = make_snapshot()
        out = compile_workflow(
            repo=repo, case=pipe_case(),
            understanding_provider=OfflineStubProvider(fail=True),
            decomposition_provider=OfflineStubProvider(mode="decomposition"),
            snapshot=snap, workflow_id="wf-pipe-fail", compiled_at=T0,
        )
        assert out["ok"] is False
        assert out["failed_stage"] == "case_understanding"
        assert out["stages"]["case_understanding"]["ok"] is False


@browser_skill
class TestValidatePlan:
    def _compiled(self, tmp_path: Path):
        repo = seeded_repo(tmp_path)
        snap = make_snapshot()
        out = compile_workflow(
            repo=repo, case=pipe_case(),
            understanding_provider=OfflineStubProvider(),
            decomposition_provider=OfflineStubProvider(mode="decomposition"),
            snapshot=snap, workflow_id="wf-pipe-vp", compiled_at=T0,
        )
        assert out["ok"] is True
        return out["workflow"]

    def test_valid_workflow_passes(self, tmp_path: Path):
        wf = self._compiled(tmp_path)
        res = validate_plan(wf)
        assert res["ok"] is True
        assert res["workflow_hash"] == wf.content_hash
        # compile 产物为 draft 治理态；validate_plan 只做结构复验不迁移状态（§8）
        assert res["status"] == "draft"
        assert res["diagnostics"] == []

    def test_dict_roundtrip_passes(self, tmp_path: Path):
        wf = self._compiled(tmp_path)
        res = validate_plan(wf.to_dict())
        assert res["ok"] is True
        assert res["workflow_hash"] == wf.content_hash

    def test_tampered_workflow_rejected(self, tmp_path: Path):
        wf = self._compiled(tmp_path)
        d = wf.to_dict()
        # 制造悬空依赖：引用不存在的步骤 → 结构复验必须拒绝
        d["steps"][0]["depends_on"] = ["S9"]
        res = validate_plan(d)
        assert res["ok"] is False
        assert any(dg["code"] == "dangling_dependency" for dg in res["diagnostics"])

    def test_invalid_input_is_schema_violation_not_pass(self, tmp_path: Path):
        for bad in (None, 42, "not-a-workflow", {"workflow_id": "wf-x"}):
            res = validate_plan(bad)
            assert res["ok"] is False
            assert res["diagnostics"][0]["code"] == "schema_violation"


@browser_skill
class TestRenderGuide:
    def _compiled(self, tmp_path: Path):
        repo = seeded_repo(tmp_path)
        snap = make_snapshot()
        out = compile_workflow(
            repo=repo, case=pipe_case(),
            understanding_provider=OfflineStubProvider(),
            decomposition_provider=OfflineStubProvider(mode="decomposition"),
            snapshot=snap, workflow_id="wf-pipe-rg", compiled_at=T0,
        )
        assert out["ok"] is True
        return out["workflow"]

    def test_guide_same_source_as_plan(self, tmp_path: Path):
        wf = self._compiled(tmp_path)
        rendered = render_guide(wf, rendered_at=T0)
        assert rendered["ok"] is True
        assert rendered["plan"]["workflow_hash"] == wf.content_hash
        assert rendered["guide"]["workflow_hash"] == wf.content_hash
        assert len(rendered["plan"]["steps"]) == len(rendered["guide"]["steps"]) == len(wf.steps)
        assert [s["step_id"] for s in rendered["plan"]["steps"]] == [s.step_id for s in wf.steps]
        assert [s["step_id"] for s in rendered["guide"]["steps"]] == [s.step_id for s in wf.steps]
        assert [s["expected"] for s in rendered["plan"]["steps"]] == [s["expected"] for s in rendered["guide"]["steps"]]

    def test_no_bound_values_in_rendered_output(self, tmp_path: Path):
        # stub 拆解默认不带参数；子类把已声明变量接到 input 步骤，验证
        # §6.6 渲染契约：参数只以变量引用出现，bound_values 不进入产物
        repo = seeded_repo(tmp_path)
        snap = make_snapshot()

        class VarDecomp(OfflineStubProvider):
            def _complete_decomposition(self, user_content: str):
                resp = super()._complete_decomposition(user_content)
                import json as _json
                payload = _json.loads(resp.text)
                for st in payload["steps"]:
                    if st["action"] == "input":
                        st["parameters"] = {"value": {"variable": "server_address"}}
                return type(resp)(text=_json.dumps(payload, ensure_ascii=False),
                                  model=resp.model, finish_reason="stop")

        out = compile_workflow(
            repo=repo, case=pipe_case(),
            understanding_provider=OfflineStubProvider(),
            decomposition_provider=VarDecomp(mode="decomposition"),
            snapshot=snap, workflow_id="wf-pipe-var", compiled_at=T0,
        )
        assert out["ok"] is True, out["diagnostics"]
        rendered = render_guide(out["workflow"], rendered_at=T0)
        assert rendered["ok"] is True
        input_steps = [s for s in rendered["plan"]["steps"] if s["action"] == "input"]
        assert input_steps and input_steps[0]["parameters"]["value"] == {
            "kind": "variable", "name": "server_address",
        }
        guide_input = [s for s in rendered["guide"]["steps"] if s["action"] == "input"]
        assert guide_input[0]["parameters"]["value"]["kind"] == "variable"

    def test_invalid_workflow_rejected(self, tmp_path: Path):
        res = render_guide({"workflow_id": "wf-bad"}, rendered_at=T0)
        assert res["ok"] is False
        assert res["stage"] == "render_guide"
        assert res["diagnostics"][0]["code"] == "schema_violation"

    def test_naive_timestamp_rejected(self, tmp_path: Path):
        wf = self._compiled(tmp_path)
        res = render_guide(wf, rendered_at="2026-11-08T07:00:00")
        assert res["ok"] is False
        assert res["stage"] == "render_guide"
