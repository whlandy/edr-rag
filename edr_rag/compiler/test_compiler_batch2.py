"""P3 batch 2: Decomposition stage tests (design.md §6.3).

正例：TaskIR+用例 → AtomicIR 草案、inference 只在导航类且附依据、schema 校验
构造 AtomicIR、输入 hash 绑定。负例（P3 验收 §10）：
- 越界/编造 source_step 拒绝（不完整用例不编造）；
- 静默删步骤（期望覆盖缺失）拒绝；
- 无来源业务操作（semantic_name 不可追溯）拒绝；
- inference 标在业务动作 / 无依据拒绝；
- gaps 与 validated 并存（谎报）拒绝；
- 悬空依赖、未知枚举、binding 非 null 等 schema 违例拒绝；
- provider 失败/不可解析输出不假通过。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from edr_rag.compiler import (
    DECOMPOSITION_PROMPT_VERSION,
    DECOMPOSITION_SCHEMA_VERSION,
    DecompositionStage,
    OfflineStubProvider,
    stage_input_hash,
)
from edr_rag.knowledge import (
    ATOMIC_ACTIONS,
    AtomicIR,
    CaseBundle,
    EvidenceRef,
    KnowledgeRepository,
    ModelError,
    NAV_ATOMIC_ACTIONS,
    SourceRevision,
    TaskIR,
    TaskVariable,
    TestCaseRevision,
    content_hash,
)

TS = "2026-11-08T01:00:00+08:00"


def make_source(source_id: str = "src-p3-2", h: str = "d" * 64) -> SourceRevision:
    return SourceRevision(
        source_id=source_id, format="platform-json", uri="offline://fixtures/p3b.json",
        source_hash=h, source_updated_at=TS, collected_at=TS,
    )


def make_case(source: SourceRevision, **overrides) -> TestCaseRevision:
    fields = dict(
        case_id="case-p3-201", product_version="5.3.0", platform="windows",
        title="配置服务器地址",
        preconditions=["已登录管理端"],
        steps=[
            {"action": "打开网络设置页", "expected": "显示服务器地址输入框"},
            {"action": "输入服务器地址具体值并保存", "expected": "提示保存成功"},
        ],
        raw_text_hash=content_hash({"raw": "p3b"}), source_ref=source.ref, created_at=TS,
    )
    fields.update(overrides)
    return TestCaseRevision(**fields)


def make_evidence(source: SourceRevision) -> list[EvidenceRef]:
    return [
        EvidenceRef("E-operation-1", source.ref, "steps[0]", "打开网络设置页", "operation"),
        EvidenceRef("E-expected-1", source.ref, "steps[0]", "显示服务器地址输入框", "expected"),
        EvidenceRef("E-operation-2", source.ref, "steps[1]", "输入服务器地址具体值并保存", "operation"),
        EvidenceRef("E-expected-2", source.ref, "steps[1]", "提示保存成功", "expected"),
    ]


def make_task(source: SourceRevision, **overrides) -> TaskIR:
    fields = dict(
        task_id="task-p3-201", product_version="5.3.0", platform="windows",
        goal="配置服务器地址并确认保存成功",
        source_case_id="case-p3-201", source_case_ref=source.ref,
        source_evidence_ids=["E-operation-1", "E-expected-1", "E-operation-2", "E-expected-2"],
        preconditions=["已登录管理端"],
        variables=[TaskVariable("server_address", "string")],
        business_result="提示保存成功", ambiguities=["step 2: 服务器地址具体值未给出"],
        created_at=TS,
    )
    fields.update(overrides)
    return TaskIR(**fields)


def seeded_repo(tmp_path: Path, source: SourceRevision, case: TestCaseRevision) -> KnowledgeRepository:
    repo = KnowledgeRepository(tmp_path / "kb")
    repo.import_case(CaseBundle(case=case, evidence=make_evidence(source)))
    return repo


def stub_decomp(**kw) -> OfflineStubProvider:
    return OfflineStubProvider(mode="decomposition", **kw)


class TestDecompositionHappyPath:
    def test_stub_draft_grounded_and_schema_valid(self, tmp_path: Path):
        s = make_source()
        case, task = make_case(s), make_task(s)
        stage = DecompositionStage(stub_decomp(), case, task)
        out = stage.run()
        assert out["ok"] is True
        assert out["schema_version"] == DECOMPOSITION_SCHEMA_VERSION
        assert out["prompt_version"] == DECOMPOSITION_PROMPT_VERSION
        assert len(out["input_hash"]) == 64
        steps: list[AtomicIR] = out["steps"]
        assert [st.source_step for st in steps] == [1, 2]
        assert all(isinstance(st, AtomicIR) for st in steps)
        assert all(st.action in ATOMIC_ACTIONS for st in steps)
        assert out["status"] == "needs_review" and out["gaps"]  # 歧义不假通过

    def test_navigate_carries_inference_with_basis(self, tmp_path: Path):
        s = make_source()
        stage = DecompositionStage(stub_decomp(), make_case(s), make_task(s))
        out = stage.run()
        nav = out["steps"][0]
        assert nav.action == "navigate"
        assert nav.inference is not None and nav.inference.get("basis")
        biz = out["steps"][1]
        assert biz.action != "navigate"
        assert biz.inference is None  # 业务动作不推断

    def test_expected_coverage_complete(self, tmp_path: Path):
        s = make_source()
        stage = DecompositionStage(stub_decomp(), make_case(s), make_task(s))
        out = stage.run()
        assert {st.source_step for st in out["steps"]} == {1, 2}

    def test_input_hash_binds_task_and_case(self, tmp_path: Path):
        s = make_source()
        case, task = make_case(s), make_task(s)
        out = DecompositionStage(stub_decomp(), case, task).run()
        expected = stage_input_hash({
            "schema_version": DECOMPOSITION_SCHEMA_VERSION,
            "prompt_version": DECOMPOSITION_PROMPT_VERSION,
            "case_source_ref": s.ref,
            "task": task.to_dict(),
            "case": case.to_dict(),
        })
        assert out["input_hash"] == expected
        # 输入变化 → hash 变化（阶段输入绑定）
        task2 = make_task(s, goal="不同的目标")
        out2 = DecompositionStage(stub_decomp(), case, task2).run()
        assert out2["input_hash"] != out["input_hash"]

    def test_roundtrip_preserves_atomic_steps(self, tmp_path: Path):
        s = make_source()
        out = DecompositionStage(stub_decomp(), make_case(s), make_task(s)).run()
        d = [st.to_dict() for st in out["steps"]]
        assert [AtomicIR.from_dict(x) for x in d] == out["steps"]


class TestDecompositionNegative:
    def test_invented_step_outside_case_rejected(self, tmp_path: Path):
        s = make_source()
        stage = DecompositionStage(stub_decomp(tamper=True), make_case(s), make_task(s))
        out = stage.run()
        assert out["ok"] is False
        assert any(d["code"] == "ungrounded_step_range" for d in out["diagnostics"])

    def test_silently_dropped_step_rejected(self, tmp_path: Path):
        """§6.5 静默删步骤在拆解层提前拒绝。"""
        s = make_source()
        case, task = make_case(s), make_task(s)

        class SkipProvider(OfflineStubProvider):
            def complete(self, messages, max_tokens: int = 1024, temperature: float = 0.0):
                resp = super().complete(messages, max_tokens, temperature)
                payload = json.loads(resp.text)
                payload["steps"] = payload["steps"][:1]
                resp.text = json.dumps(payload, ensure_ascii=False)
                return resp

        out = DecompositionStage(SkipProvider(mode="decomposition"), case, task).run()
        assert out["ok"] is False
        assert any(d["code"] == "missing_source_step_coverage" and "2" in d["message"]
                   for d in out["diagnostics"])

    def test_untraceable_business_operation_rejected(self, tmp_path: Path):
        """无来源业务操作拒绝：semantic_name 在用例原文中找不到。"""
        s = make_source()
        case, task = make_case(s), make_task(s)

        class FabricateProvider(OfflineStubProvider):
            def complete(self, messages, max_tokens: int = 1024, temperature: float = 0.0):
                resp = super().complete(messages, max_tokens, temperature)
                payload = json.loads(resp.text)
                payload["steps"][1]["target"]["semantic_name"] = "重启整个防火墙服务"
                resp.text = json.dumps(payload, ensure_ascii=False)
                return resp

        out = DecompositionStage(FabricateProvider(mode="decomposition"), case, task).run()
        assert out["ok"] is False
        assert any(d["code"] == "ungrounded_target_text" for d in out["diagnostics"])

    def test_inference_on_business_action_rejected(self, tmp_path: Path):
        s = make_source()
        case, task = make_case(s), make_task(s)

        class InferProvider(OfflineStubProvider):
            def complete(self, messages, max_tokens: int = 1024, temperature: float = 0.0):
                resp = super().complete(messages, max_tokens, temperature)
                payload = json.loads(resp.text)
                payload["steps"][1]["inference"] = {"basis": "猜的", "rationale": "r"}
                resp.text = json.dumps(payload, ensure_ascii=False)
                return resp

        out = DecompositionStage(InferProvider(mode="decomposition"), case, task).run()
        assert out["ok"] is False
        assert any(d["code"] == "inference_on_business_action" for d in out["diagnostics"])

    def test_navigation_inference_without_basis_rejected(self, tmp_path: Path):
        s = make_source()
        case, task = make_case(s), make_task(s)

        class NoBasisProvider(OfflineStubProvider):
            def complete(self, messages, max_tokens: int = 1024, temperature: float = 0.0):
                resp = super().complete(messages, max_tokens, temperature)
                payload = json.loads(resp.text)
                payload["steps"][0]["inference"] = {"rationale": "无依据"}
                resp.text = json.dumps(payload, ensure_ascii=False)
                return resp

        out = DecompositionStage(NoBasisProvider(mode="decomposition"), case, task).run()
        assert out["ok"] is False
        assert any(d["code"] == "inference_without_basis" for d in out["diagnostics"])

    def test_gaps_with_validated_status_rejected(self, tmp_path: Path):
        s = make_source()
        case, task = make_case(s), make_task(s)

        class LyingProvider(OfflineStubProvider):
            def complete(self, messages, max_tokens: int = 1024, temperature: float = 0.0):
                resp = super().complete(messages, max_tokens, temperature)
                payload = json.loads(resp.text)
                payload["gaps"] = ["参数缺失"]
                payload["status"] = "validated"
                resp.text = json.dumps(payload, ensure_ascii=False)
                return resp

        out = DecompositionStage(LyingProvider(mode="decomposition"), case, task).run()
        assert out["ok"] is False
        assert any(d["code"] == "ungrounded_status" for d in out["diagnostics"])

    def test_dangling_dependency_rejected(self, tmp_path: Path):
        s = make_source()
        case, task = make_case(s), make_task(s)

        class DepProvider(OfflineStubProvider):
            def complete(self, messages, max_tokens: int = 1024, temperature: float = 0.0):
                resp = super().complete(messages, max_tokens, temperature)
                payload = json.loads(resp.text)
                payload["steps"][1]["depends_on"] = ["S999"]
                resp.text = json.dumps(payload, ensure_ascii=False)
                return resp

        out = DecompositionStage(DepProvider(mode="decomposition"), case, task).run()
        assert out["ok"] is False
        assert any(d["code"] == "dangling_dependency" for d in out["diagnostics"])

    def test_dependency_cycle_rejected_at_stage(self, tmp_path: Path):
        """§6.5 依赖无环：环在拆解层即拒绝（接线 validate_step_graph）。"""
        s = make_source()
        case, task = make_case(s), make_task(s)

        class CycleProvider(OfflineStubProvider):
            def complete(self, messages, max_tokens: int = 1024, temperature: float = 0.0):
                resp = super().complete(messages, max_tokens, temperature)
                payload = json.loads(resp.text)
                payload["steps"][0]["depends_on"] = ["S2"]
                payload["steps"][1]["depends_on"] = ["S1"]
                resp.text = json.dumps(payload, ensure_ascii=False)
                return resp

        out = DecompositionStage(CycleProvider(mode="decomposition"), case, task).run()
        assert out["ok"] is False
        assert any(d["code"] == "dependency_cycle" for d in out["diagnostics"])

    def test_forward_reference_dependency_allowed(self, tmp_path: Path):
        """S2 depends_on S1 是正常 DAG；S1 depends_on S2（前向）不算悬空，
        由 validate_step_graph 的整体图校验判定（无环即通过）。"""
        s = make_source()
        case, task = make_case(s), make_task(s)

        class FwdProvider(OfflineStubProvider):
            def complete(self, messages, max_tokens: int = 1024, temperature: float = 0.0):
                resp = super().complete(messages, max_tokens, temperature)
                payload = json.loads(resp.text)
                payload["steps"][0]["depends_on"] = ["S2"]  # 前向引用，非环
                payload["steps"][1]["depends_on"] = []      # 清掉默认 S2→S1，避免真环
                resp.text = json.dumps(payload, ensure_ascii=False)
                return resp

        out = DecompositionStage(FwdProvider(mode="decomposition"), case, task).run()
        assert out["ok"] is True  # 无环即合法 DAG，语义排序属 Mapping 后续阶段

    def test_unknown_action_enum_rejected(self, tmp_path: Path):
        s = make_source()
        case, task = make_case(s), make_task(s)

        class BadActionProvider(OfflineStubProvider):
            def complete(self, messages, max_tokens: int = 1024, temperature: float = 0.0):
                resp = super().complete(messages, max_tokens, temperature)
                payload = json.loads(resp.text)
                payload["steps"][1]["action"] = "regedit"
                resp.text = json.dumps(payload, ensure_ascii=False)
                return resp

        out = DecompositionStage(BadActionProvider(mode="decomposition"), case, task).run()
        assert out["ok"] is False
        assert out["diagnostics"][0]["code"] == "schema_violation"
        assert "ATOMIC_ACTIONS" in out["diagnostics"][0]["message"]

    def test_non_null_binding_rejected(self, tmp_path: Path):
        s = make_source()
        case, task = make_case(s), make_task(s)

        class BindProvider(OfflineStubProvider):
            def complete(self, messages, max_tokens: int = 1024, temperature: float = 0.0):
                resp = super().complete(messages, max_tokens, temperature)
                payload = json.loads(resp.text)
                payload["steps"][0]["binding"] = {"tool": "edr_wd.click"}
                resp.text = json.dumps(payload, ensure_ascii=False)
                return resp

        out = DecompositionStage(BindProvider(mode="decomposition"), case, task).run()
        assert out["ok"] is False
        assert any("binding" in d["message"] for d in out["diagnostics"])

    def test_empty_steps_rejected(self, tmp_path: Path):
        s = make_source()
        case, task = make_case(s), make_task(s)

        class EmptyProvider(OfflineStubProvider):
            def complete(self, messages, max_tokens: int = 1024, temperature: float = 0.0):
                return super().complete(messages, max_tokens, temperature)

        out = DecompositionStage(EmptyProvider(mode="decomposition"), case, task).run()
        assert out["ok"] is True  # stub 正常输出两步
        class _None:
            pass
        stage = DecompositionStage(_None(), case, task)
        diags = stage._deterministic_diagnostics({"steps": [], "gaps": [], "status": "needs_review"})
        assert any(d["code"] == "missing_or_empty_steps" for d in diags)

    def test_failed_provider_no_fake_pass(self, tmp_path: Path):
        s = make_source()
        out = DecompositionStage(stub_decomp(fail=True), make_case(s), make_task(s)).run()
        assert out["ok"] is False
        assert "fail" in out["error"]
        assert out["input_hash"]

    def test_unparseable_response_no_fake_pass(self, tmp_path: Path):
        from edr_rag.llm.base import LLMResponse

        class JunkProvider:
            stub = True
            name = "junk"
            def complete(self, messages, max_tokens=1024, temperature=0.0):
                return LLMResponse(text="这不是 JSON", model="junk", finish_reason="stop")

        s = make_source()
        out = DecompositionStage(JunkProvider(), make_case(s), make_task(s)).run()
        assert out["ok"] is False
        assert "JSON" in out["error"]


class TestModelLevelInferenceRule:
    def test_inference_on_business_action_model_rejected(self):
        with pytest.raises(ModelError, match="navigation"):
            AtomicIR(
                step_id="S1", source_step=1, action="input",
                target={"semantic_name": "服务器地址", "locator_status": "runtime_resolution"},
                evidence_refs=["E-1"],
                inference={"basis": "b", "rationale": "r"},
            )

    def test_inference_without_basis_model_rejected(self):
        with pytest.raises(ModelError, match="basis"):
            AtomicIR(
                step_id="S1", source_step=1, action="navigate",
                target={"semantic_name": "设置页", "locator_status": "runtime_resolution"},
                evidence_refs=["E-1"],
                inference={"rationale": "无依据"},
            )

    def test_navigation_inference_roundtrip(self):
        s = AtomicIR(
            step_id="S1", source_step=1, action="navigate",
            target={"semantic_name": "设置页", "locator_status": "runtime_resolution"},
            evidence_refs=["E-1"],
            inference={"basis": "前置条件要求", "rationale": "r"},
        )
        d = s.to_dict()
        assert AtomicIR.from_dict(d).inference == {"basis": "前置条件要求", "rationale": "r"}
        assert AtomicIR.from_dict(d).to_dict() == d

    def test_nav_atomic_actions_subset(self):
        assert set(NAV_ATOMIC_ACTIONS) <= set(ATOMIC_ACTIONS)
        assert "click" not in NAV_ATOMIC_ACTIONS and "input" not in NAV_ATOMIC_ACTIONS
