"""P1 third batch: AtomicIR + WorkflowRevision (design.md §4/§7/§8).

正例：DAG 构建、roundtrip、仓库持久化与幂等。负例必须覆盖：未知 action、
悬空依赖、依赖环、自依赖、未声明变量、published 自授发布、缺证据、
binding 非 null、未知字段、ParamValue 双值/裸结构化 literal。
"""

from __future__ import annotations

import pytest

from edr_rag.knowledge import (
    AtomicIR,
    CaseBundle,
    EvidenceRef,
    KnowledgeRepository,
    ModelError,
    ParamValue,
    SourceRevision,
    StepVerification,
    TaskIR,
    TaskVariable,
    TestCaseRevision,
    WorkflowRevision,
    content_hash,
    validate_step_graph,
)

TS = "2026-11-07T15:40:00+08:00"


def make_source(source_id: str = "src-manual-1", h: str = "c" * 64) -> SourceRevision:
    return SourceRevision(
        source_id=source_id, format="chm", uri="offline://fixtures/manual.chm",
        source_hash=h, source_updated_at=TS, collected_at=TS,
    )


def make_case(source: SourceRevision, case_id: str = "case-100", product_version: str = "5.3.0") -> TestCaseRevision:
    return TestCaseRevision(
        case_id=case_id, product_version=product_version, platform="windows",
        title="配置服务器地址", steps=[
            {"action": "打开网络设置页", "expected": "显示服务器地址输入框"},
            {"action": "输入服务器地址并保存", "expected": "提示保存成功"},
        ],
        raw_text_hash=content_hash({"raw": "fixture"}), source_ref=source.ref, created_at=TS,
    )


def make_evidence(source: SourceRevision) -> list[EvidenceRef]:
    return [
        EvidenceRef("E-operation-1", source.ref, "steps[0]", "打开网络设置页", "operation"),
        EvidenceRef("E-expected-1", source.ref, "steps[0]", "显示服务器地址输入框", "expected"),
        EvidenceRef("E-expected-2", source.ref, "steps[1]", "提示保存成功", "expected"),
    ]


def make_task(source: SourceRevision, task_id: str = "task-100") -> TaskIR:
    return TaskIR(
        task_id=task_id, product_version="5.3.0", platform="windows",
        goal="完成服务器地址配置并确认保存成功",
        source_case_id="case-100", source_case_ref=source.ref,
        source_evidence_ids=["E-operation-1", "E-expected-1", "E-expected-2"],
        preconditions=["已登录管理端"],
        variables=[TaskVariable("server_address", "string")],
        business_result="配置已保存", created_at=TS,
    )


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


def make_workflow(source: SourceRevision, **overrides) -> WorkflowRevision:
    fields = dict(
        workflow_id="wf-100", product_version="5.3.0", platform="windows",
        task_id="task-100", task_source_ref=source.ref,
        steps=[step_nav(), step_input()],
        declared_variables=[TaskVariable("server_address", "string")],
        status="draft", created_at=TS,
    )
    fields.update(overrides)
    return WorkflowRevision(**fields)


def seeded_repo(tmp_path) -> tuple[KnowledgeRepository, SourceRevision]:
    repo = KnowledgeRepository(tmp_path / "kb")
    s = make_source()
    repo.import_case(CaseBundle(case=make_case(s), evidence=make_evidence(s)))
    repo.save_task(make_task(s))
    return repo, s


class TestParamValue:
    def test_variable_roundtrip(self):
        p = ParamValue(variable="server_address")
        assert ParamValue.from_dict(p.to_dict()).variable == "server_address"

    def test_literal_roundtrip(self):
        p = ParamValue(_sentinel=True, literal=3)
        assert ParamValue.from_dict(p.to_dict()).literal == 3

    def test_both_empty_rejected(self):
        with pytest.raises(ModelError, match="exactly one"):
            ParamValue()

    def test_both_set_rejected(self):
        with pytest.raises(ModelError, match="exactly one"):
            ParamValue(variable="x", _sentinel=True, literal=1)

    def test_structured_literal_rejected(self):
        # 裸结构化载荷不得当参数 (§7 不能把任意字符串/对象替换成命令)
        with pytest.raises(ModelError, match="JSON scalar"):
            ParamValue(_sentinel=True, literal={"cmd": "rm -rf /"})

    def test_from_dict_unknown_field(self):
        with pytest.raises(ModelError, match="unknown fields"):
            ParamValue.from_dict({"variable": "x", "extra": 1})

    def test_from_dict_non_object(self):
        with pytest.raises(ModelError, match="object"):
            ParamValue.from_dict("server_address")


class TestAtomicIR:
    def test_valid_step_construction(self):
        s = step_input()
        assert s.step_id == "S2" and s.depends_on == ["S1"]
        assert s.parameters["value"].variable == "server_address"

    def test_unknown_action_rejected(self):
        with pytest.raises(ModelError, match="ATOMIC_ACTIONS"):
            AtomicIR(step_id="S1", source_step=1, action="regedit",
                     target={"semantic_name": "x", "locator_status": "runtime_resolution"},
                     evidence_refs=["E-1"])

    def test_missing_evidence_rejected(self):
        with pytest.raises(ModelError, match="evidence_refs"):
            AtomicIR(step_id="S1", source_step=1, action="click",
                     target={"semantic_name": "保存", "locator_status": "runtime_resolution"},
                     evidence_refs=[])

    def test_binding_must_be_null_at_p1(self):
        with pytest.raises(ModelError, match="binding"):
            AtomicIR(step_id="S1", source_step=1, action="click",
                     target={"semantic_name": "保存", "locator_status": "fixed"},
                     evidence_refs=["E-1"], binding={"tool": "edr_wd.click"})

    def test_bad_locator_status_rejected(self):
        with pytest.raises(ModelError, match="locator_status"):
            AtomicIR(step_id="S1", source_step=1, action="click",
                     target={"semantic_name": "保存", "locator_status": "css:#ok"},
                     evidence_refs=["E-1"])

    def test_roundtrip_with_verification(self):
        s = step_input()
        d = s.to_dict()
        assert AtomicIR.from_dict(d).to_dict() == d

    def test_from_dict_unknown_field(self):
        d = step_nav().to_dict()
        d["injected"] = 1
        with pytest.raises(ModelError, match="unknown fields"):
            AtomicIR.from_dict(d)

    def test_from_dict_missing_field(self):
        d = step_nav().to_dict()
        del d["target"]
        with pytest.raises(ModelError, match="missing fields"):
            AtomicIR.from_dict(d)


class TestStepGraph:
    def test_valid_dag_passes(self):
        assert validate_step_graph([step_nav(), step_input()]) == []

    def test_dangling_dependency(self):
        s2 = step_input()
        diags = validate_step_graph([s2])
        assert any(d["code"] == "dangling_dependency" and "S1" in d["message"] for d in diags)

    def test_self_dependency(self):
        s = step_nav()
        s.depends_on = ["S1"]
        diags = validate_step_graph([s])
        assert any(d["code"] == "self_dependency" for d in diags)

    def test_duplicate_step_id(self):
        diags = validate_step_graph([step_nav(), step_nav()])
        assert any(d["code"] == "duplicate_step_id" for d in diags)

    def test_dependency_cycle_detected(self):
        a = AtomicIR(step_id="A", source_step=1, action="click",
                     target={"semantic_name": "a", "locator_status": "runtime_resolution"},
                     depends_on=["B"], evidence_refs=["E-1"])
        b = AtomicIR(step_id="B", source_step=2, action="click",
                     target={"semantic_name": "b", "locator_status": "runtime_resolution"},
                     depends_on=["A"], evidence_refs=["E-1"])
        c = AtomicIR(step_id="C", source_step=3, action="wait",
                     target={"semantic_name": "c", "locator_status": "runtime_resolution"},
                     depends_on=["A"], evidence_refs=["E-1"])
        diags = validate_step_graph([a, b, c])
        assert any(d["code"] == "dependency_cycle" for d in diags)

    def test_diamond_dag_passes(self):
        top = AtomicIR(step_id="T", source_step=1, action="navigate",
                       target={"semantic_name": "t", "locator_status": "runtime_resolution"},
                       evidence_refs=["E-1"])
        left = AtomicIR(step_id="L", source_step=2, action="click",
                        target={"semantic_name": "l", "locator_status": "runtime_resolution"},
                        depends_on=["T"], evidence_refs=["E-1"])
        right = AtomicIR(step_id="R", source_step=3, action="click",
                         target={"semantic_name": "r", "locator_status": "runtime_resolution"},
                         depends_on=["T"], evidence_refs=["E-1"])
        bottom = AtomicIR(step_id="B", source_step=4, action="assert",
                          target={"semantic_name": "b", "locator_status": "runtime_resolution"},
                          depends_on=["L", "R"], evidence_refs=["E-1"])
        assert validate_step_graph([top, left, right, bottom]) == []


class TestWorkflowRevision:
    def test_valid_workflow_persists(self, tmp_path):
        repo, s = seeded_repo(tmp_path)
        wf = make_workflow(s)
        assert repo.save_workflow(wf) is not None
        loaded = repo.get_workflow("wf-100")
        assert loaded is not None and len(loaded.steps) == 2
        assert loaded.steps[1].verification.expected_variable == "server_address"

    def test_idempotent_on_same_task_ref(self, tmp_path):
        repo, s = seeded_repo(tmp_path)
        assert repo.save_workflow(make_workflow(s)) is not None
        assert repo.save_workflow(make_workflow(s)) is None
        assert len(repo.list_workflows()) == 1

    def test_missing_task_source_rejected(self, tmp_path):
        repo = KnowledgeRepository(tmp_path / "kb")
        s = make_source()
        with pytest.raises(ModelError, match="not imported"):
            repo.save_workflow(make_workflow(s))

    def test_stale_task_ref_rejected(self, tmp_path):
        # task 已导入但 source_ref 不一致 → 悬空来源拒绝
        repo, s = seeded_repo(tmp_path)
        wf = make_workflow(s, task_source_ref="src-manual-1@rev9:deadbeef")
        with pytest.raises(ModelError, match="not imported"):
            repo.save_workflow(wf)

    def test_cycle_inside_workflow_rejected_at_save(self, tmp_path):
        repo, s = seeded_repo(tmp_path)
        a = step_nav()
        a.step_id, a.depends_on = "S9", ["S9"]
        wf = make_workflow(s, steps=[a, step_input()])
        with pytest.raises(ModelError, match="self_dependency"):
            repo.save_workflow(wf)
        assert repo.get_workflow("wf-100") is None

    def test_undeclared_variable_rejected(self, tmp_path):
        # 未声明变量不得进入计划 (变量类型化绑定)
        repo, s = seeded_repo(tmp_path)
        s2 = step_input()
        s2.parameters = {"value": ParamValue(variable="admin_password")}
        wf = make_workflow(s, steps=[step_nav(), s2], declared_variables=[TaskVariable("server_address", "string")])
        with pytest.raises(ModelError, match="admin_password"):
            repo.save_workflow(wf)

    def test_verification_unknown_variable_rejected(self, tmp_path):
        repo, s = seeded_repo(tmp_path)
        s2 = step_input()
        s2.verification = StepVerification("value_check", "ghost_var", "E-expected-2")
        wf = make_workflow(s, steps=[step_nav(), s2])
        with pytest.raises(ModelError, match="ghost_var"):
            repo.save_workflow(wf)

    def test_step_validation_diagnostics_include_field_path(self, tmp_path):
        repo, s = seeded_repo(tmp_path)
        orphan = AtomicIR(step_id="S3", source_step=3, action="click",
                          target={"semantic_name": "x", "locator_status": "runtime_resolution"},
                          depends_on=["S99"], evidence_refs=["E-operation-1"])
        wf = make_workflow(s, steps=[step_nav(), step_input(), orphan])
        diags = wf.validate_steps()
        assert any(d["code"] == "dangling_dependency" and d["path"] == "steps[S3].depends_on" for d in diags)
        # 构造允许，落库拦截且带完整诊断 —— 禁止静默删步骤
        with pytest.raises(ModelError, match="S99"):
            repo.save_workflow(wf)

    def test_published_self_grant_rejected(self):
        # 编译器不自授 published (§8)
        s = make_source()
        with pytest.raises(ModelError, match="published"):
            make_workflow(s, status="published")

    def test_execution_status_independent(self):
        s = make_source()
        wf = make_workflow(s, execution_status="failed")
        assert wf.status == "draft" and wf.execution_status == "failed"

    def test_bad_execution_status_rejected(self):
        s = make_source()
        with pytest.raises(ModelError, match="execution_status"):
            make_workflow(s, execution_status="ok")

    def test_content_hash_changes_with_steps(self):
        s = make_source()
        w1 = make_workflow(s)
        w2 = make_workflow(s)
        assert w1.content_hash == w2.content_hash
        w2.steps[0].action = "screenshot"
        assert w1.content_hash != w2.content_hash

    def test_roundtrip_preserves_dag(self):
        s = make_source()
        wf = make_workflow(s)
        d = wf.to_dict()
        got = WorkflowRevision.from_dict(d)
        assert got.to_dict() == d

    def test_from_dict_unknown_field(self):
        s = make_source()
        d = make_workflow(s).to_dict()
        d["reviewer"] = "self"
        with pytest.raises(ModelError, match="unknown fields"):
            WorkflowRevision.from_dict(d)

    def test_capability_snapshot_ref_carried(self, tmp_path):
        repo, s = seeded_repo(tmp_path)
        wf = make_workflow(s, capability_snapshot_ref="cap-snap:sha256:abc123")
        repo.save_workflow(wf)
        assert repo.get_workflow("wf-100").capability_snapshot_ref == "cap-snap:sha256:abc123"

    def test_version_filter_no_cross_merge(self, tmp_path):
        repo, s = seeded_repo(tmp_path)
        repo.save_workflow(make_workflow(s))
        repo.import_case(CaseBundle(case=make_case(s, case_id="case-201", product_version="5.4.0"),
                                    evidence=make_evidence(s)))
        repo.save_task(make_task(s, task_id="task-201"))
        wf2 = make_workflow(s, workflow_id="wf-201", task_id="task-201", product_version="5.4.0")
        repo.save_workflow(wf2)
        assert [w.workflow_id for w in repo.list_workflows(product_version="5.3.0")] == ["wf-100"]
        assert repo.get_workflow("wf-201", product_version="5.3.0") is None
