"""P1 second batch: DocumentSectionRevision adapter + minimal TaskIR.

design.md §10 P1: DocumentRevision 适配引用既有 ManualSection（来源 hash
绑定）；TaskIR 最小模型。负例必须覆盖：缺来源、缺证据、悬空证据引用、
歧义+validated 放行、跨版本混检、未知字段。
"""

from __future__ import annotations

import pytest

from edr_rag.knowledge import (
    CaseBundle,
    DocumentSectionRevision,
    EvidenceRef,
    KnowledgeRepository,
    ModelError,
    SourceRevision,
    TaskIR,
    TaskVariable,
    TestCaseRevision,
    content_hash,
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


def make_task(source: SourceRevision, **overrides) -> TaskIR:
    fields = dict(
        task_id="task-100", product_version="5.3.0", platform="windows",
        goal="完成服务器地址配置并确认保存成功",
        source_case_id="case-100", source_case_ref=source.ref,
        source_evidence_ids=["E-operation-1", "E-expected-1", "E-expected-2"],
        preconditions=["已登录管理端"],
        variables=[TaskVariable("server_address", "string")],
        business_result="配置已保存", created_at=TS,
    )
    fields.update(overrides)
    return TaskIR(**fields)


class TestDocumentSectionAdapter:
    def test_adapter_roundtrip(self, tmp_path):
        repo = KnowledgeRepository(tmp_path / "kb")
        s = make_source()
        repo.save_source(s)
        sec = DocumentSectionRevision(
            section_id="manual.section.network.proxy", document_version="5.3.0",
            source_ref=s.ref, section_content_hash=content_hash({"html": "<p>proxy</p>"}),
            toc_path=["网络", "代理设置"], page_or_chm_path="network/proxy.htm", created_at=TS,
        )
        repo.save_section(sec)
        got = repo.latest_section("manual.section.network.proxy")
        assert got is not None and got.toc_path == ["网络", "代理设置"]
        assert got.section_content_hash == sec.section_content_hash

    def test_new_source_revision_rebinds_new_hash(self, tmp_path):
        repo = KnowledgeRepository(tmp_path / "kb")
        s1 = make_source()
        s2 = make_source(h="d" * 64)
        repo.save_source(s1)
        repo.save_source(s2)
        h1 = content_hash({"html": "v1"})
        h2 = content_hash({"html": "v2"})
        repo.save_section(DocumentSectionRevision(
            section_id="sec-a", document_version="5.3.0", source_ref=s1.ref,
            section_content_hash=h1, created_at=TS))
        repo.save_section(DocumentSectionRevision(
            section_id="sec-a", document_version="5.3.0", source_ref=s2.ref,
            section_content_hash=h2, revision=2, created_at=TS))
        latest = repo.latest_section("sec-a", document_version="5.3.0")
        assert latest.source_ref == s2.ref and latest.section_content_hash == h2
        # 旧 hash 不再出现在最新 revision：源变更必须重绑定 (§4)
        assert latest.section_content_hash != h1

    def test_rejects_empty_content_hash(self):
        with pytest.raises(ModelError, match="section_content_hash"):
            DocumentSectionRevision(
                section_id="sec-b", document_version="5.3.0",
                source_ref="src@rev1:x", section_content_hash="", created_at=TS)

    def test_rejects_unknown_field(self):
        d = DocumentSectionRevision(
            section_id="sec-c", document_version="5.3.0", source_ref="src@rev1:x",
            section_content_hash="h", created_at=TS).to_dict()
        d["injected"] = True
        with pytest.raises(ModelError, match="unknown fields"):
            DocumentSectionRevision.from_dict(d)

    def test_rejects_naive_created_at(self):
        with pytest.raises(ModelError, match="timezone"):
            DocumentSectionRevision(
                section_id="sec-d", document_version="5.3.0", source_ref="src@rev1:x",
                section_content_hash="h", created_at="2026-11-07T15:40:00")

    def test_version_filter_no_cross_merge(self, tmp_path):
        repo = KnowledgeRepository(tmp_path / "kb")
        s = make_source()
        repo.save_source(s)
        mk = lambda v: DocumentSectionRevision(
            section_id="sec-v", document_version=v, source_ref=s.ref,
            section_content_hash="h-" + v, created_at=TS)
        repo.save_section(mk("5.3.0"))
        repo.save_section(DocumentSectionRevision(
            section_id="sec-v", document_version="5.4.0", source_ref=s.ref,
            section_content_hash="h-5.4.0", revision=2, created_at=TS))
        assert repo.latest_section("sec-v", document_version="5.3.0").document_version == "5.3.0"
        assert repo.latest_section("sec-v", document_version="5.4.0").revision == 2


class TestTaskIR:
    def test_task_persists_with_case_binding(self, tmp_path):
        repo = KnowledgeRepository(tmp_path / "kb")
        s = make_source()
        repo.import_case(CaseBundle(case=make_case(s), evidence=make_evidence(s)))
        saved = repo.save_task(make_task(s))
        assert saved is not None and saved.status == "needs_review"
        loaded = repo.get_task("task-100")
        assert loaded.goal.startswith("完成服务器地址配置")
        assert loaded.variables[0].var_type == "string"

    def test_duplicate_case_ref_returns_none(self, tmp_path):
        repo = KnowledgeRepository(tmp_path / "kb")
        s = make_source()
        repo.import_case(CaseBundle(case=make_case(s), evidence=make_evidence(s)))
        assert repo.save_task(make_task(s)) is not None
        assert repo.save_task(make_task(s)) is None
        assert len(repo.list_tasks()) == 1

    def test_missing_case_source_rejected(self, tmp_path):
        repo = KnowledgeRepository(tmp_path / "kb")
        s = make_source()
        # case 未导入 → 缺来源拒绝 (P1 验收：缺来源明确拒绝)
        with pytest.raises(ModelError, match="not imported"):
            repo.save_task(make_task(s))

    def test_dangling_evidence_ref_rejected(self, tmp_path):
        repo = KnowledgeRepository(tmp_path / "kb")
        s = make_source()
        repo.import_case(CaseBundle(case=make_case(s), evidence=make_evidence(s)))
        bad = make_task(s, source_evidence_ids=["E-operation-1", "E-ghost"])
        with pytest.raises(ModelError, match="dangling evidence"):
            repo.save_task(bad)

    def test_task_with_ambiguity_cannot_be_validated(self):
        s = make_source()
        with pytest.raises(ModelError, match="needs_review"):
            make_task(s, ambiguities=["服务器地址的默认值未在用例中给出"], status="validated")

    def test_ambiguous_task_kept_needs_review(self, tmp_path):
        repo = KnowledgeRepository(tmp_path / "kb")
        s = make_source()
        repo.import_case(CaseBundle(case=make_case(s), evidence=make_evidence(s)))
        t = make_task(s, ambiguities=["服务器地址的默认值未在用例中给出"])
        assert t.status == "needs_review"
        assert repo.save_task(t) is not None

    def test_empty_goal_rejected(self):
        s = make_source()
        with pytest.raises(ModelError, match="goal required"):
            make_task(s, goal="")

    def test_unknown_var_type_rejected(self):
        with pytest.raises(ModelError, match="var_type"):
            TaskVariable("x", "json_object")

    def test_version_filter_no_cross_merge(self, tmp_path):
        repo = KnowledgeRepository(tmp_path / "kb")
        s = make_source()
        repo.import_case(CaseBundle(case=make_case(s), evidence=make_evidence(s)))
        repo.save_task(make_task(s))
        repo.import_case(CaseBundle(case=make_case(s, case_id="case-201", product_version="5.4.0"),
                                    evidence=make_evidence(s)))
        t2 = make_task(s, task_id="task-201", source_case_id="case-201", product_version="5.4.0")
        repo.save_task(t2)
        assert [t.task_id for t in repo.list_tasks(product_version="5.3.0")] == ["task-100"]
        assert repo.get_task("task-201", product_version="5.3.0") is None
