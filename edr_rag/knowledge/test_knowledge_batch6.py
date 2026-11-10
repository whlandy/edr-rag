"""P2 batch 3 tests: stale-marking chain + tombstone-driven invalidation.

正例：源变更 → section stale；case supersede/tombstone → task/workflow stale。
负例必须覆盖：坏 kind/reason、缺 ref、naive recorded_at、未知字段、缺
document_version、跨版本不误标、同 hash 重采不误标、重复标记幂等。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from edr_rag.knowledge import (
    CaseBundle,
    EvidenceRef,
    KnowledgeRepository,
    ModelError,
    SourceRevision,
    TaskIR,
    TaskVariable,
    TestCaseRevision,
    WorkflowRevision,
    AtomicIR,
    ParamValue,
    content_hash,
)
from edr_rag.knowledge.staleness import (
    STALE_MARKS_FILE,
    StaleMark,
    StaleMarkRegistry,
    StalenessError,
    append_case_supersede_downstream,
)

TS = "2026-11-08T10:00:00+08:00"


def make_source(source_id="src-manual-1", h="c" * 64, revision=1) -> SourceRevision:
    return SourceRevision(
        source_id=source_id, format="chm", uri="offline://fixtures/manual.chm",
        source_hash=h, source_updated_at=TS, collected_at=TS, revision=revision,
    )


def make_case(source, case_id="case-100") -> TestCaseRevision:
    return TestCaseRevision(
        case_id=case_id, product_version="5.3.0", platform="windows",
        title="配置服务器地址", steps=[
            {"action": "打开网络设置页", "expected": "显示服务器地址输入框"},
            {"action": "输入服务器地址并保存", "expected": "提示保存成功"},
        ],
        raw_text_hash=content_hash({"raw": "fixture"}), source_ref=source.ref, created_at=TS,
    )


def make_evidence(source) -> list[EvidenceRef]:
    return [
        EvidenceRef("E-operation-1", source.ref, "steps[0]", "打开网络设置页", "operation"),
        EvidenceRef("E-expected-1", source.ref, "steps[0]", "显示服务器地址输入框", "expected"),
    ]


def make_task(source, task_id="task-100") -> TaskIR:
    return TaskIR(
        task_id=task_id, product_version="5.3.0", platform="windows",
        goal="完成服务器地址配置并确认保存成功",
        source_case_id="case-100", source_case_ref=source.ref,
        source_evidence_ids=["E-operation-1", "E-expected-1"],
        variables=[TaskVariable("server_address", "string")],
        business_result="配置已保存", created_at=TS,
    )


def make_workflow(source, task_id="task-100") -> WorkflowRevision:
    return WorkflowRevision(
        workflow_id="wf-100", product_version="5.3.0", platform="windows",
        task_id=task_id, task_source_ref=source.ref,
        steps=[AtomicIR(
            step_id="S1", source_step=1, action="navigate",
            target={"semantic_name": "网络设置页", "locator_status": "runtime_resolution"},
            evidence_refs=["E-operation-1"],
        )],
        declared_variables=[TaskVariable("server_address", "string")],
        created_at=TS,
    )


def seeded_repo(tmp_path: Path, source_id="src-manual-1", case_id="case-100", task_id="task-100"):
    repo = KnowledgeRepository(tmp_path / "kb")
    s = make_source(source_id)
    repo.import_case(CaseBundle(case=make_case(s, case_id), evidence=make_evidence(s)))
    repo.save_task(make_task(s, task_id))
    repo.save_workflow(make_workflow(s, task_id))
    return repo, s


class TestStaleMarkModel:
    def test_valid_mark_roundtrip(self):
        m = StaleMark(kind="case", ref="case-100@rev1", reason="superseded_by_new_case", recorded_at=TS)
        assert StaleMark.from_dict(m.to_dict()).key == m.key

    def test_bad_kind_rejected(self):
        with pytest.raises(StalenessError, match="kind"):
            StaleMark(kind="ghost", ref="x", reason="tombstone_deleted")

    def test_bad_reason_rejected(self):
        with pytest.raises(StalenessError, match="reason"):
            StaleMark(kind="case", ref="x", reason="magic")

    def test_empty_ref_rejected(self):
        with pytest.raises(StalenessError, match="ref"):
            StaleMark(kind="case", ref="", reason="tombstone_deleted")

    def test_naive_recorded_at_rejected(self):
        with pytest.raises(StalenessError, match="timezone"):
            StaleMark(kind="case", ref="x", reason="tombstone_deleted", recorded_at="2026-11-08T10:00:00")

    def test_unknown_field_rejected(self):
        with pytest.raises(StalenessError, match="unknown fields"):
            StaleMark.from_dict({"kind": "case", "ref": "x", "reason": "tombstone_deleted", "injected": 1})

    def test_missing_field_rejected(self):
        with pytest.raises(StalenessError, match="missing fields"):
            StaleMark.from_dict({"kind": "case", "ref": "x"})


class TestSectionStalePropagation:
    def test_new_source_revision_marks_sections_stale(self, tmp_path):
        repo = KnowledgeRepository(tmp_path / "kb")
        from edr_rag.knowledge import DocumentSectionRevision
        s1, s2 = make_source(h="c" * 64, revision=1), make_source(h="d" * 64, revision=2)
        repo.save_source(s1)
        repo.save_source(s2)
        repo.save_section(DocumentSectionRevision(
            section_id="sec-a", document_version="5.3.0", source_ref=s1.ref,
            section_content_hash=content_hash({"v": 1}), created_at=TS))
        reg = StaleMarkRegistry(repo)
        marks = reg.propagate_document_source_change(s2, "5.3.0")
        assert [m.kind for m in marks] == ["section"]
        assert marks[0].ref == "sec-a@rev1"
        assert marks[0].reason == "superseded_by_new_source"
        assert marks[0].source_ref == s1.ref

    def test_same_hash_reingestion_marks_nothing(self, tmp_path):
        repo = KnowledgeRepository(tmp_path / "kb")
        from edr_rag.knowledge import DocumentSectionRevision
        s1 = make_source(h="c" * 64, revision=1)
        s1_dup = make_source(h="c" * 64, revision=2)
        repo.save_source(s1)
        repo.save_source(s1_dup)
        repo.save_section(DocumentSectionRevision(
            section_id="sec-a", document_version="5.3.0", source_ref=s1.ref,
            section_content_hash=content_hash({"v": 1}), created_at=TS))
        marks = StaleMarkRegistry(repo).propagate_document_source_change(s1_dup, "5.3.0")
        assert marks == []

    def test_other_document_version_not_marked(self, tmp_path):
        repo = KnowledgeRepository(tmp_path / "kb")
        from edr_rag.knowledge import DocumentSectionRevision
        s1, s2 = make_source(h="c" * 64, revision=1), make_source(h="d" * 64, revision=2)
        repo.save_source(s1)
        repo.save_source(s2)
        repo.save_section(DocumentSectionRevision(
            section_id="sec-a", document_version="5.3.0", source_ref=s1.ref,
            section_content_hash=content_hash({"v": 1}), created_at=TS))
        marks = StaleMarkRegistry(repo).propagate_document_source_change(s2, "5.4.0")
        assert marks == []

    def test_missing_document_version_rejected(self, tmp_path):
        repo = KnowledgeRepository(tmp_path / "kb")
        s1, s2 = make_source(h="c" * 64, revision=1), make_source(h="d" * 64, revision=2)
        repo.save_source(s1)
        repo.save_source(s2)
        with pytest.raises(StalenessError, match="document_version"):
            StaleMarkRegistry(repo).propagate_document_source_change(s2, "")


class TestTombstoneInvalidation:
    def test_tombstone_marks_case_task_workflow(self, tmp_path):
        repo, s = seeded_repo(tmp_path)
        marks = StaleMarkRegistry(repo).apply_tombstone("case-100", s.ref)
        by_kind = {m.kind: m for m in marks}
        assert set(by_kind) == {"case", "task", "workflow"}
        assert by_kind["case"].ref == "case-100@rev1"
        assert by_kind["case"].reason == "tombstone_deleted"
        assert by_kind["task"].ref == "task-100@rev1"
        assert by_kind["task"].reason == "downstream_of_stale_case"
        assert by_kind["workflow"].ref == "wf-100@rev1"
        assert by_kind["workflow"].reason == "downstream_of_stale_case"

    def test_tombstone_idempotent(self, tmp_path):
        repo, s = seeded_repo(tmp_path)
        reg = StaleMarkRegistry(repo)
        reg.apply_tombstone("case-100", s.ref)
        assert reg.apply_tombstone("case-100", s.ref) == []

    def test_other_case_source_pair_not_marked(self, tmp_path):
        repo, _ = seeded_repo(tmp_path)
        marks = StaleMarkRegistry(repo).apply_tombstone("case-100", "other-src@rev1:x")
        assert marks == []


class TestCaseSupersedeDownstream:
    def test_supersede_marks_task_workflow_stale(self, tmp_path):
        repo, s = seeded_repo(tmp_path)
        s2 = make_source(h="d" * 64)
        repo.import_case(CaseBundle(case=make_case(s2), evidence=make_evidence(s2)))
        # import_case already appended the legacy case-supersede record
        assert [json.loads(line) for line in
                (Path(repo.root) / STALE_MARKS_FILE).read_text(encoding="utf-8").strip().splitlines()
                ] == [{"stale": "case-100@rev1", "superseded_by": "case-100@rev2"}]
        n = append_case_supersede_downstream(
            repo, "case-100", [s.ref], Path(repo.root) / STALE_MARKS_FILE)
        assert n == 2  # task + workflow
        path = Path(repo.root) / STALE_MARKS_FILE
        recs = [json.loads(line) for line in path.read_text(encoding="utf-8").strip().splitlines()]
        # append() rewrites the legacy case-supersede record in normalized form
        # (same fact, richer reason) and adds the downstream task/workflow marks
        assert recs[0]["kind"] == "case" and recs[0]["ref"] == "case-100@rev1"
        assert recs[0]["detail"] == "superseded_by case-100@rev2"
        assert {r["kind"] for r in recs[1:]} == {"task", "workflow"}
        assert all(r["reason"] == "downstream_of_stale_case" for r in recs[1:])

    def test_supersede_hook_idempotent(self, tmp_path):
        repo, s = seeded_repo(tmp_path)
        stale_path = Path(repo.root) / STALE_MARKS_FILE
        append_case_supersede_downstream(repo, "case-100", [s.ref], stale_path)
        assert append_case_supersede_downstream(repo, "case-100", [s.ref], stale_path) == 0


class TestRegistry:
    def test_legacy_case_supersede_record_normalizes(self, tmp_path):
        repo, _ = seeded_repo(tmp_path)
        path = Path(repo.root) / STALE_MARKS_FILE
        path.write_text(json.dumps({"stale": "case-001@rev1", "superseded_by": "case-001@rev2"}) + "\n", encoding="utf-8")
        reg = StaleMarkRegistry(repo)
        marks = reg.load()
        assert marks[0].kind == "case"
        assert marks[0].reason == "superseded_by_new_case"
        assert reg.is_stale("case", "case-001@rev1")

    def test_legacy_record_rewritten_with_reason_kept_deduped(self, tmp_path):
        repo, _ = seeded_repo(tmp_path)
        path = Path(repo.root) / STALE_MARKS_FILE
        path.write_text(json.dumps({"stale": "case-001@rev1", "superseded_by": "case-001@rev2"}) + "\n", encoding="utf-8")
        reg = StaleMarkRegistry(repo)
        assert reg.append([StaleMark(kind="case", ref="case-001@rev1", reason="tombstone_deleted")]) == []

    def test_unrecognized_record_rejected(self, tmp_path):
        repo, _ = seeded_repo(tmp_path)
        path = Path(repo.root) / STALE_MARKS_FILE
        path.write_text(json.dumps({"ghost": True}) + "\n", encoding="utf-8")
        with pytest.raises(StalenessError, match="unrecognized"):
            StaleMarkRegistry(repo).load()

    def test_corrupt_json_line_rejected(self, tmp_path):
        repo = KnowledgeRepository(tmp_path / "kb")
        (tmp_path / "kb" / STALE_MARKS_FILE).write_text("{broken\n", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            StaleMarkRegistry(repo).load()

    def test_duplicate_marks_in_batch_skipped(self, tmp_path):
        repo, _ = seeded_repo(tmp_path)
        reg = StaleMarkRegistry(repo)
        mark = StaleMark(kind="case", ref="c1@rev1", reason="tombstone_deleted", recorded_at=TS)
        assert reg.append([mark, mark]) == [mark]

    def test_no_fresh_marks_leaves_file_untouched(self, tmp_path):
        repo, _ = seeded_repo(tmp_path)
        path = Path(repo.root) / STALE_MARKS_FILE
        path.write_text("", encoding="utf-8")
        StaleMarkRegistry(repo).append([])
        assert path.read_text(encoding="utf-8") == ""
