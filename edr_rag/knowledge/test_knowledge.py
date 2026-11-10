"""P1 minimal acceptance tests: TestCase/Evidence model + repository.

design.md §10 P1 row: legal inputs accepted; wrong types, missing source,
cross-version, dangling references explicitly rejected. Idempotent import,
stale marking on supersede (§4).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from edr_rag.atomic_io import atomic_write_text
from edr_rag.knowledge import (
    CaseBundle,
    EvidenceRef,
    ImportResult,
    KnowledgeRepository,
    ModelError,
    SourceRevision,
    TestCaseRevision,
    content_hash,
)

TS = "2026-11-07T14:00:00+08:00"


def make_source(source_id: str = "src-case-import-1", revision: int = 1, h: str = "a" * 64) -> SourceRevision:
    return SourceRevision(
        source_id=source_id,
        format="platform-json",
        uri="offline://fixtures/case1.json",
        source_hash=h,
        source_updated_at=TS,
        collected_at=TS,
        revision=revision,
    )


def make_case(source: SourceRevision, **overrides) -> TestCaseRevision:
    fields = dict(
        case_id="case-001",
        product_version="5.3.0",
        platform="windows",
        title="配置服务器地址",
        preconditions=["已登录管理端"],
        steps=[
            {"action": "打开网络设置页", "expected": "显示服务器地址输入框"},
            {"action": "输入服务器地址并保存", "expected": "提示保存成功"},
        ],
        raw_text_hash=content_hash({"raw": "fixture"}),
        source_ref=source.ref,
        created_at=TS,
    )
    fields.update(overrides)
    return TestCaseRevision(**fields)


def make_evidence(source: SourceRevision) -> list[EvidenceRef]:
    return [
        EvidenceRef(evidence_id="E-operation-1", source_ref=source.ref, locator="steps[0]", quote="打开网络设置页", purpose="operation"),
        EvidenceRef(evidence_id="E-expected-1", source_ref=source.ref, locator="steps[0]", quote="显示服务器地址输入框", purpose="expected"),
    ]


class TestPositive:
    def test_bundle_roundtrip(self, tmp_path: Path):
        source = make_source()
        bundle = CaseBundle(case=make_case(source), evidence=make_evidence(source))
        repo = KnowledgeRepository(tmp_path / "kb")
        result = repo.import_case(bundle)
        assert isinstance(result, ImportResult)
        assert result.status == "created" and result.revision == 1

        loaded = repo.get_case("case-001")
        assert loaded is not None
        assert loaded.steps[0]["expected"] == "显示服务器地址输入框"
        ev = repo.evidence_of(loaded)
        assert [e["evidence_id"] for e in ev] == ["E-operation-1", "E-expected-1"]

    def test_idempotent_duplicate_import(self, tmp_path: Path):
        source = make_source()
        bundle = CaseBundle(case=make_case(source), evidence=make_evidence(source))
        repo = KnowledgeRepository(tmp_path / "kb")
        first = repo.import_case(bundle)
        second = repo.import_case(bundle)
        assert second.status == "duplicate"
        assert second.revision == first.revision
        assert len(repo.list_cases()) == 1

    def test_new_source_revision_supersedes_and_marks_stale(self, tmp_path: Path):
        repo = KnowledgeRepository(tmp_path / "kb")
        s1 = make_source()
        repo.import_case(CaseBundle(case=make_case(s1), evidence=make_evidence(s1)))
        s2 = make_source(h="b" * 64, revision=2)
        result = repo.import_case(CaseBundle(case=make_case(s2), evidence=make_evidence(s2)))
        assert result.status == "superseded" and result.revision == 2
        stale = (tmp_path / "kb" / "stale.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert json.loads(stale[-1])["stale"] == "case-001@rev1"

    def test_version_filter_does_not_merge(self, tmp_path: Path):
        repo = KnowledgeRepository(tmp_path / "kb")
        s = make_source()
        repo.import_case(CaseBundle(case=make_case(s), evidence=make_evidence(s)))
        repo.import_case(CaseBundle(case=make_case(s, case_id="case-002", product_version="5.4.0"), evidence=make_evidence(s)))
        assert len(repo.list_cases()) == 2
        assert [c.case_id for c in repo.list_cases(product_version="5.3.0")] == ["case-001"]


class TestNegative:
    def test_rejects_cross_source_evidence(self, tmp_path: Path):
        s1 = make_source()
        s2 = make_source(source_id="other-src")
        bad = CaseBundle(
            case=make_case(s1),
            evidence=[EvidenceRef("E-x", s2.ref, "steps[0]", "q", "operation"),
                      EvidenceRef("E-y", s1.ref, "steps[0]", "q", "expected")],
        )
        with pytest.raises(ModelError, match="source_ref"):
            KnowledgeRepository(tmp_path / "kb").import_case(bad)

    def test_rejects_dangling_evidence_purpose(self, tmp_path: Path):
        s = make_source()
        bad = CaseBundle(case=make_case(s), evidence=[
            EvidenceRef("E-op", s.ref, "steps[0]", "打开网络设置页", "parameter"),
        ])
        with pytest.raises(ModelError, match="purpose"):
            bad.validate_references()

    def test_rejects_case_without_any_evidence(self, tmp_path: Path):
        s = make_source()
        with pytest.raises(ModelError, match="at least one evidence"):
            KnowledgeRepository(tmp_path / "kb").import_case(CaseBundle(case=make_case(s), evidence=[]))

    def test_rejects_naive_timestamp(self):
        with pytest.raises(ModelError, match="timezone"):
            SourceRevision(source_id="s", format="x", uri="u", source_hash="h",
                           source_updated_at="2026-11-07T14:00:00", collected_at=TS)

    def test_rejects_missing_expected_in_step(self):
        s = make_source()
        with pytest.raises(ModelError, match="missing 'expected'"):
            TestCaseRevision(case_id="case-bad", product_version="5.3.0", platform="windows",
                             title="t", steps=[{"action": "只写操作"}], raw_text_hash="x",
                             source_ref=s.ref, created_at=TS)

    def test_rejects_unknown_top_level_field(self):
        d = make_case(make_source()).to_dict()
        d["hacker_field"] = "nope"
        with pytest.raises(ModelError, match="unknown fields"):
            TestCaseRevision.from_dict(d)

    def test_rejects_empty_product_version(self):
        s = make_source()
        with pytest.raises(ModelError, match="product_version"):
            make_case(s, product_version="")

    def test_atomic_write_leaves_valid_jsonl_on_corrupt_neighbor(self, tmp_path: Path):
        kb = tmp_path / "kb"
        kb.mkdir()
        atomic_write_text(kb / "cases.jsonl", "{not json}\n")
        repo = KnowledgeRepository(kb)
        with pytest.raises(json.JSONDecodeError):
            repo.list_cases()
        # corrupt line must not be silently skipped
