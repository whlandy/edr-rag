"""P2 batch 2 tests: platform Connector contract + sanitized fixture import.

All fixtures are synthetic sanitized data (§5: 平台未知时仅提供接口和离线导入).
Negative cases follow the P1/P2 pattern: contract violations must be rejected
with precise errors, never silently dropped or fabricated.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from edr_rag.atomic_io import atomic_write_text
from edr_rag.knowledge.connector import (
    AuthenticationRequired,
    ConnectorError,
    FixtureEntry,
    SourceDescriptor,
    WatermarkState,
    import_case_fixture,
    iter_fixture_pages,
    normalize_fixture_case,
    parse_fixture_page,
)
from edr_rag.knowledge.repository import KnowledgeRepository


def _case(case_id: str, updated_at: str, **over) -> dict:
    raw = {
        "case_id": case_id,
        "product_version": "V5R1",
        "platform": "windows",
        "title": f"case {case_id}",
        "preconditions": ["登录客户端"],
        "steps": [
            {"action": "点击[策略管理]", "expected": "进入策略管理页面"},
            {"action": "点击[新增]", "expected": "弹出新增对话框"},
        ],
        "updated_at": updated_at,
    }
    raw.update(over)
    return raw


def _page(source_id: str, cases: list[dict], next_cursor=None, **over) -> dict:
    d = {"source_id": source_id, "cases": cases, "next_cursor": next_cursor}
    d.update(over)
    return d


def _write_page(pages_dir: Path, name: str, payload: dict) -> None:
    pages_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(pages_dir / name, json.dumps(payload, ensure_ascii=False, indent=2))


def _state(source_id: str) -> WatermarkState:
    return WatermarkState(source_id=source_id, updated_at_watermark="1970-01-01T00:00:00+00:00", cursor=None)


def _repo(tmp_path: Path) -> KnowledgeRepository:
    return KnowledgeRepository(tmp_path / "kb")


@pytest.fixture()
def workdir(tmp_path: Path) -> Path:
    d = tmp_path / "fixture"
    d.mkdir()
    return d


# ---------------- descriptor / page parsing ----------------


def test_descriptor_requires_tz_aware_updated_at():
    with pytest.raises(ConnectorError, match="timezone"):
        SourceDescriptor(source_id="s", format="json", uri="file:///x", updated_at="2026-11-07T10:00:00")


def test_page_source_id_mismatch_rejected(workdir: Path):
    p = workdir / "00_page.json"
    p.write_text(json.dumps(_page("OTHER", [])), encoding="utf-8")
    with pytest.raises(ConnectorError, match="source_id"):
        parse_fixture_page(p, "proj")


def test_page_missing_required_fields_rejected(workdir: Path):
    p = workdir / "00_page.json"
    p.write_text(json.dumps(_page("proj", [{"case_id": "c1"}])), encoding="utf-8")
    with pytest.raises(ConnectorError, match="missing fields"):
        parse_fixture_page(p, "proj")


def test_page_deleted_entry_may_omit_case_fields(workdir: Path):
    p = workdir / "00_page.json"
    p.write_text(
        json.dumps(_page("proj", [{"case_id": "c9", "deleted": True}])),
        encoding="utf-8",
    )
    entries, _ = parse_fixture_page(p, "proj")
    assert entries[0].deleted is True


def test_cursor_cycle_rejected(workdir: Path):
    pages = workdir / "pages"
    _write_page(pages, "00_page.json", _page("proj", [_case("c1", "2026-11-07T01:00:00+00:00")], next_cursor="01_page.json"))
    _write_page(pages, "01_page.json", _page("proj", [], next_cursor="00_page.json"))
    with pytest.raises(ConnectorError, match="cycle"):
        list(iter_fixture_pages(workdir, "proj"))


# ---------------- normalize ----------------


def test_normalize_binds_raw_hash_and_evidence(tmp_path: Path):
    entry = FixtureEntry(raw=_case("c1", "2026-11-07T01:00:00+00:00"))
    bundle = normalize_fixture_case(entry, source_ref="proj@page1", collected_at="2026-11-07T02:00:00+00:00")
    from edr_rag.knowledge.models import content_hash

    assert bundle.case.raw_text_hash == content_hash(entry.raw)
    purposes = {e.purpose for e in bundle.evidence}
    assert purposes == {"operation", "expected"}
    assert all(e.source_ref == "proj@page1" for e in bundle.evidence)


def test_normalize_deleted_entry_rejected():
    with pytest.raises(ConnectorError, match="tombstone"):
        normalize_fixture_case(FixtureEntry(raw={"case_id": "c9"}, deleted=True), source_ref="x", collected_at="now")


def test_normalize_step_unknown_field_rejected(tmp_path: Path):
    from edr_rag.knowledge.models import ModelError

    entry = FixtureEntry(raw=_case("c1", "2026-11-07T01:00:00+00:00", steps=[{"action": "a", "expected": "e", "suspicious": 1}]))
    with pytest.raises(ModelError, match="unknown step fields"):
        normalize_fixture_case(entry, source_ref="x", collected_at="2026-11-07T02:00:00+00:00")


# ---------------- full import: pagination / watermark / idempotency ----------------


def test_import_full_chain_pagination_watermark_idempotent(workdir: Path, tmp_path: Path):
    pages = workdir / "pages"
    _write_page(pages, "00_page.json", _page("proj", [_case("c1", "2026-11-07T01:00:00+00:00")], next_cursor="01_page.json"))
    _write_page(pages, "01_page.json", _page("proj", [_case("c2", "2026-11-07T02:00:00+00:00")]))

    repo = _repo(tmp_path)
    state = _state("proj")
    stats = import_case_fixture(repo, workdir, "proj", state)

    assert stats["pages"] == 2
    assert stats["created"] == 2
    assert state.cursor is None
    assert state.updated_at_watermark == "2026-11-07T02:00:00+00:00"
    # originals persisted per page
    assert (workdir / "raw" / "00_page.json").exists()
    # rerun with the advanced watermark → nothing re-created
    stats2 = import_case_fixture(repo, workdir, "proj", state)
    assert stats2["created"] == 0
    assert stats2["skipped_watermark"] == 2
    assert stats2["duplicate"] == 0


def test_import_duplicate_case_within_store(workdir: Path, tmp_path: Path):
    pages = workdir / "pages"
    _write_page(pages, "00_page.json", _page("proj", [_case("c1", "2026-11-07T01:00:00+00:00")]))
    repo = _repo(tmp_path)
    state = _state("proj")
    import_case_fixture(repo, workdir, "proj", state)
    # fresh workdir, same case content, same source identity → duplicate
    workdir2 = workdir.parent / "fixture2"
    workdir2.mkdir()
    pages2 = workdir2 / "pages"
    _write_page(pages2, "00_page.json", _page("proj", [_case("c1", "2026-11-07T01:00:00+00:00")]))
    state2 = _state("proj")
    # watermark pre-advanced to simulate prior collection of this updated_at
    state2.updated_at_watermark = "2026-11-07T00:00:00+00:00"
    stats = import_case_fixture(repo, workdir2, "proj", state2)
    assert stats["duplicate"] == 1


def test_import_tombstone_recorded_not_dropped(workdir: Path, tmp_path: Path):
    pages = workdir / "pages"
    _write_page(
        pages,
        "00_page.json",
        _page("proj", [_case("c1", "2026-11-07T01:00:00+00:00"), {"case_id": "c0", "deleted": True}]),
    )
    repo = _repo(tmp_path)
    state = _state("proj")
    stats = import_case_fixture(repo, workdir, "proj", state)
    assert stats["tombstoned"] == 1
    tomb_lines = (workdir / "tombstones.jsonl").read_text(encoding="utf-8").strip().splitlines()
    rec = json.loads(tomb_lines[0])
    assert rec["case_id"] == "c0" and rec["source_ref"] == "proj"


def test_import_mid_batch_failure_rerun_skips_finished_pages(workdir: Path, tmp_path: Path):
    pages = workdir / "pages"
    _write_page(pages, "00_page.json", _page("proj", [_case("c1", "2026-11-07T01:00:00+00:00")], next_cursor="01_page.json"))
    # page 2 contains a contract violation → import fails on it
    _write_page(pages, "01_page.json", _page("proj", [{"case_id": "broken"}]))

    repo = _repo(tmp_path)
    state = _state("proj")
    with pytest.raises(ConnectorError):
        import_case_fixture(repo, workdir, "proj", state)

    # page 1's case must already be persisted with checkpoint advanced
    assert repo.get_case("c1") is not None
    assert state.cursor == "01_page.json"

    # fix the broken page, rerun: page 1 not re-imported (watermark), page 2 succeeds
    _write_page(pages, "01_page.json", _page("proj", [_case("c2", "2026-11-07T02:00:00+00:00")]))
    stats = import_case_fixture(repo, workdir, "proj", state)
    assert stats["created"] == 1
    assert stats["skipped_watermark"] == 1
    assert repo.get_case("c2") is not None


def test_import_session_expired_preserves_watermark(workdir: Path, tmp_path: Path):
    pages = workdir / "pages"
    _write_page(pages, "00_page.json", _page("proj", [_case("c1", "2026-11-07T01:00:00+00:00")], next_cursor="01_page.json"))
    _write_page(pages, "01_page.json", _page("proj", [_case("c2", "2026-11-07T02:00:00+00:00")]))
    # fixture contract: a sibling marker file signals the session died after this page
    (pages / "01_page.session_expired").write_text("", encoding="utf-8")

    repo = _repo(tmp_path)
    state = _state("proj")
    with pytest.raises(AuthenticationRequired) as exc:
        import_case_fixture(repo, workdir, "proj", state)
    assert exc.value.source_id == "proj"
    # page 1 was durably collected; checkpoint is mid-chain and preserved
    assert repo.get_case("c1") is not None
    assert state.cursor == "01_page.json"

    # user re-authenticates: remove marker, rerun → page 2 only
    (pages / "01_page.session_expired").unlink()
    stats = import_case_fixture(repo, workdir, "proj", state)
    assert stats["created"] == 1
    assert state.cursor is None


def test_import_malformed_page_fails_fast(workdir: Path, tmp_path: Path):
    pages = workdir / "pages"
    pages.mkdir(parents=True)
    atomic_write_text(pages / "00_page.json", "{not json")
    repo = _repo(tmp_path)
    with pytest.raises(ConnectorError, match="invalid JSON"):
        import_case_fixture(repo, workdir, "proj", _state("proj"))


def test_import_unknown_source_page_missing(workdir: Path, tmp_path: Path):
    repo = _repo(tmp_path)
    with pytest.raises(ConnectorError, match="missing"):
        import_case_fixture(repo, workdir, "proj", _state("proj"))
