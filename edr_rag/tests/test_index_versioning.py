"""P3-5 索引版本灰度切换 / 回滚 —— 单元测试。

覆盖:
  - diff_records 的增量差异判定 (added/updated/removed/unchanged)。
  - capture_version 捕获版本并置 active、保留历史，版本号自增。
  - 整批 records_hash 与逐条 fingerprints 的计算。
  - activate_version / rollback 的灰度切换与回滚语义。
  - 版本注册表原子持久化，文件不存在时优雅返回空。
"""

from __future__ import annotations

import json

from edr_rag.index_versioning import (
    VERSIONS_SCHEMA_VERSION,
    activate_version,
    capture_version,
    compute_fingerprints,
    compute_records_hash,
    current_version,
    diff_records,
    load_versions,
    previous_version,
    rollback,
    versions_path,
)


def _rec(rid: str, text: str) -> dict:
    return {"id": rid, "_searchable": text}


# ---------- diff_records ----------


def test_diff_records_basic():
    current = {"a": "fp1", "b": "fp2", "c": "fp3"}
    known = {"a": "fp1", "b": "OLD", "d": "fp4"}
    diff = diff_records(current, known)
    assert diff.added == ["c"]
    assert diff.updated == ["b"]
    assert diff.removed == ["d"]
    assert diff.unchanged == ["a"]
    assert diff.total_changed == 3


def test_diff_records_empty():
    diff = diff_records({}, {})
    assert diff.total_changed == 0


# ---------- 指纹计算 ----------


def test_compute_fingerprints_and_hash():
    records = [_rec("a", "配置代理服务器"), _rec("b", "配置邮件告警")]
    fps = compute_fingerprints(records, id_field="id", text_field="_searchable")
    assert set(fps) == {"a", "b"}
    h = compute_records_hash(records, text_field="_searchable")
    assert isinstance(h, str) and len(h) == 40
    # 记录顺序变化不改变整批 hash (逐条拼接)
    h2 = compute_records_hash(list(reversed(records)), text_field="_searchable")
    assert h == h2


# ---------- capture / activate / rollback ----------


def test_capture_versions_and_history(tmp_path):
    recs1 = [_rec("a", "内容甲"), _rec("b", "内容乙")]
    v1, p = capture_version(tmp_path, corpus="chunks", records=recs1)
    assert p == versions_path(tmp_path)
    assert v1.version_id == 1
    assert v1.active is True

    recs2 = [_rec("a", "内容甲"), _rec("b", "内容乙"), _rec("c", "内容丙")]
    v2, _ = capture_version(tmp_path, corpus="chunks", records=recs2)
    assert v2.version_id == 2
    assert v2.active is True

    registry = load_versions(tmp_path)
    versions = registry["chunks"]
    assert len(versions) == 2
    # v1 历史保留但不再 active, v2 为 active
    by_id = {v.version_id: v for v in versions}
    assert by_id[1].active is False
    assert by_id[2].active is True
    assert current_version(registry, "chunks").version_id == 2


def test_rollback_to_previous(tmp_path):
    recs1 = [_rec("a", "内容甲")]
    recs2 = [_rec("a", "内容甲"), _rec("b", "内容乙")]
    capture_version(tmp_path, corpus="procedures", records=recs1)
    capture_version(tmp_path, corpus="procedures", records=recs2)

    target = rollback(tmp_path, corpus="procedures")
    assert target is not None
    assert target.version_id == 1
    registry = load_versions(tmp_path)
    assert current_version(registry, "procedures").version_id == 1
    restored = [
        json.loads(line)
        for line in (tmp_path / "procedures.jsonl").read_text().splitlines()
    ]
    assert restored == recs1


def test_rollback_when_no_history_returns_none(tmp_path):
    capture_version(tmp_path, corpus="actions", records=[_rec("a", "x")])
    # 只有 1 个版本, 无历史可回退
    assert rollback(tmp_path, corpus="actions") is None
    # 无任何版本的 corpus 也不报错, 返回 None
    assert rollback(tmp_path, corpus="missing") is None


def test_activate_specific_version(tmp_path):
    capture_version(tmp_path, corpus="chunks", records=[_rec("a", "1")])
    capture_version(tmp_path, corpus="chunks", records=[_rec("a", "1"), _rec("b", "2")])
    capture_version(tmp_path, corpus="chunks", records=[_rec("a", "1"), _rec("b", "2"), _rec("c", "3")])
    target = activate_version(tmp_path, corpus="chunks", version_id=2)
    assert target.version_id == 2
    registry = load_versions(tmp_path)
    assert current_version(registry, "chunks").version_id == 2
    restored = [
        json.loads(line)
        for line in (tmp_path / "chunks.jsonl").read_text().splitlines()
    ]
    assert restored == [_rec("a", "1"), _rec("b", "2")]


def test_activate_refuses_metadata_only_version(tmp_path):
    capture_version(
        tmp_path,
        corpus="chunks",
        fingerprints={"a": "fingerprint-only"},
    )

    assert activate_version(tmp_path, corpus="chunks", version_id=1) is None
    assert not (tmp_path / "chunks.jsonl").exists()


def test_activate_missing_version_returns_none(tmp_path):
    capture_version(tmp_path, corpus="chunks", records=[_rec("a", "1")])
    assert activate_version(tmp_path, corpus="chunks", version_id=99) is None
    # 注册表未被改动
    registry = load_versions(tmp_path)
    assert current_version(registry, "chunks").version_id == 1


def test_capture_drop_history(tmp_path):
    capture_version(tmp_path, corpus="chunks", records=[_rec("a", "1")])
    v2, _ = capture_version(
        tmp_path, corpus="chunks", records=[_rec("a", "2")], keep_history=False
    )
    # 历史被丢弃: 注册表里只有新版本; version_id 仍单调递增 (跨库全局唯一)。
    assert v2.version_id == 2
    registry = load_versions(tmp_path)
    assert len(registry["chunks"]) == 1
    assert registry["chunks"][0].version_id == 2
    # 丢弃历史后无法回滚
    assert rollback(tmp_path, corpus="chunks") is None


def test_load_versions_empty_when_file_missing(tmp_path):
    assert load_versions(tmp_path) == {}
    assert current_version(load_versions(tmp_path), "chunks") is None


def test_versions_json_persisted(tmp_path):
    capture_version(tmp_path, corpus="chunks", records=[_rec("a", "x")], source_hash="src1")
    p = versions_path(tmp_path)
    data = json.loads(p.read_text("utf-8"))
    assert data["schema_version"] == VERSIONS_SCHEMA_VERSION
    ver = data["versions"]["chunks"][0]
    assert ver["corpus"] == "chunks"
    assert ver["source_hash"] == "src1"
    assert set(ver["fingerprints"]) == {"a"}
    assert (tmp_path / ver["snapshot_file"]).is_file()


def test_previous_version_returns_older(tmp_path):
    capture_version(tmp_path, corpus="chunks", records=[_rec("a", "1")])
    capture_version(tmp_path, corpus="chunks", records=[_rec("a", "1"), _rec("b", "2")])
    v = previous_version(load_versions(tmp_path), "chunks")
    assert v.version_id == 1
