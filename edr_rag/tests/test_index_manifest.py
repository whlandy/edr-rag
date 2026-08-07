"""P0-5 单元测试:索引 manifest 版本/hash/过期检测。

覆盖:
  1. 生成: write_index_manifest / maintain_index 会写入 index_manifest.json，
     记录 schema_version/built_at/source_hash/各产物 sha256+行数。
  2. 校验 OK: 产物未变时 validate_index_manifest 返回 OK。
  3. 内容过期: 改动某产物后校验返回 STALE_CONTENT (warning 级, 不 refuse)。
  4. 源头变化: source_hash 改变时返回 SOURCE_MISMATCH (warning 级)。
  5. 孤儿残留: 源为空 (sections.jsonl 缺失/无内容) 但索引产物仍在 -> ORPHAN_INDEX
     (refuse 级, refuse==True)。
  6. schema 不匹配: 版本不符 -> SCHEMA_MISMATCH (refuse 级)。
  7. 无索引: 未见 index_manifest.json -> NO_INDEX。
  8. 与 P1/P2 生成链路的集成: write_p1_artifacts / write_p2_artifacts 也会维护索引。
"""

from __future__ import annotations

from pathlib import Path

from edr_rag.index_manifest import (
    INDEX_MANIFEST_SCHEMA_VERSION,
    VERDICT_NO_INDEX,
    VERDICT_OK,
    VERDICT_ORPHAN_INDEX,
    VERDICT_SCHEMA_MISMATCH,
    VERDICT_SOURCE_MISMATCH,
    VERDICT_STALE_CONTENT,
    compute_artifacts_hashes,
    index_path,
    maintain_index,
    validate_index_manifest,
    write_index_manifest,
)


def _write_jsonl(path: Path, lines: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln + "\n")
    return path


def _seed_workdir(workdir: Path, sections: list[str] | None = None) -> Path:
    """写入一个最小产物目录: sections.jsonl + 可选上游清单。"""
    _write_jsonl(workdir / "sections.jsonl", sections or [
        '{"id": "s1", "manual_id": "M", "title": "第一节", "content": "hello"}',
        '{"id": "s2", "manual_id": "M", "title": "第二节", "content": "world"}',
    ])
    return workdir


def test_maintain_index_writes_schema_and_hashes(tmp_path):
    """生成: maintain_index 写入 schema_version/source_hash/artifacts hash。"""
    workdir = _seed_workdir(tmp_path)
    ip = maintain_index(workdir, chain="p1", artifacts=["sections.jsonl"])

    assert ip.exists()
    import json
    d = json.loads(ip.read_text(encoding="utf-8"))
    assert d["schema_version"] == INDEX_MANIFEST_SCHEMA_VERSION
    assert d["chain"] == "p1"
    assert d["built_at"]  # 建立时间
    assert "source_hash" in d
    art = d["artifacts"]["sections.jsonl"]
    assert art["line_count"] == 2
    assert art["sha256"]  # 内容 hash

    # hash 与直接计算一致
    hashes = compute_artifacts_hashes(workdir, ["sections.jsonl"])
    assert hashes["sections.jsonl"].sha256 == art["sha256"]


def test_validate_ok_when_unchanged(tmp_path):
    """产物未变时校验通过。"""
    workdir = _seed_workdir(tmp_path)
    maintain_index(workdir, chain="p1", artifacts=["sections.jsonl"])
    res = validate_index_manifest(workdir, chain="p1")
    assert res.verdict == VERDICT_OK
    assert res.ok
    assert not res.refuse


def test_validate_stale_content_when_artifact_changes(tmp_path):
    """改动产物后校验返回 STALE_CONTENT (content hash 不匹配)。"""
    workdir = _seed_workdir(tmp_path)
    maintain_index(workdir, chain="p1", artifacts=["sections.jsonl"])

    # 篡改产物
    _write_jsonl(workdir / "sections.jsonl", [
        '{"id": "s1", "manual_id": "M", "title": "第一节", "content": "changed"}',
        '{"id": "s2", "manual_id": "M", "title": "第二节", "content": "world"}',
    ])

    res = validate_index_manifest(workdir, chain="p1")
    assert res.verdict == VERDICT_STALE_CONTENT
    assert not res.ok
    assert not res.refuse  # 内容过期是 warning 级, 不阻止
    assert "sections.jsonl" in res.stale_files


def test_validate_source_mismatch(tmp_path):
    """source_hash 变化时返回 SOURCE_MISMATCH (warning 级)。"""
    workdir = _seed_workdir(tmp_path)
    maintain_index(
        workdir, chain="p1", artifacts=["sections.jsonl"], source_hash="abc123"
    )
    res = validate_index_manifest(workdir, chain="p1", source_hash="xyz789")
    assert res.verdict == VERDICT_SOURCE_MISMATCH
    assert not res.refuse


def test_validate_orphan_index_when_source_empty(tmp_path):
    """源为空但旧索引产物仍在 -> ORPHAN_INDEX (refuse 级)。"""
    workdir = _seed_workdir(tmp_path)
    maintain_index(workdir, chain="p1", artifacts=["sections.jsonl"])

    # 清空源头 (模拟:重新 ingest 后源缺失)
    (workdir / "sections.jsonl").unlink()

    res = validate_index_manifest(workdir, chain="p1")
    assert res.verdict == VERDICT_ORPHAN_INDEX
    assert res.refuse  # 拒绝读取
    assert "sections.jsonl" in res.stale_files


def test_validate_orhan_not_flagged_when_source_empty_and_index_empty(tmp_path):
    """源为空且索引产物也为空时不算孤儿残留 (避免误报)。"""
    workdir = Path(tmp_path)
    (workdir / "sections.jsonl").unlink(missing_ok=True)
    maintain_index(workdir, chain="p1", artifacts=["sections.jsonl"])
    res = validate_index_manifest(workdir, chain="p1")
    # 源为空 + 索引无内容 -> 不应判为孤儿, 而是内容一致 (OK)
    assert res.verdict == VERDICT_OK


def test_validate_schema_mismatch(tmp_path):
    """schema_version 不符 -> SCHEMA_MISMATCH (refuse 级)。"""
    workdir = _seed_workdir(tmp_path)
    write_index_manifest(
        workdir,
        chain="p1",
        artifacts=compute_artifacts_hashes(workdir, ["sections.jsonl"]),
        schema_version=INDEX_MANIFEST_SCHEMA_VERSION + 999,
    )
    res = validate_index_manifest(workdir, chain="p1")
    assert res.verdict == VERDICT_SCHEMA_MISMATCH
    assert res.refuse


def test_validate_no_index(tmp_path):
    """未见 index_manifest.json -> NO_INDEX。"""
    workdir = _seed_workdir(tmp_path)
    res = validate_index_manifest(workdir)
    assert res.verdict == VERDICT_NO_INDEX
    assert not res.ok
    assert not res.refuse


def test_chain_mismatch_refused(tmp_path):
    """期望链路不匹配时拒绝。"""
    workdir = _seed_workdir(tmp_path)
    maintain_index(workdir, chain="p1", artifacts=["sections.jsonl"])
    res = validate_index_manifest(workdir, chain="p2")
    assert res.verdict == VERDICT_SCHEMA_MISMATCH
    assert res.refuse


# ---------- 与 P1/P2 生成链路的集成 ----------


def test_write_p1_artifacts_maintains_index(tmp_path):
    """write_p1_artifacts 会写 procedures/chunks 并维护 index_manifest (chain=p1)。"""
    from edr_rag.p1.extract_procedures import write_p1_artifacts
    from edr_rag.p1.schema import Procedure, ProcedureSource, ProcedureStep

    workdir = _seed_workdir(tmp_path)
    proc = Procedure(
        id="procedure.net.configure",
        manual_id="M",
        product="P",
        operation="configure",
        title="配置",
        section_path=["网络"],
        source=ProcedureSource(file="a.html", section_id="s1"),
        steps=[ProcedureStep(order=1, instruction="点击设置")],
    )
    write_p1_artifacts(
        workdir, [proc], [], [],
        manual_id="M", product="P",
    )

    assert (workdir / "procedures.jsonl").exists()
    ip = index_path(workdir)
    assert ip.exists()
    import json
    d = json.loads(ip.read_text(encoding="utf-8"))
    assert d["chain"] == "p1"
    assert "procedures.jsonl" in d["artifacts"]
    assert "chunks.jsonl" in d["artifacts"]

    res = validate_index_manifest(workdir, chain="p1")
    assert res.verdict == VERDICT_OK


def test_write_p2_artifacts_maintains_index(tmp_path):
    """write_p2_artifacts 会写 actions 并维护 index_manifest (chain=p2)。"""
    from edr_rag.p2 import ActionCatalogEntry
    from edr_rag.p2.extract_actions import write_p2_artifacts

    workdir = _seed_workdir(tmp_path)
    entry = ActionCatalogEntry(
        action_id="net.configure",
        product="P",
        module="网络",
        title="配置",
        status="candidate",
        confidence=0.8,
        risk="low",
        requires_confirmation=False,
    )
    write_p2_artifacts(workdir, [entry])

    assert (workdir / "actions.jsonl").exists()
    import json
    d = json.loads(index_path(workdir).read_text(encoding="utf-8"))
    assert d["chain"] == "p2"
    assert "actions.jsonl" in d["artifacts"]

    res = validate_index_manifest(workdir, chain="p2")
    assert res.verdict == VERDICT_OK
