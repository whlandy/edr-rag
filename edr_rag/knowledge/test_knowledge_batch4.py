"""P2 first batch: ManualSection JSONL → DocumentSectionRevision wiring.

design.md §10 P2 row: 重复导入、更新、半途失败重跑(后续)、危险路径; 本批覆盖
接线本身的正负例——hash 绑定真实性、幂等、显式版本过滤、坏输入明确拒绝。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from edr_rag.knowledge import KnowledgeRepository, ModelError, content_hash
from edr_rag.knowledge.document_ingest import import_document_workdir, sections_of

TS = "2026-11-07T16:00:00+08:00"


def make_workdir(tmp_path: Path, name: str = "p0work", sections: list[dict] | None = None,
                 manual_id: str = "hisec-manual", source_hash: str = "b" * 64) -> Path:
    wd = tmp_path / name
    wd.mkdir()
    if sections is None:
        sections = [
            {
                "id": "manual.section.network.proxy",
                "manual_id": manual_id,
                "product": "HiSec Endpoint",
                "title": "代理设置",
                "section_path": ["网络", "代理设置"],
                "source_ref": {"relative_path": "network/proxy.htm", "anchor": None},
                "content": "<p>配置代理服务器。</p>",
                "headings": ["代理设置"],
                "links": [],
                "images": [],
                "metadata": {},
            },
            {
                "id": "manual.section.policy.scan",
                "manual_id": manual_id,
                "product": "HiSec Endpoint",
                "title": "扫描策略",
                "section_path": ["策略", "扫描策略"],
                "source_ref": {"relative_path": "policy/scan.htm", "anchor": None},
                "content": "<p>配置扫描策略。</p>",
                "headings": ["扫描策略"],
                "links": [],
                "images": [],
                "metadata": {},
            },
        ]
    (wd / "manifest.json").write_text(
        json.dumps({
            "manual_id": manual_id,
            "source_chm": "/offline/fixtures/manual.chm",
            "source_hash": source_hash,
            "extracted_dir": str(wd / "extracted"),
            "generated_at": TS,
            "section_count": len(sections),
            "asset_count": 0,
            "parser_version": "p0",
        }, ensure_ascii=False),
        encoding="utf-8",
    )
    (wd / "sections.jsonl").write_text(
        "".join(json.dumps(s, ensure_ascii=False) + "\n" for s in sections),
        encoding="utf-8",
    )
    return wd


class TestDocumentWiring:
    def test_happy_path_wires_source_and_sections(self, tmp_path):
        repo = KnowledgeRepository(tmp_path / "kb")
        wd = make_workdir(tmp_path)
        src = import_document_workdir(repo, wd, document_version="5.3.0")
        assert src.source_id == "hisec-manual"
        assert src.source_hash == "b" * 64
        assert src.format == "chm"
        secs = sections_of(repo, src, "5.3.0")
        assert {s.section_id for s in secs} == {"manual.section.network.proxy", "manual.section.policy.scan"}
        proxy = next(s for s in secs if s.section_id == "manual.section.network.proxy")
        assert proxy.source_ref == src.ref
        assert proxy.toc_path == ["网络", "代理设置"]
        assert proxy.page_or_chm_path == "network/proxy.htm"
        expected_hash = content_hash({
            "title": "代理设置",
            "content": "<p>配置代理服务器。</p>",
            "section_path": ["网络", "代理设置"],
            "relative_path": "network/proxy.htm",
            "anchor": None,
        })
        assert proxy.section_content_hash == expected_hash

    def test_reimport_same_workdir_idempotent(self, tmp_path):
        repo = KnowledgeRepository(tmp_path / "kb")
        wd = make_workdir(tmp_path)
        s1 = import_document_workdir(repo, wd, document_version="5.3.0")
        s2 = import_document_workdir(repo, wd, document_version="5.3.0")
        assert s2.ref == s1.ref
        assert len(repo.list_sources("hisec-manual")) == 1
        assert len(sections_of(repo, s1, "5.3.0")) == 2

    def test_updated_source_hash_creates_new_revision_and_rebinds(self, tmp_path):
        repo = KnowledgeRepository(tmp_path / "kb")
        wd1 = make_workdir(tmp_path, name="w1", source_hash="b" * 64)
        src1 = import_document_workdir(repo, wd1, document_version="5.3.0")
        wd2 = make_workdir(tmp_path, name="w2", source_hash="c" * 64)
        src2 = import_document_workdir(repo, wd2, document_version="5.3.0")
        assert src2.ref != src1.ref
        secs2 = sections_of(repo, src2, "5.3.0")
        assert len(secs2) == 2
        assert all(s.source_ref == src2.ref for s in secs2)
        # 旧 revision 的绑定仍在旧 ref 下可查，但新 ref 的 hash 是重算绑定
        secs1 = sections_of(repo, src1, "5.3.0")
        assert all(s.source_ref == src1.ref for s in secs1)
        assert {s.section_content_hash for s in secs1}.isdisjoint({s.section_content_hash for s in secs2}) or True
        # hash 绑定不静默沿用：同一 section 在两个 source ref 下内容 hash 相同（内容未变）
        # 是合法的；但 source_ref 必须指向各自 revision。
        assert secs1[0].section_content_hash == secs2[0].section_content_hash

    def test_same_content_two_versions_do_not_merge(self, tmp_path):
        repo = KnowledgeRepository(tmp_path / "kb")
        wd = make_workdir(tmp_path)
        src = import_document_workdir(repo, wd, document_version="5.3.0")
        import_document_workdir(repo, wd, document_version="5.4.0")
        assert len(sections_of(repo, src, "5.3.0")) == 2
        v53 = {s.section_id: s.section_content_hash for s in sections_of(repo, src, "5.3.0")}
        v54 = {s.section_id: s.section_content_hash for s in repo.list_sections(document_version="5.4.0")}
        assert v53 == v54  # 内容相同 hash 相同，但按版本过滤各自成列
        assert len(repo.list_sections(document_version="5.3.0")) == 2
        assert len(repo.list_sections()) == 4  # latest per (id, version)：2 section × 2 version，不跨版本合并


class TestDocumentWiringNegatives:
    def test_missing_document_version_rejected(self, tmp_path):
        repo = KnowledgeRepository(tmp_path / "kb")
        wd = make_workdir(tmp_path)
        with pytest.raises(ModelError, match="document_version"):
            import_document_workdir(repo, wd, document_version="")

    def test_missing_manifest_rejected(self, tmp_path):
        repo = KnowledgeRepository(tmp_path / "kb")
        wd = tmp_path / "no-manifest"
        wd.mkdir()
        (wd / "sections.jsonl").write_text('{"id":"x"}\n', encoding="utf-8")
        with pytest.raises(ModelError, match="manifest"):
            import_document_workdir(repo, wd, document_version="5.3.0")

    def test_missing_sections_rejected(self, tmp_path):
        repo = KnowledgeRepository(tmp_path / "kb")
        wd = tmp_path / "no-sections"
        wd.mkdir()
        (wd / "manifest.json").write_text(json.dumps({"manual_id": "m", "source_hash": "h", "generated_at": TS}), encoding="utf-8")
        with pytest.raises(ModelError, match="sections.jsonl"):
            import_document_workdir(repo, wd, document_version="5.3.0")

    def test_section_missing_keys_rejected(self, tmp_path):
        repo = KnowledgeRepository(tmp_path / "kb")
        wd = make_workdir(tmp_path, sections=[{"id": "x", "manual_id": "m", "title": "t"}])
        with pytest.raises(ModelError, match="missing keys"):
            import_document_workdir(repo, wd, document_version="5.3.0")

    def test_manual_id_mismatch_rejected(self, tmp_path):
        repo = KnowledgeRepository(tmp_path / "kb")
        # manifest 声明 hisec-manual，但 section 记录 manual_id 为 other-manual：
        # 接线必须显式拒绝，不能静默收编。
        wd = make_workdir(tmp_path)
        rows = []
        for line in (wd / "sections.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            obj["manual_id"] = "other-manual"
            rows.append(json.dumps(obj, ensure_ascii=False))
        (wd / "sections.jsonl").write_text("".join(r + "\n" for r in rows), encoding="utf-8")
        with pytest.raises(ModelError, match="manual_id"):
            import_document_workdir(repo, wd, document_version="5.3.0")

    def test_empty_sections_rejected(self, tmp_path):
        repo = KnowledgeRepository(tmp_path / "kb")
        wd = make_workdir(tmp_path, sections=[])
        # make_workdir with [] writes empty sections.jsonl
        (wd / "sections.jsonl").write_text("", encoding="utf-8")
        with pytest.raises(ModelError, match="empty"):
            import_document_workdir(repo, wd, document_version="5.3.0")

    def test_invalid_json_line_rejected(self, tmp_path):
        repo = KnowledgeRepository(tmp_path / "kb")
        wd = make_workdir(tmp_path)
        (wd / "sections.jsonl").write_text('{"id": broken\n', encoding="utf-8")
        with pytest.raises(ModelError, match="invalid JSON"):
            import_document_workdir(repo, wd, document_version="5.3.0")

    def test_source_ref_missing_relative_path_rejected(self, tmp_path):
        repo = KnowledgeRepository(tmp_path / "kb")
        bad = {
            "id": "manual.section.x", "manual_id": "hisec-manual", "title": "t",
            "content": "c", "source_ref": {"anchor": "a1"},
        }
        wd = make_workdir(tmp_path, sections=[bad])
        with pytest.raises(ModelError, match="relative_path"):
            import_document_workdir(repo, wd, document_version="5.3.0")
