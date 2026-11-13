"""同 ID 异内容身份契约（[1433 审核确认] 第二契约漏洞）。

背景：ingest 按导航路径拼 section ID，跨 topic HTML 碰撞产生同 ID 行。
``_dedupe_identical_id_rows`` 只收敛「检索等价」的重复；内容确实不同的
变体保留后，``VectorIndex._embed_upsert`` 按 ID upsert → 后写覆盖，
只有最后一个变体的向量幸存（其余变体在向量路静默不可检索）。
增量路径更糟：``index_records_incremental`` 的 by_id/new_known 都按 ID
收敛，前位变体变化时以最后变体的文本在最后位置重嵌入，doc_idx 映射错乱。

真实形状（duli sections 读取侧去重后仍残留 4 组，如
manual.section.登录_管理面…：同 ID 三变体浏览器要求 chrome 129 vs 124
vs 奇安信——不同发布版本的实质差异）。
"""

from __future__ import annotations

import json

from edr_rag.p1.search import JsonlCorpus
from edr_rag.vectorstore import SimpleVectorStore, VectorIndex


def _embed(text: str) -> list[float]:
    seed = sum(ord(c) for c in text)
    return [float(seed % 11) / 11.0, float(seed % 7) / 7.0, float(seed % 5) / 5.0]


def _sec(rid: str, content: str) -> dict:
    return {
        "id": rid,
        "type": "section",
        "title": "登录管理面",
        "content": content,
        "section_path": ["前提条件", "操作步骤"],
        "metadata": {"version": "V100R026"},
        "source_ref": {"relative_path": "t.html"},
        "links": [],
        "images": [],
    }


# 语料级：不同内容变体必须 ID 可区分（读取侧，不加磁盘改写）

class TestCorpusDisambiguatesDuplicateIds:
    def test_from_file_assigns_distinct_internal_ids_to_content_variants(self, tmp_path):
        rows = [
            _sec("s.dup", "chrome 129 及以上版本，" + "甲" * 200),
            _sec("s.dup", "chrome 124 及以上版本，" + "乙" * 200),
            _sec("s.other", "无关内容" + "丙" * 200),
        ]
        f = tmp_path / "sections.jsonl"
        f.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
            encoding="utf-8",
        )
        corpus = JsonlCorpus.from_file(f)
        ids = [str(r["_doc_id"]) for r in corpus.records]
        # 三个 record 三条可区分身份：检索层不再有两个 doc 共享同一 upsert 键
        assert len(ids) == 3
        assert len(set(ids)) == 3
        # 对外 ID 保持原语义：代表（首现）变体的公开 ID 不变
        assert corpus.records[0]["id"] == "s.dup"

    def test_equal_variants_still_collapsed_by_existing_dedupe(self, tmp_path):
        rows = [
            _sec("s.dup", "正文甲" + "x" * 200),
            _sec("s.dup", "正文甲" + "x" * 200),
        ]
        f = tmp_path / "sections.jsonl"
        f.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
            encoding="utf-8",
        )
        corpus = JsonlCorpus.from_file(f)
        assert len(corpus.records) == 1

    def test_vector_store_retrieves_every_variant_after_indexing(self, tmp_path):
        rows = [
            _sec("s.dup", "chrome 129 及以上版本，" + "甲" * 200),
            _sec("s.dup", "chrome 124 及以上版本，" + "乙" * 200),
        ]
        f = tmp_path / "sections.jsonl"
        f.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
            encoding="utf-8",
        )
        corpus = JsonlCorpus.from_file(f)
        store = SimpleVectorStore()
        index = VectorIndex(store, _embed)
        n = index.index_records(corpus.records, text_field="_searchable", id_field="_doc_id")
        assert n == 2
        assert store.count() == 2


# 索引级：直接喂同 ID 记录时 upsert 不许静默覆盖（防御 + 契约固定）

class TestVectorIndexRejectsDuplicateIdRecords:
    def test_full_rebuild_preserves_both_variants(self):
        store = SimpleVectorStore()
        index = VectorIndex(store, _embed)
        records = [
            {"_doc_id": "s.dup", "_searchable": "内容甲" + "x" * 100},
            {"_doc_id": "s.dup", "_searchable": "内容乙" + "y" * 100},
        ]
        n = index.index_records(records, text_field="_searchable", id_field="_doc_id")
        # 后写覆盖是静默数据丢失：n 报 2 而存里只有 1 条
        assert n == 2
        assert store.count() == 2

    def test_incremental_change_to_first_variant_reembeds_right_row(self):
        store = SimpleVectorStore()
        index = VectorIndex(store, _embed)
        v1 = [
            {"_doc_id": "s.dup", "_searchable": "内容甲" + "x" * 100},
            {"_doc_id": "s.dup", "_searchable": "内容乙" + "y" * 100},
        ]
        _, known = index.index_records_incremental(
            v1, known=None, text_field="_searchable", id_field="_doc_id"
        )
        # 前位变体单独变化
        v2 = [
            {"_doc_id": "s.dup", "_searchable": "内容甲改" + "x" * 100},
            {"_doc_id": "s.dup", "_searchable": "内容乙" + "y" * 100},
        ]
        result, new_known = index.index_records_incremental(
            v2, known=known, text_field="_searchable", id_field="_doc_id"
        )
        assert result.updated == ["s.dup"] or result.added
        assert store.count() == 2
        # 修复前形状：known/store 按 ID 收敛到 1 键 1 行，前位变体变化时
        # 以最后变体文本在最后位置重嵌入。修复后每个变体独享身份：
        # 用各变体自身文本自检，必须命中最先是其自己的行（stub 词表下
        # 查询相似度有字符和碰撞，不以跨行排名断言）。
        assert index.search(v2[0]["_searchable"], top_k=1)[0][1] == 0
        assert index.search(v2[1]["_searchable"], top_k=1)[0][1] == 1
