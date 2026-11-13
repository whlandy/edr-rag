"""缺陷②最小缓解回归：读取侧对「同 ID 且检索等价」的重复行去重。

背景：ingest 按导航路径拼 section ID，跨 topic HTML 碰撞产生重复 ID。
对「除 source_ref/links/images 外逐字段相同」的组（检索等价副本），
keyword/bm25 会重复计分挤占 top-k、vector upsert 按 ID 覆盖只剩一行。
修复在 JsonlCorpus.from_file 读取侧去重，不改磁盘产物、不改 ID；
「内容确实不同」的变体保持原样（哪个变体可见属 ingest 层决策）。
"""
from __future__ import annotations

import json

import pytest

from edr_rag.p1.search import JsonlCorpus


def _sec(rid, content, source_ref="a.html", links=None, images=None):
    return {
        "id": rid,
        "type": "section",
        "title": "关闭用户的双因素认证",
        "content": content,
        "section_path": ["前提条件", "操作步骤"],
        "metadata": {"version": "V100R026"},
        "source_ref": {"relative_path": source_ref, "element_path": "div"},
        "links": links or [{"text": "相关任务", "href": "x.html"}],
        "images": images or [],
    }


class TestDedupeIdenticalIdRows:
    def test_identical_variant_dropped_keeps_first_source_ref(self):
        rows = [
            _sec("s.1", "正文", source_ref="one.html"),
            _sec("s.1", "正文", source_ref="two.html"),
        ]
        out, dropped = JsonlCorpus._dedupe_identical_id_rows(rows)
        assert dropped == 1
        assert len(out) == 1
        assert out[0]["source_ref"]["relative_path"] == "one.html"

    def test_links_images_source_ref_differences_do_not_block_dedupe(self):
        rows = [
            _sec("s.1", "正文", links=[{"text": "a", "href": "1.html"}]),
            _sec("s.1", "正文", links=[{"text": "b", "href": "2.html"}],
                 images=["img2.png"], source_ref="other.html"),
        ]
        out, dropped = JsonlCorpus._dedupe_identical_id_rows(rows)
        assert dropped == 1 and out[0]["id"] == "s.1"

    def test_content_variant_kept_entirely(self):
        rows = [
            _sec("s.1", "正文A"),
            _sec("s.1", "正文B"),
        ]
        out, dropped = JsonlCorpus._dedupe_identical_id_rows(rows)
        assert dropped == 0 and len(out) == 2

    def test_mixed_group_drops_only_identical_variants(self):
        # A, A, B -> A, B：混合组内同变体仍去重，不同变体全保留
        rows = [
            _sec("s.1", "正文A", source_ref="1.html"),
            _sec("s.1", "正文A", source_ref="2.html"),
            _sec("s.1", "正文B", source_ref="3.html"),
        ]
        out, dropped = JsonlCorpus._dedupe_identical_id_rows(rows)
        assert dropped == 1
        assert len(out) == 2
        contents = {r["content"] for r in out}
        assert contents == {"正文A", "正文B"}

    def test_metadata_difference_blocks_dedupe(self):
        r1 = _sec("s.1", "正文")
        r2 = _sec("s.1", "正文")
        r2["metadata"] = {"version": "V100R027"}
        out, dropped = JsonlCorpus._dedupe_identical_id_rows([r1, r2])
        assert dropped == 0 and len(out) == 2

    def test_distinct_ids_unaffected(self):
        rows = [_sec("s.1", "正文"), _sec("s.2", "正文")]
        out, dropped = JsonlCorpus._dedupe_identical_id_rows(rows)
        assert dropped == 0 and len(out) == 2

    def test_missing_id_never_dropped(self):
        r1 = _sec(None, "正文")
        r2 = _sec(None, "正文")
        out, dropped = JsonlCorpus._dedupe_identical_id_rows([r1, r2])
        assert dropped == 0 and len(out) == 2

    def test_from_file_dedupes_without_touching_disk(self, tmp_path):
        rows = [
            _sec("s.1", "正文", source_ref="one.html"),
            _sec("s.1", "正文", source_ref="two.html"),
            _sec("s.2", "其他"),
        ]
        f = tmp_path / "sections.jsonl"
        f.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
            encoding="utf-8",
        )
        corpus = JsonlCorpus.from_file(f)
        assert len(corpus.records) == 2
        assert {r["id"] for r in corpus.records} == {"s.1", "s.2"}
        # 磁盘产物不动
        assert len(f.read_text(encoding="utf-8").strip().splitlines()) == 3

    def test_bm25_no_duplicate_hits_for_deduped_corpus(self, tmp_path):
        rows = [
            _sec("s.dup", "关闭用户的双因素认证 操作步骤"),
            _sec("s.dup", "关闭用户的双因素认证 操作步骤"),
            _sec("s.other", "完全无关的内容"),
        ]
        f = tmp_path / "sections.jsonl"
        f.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
            encoding="utf-8",
        )
        corpus = JsonlCorpus.from_file(f)
        hits = corpus.bm25_search("关闭用户的双因素认证", limit=5)
        ids = [h["id"] for h in hits]
        assert ids.count("s.dup") == 1


# ---------- 内容级去重（不同 ID 同 content 收敛为代表，[0310] 方案） ----------

def _sec_long(rid, content, title="创建病毒扫描任务"):
    r = _sec(rid, content)
    r["title"] = title
    return r


class TestDedupeContentRows:
    def test_identical_content_dedupes_to_first(self):
        body = "操作步骤 " + "x" * 250
        rows = [_sec_long("s.a", body), _sec_long("s.b", body), _sec_long("s.c", "无关" + "y" * 250)]
        out, rep_of = JsonlCorpus._dedupe_content_rows(rows)
        assert [r["id"] for r in out] == ["s.a", "s.c"]
        assert rep_of == {"s.b": "s.a"}

    def test_short_content_never_grouped(self):
        # <200 字节的短节（导航锚点/样板）不参与分组
        rows = [_sec_long("s.a", "短"), _sec_long("s.b", "短")]
        out, rep_of = JsonlCorpus._dedupe_content_rows(rows)
        assert len(out) == 2 and rep_of == {}

    def test_diff_content_not_grouped(self):
        rows = [_sec_long("s.a", "内容甲" + "x" * 250), _sec_long("s.b", "内容乙" + "x" * 250)]
        out, rep_of = JsonlCorpus._dedupe_content_rows(rows)
        assert len(out) == 2 and rep_of == {}

    def test_requires_id_dedupe_first(self):
        # 文档约定：必须先跑 _dedupe_identical_id_rows；同 ID 行
        # 「content+title+operation」键相同（D-0659-1 方案 B）时若跳过
        # 前置会错误并组——这里验证顺序由 from_file 保证。
        body = "x" * 250
        rows = [_sec_long("s.dup", body, title="甲"), _sec_long("s.dup", body, title="甲")]
        out, rep_of = JsonlCorpus._dedupe_content_rows(rows)
        # 直接调用时按内容键并组（rep 指向首个）；from_file 组合路径下
        # 同 ID 等价行先收敛——顺序契约由 test_from_file_two_level_dedupe 固定。
        assert rep_of == {"s.dup": "s.dup"}

    def test_representative_id_api(self, tmp_path):
        body = "y" * 250
        rows = [_sec_long("s.keep", body), _sec_long("s.gone", body), _sec_long("s.solo", "z" * 250)]
        f = tmp_path / "sections.jsonl"
        f.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
            encoding="utf-8",
        )
        corpus = JsonlCorpus.from_file(f)
        assert corpus.representative_id("s.gone") == "s.keep"
        assert corpus.representative_id("s.keep") == "s.keep"
        assert corpus.representative_id("s.solo") == "s.solo"
        assert corpus.representative_id("不存在的id") == "不存在的id"

    def test_from_file_two_level_dedupe(self, tmp_path):
        # 同 ID 等价行先收敛，剩余内容组再收敛：两级都不改磁盘。
        body = "w" * 250
        rows = [
            _sec_long("s.dup", body, title="同题"),
            _sec_long("s.dup", body, title="同题"),   # [1125] 同 ID 等价 → 收敛
            _sec_long("s.twin", body, title="同题"),   # 内容级孪生 → 移除
            _sec_long("s.solo", "z" * 250, title="独"),
        ]
        f = tmp_path / "sections.jsonl"
        f.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
            encoding="utf-8",
        )
        corpus = JsonlCorpus.from_file(f)
        assert [r["id"] for r in corpus.records] == ["s.dup", "s.solo"]
        assert corpus.representative_id("s.twin") == "s.dup"
        assert len(f.read_text(encoding="utf-8").strip().splitlines()) == 4


# ---------- 缺陷③：内容级去重对 text 字段语料（chunks/nodes）失效 ----------

def _chunk(rid, text):
    return {
        "id": rid,
        "manual_id": "hisec-anquan-v100r026",
        "type": "step",
        "text": text,
        "title": "配置病毒扫描任务",
        "metadata": {"section_path": ["病毒扫描", "配置步骤"]},
    }


class TestDedupeTextFieldCorpora:
    """chunks/nodes 语料正文在 text 字段（非 content），去重必须同口径。

    缺陷③ [0652]：_dedupe_content_rows 只读 content 字段，chunks 语料
    静默不参与内容级去重（实测可去重 anquan -7.8% / duli -11.2%）。
    """

    def test_text_field_identical_content_dedupes_to_first(self):
        body = "执行特征库升级操作步骤说明。" + "y" * 250
        rows = [_chunk("c.a", body), _chunk("c.b", body), _chunk("c.c", "独有" + "z" * 250)]
        out, rep_of = JsonlCorpus._dedupe_content_rows(rows)
        assert [r["id"] for r in out] == ["c.a", "c.c"]
        assert rep_of == {"c.b": "c.a"}

    def test_content_field_rows_unchanged_after_fix(self):
        # 修复不得破坏 content 字段既有行为
        body = "x" * 250
        rows = [_sec_long("s.a", body), _sec_long("s.b", body)]
        out, rep_of = JsonlCorpus._dedupe_content_rows(rows)
        assert [r["id"] for r in out] == ["s.a"] and rep_of == {"s.b": "s.a"}

    def test_neither_field_present_not_grouped(self):
        rows = [{"id": "n.1", "text": None}, {"id": "n.2", "text": None}]
        out, rep_of = JsonlCorpus._dedupe_content_rows(rows)
        assert len(out) == 2 and rep_of == {}

    def test_from_file_chunks_jsonl_two_level_dedupe(self, tmp_path):
        body = "w" * 250
        rows = [
            _chunk("c.dup", body),
            _chunk("c.dup", body),
            _chunk("c.twin", body),
            _chunk("c.solo", "z" * 250),
        ]
        f = tmp_path / "chunks.jsonl"
        f.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
            encoding="utf-8",
        )
        corpus = JsonlCorpus.from_file(f)
        assert [r["id"] for r in corpus.records] == ["c.dup", "c.solo"]
        assert corpus.representative_id("c.twin") == "c.dup"
        assert len(f.read_text(encoding="utf-8").strip().splitlines()) == 4


# ---------- D-0659-1 方案 B：去重键升级为 content|text + title + operation ----------

class TestDedupeKeyIncludesTitleOperation:
    """chunks 组内 metadata.title/operation 可不同（[0659] 实测 anquan 18 组 /
    duli 13 组），title/operation 参与 _searchable 打分 → 并组改变检索可
    发现性（跨功能同文组：创建私有镜像 vs 创建虚拟机等）。方案 B：仅
    「内容+标题+操作」都相同的行才并组；sections 组内差异仅 section_path
    （[0512] 逐字段核验），title 一致时行为不变。
    """

    def _chunk_titled(self, rid, text, title, operation):
        return {
            "id": rid,
            "manual_id": "hisec-anquan-v100r026",
            "type": "step",
            "text": text,
            "metadata": {"title": title, "operation": operation},
        }

    def test_same_text_diff_title_not_grouped(self):
        # [0659] 发现①核心场景：跨功能同文组（title 参与 _searchable 打分）
        body = "单击「确定」完成配置。" + "x" * 250
        rows = [
            self._chunk_titled("c.mirror", body, "创建私有镜像", "创建私有镜像"),
            self._chunk_titled("c.vm", body, "创建虚拟机", "创建虚拟机"),
        ]
        out, rep_of = JsonlCorpus._dedupe_content_rows(rows)
        assert [r["id"] for r in out] == ["c.mirror", "c.vm"]
        assert rep_of == {}

    def test_same_text_same_title_still_grouped(self):
        body = "执行升级操作。" + "y" * 250
        rows = [
            self._chunk_titled("c.a", body, "特征库升级", "特征库升级"),
            self._chunk_titled("c.b", body, "特征库升级", "特征库升级"),
        ]
        out, rep_of = JsonlCorpus._dedupe_content_rows(rows)
        assert [r["id"] for r in out] == ["c.a"] and rep_of == {"c.b": "c.a"}

    def test_same_text_diff_operation_not_grouped(self):
        body = "查看结果并确认。" + "z" * 250
        rows = [
            self._chunk_titled("c.a", body, "告警处理", "查看告警结果"),
            self._chunk_titled("c.b", body, "告警处理", "导出告警结果"),
        ]
        out, rep_of = JsonlCorpus._dedupe_content_rows(rows)
        assert [r["id"] for r in out] == ["c.a", "c.b"] and rep_of == {}

    def test_sections_top_level_title_path_unchanged(self):
        # sections 走顶层 title；[0512] 核验组内 title 一致 → 方案 B 行为不变
        body = "w" * 250
        rows = [_sec_long("s.a", body, title="创建病毒扫描任务"), _sec_long("s.b", body, title="创建病毒扫描任务")]
        out, rep_of = JsonlCorpus._dedupe_content_rows(rows)
        assert [r["id"] for r in out] == ["s.a"] and rep_of == {"s.b": "s.a"}

    def test_top_title_falls_back_to_metadata_title(self):
        # 取值口径与 _normalize 一致：顶层 title 缺失时回落 metadata.title
        body = "v" * 250
        rows = [
            self._chunk_titled("c.a", body, "同一标题", "同一操作"),
            self._chunk_titled("c.b", body, "不同标题", "同一操作"),
        ]
        # c.a 无顶层 title（真实 chunks schema）；c.b 同——回落路径必须生效
        out, rep_of = JsonlCorpus._dedupe_content_rows(rows)
        assert rep_of == {}

    def test_missing_title_both_rows_still_grouped(self):
        # 两条都无 title/operation（回归保护：键退化回纯内容时不互相并错）
        rows = [{"id": "c.a", "type": "step", "text": "u" * 250},
                {"id": "c.b", "type": "step", "text": "u" * 250}]
        out, rep_of = JsonlCorpus._dedupe_content_rows(rows)
        assert [r["id"] for r in out] == ["c.a"] and rep_of == {"c.b": "c.a"}
