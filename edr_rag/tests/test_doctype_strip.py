"""缺陷④回归：DOCTYPE 声明泄漏进正文与读取侧防御剥离。

背景：bs4 的 ``Doctype`` 是 ``NavigableString`` 子类，parse_html._walk 的
isinstance 判定漏过 → 两手册 sections 100%（anquan 1474/1474、duli
1788/1788）content 以 ``html PUBLIC "-//W3C//DTD ..."`` 开头；chunks/nodes
另有跨节拼接的中部泄漏。污染通道：_snippet 120 字两手册 100% 全垃圾、
_rerank_text（LLM 证据首选）均值 6.8% 被占用。

修复两层（[0652] 倾向）：
1. 根因层：parse_html._walk 排除 Doctype/Comment 子类（供未来重 ingest）；
2. 防御层：JsonlCorpus._normalize 剥离读取侧泄漏前缀（磁盘产物不改）。
"""
from __future__ import annotations

import json

import pytest

from edr_rag.p1.search import JsonlCorpus
from edr_rag.retrieval import RetrievalService

DOCTYPE_PREFIX = (
    'html\n  PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN" '
    '"http://www.w3.org/TR/html4/loose.dtd"'
)


def _sec(rid, content):
    return {
        "id": rid,
        "type": "section",
        "title": "配置病毒扫描任务",
        "content": content,
        "section_path": ["病毒防护", "配置病毒扫描任务"],
        "metadata": {"version": "V100R026"},
        "source_ref": {"relative_path": "a.html", "element_path": "div"},
    }


class TestParseHtmlExcludesDoctype:
    """根因层：Doctype/ProcessingInstruction/Comment 不得进入 content。"""

    @pytest.fixture(autouse=True)
    def _require_bs4(self):
        pytest.importorskip("bs4")

    def _parse(self, html: str):
        from pathlib import Path

        from edr_rag.parse_html import parse_html_file

        f = Path(__import__("tempfile").mkdtemp()) / "t.html"
        f.write_text(html, encoding="utf-8")
        return parse_html_file(f, f.parent)

    def test_doctype_not_in_content(self):
        page = self._parse(
            '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN" '
            '"http://www.w3.org/TR/html4/loose.dtd">'
            "<html><body><p>单击确定按钮。</p></body></html>"
        )
        assert "DOCTYPE" not in page.content
        assert "loose.dtd" not in page.content
        assert "单击确定按钮" in page.content

    def test_lowercase_doctype_not_in_content(self):
        page = self._parse(
            '<!doctype html><html><body><p>启动类型为立即启动。</p></body></html>'
        )
        assert "doctype" not in page.content.lower()

    def test_comment_not_in_content(self):
        page = self._parse(
            "<!-- build note --><html><body><p>配置完成后保存。</p></body></html>"
        )
        assert "build note" not in page.content

    def test_processing_instruction_not_in_content(self):
        page = self._parse(
            '<?xml version="1.0" encoding="gb2312"?><html><body><p>重启服务。</p>'
            "</body></html>"
        )
        assert "xml version" not in page.content


class TestReadSideDoctypeStrip:
    """防御层：_normalize 剥离泄漏前缀，磁盘产物不改。"""

    def test_prefix_stripped_from_content(self):
        c = JsonlCorpus([_sec("s.1", f"{DOCTYPE_PREFIX} 创建病毒扫描任务 前提条件")])
        assert "loose.dtd" not in c.records[0]["content"]
        assert c.records[0]["content"].startswith("创建病毒扫描任务")

    def test_prefix_stripped_from_text_field_chunks(self):
        r = _sec("s.1", None)
        del r["content"]
        r["type"] = "chunk"
        r["text"] = f"{DOCTYPE_PREFIX} 告警解释 对站点中设备进行升级"
        c = JsonlCorpus([r])
        assert "loose.dtd" not in c.records[0]["text"]
        assert c.records[0]["text"].startswith("告警解释")

    def test_mid_text_leak_also_stripped(self):
        # chunks 跨节拼接型：泄漏在中部（footer 之后跟下一节）
        c = JsonlCorpus(
            [
                _sec(
                    "s.1",
                    "版权所有 © 华为技术有限公司\n"
                    + DOCTYPE_PREFIX
                    + " 登录管理面 前提条件",
                )
            ]
        )
        assert "loose.dtd" not in c.records[0]["content"]
        assert "登录管理面" in c.records[0]["content"]

    def test_normal_content_untouched(self):
        c = JsonlCorpus([_sec("s.1", "单击立即启动。html PUBLIC 之后的内容不误删")])
        assert "html PUBLIC 之后的内容不误删" in c.records[0]["content"]

    def test_content_without_leak_unchanged(self):
        original = "正常正文，无任何泄漏。"
        c = JsonlCorpus([_sec("s.1", original)])
        assert c.records[0]["content"] == original

    def test_snippet_no_longer_all_garbage(self):
        # [0652] 量化：泄漏前 120 字 snippet 100% 全垃圾
        body = "配置病毒扫描任务。" * 20
        c = JsonlCorpus([_sec("s.1", f"{DOCTYPE_PREFIX} {body}")])
        assert "dtd" not in c._snippet(c.records[0]).lower()

    def test_rerank_text_no_longer_wasted_prefix(self):
        body = "配置病毒扫描任务。" * 200
        c = JsonlCorpus([_sec("s.1", f"{DOCTYPE_PREFIX} {body}")])
        assert c._rerank_text(c.records[0]).startswith("配置病毒扫描任务")

    def test_searchable_no_doctype_terms(self):
        # 嵌入通道：_searchable 不得含 doctype/loose.dtd 词形
        c = JsonlCorpus([_sec("s.1", f"{DOCTYPE_PREFIX} 配置病毒扫描任务")])
        assert "dtd" not in c.records[0]["_searchable"]

    def test_content_dedup_sees_stripped_content(self, tmp_path):
        # 去重键口径 = content|text；from_file 剥离必须发生在去重之前，
        # 否则「同正文仅前缀空白不同」的孪生行不并组。
        body = "正文" * 100
        rows = [
            _sec("s.1", DOCTYPE_PREFIX + " " + body),
            _sec("s.2", DOCTYPE_PREFIX + "  " + body),
        ]
        f = tmp_path / "sections.jsonl"
        f.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
            encoding="utf-8",
        )
        c = JsonlCorpus.from_file(f)
        assert c.representative_id("s.2") == "s.1"

    def test_from_file_strips_without_touching_disk(self, tmp_path):
        row = _sec("s.1", f"{DOCTYPE_PREFIX} 创建病毒扫描任务")
        f = tmp_path / "sections.jsonl"
        f.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
        before = f.read_bytes()
        c = JsonlCorpus.from_file(f)
        assert "loose.dtd" not in c.records[0]["content"]
        assert f.read_bytes() == before


class TestRecursiveStripAndLoadNodes:
    """缺陷④b（[0819] 覆盖缺口）：

    1. ``_strip_doctype_leak`` 只处理顶层 content/text —— chunks/nodes 的
       ``metadata.recovery`` 等嵌套字段同样泄漏（anquan chunks 276 行、
       nodes 304 行；经 ``get()`` / ``render_context_text`` 暴露）。
    2. ``RetrievalService._load_nodes`` 用 ``_read_jsonl`` 直接读
       nodes.jsonl（P1-4 上下文补全路径），完全绕过 JsonlCorpus 防御层
       ——anquan 1526 行 / duli 1842 行 text 泄漏渲染进 context_text。
    """

    @staticmethod
    def _proc_with_leak():
        return {
            "id": "procedure.登录管理面",
            "title": "登录管理面",
            "recovery": [
                f"{DOCTYPE_PREFIX} 连接失败请检查端口 22",
            ],
            "metadata": {
                "recovery": [f"{DOCTYPE_PREFIX} 连接失败请检查端口 22"],
                "title": "登录管理面",
            },
        }

    def test_nested_metadata_recovery_stripped(self):
        # chunks 行形状：泄漏在 metadata.recovery 列表元素内
        r = _sec("s.1", None)
        del r["content"]
        r["type"] = "chunk"
        r["text"] = "恢复步骤正文。"
        r["metadata"] = {
            "title": "恢复数据",
            "recovery": [f"{DOCTYPE_PREFIX} 恢复流程说明"],
        }
        c = JsonlCorpus([r])
        recov = c.records[0]["metadata"]["recovery"]
        assert "loose.dtd" not in recov[0]
        assert recov[0].startswith("恢复流程说明")

    def test_deeply_nested_and_normal_strings_untouched(self):
        r = _sec("s.1", None)
        del r["content"]
        r["type"] = "chunk"
        r["text"] = "正文。"
        r["metadata"] = {
            "preconditions": ["网络连接正常"],
            "recovery": [
                {"note": f"{DOCTYPE_PREFIX} 说明", "keep": "html PUBLIC 无DTD不误伤"},
                "纯文本无泄漏",
            ],
        }
        c = JsonlCorpus([r])
        meta = c.records[0]["metadata"]
        assert meta["preconditions"] == ["网络连接正常"]
        assert meta["recovery"][0]["note"] == " 说明" or not meta["recovery"][0]["note"].startswith("html")
        assert "loose.dtd" not in meta["recovery"][0]["note"]
        assert meta["recovery"][0]["keep"] == "html PUBLIC 无DTD不误伤"
        assert meta["recovery"][1] == "纯文本无泄漏"

    def test_load_nodes_strips_leak(self, tmp_path):
        # P1-4 上下文补全路径：_read_jsonl 直读 nodes.jsonl，须同口径剥离
        node = {
            "id": "node.section.sec-01",
            "level": "section",
            "text": f"{DOCTYPE_PREFIX} 配置远程登录会话 正文。",
            "metadata": {
                "title": "配置远程登录会话",
                "recovery": [f"{DOCTYPE_PREFIX} 恢复说明"],
            },
        }
        (tmp_path / "nodes.jsonl").write_text(
            json.dumps(node, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        svc = RetrievalService(tmp_path)
        loaded = svc._load_nodes()
        assert "loose.dtd" not in loaded[0]["text"]
        assert loaded[0]["text"].startswith("配置远程登录会话")
        assert "loose.dtd" not in loaded[0]["metadata"]["recovery"][0]

    def test_context_text_no_doctype_after_fix(self, tmp_path):
        from edr_rag.p1.context import assemble_context, render_context_text
        from edr_rag.p1.nodes import StructuredNode

        node = {
            "id": "node.section.sec-01",
            "level": "section",
            "text": f"{DOCTYPE_PREFIX} 配置远程登录会话 正文。",
            "metadata": {"title": "配置远程登录会话", "section_path": ["网络"]},
        }
        (tmp_path / "nodes.jsonl").write_text(
            json.dumps(node, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        svc = RetrievalService(tmp_path)
        loaded = svc._load_nodes()
        snodes = [StructuredNode.from_json(n) for n in loaded]
        ctx = assemble_context("node.section.sec-01", snodes)
        rendered = render_context_text(ctx)
        assert "loose.dtd" not in rendered
        assert "配置远程登录会话" in rendered
