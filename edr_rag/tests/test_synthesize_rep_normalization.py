"""synthesize 层接入内容级去重映射的回归（[0512] 下一任务①，[0440] 建议二）。

背景：内容级去重（JsonlCorpus.from_file 读取侧）把「不同 ID 同 content」
的行收敛为首现代表，被移除 ID 仍被 procedures/chunks 的 section_id 引用。
答案引用渲染必须输出「代表 ID + 同内容组成员列表」，不静默替换——
引用能按 ID 定位回语料现存行（设计 §9），原 ID 可追溯（原始来源不可变）。

消费契约：映射只能经 ``RetrievalService.representative_map()`` / 直接透传
``rep_of`` 获取，synthesize 层不自行重建（[0440] 结论三：三处共用一份映射）。
"""
from __future__ import annotations

import json

import pytest

from edr_rag.p2.synthesize import synthesize_answer


def _action(sid="manual.section.配置.创建工作组", **extra):
    a = {
        "id": "action.创建工作组",
        "title": "创建工作组",
        "risk": "medium",
        "confidence": 0.9,
        "source": {"file": "cfg.html", "section_id": sid},
        "procedure": [{"order": 1, "action": "点击", "target": "新建", "source_instruction": "点击「新建」"}],
        "preconditions": ["已登录"],
    }
    a.update(extra)
    return a


class TestResolveSourceRepNormalization:
    def test_removed_section_id_normalized_with_declaration(self):
        from edr_rag.p2.synthesize import _resolve_source

        rep_of = {"manual.section.配置.二级系统配置.创建工作组": "manual.section.配置.业务面配置系统管理员.创建工作组"}
        ref = _resolve_source(
            _action("manual.section.配置.二级系统配置.创建工作组"), rep_of=rep_of
        )
        assert ref["section_id"] == "manual.section.配置.业务面配置系统管理员.创建工作组"
        assert ref["same_content_as"] == ["manual.section.配置.二级系统配置.创建工作组"]

    def test_live_section_id_untouched_no_declaration(self):
        from edr_rag.p2.synthesize import _resolve_source

        ref = _resolve_source(_action(), rep_of={"s.other": "s.rep"})
        assert ref["section_id"] == "manual.section.配置.创建工作组"
        assert "same_content_as" not in ref

    def test_no_map_no_behavior_change(self):
        from edr_rag.p2.synthesize import _resolve_source

        ref = _resolve_source(
            _action("manual.section.配置.二级系统配置.创建工作组")
        )
        assert ref["section_id"] == "manual.section.配置.二级系统配置.创建工作组"
        assert "same_content_as" not in ref


class TestCitationsRepNormalization:
    def test_removed_hit_id_normalized_with_declaration(self):
        ctx = {"actions": [], "procedures": [], "sections": [
            {"id": "manual.section.配置.二级系统配置.创建工作组", "title": "创建工作组", "score": 0.9},
        ]}
        rep_of = {"manual.section.配置.二级系统配置.创建工作组": "manual.section.配置.业务面配置系统管理员.创建工作组"}
        ans = synthesize_answer("怎么创建工作组", ctx, rep_of=rep_of)
        cites = [c for c in ans.citations if c["kind"] == "section"]
        assert len(cites) == 1
        assert cites[0]["id"] == "manual.section.配置.业务面配置系统管理员.创建工作组"
        assert cites[0]["same_content_as"] == ["manual.section.配置.二级系统配置.创建工作组"]

    def test_representative_and_twin_collapse_to_one_citation(self):
        # 代表与孪生同时命中：归一后同一代表 ID，合并为一条引用 + 声明列表。
        rep = "manual.section.配置.业务面配置系统管理员.创建工作组"
        ctx = {"actions": [], "procedures": [], "sections": [
            {"id": rep, "title": "创建工作组", "score": 0.9},
            {"id": "manual.section.配置.二级系统配置.创建工作组", "title": "创建工作组", "score": 0.85},
        ]}
        rep_of = {"manual.section.配置.二级系统配置.创建工作组": rep}
        ans = synthesize_answer("怎么创建工作组", ctx, rep_of=rep_of)
        cites = [c for c in ans.citations if c["kind"] == "section"]
        assert len(cites) == 1 and cites[0]["id"] == rep
        assert cites[0]["same_content_as"] == ["manual.section.配置.二级系统配置.创建工作组"]

    def test_no_map_citations_unchanged(self):
        ctx = {"actions": [], "procedures": [], "sections": [
            {"id": "s.gone", "title": "t", "score": 0.9},
        ]}
        ans = synthesize_answer("q", ctx)
        assert ans.citations[0]["id"] == "s.gone"
        assert "same_content_as" not in ans.citations[0]


class TestStepsWarningsRepNormalization:
    def test_step_source_normalized(self):
        ctx = {"actions": [_action("s.gone")]}
        ans = synthesize_answer("q", ctx, rep_of={"s.gone": "s.keep"})
        assert ans.steps and ans.steps[0]["source"]["section_id"] == "s.keep"
        assert ans.steps[0]["source"]["same_content_as"] == ["s.gone"]


class TestRetrievalServiceRepresentativeMap:
    def test_map_shared_with_corpus(self, tmp_path):
        from edr_rag.retrieval import RetrievalService

        body = "x" * 250
        rows = [
            {"id": "s.keep", "title": "t", "content": body},
            {"id": "s.gone", "title": "t", "content": body},
        ]
        f = tmp_path / "sections.jsonl"
        f.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
            encoding="utf-8",
        )
        svc = RetrievalService(tmp_path)
        assert svc.representative_map("sections") == {"s.gone": "s.keep"}

    def test_missing_file_returns_empty(self, tmp_path):
        from edr_rag.retrieval import RetrievalService

        assert RetrievalService(tmp_path).representative_map("sections") == {}
