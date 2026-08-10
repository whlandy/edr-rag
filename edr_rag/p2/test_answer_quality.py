"""P4 Phase E —— 答案质量三项修复的行为契约。

对应成熟度评估里的严重问题 5 / 7 / 9：
  5. 置信度与检索质量完全脱钩（只看抽取阶段写死的静态字段）
  7. rerank 因个别空文本放弃整批，且只拿 120 字 snippet 打分
  9. 冲突检测只有 5 组硬编码反义词 + 标题字符串相等
"""

from __future__ import annotations

import pytest

from edr_rag import rerank as rerank_mod
from edr_rag.p2.synthesize import (
    CONFIDENCE_REFUSE_THRESHOLD,
    _compute_confidence,
    _detect_conflicts,
    synthesize_answer,
)


def _action(**over):
    base = {
        "id": "network.configure_proxy",
        "title": "配置代理服务器",
        "risk": "low",
        "status": "ready",
        "confidence": 0.85,
        "score": 9.0,
        "procedure": [
            {"order": 1, "action": "click", "target": "设置",
             "source_instruction": "打开设置"},
        ],
    }
    base.update(over)
    return base


# ---------- 严重问题 5：置信度 ----------


def test_confidence_responds_to_retrieval_decisiveness():
    """检索越决定性，置信度越高 —— 旧实现对此完全无感。"""
    decisive = {"actions": [_action(score=9.0), _action(id="x", score=0.4)]}
    ambiguous = {"actions": [_action(score=9.0), _action(id="x", score=8.9)]}
    assert _compute_confidence(decisive) > _compute_confidence(ambiguous)


def test_confidence_ignores_retrieval_when_no_scores():
    """没有分数信号时不惩罚 —— 「没有该信号」不等于「该信号很差」。"""
    with_scores = {"actions": [_action(score=9.0)]}
    without = {"actions": [_action(score=None)]}
    # 两者都应落在合理区间，不因缺分数而崩到 0
    assert _compute_confidence(without) > 0.5
    assert _compute_confidence(with_scores) > 0.5


def test_confidence_zero_for_empty_context():
    assert _compute_confidence({"actions": [], "procedures": []}) == 0.0


def test_confidence_rises_with_evidence_count():
    one = {"actions": [_action()]}
    three = {"actions": [_action(id=f"a{i}") for i in range(3)]}
    assert _compute_confidence(three) >= _compute_confidence(one)


def test_extraction_confidence_preserved_for_audit():
    """旧口径保留在 extraction_confidence 里，便于对照与审计。"""
    result = synthesize_answer("如何配置代理", {"actions": [_action(confidence=0.85)]})
    assert result.extraction_confidence == 0.85
    assert result.confidence != result.extraction_confidence


def test_low_extraction_still_refuses():
    """抽取置信度很低时仍然拒答 —— 修复不能把拒答能力弄没。"""
    result = synthesize_answer("如何配置代理", {"actions": [_action(confidence=0.15)]})
    assert result.answer_status == "refused"
    assert result.refusal_reason == "low_confidence"


def test_confidence_bounded():
    for ctx in (
        {"actions": [_action(confidence=1.0, score=1000.0)]},
        {"actions": [_action(confidence=0.0, score=0.0)]},
    ):
        assert 0.0 <= _compute_confidence(ctx) <= 1.0


def test_threshold_unchanged_semantics():
    """阈值常量仍在，语义仍是「低于则拒答」。"""
    assert 0.0 < CONFIDENCE_REFUSE_THRESHOLD < 1.0


# ---------- 严重问题 7：rerank ----------


class _StubReranker:
    """确定性假 reranker：文本里含 '相关' 的得高分。"""

    def __init__(self):
        self.seen: list[str] = []

    def __call__(self, query: str, texts: list[str]) -> list[float]:
        self.seen = list(texts)
        return [10.0 if "相关" in t else 1.0 for t in texts]


@pytest.fixture
def stub_reranker(monkeypatch):
    stub = _StubReranker()
    monkeypatch.setattr(rerank_mod, "_optional_reranker", lambda: stub)
    return stub


def test_single_empty_text_no_longer_kills_whole_batch(stub_reranker):
    """旧实现：候选池里有一条空文本 -> 整批放弃精排，reranker 形同虚设。"""
    results = [
        {"id": "a", "title": "", "snippet": ""},          # 无文本
        {"id": "b", "title": "很相关的文档", "snippet": "正文"},
        {"id": "c", "title": "无关", "snippet": "正文"},
    ]
    out, did = rerank_mod.rerank("查询", results, enabled=True)
    assert did is True, "个别空文本不应让整批精排被跳过"
    assert out[0]["id"] == "b"


def test_unscorable_items_go_to_the_end(stub_reranker):
    results = [
        {"id": "empty", "title": "", "snippet": ""},
        {"id": "hit", "title": "很相关", "snippet": "x"},
    ]
    out, did = rerank_mod.rerank("查询", results, enabled=True)
    assert did is True
    assert [r["id"] for r in out] == ["hit", "empty"]


def test_all_empty_still_skips(stub_reranker):
    results = [{"id": "a", "title": "", "snippet": ""}]
    out, did = rerank_mod.rerank("查询", results, enabled=True)
    assert did is False
    assert out == results


def test_rerank_uses_full_text_not_snippet(stub_reranker):
    """精排必须看完整正文，而不是 120 字截断的 snippet。"""
    long_body = "前置说明。" * 60 + "关键的相关内容在很后面。"
    results = [{
        "id": "a",
        "title": "标题",
        "snippet": "前置说明。" * 5,      # 120 字摘要里没有 '相关'
        "_rerank_text": long_body,          # 完整正文里有
    }]
    rerank_mod.rerank("查询", results, enabled=True)
    assert any("相关" in t for t in stub_reranker.seen), (
        "reranker 只看到了 snippet，没拿到完整正文"
    )


def test_rerank_text_is_capped(stub_reranker):
    results = [{"id": "a", "title": "T", "_rerank_text": "字" * 99999}]
    rerank_mod.rerank("查询", results, enabled=True)
    assert len(stub_reranker.seen[0]) <= rerank_mod.RERANK_TEXT_MAX


def test_rerank_text_prefers_title_first(stub_reranker):
    results = [{"id": "a", "title": "标题在前", "_rerank_text": "正文在后"}]
    rerank_mod.rerank("查询", results, enabled=True)
    assert stub_reranker.seen[0].startswith("标题在前")


def test_reranker_score_count_mismatch_is_safe(monkeypatch):
    """reranker 返回条数不对时安全跳过，而不是错位排序。"""
    monkeypatch.setattr(
        rerank_mod, "_optional_reranker", lambda: (lambda q, texts: [1.0])
    )
    results = [{"id": "a", "title": "x"}, {"id": "b", "title": "y"}]
    out, did = rerank_mod.rerank("q", results, enabled=True)
    assert did is False and out == results


def test_rerank_toggle_is_runtime_readable(monkeypatch):
    """开关改为运行时读取 —— 旧实现在 import 时求值一次，改不了。"""
    monkeypatch.setenv("EDR_RAG_RERANK", "0")
    assert rerank_mod.rerank_enabled() is False
    monkeypatch.setenv("EDR_RAG_RERANK", "1")
    assert rerank_mod.rerank_enabled() is True


def test_internal_rerank_text_not_leaked_to_callers(tmp_path):
    """_rerank_text 是内部字段，不得出现在对外响应里。"""
    import json

    from edr_rag.retrieval import RetrievalService

    (tmp_path / "sections.jsonl").write_text(
        json.dumps({"id": "s1", "title": "代理配置", "text": "正文" * 200},
                   ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    results = RetrievalService(tmp_path).search("代理", corpus_type="sections")
    assert results
    for r in results:
        assert not any(k.startswith("_") for k in r), f"内部字段泄漏: {list(r)}"


# ---------- 严重问题 9：冲突检测 ----------


def test_version_divergence_detected():
    """手册里最常见的真实冲突：同一操作在不同版本下步骤不同。"""
    items = [
        _action(title="重置系统", version="V5.0"),
        _action(id="b", title="重置系统", version="V6.0"),
    ]
    conflicts = _detect_conflicts({"actions": items, "procedures": []})
    kinds = {c["type"] for c in conflicts}
    assert "version_divergence" in kinds
    desc = next(c for c in conflicts if c["type"] == "version_divergence")["description"]
    assert "V5.0" in desc and "V6.0" in desc


def test_version_from_metadata():
    items = [
        _action(title="重置系统", metadata={"applies_to": "Windows"}),
        _action(id="b", title="重置系统", metadata={"applies_to": "Linux"}),
    ]
    conflicts = _detect_conflicts({"actions": items, "procedures": []})
    assert any(c["type"] == "version_divergence" for c in conflicts)


def test_same_version_is_not_a_conflict():
    items = [
        _action(title="重置系统", version="V6.0"),
        _action(id="b", title="重置系统", version="V6.0"),
    ]
    conflicts = _detect_conflicts({"actions": items, "procedures": []})
    assert not any(c["type"] == "version_divergence" for c in conflicts)


def test_step_count_divergence_detected():
    short = _action(title="重置系统", procedure=[
        {"order": i, "action": "click", "target": f"t{i}"} for i in range(1, 3)
    ])
    long = _action(id="b", title="重置系统", procedure=[
        {"order": i, "action": "click", "target": f"t{i}"} for i in range(1, 10)
    ])
    conflicts = _detect_conflicts({"actions": [short, long], "procedures": []})
    assert any(c["type"] == "step_count_divergence" for c in conflicts)


def test_small_step_count_difference_is_not_a_conflict():
    """多一步确认不该被误报成冲突。"""
    a = _action(title="重置系统", procedure=[
        {"order": i, "action": "click", "target": f"t{i}"} for i in range(1, 4)
    ])
    b = _action(id="b", title="重置系统", procedure=[
        {"order": i, "action": "click", "target": f"t{i}"} for i in range(1, 5)
    ])
    conflicts = _detect_conflicts({"actions": [a, b], "procedures": []})
    assert not any(c["type"] == "step_count_divergence" for c in conflicts)


@pytest.mark.parametrize("op_a,op_b", [
    ("添加", "删除"),
    ("连接", "断开"),
    ("授权", "撤销"),
    ("锁定", "解锁"),
    ("上线", "下线"),
    ("grant", "revoke"),
])
def test_expanded_opposite_operations(op_a, op_b):
    """P4 扩充的互斥操作对被识别。"""
    items = [
        _action(procedure=[{"order": 1, "action": op_a, "target": "终端策略"}]),
        _action(id="b", procedure=[{"order": 1, "action": op_b, "target": "终端策略"}]),
    ]
    conflicts = _detect_conflicts({"actions": items, "procedures": []})
    assert any(c["type"] == "opposite_operations" for c in conflicts), (
        f"未识别互斥操作 {op_a} / {op_b}"
    )


def test_conflict_sets_requires_confirmation():
    items = [
        _action(title="重置系统", version="V5.0"),
        _action(id="b", title="重置系统", version="V6.0"),
    ]
    result = synthesize_answer("怎么重置", {"actions": items})
    assert result.answer_status == "conflict"
    assert result.requires_confirmation is True


def test_no_false_conflict_on_single_item():
    conflicts = _detect_conflicts({"actions": [_action()], "procedures": []})
    assert conflicts == []
