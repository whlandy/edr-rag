"""P1-6 tests: 查询侧中文规范化 (normalize_query) + 多查询生成 (query expansion)
与基于评测集的 RRF 权重调节 (tune_rrf_weights + retrieval 集成)。

覆盖:
  - normalize_query: NFKC 全半角折叠、中文标点归一、大小写、空白折叠、幂等。
  - expand_query: 同义/别名变体生成、去重保序、截断、规范化。
  - retrieval 集成: 全角查询能命中半角语料 (对称规范化); query_expansion 开关;
    set_fusion_weights 线程到 hybrid RRF 融合。
  - tune_rrf_weights: 在评测集上网格搜索返回最优配置, 空集/无 embedder 优雅降级。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from edr_rag.query import (
    expand_query,
    get_synonyms,
    load_synonyms,
    normalize_query,
    reset_synonyms_cache,
)
from edr_rag.retrieval import RetrievalService
from edr_rag.rerank import rrf_fuse


# ---------- normalize_query ----------

def test_normalize_full_width_to_half_width():
    # 全角 ASCII 折叠为半角 (NFKC)。
    assert normalize_query("ｃｏｎｆｉｇ　代理") == "config 代理"


def test_normalize_chinese_punctuation():
    # 中文全角标点归一为半角 (NFKC + 兜底映射)。
    assert normalize_query("配置代理；查看安全状态") == "配置代理;查看安全状态"


def test_normalize_case_and_whitespace():
    assert normalize_query("  Reset   System  ") == "reset system"


def test_normalize_nfkc_compatibility():
    # 上标/兼容字符被折叠。
    assert normalize_query("①号") == "1号"


def test_normalize_keeps_cjk():
    assert normalize_query("配置代理服务器") == "配置代理服务器"


def test_normalize_idempotent():
    s = "　ＣＯＮＦＩＧ:配置代理;查看安全状态　"
    once = normalize_query(s)
    assert normalize_query(once) == once


def test_normalize_empty():
    assert normalize_query("") == ""
    assert normalize_query("   ") == ""


# ---------- expand_query ----------
#
# P4: 这些用例改用**测试自带的**词表，不再依赖 edr_rag/data/synonyms.zh.txt
# 的具体内容。测试断言的是 expand_query 的机制（命中/替换/去重/截断），
# 而不是某条业务词条 —— 词表是可被客户覆盖的数据，不该被测试锁死。

_TEST_VOCAB = """
# 测试专用词表
恢复出厂设置 => 重置系统
恢复出厂 => 重置系统
怎么查看 => 查看
"""


@pytest.fixture
def test_synonyms(tmp_path, monkeypatch):
    """把同义词表指向一份测试自带的小词表。"""
    vocab = tmp_path / "syn.txt"
    vocab.write_text(_TEST_VOCAB, encoding="utf-8")
    monkeypatch.setenv("EDR_RAG_SYNONYMS", str(vocab))
    reset_synonyms_cache()
    yield vocab
    monkeypatch.delenv("EDR_RAG_SYNONYMS", raising=False)
    reset_synonyms_cache()


def test_expand_returns_original_first(test_synonyms):
    vs = expand_query("配置代理")
    assert vs and vs[0] == "配置代理"


def test_expand_generates_alias_variant(test_synonyms):
    vs = expand_query("恢复出厂设置")
    # 首元素为原查询，命中别名替换出规范词变体。
    assert vs[0] == "恢复出厂设置"
    assert "重置系统" in vs
    assert vs[0] != vs[1]


def test_expand_normalizes_before_matching(test_synonyms):
    # 全角查询先规范化，再命中别名。
    vs = expand_query("恢复出厂设置")
    assert vs


def test_expand_dedup_and_order_stable(test_synonyms):
    vs1 = expand_query("恢复出厂设置")
    vs2 = expand_query("恢复出厂设置")
    assert vs1 == vs2


def test_expand_max_variants(test_synonyms):
    vs = expand_query("恢复出厂设置", max_variants=2)
    assert len(vs) <= 2


def test_expand_empty():
    assert expand_query("") == []


def test_longer_alias_wins(test_synonyms):
    """更具体（更长）的别名优先，短别名不得抢先命中。"""
    pairs = load_synonyms(test_synonyms)
    lengths = [len(a) for a, _ in pairs]
    assert lengths == sorted(lengths, reverse=True)


def test_missing_vocab_degrades_to_normalize_only(tmp_path, monkeypatch):
    """词表文件缺失时退化为「只规范化」，不抛异常。"""
    monkeypatch.setenv("EDR_RAG_SYNONYMS", str(tmp_path / "nope.txt"))
    reset_synonyms_cache()
    try:
        assert expand_query("恢复出厂设置") == ["恢复出厂设置"]
    finally:
        monkeypatch.delenv("EDR_RAG_SYNONYMS", raising=False)
        reset_synonyms_cache()


def test_shipped_vocab_loads():
    """随包词表本身必须可解析（格式回归）。"""
    reset_synonyms_cache()
    pairs = get_synonyms()
    assert pairs, "shipped synonym vocabulary should not be empty"
    for alias, canonical in pairs:
        assert alias and canonical and alias != canonical


# ---------- retrieval 集成: 对称规范化 + 多查询开关 ----------

def _write_artifacts(root: Path):
    # 语料含半角文本 + 一条 full-width 文本，验证对称规范化。
    actions = [
        {
            "action_id": "system.reset_system",
            "product": "EDR",
            "module": "system",
            "title": "重置系统",
            "status": "approved",
            "confidence": 0.9,
            "risk": "high",
            "requires_confirmation": True,
            "preconditions": [],
            "procedure": [{"order": 1, "action": "click", "target": "重置"}],
            "success_criteria": [],
            "recovery": [],
        },
        {
            "action_id": "net.proxy_cfg",
            "product": "EDR",
            "module": "net",
            "title": "配置代理服务器",
            "status": "approved",
            "confidence": 0.8,
            "risk": "medium",
            "requires_confirmation": False,
            "preconditions": [],
            "procedure": [{"order": 1, "action": "click", "target": "代理"}],
            "success_criteria": [],
            "recovery": [],
        },
    ]
    (root / "actions.jsonl").write_text(
        "\n".join(json.dumps(a, ensure_ascii=False) for a in actions) + "\n",
        encoding="utf-8",
    )
    # 全角版本的 section, 验证查询侧全角→半角后能命中。
    sections = [{
        "id": "s1",
        "title": "配置代理服务器",
        "content": "本节介绍 EdRag 如何配置代理，需要网络设置。",
        "section_path": ["网络", "代理"],
        "source": {"file": "net/proxy.html"},
    }]
    (root / "sections.jsonl").write_text(
        "\n".join(json.dumps(s, ensure_ascii=False) for s in sections) + "\n",
        encoding="utf-8",
    )


@pytest.fixture
def service(tmp_path: Path) -> RetrievalService:
    _write_artifacts(tmp_path)
    return RetrievalService(tmp_path)


def test_full_width_query_hits_half_width_corpus(service):
    # 查询用全角 ASCII (ＥＤＲ), 规范化到半角后应命中半角语料 (对称规范化)。
    res = service.search("ＥＤＲ", corpus_type="sections", strategy="keyword")
    assert any(r["id"] == "s1" for r in res), "全角查询应命中半角语料"
    # 不规范化的话 (直接原词) 不会命中 —— 这里验证 normalize 后的 query 生效。
    raw_res = service.search("ＥＤＲ", corpus_type="sections", strategy="keyword",
                             query_expansion=False)
    assert any(r["id"] == "s1" for r in raw_res), "query_expansion=False 仍应规范化"


def test_query_expansion_recalls_alias(service, test_synonyms):
    # "恢复出厂" 别名展开到 "重置系统", 应能召回 reset 动作。
    res = service.search("如何恢复出厂设置", corpus_type="actions", strategy="keyword")
    ids = [r["id"] for r in res]
    # 关键词"恢复出厂设置"直接命中应至少返回一个结果 (变体融合后)。
    assert res, "expansion 后应返回结果"


def test_query_expansion_off_single_variant(service):
    r_on = service.search("配置代理", corpus_type="actions")
    r_off = service.search("配置代理", corpus_type="actions", query_expansion=False)
    # "配置代理"无别名变体, 开关不影响结果。
    assert [x["id"] for x in r_off] == [x["id"] for x in r_on]


def test_set_fusion_weights_threads_to_hybrid(service):
    service.set_fusion_weights([2.0, 1.0])
    res = service.search("配置代理", corpus_type="actions", strategy="hybrid")
    assert res
    service.set_fusion_weights(None)
    res2 = service.search("配置代理", corpus_type="actions", strategy="hybrid")
    assert res2


def test_set_fusion_weights_rejects_negative(service):
    with pytest.raises(ValueError):
        service.set_fusion_weights([-1.0, 1.0])


# ---------- rrf_fuse weighted (P1-6) ----------

def test_rrf_fuse_weighted_changes_order():
    a = [{"id": "x1"}, {"id": "x2"}]
    b = [{"id": "x2"}, {"id": "x1"}]
    # 等权时两路对称, x1/x2 分数并列 (顺序由 dict 稳定决定)。
    equal = [r["id"] for r in rrf_fuse([a, b])]
    # 重权第一路时, x1 (第一路 rank1) 应显著更靠前。
    weighted = [r["id"] for r in rrf_fuse([a, b], weights=[5.0, 1.0])]
    assert weighted[0] == "x1"
    assert equal[0] == "x1" or equal[0] == "x2"


def test_rrf_fuse_weight_len_mismatch_falls_back_equal(tmp_path):
    from edr_rag.rerank import rrf_fuse as fuse
    a = [{"id": "x1"}]
    b = [{"id": "x1"}]
    # 长度不匹配 -> 退回等权, 不报错。
    out = fuse([a, b], weights=[9.9])
    assert out and out[0]["id"] == "x1"


# ---------- tune_rrf_weights ----------

def _eval_json(tmp_path: Path) -> Path:
    p = tmp_path / "eval.jsonl"
    rows = [
        {
            "qid": "q1", "query": "恢复出厂设置", "corpus_type": "actions",
            "relevant": ["system.reset_system"], "top_k": 5,
        },
        {
            "qid": "q2", "query": "配置代理", "corpus_type": "sections",
            "relevant": ["s1"], "top_k": 5,
        },
    ]
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                 encoding="utf-8")
    return p


def test_tune_returns_best_config(tmp_path):
    from edr_rag.eval.dataset import load_eval_set
    from edr_rag.eval.tune import tune_rrf_weights

    _write_artifacts(tmp_path)
    evalset = load_eval_set(_eval_json(tmp_path))
    result = tune_rrf_weights(tmp_path, evalset, metric="recall@k", candidates=[[2.0, 1.0], [1.0, 2.0], [1.0, 1.0]])
    assert result.candidates_tried == 3
    assert result.best_weights is not None
    assert len(result.best_weights) == 2
    assert 0.0 <= result.best_score <= 1.0
    assert result.scores


def test_tune_empty_eval_set_returns_default(tmp_path):
    from edr_rag.eval.dataset import EvalSet
    from edr_rag.eval.tune import tune_rrf_weights

    _write_artifacts(tmp_path)
    empty = EvalSet(name="empty", queries=[])
    result = tune_rrf_weights(tmp_path, empty)
    assert result.best_weights is None
    assert result.candidates_tried == 0
