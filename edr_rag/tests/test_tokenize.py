"""P4 Lock 4 —— 中文分词 (unigram + bigram) 的行为契约。

背景：P1 的 ``tokenize`` 把 CJK 逐字拆成 unigram，词序信息完全丢失 ——
「安全」与「全安」的 token 集合完全相同因而 BM25 同分。这是中文手册检索
精度的地基缺陷，且 keyword / bm25 / hybrid 三条策略共用这个地基。
"""

from __future__ import annotations

import pytest

from edr_rag.p1.search import JsonlCorpus, tokenize


# ---------- 核心契约 ----------


def test_word_order_is_distinguishable():
    """AC-5: 同字异序必须产出不同的 token 集合（旧实现做不到）。"""
    assert set(tokenize("安全")) != set(tokenize("全安"))
    assert set(tokenize("网络代理")) != set(tokenize("代理网络"))


def test_unigrams_are_retained_for_recall():
    """bigram 是增量，不是替换 —— 单字召回必须保留。"""
    toks = tokenize("配置代理")
    for ch in "配置代理":
        assert ch in toks


def test_bigrams_generated_within_cjk_run():
    toks = tokenize("配置代理")
    assert "配置" in toks
    assert "置代" in toks
    assert "代理" in toks


def test_bigram_does_not_cross_punctuation():
    """「代理。配置」不得拼出跨标点的无意义 bigram「理配」。"""
    toks = tokenize("代理。配置")
    assert "代理" in toks
    assert "配置" in toks
    assert "理配" not in toks


def test_bigram_does_not_cross_latin():
    toks = tokenize("代理proxy配置")
    assert "理p" not in toks
    assert "y配" not in toks
    assert "proxy" in toks


def test_latin_words_kept_whole():
    toks = tokenize("Proxy Server 2024")
    assert "proxy" in toks       # normalize_query 会做大小写折叠
    assert "server" in toks
    assert "2024" in toks


def test_empty_and_punctuation_only():
    assert tokenize("") == []
    assert tokenize("。，！") == []


def test_fullwidth_query_matches_halfwidth_corpus():
    """P1-6 的规范化对称性不能被 P4 的分词改动破坏。"""
    assert tokenize("ＰＲＯＸＹ") == tokenize("proxy")


def test_tokenize_is_deterministic():
    a = tokenize("配置代理服务器并查看安全状态")
    b = tokenize("配置代理服务器并查看安全状态")
    assert a == b


def test_term_frequency_preserved():
    """BM25 依赖词频，重复词必须保留重复 token。"""
    toks = tokenize("代理 代理")
    assert toks.count("代理") == 2


# ---------- 端到端：bigram 真的改善了排序 ----------


def test_bigram_improves_ranking_over_unigram_collision():
    """构造一个 unigram 会误判、bigram 能救回的场景。

    查询「代理配置」：
      - doc-a 讲的正是代理配置
      - doc-b 只是恰好同时出现了这四个字的其它搭配（配电、理货）
    纯 unigram 下两者的字集合高度重合；bigram 让 doc-a 拿到「代理」「配置」
    两个高区分度 term。
    """
    corpus = JsonlCorpus([
        {"id": "doc-a", "title": "代理配置", "text": "代理配置说明：如何配置代理。"},
        {"id": "doc-b", "title": "配电与理货", "text": "配电箱理货流程，代号与理由。"},
    ])
    hits = corpus.bm25_search("代理配置", limit=2)
    assert hits, "expected bm25 hits"
    assert hits[0]["id"] == "doc-a"


def test_bigram_toggle_env(monkeypatch):
    """EDR_RAG_CJK_BIGRAM=0 可回到纯 unigram（对照实验用）。"""
    import importlib

    import edr_rag.p1.search as search_mod

    monkeypatch.setenv("EDR_RAG_CJK_BIGRAM", "0")
    reloaded = importlib.reload(search_mod)
    try:
        assert set(reloaded.tokenize("安全")) == set(reloaded.tokenize("全安"))
    finally:
        monkeypatch.delenv("EDR_RAG_CJK_BIGRAM", raising=False)
        importlib.reload(search_mod)


def test_jieba_absent_is_graceful():
    """未装 jieba 时不得抛错，bigram 路径照常工作。"""
    from edr_rag.p1.search import _optional_jieba

    cut = _optional_jieba()
    assert cut is None or callable(cut)
    assert tokenize("配置代理服务器")  # 不抛异常即可
