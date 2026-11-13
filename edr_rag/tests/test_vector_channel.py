"""向量通道可观测性与回退契约（20261112-1433 事故修复）。

事故：embedder import 可用但模型加载失败时，`vector_search` 静默降级 BM25，
hybrid 侧将降级结果当"向量路"加入融合（keyword+BM25+BM25），且无任何
归档可观测痕迹；正式索引被查询自动重建写入。

契约修复：
1. `_optional_embedder` 不再只测 import——首次使用时真嵌入一个探针文本，
   失败即判**不可用**（进程内缓存），状态可经 `embedder_status()` 读取。
2. `vector_search` 对"探测可用但运行中失败"返回空列表（hybrid 侧不把
   任何 BM25 结果伪装成向量路）；单策略 vector 调用方（公开契约）仍降级 BM25。
3. `JsonlCorpus.from_file` 尊重 `EDR_RAG_VECTOR_DIR`：设置时所有语料的
   向量持久化目录重定向到该目录（评测隔离索引用），不落正式 workdir。
"""

from __future__ import annotations

import pytest


@pytest.fixture()
def _broken_embedder(monkeypatch):
    """import 探测通过、模型加载失败的 embedder（模拟事故现场）。"""
    import edr_rag.p1.search as ps

    def _boom(text: str) -> list[float]:
        raise RuntimeError("simulated model load failure")

    _boom.model_id = "hf:simulated-broken"
    monkeypatch.setattr(ps, "_embedder_checked", True)
    monkeypatch.setattr(ps, "_embedder_fn", _boom)
    # probe 缓存同样重置，让探测真实运行
    monkeypatch.setattr(ps, "_probe_done", False)
    monkeypatch.setattr(ps, "_probe_ok", None)
    yield


@pytest.fixture()
def _tiny_corpus(tmp_path):
    from edr_rag.p1.search import JsonlCorpus

    rows = [
        {"id": "s1", "content": "业务面日志在哪里查看 管理日志", "title": "查看日志"},
        {"id": "s2", "content": "终端登记管理 绑定关系", "title": "终端登记"},
        {"id": "s3", "content": "防火墙策略上限 一百条", "title": "防火墙"},
    ]
    path = tmp_path / "sections.jsonl"
    path.write_text(
        "\n".join(__import__("json").dumps(r, ensure_ascii=False) for r in rows),
        encoding="utf-8",
    )
    return JsonlCorpus.from_file(path)


def test_probe_marks_broken_embedder_unavailable(_broken_embedder):
    import edr_rag.p1.search as ps

    assert ps.embedder_status() == "failed"
    # 探测失败后 _optional_embedder 必须返回 None（不可用），而非原函数
    assert ps._optional_embedder() is None


def test_probe_ok_embedder_reports_active(monkeypatch):
    import edr_rag.p1.search as ps

    def _ok(text: str) -> list[float]:
        return [0.1, 0.2]

    _ok.model_id = "hf:simulated-ok"
    monkeypatch.setattr(ps, "_embedder_checked", True)
    monkeypatch.setattr(ps, "_embedder_fn", _ok)
    monkeypatch.setattr(ps, "_probe_done", False)
    monkeypatch.setattr(ps, "_probe_ok", None)
    assert ps.embedder_status() == "active"
    assert ps._optional_embedder() is _ok


def test_probe_absent_without_deps(monkeypatch):
    import edr_rag.p1.search as ps

    monkeypatch.setattr(ps, "_embedder_checked", True)
    monkeypatch.setattr(ps, "_embedder_fn", None)
    monkeypatch.setattr(ps, "_probe_done", False)
    monkeypatch.setattr(ps, "_probe_ok", None)
    assert ps.embedder_status() == "absent"


def test_vector_search_broken_model_returns_empty_in_hybrid_mode(_broken_embedder, _tiny_corpus):
    """hybrid 语义（fallback='empty'）下运行中失败返回 []，不伪装成向量路。"""
    assert _tiny_corpus.vector_search("日志", limit=5, fallback="empty") == []


def test_vector_search_broken_model_still_degrades_for_direct_calls(_broken_embedder, _tiny_corpus):
    """单策略 vector 公开契约不变：直接调用仍降级 BM25。"""
    res = _tiny_corpus.vector_search("日志", limit=5)
    assert res, "单策略调用应降级 BM25 而非返回空"
    assert all(r.get("id") for r in res)


def test_hybrid_no_duplicate_bm25_when_model_broken(_broken_embedder, _tiny_corpus):
    """模型失败时 hybrid 融合列表不得出现 keyword+BM25+BM25。

    验证方式：对比 hybrid 与"纯 keyword+BM25 两路"的融合——若降级 BM25
    被当作向量路重复计入，两路/三路的 RRF 排序会不同（已在事故中实证）。
    """
    import edr_rag.rerank as rr

    q = "业务面日志在哪里查看"
    kw = _tiny_corpus.keyword_search(q, limit=20)
    bm = _tiny_corpus.bm25_search(q, limit=20)
    expect = [r["id"] for r in rr.rrf_fuse([kw, bm])]

    from edr_rag.retrieval import RetrievalService
    svc = RetrievalService(str(_tiny_corpus.vector_dir.parent if _tiny_corpus.vector_dir else "."))
    # 直接驱动 _search_generic_hybrid，绕开文件加载（用注入 corpus）
    got = svc._search_generic_hybrid(
        _tiny_corpus, q, filters={}, top_k=5, rerank_enabled=False
    )
    assert [r["id"] for r in got] == expect


def test_from_file_respects_vector_dir_env(monkeypatch, tmp_path):
    """EDR_RAG_VECTOR_DIR 设置时，from_file 的语料向量目录重定向到它。"""
    import json

    import edr_rag.p1.search as ps

    override = tmp_path / "isolated-index"
    monkeypatch.setenv("EDR_RAG_VECTOR_DIR", str(override))
    src = tmp_path / "sections.jsonl"
    src.write_text(
        json.dumps({"id": "s1", "content": "x"}, ensure_ascii=False), encoding="utf-8"
    )
    c = ps.JsonlCorpus.from_file(src)
    assert c.vector_dir == override


def test_from_file_default_vector_dir_unchanged(monkeypatch, tmp_path):
    """未设置环境变量时行为不变：<workdir>/vector_index。"""
    import json

    import edr_rag.p1.search as ps

    monkeypatch.delenv("EDR_RAG_VECTOR_DIR", raising=False)
    src = tmp_path / "sections.jsonl"
    src.write_text(
        json.dumps({"id": "s1", "content": "x"}, ensure_ascii=False), encoding="utf-8"
    )
    c = ps.JsonlCorpus.from_file(src)
    assert c.vector_dir == tmp_path / "vector_index"
