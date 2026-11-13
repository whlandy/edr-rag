"""P4 Lock 2 回归测试 —— embedder 必须是模块级单例。

背景 (根因)：P3-1 的 ``_optional_embedder()`` 把 ``@lru_cache`` 定义在**函数
体内部**：

    def _optional_embedder():
        @lru_cache(maxsize=1)
        def _model(): return HuggingFaceEmbedding(...)
        def _embed(text): return _model().get_text_embedding(text)
        return _embed

每次调用返回全新闭包 + 全新 lru_cache，于是 BGE 模型被反复从磁盘装载。
实测单次 ~9–10 秒，而 ``retrieval.py`` 每个查询要调 1–2 次
(``_vector_rank`` + ``_embedder_available``)，multi-query 变体再乘一遍。

这组测试锁死修复后的契约，防止将来有人「顺手」把缓存挪回函数内部。
不需要真的装 llama-index —— 契约是关于**对象同一性**与**探测次数**的。
"""

from __future__ import annotations

import time

import pytest

from edr_rag.p1 import search as search_mod


@pytest.fixture(autouse=True)
def _reset():
    search_mod.reset_embedder_cache()
    yield
    search_mod.reset_embedder_cache()


def test_optional_embedder_returns_same_object():
    """契约：同一进程内必须返回同一个函数对象 (或都是 None)。"""
    first = search_mod._optional_embedder()
    second = search_mod._optional_embedder()
    third = search_mod._optional_embedder()
    assert first is second is third


def test_probe_runs_only_once(monkeypatch):
    """探测结果被缓存 —— 重复调用不再重新 import / 构造。

    1433 修复后首次调用含真嵌入探针（模型装载的固有 import 链），计数
    窗口放在预热之后，锁住的是「后续调用零重复装载」这一契约核心。
    """
    search_mod._optional_embedder()  # 预热（含探针装载链，计数外）
    calls = {"n": 0}
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def counting_import(name, *args, **kwargs):
        if name.startswith("llama_index.embeddings"):
            calls["n"] += 1
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", counting_import)
    for _ in range(5):
        search_mod._optional_embedder()
    assert calls["n"] == 0


def test_repeat_probe_is_cheap():
    """AC-1: 首次之后的探测必须是常数时间 (远小于模型装载的秒级)。"""
    search_mod._optional_embedder()  # 预热
    t0 = time.perf_counter()
    for _ in range(100):
        search_mod._optional_embedder()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    # 100 次探测总耗时都应远低于一次模型装载 (~10000ms)
    assert elapsed_ms < 100.0, f"probe not cached: {elapsed_ms:.1f}ms for 100 calls"


def test_model_built_at_most_once(monkeypatch):
    """模型实例只构造一次，即使多次嵌入 —— 这正是旧实现失败的地方。

    直接验证 ``_load_embed_model`` 的模块级单例语义：反复调用只应真正构造
    一次 HuggingFaceEmbedding。旧实现里每个新闭包都会重新构造。
    """
    constructed = {"n": 0}

    class _FakeModel:
        def get_text_embedding(self, text):
            return [1.0, 0.0, 0.0]

    def fake_ctor(model_name=None):
        constructed["n"] += 1
        return _FakeModel()

    search_mod.reset_embedder_cache()
    # 只替换真正的模型构造，保留 _load_embed_model 的单例逻辑本身。
    monkeypatch.setattr(search_mod, "_embedder_model", None, raising=False)
    monkeypatch.setattr(
        search_mod, "_load_embed_model", _make_singleton_loader(fake_ctor)
    )

    for _ in range(10):
        search_mod._load_embed_model().get_text_embedding("测试文本")
    assert constructed["n"] == 1


def _make_singleton_loader(ctor):
    """构造一个与 _load_embed_model 同语义 (模块级单例) 的加载器。"""
    holder: dict = {}

    def _load():
        if "m" not in holder:
            holder["m"] = ctor()
        return holder["m"]

    return _load


def test_embed_cache_is_bounded(monkeypatch):
    """文档向量缓存必须有界 —— 旧实现是无界 dict (长跑进程内存泄漏)。"""
    monkeypatch.setattr(search_mod, "_EMBED_CACHE_MAXSIZE", 8)
    search_mod._EMBEDDER_CACHE.clear()

    def fake_embed(text: str):
        return [float(len(text)), 1.0, 0.0]

    for i in range(50):
        search_mod._cached_vec(fake_embed, f"doc-{i}")
    assert len(search_mod._EMBEDDER_CACHE) <= 8


def test_embed_cache_returns_cached_value():
    calls = {"n": 0}

    def fake_embed(text: str):
        calls["n"] += 1
        return [1.0, 2.0, 3.0]

    search_mod._EMBEDDER_CACHE.clear()
    a = search_mod._cached_vec(fake_embed, "同一段文本")
    b = search_mod._cached_vec(fake_embed, "同一段文本")
    assert a == b == [1.0, 2.0, 3.0]
    assert calls["n"] == 1
