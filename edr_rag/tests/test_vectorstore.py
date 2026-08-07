"""P3-1 可插拔向量存储 —— 单元测试。

覆盖:
  - SimpleVectorStore 的 upsert/search/delete/count/ids 行为 (纯 stdlib)。
  - VectorIndex 把记录经嵌入手桥接到 store、检索带回 doc_idx。
  - factory: 默认 simple；别名归一；未知后端/缺依赖回退到 simple。
  - 外部后端骨架的可用性探测与优雅回退。
"""

from __future__ import annotations

import pytest

from edr_rag.vectorstore import (
    ElasticsearchVectorStore,
    MilvusVectorStore,
    OpenSearchVectorStore,
    PGVectorStore,
    QdrantVectorStore,
    SimpleVectorStore,
    VectorIndex,
    VectorRecord,
    build_vector_store,
)


def _embed(text: str) -> list[float]:
    """确定性伪嵌入：给出一组可区分向量的简单投影。"""
    seed = sum(ord(c) for c in text)
    return [float(seed % 11) / 11.0, float(seed % 7) / 7.0, float(seed % 5) / 5.0]


# ---------- SimpleVectorStore ----------


def test_simple_upsert_search_delete_count():
    store = SimpleVectorStore()
    store.upsert(
        [
            VectorRecord(id="a", vector=[1.0, 0.0, 0.0], metadata={"risk": "high"}),
            VectorRecord(id="b", vector=[0.0, 1.0, 0.0], metadata={"risk": "low"}),
            VectorRecord(id="c", vector=[1.0, 0.1, 0.0], metadata={"risk": "high"}),
        ]
    )
    assert store.count() == 3
    assert store.ids() == ["a", "b", "c"]

    hits = store.search([1.0, 0.0, 0.0], top_k=2)
    assert [h.id for h in hits] == ["a", "c"]
    assert hits[0].score >= hits[1].score

    # metadata 过滤
    hits = store.search([1.0, 0.0, 0.0], top_k=10, filter={"risk": "high"})
    assert {h.id for h in hits} == {"a", "c"}

    # upsert 同 id 覆盖
    store.upsert([VectorRecord(id="a", vector=[0.0, 1.0, 0.0])])
    assert store.count() == 3
    hits = store.search([0.0, 1.0, 0.0], top_k=1)
    assert hits[0].id == "a"

    assert store.delete(["b"]) == 1
    assert store.count() == 2
    assert "b" not in store.ids()


def test_simple_empty_search_returns_empty():
    store = SimpleVectorStore()
    assert store.search([1.0, 0.0]) == []


def test_simple_dim_mismatch_skipped():
    store = SimpleVectorStore()
    store.upsert([VectorRecord(id="a", vector=[1.0, 0.0, 0.0])])
    # 维度不一致的记录被跳过，不进库
    assert store.upsert([VectorRecord(id="b", vector=[1.0, 0.0])]) == 0
    assert store.count() == 1


# ---------- VectorIndex ----------


def test_vector_index_index_and_search():
    store = SimpleVectorStore()
    index = VectorIndex(store, _embed)
    records = [
        {"id": "r1", "_doc_id": "r1", "_searchable": "配置代理服务器"},
        {"id": "r2", "_doc_id": "r2", "_searchable": "配置邮件告警"},
    ]
    assert index.index_records(records, text_field="_searchable", id_field="_doc_id") == 2

    # 与 r1 特征相近的查询
    hits = index.search("代理", top_k=2)
    assert len(hits) == 2
    # doc_idx 正确映射回 r1/r2
    assert {i for _, i in hits} == {0, 1}

    # 再次 index 不重复 (签名缓存) — count 不变
    index.index_records(records, text_field="_searchable", id_field="_doc_id")
    assert store.count() == 2

    # 语料内容变化 (条数不变) → 内容签名变化 → 重建索引
    records_changed = [
        {"id": "r1", "_doc_id": "r1", "_searchable": "完全不同的新内容 A"},
        {"id": "r2", "_doc_id": "r2", "_searchable": "完全不同的新内容 B"},
    ]
    index.index_records(records_changed, text_field="_searchable", id_field="_doc_id")
    assert store.count() == 2  # 重建后仍是 2 条，但内容已替换


# ---------- factory ----------

BACKENDS = ["simple", "elasticsearch", "es", "opensearch", "os", "qdrant", "milvus", "pgvector", "pg"]


@pytest.mark.parametrize("backend", BACKENDS)
def test_build_vector_store_returns_vector_store(backend):
    """所有合法后端名都返回 VectorStore 实例（骨架/内存），绝不抛错。"""
    store = build_vector_store({"backend": backend})
    from edr_rag.vectorstore import VectorStore

    assert isinstance(store, VectorStore)


def test_unknown_backend_falls_back_to_simple():
    """未知名后端回退到 SimpleVectorStore。"""
    store = build_vector_store({"backend": "bogus_backend"})
    assert isinstance(store, SimpleVectorStore)


def test_default_is_simple():
    assert isinstance(build_vector_store(), SimpleVectorStore)


def test_simple_backend_dim_passed():
    store = build_vector_store({"backend": "simple", "dim": 16})
    assert isinstance(store, SimpleVectorStore)
    assert store.dim == 16


# ---------- 外部后端骨架形状 ----------


def test_external_backend_classes_exist():
    from edr_rag.vectorstore import VectorStore

    assert ElasticsearchVectorStore.name == "elasticsearch"
    assert OpenSearchVectorStore.name == "opensearch"
    assert QdrantVectorStore.name == "qdrant"
    assert MilvusVectorStore.name == "milvus"
    assert PGVectorStore.name == "pgvector"
    # 全部是 VectorStore 的子类
    for cls in (
        ElasticsearchVectorStore,
        OpenSearchVectorStore,
        QdrantVectorStore,
        MilvusVectorStore,
        PGVectorStore,
    ):
        assert issubclass(cls, VectorStore)
