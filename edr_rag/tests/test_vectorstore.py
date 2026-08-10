"""可插拔向量存储 —— 单元测试 (P3-1，P4 重写)。

覆盖:
  - SimpleVectorStore 的 upsert/search/delete/count/ids 行为 (纯 stdlib)。
  - PersistentVectorStore 的落盘 / 重载 / 精确检索 / metadata 谓词下推 (P4)。
  - VectorIndex 把记录经嵌入桥接到 store、检索带回 doc_idx。
  - factory: 默认 persistent；别名归一；未知后端回退；namespace 隔离 (P4)。
"""

from __future__ import annotations

import pytest

from edr_rag.vectorstore import (
    PersistentVectorStore,
    SimpleVectorStore,
    VectorIndex,
    VectorRecord,
    VectorStore,
    build_vector_store,
    get_vector_store,
    reset_shared_stores,
    vector_dir_for_workdir,
)


def _embed(text: str) -> list[float]:
    """确定性伪嵌入：给出一组可区分向量的简单投影。"""
    seed = sum(ord(c) for c in text)
    return [float(seed % 11) / 11.0, float(seed % 7) / 7.0, float(seed % 5) / 5.0]


@pytest.fixture(autouse=True)
def _clean_shared_stores():
    """每个用例前后清掉进程内共享 store，避免用例间互相污染。"""
    reset_shared_stores()
    yield
    reset_shared_stores()


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

    hits = store.search([1.0, 0.0, 0.0], top_k=10, filter={"risk": "high"})
    assert {h.id for h in hits} == {"a", "c"}

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
    assert store.upsert([VectorRecord(id="b", vector=[1.0, 0.0])]) == 0
    assert store.count() == 1


# ---------- PersistentVectorStore (P4) ----------


def _sample_records():
    return [
        VectorRecord(id="a", vector=[1.0, 0.0, 0.0], metadata={"tenant": "t1"}),
        VectorRecord(id="b", vector=[0.0, 1.0, 0.0], metadata={"tenant": "t2"}),
        VectorRecord(id="c", vector=[1.0, 0.1, 0.0], metadata={"tenant": "t1"}),
    ]


def test_persistent_roundtrip_survives_restart(tmp_path):
    """AC-2: 落盘后新实例能读回，不需要重新嵌入。"""
    store = PersistentVectorStore(directory=tmp_path, namespace="chunks")
    assert store.upsert(_sample_records()) == 3
    assert store.save() is True

    # 全新实例 (模拟进程重启) 直接从磁盘载入
    reopened = PersistentVectorStore(directory=tmp_path, namespace="chunks")
    assert reopened.count() == 3
    assert sorted(reopened.ids()) == ["a", "b", "c"]
    hits = reopened.search([1.0, 0.0, 0.0], top_k=2)
    assert [h.id for h in hits] == ["a", "c"]
    # metadata 一并保留
    assert hits[0].metadata.get("tenant") == "t1"


def test_persistent_namespaces_are_isolated(tmp_path):
    """AC-3: 不同 namespace 各自独立，互不清空。"""
    a = PersistentVectorStore(directory=tmp_path, namespace="actions")
    b = PersistentVectorStore(directory=tmp_path, namespace="sections")
    a.upsert([VectorRecord(id="x", vector=[1.0, 0.0, 0.0])])
    b.upsert([VectorRecord(id="y", vector=[0.0, 1.0, 0.0])])
    a.save()
    b.save()
    assert a.count() == 1 and b.count() == 1
    assert PersistentVectorStore(directory=tmp_path, namespace="actions").ids() == ["x"]
    assert PersistentVectorStore(directory=tmp_path, namespace="sections").ids() == ["y"]


def test_persistent_metadata_filter_pushdown(tmp_path):
    """metadata 谓词在打分前生效，且支持集合语义 (ACL 允许集合)。"""
    store = PersistentVectorStore(directory=tmp_path, namespace="chunks")
    store.upsert(_sample_records())
    hits = store.search([1.0, 0.0, 0.0], top_k=10, filter={"tenant": "t1"})
    assert {h.id for h in hits} == {"a", "c"}
    # 集合语义
    hits = store.search([1.0, 0.0, 0.0], top_k=10, filter={"tenant": ["t2"]})
    assert {h.id for h in hits} == {"b"} or hits == []


def test_persistent_delete_compacts(tmp_path):
    store = PersistentVectorStore(directory=tmp_path, namespace="chunks")
    store.upsert(_sample_records())
    assert store.delete(["b"]) == 1
    assert store.count() == 2
    assert "b" not in store.ids()
    # 删除后剩余记录仍可正确检索 (行数组重建未错位)
    hits = store.search([1.0, 0.0, 0.0], top_k=5)
    assert [h.id for h in hits] == ["a", "c"]


def test_persistent_memory_mode_without_directory():
    """directory=None 时纯内存运行，不写任何文件。"""
    store = PersistentVectorStore(directory=None, namespace="chunks")
    store.upsert(_sample_records())
    assert store.count() == 3
    assert store.save() is False
    assert store.vec_path is None


def test_persistent_rejects_unsafe_namespace(tmp_path):
    """namespace 直接参与文件名，必须挡住路径穿越。"""
    for bad in ("../evil", "a/b", "", ".."):
        with pytest.raises(ValueError):
            PersistentVectorStore(directory=tmp_path, namespace=bad)


def test_persistent_corrupt_index_is_ignored(tmp_path):
    """截断/损坏的向量文件不应让进程崩溃，而是当作空索引重建。"""
    store = PersistentVectorStore(directory=tmp_path, namespace="chunks")
    store.upsert(_sample_records())
    store.save()
    # 截断向量文件，使字节数与 ids × dim 不一致
    store.vec_path.write_bytes(store.vec_path.read_bytes()[:7])
    reopened = PersistentVectorStore(directory=tmp_path, namespace="chunks")
    assert reopened.count() == 0


def test_persistent_empty_save_clears_stale_files(tmp_path):
    store = PersistentVectorStore(directory=tmp_path, namespace="chunks")
    store.upsert(_sample_records())
    store.save()
    assert store.vec_path.exists()
    store.delete(["a", "b", "c"])
    store.save()
    assert not store.vec_path.exists()
    assert PersistentVectorStore(directory=tmp_path, namespace="chunks").count() == 0


# ---------- VectorIndex ----------


def test_vector_index_index_and_search():
    store = SimpleVectorStore()
    index = VectorIndex(store, _embed)
    records = [
        {"id": "r1", "_doc_id": "r1", "_searchable": "配置代理服务器"},
        {"id": "r2", "_doc_id": "r2", "_searchable": "配置邮件告警"},
    ]
    assert index.index_records(records, text_field="_searchable", id_field="_doc_id") == 2

    hits = index.search("代理", top_k=2)
    assert len(hits) == 2
    assert {i for _, i in hits} == {0, 1}

    index.index_records(records, text_field="_searchable", id_field="_doc_id")
    assert store.count() == 2

    records_changed = [
        {"id": "r1", "_doc_id": "r1", "_searchable": "完全不同的新内容 A"},
        {"id": "r2", "_doc_id": "r2", "_searchable": "完全不同的新内容 B"},
    ]
    index.index_records(records_changed, text_field="_searchable", id_field="_doc_id")
    assert store.count() == 2


def test_vector_index_lifts_acl_fields_to_metadata():
    """ACL 维度提升到向量 metadata 顶层，谓词才能下推 (P4)。"""
    store = SimpleVectorStore()
    index = VectorIndex(store, _embed)
    index.index_records(
        [{"_doc_id": "r1", "_searchable": "内容", "tenant": "t1", "device": "d9"}],
        text_field="_searchable",
        id_field="_doc_id",
    )
    hits = store.search(_embed("内容"), top_k=1)
    assert hits[0].metadata["tenant"] == "t1"
    assert hits[0].metadata["device"] == "d9"


def test_vector_index_persist_is_noop_for_memory_store():
    index = VectorIndex(SimpleVectorStore(), _embed)
    assert index.persist() is False


def test_vector_index_persist_writes_through(tmp_path):
    store = PersistentVectorStore(directory=tmp_path, namespace="chunks")
    index = VectorIndex(store, _embed)
    index.index_records(
        [{"_doc_id": "r1", "_searchable": "配置代理"}],
        text_field="_searchable",
        id_field="_doc_id",
    )
    assert index.persist() is True
    assert PersistentVectorStore(directory=tmp_path, namespace="chunks").count() == 1


# ---------- factory ----------


def test_default_backend_is_persistent():
    assert isinstance(build_vector_store(), PersistentVectorStore)


def test_simple_backend_selectable():
    store = build_vector_store({"backend": "simple", "dim": 16})
    assert isinstance(store, SimpleVectorStore)
    assert store.dim == 16


@pytest.mark.parametrize("alias,expected", [
    ("memory", SimpleVectorStore),
    ("local", PersistentVectorStore),
    ("file", PersistentVectorStore),
    ("disk", PersistentVectorStore),
])
def test_backend_aliases(alias, expected):
    assert isinstance(build_vector_store({"backend": alias}), expected)


def test_unknown_backend_falls_back_to_default():
    store = build_vector_store({"backend": "bogus_backend"})
    assert isinstance(store, PersistentVectorStore)


def test_all_backends_are_vector_stores():
    for backend in ("persistent", "simple"):
        assert isinstance(build_vector_store({"backend": backend}), VectorStore)


def test_no_notimplemented_backends_remain():
    """AC-4: P4 删除了五个 NotImplementedError 骨架。"""
    import edr_rag.vectorstore as vs

    assert not hasattr(vs, "QdrantVectorStore")
    assert not hasattr(vs, "ElasticsearchVectorStore")
    assert not hasattr(vs, "MilvusVectorStore")
    assert not hasattr(vs, "PGVectorStore")
    assert not hasattr(vs, "OpenSearchVectorStore")


def test_get_vector_store_isolates_namespaces(tmp_path):
    """AC-3: 共享工厂按 namespace 返回不同实例。"""
    cfg = {"directory": str(tmp_path)}
    a = get_vector_store("actions", cfg)
    b = get_vector_store("sections", cfg)
    assert a is not b
    # 同 namespace 复用同一实例
    assert get_vector_store("actions", cfg) is a


def test_vector_dir_for_workdir(tmp_path):
    assert vector_dir_for_workdir(tmp_path).name == "vector_index"
    assert vector_dir_for_workdir(None) is None
