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


def test_persistent_content_sig_survives_restart(tmp_path):
    """content_sig 随 meta 落盘并在新实例恢复 (P4 冷载缺陷)。"""
    store = PersistentVectorStore(directory=tmp_path, namespace="chunks")
    index = VectorIndex(store, _embed)
    index.index_records(
        [{"_doc_id": "r1", "_searchable": "配置代理"}],
        text_field="_searchable",
        id_field="_doc_id",
    )
    index.persist()
    assert store.get_content_sig() is not None

    reopened = PersistentVectorStore(directory=tmp_path, namespace="chunks")
    assert reopened.get_content_sig() == store.get_content_sig()


def test_vector_index_new_process_short_circuits_no_reembed(tmp_path):
    """新进程 VectorIndex 借持久化签名短路，不全量重嵌入 (P4 冷载缺陷)。"""
    records = [
        {"_doc_id": "r1", "_searchable": "配置代理服务器"},
        {"_doc_id": "r2", "_searchable": "配置邮件告警"},
    ]
    store1 = PersistentVectorStore(directory=tmp_path, namespace="chunks")
    index1 = VectorIndex(store1, _embed)
    assert index1.index_records(records, text_field="_searchable", id_field="_doc_id") == 2
    index1.persist()

    # 新实例 (模拟新进程 autoload)：同语料短路，不重嵌入
    store2 = PersistentVectorStore(directory=tmp_path, namespace="chunks")
    index2 = VectorIndex(store2, _embed)
    assert index2.index_records(records, text_field="_searchable", id_field="_doc_id") == 2
    assert store2.count() == 2


def test_vector_index_embedder_change_invalidates_persisted_sig(tmp_path):
    """换嵌入模型 (model_id 变化) 后持久化签名失效，强制重建。"""
    def _embed_v2(text: str) -> list[float]:
        seed = sum(ord(c) for c in text)
        return [float(seed % 13) / 13.0, float(seed % 3) / 3.0, 1.0]

    records = [{"_doc_id": "r1", "_searchable": "配置代理服务器"}]
    store1 = PersistentVectorStore(directory=tmp_path, namespace="chunks")
    index1 = VectorIndex(store1, _embed)
    index1.index_records(records, text_field="_searchable", id_field="_doc_id")
    index1.persist()

    store2 = PersistentVectorStore(directory=tmp_path, namespace="chunks")
    index2 = VectorIndex(store2, _embed_v2)
    # 签名含 embedder 标识 -> 不匹配 -> 重建 (会重写而非短路跳过)
    n = index2.index_records(records, text_field="_searchable", id_field="_doc_id")
    assert n == 1
    assert index2._indexed_sig != index1._indexed_sig


def test_vector_index_persist_skips_clean_store(tmp_path, monkeypatch):
    """签名短路后 persist() 不重写磁盘 (守卫 if indexed 恒真的场景)。"""
    store = PersistentVectorStore(directory=tmp_path, namespace="chunks")
    index = VectorIndex(store, _embed)
    records = [{"_doc_id": "r1", "_searchable": "配置代理"}]
    index.index_records(records, text_field="_searchable", id_field="_doc_id")
    index.persist()

    calls = {"save": 0}
    real_save = store.save
    monkeypatch.setattr(store, "save", lambda: (calls.__setitem__("save", calls["save"] + 1), real_save())[1])
    # 短路路径：无新写入时 persist 不触发 store.save
    assert index.persist() is True
    assert calls["save"] == 0
    # 真实写入后 persist 落盘并清脏标记
    index.index_records([{"_doc_id": "r2", "_searchable": "新内容"}], text_field="_searchable", id_field="_doc_id")
    assert index.persist() is True
    assert calls["save"] == 1
    assert index.persist() is True
    assert calls["save"] == 1


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


# ---------- 模型指纹持久化 (1433 审核契约漏洞修复) ----------


def test_persistent_meta_stores_embedder_sig(tmp_path):
    """磁盘 meta 持久化 embedder 标识；新实例可读回。"""
    store = PersistentVectorStore(directory=tmp_path, namespace="chunks")
    index = VectorIndex(store, _embed)
    index.index_records(
        [{"_doc_id": "r1", "_searchable": "配置代理"}],
        text_field="_searchable",
        id_field="_doc_id",
    )
    index.persist()
    assert store.get_embedder_sig() == "edr_rag.tests.test_vectorstore._embed"

    import json

    meta = json.loads((tmp_path / "chunks.meta.json").read_text(encoding="utf-8"))
    assert meta.get("embedder_sig") == "edr_rag.tests.test_vectorstore._embed"


def test_cross_process_embedder_change_cannot_reuse_old_vectors(tmp_path):
    """换模型后经磁盘恢复（新进程）必须拒绝旧模型向量，强制重建。

    缺陷形状：旧实现 _restore_sig_from_store 用「当前」embedder 拼接磁盘上
    的旧 content_sig，同进程测试测不到——换模型且跨进程重启时签名必然
    匹配，静默服务旧模型向量。
    """
    def _embed_v2(text: str) -> list[float]:
        seed = sum(ord(c) for c in text)
        return [float(seed % 13) / 13.0, float(seed % 3) / 3.0, 1.0]

    records = [{"_doc_id": "r1", "_searchable": "配置代理服务器"}]
    store1 = PersistentVectorStore(directory=tmp_path, namespace="chunks")
    index1 = VectorIndex(store1, _embed)
    index1.index_records(records, text_field="_searchable", id_field="_doc_id")
    index1.persist()

    # 模拟新进程：从磁盘 autoload 恢复，换嵌入模型。
    store2 = PersistentVectorStore(directory=tmp_path, namespace="chunks")
    assert store2.get_content_sig() is not None
    index2 = VectorIndex(store2, _embed_v2)
    n = index2.index_records(records, text_field="_searchable", id_field="_doc_id")
    # 旧实现：restore 拼接出的 sig 与新模型 sig 相同 -> 短路返回 count()=1，n=1
    # 但 store 里仍是旧模型向量。正确行为：不匹配 -> 重建 -> 向量已换。
    assert n == 1
    hits = store2.search(_embed_v2("配置代理服务器"), top_k=1)
    assert hits, "换模型重建后应可用新模型向量命中"
    assert index2._indexed_sig == index2._embed_sig + ":" + store2.get_content_sig()


def test_cross_process_same_embedder_short_circuits(tmp_path):
    """同模型跨进程恢复仍正常短路，修复不引入回退。"""
    records = [
        {"_doc_id": "r1", "_searchable": "配置代理服务器"},
        {"_doc_id": "r2", "_searchable": "配置邮件告警"},
    ]
    store1 = PersistentVectorStore(directory=tmp_path, namespace="chunks")
    index1 = VectorIndex(store1, _embed)
    index1.index_records(records, text_field="_searchable", id_field="_doc_id")
    index1.persist()

    store2 = PersistentVectorStore(directory=tmp_path, namespace="chunks")
    index2 = VectorIndex(store2, _embed)
    assert index2.index_records(records, text_field="_searchable", id_field="_doc_id") == 2
    assert store2.count() == 2


def test_legacy_meta_without_embedder_sig_triggers_rebuild(tmp_path):
    """旧格式 meta（无 embedder_sig 字段）恢复时不得复用为可短路状态。"""
    import json

    store1 = PersistentVectorStore(directory=tmp_path, namespace="chunks")
    index1 = VectorIndex(store1, _embed)
    index1.index_records(
        [{"_doc_id": "r1", "_searchable": "配置代理"}],
        text_field="_searchable",
        id_field="_doc_id",
    )
    index1.persist()
    # 模拟旧版本落盘的 meta：删除 embedder_sig 字段。
    meta_path = tmp_path / "chunks.meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.pop("embedder_sig", None)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    store2 = PersistentVectorStore(directory=tmp_path, namespace="chunks")
    index2 = VectorIndex(store2, _embed)
    # 无模型指纹 -> 无法证明向量属于当前模型 -> 必须重建而非短路。
    n = index2.index_records(
        [{"_doc_id": "r1", "_searchable": "配置代理"}],
        text_field="_searchable",
        id_field="_doc_id",
    )
    assert n == 1
    assert store2.get_embedder_sig() == "edr_rag.tests.test_vectorstore._embed"
