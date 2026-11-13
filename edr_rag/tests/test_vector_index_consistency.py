"""Regression checks for record identity, mappings and failed rebuilds."""

import json

import pytest

from edr_rag.retrieval import RetrievalService
from edr_rag.security import ACLPolicy
from edr_rag.vectorstore import PersistentVectorStore, SimpleVectorStore, VectorIndex


def embed(text):
    return [1.0, float(len(text))]


def row(rid, text="same", **extra):
    return {"id": rid, "_searchable": text, **extra}


def hits(store):
    return {h.id: h for h in store.search([1.0, 1.0], top_k=20)}


def test_full_index_refreshes_identity_and_acl_without_text_change(tmp_path):
    store = PersistentVectorStore(directory=tmp_path, namespace="test")
    index = VectorIndex(store, embed)
    index.index_records([row("old", tenant="a")])
    index.persist()
    reopened = PersistentVectorStore(directory=tmp_path, namespace="test")
    index = VectorIndex(reopened, embed)
    index.index_records([row("new", tenant="b")])
    assert reopened.ids() == ["new"]
    assert hits(reopened)["new"].metadata["tenant"] == "b"


def test_full_signature_preserves_text_boundaries():
    store = SimpleVectorStore()
    index = VectorIndex(store, embed)
    index.index_records([row("a", "ab"), row("b", "c")])
    index.index_records([row("a", "a"), row("b", "bc")])
    assert hits(store)["a"].text == "a"


def test_incremental_updates_second_duplicate_without_overwriting_first():
    store = SimpleVectorStore()
    index = VectorIndex(store, embed)
    _, known = index.index_records_incremental([row("a", "first"), row("a", "second")])
    index.index_records_incremental([row("a", "first"), row("a", "changed")], known)
    assert hits(store)["a"].text == "first"
    assert hits(store)["a#dup2"].text == "changed"


def test_incremental_remaps_unchanged_text_after_deletion():
    store = SimpleVectorStore()
    index = VectorIndex(store, embed)
    _, known = index.index_records_incremental([row("a"), row("b")])
    index.index_records_incremental([row("b")], known)
    assert hits(store)["b"].metadata["doc_idx"] == 0


def test_incremental_refreshes_acl_without_text_change():
    store = SimpleVectorStore()
    index = VectorIndex(store, embed)
    _, known = index.index_records_incremental([row("a", tenant="old")])
    index.index_records_incremental([row("a", tenant="new")], known)
    assert hits(store)["a"].metadata["tenant"] == "new"


def test_metadata_cannot_override_internal_row_mapping():
    store = SimpleVectorStore()
    index = VectorIndex(store, embed)
    index.index_records([row("a", metadata={"doc_idx": 99, "id": "forged"})])
    assert hits(store)["a"].metadata["doc_idx"] == 0
    assert hits(store)["a"].metadata["id"] == "a"


def test_duplicate_suffix_does_not_overwrite_real_id():
    store = SimpleVectorStore()
    index = VectorIndex(store, embed)
    index.index_records([row("a"), row("a"), row("a#dup2")])
    assert store.count() == 3


def test_failed_embedding_does_not_destroy_previous_index():
    store = SimpleVectorStore()
    index = VectorIndex(store, embed)
    index.index_records([row("old")])

    def fail(text):
        raise RuntimeError("model unavailable")

    index.embed = fail
    with pytest.raises(RuntimeError):
        index.index_records([row("new", "changed")])
    assert store.ids() == ["old"]


def test_incremental_empty_text_removes_old_vector():
    store = SimpleVectorStore()
    index = VectorIndex(store, embed)
    _, known = index.index_records_incremental([row("a")])
    index.index_records_incremental([row("a", "")], known)
    assert store.count() == 0


def test_incremental_no_change_does_not_embed_again():
    store = SimpleVectorStore()
    index = VectorIndex(store, embed)
    rows = [row("a"), row("a"), row("a#dup2")]
    _, known = index.index_records_incremental(rows)
    index.embed = lambda text: pytest.fail("unchanged records must not embed")
    result, _ = index.index_records_incremental(rows, known)
    assert len(result.unchanged) == 3
    assert store.count() == 3


def test_content_dedup_preserves_authorized_copy(tmp_path):
    rows = [
        {"id": "a", "title": "配置", "content": "配置说明" * 100, "tenant": "a"},
        {"id": "b", "title": "配置", "content": "配置说明" * 100, "tenant": "b"},
    ]
    (tmp_path / "sections.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8"
    )
    results = RetrievalService(tmp_path).search(
        "配置", corpus_type="sections", strategy="bm25", acl=ACLPolicy(tenants={"b"})
    )
    assert [r["id"] for r in results] == ["b"]


def test_content_dedup_preserves_version_metadata(tmp_path):
    from edr_rag.p1.search import JsonlCorpus

    rows = [
        {"id": "a", "title": "配置", "content": "配置说明" * 100, "metadata": {"version": "1"}},
        {"id": "b", "title": "配置", "content": "配置说明" * 100, "metadata": {"version": "2"}},
    ]
    path = tmp_path / "sections.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    assert len(JsonlCorpus.from_file(path).records) == 2


def test_duplicate_result_keeps_original_ndcg_rank():
    import math
    from edr_rag.eval.metrics import ndcg_at_k

    assert ndcg_at_k(["noise", "noise", "gold"], {"gold"}, 3) == pytest.approx(1 / math.log2(4))
