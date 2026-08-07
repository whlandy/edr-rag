"""P3-5 增量索引 —— 单元测试。

覆盖:
  - VectorIndex.index_records_incremental 的增量判定 (added/updated/removed/unchanged)。
  - 未变记录不重嵌入、变化记录重嵌入、消失记录被删除。
  - 返回的 new_known 可作为下一轮增量输入，可持续收敛到全量一致。
  - 整批签名短路 (index_records) 与增量后的状态保持一致。
"""

from __future__ import annotations

from edr_rag.vectorstore import IncrementalIndexResult, SimpleVectorStore, VectorIndex


def _embed(text: str) -> list[float]:
    seed = sum(ord(c) for c in text)
    return [float(seed % 11) / 11.0, float(seed % 7) / 7.0, float(seed % 5) / 5.0]


def _rec(rid: str, text: str) -> dict:
    return {"id": rid, "_doc_id": rid, "_searchable": text}


def test_incremental_first_run_all_added():
    store = SimpleVectorStore()
    index = VectorIndex(store, _embed)
    records = [_rec("a", "配置代理服务器"), _rec("b", "配置邮件告警")]
    result, new_known = index.index_records_incremental(
        records, known=None, text_field="_searchable", id_field="_doc_id"
    )
    assert result.added == ["a", "b"]
    assert result.updated == []
    assert result.removed == []
    assert result.unchanged == []
    assert result.total_changed == 2
    assert store.count() == 2
    assert set(new_known) == {"a", "b"}


def test_incremental_no_change_skips_embedding():
    store = SimpleVectorStore()
    index = VectorIndex(store, _embed)
    v1 = [_rec("a", "内容甲"), _rec("b", "内容乙")]
    _, known = index.index_records_incremental(
        v1, known=None, text_field="_searchable", id_field="_doc_id"
    )
    # 第二轮: 相同语料 -> 全部 unchanged，无新增写入
    result, new_known = index.index_records_incremental(
        v1, known=known, text_field="_searchable", id_field="_doc_id"
    )
    assert result.added == []
    assert result.updated == []
    assert result.removed == []
    assert result.unchanged == ["a", "b"]
    assert result.total_changed == 0
    assert store.count() == 2
    assert new_known == known


def test_incremental_only_changed_record_reindexed():
    store = SimpleVectorStore()
    index = VectorIndex(store, _embed)
    v1 = [_rec("a", "内容甲"), _rec("b", "内容乙")]
    _, known = index.index_records_incremental(
        v1, known=None, text_field="_searchable", id_field="_doc_id"
    )
    # b 的内容变化, a 不变
    v2 = [_rec("a", "内容甲"), _rec("b", "完全不同的乙内容")]
    result, new_known = index.index_records_incremental(
        v2, known=known, text_field="_searchable", id_field="_doc_id"
    )
    assert result.added == []
    assert result.updated == ["b"]
    assert result.removed == []
    assert result.unchanged == ["a"]
    assert store.count() == 2
    # 新内容可被检索到 (search 返回 (score, doc_idx) 元组)
    hits = index.search("乙内容", top_k=5)
    doc_idxs = {i for _, i in hits}
    # doc_idx=1 对应 v2 的第二条 (b), 其内容含 "乙内容"
    assert 1 in doc_idxs


def test_incremental_removed_records_deleted():
    store = SimpleVectorStore()
    index = VectorIndex(store, _embed)
    v1 = [_rec("a", "内容甲"), _rec("b", "内容乙"), _rec("c", "内容丙")]
    _, known = index.index_records_incremental(
        v1, known=None, text_field="_searchable", id_field="_doc_id"
    )
    # c 消失
    v2 = [_rec("a", "内容甲"), _rec("b", "内容乙")]
    result, new_known = index.index_records_incremental(
        v2, known=known, text_field="_searchable", id_field="_doc_id"
    )
    assert result.removed == ["c"]
    assert result.unchanged == ["a", "b"]
    assert store.count() == 2
    assert "c" not in store.ids()


def test_incremental_repeated_runs_converge_to_full():
    """多轮增量后，与一次性全量嵌入的结果一致 (检索回 doc 均正确)。"""
    store = SimpleVectorStore()
    index = VectorIndex(store, _embed)
    # 每轮加一条 —— 每次都只重索引新来的那一条
    all_recs = [_rec("r1", "配置代理服务器"), _rec("r2", "配置邮件告警"), _rec("r3", "配置日志轮转")]
    known: dict[str, str] = {}
    for batch_size in range(1, len(all_recs) + 1):
        batch = all_recs[:batch_size]
        result, known = index.index_records_incremental(
            batch, known=known, text_field="_searchable", id_field="_doc_id"
        )
        assert store.count() == batch_size
        assert result.total_changed <= 1  # 每轮至多重索引 1 条
    assert store.count() == 3
    hits = index.search("邮件", top_k=5)
    doc_idxs = {i for _, i in hits}
    # doc_idx=1 对应 all_recs 第二条 (r2 "配置邮件告警")
    assert 1 in doc_idxs


def test_index_records_short_circuit_after_incremental():
    """增量后紧跟 index_records (全量) 不误触发重建 (签名同步)。"""
    store = SimpleVectorStore()
    index = VectorIndex(store, _embed)
    v1 = [_rec("a", "内容甲"), _rec("b", "内容乙")]
    _, known = index.index_records_incremental(
        v1, known=None, text_field="_searchable", id_field="_doc_id"
    )
    # 直接整批重建，内容未变 -> 短路，不重复嵌入
    n = index.index_records(v1, text_field="_searchable", id_field="_doc_id")
    # store.count 应为 2 (SimpleVectorStore.count 已含重建字段)
    assert n == store.count()
    assert store.count() == 2


def test_incremental_result_is_data_class():
    r = IncrementalIndexResult(added=["a"], updated=["b"], removed=["c"], unchanged=["d"])
    assert r.total_changed == 3
    r2 = IncrementalIndexResult()
    assert r2.total_changed == 0
