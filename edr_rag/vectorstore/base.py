"""可插拔向量存储 —— 抽象接口与通用工具 (P3-1).

设计目标 (对应 P3-1 desc):
  抽象向量存储接口，使检索层的向量检索不再与某一种具体后端 (如内存余弦)
  紧耦合，而是可通过配置切换到 elasticsearch / opensearch / qdrant /
  milvus / pgvector 等外部后端；SimpleVectorStore 作为默认的本地内存实现，
  保证主链路在不引入外部服务 / 可选依赖时始终可用。

本模块提供:
  - VectorRecord / VectorHit: 写入与命中的统一数据结构。
  - VectorStore: 抽象接口 (upsert / search / delete / count / close)。
  - VectorIndex: 桥接「文档记录 + embedder」与 VectorStore 的辅助类，
    JsonlCorpus 用它把每条记录的 _searchable 文本转化为向量并索引/检索。

降级策略 (与 rerank / embedding 一致):
  - 外部后端的客户端库均属可选依赖；缺依赖或连接失败时由 factory 优雅
    回退到 SimpleVectorStore，主链路 (keyword/BM25) 始终可用，绝不抛异常。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

#: 默认值常数，供各实现复用。
DEFAULT_DIM = 512


@dataclass
class VectorRecord:
    """一条待写入向量存储的记录。

    Attributes:
        id: 文档主键 (如 action_id / node id / 'rec_{doc_idx}')。
        vector: 嵌入向量 (浮点列表)。
        metadata: 附加元数据，写入时随向量一起保存，检索时可带回。
        text: 原始文本 (可选的，供展示/渲染)。
    """

    id: str
    vector: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)
    text: str = ""


@dataclass
class VectorHit:
    """一次向量检索的命中结果。"""

    id: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    text: str = ""


class VectorStore(ABC):
    """向量存储的抽象接口。

    P3-1: 检索层 (JsonlCorpus.vector_search) 只依赖本接口，不关心底层实现；
    具体后端 (内存 / ES / OS / Qdrant / Milvus / PG) 均为本接口的实现，
    由 factory 依据配置选择。实现方在缺依赖 / 连接失败时应自行抛出可被
    factory 捕获的异常，由调用方回退到 SimpleVectorStore。
    """

    name: str = "base"

    @abstractmethod
    def upsert(self, records: Iterable[VectorRecord]) -> int:
        """批量写入/更新记录；返回实际写入条数。"""

    @abstractmethod
    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filter: Optional[dict[str, Any]] = None,
    ) -> list[VectorHit]:
        """按查询向量检索 top_k 条命中，按相关度降序。

        Args:
            query_vector: 查询向量。
            top_k: 返回条数上限。
            filter: 可选 metadata 过滤 (键值精确匹配)。

        Returns:
            降序 VectorHit 列表。
        """

    @abstractmethod
    def delete(self, ids: Iterable[str]) -> int:
        """按主键删除记录；返回删除条数。"""

    @abstractmethod
    def count(self) -> int:
        """返回当前存储中的记录数。"""

    def ids(self) -> list[str]:  # noqa: B027 - 默认尽力返回空，供重建前清空
        """返回存储中的所有主键 (尽力而为；外部后端可覆盖)。"""
        return []

    def close(self) -> None:  # noqa: B027 - 可选，默认空实现
        """释放底层连接/资源 (外部后端需要时覆盖)。"""


class VectorIndex:
    """桥接「文档记录 + embedder」与 VectorStore 的辅助类。

    JsonlCorpus 用它把每条记录的 _searchable 文本经 embedder 转成向量并
    通过 VectorStore.upsert 写入；检索时对查询文本做同样嵌入后调用
    VectorStore.search。Result 的 doc_idx 通过写进 metadata 的 'doc_idx'
    键带回，供上层 (JsonlCorpus) 还原原始记录。

    本类不缓存文档向量——缓存的职责交给 embedder (见 p1.search._cached_vec)，
    以保证不引入重复的全局状态。
    """

    def __init__(
        self,
        store: VectorStore,
        embed: Callable[[str], list[float]],
        dim: int = DEFAULT_DIM,
    ):
        self.store = store
        self.embed = embed
        self.dim = dim
        self._indexed_sig: Optional[str] = None  # 内容签名，防重复索引

    def index_records(
        self,
        records: list[dict],
        text_field: str = "_searchable",
        id_field: str = "id",
        force: bool = False,
    ) -> int:
        """把一批记录经 embedder 嵌入并写入存储。

        Args:
            records: 已规范化的 JsonlCorpus.records (含 _searchable 等内部字段)。
            text_field: 取哪一字段作为嵌入文本。
            id_field: 取哪一字段作为主键；记录缺失时回退为 'rec_{doc_idx}'。
            force: 是否忽略内容签名强制重建 (缺省按文本内容 hash 判断)。

        Returns:
            实际写入条数。
        """
        sig = _content_sig(records, text_field)
        if not force and self._indexed_sig == sig:
            return self.count()
        self.store.delete(_all_ids(self.store))
        vector_records = []
        for i, r in enumerate(records):
            text = r.get(text_field) or r.get("text") or r.get("content") or ""
            if not text.strip():
                continue
            vec = self.embed(text)
            if not vec:
                continue
            rec_id = str(r.get(id_field) or f"rec_{i}")
            meta: dict[str, Any] = {"doc_idx": i, "id": rec_id}
            meta.update(r.get("metadata") or {})
            vector_records.append(
                VectorRecord(
                    id=rec_id,
                    vector=list(vec),
                    metadata=meta,
                    text=text,
                )
            )
        n = self.store.upsert(vector_records) if vector_records else self.count()
        self._indexed_sig = sig
        return n

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter: Optional[dict[str, Any]] = None,
    ) -> list[tuple[float, int]]:
        """对查询文本嵌入后检索，返回 [(score, doc_idx)] 降序列表。

        Returns:
            [(score, doc_idx), ...]；存储为空/无命中时返回 []。
        """
        q_vec = self.embed(query)
        if not q_vec:
            return []
        hits = self.store.search(list(q_vec), top_k=top_k, filter=filter)
        out = []
        for h in hits:
            doc_idx = h.metadata.get("doc_idx")
            if doc_idx is None:
                continue
            out.append((h.score, int(doc_idx)))
        return out

    def count(self) -> int:
        return self.store.count()


def _all_ids(store: VectorStore) -> list[str]:
    """尽力枚举 Store 中已有的主键 (供 index_records 重建前清空)。"""
    try:
        return store.ids()
    except (AttributeError, NotImplementedError):
        return []


def _content_sig(records: list[dict], text_field: str) -> str:
    """对记录的嵌入文本内容做轻量 hash，用于判断是否需要重建索引。

    相比仅比较长度，内容级签名能在语料被重新摄取 (文本变化而条数不变) 时
    正确触发重建，避免检索到陈旧向量；哈希开销远低于重新嵌入，可接受。
    """
    import hashlib

    h = hashlib.sha1()
    for r in records:
        text = r.get(text_field) or r.get("text") or r.get("content") or ""
        h.update(text.encode("utf-8", "ignore"))
    return h.hexdigest()
