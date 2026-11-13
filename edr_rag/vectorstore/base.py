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

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

logger = logging.getLogger(__name__)

#: 默认值常数，供各实现复用。
DEFAULT_DIM = 512

#: 需要提升到向量 metadata 顶层的 ACL 维度 (P4)。让权限谓词可以在存储层
#: 下推，而不是「全语料召回后再过滤」。与 edr_rag.security._ACL_ATTRS 对齐。
ACL_METADATA_FIELDS = ("device", "tenant")


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


@dataclass
class IncrementalIndexResult:
    """一次增量索引 (index_records_incremental) 的结果报告 (P3-5)。

    Attributes:
        added: 新增记录主键 (上次未见)。
        updated: 内容指纹变化、需要重新嵌入的记录主键。
        removed: 上次存在但本次已消失、需要从存储删除的记录主键。
        unchanged: 内容未变、直接跳过的记录主键 (省去重新嵌入)。
    """

    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)

    @property
    def total_changed(self) -> int:
        """实际产生写入/删除操作的记录数 (added + updated + removed)。"""
        return len(self.added) + len(self.updated) + len(self.removed)


def _record_fingerprint(r: dict, text_field: str) -> str:
    """对单条记录的嵌入文本计算稳定指纹 (sha1)，用于增量变更判定。

    P3-5: 相比整批 _content_sig，逐条指纹能让「按源 hash 只重索引变更部分」
    成立——语料里只有少量记录变化时，仅需重嵌入变化的那几条。
    """
    import hashlib

    text = r.get(text_field) or r.get("text") or r.get("content") or ""
    return hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()


class VectorIndex:
    """桥接「文档记录 + embedder」与 VectorStore 的辅助类。

    JsonlCorpus 用它把每条记录的 _searchable 文本经 embedder 转成向量并
    通过 VectorStore.upsert 写入；检索时对查询文本做同样嵌入后调用
    VectorStore.search。Result 的 doc_idx 通过写进 metadata 的 'doc_idx'
    键带回，供上层 (JsonlCorpus) 还原原始记录。

    本类不缓存文档向量——缓存的职责交给 embedder (见 p1.search._cached_vec)，
    以保证不引入重复的全局状态。

    P3-5: 提供两种索引方式——
      - index_records            : 整批重建 (缺省按整批内容签名短路)；
      - index_records_incremental: 增量更新。调用方传入上一次的 id->指纹映射，
        仅重嵌入「新增/内容变化」的记录，删除「已消失」的记录，未变记录直接
        跳过，避免大语料下反复全量嵌入。
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
        self._embed_sig = _embedder_sig(embed)
        self._dirty = False  # 上次 persist 后是否有新写入
        self._restore_sig_from_store()

    def _restore_sig_from_store(self) -> None:
        """从支持持久化签名的 store 恢复签名 (P4 缺陷修复)。

        PersistentVectorStore.load() 已把上次落盘的 content_sig 载入内存；
        VectorIndex 新建时（典型场景：新进程 autoload）据此恢复短路状态，
        避免磁盘向量完整却全量重嵌入。

        磁盘签名格式为 ``content``（模型标识另存于 embedder_sig 字段）；
        恢复时必须校验磁盘记录的模型指纹与当前 embedder 一致——旧实现把
        当前 embedder 标识拼接到磁盘上旧 content 上，换模型后跨进程重启
        拼接结果必然"匹配"，静默服务旧模型向量（1433 审核确认契约漏洞）。
        指纹缺失（旧格式索引）或不一致时拒绝恢复，保持未索引状态，
        由 index_records 全量重建并写回双签名。
        """
        getter = getattr(self.store, "get_content_sig", None)
        if getter is None:
            return
        try:
            stored = getter()
        except Exception:  # pragma: no cover - 防御性
            return
        if not stored:
            return
        embedder_getter = getattr(self.store, "get_embedder_sig", None)
        stored_embedder = None
        if embedder_getter is not None:
            try:
                stored_embedder = embedder_getter()
            except Exception:  # pragma: no cover - 防御性
                stored_embedder = None
        if not stored_embedder or stored_embedder != self._embed_sig:
            logger.warning(
                "vector index embedder fingerprint mismatch or missing "
                "(stored=%r, current=%r); ignoring persisted short-circuit "
                "state to avoid serving stale-model vectors",
                stored_embedder, self._embed_sig,
            )
            return
        self._indexed_sig = self._embed_sig + ":" + stored

    def _note_indexed_sig(self, records: list[dict], text_field: str) -> None:
        """索引成功后记录签名，并回写到持久化 store 供下次进程恢复。

        content 与生成它的嵌入模型标识分开落盘：恢复时二者都需匹配
        当前语料与 embedder 才允许短路，见 ``_restore_sig_from_store``。
        """
        self._dirty = True
        content = _content_sig(records, text_field)
        self._indexed_sig = self._embed_sig + ":" + content
        setter = getattr(self.store, "set_content_sig", None)
        if setter is not None:
            try:
                setter(content)
            except Exception:  # pragma: no cover - best-effort
                pass
        embedder_setter = getattr(self.store, "set_embedder_sig", None)
        if embedder_setter is not None:
            try:
                embedder_setter(self._embed_sig)
            except Exception:  # pragma: no cover - best-effort
                pass

    def index_records(
        self,
        records: list[dict],
        text_field: str = "_searchable",
        id_field: str = "id",
        force: bool = False,
    ) -> int:
        """把一批记录经 embedder 嵌入并写入存储 (整批重建)。

        Args:
            records: 已规范化的 JsonlCorpus.records (含 _searchable 等内部字段)。
            text_field: 取哪一字段作为嵌入文本。
            id_field: 取哪一字段作为主键；记录缺失时回退为 'rec_{doc_idx}'。
            force: 是否忽略内容签名强制重建 (缺省按文本内容 hash 判断)。

        Returns:
            实际写入条数。
        """
        sig = self._embed_sig + ":" + _content_sig(records, text_field)
        if not force and self._indexed_sig == sig:
            return self.count()
        self.store.delete(_all_ids(self.store))
        n = self._embed_upsert(records, text_field, id_field)
        self._note_indexed_sig(records, text_field)
        return n

    def _embed_upsert(
        self,
        records: list[dict],
        text_field: str,
        id_field: str,
        doc_indices: Optional[list[int]] = None,
    ) -> int:
        """把一批记录嵌入并 upsert 进 store，返回实际写入 (含替换) 条数。

        不清理既有记录、不碰内容签名——仅供 index_records / 增量索引内部复用。

        Args:
            doc_indices: 可选，与 `records` 等长的、每条记录在**完整语料**中的
                位置 (doc_idx)。缺省时用本条记录在 `records` 内的下标。
                JsonlCorpus 依赖 doc_idx 精确映射回 self.records，因此增量索引
                重嵌入子集时必须显式传入原始位置 (P3-5)。
        """
        vector_records = []
        # 单批内同 ID 检测：PersistentVectorStore.upsert 按 ID 覆盖，静默
        # 后写覆盖 = 数据丢失（1433 审核漏洞 #2）。批内消歧为唯一存储键，
        # 原始 id 保留在 metadata（检索经 doc_idx 回记录，后缀不外泄）。
        seen_ids: dict[str, int] = {}
        for i, r in enumerate(records):
            text = r.get(text_field) or r.get("text") or r.get("content") or ""
            if not text.strip():
                continue
            vec = self.embed(text)
            if not vec:
                continue
            rec_id = str(r.get(id_field) or f"rec_{i}")
            n = seen_ids.get(rec_id, 0) + 1
            seen_ids[rec_id] = n
            store_id = rec_id if n == 1 else f"{rec_id}#dup{n}"
            real_idx = doc_indices[i] if doc_indices is not None else i
            meta: dict[str, Any] = {"doc_idx": real_idx, "id": rec_id}
            meta.update(r.get("metadata") or {})
            # ACL 维度提升为顶层 metadata，使权限谓词能在存储层下推 (P4)。
            # 顶层字段优先于 metadata 内嵌值，与 security._record_acl_field
            # 的读取顺序保持一致。
            for attr in ACL_METADATA_FIELDS:
                val = r.get(attr)
                if val is not None:
                    meta[attr] = val
            vector_records.append(
                VectorRecord(
                    id=store_id,
                    vector=list(vec),
                    metadata=meta,
                    text=text,
                )
            )
        return self.store.upsert(vector_records) if vector_records else 0

    def index_records_incremental(
        self,
        records: list[dict],
        known: Optional[dict[str, str]] = None,
        text_field: str = "_searchable",
        id_field: str = "id",
    ) -> tuple[IncrementalIndexResult, dict[str, str]]:
        """按「源 hash/内容指纹」增量更新索引，只重索引变更部分 (P3-5)。

        对比上一轮的 id->内容指纹映射 `known`，得出四类记录：
          - 新增 (added)     : 本次新出现，需嵌入；
          - 变化 (updated)   : 内容指纹不同，需重新嵌入；
          - 消失 (removed)   : 从存储删除；
          - 未变 (unchanged) : 跳过，不重嵌入 (省去反复嵌入的开销)。
        仅对 added+updated 做嵌入、对 removed 做删除；未变记录保持原向量不动。

        Args:
            records: 当前语料记录列表。
            known: 上一轮的 {记录主键: 内容指纹} 映射；None/空表示首次全量。
            text_field: 嵌入文本字段。
            id_field: 主键字段；缺失时回退为 'rec_{doc_idx}'。

        Returns:
            (result, new_known)——result 为本次变更报告；new_known 为全新的
            {主键: 指纹} 映射，调用方应持久化，供下一轮增量使用。
        """
        known = known or {}
        new_known: dict[str, str] = {}
        by_id: dict[str, dict] = {}
        orig_index: dict[str, int] = {}
        added: list[str] = []
        updated: list[str] = []
        unchanged: list[str] = []
        # 同批内同 ID 消歧（与 _embed_upsert 同一方案）：重复 ID 是实质
        # 不同的变体，共享键会让 by_id/orig_index/new_known 收敛到最后一
        # 变体——前位变体变化时以错误文本在错误位置重嵌入（1433 审核漏洞
        # #2 的增量形状）。批内第 2..n 次出现的 ID 记为 <id>#dupN。
        seen_ids: dict[str, int] = {}

        for i, r in enumerate(records):
            rec_id = str(r.get(id_field) or f"rec_{i}")
            n = seen_ids.get(rec_id, 0) + 1
            seen_ids[rec_id] = n
            key = rec_id if n == 1 else f"{rec_id}#dup{n}"
            fp = _record_fingerprint(r, text_field)
            new_known[key] = fp
            by_id[key] = r
            orig_index[key] = i
            if key not in known:
                added.append(key)
            elif known[key] != fp:
                updated.append(key)
            else:
                unchanged.append(key)

        removed = [rid for rid in known if rid not in new_known]

        # 只重嵌入新增 + 变化的部分；消失的记录从存储删除。
        changed_ids = added + updated
        to_index = [by_id[rid] for rid in changed_ids]
        # 关键: doc_idx 是记录在**完整语料**中的位置，不是子集内位置——
        # JsonlCorpus 依靠它映射回 self.records (P3-5)。
        doc_indices = [orig_index[rid] for rid in changed_ids]
        if to_index:
            self._embed_upsert(to_index, text_field, id_field, doc_indices=doc_indices)
        if removed:
            self.store.delete(removed)

        # 让 index_records 的整批签名短路与新状态一致，避免紧随其后的
        # 全量重建被误触发。
        self._note_indexed_sig(records, text_field)

        return (
            IncrementalIndexResult(
                added=added, updated=updated, removed=removed, unchanged=unchanged
            ),
            new_known,
        )

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

    def persist(self) -> bool:
        """把底层存储落盘 (P4)。存储不支持持久化时返回 False，不抛异常。

        由调用方在**实际发生写入之后**调用；``index_records`` 的内容签名短路
        命中时磁盘已是最新，本方法直接返回 True 不重写存储，避免每次查询
        都产生无谓 IO。
        """
        save = getattr(self.store, "save", None)
        if save is None:
            return False
        if not self._dirty:
            return True
        try:
            saved = bool(save())
            if saved:
                self._dirty = False
            return saved
        except Exception:  # pragma: no cover - best-effort 持久化
            import logging

            logging.getLogger(__name__).warning(
                "vector index persist failed", exc_info=True
            )
            return False


def _all_ids(store: VectorStore) -> list[str]:
    """尽力枚举 Store 中已有的主键 (供 index_records 重建前清空)。"""
    try:
        return store.ids()
    except (AttributeError, NotImplementedError):
        return []


def _embedder_sig(embed: Callable[[str], list[float]]) -> str:
    """嵌入函数的稳定标识，并入内容签名。

    同一批文本换嵌入模型后向量空间不同，签名必须跟着变，否则持久化
    短路会静默服务陈旧向量。优先读 ``model_id`` 属性 (真实 embedder
    构造时由 p1.search 附加模型名)；测试/自定义函数回退到
    ``module.qualname`` —— 同一进程内同一函数对象标识稳定。
    """
    model_id = getattr(embed, "model_id", None)
    if model_id:
        return str(model_id)
    return f"{getattr(embed, '__module__', '?')}.{getattr(embed, '__qualname__', '?')}"


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
