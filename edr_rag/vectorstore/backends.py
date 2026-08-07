"""外部向量数据库的后端适配器骨架 (P3-1)。

这些类实现 VectorStore 接口，但作为「骨架」: 客户端库均为可选依赖且
此处不做真实网络调用——只定义每个后端所需的连接配置、可用性探测
(available() 检查客户端库是否可导入) 与调用形状 (方法签名/参数约束)，
留出真实的 upsert/search/delete 实现位 (标有 ``# TODO(P3-1): 接通真实
客户端``)。缺依赖时 available() 返回 False，factory 会优雅回退到
SimpleVectorStore，主链路不报错。

覆盖后端:
  - ElasticsearchVectorStore / OpenSearchVectorStore (共用 _ESLikeStore)
  - QdrantVectorStore
  - MilvusVectorStore
  - PGVectorStore (pgvector)

每个适配器通过 dict 配置构造，约定的键 (供 factory 从 env 解析后传入):
  url / host / port / index (ES/OS) | collection (Qdrant) | table (PG)
  可选: dim, username, password
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Optional

from .base import VectorHit, VectorRecord, VectorStore

logger = logging.getLogger(__name__)


def _client(name: str):
    """按客户端库名导入，缺依赖返回 None (不抛异常)。"""
    try:
        if name == "elasticsearch":
            import elasticsearch  # type: ignore
            return elasticsearch
        if name == "opensearch":
            import opensearchpy  # type: ignore
            return opensearchpy
        if name == "qdrant":
            import qdrant_client  # type: ignore
            return qdrant_client
        if name == "milvus":
            from pymilvus import connections  # type: ignore
            return connections
        if name == "pgvector":
            import psycopg  # type: ignore
            return psycopg
    except Exception:  # pragma: no cover - 可选依赖缺失
        return None
    return None


class _ESLikeStore(VectorStore):
    """Elasticsearch / OpenSearch 骨架 (二者 API 相近，共用实现位)。"""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 9200,
        index: str = "edr_rag_vectors",
        dim: int | None = None,
        username: str | None = None,
        password: str | None = None,
    ):
        self.host = host
        self.port = int(port)
        self.index = index
        self.dim = dim
        self.username = username
        self.password = password
        self._client = None
        self._connected = False

    @property
    def _lib_available(self) -> bool:
        return _client("elasticsearch") is not None or _client("opensearch") is not None

    def _connect(self):
        # TODO(P3-1): 用 elasticsearch / opensearchpy 客户端连接远端，
        # 创建 `index` 的 dense_vector 映射后置 _connected=True。
        if not self._lib_available:
            raise RuntimeError("elasticsearch/opensearch client not installed")
        raise NotImplementedError("ES/OS backend skeleton: real client wiring pending")

    def upsert(self, records: Iterable[VectorRecord]) -> int:
        if not self._connected:
            self._connect()
        # TODO(P3-1): bulk index records (id -> vector + metadata)
        raise NotImplementedError("ES/OS upsert skeleton")

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filter: Optional[dict[str, Any]] = None,
    ) -> list[VectorHit]:
        if not self._connected:
            self._connect()
        # TODO(P3-1): knn / script_score query
        raise NotImplementedError("ES/OS search skeleton")

    def delete(self, ids: Iterable[str]) -> int:
        if not self._connected:
            self._connect()
        # TODO(P3-1): delete_by_query on ids
        raise NotImplementedError("ES/OS delete skeleton")

    def count(self) -> int:
        if not self._connected:
            self._connect()
        return 0

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:  # pragma: no cover
                pass
        self._connected = False


class ElasticsearchVectorStore(_ESLikeStore):
    """Elasticsearch 向量后端骨架。"""

    name = "elasticsearch"


class OpenSearchVectorStore(_ESLikeStore):
    """OpenSearch 向量后端骨架。"""

    name = "opensearch"


class QdrantVectorStore(VectorStore):
    """Qdrant 向量后端骨架 (collection 存储)。"""

    name = "qdrant"

    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection: str = "edr_rag",
        dim: int | None = None,
        api_key: str | None = None,
    ):
        self.url = url
        self.collection = collection
        self.dim = dim
        self.api_key = api_key
        self._client = None
        self._connected = False

    def _connect(self):
        if _client("qdrant") is None:
            raise RuntimeError("qdrant-client not installed")
        # TODO(P3-1): QdrantClient(url=..., api_key=...); ensure collection exists
        raise NotImplementedError("Qdrant backend skeleton: real client wiring pending")

    def upsert(self, records: Iterable[VectorRecord]) -> int:
        if not self._connected:
            self._connect()
        # TODO(P3-1): client.upsert(collection_name, points=[...])
        raise NotImplementedError("Qdrant upsert skeleton")

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filter: Optional[dict[str, Any]] = None,
    ) -> list[VectorHit]:
        if not self._connected:
            self._connect()
        # TODO(P3-1): client.search(collection_name, query_vector=..., limit=top_k)
        raise NotImplementedError("Qdrant search skeleton")

    def delete(self, ids: Iterable[str]) -> int:
        if not self._connected:
            self._connect()
        raise NotImplementedError("Qdrant delete skeleton")

    def count(self) -> int:
        if not self._connected:
            self._connect()
        return 0


class MilvusVectorStore(VectorStore):
    """Milvus 向量后端骨架 (collection 存储)。"""

    name = "milvus"

    def __init__(
        self,
        host: str = "localhost",
        port: int = 19530,
        collection: str = "edr_rag",
        dim: int | None = None,
    ):
        self.host = host
        self.port = int(port)
        self.collection = collection
        self.dim = dim
        self._connected = False

    def _connect(self):
        if _client("milvus") is None:
            raise RuntimeError("pymilvus not installed")
        # TODO(P3-1): connections.connect(host=..., port=...); ensure collection
        raise NotImplementedError("Milvus backend skeleton: real client wiring pending")

    def upsert(self, records: Iterable[VectorRecord]) -> int:
        if not self._connected:
            self._connect()
        raise NotImplementedError("Milvus upsert skeleton")

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filter: Optional[dict[str, Any]] = None,
    ) -> list[VectorHit]:
        if not self._connected:
            self._connect()
        raise NotImplementedError("Milvus search skeleton")

    def delete(self, ids: Iterable[str]) -> int:
        if not self._connected:
            self._connect()
        raise NotImplementedError("Milvus delete skeleton")

    def count(self) -> int:
        if not self._connected:
            self._connect()
        return 0


class PGVectorStore(VectorStore):
    """PostgreSQL + pgvector 后端骨架 (table 存储)。"""

    name = "pgvector"

    def __init__(
        self,
        dsn: str = "postgresql://localhost:5432/edr_rag",
        table: str = "edr_vectors",
        dim: int | None = None,
    ):
        self.dsn = dsn
        self.table = table
        self.dim = dim
        self._conn = None

    def _connect(self):
        if _client("pgvector") is None:
            raise RuntimeError("psycopg not installed")
        # TODO(P3-1): psycopg.connect(dsn); CREATE TABLE ... vector(dim)
        raise NotImplementedError("PGVector backend skeleton: real client wiring pending")

    def upsert(self, records: Iterable[VectorRecord]) -> int:
        if self._conn is None:
            self._connect()
        raise NotImplementedError("PGVector upsert skeleton")

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filter: Optional[dict[str, Any]] = None,
    ) -> list[VectorHit]:
        if self._conn is None:
            self._connect()
        # TODO(P3-1): SELECT ... ORDER BY embedding <=> %s LIMIT top_k
        raise NotImplementedError("PGVector search skeleton")

    def delete(self, ids: Iterable[str]) -> int:
        if self._conn is None:
            self._connect()
        raise NotImplementedError("PGVector delete skeleton")

    def count(self) -> int:
        if self._conn is None:
            self._connect()
        return 0

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:  # pragma: no cover
                pass
            self._conn = None
