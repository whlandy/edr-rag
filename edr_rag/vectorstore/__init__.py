"""可插拔向量存储 (P3-1)。

对外暴露的公共 API:
  - VectorStore / VectorRecord / VectorHit / VectorIndex: 抽象与通用工具
  - SimpleVectorStore: 默认本地内存实现
  - build_vector_store / get_vector_store: 配置化工厂
  - 外部后端骨架: ElasticsearchVectorStore / OpenSearchVectorStore /
    QdrantVectorStore / MilvusVectorStore / PGVectorStore
"""

from __future__ import annotations

from .base import (
    IncrementalIndexResult,
    VectorHit,
    VectorIndex,
    VectorRecord,
    VectorStore,
)
from .backends import (
    ElasticsearchVectorStore,
    MilvusVectorStore,
    OpenSearchVectorStore,
    PGVectorStore,
    QdrantVectorStore,
)
from .factory import build_vector_store, get_vector_store
from .simple import SimpleVectorStore

__all__ = [
    "IncrementalIndexResult",
    "VectorHit",
    "VectorIndex",
    "VectorRecord",
    "VectorStore",
    "SimpleVectorStore",
    "ElasticsearchVectorStore",
    "OpenSearchVectorStore",
    "QdrantVectorStore",
    "MilvusVectorStore",
    "PGVectorStore",
    "build_vector_store",
    "get_vector_store",
]
