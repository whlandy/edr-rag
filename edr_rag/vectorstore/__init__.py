"""可插拔向量存储 (P3-1，P4 重写后端集合)。

对外暴露的公共 API:
  - VectorStore / VectorRecord / VectorHit / VectorIndex: 抽象与通用工具
  - PersistentVectorStore: 本地文件持久化 + 精确检索 (生产默认)
  - SimpleVectorStore: 纯进程内实现 (测试 / 轻量调用方)
  - build_vector_store / get_vector_store: 按 namespace 的配置化工厂

P4 移除了 ElasticsearchVectorStore / OpenSearchVectorStore /
QdrantVectorStore / MilvusVectorStore / PGVectorStore 五个
``NotImplementedError`` 骨架 —— 详见 docs/requirements/P4-HARDENING-DESIGN.md
§4 Lock 1。VectorStore 抽象接口保留，语料规模真的需要外部向量库时新增一个
实现即可，检索层无需改动。
"""

from __future__ import annotations

from .base import (
    IncrementalIndexResult,
    VectorHit,
    VectorIndex,
    VectorRecord,
    VectorStore,
)
from .factory import (
    VECTOR_SUBDIR,
    build_vector_store,
    get_vector_store,
    reset_shared_stores,
    vector_dir_for_workdir,
)
from .persistent import PersistentVectorStore, validate_namespace
from .simple import SimpleVectorStore

__all__ = [
    "IncrementalIndexResult",
    "VectorHit",
    "VectorIndex",
    "VectorRecord",
    "VectorStore",
    "PersistentVectorStore",
    "SimpleVectorStore",
    "build_vector_store",
    "get_vector_store",
    "reset_shared_stores",
    "vector_dir_for_workdir",
    "validate_namespace",
    "VECTOR_SUBDIR",
]
