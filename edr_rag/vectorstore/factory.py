"""向量存储工厂 —— 配置化选择后端 (P3-1).

build_vector_store(config) 依据配置的 ``backend`` 字段 (或环境变量
EDR_RAG_VECTOR_BACKEND) 实例化对应后端:

  - simple          -> SimpleVectorStore (默认，内存 cosine，无外部依赖)
  - elasticsearch   -> ElasticsearchVectorStore (骨架)
  - opensearch      -> OpenSearchVectorStore (骨架)
  - qdrant          -> QdrantVectorStore (骨架)
  - milvus          -> MilvusVectorStore (骨架)
  - pgvector        -> PGVectorStore (骨架)

外部端点为「骨架」实现 (见 backends.py): 客户端库属可选依赖且此处不做
真实网络调用。当目标后端的客户端库缺失、或出现任何构造/连接异常时，
factory 记录 warning 并**优雅回退**到 SimpleVectorStore——主链路
(keyword/BM25) 始终可用，绝不因配置了外部向量库而抛错。

环境变量 (等价于 config 键，config 优先):
  EDR_RAG_VECTOR_BACKEND   后端名 (默认 'simple')
  EDR_RAG_VECTOR_URL       qdrant / pgvector 的连接串或 url
  EDR_RAG_VECTOR_HOST      外部服务 host (ES/OS/Milvus)
  EDR_RAG_VECTOR_PORT      外部服务端口
  EDR_RAG_VECTOR_INDEX     索引 / collection / table 名 (后端按需取用)
  EDR_RAG_VECTOR_DIM        向量维度
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from .base import VectorStore
from .backends import (
    ElasticsearchVectorStore,
    MilvusVectorStore,
    OpenSearchVectorStore,
    PGVectorStore,
    QdrantVectorStore,
)
from .simple import SimpleVectorStore

logger = logging.getLogger(__name__)

#: 允许的后端名 -> 构造器 (backend, 需连接参数)。
_BACKENDS: dict[str, type] = {
    "simple": SimpleVectorStore,
    "elasticsearch": ElasticsearchVectorStore,
    "opensearch": OpenSearchVectorStore,
    "qdrant": QdrantVectorStore,
    "milvus": MilvusVectorStore,
    "pgvector": PGVectorStore,
}

#: 便捷别名 (elasticsearch == es；opensearch == os；pgvector == pg)。
_ALIASES = {
    "es": "elasticsearch",
    "os": "opensearch",
    "pg": "pgvector",
    "memory": "simple",
    "local": "simple",
}


def _resolve_backend(name: str | None) -> str:
    name = (name or os.environ.get("EDR_RAG_VECTOR_BACKEND", "") or "simple").strip().lower()
    name = _ALIASES.get(name, name)
    if name not in _BACKENDS:
        logger.warning(
            "unknown vector backend '%s'; falling back to 'simple'", name
        )
        return "simple"
    return name


def _env_config() -> dict[str, Any]:
    """把向量相关环境变量收敛成 config dict。"""
    cfg: dict[str, Any] = {}
    for key, env in (
        ("url", "EDR_RAG_VECTOR_URL"),
        ("host", "EDR_RAG_VECTOR_HOST"),
        ("dsn", "EDR_RAG_VECTOR_URL"),
        ("index", "EDR_RAG_VECTOR_INDEX"),
        ("collection", "EDR_RAG_VECTOR_INDEX"),
        ("table", "EDR_RAG_VECTOR_INDEX"),
    ):
        val = os.environ.get(env)
        if val:
            cfg[key] = val
    port = os.environ.get("EDR_RAG_VECTOR_PORT")
    if port:
        try:
            cfg["port"] = int(port)
        except ValueError:  # pragma: no cover - 防御性
            logger.warning("invalid EDR_RAG_VECTOR_PORT=%r ignored", port)
    dim = os.environ.get("EDR_RAG_VECTOR_DIM")
    if dim:
        try:
            cfg["dim"] = int(dim)
        except ValueError:  # pragma: no cover - 防御性
            logger.warning("invalid EDR_RAG_VECTOR_DIM=%r ignored", dim)
    return cfg


def build_vector_store(config: Optional[dict[str, Any]] = None) -> VectorStore:
    """按配置实例化向量存储，缺依赖/异常时优雅回退 SimpleVectorStore。

    Args:
        config: dict，可含 backend / url / host / port / index / collection /
            table / dim 等键。None 时仅从环境变量读取。

    Returns:
        一个 VectorStore 实现。外部后端缺依赖或连接失败时回退
        SimpleVectorStore (主链路始终可用)。
    """
    cfg = dict(config or {})
    # 环境变量作为兜底，config 显式键优先。
    env = _env_config()
    for k, v in env.items():
        cfg.setdefault(k, v)

    backend = _resolve_backend(cfg.get("backend"))
    store_type = _BACKENDS[backend]

    if backend == "simple":
        return store_type(dim=cfg.get("dim"))

    # 外部后端: 取该后端所需参数，构造失败即回退。
    try:
        return _instantiate(store_type, backend, cfg)
    except Exception as exc:  # noqa: BLE001 - 所有失败统一回退到本地
        logger.warning(
            "vector backend '%s' unavailable (%s: %s); falling back to SimpleVectorStore",
            backend, type(exc).__name__, exc,
        )
        return SimpleVectorStore(dim=cfg.get("dim"))


#: 模块级进程内共享的向量存储实例 (惰性，由 get_vector_store 维护)。
_shared_store: Optional[VectorStore] = None


def get_vector_store(config: Optional[dict[str, Any]] = None) -> VectorStore:
    """返回进程内共享的向量存储 (惰性创建，复用已有实例)。

    Default 后端由环境变量 EDR_RAG_VECTOR_BACKEND 决定；显式 config 的
    ``backend`` 会覆盖它。每次调用不重建——同一进程内检索共享同一后端，
    避免反复重新嵌入/连库。若调用方需要指定不同后端，请直接用
    build_vector_store。
    """
    global _shared_store
    if _shared_store is None:
        _shared_store = build_vector_store(config)
    return _shared_store


def _instantiate(store_type: type, backend: str, cfg: dict[str, Any]) -> VectorStore:
    """按后端所需的连接参数构造实例。"""
    common = {"dim": cfg.get("dim")}
    if backend in ("elasticsearch", "opensearch"):
        return store_type(
            host=cfg.get("host", "localhost"),
            port=cfg.get("port", 9200),
            index=cfg.get("index", "edr_rag_vectors"),
            username=cfg.get("username"),
            password=cfg.get("password"),
            **common,
        )
    if backend == "qdrant":
        return store_type(
            url=cfg.get("url", "http://localhost:6333"),
            collection=cfg.get("collection", "edr_rag"),
            api_key=cfg.get("api_key"),
            **common,
        )
    if backend == "milvus":
        return store_type(
            host=cfg.get("host", "localhost"),
            port=cfg.get("port", 19530),
            collection=cfg.get("collection", "edr_rag"),
            **common,
        )
    if backend == "pgvector":
        return store_type(
            dsn=cfg.get("dsn", "postgresql://localhost:5432/edr_rag"),
            table=cfg.get("table", "edr_vectors"),
            **common,
        )
    raise ValueError(f"unhandled backend '{backend}'")
