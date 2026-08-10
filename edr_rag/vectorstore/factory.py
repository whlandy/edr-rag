"""向量存储工厂 —— 按 namespace 配置化选择后端 (P3-1，P4 重写)。

P4 变更 (见 docs/requirements/P4-HARDENING-DESIGN.md §4 Lock 1):
  - **删除** ES / OpenSearch / Qdrant / Milvus / pgvector 五个
    ``NotImplementedError`` 骨架。它们提供的是「支持五种后端」的错觉：任何
    一个被配置上去都会在第一次查询时崩掉、然后被 factory 静默回退。
  - **新增** ``persistent`` 后端 (PersistentVectorStore)：本地文件持久化 +
    精确检索，是内网离线部署的生产默认。
  - **namespace 隔离**：``get_vector_store`` 现在按 namespace 返回独立实例。
    旧实现全进程共用一个 store，``planner_retrieve`` 逐层检索 actions →
    procedures → sections 时会互相清空对方的向量并触发全量重嵌。

可用后端:
  - persistent -> PersistentVectorStore (默认，落盘；directory 为 None 时纯内存)
  - simple     -> SimpleVectorStore     (纯进程内 dict，保留给测试/轻量调用方)

环境变量 (config 显式键优先):
  EDR_RAG_VECTOR_BACKEND   后端名 (默认 'persistent')
  EDR_RAG_VECTOR_DIR       持久化目录 (缺省由调用方传入的 workdir 推导)
  EDR_RAG_VECTOR_DIM       向量维度
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any, Optional

from .base import VectorStore
from .persistent import PersistentVectorStore
from .simple import SimpleVectorStore

logger = logging.getLogger(__name__)

#: 允许的后端名 -> 构造器。
_BACKENDS: dict[str, type] = {
    "persistent": PersistentVectorStore,
    "simple": SimpleVectorStore,
}

#: 便捷别名。
_ALIASES = {
    "memory": "simple",
    "local": "persistent",
    "file": "persistent",
    "disk": "persistent",
}

DEFAULT_BACKEND = "persistent"

#: 向量文件在 workdir 下的子目录名。
VECTOR_SUBDIR = "vector_index"


def _resolve_backend(name: str | None) -> str:
    name = (
        name
        or os.environ.get("EDR_RAG_VECTOR_BACKEND", "")
        or DEFAULT_BACKEND
    ).strip().lower()
    name = _ALIASES.get(name, name)
    if name not in _BACKENDS:
        logger.warning(
            "unknown vector backend '%s'; falling back to '%s'", name, DEFAULT_BACKEND
        )
        return DEFAULT_BACKEND
    return name


def _env_config() -> dict[str, Any]:
    """把向量相关环境变量收敛成 config dict。"""
    cfg: dict[str, Any] = {}
    directory = os.environ.get("EDR_RAG_VECTOR_DIR")
    if directory:
        cfg["directory"] = directory
    dim = os.environ.get("EDR_RAG_VECTOR_DIM")
    if dim:
        try:
            cfg["dim"] = int(dim)
        except ValueError:  # pragma: no cover - 防御性
            logger.warning("invalid EDR_RAG_VECTOR_DIM=%r ignored", dim)
    return cfg


def vector_dir_for_workdir(workdir: str | Path | None) -> Optional[Path]:
    """由 workdir 推导向量持久化目录 (``<workdir>/vector_index``)。"""
    if workdir is None:
        return None
    return Path(workdir) / VECTOR_SUBDIR


def build_vector_store(
    config: Optional[dict[str, Any]] = None,
    namespace: str = "default",
) -> VectorStore:
    """按配置实例化向量存储。

    Args:
        config: 可含 backend / directory / dim。None 时仅从环境变量读取。
        namespace: 语料名，决定持久化文件名与隔离边界。

    Returns:
        VectorStore 实例。持久化目录不可写时降级为纯内存模式 (仍可用)。
    """
    cfg = dict(config or {})
    for k, v in _env_config().items():
        cfg.setdefault(k, v)

    backend = _resolve_backend(cfg.get("backend"))
    dim = cfg.get("dim")

    if backend == "simple":
        return SimpleVectorStore(dim=dim)

    directory = cfg.get("directory")
    try:
        return PersistentVectorStore(
            directory=directory, namespace=namespace, dim=dim
        )
    except ValueError:
        # 非法 namespace 是调用方 bug，必须暴露。
        raise
    except Exception as exc:  # noqa: BLE001 - 磁盘不可用时保住主链路
        logger.warning(
            "persistent vector store unavailable (%s: %s); using in-memory store",
            type(exc).__name__, exc,
        )
        return SimpleVectorStore(dim=dim)


#: 进程内共享的 store 实例，按 (backend, directory, namespace) 缓存。
_shared_stores: dict[tuple[str, str, str], VectorStore] = {}
_shared_lock = threading.Lock()


def get_vector_store(
    namespace: str = "default",
    config: Optional[dict[str, Any]] = None,
) -> VectorStore:
    """返回进程内共享的向量存储，**按 namespace 隔离** (惰性创建)。

    这是 P4 修复跨语料索引冲刷的入口：每个语料拿到自己的 store，互不影响。

    Args:
        namespace: 语料名 (sections / procedures / actions / chunks / ...)。
        config: 同 build_vector_store。

    Returns:
        该 namespace 的共享 VectorStore。
    """
    cfg = dict(config or {})
    for k, v in _env_config().items():
        cfg.setdefault(k, v)
    backend = _resolve_backend(cfg.get("backend"))
    directory = str(cfg.get("directory") or "")
    key = (backend, directory, namespace)
    with _shared_lock:
        store = _shared_stores.get(key)
        if store is None:
            store = build_vector_store(cfg, namespace=namespace)
            _shared_stores[key] = store
        return store


def reset_shared_stores() -> None:
    """清空共享 store 缓存 (测试用；生产不应调用)。"""
    with _shared_lock:
        for store in _shared_stores.values():
            try:
                store.close()
            except Exception:  # pragma: no cover - best-effort
                pass
        _shared_stores.clear()
