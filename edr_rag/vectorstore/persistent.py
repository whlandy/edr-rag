"""本地持久化向量存储 —— P4 的生产默认后端。

替代 P3-1 的五个 ``NotImplementedError`` 外部后端骨架 (已删除)。设计取舍见
``docs/requirements/P4-HARDENING-DESIGN.md`` §4 Lock 1，摘要:

  - **内网离线部署**：交付物是 pip 包 + 模型文件 + workdir，不要求客户现场
    运维任何中间件 (Qdrant / ES / Postgres 都是要装、要备份、要升级的东西)。
  - **精确检索而非 ANN**：单产品手册量级 (1k–5万 chunk) 下 numpy 向量化余弦
    是毫秒级，HNSW 收益接近零；而精确检索**没有召回损失**，这对 ACL 过滤后
    仍要保证召回完整性很重要 (filtered-HNSW 在高选择性过滤时会漏召回)。
  - **持久化**：向量落盘，进程重启不重新嵌入。这是 P3-1 SimpleVectorStore
    最大的生产缺陷 —— 它是纯进程内 dict。

存储格式 (纯 stdlib 可读写，numpy 可选加速):
  ``<dir>/<namespace>.vec``        原始 float32 小端字节流，N × dim 行主序
  ``<dir>/<namespace>.meta.json``  {schema_version, dim, ids, metadata, texts}

选择裸 float32 而不是 JSON：5万 × 512 维的 JSON 文本约 200MB，float32 只有
100MB 且可被 numpy ``frombuffer`` 零拷贝加载。选择 stdlib ``array`` 做兜底，
保证没有 numpy 的最小化环境同样能读写同一份文件 (格式与 numpy 完全一致)。

namespace 契约 (P4 Lock，修复跨语料索引冲刷):
  每个语料 (sections / procedures / actions / chunks / nodes / images) **必须**
  使用独立 namespace。旧实现全进程共用一个 store，换语料就触发
  ``delete(全部) + 全量重嵌``，``planner_retrieve`` 三层检索会互相清空对方的
  向量。namespace 让每个语料拥有独立的文件与内存矩阵。
"""

from __future__ import annotations

import array
import json
import logging
import math
import re
import threading
from pathlib import Path
from typing import Any, Iterable, Optional

from .base import VectorHit, VectorRecord, VectorStore

logger = logging.getLogger(__name__)

#: 持久化格式版本；不兼容变更时递增，读到更高版本视为不可读并重建。
PERSIST_SCHEMA_VERSION = 1

#: 向量元素类型 —— float32 小端。`array('f')` 与 numpy float32 二进制等价。
_VEC_TYPECODE = "f"

#: namespace 合法字符 (直接参与文件名，必须防路径穿越)。
_NS_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def _numpy():
    """返回 numpy 模块，缺失时返回 None (纯 Python 兜底)。"""
    try:
        import numpy  # type: ignore

        return numpy
    except Exception:  # pragma: no cover - 可选加速依赖
        return None


def validate_namespace(namespace: str) -> str:
    """校验 namespace 可安全作为文件名；非法时抛 ValueError。"""
    ns = (namespace or "").strip()
    if not ns or not _NS_RE.match(ns) or ns in (".", ".."):
        raise ValueError(
            f"invalid vector namespace {namespace!r}; "
            "expected [A-Za-z0-9._-]+ (used verbatim as a filename)"
        )
    return ns


class PersistentVectorStore(VectorStore):
    """本地文件持久化的精确向量存储 (每 namespace 一份)。

    线程安全：所有公开方法在实例级 ``RLock`` 下执行 (P4 严重问题 10 —— 旧
    SimpleVectorStore 明确不保证线程安全，而 HTTP 服务是多线程的)。

    Attributes:
        directory: 持久化目录 (``<workdir>/vector_index``)。None 表示纯内存
            模式 (测试 / 无 workdir 的调用方)，行为与旧 SimpleVectorStore 一致。
        namespace: 语料名，决定文件名与隔离边界。
    """

    name = "persistent"

    def __init__(
        self,
        directory: str | Path | None = None,
        namespace: str = "default",
        dim: int | None = None,
        autoload: bool = True,
    ):
        self.namespace = validate_namespace(namespace)
        self.directory = Path(directory) if directory is not None else None
        self.dim = dim
        self._lock = threading.RLock()
        # 行主序存储：_ids[i] 对应 _vectors 的第 i 行。
        self._ids: list[str] = []
        self._metadata: list[dict[str, Any]] = []
        self._texts: list[str] = []
        self._vectors = array.array(_VEC_TYPECODE)
        self._pos: dict[str, int] = {}      # id -> 行号
        self._content_sig: Optional[str] = None  # 整批内容签名，供 VectorIndex 短路
        self._embedder_sig: Optional[str] = None  # 生成这些向量所用嵌入模型的标识
        self._matrix = None                  # numpy 归一化矩阵缓存
        self._matrix_dirty = True
        self._np = _numpy()
        if autoload and self.directory is not None:
            self.load()

    # ---------- 路径 ----------

    @property
    def vec_path(self) -> Optional[Path]:
        if self.directory is None:
            return None
        return self.directory / f"{self.namespace}.vec"

    @property
    def meta_path(self) -> Optional[Path]:
        if self.directory is None:
            return None
        return self.directory / f"{self.namespace}.meta.json"

    # ---------- 持久化 ----------

    def load(self) -> bool:
        """从磁盘载入向量与元数据。文件缺失/损坏时返回 False 并保持空状态。"""
        vec_path, meta_path = self.vec_path, self.meta_path
        if vec_path is None or meta_path is None:
            return False
        if not vec_path.is_file() or not meta_path.is_file():
            return False
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, ValueError) as e:
            logger.warning("vector meta %s unreadable: %s", meta_path, e)
            return False

        if int(meta.get("schema_version", 0)) > PERSIST_SCHEMA_VERSION:
            logger.warning(
                "vector index %s written by a newer schema (%s > %s); ignoring",
                meta_path, meta.get("schema_version"), PERSIST_SCHEMA_VERSION,
            )
            return False

        dim = int(meta.get("dim") or 0)
        ids = list(meta.get("ids") or [])
        metadata = list(meta.get("metadata") or [])
        texts = list(meta.get("texts") or [])
        sig = str(meta.get("content_sig") or "") or None
        embedder_sig = str(meta.get("embedder_sig") or "") or None
        if dim <= 0 or not ids:
            return False

        try:
            raw = vec_path.read_bytes()
        except OSError as e:  # pragma: no cover - 防御性
            logger.warning("vector data %s unreadable: %s", vec_path, e)
            return False

        vectors = array.array(_VEC_TYPECODE)
        try:
            vectors.frombytes(raw)
        except ValueError as e:  # pragma: no cover - 截断文件
            logger.warning("vector data %s corrupt: %s", vec_path, e)
            return False

        # 一致性校验：字节数必须正好是 len(ids) * dim 个 float。
        if len(vectors) != len(ids) * dim:
            logger.warning(
                "vector index %s inconsistent (%d floats for %d ids × %d dims); rebuilding",
                vec_path, len(vectors), len(ids), dim,
            )
            return False

        # metadata / texts 长度不足时补齐，避免索引越界。
        metadata += [{} for _ in range(len(ids) - len(metadata))]
        texts += ["" for _ in range(len(ids) - len(texts))]

        with self._lock:
            self.dim = dim
            self._ids = [str(i) for i in ids]
            self._metadata = [m if isinstance(m, dict) else {} for m in metadata[: len(ids)]]
            self._texts = [str(t) for t in texts[: len(ids)]]
            self._vectors = vectors
            self._pos = {rid: i for i, rid in enumerate(self._ids)}
            self._content_sig = sig
            self._embedder_sig = embedder_sig
            self._matrix_dirty = True
        logger.debug(
            "loaded %d vectors (dim=%d) for namespace '%s'",
            len(self._ids), dim, self.namespace,
        )
        return True

    def save(self) -> bool:
        """原子落盘。目录未配置 (纯内存模式) 时返回 False。"""
        vec_path, meta_path = self.vec_path, self.meta_path
        if vec_path is None or meta_path is None:
            return False
        with self._lock:
            if not self._ids:
                # 空索引：清掉残留文件，避免下次载入到陈旧数据。
                self._content_sig = None
                self._embedder_sig = None
                for p in (vec_path, meta_path):
                    try:
                        p.unlink()
                    except FileNotFoundError:
                        pass
                    except OSError as e:  # pragma: no cover - 防御性
                        logger.warning("cannot remove %s: %s", p, e)
                return True
            payload = {
                "schema_version": PERSIST_SCHEMA_VERSION,
                "namespace": self.namespace,
                "dim": self.dim,
                "count": len(self._ids),
                "ids": self._ids,
                "metadata": self._metadata,
                "texts": self._texts,
                # 整批内容签名：VectorIndex 据此在新进程跳过全量重嵌入。
                # 老索引无此字段时 load 恢复为 None，首次查询仍会重建一次。
                "content_sig": self._content_sig,
                # 生成这些向量的嵌入模型标识：VectorIndex 恢复短路状态前
                # 校验，防止换模型后跨进程复用旧模型向量 (1433 审核契约漏洞)。
                "embedder_sig": self._embedder_sig,
            }
            vec_bytes = self._vectors.tobytes()
            meta_text = json.dumps(payload, ensure_ascii=False)
        try:
            vec_path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write_bytes(vec_path, vec_bytes)
            _atomic_write_bytes(meta_path, meta_text.encode("utf-8"))
        except OSError as e:
            logger.warning("cannot persist vector index %s: %s", vec_path, e)
            return False
        return True

    # ---------- VectorStore 接口 ----------

    def upsert(self, records: Iterable[VectorRecord]) -> int:
        n = 0
        with self._lock:
            for r in records:
                if not r.vector:
                    continue
                if self.dim is None:
                    self.dim = len(r.vector)
                if len(r.vector) != self.dim:
                    # 维度不一致会让余弦退化，静默跳过 (与旧实现一致)。
                    continue
                idx = self._pos.get(r.id)
                if idx is None:
                    self._pos[r.id] = len(self._ids)
                    self._ids.append(r.id)
                    self._metadata.append(dict(r.metadata))
                    self._texts.append(r.text)
                    self._vectors.extend(float(x) for x in r.vector)
                else:
                    self._metadata[idx] = dict(r.metadata)
                    self._texts[idx] = r.text
                    base = idx * self.dim
                    for j, x in enumerate(r.vector):
                        self._vectors[base + j] = float(x)
                n += 1
            if n:
                self._matrix_dirty = True
        return n

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filter: Optional[dict[str, Any]] = None,
    ) -> list[VectorHit]:
        """精确余弦检索，可选 metadata 预过滤。

        过滤在**打分之前**执行 (predicate push-down) —— 这是 P4 严重问题 6 的
        修复基础：ACL 受限查询不必对全语料打分再截断。
        """
        with self._lock:
            if not query_vector or not self._ids or not self.dim:
                return []
            if len(query_vector) != self.dim:
                return []
            candidates = self._candidate_rows(filter)
            if not candidates:
                return []
            top_k = max(int(top_k), 0)
            if top_k == 0:
                return []
            if self._np is not None:
                scored = self._search_numpy(query_vector, candidates, top_k)
            else:
                scored = self._search_python(query_vector, candidates, top_k)
            return [
                VectorHit(
                    id=self._ids[i],
                    score=round(float(s), 6),
                    metadata=dict(self._metadata[i]),
                    text=self._texts[i],
                )
                for s, i in scored
            ]

    def _candidate_rows(self, filter: Optional[dict[str, Any]]) -> list[int]:
        """按 metadata 过滤出候选行号。

        过滤值为 list/set/tuple 时语义是「属于集合」(ACL 的允许集合)；
        标量时是精确相等。
        """
        if not filter:
            return list(range(len(self._ids)))
        rows: list[int] = []
        for i, meta in enumerate(self._metadata):
            ok = True
            for k, want in filter.items():
                got = meta.get(k)
                if isinstance(want, (list, set, tuple)):
                    if got not in want:
                        ok = False
                        break
                elif got != want:
                    ok = False
                    break
            if ok:
                rows.append(i)
        return rows

    def _ensure_matrix(self):
        """构建/复用 numpy 的 L2 归一化矩阵 (N × dim)。"""
        np = self._np
        if self._matrix is None or self._matrix_dirty:
            buf = np.frombuffer(
                self._vectors.tobytes(), dtype=np.float32
            ).reshape(len(self._ids), self.dim)
            norms = np.linalg.norm(buf, axis=1, keepdims=True)
            # 零向量的范数置 1，避免除零；其余保持单位化。
            norms[norms == 0.0] = 1.0
            self._matrix = (buf / norms).astype(np.float32)
            self._matrix_dirty = False
        return self._matrix

    def _search_numpy(self, query_vector, candidates, top_k) -> list[tuple[float, int]]:
        np = self._np
        matrix = self._ensure_matrix()
        q = np.asarray(query_vector, dtype=np.float32)
        qn = float(np.linalg.norm(q))
        if qn == 0.0:
            return []
        q = q / qn
        # 全量候选时跳过取行，直接整矩阵点乘 (最常见路径)。
        if len(candidates) == len(self._ids):
            sims = matrix @ q
            rows = None
        else:
            idx = np.asarray(candidates, dtype=np.int64)
            sims = matrix[idx] @ q
            rows = idx
        k = min(top_k, sims.shape[0])
        if k == 0:
            return []
        # argpartition 取 top-k 再排序：O(N) 而非 O(N log N)。
        part = np.argpartition(-sims, k - 1)[:k]
        part = part[np.argsort(-sims[part])]
        out: list[tuple[float, int]] = []
        for p in part:
            score = float(sims[p])
            if score <= 0.0:
                continue
            row = int(rows[p]) if rows is not None else int(p)
            out.append((score, row))
        return out

    def _search_python(self, query_vector, candidates, top_k) -> list[tuple[float, int]]:
        """无 numpy 时的兜底：纯 Python 余弦 (小语料够用)。"""
        qn = math.sqrt(sum(x * x for x in query_vector))
        if qn == 0.0:
            return []
        dim = self.dim
        vecs = self._vectors
        scored: list[tuple[float, int]] = []
        for i in candidates:
            base = i * dim
            dot = 0.0
            nn = 0.0
            for j in range(dim):
                v = vecs[base + j]
                dot += query_vector[j] * v
                nn += v * v
            if nn == 0.0:
                continue
            score = dot / (qn * math.sqrt(nn))
            if score > 0.0:
                scored.append((score, i))
        scored.sort(key=lambda x: -x[0])
        return scored[:top_k]

    def delete(self, ids: Iterable[str]) -> int:
        """删除记录。整体重建行数组 —— 删除是低频操作，换取存储紧凑。"""
        with self._lock:
            drop = {str(i) for i in ids}
            if not drop:
                return 0
            keep = [i for i, rid in enumerate(self._ids) if rid not in drop]
            removed = len(self._ids) - len(keep)
            if removed == 0:
                return 0
            if not keep:
                self._ids, self._metadata, self._texts = [], [], []
                self._vectors = array.array(_VEC_TYPECODE)
                self._pos = {}
                self._matrix_dirty = True
                return removed
            dim = self.dim or 0
            new_vec = array.array(_VEC_TYPECODE)
            for i in keep:
                new_vec.extend(self._vectors[i * dim:(i + 1) * dim])
            self._ids = [self._ids[i] for i in keep]
            self._metadata = [self._metadata[i] for i in keep]
            self._texts = [self._texts[i] for i in keep]
            self._vectors = new_vec
            self._pos = {rid: i for i, rid in enumerate(self._ids)}
            self._matrix_dirty = True
            return removed

    def count(self) -> int:
        with self._lock:
            return len(self._ids)

    def ids(self) -> list[str]:
        with self._lock:
            return list(self._ids)

    def get_content_sig(self) -> Optional[str]:
        """上次落盘的整批内容签名；纯内存或未恢复时为 None。"""
        with self._lock:
            return self._content_sig

    def set_content_sig(self, sig: Optional[str]) -> None:
        """记录当前内容的签名（由 VectorIndex 在成功重建后调用）。

        只改内存，随下一次 ``save()`` 落盘 —— 与向量数据同一原子写，
        保证签名与向量内容不会出现半更新的错配。
        """
        with self._lock:
            self._content_sig = sig

    def get_embedder_sig(self) -> Optional[str]:
        """生成当前向量所用的嵌入模型标识；未记录时为 None。"""
        with self._lock:
            return self._embedder_sig

    def set_embedder_sig(self, sig: Optional[str]) -> None:
        """记录生成这些向量的嵌入模型标识（由 VectorIndex 在重建后调用）。

        与 content_sig 同一原子写落盘；None 表示清除（空索引路径）。
        """
        with self._lock:
            self._embedder_sig = sig

    def close(self) -> None:
        """落盘并释放矩阵缓存。"""
        self.save()
        with self._lock:
            self._matrix = None


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """同目录临时文件 + rename 的原子写 (避免半截文件被下次启动读到)。"""
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        f.write(data)
        f.flush()
    tmp.replace(path)
