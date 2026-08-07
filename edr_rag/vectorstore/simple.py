"""本地内存向量存储 —— 默认实现 (P3-1)。

SimpleVectorStore 是 VectorStore 的默认本地实现: 把所有记录缓存在进程内
dict 中，检索时做暴力余弦打分。它不依赖任何第三方库 (纯 stdlib)，因此
是主链路 (无外部服务、无可选依赖) 下 vector 检索的兜底后端。

与既有 p1.search.vector_search 行为一致: 内存余弦、按相关度降序、
score>0 才保留；差别在于这里把「向量库」从检索逻辑中抽象成了独立组件，
便于日后无痛切到外部后端。
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Optional

from .base import VectorHit, VectorRecord, VectorStore


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    denom = math.sqrt(na) * math.sqrt(nb)
    if denom == 0:
        return 0.0
    return dot / denom


class SimpleVectorStore(VectorStore):
    """进程内内存向量存储，暴力余弦检索。

    线程安全不作保证 (单进程检索场景足够)；写入即替换同 id 旧记录。
    """

    name = "simple"

    def __init__(self, dim: int | None = None):
        self.dim = dim
        self._records: dict[str, VectorRecord] = {}

    def upsert(self, records: Iterable[VectorRecord]) -> int:
        n = 0
        for r in records:
            if not r.vector:
                continue
            if self.dim is None:
                self.dim = len(r.vector)
            if self.dim and len(r.vector) != self.dim:
                continue  # 维度不一致的记录跳过，避免余弦退化
            self._records[r.id] = r
            n += 1
        return n

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filter: Optional[dict[str, Any]] = None,
    ) -> list[VectorHit]:
        if not query_vector:
            return []
        scored = []
        for r in self._records.values():
            if filter:
                skip = False
                for k, v in filter.items():
                    if r.metadata.get(k) != v:
                        skip = True
                        break
                if skip:
                    continue
            score = _cosine(query_vector, r.vector)
            if score > 0:
                scored.append((score, r))
        scored.sort(key=lambda x: -x[0])
        return [
            VectorHit(id=r.id, score=round(s, 6), metadata=dict(r.metadata), text=r.text)
            for s, r in scored[:top_k]
        ]

    def delete(self, ids: Iterable[str]) -> int:
        n = 0
        for i in ids:
            if i in self._records:
                del self._records[i]
                n += 1
        return n

    def count(self) -> int:
        return len(self._records)

    def ids(self) -> list[str]:
        return list(self._records.keys())
