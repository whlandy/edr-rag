"""P1 local JSONL search — keyword + BM25.

Runs over the artifacts written by P0 (sections.jsonl) and P1
(procedures.jsonl, chunks.jsonl). No vector DB. Deterministic.

Search strategies (design §5):
  - keyword: substring / token overlap on text + title + path
  - bm25: lightweight BM25 over the corpus (used when --scoring bm25)
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import threading
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Callable, Optional

from edr_rag.query import normalize_query

logger = logging.getLogger(__name__)

# ---------- Tokenization (CJK-aware) ----------

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]+")
_WORD_RE = re.compile(r"[A-Za-z0-9_]+")
_SPLIT_RE = re.compile(r"[^\u4e00-\u9fffA-Za-z0-9_]+")


#: P4 Lock 4 —— 是否产出 CJK bigram。可用 EDR_RAG_CJK_BIGRAM=0 关闭以回到
#: P1 的纯 unigram 行为 (仅用于对照实验，不建议生产关闭)。
CJK_BIGRAM = os.environ.get("EDR_RAG_CJK_BIGRAM", "1") != "0"

#: jieba 词典分词探测结果 (模块级单例，语义同 _optional_embedder)。
_jieba_checked = False
_jieba_cut = None


def _optional_jieba():
    """返回 jieba 的切词函数；未安装时返回 None (bigram 已能兜住主要收益)。"""
    global _jieba_checked, _jieba_cut
    if _jieba_checked:
        return _jieba_cut
    try:
        import jieba  # type: ignore

        # search 模式产出更细粒度的重叠切分，利于召回。
        _jieba_cut = jieba.lcut_for_search
    except Exception:  # pragma: no cover - optional dep absent
        _jieba_cut = None
    _jieba_checked = True
    return _jieba_cut


def reset_tokenizer_cache() -> None:
    """重置 jieba 探测缓存 (测试用)。"""
    global _jieba_checked, _jieba_cut
    _jieba_checked = False
    _jieba_cut = None


def tokenize(text: str) -> list[str]:
    """把文本切成检索 token：拉丁词整词 + CJK unigram + CJK bigram。

    P1-6: 先做与查询侧一致的 NFKC 规范化（全半角/大小写），保证查询 token
    与文档 token 对称，全角查询也能命中半角语料。

    P4 Lock 4 —— 加 bigram。纯 unigram 的中文 BM25 有个经典缺陷：词序信息
    完全丢失，「安全」与「全安」的 token 集合完全相同因而同分，「网络代理」
    与「代理网络」也无法区分。bigram 是**纯 stdlib** 就能拿到大部分收益的
    方案，不给内网离线部署增加任何依赖。

    bigram 只在**连续 CJK 片段内部**生成，不跨标点/拉丁字符组合，避免
    「代理。配置」被拼出无意义的「理配」。

    jieba 可用时额外叠加词典切分的长词（>2 字），进一步提升领域词精度；
    不可用时静默跳过。

    Returns:
        token 列表，顺序为 [拉丁词..., CJK unigram..., CJK bigram..., jieba 长词...]。
        保留重复项 —— BM25 需要词频。
    """
    text = normalize_query(text)
    tokens: list[str] = []
    # 拉丁词 + 数字整词
    for w in _WORD_RE.findall(text):
        tokens.append(w)
    # CJK 单字 unigram（保底召回）
    for m in _CJK_RE.finditer(text):
        tokens.append(m.group(0))
    # CJK bigram（词序/搭配精度），仅在连续汉字片段内部滑窗
    if CJK_BIGRAM:
        for run in _CJK_RUN_RE.findall(text):
            for i in range(len(run) - 1):
                tokens.append(run[i:i + 2])
    # 可选：jieba 词典切分，补充 2 字以上的领域词
    cut = _optional_jieba()
    if cut is not None:
        try:
            for w in cut(text):
                w = w.strip()
                # 单字已由 unigram 覆盖、2 字已由 bigram 覆盖，这里只补更长的词
                if len(w) > 2 and _CJK_RE.search(w):
                    tokens.append(w)
        except Exception:  # pragma: no cover - optional path
            logger.debug("jieba tokenization failed; using bigram tokens only")
    return tokens


# ---------- Optional embedding helpers (graceful degradation) ----------

#: 嵌入模型名。可用 EDR_RAG_EMBED_MODEL 覆盖 (离线部署指向本地模型目录)。
EMBED_MODEL = os.environ.get("EDR_RAG_EMBED_MODEL", "BAAI/bge-small-zh-v1.5")

#: 文档向量缓存上限。旧实现是无界 dict —— 长跑进程上是内存泄漏。
_EMBED_CACHE_MAXSIZE = int(os.environ.get("EDR_RAG_EMBED_CACHE_SIZE", "20000"))
_EMBEDDER_CACHE: "OrderedDict[str, list[float]]" = OrderedDict()
_EMBED_CACHE_LOCK = threading.Lock()

#: P4 关键修复 —— 模块级单例。
#:
#: 旧实现把 ``@lru_cache`` 定义在 ``_optional_embedder()`` **函数体内部**，
#: 于是每次调用都生成全新闭包 + 全新缓存，BGE 模型被反复从磁盘装载。实测
#: 每次 ~9–10 秒，而 retrieval.py 每个查询要调 1–2 次。这是 P4 的头号性能
#: 阻塞项 (见 P4-HARDENING-DESIGN.md §4 Lock 2)。
#:
#: 契约：同一进程内 ``_optional_embedder()`` 必须返回**同一个函数对象**。
#: 由 tests/test_embedder_singleton.py 锁死。
_embedder_checked = False
_embedder_fn: Optional[Callable[[str], list[float]]] = None
_embedder_model = None
_embedder_lock = threading.Lock()


def _load_embed_model():
    """惰性构造嵌入模型 (进程内仅一次)。"""
    global _embedder_model
    if _embedder_model is None:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # type: ignore

        _embedder_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
    return _embedder_model


def _optional_embedder() -> Optional[Callable[[str], list[float]]]:
    """Return an embedding function if the optional `rag` extras are present.

    Uses LlamaIndex's HuggingFaceEmbedding when importable; otherwise returns
    None and callers (vector_search) fall back to BM25. Never raises on a
    missing optional dependency — the main chain stays on keyword/BM25.

    P4: 探测结果与模型实例均为模块级单例，重复调用零开销。
    """
    global _embedder_checked, _embedder_fn
    if _embedder_checked:
        return _embedder_fn
    with _embedder_lock:
        if _embedder_checked:  # 双检锁：等锁期间可能已被别的线程完成探测
            return _embedder_fn
        try:
            # 只探测可导入性，模型本身留到首次真正嵌入时再构造，
            # 避免「只是问一句 embedder 在不在」就付出装载成本。
            import llama_index.embeddings.huggingface  # type: ignore  # noqa: F401

            def _embed(text: str) -> list[float]:
                vec = _load_embed_model().get_text_embedding(text)
                return list(vec)

            _embedder_fn = _embed
        except Exception:  # pragma: no cover - optional deps absent
            logger.debug("optional RAG embedder unavailable; using BM25 fallback")
            _embedder_fn = None
        _embedder_checked = True
    return _embedder_fn


def reset_embedder_cache() -> None:
    """重置嵌入探测与向量缓存 (测试用；生产不应调用)。"""
    global _embedder_checked, _embedder_fn, _embedder_model
    with _embedder_lock:
        _embedder_checked = False
        _embedder_fn = None
        _embedder_model = None
    with _EMBED_CACHE_LOCK:
        _EMBEDDER_CACHE.clear()


def _cached_vec(embed_fn: Callable[[str], list[float]], text: str) -> Optional[list[float]]:
    """Cache per-document embedding vectors (LRU, keyed by content hash)."""
    key = hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()
    with _EMBED_CACHE_LOCK:
        vec = _EMBEDDER_CACHE.get(key)
        if vec is not None:
            _EMBEDDER_CACHE.move_to_end(key)
            return vec
    try:
        vec = embed_fn(text)
    except Exception:  # pragma: no cover - optional path only
        return None
    if vec:
        with _EMBED_CACHE_LOCK:
            _EMBEDDER_CACHE[key] = vec
            _EMBEDDER_CACHE.move_to_end(key)
            while len(_EMBEDDER_CACHE) > _EMBED_CACHE_MAXSIZE:
                _EMBEDDER_CACHE.popitem(last=False)
    return vec


def _cosine(a: list[float], b: list[float]) -> float:
    import math as _m

    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = _m.sqrt(sum(x * x for x in a))
    nb = _m.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


#: 进程内共享的 VectorIndex，**按 (namespace, 持久化目录) 缓存** (P4)。
#:
#: P3-1 的实现按 ``id(store)`` 缓存，而 ``get_vector_store()`` 全进程只返回
#: 一个 store —— 于是所有语料共用同一个 VectorIndex。``index_records`` 按整批
#: 内容签名短路，一换语料签名就不匹配，触发 ``delete(全部) + 全量重嵌``。
#: ``planner_retrieve`` 逐层走 actions → procedures → sections，三层会互相
#: 清空对方的向量，vector 策略下每层都要重嵌一次整个语料。
#:
#: P4 让每个语料拿到独立 namespace 的 store 与独立 VectorIndex。
_INDEX_CACHE: dict[tuple[str, str], Any] = {}
_INDEX_CACHE_LOCK = threading.Lock()


def _get_shared_index(
    embed_fn: Callable[[str], list[float]],
    namespace: str = "default",
    directory: Optional[Path] = None,
):
    """返回该 namespace 的进程内共享 VectorIndex，惰性创建。

    Args:
        embed_fn: 嵌入函数。
        namespace: 语料名 —— 隔离边界，不同语料必须不同。
        directory: 持久化目录；None 表示纯内存 (无 workdir 的调用方)。
    """
    from edr_rag.vectorstore import VectorIndex, get_vector_store

    dir_key = str(directory) if directory is not None else ""
    key = (namespace, dir_key)
    with _INDEX_CACHE_LOCK:
        idx = _INDEX_CACHE.get(key)
        if idx is None:
            cfg: dict[str, Any] = {}
            if directory is not None:
                cfg["directory"] = str(directory)
            else:
                # 无持久化目录时退回纯进程内实现，避免在 CWD 里乱写文件。
                cfg["backend"] = "simple"
            store = get_vector_store(namespace=namespace, config=cfg)
            idx = VectorIndex(store, embed_fn)
            _INDEX_CACHE[key] = idx
        return idx


def reset_index_cache() -> None:
    """清空 VectorIndex 缓存 (测试用；生产不应调用)。"""
    with _INDEX_CACHE_LOCK:
        _INDEX_CACHE.clear()


# ---------- Corpus ----------

class JsonlCorpus:
    """A searchable corpus over one or more JSONL artifact files.

    P4: 语料携带 ``namespace`` 与 ``vector_dir``，供向量层做隔离与持久化。
    ``namespace`` 由来源文件名推导 (sections.jsonl -> 'sections')；从裸记录
    构造时为 None，此时向量检索走纯内存、不落盘 (测试与轻量调用方)。
    """

    def __init__(
        self,
        records: list[dict],
        namespace: Optional[str] = None,
        vector_dir: Optional[Path] = None,
    ):
        # Normalize each record into a searchable doc
        self.records = []
        for r in records:
            self.records.append(self._normalize(r))

        self.namespace = namespace
        self.vector_dir = vector_dir

        # Build BM25 index
        self._df: Counter = Counter()          # term -> #docs containing
        self._postings: dict[str, list[tuple[int, int]]] = {}  # term -> [(doc_idx, tf)]
        self._doc_len: list[int] = []
        self._avgdl = 0.0
        self._build_bm25()

    @staticmethod
    def from_file(path: Path) -> "JsonlCorpus":
        """从 JSONL 构造语料，并由文件名/位置推导向量 namespace 与持久化目录。"""
        path = Path(path)
        records = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        # sections.jsonl -> 'sections'；ocr.jsonl -> 'ocr'。
        namespace = path.name[: -len(".jsonl")] if path.name.endswith(".jsonl") else path.stem
        from edr_rag.vectorstore import vector_dir_for_workdir

        return JsonlCorpus(
            records,
            namespace=namespace or None,
            vector_dir=vector_dir_for_workdir(path.parent),
        )

    @staticmethod
    def _normalize(r: dict) -> dict:
        """Merge title/path/text into a searchable 'searchable' field."""
        text = r.get("text") or r.get("content") or ""
        title = r.get("title") or ""
        section_path = r.get("section_path") or []
        paths = " ".join(section_path) if section_path else ""
        source = r.get("source") or {}
        file = source.get("file") or r.get("source_file") or ""

        # Chunk records keep module/section context inside `metadata`
        # (built by P1 build_chunks). Fold that in so queries relying on
        # module/module-path context can still match.
        meta = r.get("metadata") or {}
        meta_paths = meta.get("section_path") or []
        if isinstance(meta_paths, str):
            meta_paths = [meta_paths]
        meta_title = meta.get("title") or ""
        module = meta.get("module") or ""
        version = meta.get("version") or ""
        expected_result = meta.get("expected_result") or ""
        folded_meta = " ".join(
            [
                meta_title,
                module,
                version,
                expected_result,
                " ".join(meta_paths),
                meta.get("operation") or "",
            ]
        )

        r["_searchable"] = normalize_query(" ".join([title, paths, file, folded_meta, text]))
        r["_title"] = title or meta_title
        r["_doc_id"] = r.get("id") or ""
        return r

    def _build_bm25(self):
        for i, r in enumerate(self.records):
            toks = tokenize(r["_searchable"])
            self._doc_len.append(len(toks))
            counts = Counter(toks)
            for term, tf in counts.items():
                self._df[term] += 1
                self._postings.setdefault(term, []).append((i, tf))
        total = sum(self._doc_len)
        self._avgdl = total / len(self._doc_len) if self._doc_len else 0.0

    def keyword_search(self, query: str, limit: int = 5) -> list[dict]:
        """Simple substring / token-overlap scoring."""
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        scores: list[tuple[float, int]] = []
        for i, r in enumerate(self.records):
            field = r["_searchable"]
            score = 0.0
            for t in q_tokens:
                if t in field:
                    # title+path matches weighted higher
                    weight = 3.0 if (t in r["_title"] or t in r.get("_searchable", "")[:200]) else 1.0
                    score += field.count(t) * weight
            if score > 0:
                scores.append((score, i))
        scores.sort(key=lambda x: -x[0])
        return [self._to_result(i, s) for s, i in scores[:limit]]

    def bm25_search(self, query: str, limit: int = 5, k1: float = 1.5, b: float = 0.75) -> list[dict]:
        q_tokens = set(tokenize(query))
        if not q_tokens:
            return []
        n = len(self.records)
        idf_cache = {}
        for t in q_tokens:
            df = self._df.get(t, 0)
            idf_cache[t] = 0.0 if df == 0 else math.log((n - df + 0.5) / (df + 0.5) + 1)
        scores: dict[int, float] = {}
        for t in q_tokens:
            idf = idf_cache[t]
            if idf == 0:
                continue
            for doc_idx, tf in self._postings.get(t, []):
                dl = self._doc_len[doc_idx]
                tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (dl / self._avgdl))) if self._avgdl else 0.0
                scores[doc_idx] = scores.get(doc_idx, 0.0) + idf * tf_norm
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return [self._to_result(i, s) for i, s in ranked[:limit]]

    def vector_search(
        self,
        query: str,
        limit: int = 5,
        filter: Optional[dict[str, Any]] = None,
    ) -> list[dict]:
        """Embedding-based retrieval via the pluggable vector store (P3-1).

        Uses LlamaIndex + a BGE embedder when the optional `rag` extras are
        installed, indexing this corpus's records into the configured vector
        backend (default: PersistentVectorStore, on-disk exact cosine). If the
        optional deps are missing, gracefully degrades to BM25 so the main
        retrieval chain never errors. The configured backend is determined by
        EDR_RAG_VECTOR_BACKEND (see edr_rag/vectorstore).

        Args:
            filter: 可选 metadata 谓词，在**打分之前**下推到存储层
                (P4 ACL 谓词下推的基础)。值为 list/set/tuple 时语义是
                「属于集合」，标量时是精确相等。
        """
        embed_fn = _optional_embedder()
        if embed_fn is None:
            # Degrade to BM25 when LlamaIndex / embedding model is absent —
            # main chain stays green without the optional deps.
            return self.bm25_search(query, limit=limit)

        try:
            index = _get_shared_index(
                embed_fn,
                namespace=self.namespace or "default",
                directory=self.vector_dir if self.namespace else None,
            )
            indexed = index.index_records(
                self.records,
                text_field="_searchable",
                id_field="_doc_id",
            )
            # 只有真正发生写入时才落盘 —— index_records 的签名短路命中时
            # (常见路径) 不做任何 IO。
            if indexed and self.vector_dir is not None:
                index.persist()
            hits = index.search(query, top_k=limit, filter=filter)
            return [self._to_result(doc_idx, score) for score, doc_idx in hits]
        except Exception:  # pragma: no cover - optional path only
            logger.warning("vector_search failed; falling back to BM25", exc_info=True)
            return self.bm25_search(query, limit=limit)

    def _to_result(self, doc_idx: int, score: float) -> dict:
        r = self.records[doc_idx]
        return {
            "id": r["_doc_id"],
            "type": r.get("type", "section"),
            "score": round(score, 3),
            "title": r["_title"],
            "source": r.get("source") or {
                "file": r.get("source_file", ""),
            },
            "snippet": self._snippet(r),
        }

    def _snippet(self, r: dict, max_len: int = 120) -> str:
        text = r.get("text") or r.get("content") or ""
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) <= max_len:
            return text
        return text[:max_len] + "..."
