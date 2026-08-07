"""P1 local JSONL search — keyword + BM25.

Runs over the artifacts written by P0 (sections.jsonl) and P1
(procedures.jsonl, chunks.jsonl). No vector DB. Deterministic.

Search strategies (design §5):
  - keyword: substring / token overlap on text + title + path
  - bm25: lightweight BM25 over the corpus (used when --scoring bm25)
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Optional

from edr_rag.query import normalize_query

logger = logging.getLogger(__name__)

# ---------- Tokenization (CJK-aware) ----------

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_WORD_RE = re.compile(r"[A-Za-z0-9_]+")
_SPLIT_RE = re.compile(r"[^\u4e00-\u9fffA-Za-z0-9_]+")


def tokenize(text: str) -> list[str]:
    """Split text into tokens: keep CJK chars individually (unigram-ish),
    latin words as whole tokens.

    P1-6: 先做与查询侧一致的 NFKC 规范化（全半角/大小写），保证查询 token
    与文档 token 对称，全角查询也能命中半角语料。
    """
    text = normalize_query(text)
    tokens: list[str] = []
    # Handle latin words + numbers
    for w in _WORD_RE.findall(text):
        tokens.append(w)
    # Handle CJK chars (char unigrams)
    for m in _CJK_RE.finditer(text):
        tokens.append(m.group(0))
    return tokens


# ---------- Optional embedding helpers (graceful degradation) ----------

_EMBEDDER_CACHE: dict[str, list[float]] = {}


def _optional_embedder() -> Optional[Callable[[str], list[float]]]:
    """Return an embedding function if the optional `rag` extras are present.

    Uses LlamaIndex's HuggingFaceEmbedding when importable; otherwise returns
    None and callers (vector_search) fall back to BM25. Never raises on a
    missing optional dependency — the main chain stays on keyword/BM25.
    """
    try:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # type: ignore
        from functools import lru_cache

        @lru_cache(maxsize=1)
        def _model():
            return HuggingFaceEmbedding(model_name="BAAI/bge-small-zh-v1.5")

        def _embed(text: str) -> list[float]:
            model = _model()
            vec = model.get_text_embedding(text)
            return list(vec)

        return _embed
    except Exception:  # pragma: no cover - optional deps absent
        logger.debug("optional RAG embedder unavailable; using BM25 fallback")
        return None


def _cached_vec(embed_fn: Callable[[str], list[float]], text: str) -> Optional[list[float]]:
    """Cache per-document embedding vectors (document text keyed by md5)."""
    import hashlib

    key = hashlib.md5(text.encode("utf-8", "ignore")).hexdigest()
    if key in _EMBEDDER_CACHE:
        return _EMBEDDER_CACHE[key]
    try:
        vec = embed_fn(text)
    except Exception:  # pragma: no cover - optional path only
        return None
    if vec:
        _EMBEDDER_CACHE[key] = vec
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


# ---------- Corpus ----------

class JsonlCorpus:
    """A searchable corpus over one or more JSONL artifact files."""

    def __init__(self, records: list[dict]):
        # Normalize each record into a searchable doc
        self.records = []
        for r in records:
            self.records.append(self._normalize(r))

        # Build BM25 index
        self._df: Counter = Counter()          # term -> #docs containing
        self._postings: dict[str, list[tuple[int, int]]] = {}  # term -> [(doc_idx, tf)]
        self._doc_len: list[int] = []
        self._avgdl = 0.0
        self._build_bm25()

    @staticmethod
    def from_file(path: Path) -> "JsonlCorpus":
        records = []
        with Path(path).open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return JsonlCorpus(records)

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

    def vector_search(self, query: str, limit: int = 5) -> list[dict]:
        """Embedding-based retrieval via optional dependencies.

        Uses LlamaIndex + a BGE reranker when the optional `rag` extras are
        installed. If they are missing, gracefully degrades to BM25 so the
        main retrieval chain never errors. This keeps the standalone keyword
        / BM25 path as the always-available fallback.
        """
        embed_fn = _optional_embedder()
        if embed_fn is None:
            # Degrade to BM25 when LlamaIndex / embedding model is absent —
            # main chain stays green without the optional deps.
            return self.bm25_search(query, limit=limit)

        try:
            q_vec = embed_fn(query)
            scored = []
            for i, r in enumerate(self.records):
                vec = _cached_vec(embed_fn, r["_searchable"])
                if vec is None:
                    continue
                score = _cosine(q_vec, vec)
                scored.append((score, i))
            scored.sort(key=lambda x: -x[0])
            return [self._to_result(i, s) for s, i in scored[:limit]]
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
