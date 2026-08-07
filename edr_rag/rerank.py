"""P1-3 本地 reranker（cross-encoder）与多路召回融合（RRF）。

设计目标（对应 P1-3 desc）:
  检索第二阶段加 cross-encoder rerank（如 BAAI/bge-reranker-base），作为
  可选依赖。整体检索流程为:

      多路召回 -> RRF 融合 -> rerank -> top K

  本模块提供:
    - rrf_fuse:          Reciprocal Rank Fusion，把多路排序结果(按 id)融合
                         成一份综合排名 (确定性，无需任何可选依赖)。
    - rerank:            cross-encoder 精排；可选依赖缺失时优雅跳过，返回
                         原顺序 (主链路始终可用)。
    - available() / _optional_reranker: 探测 reranker 是否可用。

降级策略 (与 vector 的 embedding 一致):
  - bge-reranker 的依赖 (FlagEmbedding / sentence-transformers / torch) 是
    可选 `rag` extras；缺依赖时 rerank 直接跳过，保留 RRF 融合结果。
  - 绝不抛出异常，主链路 (keyword/BM25/RRF) 始终可用。

开关 (关闭开关):
  - 在调用检索时传 rerank=False 可关闭精排。
  - 亦可通过环境变量 EDR_RAG_RERANK=0 全局关闭 (默认 "auto"：可用即启用)。
"""

from __future__ import annotations

import logging
import os
from typing import Callable, Optional

logger = logging.getLogger(__name__)

#: RRF 融合的平滑常数 k (BM25 论文常用 60)。
RRF_K = 60.0

#: 开关: "0" -> 全局关闭精排；其它非空值或缺失 -> auto (可用即启用)。
_ENV_RERANK = os.environ.get("EDR_RAG_RERANK", "").strip().lower()
RERANK_ENABLED = _ENV_RERANK != "0"


def rrf_fuse(
    ranked_lists: list[list[dict]],
    key: str = "id",
    k: float = RRF_K,
) -> list[dict]:
    """Reciprocal Rank Fusion 融合多路排序结果。

    Args:
        ranked_lists: 多路召回结果列表，每路内部已按相关度降序。
            每个元素是有 ``key`` 字段的 dict。
        key: 用于去重/合并的主键字段 (generic 语料用 'id'，
            动作富字段结果同样以 'id' = action_id 标识)。
        k: RRF 平滑常数，默认 60。

    Returns:
        按 RRF 分数降序的融合结果列表 (去重，每路中各条取最高分)，
        其余字段来自第一次出现该 id 的那条。带 ``rrf_score`` 附加字段。
    """
    if not ranked_lists:
        return []

    # 每条: id -> rrf score 累加
    scores: dict[str, float] = {}
    first: dict[str, dict] = {}
    for ranking in ranked_lists:
        if not ranking:
            continue
        for rank, item in enumerate(ranking):
            ident = item.get(key)
            if ident is None:
                continue
            scores[ident] = scores.get(ident, 0.0) + 1.0 / (k + rank + 1)
            if ident not in first:
                first[ident] = item

    fused = [
        {**first[ident], "rrf_score": round(sc, 4)}
        for ident, sc in sorted(scores.items(), key=lambda kv: -kv[1])
    ]
    return fused


# ---------- optional cross-encoder reranker ----------
#: 缓存检测结果 (惰性探测一次): checked=True 后，fn 即为最终值或 None。
_reranker_checked = False
_reranker_fn: Optional[Callable[[str, list[str]], list[float]]] = None


def _optional_reranker() -> Optional[Callable[[str, list[str]], list[float]]]:
    """若可选 `rag` extras 可用，返回 ``(query, texts) -> scores`` 函数。

    依次尝试:
      1. FlagEmbedding.FlagReranker (bge-reranker 官方实现)
      2. llama_index cross-encoder (HuggingFaceCrossEncoder)

    缺所有可选依赖时返回 None，调用方跳过精排。绝不抛出。
    """
    global _reranker_checked, _reranker_fn
    if _reranker_checked:
        return _reranker_fn

    fn: Optional[Callable[[str, list[str]], list[float]]] = None

    # 1) FlagEmbedding.FlagReranker
    try:
        from FlagEmbedding import FlagReranker  # type: ignore

        _model = FlagReranker("BAAI/bge-reranker-base", use_fp16=False)

        def _via_flag(query: str, texts: list[str]) -> list[float]:
            pairs = [[query, t] for t in texts]
            scores = _model.compute_score(pairs)
            if isinstance(scores, float):
                scores = [scores]
            return [float(s) for s in scores]

        fn = _via_flag
        logger.debug("P1-3 flag reranker available (FlagEmbedding)")
    except Exception:
        pass

    # 2) llama_index HuggingFaceCrossEncoder / SentenceTransformerRerank
    if fn is None:
        try:
            from llama_index.postprocessor import SentenceTransformerRerank  # type: ignore

            _model = SentenceTransformerRerank(
                model="BAAI/bge-reranker-base", top_n=1
            )

            def _via_li(query: str, texts: list[str]) -> list[float]:
                from llama_index.core.schema import NodeWithScore, TextNode

                nodes = [NodeWithScore(node=TextNode(text=t), score=0.0) for t in texts]
                scored = _model._postprocess_nodes(  # type: ignore[attr-defined]
                    nodes, query_str=query
                )
                by_text = {n.node.get_content(): n.score for n in scored}
                return [float(by_text.get(t, 0.0)) for t in texts]

            fn = _via_li
            logger.debug("P1-3 reranker available (llama_index SentenceTransformerRerank)")
        except Exception:
            pass

    _reranker_checked = True
    _reranker_fn = fn
    return fn


def available() -> bool:
    """cross-encoder reranker 当前是否可用。"""
    return _optional_reranker() is not None


def rerank(
    query: str,
    results: list[dict],
    enabled: bool = True,
) -> tuple[list[dict], bool]:
    """对检索结果做 cross-encoder 精排。

    Args:
        query: 原始查询。
        results: 检索结果 (dict 列表，需含可展示文本字段)。
        enabled: 是否启用精排 (调用方开关)。默认 True。

    Returns:
        (reranked_results, did_rerank)。did_rerank=True 表示真正执行了
        cross-encoder 精排；否则 (开关关闭或可选依赖缺失) 原顺序返回。
    """
    if not enabled or not results:
        return results, False

    reranker = _optional_reranker()
    if reranker is None:
        logger.debug("P1-3 reranker unavailable; skipping cross-encoder rerank")
        return results, False

    # 抽取每条的可展示文本用于打分，优先用富文本字段，回落到 title/snippet。
    texts: list[str] = []
    for r in results:
        text = (
            r.get("text")
            or r.get("content")
            or r.get("title")
            or r.get("snippet")
            or ""
        )
        texts.append(text)

    # 空文本一律跳过精排 (无法打分)，保留原顺序。
    if any(not t.strip() for t in texts):
        return results, False

    try:
        scores = reranker(query, texts)
        scored = sorted(
            zip(results, scores), key=lambda pair: -pair[1]
        )
        out = [
            {**r, "score": round(float(s), 4), "rerank_score": round(float(s), 4)}
            for r, s in scored
        ]
        return out, True
    except Exception:  # pragma: no cover - optional path only
        logger.warning("P1-3 cross-encoder rerank failed; skipping", exc_info=True)
        return results, False
