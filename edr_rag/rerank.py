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
#:
#: P4: 改为函数读取。旧实现在 import 时求值一次，运行时无法调整 ——
#: 长跑的服务进程想临时关掉精排只能重启。``RERANK_ENABLED`` 作为模块级
#: 常量保留（向后兼容既有 import），但检索链路统一走 ``rerank_enabled()``。
RERANK_ENABLED = os.environ.get("EDR_RAG_RERANK", "").strip().lower() != "0"


def rerank_enabled() -> bool:
    """当前是否默认启用精排（每次读环境变量，支持运行时调整）。"""
    return os.environ.get("EDR_RAG_RERANK", "").strip().lower() != "0"


def rrf_fuse(
    ranked_lists: list[list[dict]],
    key: str = "id",
    k: float = RRF_K,
    weights: Optional[list[float]] = None,
) -> list[dict]:
    """Reciprocal Rank Fusion 融合多路排序结果。

    Args:
        ranked_lists: 多路召回结果列表，每路内部已按相关度降序。
            每个元素是有 ``key`` 字段的 dict。
        key: 用于去重/合并的主键字段 (generic 语料用 'id'，
            动作富字段结果同样以 'id' = action_id 标识)。
        k: RRF 平滑常数，默认 60。
        weights: 每路召回的权重（与 ranked_lists 等长，非负）。缺省 None
            时各路等权（P1-6）。权重由评测集调节（见 edr_rag/eval/tune.py），
            例如 keyword 召回权重高于 bm25，或 embedding 可用时提高 vector
            权重。长度不匹配时忽略该参数退回等权。

    Returns:
        按 RRF 分数降序的融合结果列表 (去重，每路中各条取最高分)，
        其余字段来自第一次出现该 id 的那条。带 ``rrf_score`` 附加字段。
    """
    if not ranked_lists:
        return []

    if weights is None or len(weights) != len(ranked_lists):
        weights = [1.0] * len(ranked_lists)

    # 每条: id -> rrf score 累加
    scores: dict[str, float] = {}
    first: dict[str, dict] = {}
    for w, ranking in zip(weights, ranked_lists):
        if not ranking:
            continue
        for rank, item in enumerate(ranking):
            ident = item.get(key)
            if ident is None:
                continue
            scores[ident] = scores.get(ident, 0.0) + w / (k + rank + 1)
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


#: 精排文本的截断上限。cross-encoder 的输入窗口有限，但 120 字的 snippet
#: 远远不够 —— P4 让检索结果携带 ``_rerank_text`` (更完整的正文)。
RERANK_TEXT_MAX = int(os.environ.get("EDR_RAG_RERANK_TEXT_MAX", "1200"))


def rerank_text_for(r: dict) -> str:
    """取一条结果用于精排打分的文本。

    P4 修复：旧实现按 ``text / content / title / snippet`` 取值，而 generic
    语料的 ``_to_result`` 只产出 ``title`` 与 **120 字截断**的 ``snippet``。
    cross-encoder 拿 120 字判相关性，收益接近噪声。

    现在优先用 ``_rerank_text``（由 JsonlCorpus 注入的完整正文，见
    p1/search.py），并把标题拼在前面 —— 标题往往是最强的相关性信号。
    """
    title = str(r.get("title") or r.get("_title") or "").strip()
    body = str(
        r.get("_rerank_text")
        or r.get("text")
        or r.get("content")
        or r.get("snippet")
        or ""
    ).strip()
    parts = [p for p in (title, body) if p]
    text = " ".join(parts)
    return text[:RERANK_TEXT_MAX]


def rerank(
    query: str,
    results: list[dict],
    enabled: bool = True,
) -> tuple[list[dict], bool]:
    """对检索结果做 cross-encoder 精排。

    P4 修复（严重问题 7）：
      1. **不再因个别空文本放弃整批**。旧实现是
         ``if any(not t.strip() for t in texts): return results, False`` ——
         候选池里只要有一条空文本（动作目录里标题为空的记录并不罕见），
         整批精排就被跳过，reranker 形同虚设。现在只对有文本的条目精排，
         无文本的按原相对顺序**排在已精排结果之后**（无法打分 != 相关）。
      2. **用完整正文而非 120 字 snippet** 打分，见 ``rerank_text_for``。

    Args:
        query: 原始查询。
        results: 检索结果 (dict 列表)。
        enabled: 是否启用精排 (调用方开关)。默认 True。

    Returns:
        (reranked_results, did_rerank)。did_rerank=True 表示真正执行了
        cross-encoder 精排；否则 (开关关闭 / 可选依赖缺失 / 全部无文本)
        原顺序返回。
    """
    if not enabled or not results:
        return results, False

    reranker = _optional_reranker()
    if reranker is None:
        logger.debug("P1-3 reranker unavailable; skipping cross-encoder rerank")
        return results, False

    # 分成「可打分」与「无文本」两组，保留各自原相对顺序。
    scorable: list[tuple[int, dict, str]] = []
    unscorable: list[tuple[int, dict]] = []
    for i, r in enumerate(results):
        text = rerank_text_for(r)
        if text.strip():
            scorable.append((i, r, text))
        else:
            unscorable.append((i, r))

    if not scorable:
        # 整批都没有可打分文本 —— 这时精排确实无从谈起。
        logger.debug("no scorable text in candidate pool; skipping rerank")
        return results, False

    try:
        scores = reranker(query, [t for _, _, t in scorable])
    except Exception:  # pragma: no cover - optional path only
        logger.warning("P1-3 cross-encoder rerank failed; skipping", exc_info=True)
        return results, False

    if len(scores) != len(scorable):  # pragma: no cover - 防御性
        logger.warning(
            "reranker returned %d scores for %d candidates; skipping",
            len(scores), len(scorable),
        )
        return results, False

    ranked = sorted(
        zip(scorable, scores), key=lambda pair: -float(pair[1])
    )
    out = [
        {**r, "score": round(float(s), 4), "rerank_score": round(float(s), 4)}
        for (_, r, _), s in ranked
    ]
    # 无法打分的条目缀在后面，保持它们之间的原有顺序。
    out.extend(dict(r) for _, r in unscorable)
    return out, True
