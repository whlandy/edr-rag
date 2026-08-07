"""统一 RetrievalService — 收敛所有检索入口 (P0-2).

设计目标:
  - 新建 edr_rag/retrieval.py，提供统一的 RetrievalService，作为纯字 API
    (EdrRagAPI)、CLI (p2/cli.py)、MCP (EdrRagAgent) 的唯一检索入口。
  - 禁止调用方各自直接 new JsonlCorpus 或直接读 JSONL；全部通过
    RetrievalService.search(...) 走同一套 corpora + 打分 + 过滤逻辑。

支持的 corpus_type:
  - sections   -> sections.jsonl   (P0 章节/手册正文)
  - procedures -> procedures.jsonl (P1 流程)
  - actions    -> actions.jsonl    (P2 动作目录，返回富字段)
  - chunks     -> chunks.jsonl     (P1 步骤分片)

搜索策略 strategy: 'keyword' | 'bm25' | 'vector' | 'hybrid'
  - keyword: 子串/词元重叠打分，动作对 title/action_id 加权
  - bm25:    轻量 BM25
  - vector:  可选 RAG extras 的向量检索；缺依赖时优雅降级到 bm25
             (主链路始终可用 keyword/BM25)
  - hybrid:  未来融合策略占位；当前实现先用 bm25（保证所有入口即使
             指定 hybrid 也绝不报错，后续在此处接入 keyword+vector 融合）。

planner_retrieve 保持既有的 action -> procedure -> manual 优先级，
并把调用方传入的 strategy 透传给每一层检索。
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Optional

from .p1.search import JsonlCorpus

# 允许的 corpus_type -> 对应 JSONL 文件名
_CORPUS_FILES: dict[str, str] = {
    "sections": "sections.jsonl",
    "procedures": "procedures.jsonl",
    "actions": "actions.jsonl",
    "chunks": "chunks.jsonl",
}

_ALLOWED_STRATEGIES = ("keyword", "bm25", "vector", "hybrid")
# hybrid 为未来融合策略占位：当前实现落到 bm25（对齐 vector 缺依赖时的降级行为），
# 后续接入 keyword+vector 融合时只需改进 _resolve_strategy。
_DEFAULT_STRATEGY = "keyword"

#: 缓存签名类型：用文件 mtime(纳秒)+size 做廉价失效判断，避免每次调用重读 JSONL。
#: mtime/size 任一变化即视为文件已重建，触发重载并重建内存索引。
_CacheSig = tuple[int, int] | None  # (st_mtime_ns, st_size)；文件缺失为 None


def _file_signature(path: Path) -> _CacheSig:
    """返回文件 (mtime_ns, size) 签名；文件不存在时返回 None。"""
    try:
        st = path.stat()
    except FileNotFoundError:
        return None
    return (st.st_mtime_ns, st.st_size)


class RetrievalService:
    """统一检索服务。

    由 workdir 懒加载各 corpus 并缓存；对外仅暴露 search / get /
    planner_retrieve 三个检索入口。
    """

    def __init__(self, workdir: str | Path):
        self.workdir = Path(workdir)
        # 每个 corpus_type 缓存 None(未加载/文件缺失) 或 (签名, corpus)。
        self._corpora: dict[str, tuple[_CacheSig, JsonlCorpus] | None] = {
            "sections": None,
            "procedures": None,
            "actions": None,
            "chunks": None,
        }

    # ---------- Corpus 懒加载 ----------

    def _corpus(self, corpus_type: str) -> Optional[JsonlCorpus]:
        """按 corpus_type 返回缓存 corpus；文件不存在或为空时返回 None。

        P0-6: 以文件 (mtime_ns, size) 签名做失效判断——同一服务实例多次检索
        时不重复读文件/重建索引；一旦 JSONL 被重建 (例如新跑一遍 ingest)，
        签名变化即自动重载，无需重建整个服务。
        """
        if corpus_type not in _CORPUS_FILES:
            raise ValueError(
                f"unknown corpus_type '{corpus_type}'; "
                f"expected one of {sorted(_CORPUS_FILES)}"
            )
        path = self.workdir / _CORPUS_FILES[corpus_type]
        sig = _file_signature(path)

        entry = self._corpora.get(corpus_type)
        if entry is not None and entry[0] == sig:
            # 命中：签名未变，直接复用内存中的 corpus (含已建好的索引)。
            return entry[1]

        if sig is None:
            # 文件缺失 (或已删除)：不缓存 corpus，返回 None。
            self._corpora[corpus_type] = None
            return None

        # 首次加载或签名变化 -> 重建 corpus 及其内存索引。
        corpus = JsonlCorpus.from_file(path)
        self._corpora[corpus_type] = (sig, corpus)
        return corpus

    # ---------- 搜索 ----------

    def search(
        self,
        query: str,
        corpus_type: str = "actions",
        strategy: str = _DEFAULT_STRATEGY,
        filters: Optional[dict[str, Any]] = None,
        top_k: int = 5,
        recall_budget: int = 4,
    ) -> list[dict]:
        """统一检索入口。

        Args:
            query: 检索词 (支持中文)。
            corpus_type: 'sections' | 'procedures' | 'actions' | 'chunks'。
            strategy: 'keyword' | 'bm25' | 'vector' | 'hybrid'；
                vector/hybrid 缺依赖或为占位时降级 bm25。
            filters: dict，支持:
                - 'risk':   动作过滤 (low/medium/high)
                - 'status': 动作过滤 (candidate/needs_review/approved/published)
                - 'id':     精确匹配 id（actions 用 action_id）做获取式查询
            top_k: 返回条数上限。
            recall_budget: 动作检索的召回预算倍数 (P0-4)。先扩大候选池
                (top_k * recall_budget) 打分排序，再对候选池做 metadata
                过滤；结果不足时循环扩大候选池直到满足或耗尽语料。该倍数
                只影响候选池大小，不改变最终返回条数 (仍 ≤ top_k)。

        Returns:
            结果 dict 列表。actions 返回富字段
            (id/title/risk/status/confidence/score/snippet)；
            其余类型返回 JsonlCorpus 的标准结果
            (id/type/score/title/source/snippet)。
        """
        filters = filters or {}
        top_k = max(int(top_k), 0)
        recall_budget = max(int(recall_budget), 1)

        # 精确 id 获取模式：不做全文打分，直接按主键取。
        if filters.get("id") is not None:
            rec = self.get(corpus_type, filters["id"])
            if rec is None:
                return []
            if corpus_type == "actions":
                # 动作返回统一富字段结果 (id = action_id)。
                return [self._record_to_action_result(rec)]
            return [dict(rec)]

        if strategy not in _ALLOWED_STRATEGIES:
            raise ValueError(
                f"unknown strategy '{strategy}'; expected one of {_ALLOWED_STRATEGIES}"
            )

        strategy = self._resolve_strategy(strategy)

        if corpus_type == "actions":
            results = self._search_actions(
                query,
                strategy=strategy,
                filters=filters,
                top_k=top_k,
                recall_budget=recall_budget,
            )
        else:
            corpus = self._corpus(corpus_type)
            if corpus is None:
                return []
            results = self._search_generic(corpus, query, strategy, top_k)
            results = _apply_filter(results, filters)

        return results

    @staticmethod
    def _resolve_strategy(strategy: str) -> str:
        """把可能降级的策略解析为实际执行的检索策略。

        - 'hybrid' 当前为占位：先落到 bm25（保证主链路可用），
          后续接入 keyword+vector 融合时在这里扩展。
        - 其余 ('keyword'/'bm25'/'vector') 原样返回；vector 的缺依赖降级
          由底层 JsonlCorpus.vector_search 处理。
        """
        if strategy == "hybrid":
            return "bm25"
        return strategy

    def _search_actions(
        self,
        query: str,
        strategy: str,
        filters: dict[str, Any],
        top_k: int,
        recall_budget: int = 4,
    ) -> list[dict]:
        """动作目录检索，输出富字段结果。

        P0-4 召回预算逻辑:
          先按扩大后的候选预算 (top_k * recall_budget) 打分排序，再对候选
          池应用 metadata 过滤 (risk/status)，避免「在打分循环内过滤」导致
          结果不足或漏召回。若过滤后条数仍不足 top_k，且候选池尚未覆盖整个
          语料，则循环倍增预算继续打分补召回，直到满足或耗尽语料为止。

        keyword 策略对 title/action_id 加权，与历史 EdrRagAPI.action_search
        行为保持一致。
        """
        corpus = self._corpus("actions")
        if corpus is None:
            return []
        total = len(corpus.records)
        budget = max(top_k * recall_budget, top_k)

        while True:
            scored = self._score_actions(corpus, query, strategy, budget)
            filtered = _apply_filter(scored, filters, action_shape=True)
            if len(filtered) >= top_k or budget >= total:
                break
            budget = min(total, max(budget * 2, budget + 1))

        return filtered[:top_k]

    def _score_actions(
        self,
        corpus: JsonlCorpus,
        query: str,
        strategy: str,
        budget: int,
    ) -> list[dict]:
        """对语料打分排序，返回前 `budget` 个命中的动作富字段结果。

        bm25/vector 底层通用结果缺少 risk/status，动作的 metadata 过滤
        (risk/status) 依赖富字段，因此这里统一用动作富字段重打分排序，
        保证所有策略都能做一致的 metadata 过滤 (P0-4)。
        """
        if strategy == "bm25":
            pairs = self._bm25_rank(corpus, query)
        elif strategy == "vector":
            pairs = self._vector_rank(corpus, query)
        else:  # keyword
            return self._keyword_actions(corpus, query, budget)
        return [self._action_result(corpus, i, s) for s, i in pairs[:budget]]

    def _bm25_rank(self, corpus: JsonlCorpus, query: str) -> list[tuple[float, int]]:
        """轻量 BM25 打分，返回 (score, doc_idx) 降序列表 (动作富字段用)。"""
        from .p1.search import tokenize

        q_tokens = set(tokenize(query))
        if not q_tokens:
            return []
        n = len(corpus.records)
        idf_cache = {}
        for t in q_tokens:
            df = corpus._df.get(t, 0)
            idf_cache[t] = 0.0 if df == 0 else math.log((n - df + 0.5) / (df + 0.5) + 1)
        scores: dict[int, float] = {}
        for t in q_tokens:
            idf = idf_cache[t]
            if idf == 0:
                continue
            for doc_idx, tf in corpus._postings.get(t, []):
                dl = corpus._doc_len[doc_idx]
                tf_norm = (
                    (tf * (1.5 + 1)) / (tf + 1.5 * (1 - 0.75 + 0.75 * (dl / corpus._avgdl)))
                    if corpus._avgdl
                    else 0.0
                )
                scores[doc_idx] = scores.get(doc_idx, 0.0) + idf * tf_norm
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return [(s, i) for i, s in ranked]

    def _vector_rank(self, corpus: JsonlCorpus, query: str) -> list[tuple[float, int]]:
        """向量打分 (缺依赖时降级 BM25) 的 (score, doc_idx) 降序列表。

        复用 p1.search 的可选 embedding 助手，保留对可选 RAG extras 的
        支持；缺依赖时优雅降级到 BM25，主链路始终可用。
        """
        from .p1.search import _cached_vec, _cosine, _optional_embedder

        embed_fn = _optional_embedder()
        if embed_fn is None:
            return self._bm25_rank(corpus, query)
        try:
            q_vec = embed_fn(query)
            scored: list[tuple[float, int]] = []
            for i, r in enumerate(corpus.records):
                vec = _cached_vec(embed_fn, r["_searchable"])
                if vec is None:
                    continue
                score = _cosine(q_vec, vec)
                if score > 0:
                    scored.append((score, i))
            scored.sort(key=lambda x: -x[0])
            return scored
        except Exception:
            logging.getLogger(__name__).warning(
                "vector_score failed; falling back to BM25", exc_info=True
            )
            return self._bm25_rank(corpus, query)

    def _keyword_actions(self, corpus: JsonlCorpus, query: str, top_k: int) -> list[dict]:
        """动作专用 keyword 打分：对 action_id/title 加权。"""
        from .p1.search import tokenize

        q_tokens = set(tokenize(query))
        if not q_tokens:
            return []

        # 需要访问每个记录的搜索文本。直接复用 JsonlCorpus 内部字段。
        scored: list[tuple[float, int]] = []
        for i, r in enumerate(corpus.records):
            field = r["_searchable"]
            title = r.get("_title", "")
            action_id = r.get("action_id", "")
            score = 0.0
            for t in q_tokens:
                if t in field:
                    weight = 3.0 if (t in title or t in action_id) else 1.0
                    score += field.count(t) * weight
            if score > 0:
                scored.append((score, i))
        scored.sort(key=lambda x: -x[0])
        return [self._action_result(corpus, i, s) for s, i in scored[:top_k]]

    def _action_result(self, corpus: JsonlCorpus, doc_idx: int, score: float) -> dict:
        r = corpus.records[doc_idx]
        result = self._record_to_action_result(r)
        result["score"] = round(score, 3)
        return result

    @staticmethod
    def _record_to_action_result(r: dict) -> dict:
        """把动作原始 record 转成统一富字段结果 (id = action_id)。"""
        snippet = r.get("title", "")
        pre = r.get("preconditions", [])
        if pre:
            snippet += " | 前置条件: " + "；".join(pre)
        return {
            "id": r.get("action_id", ""),
            "title": r.get("title", ""),
            "risk": r.get("risk", ""),
            "status": r.get("status", ""),
            "confidence": r.get("confidence", 0.0),
            "score": 0.0,
            "snippet": snippet,
        }

    def _search_generic(
        self,
        corpus: JsonlCorpus,
        query: str,
        strategy: str,
        top_k: int,
    ) -> list[dict]:
        if strategy == "bm25":
            return corpus.bm25_search(query, limit=top_k)
        if strategy == "vector":
            return corpus.vector_search(query, limit=top_k)
        return corpus.keyword_search(query, limit=top_k)

    # ---------- 精确获取 ----------

    def get(
        self,
        corpus_type: str,
        key: str,
        key_field: Optional[str] = None,
    ) -> Optional[dict]:
        """按主键精确获取一条记录 (actions 用 action_id，其余用 id)。

        Args:
            corpus_type: 同上。
            key: 主键值。
            key_field: 显式指定匹配字段；缺省 actions->action_id，其余->id。
        """
        corpus = self._corpus(corpus_type)
        if corpus is None:
            return None

        if key_field is None:
            key_field = "action_id" if corpus_type == "actions" else "id"

        target = key
        # 兼容 'action.' / 'procedure.' 前缀的 action_id 查询。
        if corpus_type == "actions":
            for prefix in ("action.", "procedure."):
                if target.startswith(prefix):
                    target = target[len(prefix):]
                    break

        for r in corpus.records:
            if r.get(key_field, "") == target:
                # 剥掉 JsonlCorpus 注入的内部检索字段，仅返回原始记录。
                return {k: v for k, v in r.items() if not k.startswith("_")}
        return None

    # ---------- Planner 编排 (action -> procedure -> manual) ----------

    def planner_retrieve(
        self,
        user_request: str,
        limit: int = 5,
        strategy: str = _DEFAULT_STRATEGY,
    ) -> dict:
        """完整的 planner 检索流程，保持既有优先级:

          1. 先检索动作目录 (actions)
          2. 无动作结果 -> 检索流程 (procedures)
          3. 仍无结果 -> 检索手册章节 (sections)

        传入的 strategy 会被透传给每一层检索，保证 Planner、API、CLI、
        MCP 共享同一套可配置检索策略。

        Returns:
            dict: {query, actions, [procedures], [sections]}
        """
        actions = self.search(user_request, corpus_type="actions", top_k=limit, strategy=strategy)
        result: dict[str, Any] = {"query": user_request, "actions": actions}

        if not actions:
            procedures = self.search(
                user_request, corpus_type="procedures", top_k=limit, strategy=strategy
            )
            result["procedures"] = procedures
            if not procedures:
                sections = self.search(
                    user_request, corpus_type="sections", top_k=limit, strategy=strategy
                )
                result["sections"] = sections
        else:
            result["procedures"] = []
            result["sections"] = []

        return result


# ---------- 过滤辅助 ----------

def _apply_filter(
    results: list[dict],
    filters: dict[str, Any],
    action_shape: bool = False,
) -> list[dict]:
    """对结果应用健康过滤 (risk/status)。

    action_shape=True 时从富字段读取 risk/status；否则动作也回落到原始
    record 字段 (generic 结果没有 risk/status 键，直接跳过)。
    """
    risk = filters.get("risk")
    status = filters.get("status")
    if not risk and not status:
        return results

    out = []
    for r in results:
        if risk and r.get("risk") != risk:
            continue
        if status and r.get("status") != status:
            continue
        out.append(r)
    return out
