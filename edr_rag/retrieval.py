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

搜索策略 strategy: 'keyword' | 'bm25' | 'vector'
  - keyword: 子串/词元重叠打分，动作对 title/action_id 加权
  - bm25:    轻量 BM25
  - vector:  可选 RAG extras 的向量检索；缺依赖时优雅降级到 bm25
              (主链路始终可用 keyword/BM25)。

planner_retrieve 保持既有的 action -> procedure -> manual 优先级。
"""

from __future__ import annotations

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

_ALLOWED_STRATEGIES = ("keyword", "bm25", "vector")


class RetrievalService:
    """统一检索服务。

    由 workdir 懒加载各 corpus 并缓存；对外仅暴露 search / get /
    planner_retrieve 三个检索入口。
    """

    def __init__(self, workdir: str | Path):
        self.workdir = Path(workdir)
        self._corpora: dict[str, JsonlCorpus | None] = {
            "sections": None,
            "procedures": None,
            "actions": None,
            "chunks": None,
        }

    # ---------- Corpus 懒加载 ----------

    def _corpus(self, corpus_type: str) -> Optional[JsonlCorpus]:
        """按 corpus_type 返回缓存 corpus；文件不存在或为空时返回 None。"""
        if corpus_type not in _CORPUS_FILES:
            raise ValueError(
                f"unknown corpus_type '{corpus_type}'; "
                f"expected one of {sorted(_CORPUS_FILES)}"
            )
        cached = self._corpora.get(corpus_type)
        if cached is not None:
            return cached
        path = self.workdir / _CORPUS_FILES[corpus_type]
        if not path.exists():
            self._corpora[corpus_type] = None
            return None
        corpus = JsonlCorpus.from_file(path)
        self._corpora[corpus_type] = corpus
        return corpus

    # ---------- 搜索 ----------

    def search(
        self,
        query: str,
        corpus_type: str = "actions",
        strategy: str = "keyword",
        filters: Optional[dict[str, Any]] = None,
        top_k: int = 5,
    ) -> list[dict]:
        """统一检索入口。

        Args:
            query: 检索词 (支持中文)。
            corpus_type: 'sections' | 'procedures' | 'actions' | 'chunks'。
            strategy: 'keyword' | 'bm25' | 'vector'；vector 缺依赖降级 bm25。
            filters: dict，支持:
                - 'risk':   动作过滤 (low/medium/high)
                - 'status': 动作过滤 (candidate/needs_review/approved/published)
                - 'id':     精确匹配 id（actions 用 action_id）做获取式查询
            top_k: 返回条数上限。

        Returns:
            结果 dict 列表。actions 返回富字段
            (id/title/risk/status/confidence/score/snippet)；
            其余类型返回 JsonlCorpus 的标准结果
            (id/type/score/title/source/snippet)。
        """
        filters = filters or {}
        top_k = max(int(top_k), 0)

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

        if corpus_type == "actions":
            results = self._search_actions(
                query, strategy=strategy, filters=filters, top_k=top_k
            )
        else:
            corpus = self._corpus(corpus_type)
            if corpus is None:
                return []
            results = self._search_generic(corpus, query, strategy, top_k)
            results = _apply_filter(results, filters)

        return results

    def _search_actions(
        self,
        query: str,
        strategy: str,
        filters: dict[str, Any],
        top_k: int,
    ) -> list[dict]:
        """动作目录检索，输出富字段结果。

        keyword 策略对 title/action_id 加权，与历史 EdrRagAPI.action_search
        行为保持一致；过滤 (risk/status) 在打分后应用。
        """
        corpus = self._corpus("actions")
        if corpus is None:
            return []

        if strategy == "bm25":
            results = corpus.bm25_search(query, limit=top_k * 4)
        elif strategy == "vector":
            results = corpus.vector_search(query, limit=top_k * 4)
        else:  # keyword
            results = self._keyword_actions(corpus, query, top_k)

        results = _apply_filter(results, filters, action_shape=True)
        return results[:top_k]

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

    def planner_retrieve(self, user_request: str, limit: int = 5) -> dict:
        """完整的 planner 检索流程，保持既有优先级:

          1. 先检索动作目录 (actions)
          2. 无动作结果 -> 检索流程 (procedures)
          3. 仍无结果 -> 检索手册章节 (sections)

        Returns:
            dict: {query, actions, [procedures], [sections]}
        """
        actions = self.search(user_request, corpus_type="actions", top_k=limit)
        result: dict[str, Any] = {"query": user_request, "actions": actions}

        if not actions:
            procedures = self.search(user_request, corpus_type="procedures", top_k=limit)
            result["procedures"] = procedures
            if not procedures:
                sections = self.search(user_request, corpus_type="sections", top_k=limit)
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
