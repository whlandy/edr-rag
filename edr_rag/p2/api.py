"""P2 Python API — planner retrieval interface.

Design (§6, Planner Retrieval Flow):

  The planner should call the manual knowledge layer in this order:
    1. action_search      — search action catalog first
    2. get_action         — get full action detail
    3. procedure_search   — if action_search has no result
    4. manual_search      — only for explanation or troubleshooting

All search functions return results as dicts (JSON-serializable), so they
can be passed directly to the planner or over MCP.

Since P0-2, this API is a thin, backward-compatible wrapper over the unified
RetrievalService (edr_rag/retrieval.py) — it no longer touches JSONL /
JsonlCorpus directly; all retrieval is delegated to the service so pure-API,
CLI, and MCP share one code path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from ..retrieval import RetrievalService
from ..security import ACLPolicy


class EdrRagAPI:
    """P2 Retrieval API for the EDR-RAG system.

    Provides unified access to:
      - action_search: search action catalog entries
      - get_action: get full action entry by id
      - procedure_search: search P1 procedures (fallback)
      - manual_search: search P0 sections (troubleshooting only)

    All operations delegate to a shared RetrievalService instance.
    """

    def __init__(self, workdir: str | Path):
        self.workdir = Path(workdir)
        self._service = RetrievalService(self.workdir)

    # ---------- Action search (first priority) ----------

    def action_search(
        self,
        query: str,
        limit: int = 5,
        risk_filter: Optional[str] = None,
        status_filter: Optional[str] = None,
        strategy: str = "keyword",
        recall_budget: int = 4,
        rerank: Optional[bool] = None,
        acl: Optional["ACLPolicy"] = None,
    ) -> list[dict]:
        """Search the action catalog (delegated to RetrievalService).

        strategy: 'keyword' | 'bm25' | 'vector' | 'hybrid' (default 'keyword').

        recall_budget (P0-4): 召回预算倍数。先按 top_k*recall_budget 扩大
        候选池打分排序，再对候选池做 risk/status metadata 过滤，结果不足时
        循环扩大候选池直到满足或耗尽语料；避免严格过滤器导致漏召回。

        rerank (P1-3): cross-encoder 精排开关，仅对 hybrid 生效；None 跟随
        全局默认 (可选依赖可用即启用)。

        acl (P3-3): 文档级 ACL 过滤策略；None 表示不过滤。
        """
        filters = {}
        if risk_filter:
            filters["risk"] = risk_filter
        if status_filter:
            filters["status"] = status_filter
        return self._service.search(
            query,
            corpus_type="actions",
            strategy=strategy,
            filters=filters,
            top_k=limit,
            recall_budget=recall_budget,
            rerank=rerank,
            acl=acl,
        )

    def get_action(self, action_id: str) -> Optional[dict]:
        """Get the full action catalog entry by action_id (stripped).

        Handles 'action.' / 'procedure.' prefixes via the service's get.
        """
        return self._service.get("actions", action_id, key_field="action_id")

    # ---------- Procedure search (second priority) ----------

    def procedure_search(
        self,
        query: str,
        limit: int = 5,
        scoring: Optional[str] = None,
        strategy: Optional[str] = None,
        rerank: Optional[bool] = None,
        acl: Optional[ACLPolicy] = None,
    ) -> list[dict]:
        """Search P1 procedures (fallback if action_search has no results).

        strategy: 'keyword' | 'bm25' | 'vector' | 'hybrid' (default 'keyword').
        `scoring` 为历史别名参数，与 `strategy` 等价；两者都传时以
        `strategy` 为准（保持 backward compatibility）。

        rerank (P1-3): cross-encoder 精排开关，仅对 hybrid 生效。
        acl (P3-3): 文档级 ACL 过滤策略；None 表示不过滤。
        """
        if strategy is None:
            strategy = scoring or "keyword"
        return self._service.search(
            query,
            corpus_type="procedures",
            strategy=strategy,
            top_k=limit,
            rerank=rerank,
            acl=acl,
        )

    # ---------- Manual search (last priority) ----------

    def manual_search(
        self,
        query: str,
        limit: int = 5,
        scoring: Optional[str] = None,
        strategy: Optional[str] = None,
        rerank: Optional[bool] = None,
        acl: Optional[ACLPolicy] = None,
    ) -> list[dict]:
        """Search P0 sections (for explanation or troubleshooting only).

        strategy: 'keyword' | 'bm25' | 'vector' | 'hybrid' (default 'keyword').
        `scoring` 为历史别名参数，与 `strategy` 等价；两者都传时以
        `strategy` 为准。

        rerank (P1-3): cross-encoder 精排开关，仅对 hybrid 生效。
        acl (P3-3): 文档级 ACL 过滤策略；None 表示不过滤。
        """
        if strategy is None:
            strategy = scoring or "keyword"
        return self._service.search(
            query,
            corpus_type="sections",
            strategy=strategy,
            top_k=limit,
            rerank=rerank,
            acl=acl,
        )

    # ---------- 结构化节点检索 & 上下文补全 (P1-4) ----------

    def node_search(
        self,
        query: str,
        limit: int = 5,
        strategy: str = "keyword",
        with_context: bool = True,
        adjacency: int = 1,
    ) -> list[dict]:
        """检索结构化节点（P1-2），并对命中节点做父子/相邻上下文补全 (P1-4)。

        返回结果中每条含 `context`（assemble_context 结构化结果）与
        `context_text`（带标签渲染文本）。nodes.jsonl 缺失时退化为普通检索
        （不报错，无 context 键）。
        """
        return self._service.search(
            query,
            corpus_type="nodes",
            strategy=strategy,
            top_k=limit,
            with_context=with_context,
            adjacency=adjacency,
        )

    def get_context(
        self,
        node_id: str,
        *,
        adjacency: int = 1,
        include_section: bool = True,
    ) -> Optional[dict]:
        """按节点 id 获取结构化上下文（父子/相邻/前置条件/注意事项/恢复）。

        nodes.jsonl 缺失或 node_id 不存在时返回 None。
        """
        return self._service.context_for(
            node_id,
            adjacency=adjacency,
            include_section=include_section,
        )

    # ---------- Planner orchestration ----------

    def planner_retrieve(
        self,
        user_request: str,
        limit: int = 5,
        strategy: str = "keyword",
        rerank: Optional[bool] = None,
        acl: Optional[ACLPolicy] = None,
    ) -> dict:
        """Orchestrate the full planner retrieval flow (action → procedure → manual).

        Delegates to RetrievalService.planner_retrieve which preserves the
        §6 priority order and threads `strategy` to each retrieval tier.
        `rerank` (P1-3) is likewise threaded through. `acl` (P3-3) filters
        each tier's results by document-level device/tenant metadata.
        """
        return self._service.planner_retrieve(
            user_request, limit=limit, strategy=strategy, rerank=rerank, acl=acl
        )

    # ---------- 结构化答案合成 (P2-1) ----------

    def synthesize(
        self,
        user_request: str,
        context: Optional[dict] = None,
        llm: Optional[Callable] = None,
        limit: int = 5,
        strategy: str = "keyword",
        min_confidence: Optional[float] = None,
        acl: Optional[ACLPolicy] = None,
        sanitize: bool = True,
    ) -> dict:
        """基于检索上下文生成结构化答案（P2-1）。

        Args:
            user_request: 用户请求原文。
            context: 可选检索上下文 dict（planner_retrieve 形状）。为 None 时
                内部自动调用 planner_retrieve 获取。
            llm: 可选的可调用对象 `(user_request, context_text) -> str`，
                用于生成自然语言 answer；缺省回落为结构化组装（主链路不报错）。
            limit / strategy: 自动检索时的检索参数。
            min_confidence: 证据不足拒答的置信度阈值（P2-3）。为 None 时使用
                模块默认 CONFIDENCE_REFUSE_THRESHOLD。
            acl: (P3-3) 文档级 ACL 过滤策略；None 表示不过滤。自动检索时
                透传给每一层，越权的文档不会进入上下文。
            sanitize: (P3-3) 是否对喂给 LLM 的检索上下文做注入清洗与
                system prompt 隔离（默认 True）。组装模式下不影响结构化字段。

        Returns:
            SynthesizedAnswer.to_jsonl() 的 dict:
            {answer, steps, warnings, warning_sources, assertions, preconditions,
             citations, confidence, requires_confirmation, generated_by,
             answer_status, refusal_reason, conflicts}。
            P2-2: 每个 step 含 source；warning_sources 与 warnings 一一对应；
            assertions 为带 source 的断言列表，均可追溯到 artifacts。
            P2-3: answer_status 为 answered / refused / conflict；refused 时
            不编造，conflict 时标注冲突并置 requires_confirmation=True。
        """
        from .synthesize import synthesize_answer

        if context is None:
            context = self._service.planner_retrieve(
                user_request, limit=limit, strategy=strategy, acl=acl
            )
        # 检索返回的动作结果是薄形状（无 procedure/preconditions/recovery），
        # 用全量动作记录补全，让合成器能输出完整步骤/前置条件/注意事项。
        context = _enrich_action_details(self._service, context)
        return synthesize_answer(
            user_request,
            context,
            llm=llm,
            min_confidence=min_confidence,
            sanitize=sanitize,
        ).to_jsonl()


    # ---------- AskAnswer: 结构化答案的 agent 消费入口 (P2-4 / P2-5) ----------

    def ask_answer(
        self,
        user_request: str,
        llm: Optional[Callable] = None,
        limit: int = 5,
        strategy: str = "keyword",
        min_confidence: Optional[float] = None,
        confirm: Optional[bool] = None,
        acl: Optional[ACLPolicy] = None,
        sanitize: bool = True,
    ) -> dict:
        """AskAnswer：把问题检索并合成为结构化答案，供 agent/规划器消费 (P2-4)。

        与 `synthesize` 等价，但返回字典额外带便捷字段，让 agent 无需自己从
        steps/citations 重新聚合即可拿到:
          - dangerous:  bool — 是否涉及高风险或需要人工确认（requires_confirmation
                        或 answer_status=='conflict'）。agent 可据此决定是否把
                        控制权交给人 / 降低自动化程度。
          - sources:    所有来源 file 的列表（去重），便于 agent 追溯。

        结构化字段（steps/risk/preconditions/warnings/assertions/answer_status 等）
        与 `synthesize` 完全一致，agent 应优先消费这些字段而不仅是 `answer` 文本。

        P2-5 显式确认门控:
          - confirm=None（默认）：向后兼容的信息模式，返回完整结构化答案（仅带
            requires_confirmation / dangerous 等标记字段，不做门控）。
          - confirm=True / False：对该答案应用显式确认门控。当答案命中
            requires_confirmation 的 Action 且 confirm 非 True 时，返回
            confirmation_status='required' 的门控答案（扣留可执行步骤，改为
            请确认提示）；confirm=True 则放行并标注 confirmation_status='confirmed'。

        acl (P3-3): 文档级 ACL 过滤策略；None 表示不过滤。
        sanitize (P3-3): 是否对喂给 LLM 的检索上下文做注入清洗与 system
            prompt 隔离（默认 True）。
        """
        result = self.synthesize(
            user_request,
            llm=llm,
            limit=limit,
            strategy=strategy,
            min_confidence=min_confidence,
            acl=acl,
            sanitize=sanitize,
        )
        result = _augment_ask_answer(result)
        if confirm is None:
            # 兼容模式：不门控，仅返回结构化标记（含 confirmation 信息）。
            from . import confirmation as confirm_mod

            result.setdefault("confirmation_status", (
                confirm_mod.CONFIRMATION_CONFIRMED
                if not confirm_mod.needs_explicit_confirmation(result)
                else confirm_mod.CONFIRMATION_REQUIRED
            ))
            result.setdefault("needs_confirmation",
                              confirm_mod.needs_explicit_confirmation(result))
            return result
        from . import confirmation as confirm_mod

        return confirm_mod.gate_answer(result, confirm=bool(confirm))


def _augment_ask_answer(result: dict) -> dict:
    """在 synthesize 的结构化结果上追加 agent 便捷字段 (P2-4)。"""
    out = dict(result)
    # dangerous: 需要人工确认或检测到冲突 → 视为高风险，agent 应保守处理。
    out["dangerous"] = bool(
        out.get("requires_confirmation") or out.get("answer_status") == "conflict"
    )
    # sources: 聚合所有步骤/警告/断言的来源 file，去重保序
    files: list[str] = []
    seen: set[str] = set()
    for s in out.get("steps") or []:
        f = (s.get("source") or {}).get("file", "")
        if f and f not in seen:
            seen.add(f)
            files.append(f)
    for w in out.get("warning_sources") or []:
        f = (w or {}).get("file", "")
        if f and f not in seen:
            seen.add(f)
            files.append(f)
    for a in out.get("assertions") or []:
        f = (a.get("source") or {}).get("file", "")
        if f and f not in seen:
            seen.add(f)
            files.append(f)
    out["sources"] = files
    return out


def _enrich_action_details(service, context: dict) -> dict:
    """把薄形状的动作检索结果补全为全量动作记录。

    检索层 (_record_to_action_result) 只返回 id/title/risk/status/confidence/score，
    丢失了 procedure/preconditions/recovery/requires_confirmation 等结构化字段。
    这里通过 service.get 逐条取回全量记录，供答案合成器提取完整步骤。
    """
    ctx = dict(context)
    actions = ctx.get("actions") or []
    enriched = []
    for a in actions:
        full = service.get("actions", a.get("id") or a.get("action_id", ""))
        if full:
            # 保留检索层的 score 与薄字段，合并全量字段。
            merged = dict(a)
            for k, v in full.items():
                if k not in merged or merged.get(k) in (None, "", 0.0, []):
                    merged[k] = v
            enriched.append(merged)
        else:
            enriched.append(a)
    ctx["actions"] = enriched
    return ctx
