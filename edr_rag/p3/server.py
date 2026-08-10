"""P3: edr-rag MCP server — separate from edr-wd.

P3 scope (design §7 + §12):
  1. expose MCP tools: search, action retrieval, feedback ingestion
  2. execution feedback: capture, classify, propose patches
  3. re-index trigger

Design decisions:
  - Separate MCP server (edr-rag) from edr-wd execution server
  - stdio transport for simplicity; can be wrapped in SSH tunnel
  - Artifact-path-based: reads JSONL artifacts from a work directory
  - Feedback stored as JSONL alongside artifacts

Usage:
    python -m tools.edr_rag.p3.server --workdir ~/Desktop/edr-chm-rag-work
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# ---------- Optional MCP SDK import ----------

try:
    from mcp.server import Server, NotificationOptions
    from mcp.server.models import InitializationOptions
    import mcp.server.stdio
    import mcp.types as types
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False


# ---------- Import P0/P1/P2 components ----------

def _add_cwd_to_path():
    """Add tools directory to sys.path for imports."""
    this_dir = Path(__file__).resolve().parent  # p3/
    tools_dir = this_dir.parent.parent.parent  # .../tools/
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))


_add_cwd_to_path()

from edr_rag import normalize  # noqa: E402
from edr_rag.p1 import schema as p1schema  # noqa: E402
from edr_rag.p2 import ActionCatalogEntry  # noqa: E402
from edr_rag.p3.feedback import FeedbackItem, FeedbackEvidence, FeedbackStore, SuggestedPatch, FAILURE_TYPES  # noqa: E402, E501


# ---------- P3 agent implementation ----------

class EdrRagAgent:
    """P3 edr-rag agent — bridges artifacts with MCP tools.

    This is not the RAGCycle from src/mcp_edr_rag/ — it's the P3 specific
    implementation that reads P0/P1/P2 artifacts and exposes MCP tools.
    """

    def __init__(self, workdir: str, trace: Optional[bool] = None):
        self.workdir = Path(workdir)
        self.feedback_store = FeedbackStore(str(self.workdir / "feedback.jsonl"))
        # Unified retrieval entry (P0-2) — the only way to access JSONL artifacts.
        from edr_rag.retrieval import RetrievalService
        self.retrieval = RetrievalService(self.workdir)
        # P3-2: 结构化可观测性 —— 默认开启 JSONL 追踪 (trace.jsonl)。
        # trace=False 或环境变量 EDR_RAG_TRACE=0 时关闭，保证可独立于观测运行。
        from edr_rag.p3.tracing import EventTracer
        if trace is None:
            trace = os.environ.get("EDR_RAG_TRACE", "1") not in ("0", "false", "False")
        self.trace_enabled = bool(trace)
        self.tracer = (
            EventTracer(self.workdir / "trace.jsonl") if self.trace_enabled else None
        )

    # P3-2: 统一检索追踪辅助 —— 计时 + 记录检索事件，返回 (结果, 耗时ms)。
    def _traced_search(self, ctx, corpus_type, query, limit, strategy, rerank):
        from edr_rag.p3.tracing import TraceContext
        t0 = time.perf_counter()
        try:
            results = self.retrieval.search(
                query, corpus_type=corpus_type, top_k=limit,
                strategy=strategy, rerank=rerank,
            )
        except Exception:  # 记录失败事件后重新抛出，主链路异常语义不变
            latency = (time.perf_counter() - t0) * 1000.0
            if self.tracer is not None and ctx is None:
                TraceContext(self.tracer, context=f"{corpus_type}.search").record_retrieval(
                    query, corpus_type, strategy, latency, [], ok=False,
                )
            raise
        latency = (time.perf_counter() - t0) * 1000.0
        if ctx is not None:
            ctx.record_retrieval(query, corpus_type, strategy, latency, results)
        return results, latency

    # ----- Knowledge search tools -----

    def search_sections(self, query: str, limit: int = 5, strategy: str = "keyword",
                        rerank: Optional[bool] = None) -> list[dict]:
        """Search P0 sections.jsonl (via RetrievalService).

        strategy: 'keyword' | 'bm25' | 'vector' | 'hybrid' (default 'keyword').
        rerank (P1-3): cross-encoder 精排开关，仅对 hybrid 生效。
        """
        from edr_rag.p3.tracing import TraceContext
        ctx = TraceContext(self.tracer, context="search_sections") if self.tracer else None
        results, _ = self._traced_search(ctx, "sections", query, limit, strategy, rerank)
        return results

    def search_procedures(self, query: str, limit: int = 5, strategy: str = "keyword",
                          rerank: Optional[bool] = None) -> list[dict]:
        """Search P1 procedures.jsonl (via RetrievalService).

        strategy: 'keyword' | 'bm25' | 'vector' | 'hybrid' (default 'keyword').
        rerank (P1-3): cross-encoder 精排开关，仅对 hybrid 生效。
        """
        from edr_rag.p3.tracing import TraceContext
        ctx = TraceContext(self.tracer, context="search_procedures") if self.tracer else None
        results, _ = self._traced_search(ctx, "procedures", query, limit, strategy, rerank)
        return results

    def search_actions(self, query: str, limit: int = 5, strategy: str = "keyword",
                       rerank: Optional[bool] = None) -> list[dict]:
        """Search P2 actions.jsonl (rich action result, via RetrievalService).

        strategy: 'keyword' | 'bm25' | 'vector' | 'hybrid' (default 'keyword').
        rerank (P1-3): cross-encoder 精排开关，仅对 hybrid 生效。
        """
        from edr_rag.p3.tracing import TraceContext
        ctx = TraceContext(self.tracer, context="search_actions") if self.tracer else None
        results, _ = self._traced_search(ctx, "actions", query, limit, strategy, rerank)
        return results

    def search_chunks(self, query: str, limit: int = 5, strategy: str = "keyword",
                      rerank: Optional[bool] = None) -> list[dict]:
        """Search P1 chunks.jsonl (via RetrievalService).

        strategy: 'keyword' | 'bm25' | 'vector' | 'hybrid' (default 'keyword').
        rerank (P1-3): cross-encoder 精排开关，仅对 hybrid 生效。
        """
        from edr_rag.p3.tracing import TraceContext
        ctx = TraceContext(self.tracer, context="search_chunks") if self.tracer else None
        results, _ = self._traced_search(ctx, "chunks", query, limit, strategy, rerank)
        return results

    def search_images(self, query: str, limit: int = 5, strategy: str = "keyword",
                      rerank: Optional[bool] = None) -> list[dict]:
        """Search P3-4 ocr.jsonl (图片 OCR 文本/截图检索, via RetrievalService).

        依据截图 OCR 出的文字检索，用于「我只记得一张图/截图里写着 XX」的
        场景，返回命中的图片 asset（含 path / 所属章节）。corpus_type="images"。
        strategy: 'keyword' | 'bm25' | 'vector' | 'hybrid' (default 'keyword').
        rerank: cross-encoder 精排开关，仅对 hybrid 生效。
        """
        from edr_rag.p3.tracing import TraceContext
        ctx = TraceContext(self.tracer, context="search_images") if self.tracer else None
        results, _ = self._traced_search(ctx, "images", query, limit, strategy, rerank)
        return results

    def get_action(self, action_id: str) -> Optional[dict]:
        """Retrieve a specific action by action_id."""
        return self.retrieval.get("actions", action_id, key_field="action_id")

    def get_procedure(self, procedure_id: str) -> Optional[dict]:
        """Retrieve a specific procedure by id."""
        return self.retrieval.get("procedures", procedure_id, key_field="id")

    # ----- AskAnswer: 结构化答案 (P2-4) -----

    def ask_answer(self, query: str, limit: int = 5, strategy: str = "keyword",
                   min_confidence: Optional[float] = None,
                   confirm: Optional[bool] = None) -> dict:
        """AskAnswer：检索并合成为结构化答案，供 agent/规划器消费 (P2-4)。

        返回结构化答案 dict（steps/risk/preconditions/warnings/assertions/
        answer_status/confidence/requires_confirmation/dangerous/sources 等），
        让 agent 消费结构化字段而不只是纯文本。缺依赖时自动降级到 keyword。

        P2-5: `confirm` 用于显式确认门控。为 None 时（默认）返回完整带标记的
        答案；为 True/False 时执行门控，命中 requires_confirmation 的 Action
        且未确认则扣留可执行步骤（confirmation_status='required'）。
        """
        from edr_rag.p2.api import EdrRagAPI

        api = EdrRagAPI(self.workdir)
        from edr_rag.p3.tracing import TraceContext

        ctx = TraceContext(self.tracer, context="ask_answer") if self.tracer else None
        t0 = time.perf_counter()
        try:
            result = api.ask_answer(
                query, limit=limit, strategy=strategy, min_confidence=min_confidence,
                confirm=confirm,
            )
        except Exception:
            if ctx is not None:
                latency = (time.perf_counter() - t0) * 1000.0
                ctx.record_generation(query, latency, {"answer_status": "error"}, ok=False)
            raise
        latency = (time.perf_counter() - t0) * 1000.0
        if ctx is not None:
            ctx.record_generation(query, latency, result)
        return result

    # ----- Feedback tools -----

    def capture_feedback(self, feedback: dict) -> str:
        """Capture a feedback item from an execution run.

        Returns feedback_id.
        """
        feedback_id = feedback.get("feedback_id", f"fb_{int(datetime.now().timestamp())}")
        item = FeedbackItem(
            feedback_id=feedback_id,
            manual_id=feedback.get("manual_id", ""),
            action_id=feedback.get("action_id"),
            procedure_id=feedback.get("procedure_id"),
            failure_type=feedback.get("failure_type", ""),
            expected=feedback.get("expected", ""),
            observed=feedback.get("observed", ""),
            evidence=FeedbackEvidence(
                screenshot=feedback.get("evidence", {}).get("screenshot"),
                dump_tree=feedback.get("evidence", {}).get("dump_tree"),
                error_message=feedback.get("evidence", {}).get("error_message"),
            ) if feedback.get("evidence") else None,
            suggested_patch=SuggestedPatch(
                patch_type=feedback.get("suggested_patch", {}).get("patch_type"),
                field=feedback.get("suggested_patch", {}).get("field"),
                old_value=feedback.get("suggested_patch", {}).get("old_value"),
                new_value=feedback.get("suggested_patch", {}).get("new_value"),
            ) if feedback.get("suggested_patch") else None,
            status="captured",
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self.feedback_store.append(item)
        return feedback_id

    def _find_feedback(self, feedback_id: str):
        """按 id 取一条反馈；不存在返回 None。"""
        for item in self.feedback_store.read_all():
            if item.feedback_id == feedback_id:
                return item
        return None

    def classify_failure(self, feedback_id: str) -> dict:
        """按规则表分类失败类型 (P4 Phase H)。

        P3 版本是 4 条 if-elif 子串匹配，覆盖 FAILURE_TYPES 里 11 种中的 4 种，
        且 ``"ui" in msg`` 会把任何含 build/guide/require 的消息误判成 UI 错误
        （子串匹配没有词边界）。现在走 ``classify`` 的带优先级规则表，
        expected / observed / error_message / dump_tree 都参与判定。

        Returns:
            dict: {ok, feedback_id, failure_type, matched, rationale}。
                未命中任何规则时保留原有 failure_type（可能是 edr-wd 已判好的），
                而不是硬塞一个错的。
        """
        from edr_rag.p3.classify import classify

        target = self._find_feedback(feedback_id)
        if target is None:
            return {"ok": False, "feedback_id": feedback_id, "error": "not found"}

        result = classify(target)
        self.feedback_store.update_item(
            feedback_id,
            failure_type=result.failure_type or target.failure_type,
            status="needs_review",
        )
        payload = {"ok": True, "feedback_id": feedback_id}
        payload.update(result.to_dict())
        return payload

    def propose_patch(self, feedback_id: str) -> Optional[dict]:
        """由反馈**生成**一个动作目录补丁 (P4 Phase H)。

        P3 版本只是把调用方自己传进来的 patch 原样退回并改个状态——既不生成
        也不应用，反馈回路在这里就断了。现在按 failure_type + evidence 推导
        补丁；调用方已显式给出 patch 时以它为准（人比启发式可靠）。

        生成的补丁**不会自动应用**，需要显式调用 ``apply_patch``。
        """
        from edr_rag.p3.patch import propose

        target = self._find_feedback(feedback_id)
        if target is None:
            return None
        patch = propose(target)
        if patch is None:
            return None
        self.feedback_store.update_item(
            feedback_id, suggested_patch=patch, status="patch_proposed"
        )
        return patch.to_jsonl()

    def apply_patch(self, feedback_id: str, confirm: bool = False) -> dict:
        """把已提议的补丁真正写进 actions.jsonl (P4 Phase H)。

        动作目录直接驱动客户终端上的 GUI 自动化——一条错补丁 = 在客户机器上
        点错按钮。因此：
          - 只应用 status 已是 patch_proposed 的条目（必须先走提议）
          - 高风险动作需要 confirm=True
          - 应用前捕获索引版本快照，可用 `edr-rag index rollback` 回退

        Returns:
            PatchResult.to_dict()，附带 feedback_id。
        """
        from edr_rag.p3.patch import PATCH_APPLIED, PATCH_PROPOSED, apply

        target = self._find_feedback(feedback_id)
        if target is None:
            return {"ok": False, "feedback_id": feedback_id, "error": "not found"}
        if target.status != PATCH_PROPOSED:
            return {
                "ok": False,
                "feedback_id": feedback_id,
                "error": (
                    f"feedback status is '{target.status}'; "
                    f"call propose_patch first (expected '{PATCH_PROPOSED}')"
                ),
            }

        result = apply(self.workdir, target, confirm=confirm)
        payload = result.to_dict()
        payload["feedback_id"] = feedback_id
        if result.ok:
            self.feedback_store.update_item(feedback_id, status=PATCH_APPLIED)
        return payload

    def reject_patch(self, feedback_id: str, reason: str = "") -> dict:
        """驳回一个补丁提议 (P4 Phase H) —— 审核状态机的另一条出边。"""
        from edr_rag.p3.patch import PATCH_REJECTED

        if self._find_feedback(feedback_id) is None:
            return {"ok": False, "feedback_id": feedback_id, "error": "not found"}
        self.feedback_store.update_item(
            feedback_id, status=PATCH_REJECTED, review_note=reason
        )
        return {"ok": True, "feedback_id": feedback_id, "status": PATCH_REJECTED}

    # ----- Re-index -----

    def reindex(self, manual_id: str, product: str) -> dict:
        """Trigger a full re-index from P0 artifacts through P2.

        Returns summary dict.
        """
        # P0: re-run ingestion (requires original .chm file)
        # For P3 demo, we skip actual re-extraction and just note it.
        # In production: find .chm files in workdir/input/, run ingest, then P1, then P2.
        chm_dir = self.workdir / "input"
        chm_files = list(chm_dir.glob("*.chm")) if chm_dir.exists() else []
        if not chm_files:
            return {
                "ok": False,
                "error": "No .chm files found in workdir/input/",
                "manual_id": manual_id,
                "product": product,
            }

        # For now: placeholder — log the intent and return instructions
        return {
            "ok": True,
            "manual_id": manual_id,
            "product": product,
            "chm_files": [str(p) for p in chm_files],
            "steps": [
                "1. python -m tools.edr_rag.cli ingest --chm <file> --manual-id <id> --product <p> --workdir .",
                "2. python -m tools.edr_rag.p1.cli extract-procedures --workdir .",
                "3. python -m tools.edr_rag.p2.cli build-actions --workdir .",
            ],
            "message": "Re-index requires manual steps listed above, or call via CLI directly",
        }


# ---------- MCP server ----------

def create_mcp_server(workdir: str) -> Server:
    """Create an MCP server with edr-rag tools."""
    agent = EdrRagAgent(workdir)
    app = Server("edr-rag")

    # shared strategy property for search tools (keyword default; bm25 / vector / hybrid)
    _STRATEGY_PROP = {
        "type": "string",
        "description": ("Retrieval strategy: keyword (default) | bm25 | vector | hybrid. "
                        "vector/hybrid gracefully degrade to bm25 when optional RAG deps are absent."),
        "enum": ["keyword", "bm25", "vector", "hybrid"],
        "default": "keyword",
    }
    # shared cross-encoder rerank property (P1-3), only meaningful for hybrid.
    _RERANK_PROP = {
        "type": "string",
        "description": ("Cross-encoder rerank toggle for hybrid: auto (default, enable when "
                        "optional RAG deps available, else skip) | on | off."),
        "enum": ["auto", "on", "off"],
        "default": "auto",
    }

    def _rerank_arg(value) -> Optional[bool]:
        """把工具的 rerank (auto/on/off) 映射为 service 的 Optional[bool]。"""
        if not value or value == "auto":
            return None
        return value == "on"

    @app.list_tools()
    async def handle_list_tools():
        return [
            types.Tool(
                name="search_knowledge",
                description="Search ManualSection knowledge base by keyword. Use this BEFORE procedures/actions when you need context.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query (CJK-safe, keyword based)"},
                        "limit": {"type": "integer", "description": "Max results", "default": 5},
                        "strategy": _STRATEGY_PROP,
                        "rerank": _RERANK_PROP,
                    },
                    "required": ["query"],
                },
            ),
            types.Tool(
                name="search_procedures",
                description="Search extracted procedures by keyword. Use this when you need step-by-step instructions for a task.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query (CJK-safe)"},
                        "limit": {"type": "integer", "description": "Max results", "default": 5},
                        "strategy": _STRATEGY_PROP,
                        "rerank": _RERANK_PROP,
                    },
                    "required": ["query"],
                },
            ),
            types.Tool(
                name="search_actions",
                description="Search action catalog by keyword. This is the FIRST lookup for planner: retrieval priority 1. Use before procedures.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query (CJK-safe)"},
                        "limit": {"type": "integer", "description": "Max results", "default": 5},
                        "strategy": _STRATEGY_PROP,
                        "rerank": _RERANK_PROP,
                    },
                    "required": ["query"],
                },
            ),
            types.Tool(
                name="search_images",
                description="Search P3-4 image OCR text (ocr.jsonl). Use when the user remembers text visible in a screenshot/diagram inside the manual, to locate the image asset (path + owning section).",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query (CJK-safe), e.g. text seen in a screenshot"},
                        "limit": {"type": "integer", "description": "Max results", "default": 5},
                        "strategy": _STRATEGY_PROP,
                        "rerank": _RERANK_PROP,
                    },
                    "required": ["query"],
                },
            ),
            types.Tool(
                name="get_action",
                description="Retrieve a full action definition by action_id. Returns risk, preconditions, steps, success criteria.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "action_id": {"type": "string", "description": "e.g., 'network.configure_proxy'"},
                    },
                    "required": ["action_id"],
                },
            ),
            types.Tool(
                name="get_procedure",
                description="Retrieve a full procedure by procedure id.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "procedure_id": {"type": "string", "description": "e.g., 'procedure.network.configure_proxy'"},
                    },
                    "required": ["procedure_id"],
                },
            ),
            types.Tool(
                name="ask",
                description=("AskAnswer (P2-4): 检索并合成一个「结构化答案」，返回 steps / "
                             "preconditions / warnings / risk / confidence / answer_status / "
                             "sources 等结构化字段。agent 应消费这些字段而不是纯文本 answer。"
                             "这是把问题变成可执行计划的首选入口。\n\n"
                             "P2-5 显式确认：`confirm` 字段用于显式确认门控。当答案命中 "
                             "requires_confirmation=true 的高风险 Action（或检测到冲突）且未传 "
                             "confirm=true 时，本工具会扣留可执行步骤并返回 confirmation_status="
                             "'required' 的门控答案；你必须先向用户呈现确认请求、得到明确确认后，"
                             "再以 confirm=true 重新调用以放行完整操作步骤。"),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "User request / question (CJK-safe)"},
                        "limit": {"type": "integer", "description": "Retrieval limit per tier", "default": 5},
                        "strategy": _STRATEGY_PROP,
                        "min_confidence": {"type": "number", "description": "证据不足拒答的置信度阈值 (P2-3, 默认 0.4)", "default": None},
                        "confirm": {
                            "type": "boolean",
                            "description": ("P2-5 显式确认：是否确认放行高风险 Action 的完整步骤。"
                                            "默认 false；当答案带 requires_confirmation=true 的 Action "
                                            "且 confirm 非 true 时返回门控答案，需用户明确确认后以 "
                                            "confirm=true 重新调用。"),
                            "default": False,
                        },
                    },
                    "required": ["query"],
                },
            ),
            types.Tool(
                name="capture_feedback",
                description="Capture execution feedback for later analysis and catalog improvement. Edr-wd should call this after each action attempt.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "feedback_id": {"type": "string", "description": "Optional; auto-generated if omitted"},
                        "manual_id": {"type": "string", "description": "Manual identifier"},
                        "action_id": {"type": "string", "description": "Action that was attempted"},
                        "procedure_id": {"type": "string", "description": "Procedure that was attempted"},
                        "failure_type": {"type": "string", "description": f"One of: {', '.join(sorted(FAILURE_TYPES))}"},
                        "expected": {"type": "string", "description": "What was expected"},
                        "observed": {"type": "string", "description": "What actually occurred"},
                        "evidence": {
                            "type": "object",
                            "description": "Evidence including screenshot path, dump_tree, error_message",
                            "properties": {
                                "screenshot": {"type": "string"},
                                "dump_tree": {"type": "string"},
                                "error_message": {"type": "string"},
                            },
                        },
                        "suggested_patch": {
                            "type": "object",
                            "description": "Suggested patch for catalog update",
                            "properties": {
                                "patch_type": {"type": "string"},
                                "field": {"type": "string"},
                                "old_value": {"type": "string"},
                                "new_value": {"type": "string"},
                            },
                        },
                    },
                    "required": ["manual_id"],
                },
            ),
            types.Tool(
                name="classify_failure",
                description="Classify failure type from feedback evidence using heuristics. Call after capturing feedback.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "feedback_id": {"type": "string", "description": "Feedback item id to classify"},
                    },
                    "required": ["feedback_id"],
                },
            ),
            types.Tool(
                name="propose_patch",
                description="Propose a catalog patch from a feedback item. The patch must be reviewed before applying.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "feedback_id": {"type": "string", "description": "Feedback item id to generate patch from"},
                    },
                    "required": ["feedback_id"],
                },
            ),
            types.Tool(
                name="apply_patch",
                description=("P4: 把已提议的补丁真正写入动作目录 (actions.jsonl)。"
                             "必须先调用 propose_patch。动作目录直接驱动客户终端上的 GUI "
                             "自动化——一条错误的补丁会导致在客户机器上点错按钮，因此高风险 "
                             "Action 的补丁需要 confirm=true。应用前会捕获索引版本快照，"
                             "可用 CLI `edr-rag index rollback` 回退。"),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "feedback_id": {"type": "string", "description": "已提议补丁的反馈 id"},
                        "confirm": {
                            "type": "boolean",
                            "description": ("高风险 Action 的目录改动需显式确认。默认 false；"
                                            "未确认时返回 requires_confirmation=true 的门控结果。"),
                            "default": False,
                        },
                    },
                    "required": ["feedback_id"],
                },
            ),
            types.Tool(
                name="reject_patch",
                description="P4: 驳回一个补丁提议，并记录驳回原因（审核留痕）。",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "feedback_id": {"type": "string", "description": "待驳回的反馈 id"},
                        "reason": {"type": "string", "description": "驳回原因"},
                    },
                    "required": ["feedback_id"],
                },
            ),
            types.Tool(
                name="reindex",
                description="Trigger re-index from P0 artifacts. Requires .chm files in workdir/input/.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "manual_id": {"type": "string", "description": "Manual identifier"},
                        "product": {"type": "string", "description": "Product name"},
                    },
                    "required": ["manual_id", "product"],
                },
            ),
        ]

    @app.call_tool()
    async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
        try:
            if name == "search_knowledge":
                results = agent.search_sections(arguments["query"], arguments.get("limit", 5), strategy=arguments.get("strategy", "keyword"), rerank=_rerank_arg(arguments.get("rerank")))
            elif name == "search_procedures":
                results = agent.search_procedures(arguments["query"], arguments.get("limit", 5), strategy=arguments.get("strategy", "keyword"), rerank=_rerank_arg(arguments.get("rerank")))
            elif name == "search_actions":
                results = agent.search_actions(arguments["query"], arguments.get("limit", 5), strategy=arguments.get("strategy", "keyword"), rerank=_rerank_arg(arguments.get("rerank")))
            elif name == "search_images":
                results = agent.search_images(arguments["query"], arguments.get("limit", 5), strategy=arguments.get("strategy", "keyword"), rerank=_rerank_arg(arguments.get("rerank")))
            elif name == "get_action":
                result = agent.get_action(arguments["action_id"])
                results = result if result else {"error": "not found"}
            elif name == "get_procedure":
                result = agent.get_procedure(arguments["procedure_id"])
                results = result if result else {"error": "not found"}
            elif name == "ask":
                results = agent.ask_answer(
                    arguments["query"],
                    limit=arguments.get("limit", 5),
                    strategy=arguments.get("strategy", "keyword"),
                    min_confidence=arguments.get("min_confidence"),
                    # P2-5: MCP 工具默认未确认（confirm=false），命中高风险的
                    # requires_confirmation Action 时会返回门控答案；用户显式
                    # 确认后以 confirm=true 重新调用才放行完整步骤。
                    confirm=arguments.get("confirm", False),
                )
            elif name == "capture_feedback":
                fb_id = agent.capture_feedback(arguments)
                results = {"feedback_id": fb_id, "status": "captured"}
            elif name == "classify_failure":
                # P4: 返回带分类依据的结构化结果，而不只是一个 bool。
                results = agent.classify_failure(arguments["feedback_id"])
            elif name == "propose_patch":
                patch = agent.propose_patch(arguments["feedback_id"])
                results = (
                    {"ok": True, "patch": patch}
                    if patch
                    else {"ok": False, "error": "no patch could be derived from this feedback"}
                )
            elif name == "apply_patch":
                results = agent.apply_patch(
                    arguments["feedback_id"], confirm=arguments.get("confirm", False)
                )
            elif name == "reject_patch":
                results = agent.reject_patch(
                    arguments["feedback_id"], reason=arguments.get("reason", "")
                )
            elif name == "reindex":
                results = agent.reindex(arguments["manual_id"], arguments["product"])
            else:
                return [types.TextContent(type="text", text=json.dumps({"error": f"unknown tool: {name}"}))]

            return [types.TextContent(type="text", text=json.dumps(results, ensure_ascii=False, indent=2))]
        except Exception as e:
            return [types.TextContent(type="text", text=json.dumps({"error": str(e)}, ensure_ascii=False))]

    return app


async def run_server(workdir: str):
    app = create_mcp_server(workdir)
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="edr-rag",
                server_version="0.1.0",
                capabilities=app.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


def main():
    parser = argparse.ArgumentParser(description="edr-rag P3 MCP server")
    parser.add_argument("--workdir", required=True, help="Path to work directory with P0/P1/P2 artifacts")
    args = parser.parse_args()

    if not MCP_AVAILABLE:
        print("Error: mcp package not installed. Install with: pip install 'mcp>=1.0.0'", file=sys.stderr)
        sys.exit(1)

    import asyncio
    asyncio.run(run_server(args.workdir))


if __name__ == "__main__":
    main()
