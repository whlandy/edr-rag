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
from typing import Optional

from ..retrieval import RetrievalService


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
    ) -> list[dict]:
        """Search the action catalog (delegated to RetrievalService).

        strategy: 'keyword' | 'bm25' | 'vector' | 'hybrid' (default 'keyword').
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
    ) -> list[dict]:
        """Search P1 procedures (fallback if action_search has no results).

        strategy: 'keyword' | 'bm25' | 'vector' | 'hybrid' (default 'keyword').
        `scoring` 为历史别名参数，与 `strategy` 等价；两者都传时以
        `strategy` 为准（保持 backward compatibility）。
        """
        if strategy is None:
            strategy = scoring or "keyword"
        return self._service.search(
            query,
            corpus_type="procedures",
            strategy=strategy,
            top_k=limit,
        )

    # ---------- Manual search (last priority) ----------

    def manual_search(
        self,
        query: str,
        limit: int = 5,
        scoring: Optional[str] = None,
        strategy: Optional[str] = None,
    ) -> list[dict]:
        """Search P0 sections (for explanation or troubleshooting only).

        strategy: 'keyword' | 'bm25' | 'vector' | 'hybrid' (default 'keyword').
        `scoring` 为历史别名参数，与 `strategy` 等价；两者都传时以
        `strategy` 为准。
        """
        if strategy is None:
            strategy = scoring or "keyword"
        return self._service.search(
            query,
            corpus_type="sections",
            strategy=strategy,
            top_k=limit,
        )

    # ---------- Planner orchestration ----------

    def planner_retrieve(
        self,
        user_request: str,
        limit: int = 5,
        strategy: str = "keyword",
    ) -> dict:
        """Orchestrate the full planner retrieval flow (action → procedure → manual).

        Delegates to RetrievalService.planner_retrieve which preserves the
        §6 priority order and threads `strategy` to each retrieval tier.
        """
        return self._service.planner_retrieve(user_request, limit=limit, strategy=strategy)
