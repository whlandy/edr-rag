"""P2 Python API — planner retrieval interface.

Design (§6, Planner Retrieval Flow):

  The planner should call the manual knowledge layer in this order:
    1. action_search      — search action catalog first
    2. get_action         — get full action detail
    3. procedure_search   — if action_search has no result
    4. manual_search      — only for explanation or troubleshooting

All search functions return results as dicts (JSON-serializable), so they
can be passed directly to the planner or over MCP.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

from .extract_actions import build_actions_from_file
from ..p1.search import JsonlCorpus, tokenize


class EdrRagAPI:
    """P2 Retrieval API for the EDR-RAG system.

    Provides unified access to:
      - action_search: search action catalog entries
      - get_action: get full action entry by id
      - procedure_search: search P1 procedures (fallback)
      - manual_search: search P0 sections (troubleshooting only)
    """

    def __init__(self, workdir: str | Path):
        self.workdir = Path(workdir)

        # Lazy-loaded corpora
        self._actions: list[dict] | None = None
        self._procedures_corpus: JsonlCorpus | None = None
        self._sections_corpus: JsonlCorpus | None = None

    # ---------- Corpora loading (lazy) ----------

    def _load_actions(self) -> list[dict]:
        if self._actions is not None:
            return self._actions
        path = self.workdir / "actions.jsonl"
        if not path.exists():
            self._actions = []
            return self._actions
        records = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        self._actions = records
        return records

    def _load_procedure_corpus(self) -> JsonlCorpus | None:
        if self._procedures_corpus is not None:
            return self._procedures_corpus
        path = self.workdir / "procedures.jsonl"
        if not path.exists():
            self._procedures_corpus = JsonlCorpus([])
            return None
        self._procedures_corpus = JsonlCorpus.from_file(path)
        return self._procedures_corpus

    def _load_section_corpus(self) -> JsonlCorpus | None:
        if self._sections_corpus is not None:
            return self._sections_corpus
        path = self.workdir / "sections.jsonl"
        if not path.exists():
            self._sections_corpus = JsonlCorpus([])
            return None
        self._sections_corpus = JsonlCorpus.from_file(path)
        return self._sections_corpus

    # ---------- Action search (first priority) ----------

    def action_search(
        self,
        query: str,
        limit: int = 5,
        risk_filter: Optional[str] = None,
        status_filter: Optional[str] = None,
    ) -> list[dict]:
        """Search the action catalog.

        Matches against action_id, title, module, preconditions, and
        procedure step instructions.

        Args:
            query: Free-text search query (supports CJK).
            limit: Max results to return.
            risk_filter: Optional filter: 'low', 'medium', 'high'.
            status_filter: Optional filter: 'candidate', 'needs_review',
                          'approved', 'published'.

        Returns:
            List of result dicts with id, title, risk, status, snippet.
        """
        actions = self._load_actions()
        if not actions:
            return []

        query_tokens = set(tokenize(query))
        if not query_tokens:
            return []

        scored = []
        for action in actions:
            # Build searchable text
            searchable = " ".join([
                action.get("action_id", ""),
                action.get("title", ""),
                action.get("module", ""),
                " ".join(action.get("preconditions", [])),
                " ".join(s.get("source_instruction", "") for s in action.get("procedure", [])),
            ])

            score = 0.0
            for t in query_tokens:
                if t in searchable:
                    # Title and action_id matches weighted higher
                    weight = 3.0 if (t in action.get("title", "") or t in action.get("action_id", "")) else 1.0
                    score += searchable.count(t) * weight

            if score > 0:
                # Apply filters
                if risk_filter and action.get("risk") != risk_filter:
                    continue
                if status_filter and action.get("status") != status_filter:
                    continue
                scored.append((score, action))

        scored.sort(key=lambda x: -x[0])

        results = []
        for score, action in scored[:limit]:
            snippet = action.get("title", "")
            pre = action.get("preconditions", [])
            if pre:
                snippet += " | 前置条件: " + "；".join(pre)
            results.append({
                "id": action.get("action_id", ""),
                "title": action.get("title", ""),
                "risk": action.get("risk", ""),
                "status": action.get("status", ""),
                "confidence": action.get("confidence", 0.0),
                "score": round(score, 3),
                "snippet": snippet,
            })

        return results

    def get_action(self, action_id: str) -> Optional[dict]:
        """Get the full action catalog entry by action_id (stripped).

        Matches against action_id (e.g. 'network.configure_proxy'
        or 'procedure.network.configure_proxy' — both work).
        """
        actions = self._load_actions()
        if not actions:
            return None

        # Normalize lookup key: strip 'action.' prefix if present
        lookup = action_id
        for prefix in ("action.", "procedure."):
            if lookup.startswith(prefix):
                lookup = lookup[len(prefix):]
                break

        for action in actions:
            aid = action.get("action_id", "")
            if aid == lookup:
                return action

        return None

    # ---------- Procedure search (second priority) ----------

    def procedure_search(
        self,
        query: str,
        limit: int = 5,
        scoring: str = "keyword",
    ) -> list[dict]:
        """Search P1 procedures (fallback if action_search has no results).

        Args:
            query: Search query string.
            limit: Max results.
            scoring: 'keyword' or 'bm25'.

        Returns:
            List of result dicts with id, title, snippet, source.
        """
        corpus = self._load_procedure_corpus()
        if not corpus:
            return []

        if scoring == "bm25":
            return corpus.bm25_search(query, limit=limit)
        return corpus.keyword_search(query, limit=limit)

    # ---------- Manual search (last priority) ----------

    def manual_search(
        self,
        query: str,
        limit: int = 5,
        scoring: str = "keyword",
    ) -> list[dict]:
        """Search P0 sections (for explanation or troubleshooting only).

        Args:
            query: Search query string.
            limit: Max results.
            scoring: 'keyword' or 'bm25'.

        Returns:
            List of result dicts with id, title, snippet, source.
        """
        corpus = self._load_section_corpus()
        if not corpus:
            return []

        if scoring == "bm25":
            return corpus.bm25_search(query, limit=limit)
        return corpus.keyword_search(query, limit=limit)

    # ---------- Planner orchestration ----------

    def planner_retrieve(
        self,
        user_request: str,
        limit: int = 5,
    ) -> dict:
        """Orchestrate the full planner retrieval flow.

        Implements design §6 planner retrieval order:
          1. Search action catalog first.
          2. If no actions found, search procedures.
          3. Only fall back to manual sections for explanation.

        Returns:
            dict with:
              - query: original query
              - actions: list of action search results
              - procedures: list of procedure search results (if actions empty)
              - sections: list of section search results (only if both empty)
        """
        actions = self.action_search(user_request, limit=limit)

        result = {
            "query": user_request,
            "actions": actions,
        }

        if not actions:
            procedures = self.procedure_search(user_request, limit=limit)
            result["procedures"] = procedures

            if not procedures:
                sections = self.manual_search(user_request, limit=limit)
                result["sections"] = sections
        else:
            result["procedures"] = []
            result["sections"] = []

        return result
