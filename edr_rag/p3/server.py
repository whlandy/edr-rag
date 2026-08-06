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

    def __init__(self, workdir: str):
        self.workdir = Path(workdir)
        self.feedback_store = FeedbackStore(str(self.workdir / "feedback.jsonl"))

    # ----- Knowledge search tools -----

    def search_sections(self, query: str, limit: int = 5) -> list[dict]:
        """Search P0 sections.jsonl by keyword."""
        sections_path = self.workdir / "sections.jsonl"
        if not sections_path.exists():
            return []
        from edr_rag.p1.search import JsonlCorpus
        corpus = JsonlCorpus.from_file(sections_path)
        return corpus.keyword_search(query, limit=limit)

    def search_procedures(self, query: str, limit: int = 5) -> list[dict]:
        """Search P1 procedures.jsonl by keyword."""
        procs_path = self.workdir / "procedures.jsonl"
        if not procs_path.exists():
            return []
        from edr_rag.p1.search import JsonlCorpus
        corpus = JsonlCorpus.from_file(procs_path)
        return corpus.keyword_search(query, limit=limit)

    def search_actions(self, query: str, limit: int = 5) -> list[dict]:
        """Search P2 actions.jsonl by keyword."""
        actions_path = self.workdir / "actions.jsonl"
        if not actions_path.exists():
            return []
        from edr_rag.p1.search import JsonlCorpus
        corpus = JsonlCorpus.from_file(actions_path)
        return corpus.keyword_search(query, limit=limit)

    def search_chunks(self, query: str, limit: int = 5) -> list[dict]:
        """Search P1 chunks.jsonl by keyword."""
        chunks_path = self.workdir / "chunks.jsonl"
        if not chunks_path.exists():
            return []
        from edr_rag.p1.search import JsonlCorpus
        corpus = JsonlCorpus.from_file(chunks_path)
        return corpus.keyword_search(query, limit=limit)

    def get_action(self, action_id: str) -> Optional[dict]:
        """Retrieve a specific action by action_id."""
        actions_path = self.workdir / "actions.jsonl"
        if not actions_path.exists():
            return None
        with actions_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    if data.get("action_id") == action_id:
                        return data
        return None

    def get_procedure(self, procedure_id: str) -> Optional[dict]:
        """Retrieve a specific procedure by id."""
        procs_path = self.workdir / "procedures.jsonl"
        if not procs_path.exists():
            return None
        with procs_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    if data.get("id") == procedure_id:
                        return data
        return None

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

    def classify_failure(self, feedback_id: str) -> bool:
        """Classify a failure type from evidence for a feedback item. Placeholder.

        In P3 this uses heuristics (no LLM). Returns True if reclassified.
        """
        items = self.feedback_store.read_all()
        target = None
        for item in items:
            if item.feedback_id == feedback_id:
                target = item
                break
        if not target:
            return False

        # Heuristic: if evidence has error_message, try to classify
        if target.evidence and target.evidence.error_message:
            msg = target.evidence.error_message.lower()
            if "not found" in msg and "button" in msg:
                target.failure_type = "button_text_not_found"
            elif "timeout" in msg:
                target.failure_type = "timeout"
            elif "network" in msg or "connection" in msg:
                target.failure_type = "network_error"
            elif "ui" in msg or "element" in msg:
                target.failure_type = "missing_ui_label"

        target.status = "needs_review"
        self.feedback_store.update_status(feedback_id, target.status)
        # Update failure_type in file by rewriting
        items = self.feedback_store.read_all()
        for item in items:
            if item.feedback_id == feedback_id:
                item.failure_type = target.failure_type
                break
        with open(self.feedback_store.path, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item.to_jsonl(), ensure_ascii=False) + "\n")
        return True

    def propose_patch(self, feedback_id: str) -> Optional[dict]:
        """Propose a patch from a feedback item. Placeholder.

        Returns the suggested patch dict, or None if no patch to propose.
        """
        items = self.feedback_store.read_all()
        target = None
        for item in items:
            if item.feedback_id == feedback_id:
                target = item
                break
        if not target or not target.suggested_patch:
            return None

        self.feedback_store.update_status(feedback_id, "patch_proposed")
        return target.suggested_patch.to_jsonl()

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
                results = agent.search_sections(arguments["query"], arguments.get("limit", 5))
            elif name == "search_procedures":
                results = agent.search_procedures(arguments["query"], arguments.get("limit", 5))
            elif name == "search_actions":
                results = agent.search_actions(arguments["query"], arguments.get("limit", 5))
            elif name == "get_action":
                result = agent.get_action(arguments["action_id"])
                results = result if result else {"error": "not found"}
            elif name == "get_procedure":
                result = agent.get_procedure(arguments["procedure_id"])
                results = result if result else {"error": "not found"}
            elif name == "capture_feedback":
                fb_id = agent.capture_feedback(arguments)
                results = {"feedback_id": fb_id, "status": "captured"}
            elif name == "classify_failure":
                ok = agent.classify_failure(arguments["feedback_id"])
                results = {"ok": ok, "feedback_id": arguments["feedback_id"]}
            elif name == "propose_patch":
                patch = agent.propose_patch(arguments["feedback_id"])
                results = {"ok": patch is not None, "patch": patch} if patch else {"ok": False, "error": "no patch to propose"}
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
