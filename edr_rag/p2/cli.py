"""P2 CLI subcommands for action catalog building and search.

Usage:
    python -m tools.edr_rag.p2.cli build-actions --workdir <path>
    python -m tools.edr_rag.p2.cli action-search --query <q> --workdir <path> [--limit 5] [--strategy keyword|bm25|vector|hybrid] [--risk medium] [--status published]
    python -m tools.edr_rag.p2.cli get-action <action_id> --workdir <path>
    python -m tools.edr_rag.p2.cli planner-retrieve --query <q> --workdir <path> [--strategy keyword] [--limit 5]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional


def _cli_build_actions(args: argparse.Namespace):
    from .extract_actions import build_actions_from_file, write_p2_artifacts

    procedures_path = Path(args.workdir) / "procedures.jsonl"
    if not procedures_path.exists():
        print(f"Error: procedures not found at {procedures_path}", file=sys.stderr)
        print("Run P1 extract-procedures first", file=sys.stderr)
        sys.exit(1)

    entries = build_actions_from_file(procedures_path, product=args.product)

    if not entries:
        print("No action candidates generated (no procedures found or all malformed)")
        sys.exit(1)

    jsonl_path, yaml_path = write_p2_artifacts(Path(args.workdir), entries)

    by_risk = {}
    by_status = {}
    for e in entries:
        by_risk[e.risk] = by_risk.get(e.risk, 0) + 1
        by_status[e.status] = by_status.get(e.status, 0) + 1

    print(f"Generated {len(entries)} action candidates:")
    print(f"  by risk:   {', '.join(f'{k}={v}' for k, v in sorted(by_risk.items()))}")
    print(f"  by status: {', '.join(f'{k}={v}' for k, v in sorted(by_status.items()))}")
    print(f"  JSONL: {jsonl_path}")
    print(f"  YAML:  {yaml_path}")


def _rerank_arg(value: Optional[str]) -> Optional[bool]:
    """把 CLI 的 --rerank (auto/on/off) 映射为 service 的 Optional[bool]。"""
    if value is None or value == "auto":
        return None
    return value == "on"


def _cli_action_search(args: argparse.Namespace):
    from ..retrieval import RetrievalService

    service = RetrievalService(args.workdir)
    filters = {}
    if args.risk:
        filters["risk"] = args.risk
    if args.status:
        filters["status"] = args.status
    results = service.search(
        args.query,
        corpus_type="actions",
        strategy=args.strategy,
        filters=filters,
        top_k=args.limit,
        rerank=_rerank_arg(args.rerank),
    )

    if not results:
        print(f"No matching actions found for: {args.query}")
        if args.risk:
            print(f"  (filtered by risk={args.risk})")
        if args.status:
            print(f"  (filtered by status={args.status})")
        sys.exit(0)

    print(f"Action search results for '{args.query}':")
    for r in results:
        print(f"\n  [{r.get('risk', '-')}] {r.get('id', r.get('title', '?'))}  (score: {r.get('score')})")
        print(f"     title: {r.get('title')}")
        if r.get("status") is not None:
            print(f"     status: {r.get('status')}")
        if r.get("confidence") is not None:
            print(f"     confidence: {r.get('confidence')}")
        if r.get("snippet"):
            print(f"     {r['snippet']}")


def _cli_get_action(args: argparse.Namespace):
    from ..retrieval import RetrievalService

    service = RetrievalService(args.workdir)
    action = service.get("actions", args.action_id, key_field="action_id")

    if not action:
        print(f"Action '{args.action_id}' not found", file=sys.stderr)
        sys.exit(1)

    print(json.dumps(action, ensure_ascii=False, indent=2))


def _cli_planner_retrieve(args: argparse.Namespace):
    from ..retrieval import RetrievalService

    service = RetrievalService(args.workdir)
    result = service.planner_retrieve(
        args.query, limit=args.limit, strategy=args.strategy, rerank=_rerank_arg(args.rerank)
    )

    print(f"Planner retrieval for: {args.query}")
    print()

    if result.get("actions"):
        print("=== Actions (priority 1) ===")
        for r in result["actions"]:
            print(f"  [{r.get('risk', '-')}] {r.get('id', r.get('title', '?'))} (score: {r.get('score')})")
            print(f"    title: {r.get('title')}")
            if r.get("status") is not None:
                print(f"    status: {r.get('status')}")
            if r.get("snippet"):
                print(f"    snippet: {r['snippet'][:100]}")
            print()
    elif result.get("procedures"):
        print("=== Procedures (priority 2: no actions found) ===")
        for r in result["procedures"]:
            print(f"  {r['id']}  (score: {r['score']})")
            print(f"    title: {r['title']}")
            if r.get("snippet"):
                print(f"    snippet: {r['snippet'][:100]}")
            print()
    elif result.get("sections"):
        print("=== Sections (priority 3: fallback) ===")
        for r in result["sections"]:
            print(f"  {r['id']}  (score: {r['score']})")
            print(f"    title: {r['title']}")
            if r.get("snippet"):
                print(f"    snippet: {r['snippet'][:100]}")
            print()
    else:
        print("  No results found.")


def _cli_answer(args: argparse.Namespace):
    from .api import EdrRagAPI
    from .synthesize import answer_to_text

    api = EdrRagAPI(args.workdir)
    if args.confirm is not None:
        # P2-5: 显式确认门控路径。命中 requires_confirmation 的 Action 且未
        # confirm 时返回门控答案（扣留可执行步骤）。
        result = api.ask_answer(
            args.query, limit=args.limit, strategy=args.strategy,
            min_confidence=args.min_confidence, confirm=args.confirm,
        )
    else:
        result = api.synthesize(
            args.query, limit=args.limit, strategy=args.strategy,
            min_confidence=args.min_confidence,
        )
    if args.json:
        import json as _json
        print(_json.dumps(result, ensure_ascii=False, indent=2))
        return
    # result 是 dict，转回 SynthesizedAnswer 用于文本渲染（过滤掉 agent 便捷
    # 字段与 P2-5 confirmation 元数据，避免 dataclass 收到未知 kwargs）。
    from .synthesize import SynthesizedAnswer
    field_names = {f for f in SynthesizedAnswer.__dataclass_fields__}
    base = {k: v for k, v in result.items() if k in field_names}
    print(answer_to_text(SynthesizedAnswer(**base)))
    print()
    if result.get("confirmation_status") == "required":
        print("[confirmation_status=required] 该答案含需显式确认的高风险 Action，"
              "可执行步骤已扣留；请确认后以 --confirm 重跑以放行。")
    print(f"[generated_by={result['generated_by']} confidence={result['confidence']} "
          f"requires_confirmation={str(result['requires_confirmation']).lower()} "
          f"answer_status={result['answer_status']} "
          f"confirmation_status={result.get('confirmation_status', '-')} "
          f"steps={len(result.get('steps') or [])} "
          f"preconditions={len(result.get('preconditions') or [])}]")


def main():
    parser = argparse.ArgumentParser(
        description="P2: action catalog and planner retrieval CLI",
    )
    subparsers = parser.add_subparsers(dest="subcommand")

    # build-actions
    build_parser = subparsers.add_parser(
        "build-actions",
        help="Build action candidates from P1 procedures",
    )
    build_parser.add_argument("--workdir", required=True, help="Work directory (same as P0/P1)")
    build_parser.add_argument("--product", default=None, help="Product name override")

    # action-search
    search_parser = subparsers.add_parser(
        "action-search",
        help="Search action catalog",
    )
    search_parser.add_argument("--query", required=True, help="Search query")
    search_parser.add_argument("--workdir", required=True, help="Work directory")
    search_parser.add_argument("--limit", type=int, default=5, help="Max results")
    search_parser.add_argument("--strategy", default="keyword",
                               choices=["keyword", "bm25", "vector", "hybrid"],
                               help="Retrieval strategy (default: keyword; vector degrades to bm25 and hybrid to RRF fusion without optional RAG deps)")
    search_parser.add_argument("--rerank", type=str, choices=["auto", "on", "off"], default=None,
                               help="Cross-encoder rerank toggle for hybrid (default: auto = enable when deps available, else skip)")
    search_parser.add_argument("--risk", default=None, choices=["low", "medium", "high"],
                               help="Filter by risk level")
    search_parser.add_argument("--status", default=None,
                               choices=["candidate", "needs_review", "approved", "published"],
                               help="Filter by status")

    # get-action
    get_parser = subparsers.add_parser(
        "get-action",
        help="Get full action catalog entry by ID",
    )
    get_parser.add_argument("action_id", help="Action ID (e.g. network.configure_proxy)")
    get_parser.add_argument("--workdir", required=True, help="Work directory")

    # planner-retrieve
    planner_parser = subparsers.add_parser(
        "planner-retrieve",
        help="Full planner retrieval flow (action → procedure → manual)",
    )
    planner_parser.add_argument("--query", required=True, help="User request or query")
    planner_parser.add_argument("--workdir", required=True, help="Work directory")
    planner_parser.add_argument("--limit", type=int, default=5, help="Max results per tier")
    planner_parser.add_argument("--strategy", default="keyword",
                                choices=["keyword", "bm25", "vector", "hybrid"],
                                help="Retrieval strategy threaded to every tier (default: keyword)")

    # answer (P2-1) — structured answer synthesizer
    answer_parser = subparsers.add_parser(
        "answer",
        help="Generate a structured answer from retrieval context (P2-1)",
    )
    answer_parser.add_argument("--query", required=True, help="User request / question")
    answer_parser.add_argument("--workdir", required=True, help="Work directory")
    answer_parser.add_argument("--limit", type=int, default=5, help="Retrieval limit per tier")
    answer_parser.add_argument("--strategy", default="keyword",
                               choices=["keyword", "bm25", "vector", "hybrid"],
                               help="Retrieval strategy (default: keyword)")
    answer_parser.add_argument("--json", action="store_true",
                               help="Emit the full structured answer as JSON")
    answer_parser.add_argument("--min-confidence", type=float, default=None,
                               help="证据不足拒答的置信度阈值 (P2-3, 默认 0.4)")
    answer_parser.add_argument("--confirm", type=lambda s: s.strip().lower() in ("1", "true", "yes"),
                               default=None, nargs="?", const=True,
                               help="P2-5 显式确认高风险 Action：传 --confirm 表示确认放行完整步骤；"
                                    "不传则命中 requires_confirmation 时返回门控答案。")

    args = parser.parse_args()
    if not args.subcommand:
        parser.print_help()
        sys.exit(1)

    if args.subcommand == "build-actions":
        _cli_build_actions(args)
    elif args.subcommand == "action-search":
        _cli_action_search(args)
    elif args.subcommand == "get-action":
        _cli_get_action(args)
    elif args.subcommand == "planner-retrieve":
        _cli_planner_retrieve(args)
    elif args.subcommand == "answer":
        _cli_answer(args)


if __name__ == "__main__":
    main()
