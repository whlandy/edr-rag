"""P2 CLI subcommands for action catalog building and search.

Usage:
    python -m tools.edr_rag.p2.cli build-actions --workdir <path>
    python -m tools.edr_rag.p2.cli action-search --query <q> --workdir <path> [--limit 5] [--risk medium] [--status published]
    python -m tools.edr_rag.p2.cli get-action <action_id> --workdir <path>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


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


def _cli_action_search(args: argparse.Namespace):
    from .api import EdrRagAPI

    api = EdrRagAPI(args.workdir)
    results = api.action_search(
        args.query,
        limit=args.limit,
        risk_filter=args.risk,
        status_filter=args.status,
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
        print(f"\n  [{r['risk']}] {r['id']}  (score: {r['score']})")
        print(f"     title: {r['title']}")
        print(f"     status: {r['status']} | confidence: {r['confidence']}")
        if r.get("snippet"):
            print(f"     {r['snippet']}")


def _cli_get_action(args: argparse.Namespace):
    from .api import EdrRagAPI

    api = EdrRagAPI(args.workdir)
    action = api.get_action(args.action_id)

    if not action:
        print(f"Action '{args.action_id}' not found", file=sys.stderr)
        sys.exit(1)

    print(json.dumps(action, ensure_ascii=False, indent=2))


def _cli_planner_retrieve(args: argparse.Namespace):
    from .api import EdrRagAPI

    api = EdrRagAPI(args.workdir)
    result = api.planner_retrieve(args.query, limit=args.limit)

    print(f"Planner retrieval for: {args.query}")
    print()

    if result.get("actions"):
        print("=== Actions (priority 1) ===")
        for r in result["actions"]:
            print(f"  [{r['risk']}] {r['id']} (score: {r['score']})")
            print(f"    title: {r['title']}")
            print(f"    status: {r['status']}")
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


if __name__ == "__main__":
    main()
