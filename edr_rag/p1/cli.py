"""P1 CLI subcommands for procedure extraction and local search.

Usage:
    python -m tools.edr_rag.p1.cli extract-procedures --manual-id <id> --workdir <path>
    python -m tools.edr_rag.p1.cli search --query <query> --workdir <path> [--type procedure] [--scoring keyword|bm25] [--limit 5]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Point working directory: P0 artifacts expected at <workdir>/
# P1 custom paths:
#   <workdir>/procedures.jsonl
#   <workdir>/chunks.jsonl
#   <workdir>/p1_metadata.json


def _cli_extract_procedures(args: argparse.Namespace):
    from .extract_procedures import extract_procedures_from_file, write_p1_artifacts

    sections_path = Path(args.workdir) / "sections.jsonl"
    if not sections_path.exists():
        print(f"Error: sections not found at {sections_path}", file=sys.stderr)
        print("Run P0 ingestion first: python -m tools.edr_rag.cli ingest ...", file=sys.stderr)
        sys.exit(1)

    # Also read manifest for manual_id / product defaults
    manifest_path = Path(args.workdir) / "manifest.json"
    if manifest_path.exists():
        import json
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manual_id = args.manual_id or manifest.get("manual_id")
        product = args.product or manifest.get("product")
    else:
        manual_id = args.manual_id
        product = args.product or manual_id

    sections_raw = []
    import json
    with sections_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                sections_raw.append(json.loads(line))

    procedures, chunks = extract_procedures_from_file(
        sections_path, product=product,
    )
    write_p1_artifacts(
        Path(args.workdir),
        procedures, chunks, sections_raw,
        manual_id=manual_id,
        product=product,
    )
    print(f"Extracted {len(procedures)} procedures, {len(chunks)} chunks")
    print(f"  procedures -> {Path(args.workdir)/'procedures.jsonl'}")
    print(f"  chunks     -> {Path(args.workdir)/'chunks.jsonl'}")
    print(f"  metadata   -> {Path(args.workdir)/'p1_metadata.json'}")


def _cli_search(args: argparse.Namespace):
    from .search import JsonlCorpus

    # Determine which artifact file to search
    type_map = {
        "section": "sections.jsonl",
        "procedure": "procedures.jsonl",
        "step": "chunks.jsonl",
    }

    file_name = type_map.get(args.type)
    if not file_name:
        print(f"Error: unknown type '{args.type}'. Supported: section, procedure, step", file=sys.stderr)
        sys.exit(1)

    path = Path(args.workdir) / file_name
    if not path.exists():
        print(f"Error: {path} not found", file=sys.stderr)
        suggest = file_name != "sections.jsonl" and (Path(args.workdir) / "sections.jsonl").exists()
        if suggest:
            hint = "\n  Available: sections.jsonl (P0). Did you run extract-procedures first?"
        else:
            hint = ""
        print(f"  P0 artifacts at {Path(args.workdir)}{hint}", file=sys.stderr)
        sys.exit(1)

    corpus = JsonlCorpus.from_file(path)

    if args.scoring == "bm25":
        results = corpus.bm25_search(args.query, limit=args.limit)
    else:
        results = corpus.keyword_search(args.query, limit=args.limit)

    if not results:
        print(f"No results found for query: {args.query}")
        sys.exit(0)

    print(f"Search results for '{args.query}' ({args.type}, {args.scoring}):")
    for r in results:
        print(f"\n  [{r['type']}] {r['id']}  (score: {r['score']})")
        if r.get("source") and r["source"].get("file"):
            print(f"     source: {r['source']['file']}")
        print(f"     title: {r['title']}")
        snip = r.get("snippet", "")
        if snip:
            print(f"     snippet: {snip[:100]}")


def main():
    parser = argparse.ArgumentParser(
        description="P1: procedure extraction and local search CLI",
    )
    subparsers = parser.add_subparsers(dest="subcommand")

    # extract-procedures
    ep_parser = subparsers.add_parser(
        "extract-procedures",
        help="Extract procedures from P0 sections.jsonl",
    )
    ep_parser.add_argument("--manual-id", default=None, help="Manual identifier (overrides manifest)")
    ep_parser.add_argument("--product", default=None, help="Product name (overrides manifest)")
    ep_parser.add_argument("--workdir", required=True, help="Work directory (same as P0 --workdir)")

    # search
    search_parser = subparsers.add_parser(
        "search",
        help="Search P0/P1 artifacts",
    )
    search_parser.add_argument("--query", required=True, help="Search query (keyword or phrase)")
    search_parser.add_argument("--workdir", required=True, help="Work directory (same as P0 --workdir)")
    search_parser.add_argument("--type", default="procedure", choices=["section", "procedure", "step"],
                               help="Artifact type to search (default: procedure)")
    search_parser.add_argument("--scoring", default="keyword", choices=["keyword", "bm25"],
                               help="Scoring method (default: keyword)")
    search_parser.add_argument("--limit", type=int, default=5, help="Max results")

    args = parser.parse_args()
    if not args.subcommand:
        parser.print_help()
        sys.exit(1)

    if args.subcommand == "extract-procedures":
        _cli_extract_procedures(args)
    elif args.subcommand == "search":
        _cli_search(args)


if __name__ == "__main__":
    main()
