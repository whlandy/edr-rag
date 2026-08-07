"""P1 CLI subcommands for procedure extraction and local search.

Usage:
    python -m tools.edr_rag.p1.cli extract-procedures --manual-id <id> --workdir <path>
    python -m tools.edr_rag.p1.cli search --query <query> --workdir <path> [--type procedure] [--scoring keyword|bm25] [--strategy keyword|bm25] [--limit 5]
    python -m tools.edr_rag.p1.cli ingest-llama --workdir <path> [--corpus chunks] [--force-llama]
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

    # 统一检索策略：--strategy 为统一命名，--scoring 为历史别名。
    # 两者都指定时以 --strategy 为准（保持 backward compatibility）。
    strategy = args.strategy or args.scoring or "keyword"

    if strategy == "bm25":
        results = corpus.bm25_search(args.query, limit=args.limit)
    else:
        results = corpus.keyword_search(args.query, limit=args.limit)

    if not results:
        print(f"No results found for query: {args.query}")
        sys.exit(0)

    print(f"Search results for '{args.query}' ({args.type}, {strategy}):")
    for r in results:
        print(f"\n  [{r['type']}] {r['id']}  (score: {r['score']})")
        if r.get("source") and r["source"].get("file"):
            print(f"     source: {r['source']['file']}")
        print(f"     title: {r['title']}")
        snip = r.get("snippet", "")
        if snip:
            print(f"     snippet: {snip[:100]}")


def _cli_ingest_llama(args: argparse.Namespace):
    """运行可选的 LlamaIndex 摄取管线（缺依赖自动降级到标准库结构化切分）。"""
    from .llama_ingest import IngestResult, llama_index_available, run_ingestion_pipeline

    workdir = Path(args.workdir)
    corpus = args.corpus

    if args.force_llama and not llama_index_available():
        print(
            "Error: --force-llama requires llama-index-core. "
            "Install with: pip install -e '.[rag]'",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        result: IngestResult = run_ingestion_pipeline(
            workdir,
            corpus,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            force_llama=args.force_llama,
        )
    except Exception as e:  # force-llama 时 LlamaIndexUnavailable 会冒出
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    mode = result.mode
    if mode == "llama":
        mode_note = "LlamaIndex IngestionPipeline"
    else:
        mode_note = "标准库结构化切分 (llama-index-core 未安装或不可用)"

    print(
        f"ingested corpus={corpus} mode={mode} ({mode_note})"
    )
    print(f"  source_records     = {result.source_records}")
    print(f"  produced_nodes     = {result.produced_nodes}")
    print(f"  new_nodes          = {result.new_nodes}")
    print(f"  duplicates_skipped = {result.duplicates_skipped}")
    print(f"  cache              = {result.cache_path}")
    if result.error:
        print(f"  note               = {result.error}")


def _cli_build_nodes(args: argparse.Namespace):
    """构建结构化节点（章节/Procedure/Step/Action 分层），写出 nodes.jsonl。"""
    from .nodes import (
        build_context_window,
        build_structured_nodes_file,
        load_structured_nodes,
    )

    workdir = Path(args.workdir)
    nodes, nodes_path = build_structured_nodes_file(workdir)
    counts = {"section": 0, "procedure": 0, "step": 0, "action": 0}
    for n in nodes:
        counts[n.level] = counts.get(n.level, 0) + 1
    print(f"built {len(nodes)} structured nodes -> {nodes_path}")
    for lvl in ("section", "procedure", "step", "action"):
        print(f"  {lvl:<10} {counts.get(lvl, 0)}")

    if args.context and nodes:
        loaded = load_structured_nodes(nodes_path)
        node = loaded[0]
        window = build_context_window(node.id, loaded)
        print(f"\ncontext window for {node.id} ({node.level}):")
        for w in window:
            print(f"  [{w.level}] {w.id}")
            print("     " + (w.text[:60].replace("\n", " ") if w.text else ""))


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
                               help="Scoring method (default: keyword; historical alias of --strategy)")
    search_parser.add_argument("--strategy", default=None, choices=["keyword", "bm25", "vector", "hybrid"],
                               help="Retrieval strategy (unified naming; overrides --scoring). vector/hybrid degrade to bm25 at this level")
    search_parser.add_argument("--limit", type=int, default=5, help="Max results")

    # ingest-llama (P1-1: 可选 LlamaIndex 摄取管线)
    il_parser = subparsers.add_parser(
        "ingest-llama",
        help="Run optional LlamaIndex ingestion pipeline (degrades to stdlib split)",
    )
    il_parser.add_argument("--workdir", required=True, help="Work directory (same as P0 --workdir)")
    il_parser.add_argument("--corpus", default="chunks",
                           choices=["chunks", "procedures", "sections", "actions"],
                           help="Corpus type to ingest (default: chunks)")
    il_parser.add_argument("--chunk-size", type=int, default=512,
                           help="Max node length in chars (default: 512)")
    il_parser.add_argument("--overlap", type=int, default=0,
                           help="Overlap for re-splitting long nodes (default: 0)")
    il_parser.add_argument("--force-llama", action="store_true",
                           help="Require LlamaIndex; fail instead of degrading if absent")

    # build-nodes (P1-2: 结构化 node 分层)
    bn_parser = subparsers.add_parser(
        "build-nodes",
        help="Build structured nodes (section/procedure/step/action) from artifacts",
    )
    bn_parser.add_argument("--workdir", required=True, help="Work directory (same as P0 --workdir)")
    bn_parser.add_argument("--context", action="store_true",
                           help="Print a context window for the first node as a demo")

    args = parser.parse_args()
    if not args.subcommand:
        parser.print_help()
        sys.exit(1)

    if args.subcommand == "extract-procedures":
        _cli_extract_procedures(args)
    elif args.subcommand == "search":
        _cli_search(args)
    elif args.subcommand == "ingest-llama":
        _cli_ingest_llama(args)
    elif args.subcommand == "build-nodes":
        _cli_build_nodes(args)


if __name__ == "__main__":
    main()
