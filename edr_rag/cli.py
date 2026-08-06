"""CLI entry point for the CHM ingestion pipeline.

Commands:
  python -m tools.edr_rag.cli ingest --chm <path> [options]

    P0 ingest: extract CHM + TOC + HTML → manifest.json + sections.jsonl + assets.jsonl

  python -m tools.edr_rag.cli validate --workdir <path> [--manual-id <id>]
  edr-rag index --workdir <path> [--type all] [--embed-model <model>]
  edr-rag search --workdir <path> --query <text> [--strategy hybrid]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from . import normalize

logger = logging.getLogger(__name__)


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )


def _cmd_ingest(args: argparse.Namespace) -> None:
    """Run P0 ingestion."""
    chm_path = Path(args.chm).expanduser().resolve()
    manual_id = args.manual_id
    product = args.product
    workdir = Path(args.workdir).expanduser().resolve()

    if not chm_path.is_file():
        print(f"Error: CHM file not found: {chm_path}", file=sys.stderr)
        sys.exit(1)

    manifest = normalize.run_ingest(chm_path, manual_id, product, workdir)
    print(f"{json.dumps(manifest.to_json(), ensure_ascii=False, indent=2)}")


def _cmd_validate(args: argparse.Namespace) -> None:
    """Validate the output of a P0 ingestion run.

    P0 contract check (review Blocker 5):
    sections.jsonl:
      required: id, manual_id, title, content, section_path, source_ref.relative_path
    assets.jsonl:
      required: id, manual_id, type, path
    """
    workdir = Path(args.workdir).expanduser().resolve()
    manual_id = args.manual_id

    try:
        section_count, asset_count, errors = normalize.validate_workdir(workdir)
    except FileNotFoundError as e:
        print(f"Validation FAILED: {e}", file=sys.stderr)
        sys.exit(1)

    if manual_id:
        manifest_path = workdir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("manual_id") != manual_id:
            print(
                f"Warning: manifest manual_id ({manifest.get('manual_id')}) "
                f"does not match argument ({manual_id})",
                file=sys.stderr,
            )

    if errors:
        sys.exit(1)


def _cmd_index(args: argparse.Namespace) -> None:
    from .vector import ARTIFACT_FILES, LlamaIndexStore, LlamaIndexUnavailableError

    artifact_types = (
        [kind for kind, filename in ARTIFACT_FILES.items() if (Path(args.workdir) / filename).exists()]
        if args.type == "all" else [args.type]
    )
    if not artifact_types:
        print(f"Error: no supported JSONL artifacts found in {args.workdir}", file=sys.stderr)
        sys.exit(1)
    try:
        counts = LlamaIndexStore(
            args.workdir, embed_model=args.embed_model, index_dir=args.index_dir
        ).build(artifact_types, show_progress=args.show_progress)
    except (FileNotFoundError, ValueError, LlamaIndexUnavailableError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    print(json.dumps({"indexed": counts, "embed_model": args.embed_model}, ensure_ascii=False, indent=2))


def _cmd_search(args: argparse.Namespace) -> None:
    from .p1.search import JsonlCorpus
    from .vector import (
        ARTIFACT_FILES,
        LlamaIndexStore,
        LlamaIndexUnavailableError,
        reciprocal_rank_fusion,
    )

    source = Path(args.workdir) / ARTIFACT_FILES[args.type]
    lexical = []
    if args.strategy in ("keyword", "bm25", "hybrid"):
        if not source.exists():
            print(f"Error: missing artifact: {source}", file=sys.stderr)
            sys.exit(1)
        corpus = JsonlCorpus.from_file(source)
        method = corpus.keyword_search if args.strategy == "keyword" else corpus.bm25_search
        lexical = method(args.query, limit=args.limit * 2)

    semantic = []
    if args.strategy in ("vector", "hybrid"):
        try:
            semantic = LlamaIndexStore(
                args.workdir, embed_model=args.embed_model, index_dir=args.index_dir
            ).search(args.query, args.type, limit=args.limit * 2)
        except (FileNotFoundError, LlamaIndexUnavailableError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)

    if args.strategy == "hybrid":
        results = reciprocal_rank_fusion(lexical, semantic, limit=args.limit)
    else:
        results = (semantic or lexical)[:args.limit]
    print(json.dumps(results, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CHM ingestion pipeline — P0: extract, parse, normalize",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ingest
    ingest_parser = subparsers.add_parser("ingest", help="Run P0 ingestion")
    ingest_parser.add_argument("--chm", required=True, help="Path to the .chm file")
    ingest_parser.add_argument("--manual-id", required=True, help="Manual identifier (e.g., product-manual)")
    ingest_parser.add_argument("--product", required=True, help="Product name (e.g., ProductName)")
    ingest_parser.add_argument("--workdir", default="~/Desktop/edr-chm-rag-work",
                               help="Working directory (default: ~/Desktop/edr-chm-rag-work)")

    # validate
    validate_parser = subparsers.add_parser("validate", help="Validate P0 output")
    validate_parser.add_argument("--workdir", default="~/Desktop/edr-chm-rag-work",
                                 help="Working directory (default: ~/Desktop/edr-chm-rag-work)")
    validate_parser.add_argument("--manual-id", default=None, help="Expected manual_id for validation")

    # LlamaIndex vector index
    index_parser = subparsers.add_parser("index", help="Build persisted LlamaIndex vector indexes")
    index_parser.add_argument("--workdir", required=True, help="Work directory containing JSONL artifacts")
    index_parser.add_argument("--type", default="all",
                              choices=["all", "section", "procedure", "step", "action"])
    index_parser.add_argument("--embed-model", default="BAAI/bge-small-zh-v1.5",
                              help="Hugging Face embedding model name")
    index_parser.add_argument("--index-dir", default=None, help="Index directory (default: WORKDIR/.llamaindex)")
    index_parser.add_argument("--show-progress", action="store_true", help="Show embedding progress")

    search_parser = subparsers.add_parser("search", help="Keyword, BM25, vector, or hybrid search")
    search_parser.add_argument("--workdir", required=True)
    search_parser.add_argument("--query", required=True)
    search_parser.add_argument("--type", default="section",
                               choices=["section", "procedure", "step", "action"])
    search_parser.add_argument("--strategy", default="hybrid",
                               choices=["keyword", "bm25", "vector", "hybrid"])
    search_parser.add_argument("--limit", type=int, default=5)
    search_parser.add_argument("--embed-model", default="BAAI/bge-small-zh-v1.5")
    search_parser.add_argument("--index-dir", default=None)

    args = parser.parse_args()
    _configure_logging(args.verbose)

    if args.command == "ingest":
        _cmd_ingest(args)
    elif args.command == "validate":
        _cmd_validate(args)
    elif args.command == "index":
        _cmd_index(args)
    elif args.command == "search":
        _cmd_search(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
