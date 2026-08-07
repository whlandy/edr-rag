"""CLI entry point for the CHM ingestion pipeline.

Commands:
  python -m tools.edr_rag.cli ingest --chm <path> [options]

    P0 ingest: extract CHM + TOC + HTML → manifest.json + sections.jsonl + assets.jsonl

  python -m tools.edr_rag.cli validate --workdir <path> [--manual-id <id>]


  python -m tools.edr_rag.cli search --query <text> [--workdir <path>] [options]
    (Planned for P1 — not implemented yet)
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


def _cmd_validate_index(args: argparse.Namespace) -> None:
    """Validate the index manifest (P0-5): schema/hash/source/orphan checks."""
    from .index_manifest import validate_index_manifest

    res = validate_index_manifest(
        args.workdir, chain=args.chain, artifacts=args.artifacts or None
    )
    print(res.message)
    if res.stale_files:
        print(f"  内容已变化: {', '.join(res.stale_files)}")
    if res.ok:
        return
    # refuse 级结论 (schema 不匹配 / 孤儿残留) -> 非零退出
    if res.refuse:
        print("拒绝读取: 请重新 ingest / generate 后再使用索引", file=sys.stderr)
        sys.exit(2)
    # warning 级 (内容/源头过期) -> 非零退出 (提示但不阻止重建)
    sys.exit(1)


def _cmd_index(sub_args: argparse.Namespace) -> None:
    """Index version management (P3-5): list / capture / rollback."""
    from .index_versioning import (
        activate_version,
        capture_version,
        load_versions,
        rollback,
    )

    workdir = Path(sub_args.workdir).expanduser().resolve()

    if sub_args.action == "list":
        registry = load_versions(workdir)
        if not registry:
            print(f"无索引版本记录 ({workdir / 'index_versions.json'})")
            return
        for corpus in sorted(registry):
            print(f"[{corpus}]")
            for v in registry[corpus]:
                mark = "  (active)" if v.active else ""
                print(
                    f"  v{v.version_id}{mark} built_at={v.built_at} "
                    f"records={len(v.fingerprints)} source_hash={v.source_hash[:12] or '-'}"
                )
        return

    if sub_args.action == "capture":
        corpus = sub_args.corpus
        ver, _p = capture_version(
            workdir,
            corpus=corpus,
            source_hash=sub_args.source_hash or "",
            keep_history=not sub_args.drop_history,
        )
        print(f"captured {corpus} -> v{ver.version_id} (active)")
        return

    if sub_args.action == "rollback":
        corpus = sub_args.corpus
        target = rollback(workdir, corpus=corpus)
        if target is None:
            print(
                f"无法回滚 {corpus}: 无版本历史或未找到回滚目标",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"rolled back {corpus} -> v{target.version_id} (active)")
        return

    if sub_args.action == "activate":
        corpus = sub_args.corpus
        target = activate_version(workdir, corpus=corpus, version_id=sub_args.version)
        if target is None:
            print(
                f"未找到 {corpus} 的 v{sub_args.version}",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"activated {corpus} -> v{target.version_id} (active)")
        return


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

    # validate-index
    vi_parser = subparsers.add_parser("validate-index", help="Validate index manifest (schema/hash/source/orphan)")
    vi_parser.add_argument("--workdir", required=True, help="Work directory")
    vi_parser.add_argument("--chain", default=None, choices=["p1", "p2"],
                           help="Expected chain (optional)")
    vi_parser.add_argument("--artifact", dest="artifacts", action="append", default=None,
                           help="Artifact filename(s) to hash-check (repeatable; default: all tracked)")

    # index (P3-5: 版本灰度切换 / 回滚)
    index_parser = subparsers.add_parser(
        "index",
        help="Index version management (P3-5): list / capture / rollback / activate",
    )
    index_sub = index_parser.add_subparsers(dest="action", required=True)

    idx_list = index_sub.add_parser("list", help="List index versions per corpus")
    idx_list.add_argument("--workdir", required=True, help="Work directory")

    idx_cap = index_sub.add_parser("capture", help="Capture a new version for a corpus (becomes active)")
    idx_cap.add_argument("--workdir", required=True, help="Work directory")
    idx_cap.add_argument("--corpus", required=True, help="Corpus type, e.g. chunks / procedures / actions")
    idx_cap.add_argument("--source-hash", default="", help="Source hash at build time")
    idx_cap.add_argument("--drop-history", action="store_true",
                         help="Discard older versions for this corpus (disable rollback)")

    idx_roll = index_sub.add_parser("rollback", help="Roll back a corpus to the previous version")
    idx_roll.add_argument("--workdir", required=True, help="Work directory")
    idx_roll.add_argument("--corpus", required=True, help="Corpus type")

    idx_act = index_sub.add_parser("activate", help="Activate a specific version of a corpus")
    idx_act.add_argument("--workdir", required=True, help="Work directory")
    idx_act.add_argument("--corpus", required=True, help="Corpus type")
    idx_act.add_argument("--version", type=int, required=True, help="Version id to activate")

    args = parser.parse_args()
    _configure_logging(args.verbose)

    if args.command == "ingest":
        _cmd_ingest(args)
    elif args.command == "validate":
        _cmd_validate(args)
    elif args.command == "validate-index":
        _cmd_validate_index(args)
    elif args.command == "index":
        _cmd_index(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
