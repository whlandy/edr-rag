"""P1-5 评测 CLI (edr-rag-eval)。

用法:
    python -m edr_rag.eval.cli compare --workdir <path> --data <eval.jsonl>
            [--strategies keyword bm25 hybrid] [--paraphrases]
    python -m edr_rag.eval.cli run --workdir <path> --data <eval.jsonl> --strategy bm25
    python -m edr_rag.eval.cli validate --data <eval.jsonl>

子命令:
  compare   对比多个策略并输出汇总 + 逐查询明细 (推荐入口)。
  run       只跑单个策略，输出汇总。
  validate  仅校验评测集格式/一致性。
"""

from __future__ import annotations

import argparse
import sys

from .dataset import EvalSet, EvalSetValidationError, load_eval_set
from .runner import DEFAULT_STRATEGIES, format_summary, run_eval


def _load(args) -> "EvalSet":
    try:
        return load_eval_set(args.data)
    except (FileNotFoundError, ValueError, EvalSetValidationError) as e:
        print(f"Error loading eval set: {e}", file=sys.stderr)
        sys.exit(1)


def _cli_compare(args: argparse.Namespace):
    evalset = _load(args)
    strategies = tuple(args.strategies) if args.strategies else DEFAULT_STRATEGIES
    report = run_eval(
        args.workdir,
        evalset,
        strategies=strategies,
        use_paraphrases=args.paraphrases,
    )
    print(format_summary(report))
    print()
    print("=== 逐查询明细 ===")
    for row in report.rows:
        m = row.metrics
        print(
            f"[{row.strategy:<3}] {row.qid}: "
            f"R@K={m.get('recall@k', 0):.3f} MRR={m.get('mrr', 0):.3f} "
            f"nDCG@K={m.get('ndcg@k', 0):.3f}"
            + (
                f" FilterAcc={m.get('filter_accuracy', 0):.3f}"
                if "filter_accuracy" in m
                else ""
            )
        )
        print(f"        ids: {row.ranked_ids}")


def _cli_run(args: argparse.Namespace):
    evalset = _load(args)
    report = run_eval(
        args.workdir,
        evalset,
        strategies=(args.strategy,),
        use_paraphrases=args.paraphrases,
    )
    print(format_summary(report))


def _cli_validate(args: argparse.Namespace):
    evalset = _load(args)
    print(f"OK: {evalset.name} — 主查询 {len(evalset.queries)} 条, "
          f"同义改写 {evalset.paraphrase_total} 条, "
          f"变体合计 {evalset.variant_total} 条")
    print("corpus 覆盖:", sorted({q.corpus_type for q in evalset.queries}))


def main():
    parser = argparse.ArgumentParser(
        description="P1-5: EDR 查询评测 runner (对比 keyword/bm25/hybrid)",
    )
    sub = parser.add_subparsers(dest="subcommand")

    p_compare = sub.add_parser("compare", help="对比多策略")
    p_compare.add_argument("--workdir", required=True)
    p_compare.add_argument("--data", required=True, help="评测集路径 (.json/.jsonl)")
    p_compare.add_argument("--strategies", nargs="*", default=None,
                           choices=["keyword", "bm25", "vector", "hybrid"])
    p_compare.add_argument("--paraphrases", action="store_true",
                           help="把同义改写也当独立查询评测")
    p_compare.set_defaults(func=_cli_compare)

    p_run = sub.add_parser("run", help="单策略评测")
    p_run.add_argument("--workdir", required=True)
    p_run.add_argument("--data", required=True)
    p_run.add_argument("--strategy", default="keyword",
                       choices=["keyword", "bm25", "vector", "hybrid"])
    p_run.add_argument("--paraphrases", action="store_true")
    p_run.set_defaults(func=_cli_run)

    p_val = sub.add_parser("validate", help="校验评测集")
    p_val.add_argument("--data", required=True)
    p_val.set_defaults(func=_cli_validate)

    args = parser.parse_args()
    if not args.subcommand:
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
