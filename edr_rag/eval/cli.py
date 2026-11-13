"""P1-5 评测 CLI (edr-rag-eval)。

用法:
    python -m edr_rag.eval.cli compare --workdir <path> --data <eval.jsonl>
            [--strategies keyword bm25 hybrid] [--paraphrases]
    python -m edr_rag.eval.cli run --workdir <path> --data <eval.jsonl> --strategy bm25
    python -m edr_rag.eval.cli validate --data <eval.jsonl>
    python -m edr_rag.eval.cli tune --workdir <path> --data <eval.jsonl> [--metric mrr]

子命令:
  compare   对比多个策略并输出汇总 + 逐查询明细 (推荐入口)。
  run       只跑单个策略，输出汇总。
  validate  仅校验评测集格式/一致性。
  tune      (P1-6) 在评测集上网格搜索 hybrid 的 RRF 融合权重并输出最优配置。
"""

from __future__ import annotations

import argparse
import sys

from .dataset import EvalSet, EvalSetValidationError, load_eval_set
from .runner import DEFAULT_STRATEGIES, format_summary, run_eval
from .vocab import EVAL_SLICES, SLICE_OOV, matched_synonyms


def _load(args) -> "EvalSet":
    try:
        return load_eval_set(args.data)
    except (FileNotFoundError, ValueError, EvalSetValidationError) as e:
        print(f"Error loading eval set: {e}", file=sys.stderr)
        sys.exit(1)


def _llm_from_args(args: argparse.Namespace):
    """解析 --llm 开关：'env' 用环境配置的端点，'off'/缺省走组装路径。"""
    mode = getattr(args, "llm", None) or "off"
    if mode == "off":
        return None
    from ..llm import get_llm_client, make_synthesize_callable

    client = get_llm_client()
    if client is None:
        print(
            "警告: --llm 已指定但未配置 EDR_RAG_LLM_BASE_URL/EDR_RAG_LLM_MODEL，"
            "走确定性组装路径。",
            file=sys.stderr,
        )
        return None
    return make_synthesize_callable(client)


def _channel_banner() -> str:
    """评测输出头部的通道声明（1433 事故：归档数值须自带模式证据）。

    active  真嵌入探针通过，hybrid = keyword+BM25+vector 三路
    failed  import 可用但模型加载/推理失败，hybrid = keyword+BM25 两路
    absent  依赖缺失，hybrid = keyword+BM25 两路
    """
    from ..retrieval import RetrievalService

    status = RetrievalService.embedder_status()
    label = {
        "active": "vector=active (hybrid=keyword+bm25+vector 三路)",
        "failed": "vector=FAILED-DEGRADED (hybrid=keyword+bm25 两路; 模型加载/推理失败)",
        "absent": "vector=absent (hybrid=keyword+bm25 两路; 依赖缺失)",
    }.get(status, f"vector={status}")
    return f"检索通道: {label}"


def _cli_compare(args: argparse.Namespace):
    evalset = _load(args)
    strategies = tuple(args.strategies) if args.strategies else DEFAULT_STRATEGIES
    report = run_eval(
        args.workdir,
        evalset,
        strategies=strategies,
        use_paraphrases=args.paraphrases,
        llm=_llm_from_args(args),
    )
    print(_channel_banner())
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
                f" factR@K={m.get('fact_recall@k', 0):.3f} factMRR={m.get('fact_mrr', 0):.3f}"
                if "fact_recall@k" in m
                else ""
            )
            + (
                f" FilterAcc={m.get('filter_accuracy', 0):.3f}"
                if "filter_accuracy" in m
                else ""
            )
        )
        # P2-6: 逐查询 generation 维度摘要（生成维度只汇报有标注的指标）。
        g = row.generation_metrics
        if g:
            gsum = " ".join(
                f"{k}={v:.3f}"
                for k, v in g.items()
            )
            print(f"        gen: {gsum}" + (f" [{row.answer.answer_status}]" if row.answer else ""))
        print(f"        ids: {row.ranked_ids}")


def _cli_run(args: argparse.Namespace):
    evalset = _load(args)
    report = run_eval(
        args.workdir,
        evalset,
        strategies=(args.strategy,),
        use_paraphrases=args.paraphrases,
        llm=_llm_from_args(args),
    )
    print(_channel_banner())
    print(format_summary(report))


def _cli_validate(args: argparse.Namespace):
    evalset = _load(args)
    print(f"OK: {evalset.name} — 主查询 {len(evalset.queries)} 条, "
          f"同义改写 {evalset.paraphrase_total} 条, "
          f"变体合计 {evalset.variant_total} 条")
    print("corpus 覆盖:", sorted({q.corpus_type for q in evalset.queries}))

    # 词表覆盖：命中 SYNONYMS 别名表的查询，其召回含查表贡献，评测时要单独看。
    covered: list[tuple[str, str, list[str]]] = []
    for q in evalset.queries:
        for qid, text in q.effective_queries(use_paraphrases=True):
            hits = matched_synonyms(text)
            if hits:
                covered.append((qid, text, hits))
    total_variants = evalset.variant_total
    print(
        f"词表覆盖: {len(covered)}/{total_variants} 条变体命中 query.py 的 "
        f"SYNONYMS 别名表 (in_vocab)"
    )
    if covered:
        print("  这些变体的召回含查表贡献，泛化能力请看 compare 输出里的 oov 切片：")
        for qid, text, hits in covered:
            print(f"    {qid:<10} {text}  ← {'、'.join(hits)}")


def _cli_tune(args: argparse.Namespace):
    """(P1-6，P4 加 holdout) 网格搜索 hybrid 的 RRF 融合权重。"""
    evalset = _load(args)
    from .tune import tune_rrf_weights

    result = tune_rrf_weights(
        args.workdir, evalset, metric=args.metric, eval_slice=args.slice,
        holdout_ratio=args.holdout_ratio,
    )
    if result.candidates_tried == 0:
        print("评测集为空，跳过权重调节 (返回默认等权)。")
        return
    print(f"目标指标: {result.metric}")
    print(f"选优切片: {result.eval_slice}"
          + (f"  (请求 {args.slice}，该切片无样本，已退回全集)"
             if result.eval_slice != args.slice else ""))
    print(f"切分:     train {result.train_size} 条 / holdout {result.holdout_size} 条")
    print(f"候选网格: {result.candidates_tried} 个配置 (在 train 上选)")
    for cfg, score in sorted(result.scores.items(), key=lambda kv: -kv[1]):
        print(f"    {cfg:<22} {score:.4f}")
    print()
    print(f"最优权重:        {result.best_weights}  (fusion [keyword, bm25, (vector)])")
    print(f"train {result.metric}:  {result.best_score:.4f}   <- 乐观，不要对外汇报这个数")
    print(f"holdout {result.metric}:{result.holdout_score:.4f}   <- 可汇报")
    print(f"等权基线:        {result.baseline_holdout_score:.4f}")
    print(f"过拟合间隙:      {result.overfit_gap:+.4f}  (train - holdout)")
    print(f"调参净收益:      {result.tuning_gain:+.4f}  (holdout - 等权基线)")

    if not result.split_applied:
        print()
        print("⚠  评测集过小或未切分：holdout 与 train 同源，以上分数全部不可信。")
        print("   请把评测集扩充到至少 20 条真实查询后重跑。")
    elif result.tuning_gain <= 0:
        print()
        print("⚠  调参在 holdout 上没有跑赢等权基线 —— 建议直接用等权 (None)。")
    elif result.overfit_gap > 0.15:
        print()
        print("⚠  train 明显高于 holdout，网格搜索在过拟合调参集。")
        print("   增大评测集规模，或缩小候选网格。")


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
    p_compare.add_argument("--llm", default="off", choices=["off", "env"],
                           help="生成侧：env=用 EDR_RAG_LLM_* 环境变量配置的"
                                "端点驱动 LLM 生成/拒答；off=确定性组装（默认）")
    p_compare.set_defaults(func=_cli_compare)

    p_run = sub.add_parser("run", help="单策略评测")
    p_run.add_argument("--workdir", required=True)
    p_run.add_argument("--data", required=True)
    p_run.add_argument("--strategy", default="keyword",
                       choices=["keyword", "bm25", "vector", "hybrid"])
    p_run.add_argument("--paraphrases", action="store_true")
    p_run.add_argument("--llm", default="off", choices=["off", "env"],
                       help="生成侧：env=用 EDR_RAG_LLM_* 环境变量配置的端点")
    p_run.set_defaults(func=_cli_run)

    p_val = sub.add_parser("validate", help="校验评测集")
    p_val.add_argument("--data", required=True)
    p_val.set_defaults(func=_cli_validate)

    p_tune = sub.add_parser("tune", help="(P1-6) 网格搜索 RRF 融合权重")
    p_tune.add_argument("--workdir", required=True)
    p_tune.add_argument("--data", required=True)
    p_tune.add_argument(
        "--holdout-ratio", dest="holdout_ratio", type=float, default=0.3,
        help="留出集比例 (P4)；0 表示不切分 (分数将不可信)",
    )
    p_tune.add_argument("--metric", default="recall@k",
                        choices=["recall@k", "mrr", "ndcg@k", "filter_accuracy"])
    p_tune.add_argument("--slice", default=SLICE_OOV, choices=list(EVAL_SLICES),
                        help="用于选优的词表覆盖切片 (缺省 oov：只在未被 "
                             "SYNONYMS 别名表覆盖的查询上选，避免在查表已保证"
                             "命中的样本上调参)")
    p_tune.set_defaults(func=_cli_tune)

    args = parser.parse_args()
    if not args.subcommand:
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
