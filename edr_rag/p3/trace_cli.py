"""P3-2: trace 查看 / 聚合 CLI。

读取 trace.jsonl 并输出可观测数据:
  - traces       列出最近事件（按时间倒序）
  - stats        延迟分位数 + 质量指标聚合
  - latest       最新一条事件（便于快速调试）

Usage:
    python -m edr_rag.p3.trace_cli --trace <trace.jsonl> stats
    python -m edr_rag.p3.trace_cli --trace <trace.jsonl> traces [--limit 20]
    python -m edr_rag.p3.trace_cli --trace <trace.jsonl> latest
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional


def _fmt_ms(v) -> str:
    return "—" if v is None else f"{v:.1f}ms"


def _fmt_num(v) -> str:
    return "—" if v is None else f"{v:.4f}"


def _cli_stats(store):
    metrics = store.quality_metrics()
    lat = store.latency_by_type()

    print("== 质量指标 (quality metrics) ==")
    print(f"  事件总数 (event_count): {metrics['event_count']}")
    r = metrics["retrieval"]
    print("  [retrieval]")
    print(f"    count           : {r['count']}")
    print(f"    empty_rate      : {_fmt_num(r['empty_rate'])}  (空召回占比)")
    print(f"    avg_max_score   : {_fmt_num(r['avg_max_score'])}")
    print(f"    avg_top_confidence: {_fmt_num(r['avg_top_confidence'])}")
    g = metrics["generation"]
    print("  [generation]")
    print(f"    count           : {g['count']}")
    print(f"    refusal_rate    : {_fmt_num(g['refusal_rate'])}  (拒答率)")
    print(f"    error_rate      : {_fmt_num(g['error_rate'])}")
    print(f"    avg_confidence  : {_fmt_num(g['avg_confidence'])}")
    if g["by_status"]:
        print(f"    by_status       : {json.dumps(g['by_status'], ensure_ascii=False)}")
    if g["by_generated_by"]:
        print(f"    by_generated_by : {json.dumps(g['by_generated_by'], ensure_ascii=False)}")
    print(f"  model_call_count: {metrics['model_call_count']}")

    print()
    print("== 延迟分位数 (by type, successful only) ==")
    if not lat:
        print("  (无 successful 事件)")
    for t, q in sorted(lat.items()):
        print(f"  [{t}] count={q['count']} p50={_fmt_ms(q['p50'])} "
              f"p95={_fmt_ms(q['p95'])} max={_fmt_ms(q['max'])} sum={_fmt_ms(q['sum'])}")


def _cli_traces(store, limit: int):
    events = store.read_all()
    if not events:
        print("(empty trace)")
        return
    for ev in reversed(events[-limit:]):
        p = ev.get("payload") or {}
        et = ev.get("event_type", "?")
        ok = "ok" if ev.get("ok") else "ERR"
        lat = _fmt_ms(ev.get("latency_ms"))
        extra = ""
        if et == "retrieval":
            extra = (f"{p.get('corpus_type')} strategy={p.get('strategy')} "
                     f"count={p.get('result_count')} max_score={_fmt_num(p.get('max_score'))} "
                     f"q={p.get('query')!r:.40}")
        elif et == "generation":
            extra = (f"status={p.get('answer_status')} conf={_fmt_num(p.get('confidence'))} "
                     f"by={p.get('generated_by')} steps={p.get('step_count')} q={p.get('query')!r:.40}")
        elif et == "model_call":
            extra = f"model={p.get('model')} prompt={p.get('prompt_chars')} comp={p.get('completion_chars')}"
        print(f"[{ev.get('ts','')}] {et:10s} {ok} {lat:>12s}  {extra}")


def _cli_latest(store):
    events = store.read_all()
    if not events:
        print("(empty trace)")
        return
    print(json.dumps(events[-1], ensure_ascii=False, indent=2))


def main(argv: Optional[list[str]] = None) -> int:
    from edr_rag.p3.tracing import TraceStore

    parser = argparse.ArgumentParser(description="P3-2 trace 查看/聚合 CLI")
    parser.add_argument("--trace", required=True, help="trace.jsonl 路径")
    sub = parser.add_subparsers(dest="subcommand")

    sub.add_parser("stats", help="质量指标 + 延迟分位数")
    p_traces = sub.add_parser("traces", help="列出事件")
    p_traces.add_argument("--limit", type=int, default=20)
    sub.add_parser("latest", help="最新一条事件")

    args = parser.parse_args(argv)
    if not args.subcommand:
        parser.print_help()
        return 1

    store = TraceStore(args.trace)
    if args.subcommand == "stats":
        _cli_stats(store)
    elif args.subcommand == "traces":
        _cli_traces(store, args.limit)
    elif args.subcommand == "latest":
        _cli_latest(store)
    return 0


if __name__ == "__main__":
    sys.exit(main())
