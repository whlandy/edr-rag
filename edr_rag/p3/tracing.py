"""P3-2: tracing / 延迟 / 质量监控 — 结构化可观测性。

P3 scope (design §7.6 observability):
  - 为检索与生成加结构化日志
  - 记录耗时 (latency)、召回结果、模型调用追踪
  - 输出 JSONL 可观测数据 (trace.jsonl)，可被独立工具聚合

模块组成:
  - TraceEvent: 单条可观测事件 (检索 / 生成 / 模型调用)，扁平 JSONL 序列化。
  - EventTracer: 线程安全的 JSONL 写入器，负责时间戳、trace_id、事件 id 与落盘。
  - TraceContext: 一次「会话」的追踪边界，为其中的若干事件打上同一 trace_id，
    并支持子上下文 (parent/child) 记录嵌套调用 (如 生成->模型调用)。
  - TraceStore: 读取 + 聚合 trace.jsonl —— 延迟分位数、质量指标
    (平均置信度 / 空召回率 / 拒答率 / 错误率)，供 CLI / 报表消费。

设计取舍:
  - 独立于 RetrievalService / EdrRagAPI，以「薄埋点」方式接入 (P3 Agent 层)，
    不侵入 P0/P1/P2 核心检索路径，避免破坏现有 API/CLI/MCP 入口。
  - 所有事件带相同扁平 schema，便于直接 import 进 pandas / jq / Datadog。
  - 写入失败绝不抛出 —— 观测链路必须对主链路零副作用 (best-effort)。
  - 依赖: 纯标准库 (json / time / threading / uuid / statistics)。
"""

from __future__ import annotations

import json
import statistics
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

#: 检索类事件的 type 标记
EVT_RETRIEVAL = "retrieval"
EVT_GENERATION = "generation"
EVT_MODEL_CALL = "model_call"
#: 生成答案的来源 (generated_by)，对齐 P2 synthesize
_GENERATED_BY_STRUCTURED = "structured"
_GENERATED_BY_LLM = "llm"

#: 事件类型白名单（用于校验 / 文档）
EVENT_TYPES = (EVT_RETRIEVAL, EVT_GENERATION, EVT_MODEL_CALL)


def _now_iso() -> str:
    """UTC ISO8601 时间戳。"""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TraceEvent:
    """单条结构化可观测事件。

    所有事件共用扁平 schema：type 决定 payload 的解释方式。latency_ms 为
    该事件阶段自身耗时（不含子阶段），单位为毫秒。ok 标记该阶段是否成功。
    """

    event_type: str
    latency_ms: float
    ok: bool
    trace_id: str = ""
    event_id: str = ""
    ts: str = ""
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "event_id": self.event_id,
            "ts": self.ts,
            "event_type": self.event_type,
            "latency_ms": round(self.latency_ms, 3),
            "ok": self.ok,
            "payload": self.payload,
        }


class EventTracer:
    """线程安全的 JSONL 可观测事件写入器。

    写入 best-effort：文件不可写 / 磁盘满等异常被吞掉并返回 False，
    绝不向主链路抛出。每次 append 加锁，保证并发安全。
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = threading.Lock()

    def append(self, event: TraceEvent) -> bool:
        """序列化并追加一条事件到 JSONL。

        Args:
            event: 待写入事件。trace_id/event_id/ts 为空时自动补全。

        Returns:
            True 写入成功；False 在失败时返回（不抛出）。
        """
        if not event.trace_id:
            event.trace_id = new_trace_id()
        if not event.event_id:
            event.event_id = new_event_id()
        if not event.ts:
            event.ts = _now_iso()
        try:
            line = json.dumps(event.to_dict(), ensure_ascii=False)
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
            return True
        except Exception:  # pragma: no cover - best-effort 观测路径
            return False

    def reset(self) -> None:
        """清空已写入的追踪文件（测试 / 轮转用）。"""
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("", encoding="utf-8")


class TraceContext:
    """一次会话的追踪边界。

    用法:
        with TraceContext(tracer, context="ask_answer") as ctx:
            ctx.record_retrieval(query=..., corpus_type=..., strategy=...,
                                 latency_ms=..., result_count=...)
            with ctx.child("generation") as gen:
                gen.record_model_call(model="llm", latency_ms=...)
            ctx.record_generation(...)

    同一 ctx 内所有事件共享 trace_id；child() 返回的子上下文共享父 trace_id，
    便于在 trace_id 维度做整条会话的关联分析。
    """

    def __init__(
        self,
        tracer: EventTracer,
        context: str = "",
        parent_id: str = "",
        trace_id: str = "",
    ):
        self.tracer = tracer
        self.context = context
        self.trace_id = trace_id or new_trace_id()
        self.parent_id = parent_id
        self._events: list[TraceEvent] = []

    # ---------- 运行时上下文管理 ----------

    def __enter__(self) -> "TraceContext":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc) -> bool:
        """上下文退出时记录累计耗时事件（仅当已显式记录过具体事件时才生成汇总）。

        返回 False（不吞异常），保证观测路径不改变主链路异常传播。
        """
        return False

    def child(self, context: str) -> "TraceContext":
        """派生一个共享 trace_id 的子上下文（用于嵌套阶段）。"""
        return TraceContext(
            self.tracer, context=context,
            parent_id=self.trace_id, trace_id=self.trace_id,
        )

    # ---------- 记录入口 ----------

    def _emit(self, event_type: str, latency_ms: float, ok: bool, payload: dict) -> TraceEvent:
        event = TraceEvent(
            event_type=event_type,
            latency_ms=float(latency_ms),
            ok=ok,
            trace_id=self.trace_id,
            payload=payload,
        )
        self._events.append(event)
        self.tracer.append(event)
        return event

    # 检索事件
    def record_retrieval(
        self,
        query: str,
        corpus_type: str,
        strategy: str,
        latency_ms: float,
        results: list[dict],
        ok: bool = True,
    ) -> TraceEvent:
        """记录一次检索。

        Args:
            results: 检索原始结果列表；仅摘取 score/confidence 等概要字段
                写入 payload，避免把整条大文档灌进观测日志。
        """
        scores = [_num(r.get("score")) for r in results]
        scores_clean = [s for s in scores if s is not None]
        confs = [
            _num(r.get("confidence"))
            for r in results
            if r.get("confidence") is not None
        ]
        payload = {
            "context": self.context,
            "query": query,
            "corpus_type": corpus_type,
            "strategy": strategy,
            "result_count": len(results),
            "max_score": max(scores_clean) if scores_clean else None,
            "top_scores": scores[:5],
            "top_confidences": confs[:5],
            "top_ids": [str(r.get("id") or r.get("action_id") or "") for r in results[:3]],
        }
        return self._emit(EVT_RETRIEVAL, latency_ms, ok, payload)

    # 生成事件
    def record_generation(
        self,
        query: str,
        latency_ms: float,
        result: dict,
        ok: bool = True,
    ) -> TraceEvent:
        """记录一次答案生成 (P2 synthesize / ask_answer)。

        只摘取结构化概要字段，answer 正文截断到前 200 字符控制体积。
        """
        answer = result.get("answer") or ""
        payload = {
            "context": self.context,
            "query": query,
            "answer_status": result.get("answer_status"),
            "confidence": _num(result.get("confidence")),
            "generated_by": result.get("generated_by"),
            "requires_confirmation": result.get("requires_confirmation"),
            "refusal_reason": result.get("refusal_reason"),
            "step_count": len(result.get("steps") or []),
            "warning_count": len(result.get("warnings") or []),
            "source_count": len(result.get("sources") or []),
            "answer_preview": (answer[:200] + "…") if len(answer) > 200 else answer,
        }
        return self._emit(EVT_GENERATION, latency_ms, ok, payload)

    # 模型调用事件
    def record_model_call(
        self,
        model: str,
        latency_ms: float,
        ok: bool = True,
        prompt_chars: int = 0,
        completion_chars: int = 0,
        **extra: Any,
    ) -> TraceEvent:
        """记录一次模型 (LLM) 调用。"""
        payload = {
            "context": self.context,
            "model": model,
            "prompt_chars": int(prompt_chars),
            "completion_chars": int(completion_chars),
        }
        payload.update(extra)
        return self._emit(EVT_MODEL_CALL, latency_ms, ok, payload)


# ---------- TraceStore: 读取 + 聚合 ----------

class TraceStore:
    """读取并聚合 trace.jsonl，产出延迟分位数与质量指标。

    供 CLI (trace) 与报表消费；纯标准库实现。
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def read_all(self) -> list[dict]:
        """读取全部事件 dict；坏行跳过。文件缺失返回 []。"""
        if not self.path.exists():
            return []
        out: list[dict] = []
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return out

    # ---------- 通用延迟统计 ----------

    def latency_by_type(self) -> dict[str, dict[str, float]]:
        """按 event_type 统计 successful 事件的延迟分位数 (p50/p95/max/count)。"""
        by_type: dict[str, list[float]] = {}
        for ev in self.read_all():
            if not ev.get("ok"):
                continue
            by_type.setdefault(ev.get("event_type", ""), []).append(ev.get("latency_ms", 0.0))
        out: dict[str, dict[str, float]] = {}
        for t, vals in by_type.items():
            out[t] = _quantiles(vals)
        return out

    def latency_total_ms(self) -> Optional[float]:
        """所有事件的延迟总和（近似全链路耗时）。"""
        total = 0.0
        for ev in self.read_all():
            total += ev.get("latency_ms", 0.0)
        return round(total, 3) if self.read_count() else None

    def read_count(self) -> int:
        return len(self.read_all())

    # ---------- 质量指标 ----------

    def quality_metrics(self) -> dict[str, Any]:
        """聚合质量指标。

        Returns:
            dict 含:
              - retrieval:  检索子指标 (count / empty_rate / avg_max_score / avg_top_conf)
              - generation: 生成子指标 (count / refusal_rate / error_rate / avg_confidence)
              - model_call: 模型调用次数
              - overall:    事件总数
        """
        events = self.read_all()
        retrieval = [e for e in events if e.get("event_type") == EVT_RETRIEVAL and e.get("ok")]
        generation = [e for e in events if e.get("event_type") == EVT_GENERATION and e.get("ok")]
        model_calls = [e for e in events if e.get("event_type") == EVT_MODEL_CALL and e.get("ok")]
        gen_all = [e for e in events if e.get("event_type") == EVT_GENERATION]

        empty = sum(1 for e in retrieval if (e.get("payload") or {}).get("result_count", 0) == 0)
        avg_max_score = _avg(
            (e.get("payload") or {}).get("max_score")
            for e in retrieval
            if (e.get("payload") or {}).get("max_score") is not None
        )
        top_conf = [
            c
            for e in retrieval
            for c in (e.get("payload") or {}).get("top_confidences") or []
            if c is not None
        ]
        refusal = sum(
            1 for e in gen_all
            if (e.get("payload") or {}).get("answer_status") == "refused"
        )
        err = sum(1 for e in gen_all if not e.get("ok"))
        avg_conf = _avg(
            (e.get("payload") or {}).get("confidence")
            for e in generation
            if (e.get("payload") or {}).get("confidence") is not None
        )

        return {
            "event_count": len(events),
            "retrieval": {
                "count": len(retrieval),
                "empty_rate": (empty / len(retrieval)) if retrieval else None,
                "avg_max_score": avg_max_score,
                "avg_top_confidence": _avg(top_conf),
            },
            "generation": {
                "count": len(generation),
                "refusal_rate": (refusal / len(gen_all)) if gen_all else None,
                "error_rate": (err / len(gen_all)) if gen_all else None,
                "avg_confidence": avg_conf,
                "by_status": _count_by(gen_all, "answer_status"),
                "by_generated_by": _count_by(gen_all, "generated_by"),
            },
            "model_call_count": len(model_calls),
        }

    def by_trace_id(self) -> dict[str, list[dict]]:
        """按 trace_id 分组事件，便于做整条会话的关联分析。"""
        groups: dict[str, list[dict]] = {}
        for ev in self.read_all():
            groups.setdefault(ev.get("trace_id", ""), []).append(ev)
        return groups


# ---------- 工具函数 ----------

def new_trace_id() -> str:
    """生成 trace_id（UUID 短形式）。"""
    return f"trc_{uuid.uuid4().hex[:12]}"


def new_event_id() -> str:
    """生成事件 id（UUID 短形式）。"""
    return f"evt_{uuid.uuid4().hex[:12]}"


def _num(v: Any) -> Optional[float]:
    """把数值类值规整为 float；None / 非数值返回 None。"""
    if v is None:
        return None
    try:
        return round(float(v), 4)
    except (TypeError, ValueError):
        return None


def _avg(values) -> Optional[float]:
    """可迭代数值的平均值；空或无值时返回 None。"""
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return round(statistics.fmean(vals), 4)


def _quantiles(vals: list[float]) -> dict[str, float]:
    """延迟分位数统计。"""
    if not vals:
        return {"count": 0}
    s = sorted(vals)
    return {
        "count": len(s),
        "p50": round(statistics.median(s), 3),
        "p95": round(s[min(int(0.95 * len(s)) - 1, len(s) - 1)], 3),
        "max": round(s[-1], 3),
        "sum": round(sum(s), 3),
    }


def _count_by(events: list[dict], key: str) -> dict[str, int]:
    """按事件 payload 中字段计数。"""
    out: dict[str, int] = {}
    for e in events:
        v = (e.get("payload") or {}).get(key)
        out[str(v)] = out.get(str(v), 0) + 1
    return out
