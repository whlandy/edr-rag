"""P3-2: tracing/延迟/质量监控 单元测试。

覆盖:
  - EventTracer JSONL 写入 & 自动补全 trace_id/event_id/ts
  - TraceContext 检索 / 生成 / 模型调用事件记录
  - TraceStore 延迟分位数 & 质量指标聚合
  - EdrRagAgent 端到端埋点 (search_* 产生 retrieval 事件, ask_answer 产生 generation)
"""

import json
from pathlib import Path

import pytest

from edr_rag.p3.tracing import (
    EventTracer,
    TraceContext,
    TraceEvent,
    TraceStore,
    EVT_RETRIEVAL,
    EVT_GENERATION,
    EVT_MODEL_CALL,
)


@pytest.fixture
def trace_path(tmp_path: Path) -> Path:
    return tmp_path / "trace.jsonl"


# ---------- EventTracer ----------

def test_tracer_append_auto_fills_fields(trace_path: Path):
    tracer = EventTracer(trace_path)
    # record_* 会通过 tracer 自动落盘，并回填 trace_id/event_id/ts
    ev = TraceContext(tracer, context="test").record_retrieval(
        "代理", "actions", "keyword", 1.5, [{"id": "a1", "score": 0.9}]
    )
    assert ev.trace_id
    assert ev.event_id
    assert ev.ts
    assert trace_path.exists()
    line = json.loads(trace_path.read_text(encoding="utf-8").strip())
    assert line["event_type"] == EVT_RETRIEVAL
    assert line["trace_id"].startswith("trc_")
    assert line["event_id"].startswith("evt_")
    assert line["ts"]
    assert line["latency_ms"] == pytest.approx(1.5, abs=0.01)
    assert line["payload"]["result_count"] == 1
    assert line["payload"]["max_score"] == pytest.approx(0.9)


def test_tracer_best_effort_no_throw(tmp_path: Path):
    # 把路径指向一个会写失败的位置：父目录是文件而非目录 → mkdir 失败
    blocker = tmp_path / "blocker"
    blocker.write_text("x", encoding="utf-8")
    tracer = EventTracer(Path(blocker) / "t.jsonl")
    assert tracer.append(TraceEvent(EVT_RETRIEVAL, 1.0, True)) is False  # 失败但不抛
    # 正常路径仍可用
    good = EventTracer(tmp_path / "t.jsonl")
    assert good.append(TraceEvent(EVT_RETRIEVAL, 1.0, True)) is True


def test_tracer_reset(trace_path: Path):
    tracer = EventTracer(trace_path)
    ctx = TraceContext(tracer, context="x")
    ctx.record_retrieval("q", "actions", "keyword", 1.0, [])
    assert trace_path.exists()
    tracer.reset()
    assert trace_path.read_text(encoding="utf-8") == ""


# ---------- TraceContext ----------

def test_context_child_shares_trace_id(trace_path: Path):
    tracer = EventTracer(trace_path)
    with TraceContext(tracer, context="ask_answer") as ctx:
        ctx.record_retrieval("q", "actions", "keyword", 1.0, [])
        with ctx.child("generation") as gen:
            gen.record_model_call("fake-llm", 3.0, prompt_chars=10, completion_chars=5)
        ctx.record_generation("q", 4.0, {"answer_status": "answered", "confidence": 0.8})
    events = TraceStore(trace_path).read_all()
    assert len(events) == 3
    trace_ids = {e["trace_id"] for e in events}
    assert len(trace_ids) == 1  # 同一条会话共享 trace_id
    types = {e["event_type"] for e in events}
    assert types == {EVT_RETRIEVAL, EVT_GENERATION, EVT_MODEL_CALL}


def test_record_generation_previews_answer(trace_path: Path):
    tracer = EventTracer(trace_path)
    ctx = TraceContext(tracer, context="ask_answer")
    ctx.record_generation(
        "q", 1.0,
        {"answer_status": "answered", "confidence": 0.7, "answer": "x" * 300,
         "generated_by": "structured", "steps": [1, 2, 3], "warnings": [], "sources": ["a"]},
    )
    ev = TraceStore(trace_path).read_all()[0]["payload"]
    assert ev["step_count"] == 3
    assert len(ev["answer_preview"]) <= 201  # 200 + 省略号


# ---------- TraceStore ----------

def _seed(trace_path: Path):
    tracer = EventTracer(trace_path)
    # 3 次成功检索
    for i, q in enumerate(["网络", "代理", "防火墙"]):
        TraceContext(tracer, context=f"r{i}").record_retrieval(
            q, "actions", "keyword", 10.0 + i, [{"id": "a", "score": 0.9 - i * 0.1}]
        )
    # 1 次空召回
    TraceContext(tracer, context="empty").record_retrieval(
        "绝对不存在的词", "actions", "keyword", 5.0, []
    )
    # 2 次生成（1 正常 1 拒答）
    TraceContext(tracer, context="g1").record_generation(
        "q", 20.0, {"answer_status": "answered", "confidence": 0.8, "answer": "ok"}
    )
    TraceContext(tracer, context="g2").record_generation(
        "q", 15.0, {"answer_status": "refused", "confidence": 0.1, "answer": ""}
    )


def test_store_quality_metrics(trace_path: Path):
    _seed(trace_path)
    m = TraceStore(trace_path).quality_metrics()
    assert m["event_count"] == 6
    r = m["retrieval"]
    assert r["count"] == 4
    assert r["empty_rate"] == pytest.approx(0.25)
    assert r["avg_max_score"] == pytest.approx((0.9 + 0.8 + 0.7) / 3)
    g = m["generation"]
    assert g["count"] == 2
    assert g["refusal_rate"] == pytest.approx(0.5)
    assert g["avg_confidence"] == pytest.approx((0.8 + 0.1) / 2)
    assert g["error_rate"] == 0.0
    assert g["by_status"] == {"answered": 1, "refused": 1}


def test_store_latency_quantiles(trace_path: Path):
    _seed(trace_path)
    lat = TraceStore(trace_path).latency_by_type()
    assert EVT_RETRIEVAL in lat
    # latency 10,11,12,5 -> sorted [5,10,11,12]
    q = lat[EVT_RETRIEVAL]
    assert q["count"] == 4
    assert q["max"] == pytest.approx(12.0)
    assert q["p50"] == pytest.approx(10.5)


def test_store_empty_file(tmp_path: Path):
    store = TraceStore(tmp_path / "missing.jsonl")
    assert store.read_all() == []
    assert store.quality_metrics()["event_count"] == 0
    assert store.latency_by_type() == {}


# ---------- EdrRagAgent 集成埋点 ----------

def test_agent_search_emits_retrieval_trace(tmp_path: Path):
    from edr_rag.p3.server import EdrRagAgent

    # 构造最小 sections.jsonl
    wd = tmp_path / "workdir"
    wd.mkdir(parents=True)
    (wd / "sections.jsonl").write_text(
        json.dumps({"id": "s1", "title": "配置代理", "text": "打开网络设置选择代理", "path": "network/proxy"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    agent = EdrRagAgent(str(wd), trace=True)
    results = agent.search_sections("代理", limit=3)
    assert len(results) >= 1

    store = TraceStore(wd / "trace.jsonl")
    events = store.read_all()
    assert len(events) >= 1
    assert events[-1]["event_type"] == EVT_RETRIEVAL
    assert events[-1]["payload"]["result_count"] >= 1
    assert events[-1]["latency_ms"] >= 0


def test_agent_trace_disabled_writes_nothing(tmp_path: Path):
    from edr_rag.p3.server import EdrRagAgent

    wd = tmp_path / "workdir"
    wd.mkdir(parents=True)
    (wd / "sections.jsonl").write_text(
        json.dumps({"id": "s1", "title": "配置代理", "text": "打开网络设置", "path": "n/p"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    agent = EdrRagAgent(str(wd), trace=False)
    agent.search_sections("代理", limit=3)
    assert not (wd / "trace.jsonl").exists()
