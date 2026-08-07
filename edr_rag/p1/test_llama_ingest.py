"""P1-1 测试: LlamaIndex 摄取管线（可选依赖）与优雅降级。

测试目标：
  - 纯标准库结构化切分 + 持久化缓存 + 去重（不缺依赖时也不受影响）。
  - 缺 llama-index-core 时优雅降级到标准库路径，mode='fallback'，不报错。
  - 持久化缓存可复用、去重正确（指纹 + id）。
  - CLI ingest-llama 在两种依赖状态下均可工作（fallback 为主）。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from edr_rag.p1 import llama_ingest as li


def _write_sample_chunks(workdir: Path) -> Path:
    """写出一个简单 chunks.jsonl 语料。"""
    chunks = [
        {
            "id": "chunk.section.s1.s1",
            "type": "step",
            "manual_id": "test-manual",
            "title": "打开网络设置",
            "text": "要配置代理服务器：1. 打开网络设置 2. 选择代理 3. 输入代理地址",
            "section_path": ["网络", "代理"],
            "source": {"file": "network/proxy.html"},
            "metadata": {"section_path": ["网络", "代理"], "module": "网络 / 代理"},
        },
        {
            "id": "chunk.section.s2.s1",
            "type": "step",
            "manual_id": "test-manual",
            "title": "查看安全状态",
            "text": "安全状态页面显示当前防护状态。",
            "section_path": ["安全状态"],
            "source": {"file": "status.html"},
            "metadata": {"section_path": ["安全状态"], "module": "安全状态"},
        },
    ]
    path = workdir / "chunks.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    return path


# ---------- 结构化切分 ----------

def test_structured_split_produces_nodes(tmp_path: Path):
    workdir = tmp_path / "w"
    workdir.mkdir()
    _write_sample_chunks(workdir)
    records = li.read_jsonl(workdir / "chunks.jsonl")
    nodes = li.structured_split_records(records, "chunks")
    assert len(nodes) == 2
    assert all(n.id and n.text and n.fingerprint for n in nodes)
    assert nodes[0].id.startswith("chunks.")
    assert "代理" in nodes[0].text


def test_structured_split_long_text_rechunks(tmp_path: Path):
    workdir = tmp_path / "w"
    workdir.mkdir()
    long_text = "内容。" * 200  # 600 字，超过 chunk_size 默认
    (workdir / "chunks.jsonl").write_text(
        json.dumps(
            {"id": "chunk.a", "type": "section", "manual_id": "m",
             "title": "", "text": long_text, "section_path": [], "source": {}}
        )
        + "\n",
        encoding="utf-8",
    )
    records = li.read_jsonl(workdir / "chunks.jsonl")
    nodes = li.structured_split_records(records, "chunks", chunk_size=200, overlap=20)
    assert len(nodes) >= 3  # 600 字 / 步长 180
    # 每条节点不超过 chunk_size
    assert all(len(n.text) <= 200 for n in nodes)


# ---------- 去重 ----------

def test_dedupe_by_fingerprint_and_id():
    from edr_rag.p1.llama_ingest import SplitNode, dedupe_nodes

    n1 = SplitNode(id="a", text="hello world", fingerprint="fp-common")
    n2 = SplitNode(id="b", text="hello world", fingerprint="fp-common")  # 同指纹
    n3 = SplitNode(id="a", text="different", fingerprint="fp-x")  # 同 id
    n4 = SplitNode(id="c", text="unique", fingerprint="fp-y")

    out = dedupe_nodes([n1, n2, n3, n4], seen=set())
    assert [n.id for n in out] == ["a", "c"]


def test_dedupe_against_persisted_cache(tmp_path: Path):
    workdir = tmp_path / "w"
    workdir.mkdir()
    _write_sample_chunks(workdir)

    # 第一次摄取
    r1 = li.run_ingestion_pipeline(workdir, "chunks")
    assert r1.source_records == 2
    assert r1.new_nodes == 2
    assert r1.mode in ("llama", "fallback")

    # 第二次摄取相同语料 -> 全部被去重
    r2 = li.run_ingestion_pipeline(workdir, "chunks")
    assert r2.source_records == 2
    assert r2.new_nodes == 0
    assert r2.duplicates_skipped == r2.produced_nodes


# ---------- 持久化缓存 ----------

def test_persistence_cache_written_and_loadable(tmp_path: Path):
    workdir = tmp_path / "w"
    workdir.mkdir()
    _write_sample_chunks(workdir)
    li.run_ingestion_pipeline(workdir, "chunks")

    cache = workdir / "llama_nodes.json"
    report = workdir / "llama_ingest_report.json"
    assert cache.exists()
    assert report.exists()

    payload = json.loads(cache.read_text(encoding="utf-8"))
    assert payload["node_count"] == 2
    assert len(payload["nodes"]) == 2

    loaded = li.load_cached_nodes(workdir)
    assert len(loaded) == 2

    rep = json.loads(report.read_text(encoding="utf-8"))
    assert rep["produced_nodes"] == 2
    assert rep["new_nodes"] == 2


# ---------- 优雅降级 ----------

def test_graceful_fallback_when_llama_absent(tmp_path: Path):
    """缺 llama-index 时 mode='fallback'，不抛异常，仍有节点产出。"""
    if li.llama_index_available():
        pytest.skip("llama-index-core installed; cannot test fallback naturally")

    workdir = tmp_path / "w"
    workdir.mkdir()
    _write_sample_chunks(workdir)
    result = li.run_ingestion_pipeline(workdir, "chunks")
    assert result.mode == "fallback"
    assert result.new_nodes == 2
    assert result.error is None

    # 强制 llama 时会抛出
    with pytest.raises(li.LlamaIndexUnavailable):
        li.run_ingestion_pipeline(workdir, "chunks", force_llama=True)


# ---------- CLI ----------

def test_cli_ingest_llama_fallback(tmp_path: Path):
    workdir = tmp_path / "w"
    workdir.mkdir()
    _write_sample_chunks(workdir)

    proc = subprocess.run(
        [sys.executable, "-m", "edr_rag.p1.cli", "ingest-llama",
         "--workdir", str(workdir), "--corpus", "chunks"],
        capture_output=True, text=True, cwd=str(Path(__file__).parents[2]),
    )
    assert proc.returncode == 0, proc.stderr
    assert "mode=fallback" in proc.stdout or "mode=llama" in proc.stdout
    assert (workdir / "llama_nodes.json").exists()


def test_cli_ingest_llama_unknown_corpus(tmp_path: Path):
    workdir = tmp_path / "w"
    workdir.mkdir()
    _write_sample_chunks(workdir)
    # 不存在的语料类型 -> 0 条源记录，不报错
    proc = subprocess.run(
        [sys.executable, "-m", "edr_rag.p1.cli", "ingest-llama",
         "--workdir", str(workdir), "--corpus", "actions"],
        capture_output=True, text=True, cwd=str(Path(__file__).parents[2]),
    )
    assert proc.returncode == 0, proc.stderr
