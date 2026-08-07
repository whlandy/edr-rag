"""P1-4 服务层测试：RetrievalService 上下文补全集成。

覆盖：
  1. `context_for(node_id)` 返回包含 procedure/preconditions/warnings/
     相邻 steps/section 的结构化上下文。
  2. `search(corpus_type='nodes', with_context=True)` 给结果附加 context 与
     context_text。
  3. 缺 nodes.jsonl 时优雅降级（context_for 返回 None、with_context 原样返回）。
  4. 未知 node_id 时 context_for 返回 None。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from edr_rag.p1.nodes import build_structured_nodes
from edr_rag.retrieval import RetrievalService

SECTION = {
    "id": "sec-01",
    "manual_id": "M-001",
    "title": "配置远程登录会话",
    "section_path": ["网络配置", "远程管理"],
    "content": "本文介绍如何配置远程登录会话。",
}

PROCEDURE = {
    "id": "procedure.网络配置.配置远程登录会话",
    "manual_id": "M-001",
    "operation": "配置远程登录会话",
    "title": "配置远程登录会话",
    "section_path": ["网络配置", "远程管理"],
    "preconditions": ["以管理员身份登录设备"],
    "expected_result": "能够通过远程会话成功管理设备",
    "warnings": ["修改加密配置前请备份"],
    "recovery": ["连接失败请检查端口 22"],
    "source": {"section_id": "sec-01"},
    "steps": [
        {"order": 1, "instruction": "输入代理服务器地址"},
        {"order": 2, "instruction": "启用加密传输协议 TLS"},
        {"order": 3, "instruction": "保存配置"},
    ],
}


def _write_nodes(root: Path):
    nodes = build_structured_nodes([SECTION], [PROCEDURE])
    payload = "\n".join(json.dumps(n.to_json(), ensure_ascii=False) for n in nodes) + "\n"
    (root / "nodes.jsonl").write_text(payload, encoding="utf-8")


def _step(nodes, order):
    return next(
        n for n in nodes
        if n.level == "step" and n.metadata["step"] == order
    )


def test_context_for_step_completes_procedure_fields(tmp_path):
    _write_nodes(tmp_path)
    svc = RetrievalService(tmp_path)
    nodes = build_structured_nodes([SECTION], [PROCEDURE])
    middle = _step(nodes, 2)

    ctx = svc.context_for(middle.id)
    assert ctx is not None
    proc = ctx["procedure"]
    assert proc["procedure_id"] == PROCEDURE["id"]
    assert proc["preconditions"] == ["以管理员身份登录设备"]
    assert proc["warnings"] == ["修改加密配置前请备份"]
    assert proc["recovery"] == ["连接失败请检查端口 22"]
    assert proc["expected_result"] == "能够通过远程会话成功管理设备"
    relations = {a["relation"] for a in ctx["adjacent_steps"]}
    assert relations == {"prev", "next"}
    assert ctx["section"]["module"] == "网络配置 / 远程管理"


def test_context_for_unknown_id_returns_none(tmp_path):
    _write_nodes(tmp_path)
    svc = RetrievalService(tmp_path)
    assert svc.context_for("node.step.nonexistent.s1") is None


def test_context_for_missing_nodes_file_returns_none(tmp_path):
    # 不写 nodes.jsonl
    svc = RetrievalService(tmp_path)
    assert svc.context_for("node.step.x.s1") is None


def test_search_nodes_with_context_enriches_results(tmp_path):
    _write_nodes(tmp_path)
    svc = RetrievalService(tmp_path)
    results = svc.search("启用加密传输协议", corpus_type="nodes", strategy="keyword",
                         with_context=True)
    assert results, "should find the TLS step node"
    r = results[0]
    assert "context" in r and "context_text" in r
    assert r["context"]["procedure"]["preconditions"] == ["以管理员身份登录设备"]
    assert "[前置条件]" in r["context_text"]


def test_search_without_context_has_no_context_key(tmp_path):
    _write_nodes(tmp_path)
    svc = RetrievalService(tmp_path)
    results = svc.search("启用加密传输协议", corpus_type="nodes", strategy="keyword")
    assert results
    assert "context" not in results[0]


def test_search_nodes_with_context_missing_file_no_crash(tmp_path):
    # 无 nodes.jsonl：with_context 应优雅降级，返回原样结果（空）
    svc = RetrievalService(tmp_path)
    results = svc.search("代理", corpus_type="nodes", strategy="keyword", with_context=True)
    assert results == []


def test_api_get_context_and_node_search(tmp_path):
    """EdrRagAPI 的 get_context / node_search 代理到服务并返回补全上下文。"""
    from edr_rag.p2.api import EdrRagAPI

    _write_nodes(tmp_path)
    api = EdrRagAPI(tmp_path)
    nodes = build_structured_nodes([SECTION], [PROCEDURE])
    middle = _step(nodes, 2)

    ctx = api.get_context(middle.id)
    assert ctx is not None
    assert ctx["procedure"]["preconditions"] == ["以管理员身份登录设备"]

    results = api.node_search("启用加密传输协议")
    assert results
    assert "context" in results[0]
    assert "[前置条件]" in results[0]["context_text"]

    # 未知 id / 缺文件返回 None
    assert api.get_context("node.step.nonexistent.s1") is None

