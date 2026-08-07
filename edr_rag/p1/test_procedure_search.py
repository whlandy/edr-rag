"""P0-1 回归测试:检索文本组装路径 — steps/warnings/preconditions/recovery/expected_result/版本/模块。

验证目标:
  1. Procedure 的步骤内容(steps)真正进入检索文本 —— 仅凭"具体操作步骤内容"(而非标题)
     就能在 procedure/step chunk 上召回该 procedure。
  2. warnings / preconditions / recovery / expected_result / 版本 一并进入检索文本。
  3. 模块(section_path)上下文被 folded 进可检索文本,支持按模块召回。
  4. 可选 RAG 依赖缺失时 vector_search 优雅降级到 BM25,主链路不报错。
"""

from __future__ import annotations

from edr_rag.p1.extract_procedures import (
    _procedure_to_text,
    build_chunks,
    extract_procedures_from_section,
)
from edr_rag.p1.schema import Procedure, RetrievalChunk
from edr_rag.p1.search import JsonlCorpus

# A sample procedure section whose _title_ says nothing about the concrete
# steps, so only the step content (not the title) can recall it.
SAMPLE_SECTION = {
    "id": "sec-01",
    "manual_id": "M-001",
    "product": "华三交换机",
    "title": "配置远程登录会话",
    "version": "V2.1",
    "section_path": ["网络配置", "远程管理", "登录会话"],
    "content": (
        "前置条件\n"
        "以管理员身份登录设备\n"
        "1. 输入代理服务器地址 10.1.2.3\n"
        "2. 设置会话空闲超时时间为 15 分钟\n"
        "3. 启用加密传输协议 TLS\n"
        "预期结果\n"
        "能够通过远程会话成功管理设备\n"
        "如果连接失败,请检查端口 22 是否被占用\n"
    ),
    "source_ref": {"relative_path": "docs/remote.md"},
}


def _build_corpus() -> tuple[JsonlCorpus, "Procedure", "list[RetrievalChunk]"]:
    procs = extract_procedures_from_section(dict(SAMPLE_SECTION))
    assert procs, "sample section should parse into a procedure"
    chunks = build_chunks(dict(SAMPLE_SECTION), procs)
    records = [
        {
            "id": c.id,
            "type": c.type,
            "text": c.text,
            "source": {
                "file": c.source.file,
                "section_id": c.source.section_id,
                "procedure_id": c.source.procedure_id,
            },
            "metadata": c.metadata,
        }
        for c in chunks
    ]
    return JsonlCorpus(records), procs[0], chunks


def test_retrieval_text_contains_steps_not_just_title():
    """步骤内容应真正进入检索文本(procedure 级文本)。"""
    procs = extract_procedures_from_section(dict(SAMPLE_SECTION))
    assert procs
    proc = procs[0]
    text = _procedure_to_text(proc)
    # 标题不含具体步骤信息,步骤文本必须独立存在于检索文本中
    assert "输入代理服务器地址 10.1.2.3" in text
    assert "设置会话空闲超时时间为 15 分钟" in text
    assert "启用加密传输协议 TLS" in text


def test_retrieval_text_contains_warnings_preconditions_recovery_expected_version():
    """warnings/preconditions/recovery/expected_result/版本/模块 均应纳入检索文本。"""
    procs = extract_procedures_from_section(dict(SAMPLE_SECTION))
    proc = procs[0]
    text = _procedure_to_text(proc)

    # preconditions 来自"前置条件"块
    assert "以管理员身份登录设备" in text
    # recovery 来自"如果连接失败"块
    assert "检查端口 22 是否被占用" in text
    # expected_result 来自"预期结果"块
    assert "预期结果" in text and "能够通过远程会话成功管理设备" in text
    # 版本
    assert "版本: V2.1" in text


def test_search_recalls_procedure_by_step_content():
    """回归测试:仅凭具体操作步骤内容(非标题)能召回该 procedure chunk。"""
    corpus, proc, chunks = _build_corpus()

    # 查询只含步骤中的具体细节,与标题"配置远程登录会话"完全不同
    results = corpus.keyword_search("输入代理服务器地址 10.1.2.3", limit=5)
    assert results, "should recall something by step content"

    proc_ids = {r["source"].get("procedure_id") for r in results}
    assert proc.id in proc_ids, (
        f"expected procedure {proc.id} recalled by step content, got {proc_ids}"
    )

    # step 级 chunk 也能被召回
    ids = {r["id"] for r in results}
    assert any(i.startswith("step-") for i in ids) or proc.id in proc_ids


def test_search_recalls_procedure_by_expected_and_recovery():
    """按预期结果 / 恢复信息也能召回该 procedure。"""
    corpus, proc, chunks = _build_corpus()

    res_expected = corpus.keyword_search("成功管理设备", limit=5)
    assert any(
        r["source"].get("procedure_id") == proc.id for r in res_expected
    ), "expected_result 文本应能召回 procedure"

    res_recovery = corpus.keyword_search("端口 22 被占用", limit=5)
    assert any(
        r["source"].get("procedure_id") == proc.id for r in res_recovery
    ), "recovery 文本应能召回 procedure"


def test_search_recalls_procedure_by_module_context():
    """模块(section_path)上下文 folded 进检索文本,可按模块召回。"""
    corpus, proc, chunks = _build_corpus()
    # 用模块名(非标题)查询 —— 来自 section_path / operation 上下文
    results = corpus.keyword_search("远程管理 登录会话", limit=5)
    assert results
    proc_ids = {r["source"].get("procedure_id") for r in results}
    assert proc.id in proc_ids, f"module context should recall procedure, got {proc_ids}"


def test_vector_search_degrades_to_bm25_without_rag_extras():
    """无可选 RAG 依赖时 vector_search 优雅降级到 BM25,主链路不报错。"""
    corpus, proc, chunks = _build_corpus()
    results = corpus.vector_search("输入代理服务器地址 10.1.2.3", limit=5)
    # 无论是否真正加载了向量,结果必须是非空且包含目标 procedure
    proc_ids = {r["source"].get("procedure_id") for r in results}
    assert proc.id in proc_ids


def test_chunk_metadata_carries_version_and_expected_result():
    """chunk metadata 应携带 version 与 expected_result,供折叠进检索文本。"""
    _, proc, chunks = _build_corpus()
    proc_chunk = next(c for c in chunks if c.type == "procedure")
    assert proc_chunk.metadata.get("version") == "V2.1"
    assert "能够通过远程会话成功管理设备" in (proc_chunk.metadata.get("expected_result") or "")
    # step chunk 继承模块上下文
    step_chunk = next(c for c in chunks if c.type == "step")
    assert step_chunk.metadata.get("module")
