"""P0-1 regression tests: procedure/step retrieval text completeness.

Asserts that procedure and step chunks produced by P1 carry the real
operation content (steps, warnings, preconditions, recovery, module
context) so that search by *step content* (not just title) and by
*module context* both recall the procedure.

Addresses review finding: retrieval text must include the actual
steps and module/section context, not only the title.
"""
from __future__ import annotations

from edr_rag.p1.extract_procedures import build_chunks, _procedure_to_text
from edr_rag.p1.schema import Procedure, ProcedureSource, ProcedureStep
from edr_rag.p1.search import JsonlCorpus


def _make_proc() -> Procedure:
    return Procedure(
        id="procedure.test_configure_proxy",
        manual_id="m1",
        product="EDR",
        operation="配置代理服务器代理设置",
        title="配置代理服务器",
        section_path=["网络", "代理", "配置代理服务器"],
        source=ProcedureSource(file="net/proxy.html", section_id="s1"),
        preconditions=["已安装客户端"],
        steps=[
            ProcedureStep(order=1, instruction="打开系统设置，进入网络与代理页面"),
            ProcedureStep(order=2, instruction="在代理服务器输入框中填入地址 127.0.0.1"),
            ProcedureStep(order=3, instruction="点击保存并重启服务"),
        ],
        warnings=["修改代理会中断当前连接"],
        recovery=["恢复默认代理设置"],
        confidence=0.9,
        status="verified",
    )


def _make_section() -> dict:
    return {
        "id": "s1",
        "manual_id": "m1",
        "product": "EDR",
        "title": "配置代理服务器",
        "content": "本节介绍如何配置代理",
        "section_path": ["网络", "代理", "配置代理服务器"],
        "source_ref": {"relative_path": "net/proxy.html"},
    }


def test_procedure_to_text_includes_steps_and_context():
    proc = _make_proc()
    text = _procedure_to_text(proc)
    # Steps (the core P0-1 concern): real step instructions must be present.
    assert "打开系统设置" in text
    assert "127.0.0.1" in text
    assert "点击保存并重启服务" in text
    # Context fields must be present.
    assert "前置条件" in text and "已安装客户端" in text
    assert "注意" in text and "中断当前连接" in text
    assert "恢复" in text and "恢复默认代理设置" in text
    assert "操作" in text and "配置代理服务器" in text


def test_procedure_chunk_recall_by_step_content():
    """The procedure chunk must be findable by a *step* query, not just title."""
    section = _make_section()
    chunks = build_chunks(section, [_make_proc()])
    corpus = JsonlCorpus([c.to_jsonl() for c in chunks])

    # Query a distinctive step phrase that is NOT in the title.
    res = corpus.keyword_search("代理服务器输入框 填入地址", limit=5)
    assert res, "procedure chunk not recalled by step content"
    hit = res[0]
    assert hit["type"] == "procedure" or hit["type"] == "step"
    assert "127.0.0.1" in hit["snippet"] or hit["type"] == "procedure"


def test_procedure_chunk_recall_by_module_context():
    """Module/section context must be searchable even when the query only
    names the module root ('网络'), which is metadata, not step text."""
    section = _make_section()
    chunks = build_chunks(section, [_make_proc()])
    corpus = JsonlCorpus([c.to_jsonl() for c in chunks])

    # The normalized record folds metadata.module / section_path into text.
    rec = corpus.records[0]
    assert "网络" in rec["_searchable"], "module/section_path not folded into searchable text"
    assert rec["_title"], "chunk title should be populated for weighting"

    res = corpus.keyword_search("网络 配置代理", limit=5)
    assert res, "procedure chunk not recalled by module context"


def test_procedure_chunk_metadata_carries_module_and_section():
    section = _make_section()
    chunks = build_chunks(section, [_make_proc()])
    proc_chunk = next(c for c in chunks if c.type == "procedure")
    meta = proc_chunk.metadata
    assert meta["module"], "module metadata missing"
    assert meta["section_path"], "section_path metadata missing"
    assert meta["operation"] == "配置代理服务器代理设置"
    assert meta["source_file"] == "net/proxy.html"

    # Step chunks inherit parent context (warnings / preconditions / recovery).
    step_chunks = [c for c in chunks if c.type == "step"]
    assert step_chunks
    for sc in step_chunks:
        assert sc.metadata["warnings"] == ["修改代理会中断当前连接"]
        assert sc.metadata["preconditions"] == ["已安装客户端"]
        assert sc.metadata["recovery"] == ["恢复默认代理设置"]


if __name__ == "__main__":
    test_procedure_to_text_includes_steps_and_context()
    test_procedure_chunk_recall_by_step_content()
    test_procedure_chunk_recall_by_module_context()
    test_procedure_chunk_metadata_carries_module_and_section()
    print("[PASS] P0-1 retrieval text tests")
