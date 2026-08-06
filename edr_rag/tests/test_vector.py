"""Dependency-free tests for the LlamaIndex adapter's data and fusion logic."""

from edr_rag.vector import _record_text, reciprocal_rank_fusion
from edr_rag.p1.search import JsonlCorpus


def test_record_text_includes_action_fields():
    text = _record_text({
        "title": "Configure proxy",
        "module": "Network",
        "preconditions": ["Administrator"],
        "procedure": [{"source_instruction": "Open Settings", "target": "Proxy"}],
    })

    assert "Configure proxy" in text
    assert "Administrator" in text
    assert "Open Settings" in text
    assert "Proxy" in text


def test_reciprocal_rank_fusion_rewards_results_from_both_lists():
    lexical = [{"id": "a", "score": 9}, {"id": "b", "score": 8}]
    semantic = [{"id": "b", "score": 0.9}, {"id": "c", "score": 0.8}]

    results = reciprocal_rank_fusion(lexical, semantic, limit=3)

    assert results[0]["id"] == "b"
    assert results[0]["scoring"] == "hybrid_rrf"
    assert {result["id"] for result in results} == {"a", "b", "c"}


def test_jsonl_search_exposes_p0_source_ref():
    result = JsonlCorpus([{
        "id": "a", "title": "Proxy", "content": "Configure proxy",
        "source_ref": {"relative_path": "network/proxy.html"},
    }]).keyword_search("proxy")[0]

    assert result["source"]["file"] == "network/proxy.html"
