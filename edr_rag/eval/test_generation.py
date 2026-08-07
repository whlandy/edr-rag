"""P2-6 generation 维度指标的单测。

覆盖 generation.py 中的纯函数：
  - citation_accuracy（引用正确率）
  - refusal_hit / refusal_rate（拒答率 + 误拒答率）
  - high_risk_flag_hit / high_risk_identify_rate（高风险识别率）
  - query_generation_metrics（单查询汇总）
均用构造的 SynthesizedAnswer / 简单实数核对，不依赖真实语料。
"""

from __future__ import annotations

from edr_rag.eval.generation import (
    citation_accuracy,
    high_risk_flag_hit,
    high_risk_identify_rate,
    query_generation_metrics,
    refusal_hit,
    refusal_rate,
)
from edr_rag.p2.synthesize import ANSWER_ANSWERED, ANSWER_REFUSED, SynthesizedAnswer


def _answer(
    answer_status: str = ANSWER_ANSWERED,
    cited_ids: list[str] | None = None,
    warnings: list[str] | None = None,
    requires_confirmation: bool = False,
) -> SynthesizedAnswer:
    return SynthesizedAnswer(
        answer="...",
        warnings=warnings or [],
        citations=[{"id": cid, "title": cid} for cid in (cited_ids or [])],
        requires_confirmation=requires_confirmation,
        answer_status=answer_status,
    )


# ---------- citation_accuracy ----------


def test_citation_accuracy_both_empty():
    assert citation_accuracy([], []) == 1.0


def test_citation_accuracy_no_citation_but_relevant():
    assert citation_accuracy([], ["a"]) == 0.0


def test_citation_accuracy_all_correct():
    assert citation_accuracy(["a", "b"], ["a", "b", "c"]) == 1.0


def test_citation_accuracy_partial():
    assert citation_accuracy(["a", "b", "x"], ["a", "b"]) == 2 / 3


def test_citation_accuracy_none_correct():
    assert citation_accuracy(["x", "y"], ["a"]) == 0.0


# ---------- refusal ----------


def test_refusal_hit_expect_refuse_and_refused():
    assert refusal_hit(ANSWER_REFUSED, True) == 1.0


def test_refusal_hit_expect_refuse_but_answered():
    assert refusal_hit(ANSWER_ANSWERED, True) == 0.0


def test_refusal_hit_expect_answer_and_answered():
    assert refusal_hit(ANSWER_ANSWERED, False) == 1.0


def test_refusal_hit_false_refusal():
    assert refusal_hit(ANSWER_REFUSED, False) == 0.0


def test_refusal_rate_mixed():
    rows = [
        (ANSWER_REFUSED, True),   # 应拒答 -> 拒答  ✓
        (ANSWER_ANSWERED, True),  # 应拒答 -> 作答  ✗
        (ANSWER_ANSWERED, False), # 应作答 -> 作答  ✓
        (ANSWER_ANSWERED, False), # 应作答 -> 作答  ✓
        (ANSWER_REFUSED, False),  # 应作答 -> 误拒答 ✗
    ]
    out = refusal_rate(rows)
    assert out["refusal_recall"] == 0.5   # 2 条应拒答里 1 条拒答
    assert out["false_refusal"] == 1 / 3  # 3 条应作答里 1 条误拒


def test_refusal_rate_no_samples():
    out = refusal_rate([])
    assert out["refusal_recall"] == 0.0
    assert out["false_refusal"] == 0.0


# ---------- high risk ----------


def test_high_risk_flag_confirmation():
    a = _answer(requires_confirmation=True)
    assert high_risk_flag_hit(a, True) == 1.0
    assert high_risk_flag_hit(a, False) == 0.0


def test_high_risk_flag_warning_text():
    a = _answer(warnings=["注意：该操作存在高风险，请确认"])
    assert high_risk_flag_hit(a, True) == 1.0


def test_high_risk_flag_not_flagged():
    a = _answer()
    assert high_risk_flag_hit(a, True) == 0.0
    assert high_risk_flag_hit(a, False) == 1.0


def test_high_risk_unnannotated_returns_none():
    assert high_risk_flag_hit(_answer(), None) is None


def test_high_risk_identify_rate():
    rows = [
        (_answer(requires_confirmation=True), True),  # ✓
        (_answer(requires_confirmation=True), True),  # ✓
        (_answer(), True),                            # ✗
        (_answer(), False),                           # 不计入（expected False）
    ]
    assert high_risk_identify_rate(rows) == 2 / 3


def test_high_risk_identify_no_samples():
    assert high_risk_identify_rate([]) == 0.0


# ---------- query_generation_metrics ----------


def test_query_generation_metrics_full():
    a = _answer(cited_ids=["a", "x"], warnings=["高风险"], requires_confirmation=True)
    out = query_generation_metrics(a, ["a", "b"], expected_refusal=False, high_risk=True)
    assert out["citation_accuracy"] == 0.5
    assert out["refusal_correct"] == 1.0   # 应作答，未拒答
    assert out["high_risk_correct"] == 1.0


def test_query_generation_metrics_no_annotations():
    a = _answer(cited_ids=["a"])
    out = query_generation_metrics(a, ["a"])
    # 只出现始终计算的引用正确率，无拒答/高风键。
    assert set(out.keys()) == {"citation_accuracy"}
    assert out["citation_accuracy"] == 1.0


def test_query_generation_metrics_none_answer():
    out = query_generation_metrics(None, ["a"])
    assert out["citation_accuracy"] == 0.0
    assert "refusal_correct" not in out
