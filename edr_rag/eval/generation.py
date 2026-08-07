"""P2-6 评测框架的 generation（生成）维度。

与 retrieval 维度（Recall/MRR/nDCG，见 ``metrics.py``）并列，评测「答案生成」
的质量，而不再只看最终一句话。生成维度指标:

  - 引用正确率 (citation accuracy)：答案的引用（``answer.citations[].id``）
    中，有多比例是对的（落在标注的 ``relevant`` 集合里）。
  - 拒答率 / 误拒答率 (refusal recall / false refusal)：应拒答的查询里真正
    拒答的比例；以及本应作答却被误拒答的比例。
  - 高风险识别率 (high-risk identification)：涉及高风险操作的查询，答案是否
    正确识别并提示了风险（requires_confirmation 或警告提及风险）。

设计:
  - 所有指标都是纯函数 / 确定性逻辑，基于 ``SynthesizedAnswer`` 与评测标注
    计算，可独立单测（无 IO / 无外部依赖）。
  - 生成逻辑复用 ``edr_rag/p2/synthesize.py`` 的 ``synthesize_answer`` ——
    与 CLI / MCP 共享同一套答案生成链路，保证评测行为与真实使用一致。
  - 生成维度无需任何可选 ``rag`` 依赖；缺依赖时检索降级已由
    RetrievalService 处理，生成侧始终走确定性的组装路径。
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

from ..p2.synthesize import ANSWER_REFUSED, SynthesizedAnswer


# ---------- 引用正确率 ----------


def citation_accuracy(cited_ids: Sequence[str], relevant: Iterable[str]) -> float:
    """引用正确率：答案给出的引用里，正确（在相关集合内）的比例。

    定义（precision-like，衡量「给出的引用有多靠谱」）:
      - relevant 与 cited 均为空 -> 1.0（无需引用，也正确地没引用任何东西）。
      - 引用了但一条都不在 relevant 内 -> 0.0。
      - 否则 = |cited ∩ relevant| / |cited|。

    Args:
        cited_ids: 答案实际引用的 id 列表（answer.citations[].id）。
        relevant: 标注的相关文档 id 集合。

    Returns:
        [0.0, 1.0] 的引用正确率。
    """
    relevant = set(relevant)
    cited = set(cited_ids)
    if not relevant and not cited:
        return 1.0
    if not cited:
        return 0.0
    return len(cited & relevant) / len(cited)


# ---------- 拒答率 / 误拒答率 ----------


def refusal_hit(
    answer_status: str,
    expected_refusal: bool,
) -> float:
    """单个查询的拒答判定是否正确：命中为 1.0，否则 0.0。

    Args:
        answer_status: 生成答案的 answer_status（answered/refused/conflict）。
        expected_refusal: 该查询是否应拒答（评测标注）。

    Returns:
        1.0 表示「应拒答 -> 真拒答」或「应作答 -> 真作答」，否则 0.0。
    """
    generated_refused = answer_status == ANSWER_REFUSED
    return 1.0 if generated_refused == bool(expected_refusal) else 0.0


def refusal_rate(
    rows: Sequence[tuple[str, bool]],
) -> dict[str, float]:
    """汇总拒答维度：拒答率（应拒答中被拒答比例）+ 误拒答率。

    Args:
        rows: [(answer_status, expected_refusal), ...]，逐查询。

    Returns:
        {
          "refusal_recall":   拒答率 —— 标注应拒答的查询里真正拒答的比例。
          "false_refusal":    误拒答率 —— 标注应作答(不应拒答)却被拒答的比例。
        }
        若某侧没有样本则对应值为 0.0。
    """
    expected_refuse = [(s, e) for s, e in rows if e]
    expected_answer = [(s, e) for s, e in rows if not e]

    refused_of_expected = sum(
        1 for s, _ in expected_refuse if s == ANSWER_REFUSED
    )
    falsely_refused = sum(
        1 for s, _ in expected_answer if s == ANSWER_REFUSED
    )
    return {
        "refusal_recall": refused_of_expected / len(expected_refuse) if expected_refuse else 0.0,
        "false_refusal": falsely_refused / len(expected_answer) if expected_answer else 0.0,
    }


# ---------- 高风险识别率 ----------


def high_risk_flag_hit(
    answer: Optional[SynthesizedAnswer],
    expected_high_risk: Optional[bool],
) -> Optional[float]:
    """单个查询的高风险识别是否正确（命中 1.0 / 未命中 0.0）。

    判定答案是否「识别出高风险」：答案要求确认（requires_confirmation）或
    任一警告明确提及风险。

    Args:
        answer: 生成的答案；None 表示未生成（视为未识别）。
        expected_high_risk: 该查询是否应识别高风险（评测标注）。

    Returns:
        None 当 expected_high_risk 未提供（调用方决定是否纳入聚合）；
        否则返回 1.0 / 0.0。
    """
    if expected_high_risk is None:
        return None
    if answer is None:
        flagged = False
    else:
        flagged = bool(answer.requires_confirmation) or any(
            ("风险" in w) or ("high" in w.lower())
            for w in answer.warnings
        )
    return 1.0 if flagged == bool(expected_high_risk) else 0.0


def high_risk_identify_rate(
    rows: Sequence[tuple[Optional[SynthesizedAnswer], bool]],
) -> float:
    """高风险识别率：标注应识别高风险的查询里，被正确识别出风险的比例。

    Args:
        rows: [(answer, expected_high_risk), ...]；仅统计 expected_high_risk
            为 True 的样本。

    Returns:
        [0.0, 1.0]；没有应识别高风险的样本时返回 0.0。
    """
    need = [(a, e) for a, e in rows if e]
    if not need:
        return 0.0
    hits = sum(1 for a, _ in need if high_risk_flag_hit(a, True) == 1.0)
    return hits / len(need)


# ---------- 单查询生成维度汇总 ----------


def query_generation_metrics(
    answer: Optional[SynthesizedAnswer],
    relevant: Iterable[str],
    expected_refusal: Optional[bool] = None,
    high_risk: Optional[bool] = None,
) -> dict[str, float]:
    """汇总单条查询的生成维度指标（供 runner 逐查询记录与聚合）。

    Args:
        answer: 生成的答案（None -> 视为未引用 / 未拒答 / 未识别风险）。
        relevant: 标注的相关 id 集合（做引用正确率的参照）。
        expected_refusal: 是否应拒答的标注；None 表示未标注，不产生该指标。
        high_risk: 是否应识别高风险的标注；None 表示未标注。

    Returns:
        dict，可能包含键：
          - "citation_accuracy": 引用正确率（始终计算）。
          - "refusal_correct":   拒答判定是否正确（仅当 expected_refusal 标注）。
          - "high_risk_correct": 高风险识别是否正确（仅当 high_risk 标注）。
    """
    cited_ids = []
    if answer is not None:
        cited_ids = [c.get("id", "") for c in answer.citations if c.get("id")]

    out: dict[str, float] = {"citation_accuracy": citation_accuracy(cited_ids, relevant)}

    if expected_refusal is not None and answer is not None:
        out["refusal_correct"] = refusal_hit(answer.answer_status, expected_refusal)

    hr = high_risk_flag_hit(answer, high_risk if high_risk is not None else False)
    if high_risk is not None and hr is not None:
        out["high_risk_correct"] = hr

    return out
