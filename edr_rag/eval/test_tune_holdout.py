"""P4 —— 调参的 train/holdout 切分。

背景 (根因)：P1-6 的 ``tune_rrf_weights`` 在**同一份**评测集上做网格搜索并
报告最优分数。那个数天然是乐观的 —— 它回答不了「换一批查询还成立吗」。
配合当时从评测集反抄的同义词表，整条评测链路构成自证闭环。

P4 让调参在 train 上选、在 holdout 上报，并给出等权基线，使
「过拟合」和「调参是否真有收益」都变成可读的数字。
"""

from __future__ import annotations

from edr_rag.eval.dataset import EvalQuery, EvalSet
from edr_rag.eval.tune import (
    DEFAULT_HOLDOUT_RATIO,
    TuningResult,
    split_eval_set,
)


def _mk(n: int) -> EvalSet:
    return EvalSet(
        queries=[
            EvalQuery(qid=f"q{i:03d}", query=f"查询{i}", corpus_type="actions",
                      relevant=[f"a{i}"])
            for i in range(n)
        ],
        name="synthetic",
    )


# ---------- 切分本身 ----------


def test_split_is_disjoint_and_covers_everything():
    """AC-7: train / holdout 不重叠，且并集等于全集。"""
    eval_set = _mk(40)
    train, holdout = split_eval_set(eval_set)
    train_ids = {q.qid for q in train.queries}
    holdout_ids = {q.qid for q in holdout.queries}
    assert not (train_ids & holdout_ids), "train 与 holdout 不得重叠"
    assert train_ids | holdout_ids == {q.qid for q in eval_set.queries}


def test_split_is_deterministic():
    a1, b1 = split_eval_set(_mk(40))
    a2, b2 = split_eval_set(_mk(40))
    assert [q.qid for q in a1.queries] == [q.qid for q in a2.queries]
    assert [q.qid for q in b1.queries] == [q.qid for q in b2.queries]


def test_split_ratio_is_approximately_honoured():
    eval_set = _mk(200)
    _, holdout = split_eval_set(eval_set, holdout_ratio=0.3)
    frac = len(holdout.queries) / 200
    assert 0.2 < frac < 0.4, f"holdout fraction {frac:.2f} far from 0.3"


def test_split_is_stable_when_eval_set_grows():
    """按 qid 哈希分桶：追加新条目不改变既有条目的归属，历史分数仍可比。"""
    small_train, small_holdout = split_eval_set(_mk(40))
    big_train, big_holdout = split_eval_set(_mk(80))
    small_holdout_ids = {q.qid for q in small_holdout.queries}
    big_holdout_ids = {q.qid for q in big_holdout.queries}
    # 前 40 条在两次切分中的归属必须一致
    for q in small_train.queries:
        assert q.qid not in big_holdout_ids, f"{q.qid} 归属发生漂移"
    for qid in small_holdout_ids:
        assert qid in big_holdout_ids, f"{qid} 归属发生漂移"


def test_tiny_eval_set_is_not_split():
    """< 4 条时不切分 —— 基于 1 条查询的 holdout 分数没有意义。"""
    eval_set = _mk(3)
    train, holdout = split_eval_set(eval_set)
    assert len(train.queries) == len(holdout.queries) == 3


def test_degenerate_ratio_disables_split():
    for ratio in (0.0, 1.0, -0.5, 2.0):
        train, holdout = split_eval_set(_mk(40), holdout_ratio=ratio)
        assert len(train.queries) == len(holdout.queries) == 40


def test_empty_eval_set():
    train, holdout = split_eval_set(EvalSet(queries=[], name="empty"))
    assert train.queries == [] and holdout.queries == []


# ---------- TuningResult 的诊断字段 ----------


def test_overfit_gap_and_tuning_gain():
    r = TuningResult(
        best_weights=[2.0, 1.0], best_score=0.90, scores={}, metric="recall@k",
        candidates_tried=7, holdout_score=0.62, baseline_holdout_score=0.60,
        train_size=28, holdout_size=12,
    )
    # train 0.90 vs holdout 0.62 -> 明显过拟合
    assert r.overfit_gap == 0.28
    # 调参相对等权只赚了 0.02
    assert r.tuning_gain == 0.02


def test_tuning_gain_negative_means_dont_tune():
    r = TuningResult(
        best_weights=[2.0, 1.0], best_score=0.95, scores={}, metric="recall@k",
        candidates_tried=7, holdout_score=0.55, baseline_holdout_score=0.61,
    )
    assert r.tuning_gain < 0


def test_default_holdout_ratio_is_sane():
    assert 0.0 < DEFAULT_HOLDOUT_RATIO < 0.5
