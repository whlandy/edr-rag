"""评测集词表覆盖切片 (in_vocab / oov) 测试。

背景见 edr_rag/eval/vocab.py：命中 query.py 词表（P4 起外置到
data/synonyms.zh.txt，可经 EDR_RAG_SYNONYMS 换表）的查询，其召回数字含
查表贡献，不能用来判断泛化能力。这里覆盖:

  - 覆盖判定 (matched_synonyms / is_in_vocabulary / slice_of)
  - 判定随别名表实时变化，而非写死
  - EvalReport 的切片聚合与切片计数
  - format_summary 的切片表出现/省略条件
  - tune 默认在 oov 切片选优，切片为空时退回 all

判定测试统一走 monkeypatch 注入的受控词表，不依赖随包词表文件的具体条目
（文件是现场可换的产品数据，测试钉它的内容会把评测和词表维护耦死）；
管线测试 (run_eval / tune) 用 _mixed_eval_set 的 autouse fixture 固定
词表后，切片标签也是确定的。
"""

from pathlib import Path

import pytest

import edr_rag.query as _query
from edr_rag.eval.dataset import EvalQuery, EvalSet
from edr_rag.eval.runner import format_summary, run_eval
from edr_rag.eval.test_runner import _write_workdir
from edr_rag.eval.vocab import (
    SLICE_ALL,
    SLICE_IN_VOCAB,
    SLICE_OOV,
    in_slice,
    is_in_vocabulary,
    matched_synonyms,
    slice_of,
)


@pytest.fixture(autouse=True)
def _pinned_vocab(monkeypatch, tmp_path_factory):
    """把词表钉成受控单规则表，让本文件所有切片判定与随包词表内容解耦。

    「设置代理 → 配置代理」：_mixed_eval_set 的 v1 靠它成为 in_vocab，
    o1（规范词本身）与 o2（导出审计日志，与词表及 workdir 语料均无关）
    成为 oov，且 o2 在 workdir 里检索不到 relevant 动作——切片间指标
    有确定性的差。

    检索侧 (expand_query → get_synonyms) 与切片侧 (vocab → query.SYNONYMS
    模块属性) 是两条读取路径：前者读 EDR_RAG_SYNONYMS 词表文件（带缓存），
    后者读模块属性以便运行时换表可观测。fixture 同时钉住两条路径——
    monkeypatch _synonyms_path 指向临时词表文件并清缓存，再同步模块属性
    快照——保证「覆盖判定」与「真实检索行为」用词一致，这正是切片语义
    的前提；测完恢复随包词表，避免污染其他测试。
    """
    table = [
        ("设置代理", "配置代理"),
    ]
    vocab_file = tmp_path_factory.mktemp("vocab") / "synonyms_test.zh.txt"
    vocab_file.write_text(
        "\n".join(f"{a} => {c}" for a, c in table) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(_query, "_synonyms_path", lambda: vocab_file)
    monkeypatch.setattr(_query, "SYNONYMS", list(table))
    _query.reset_synonyms_cache()
    yield
    _query.reset_synonyms_cache()


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    _write_workdir(tmp_path)
    return tmp_path


# ---------- 覆盖判定 ----------

def test_matched_synonyms_detects_alias():
    # 「怎么设置代理」命中「设置代理 → 配置代理」（_pinned_vocab 词表）。
    hits = matched_synonyms("怎么设置代理")
    assert "设置代理" in hits
    assert is_in_vocabulary("怎么设置代理") is True
    assert slice_of("怎么设置代理") == SLICE_IN_VOCAB


def test_canonical_query_is_oov():
    # 别名表是「别名 -> 规范词」，规范词本身不该被判为 in_vocab，
    # 否则主查询会被整批错划进 in_vocab，oov 切片就空了。
    assert is_in_vocabulary("配置代理") is False
    assert slice_of("配置代理") == SLICE_OOV


def test_unrelated_query_is_oov():
    assert matched_synonyms("重置系统") == []
    assert is_in_vocabulary("重置系统") is False


def test_empty_query_is_oov():
    assert matched_synonyms("") == []
    assert is_in_vocabulary("") is False


def test_normalizes_before_matching():
    # 全角/大小写差异不该让覆盖判定与真实检索行为脱节。
    assert is_in_vocabulary("怎么设置代理　") is True


def test_coverage_follows_synonym_table_at_runtime(monkeypatch):
    """切片标签必须由别名表实时推导——往表里加规则，切片要跟着变。

    这是选 B 方案（打标签而非拆 held-out）的前提：手工标注会随别名表演进
    而腐坏，实时推导不会。两条读取路径（检索侧文件、切片侧模块属性）都要
    换，见 _pinned_vocab 的说明。
    """
    assert is_in_vocabulary("重置系统") is False
    new_table = [("重置系统", "恢复出厂设置")]
    vocab_file = _query._synonyms_path()
    vocab_file.write_text(
        "\n".join(f"{a} => {c}" for a, c in new_table) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(_query, "SYNONYMS", new_table)
    _query.reset_synonyms_cache()
    assert is_in_vocabulary("重置系统") is True
    assert is_in_vocabulary("怎么设置代理") is False  # 旧规则已不在表中


def test_in_slice_semantics():
    assert in_slice(True, SLICE_IN_VOCAB) is True
    assert in_slice(True, SLICE_OOV) is False
    assert in_slice(False, SLICE_OOV) is True
    assert in_slice(True, SLICE_ALL) is True
    assert in_slice(False, SLICE_ALL) is True
    # 未知切片名按 all 处理：汇报不该因为笔误而变成空表。
    assert in_slice(True, "typo") is True


# ---------- EvalReport 切片聚合 ----------

def _mixed_eval_set() -> EvalSet:
    return EvalSet(
        name="mixed",
        queries=[
            # in_vocab：命中「设置代理」别名
            EvalQuery(qid="v1", query="怎么设置代理", corpus_type="actions",
                      relevant=["network.configure_proxy"], top_k=5),
            # oov：规范词本身，不命中别名表
            EvalQuery(qid="o1", query="配置代理", corpus_type="actions",
                      relevant=["network.configure_proxy"], top_k=5),
            EvalQuery(qid="o2", query="导出审计日志", corpus_type="actions",
                      relevant=["network.configure_proxy"], top_k=5),
        ],
    )


def test_rows_carry_vocabulary_flag(workdir):
    report = run_eval(workdir, _mixed_eval_set(), strategies=("keyword",))
    flags = {r.qid: r.in_vocabulary for r in report.rows}
    assert flags == {"v1": True, "o1": False, "o2": False}


def test_slice_sizes(workdir):
    report = run_eval(workdir, _mixed_eval_set(), strategies=("keyword", "bm25"))
    sizes = report.slice_sizes()
    # 跨策略按 qid 去重，所以是查询数而不是行数。
    assert sizes == {SLICE_ALL: 3, SLICE_IN_VOCAB: 1, SLICE_OOV: 2}


def test_summary_slicing_partitions_rows(workdir):
    report = run_eval(workdir, _mixed_eval_set(), strategies=("keyword",))
    assert len(report.by_strategy(SLICE_IN_VOCAB)["keyword"]) == 1
    assert len(report.by_strategy(SLICE_OOV)["keyword"]) == 2
    assert len(report.by_strategy(SLICE_ALL)["keyword"]) == 3
    # 默认参数保持旧行为（不切片）。
    assert report.summary() == report.summary(slice_name=SLICE_ALL)


def test_summary_slices_can_differ(workdir):
    """in_vocab 与 oov 的指标应当能分开——分不开就说明切片没生效。"""
    report = run_eval(workdir, _mixed_eval_set(), strategies=("keyword",))
    iv = report.summary(keys=("recall@k",), slice_name=SLICE_IN_VOCAB)["keyword"]
    oov = report.summary(keys=("recall@k",), slice_name=SLICE_OOV)["keyword"]
    # o2「导出审计日志」检索不到代理动作，必然拉低 oov 切片。
    assert iv["recall@k"] > oov["recall@k"]


# ---------- format_summary ----------

def test_format_summary_shows_slice_table_when_both_present(workdir):
    report = run_eval(workdir, _mixed_eval_set(), strategies=("keyword",))
    text = format_summary(report)
    assert "词表覆盖切片" in text
    assert SLICE_IN_VOCAB in text
    assert SLICE_OOV in text


def test_format_summary_omits_slice_table_when_single_slice(workdir):
    """全是 oov 时切片表就是全集表的复制品，应当省略。"""
    only_oov = EvalSet(name="oov", queries=[
        EvalQuery(qid="o1", query="配置代理", corpus_type="actions",
                  relevant=["network.configure_proxy"], top_k=5),
    ])
    text = format_summary(run_eval(workdir, only_oov, strategies=("keyword",)))
    assert "词表覆盖切片" not in text


# ---------- tune 选优切片 ----------

def test_tune_selects_on_oov_slice_by_default(workdir):
    from edr_rag.eval.tune import tune_rrf_weights

    result = tune_rrf_weights(workdir, _mixed_eval_set(), metric="recall@k")
    assert result.eval_slice == SLICE_OOV


def test_tune_falls_back_to_all_when_slice_empty(workdir):
    """全部查询都被别名表覆盖时，oov 切片为空，必须退回 all 而不是静默给 0 分。"""
    from edr_rag.eval.tune import tune_rrf_weights

    all_in_vocab = EvalSet(name="iv", queries=[
        EvalQuery(qid="v1", query="怎么设置代理", corpus_type="actions",
                  relevant=["network.configure_proxy"], top_k=5),
    ])
    result = tune_rrf_weights(workdir, all_in_vocab, metric="recall@k")
    assert result.eval_slice == SLICE_ALL
    assert result.best_score > 0.0


def test_tune_honours_explicit_slice(workdir):
    from edr_rag.eval.tune import tune_rrf_weights

    result = tune_rrf_weights(
        workdir, _mixed_eval_set(), metric="recall@k", eval_slice=SLICE_ALL
    )
    assert result.eval_slice == SLICE_ALL
