"""P4 —— 查询扩展词表的**反过拟合结构性检查**。

背景 (根因)：P1-6 把 16 条同义词硬编码在 ``query.py`` 的 ``SYNONYMS`` 里，
而这 16 条逐条对应 ``eval/data/eval_set_default.jsonl`` 中 3 个 action 的
``paraphrases``：

    评测集 q002.paraphrases = ["恢复出厂设置", "系统重置一下", "把系统初始化", ...]
    代码   SYNONYMS         = [("恢复出厂设置","重置系统"), ("系统重置","重置系统"), ...]

然后再用**同一份评测集**跑 ``eval/tune.py`` 的网格搜索去调 RRF 权重。
这构成一个自证闭环：指标只是在验证「我们把答案抄进了代码」，对真实查询
没有任何预测力。

这组测试把「不许从评测集反抄」变成一条会失败的断言，而不是一条注释。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from edr_rag.eval.dataset import load_eval_set
from edr_rag.query import expand_query, get_synonyms, reset_synonyms_cache

_EVAL_SET = Path(__file__).parent.parent / "eval" / "data" / "eval_set_default.jsonl"


@pytest.fixture(autouse=True)
def _fresh_vocab():
    reset_synonyms_cache()
    yield
    reset_synonyms_cache()


def _eval_paraphrases() -> set[str]:
    if not _EVAL_SET.exists():  # pragma: no cover - 评测集应随包携带
        pytest.skip("default eval set not present")
    eval_set = load_eval_set(_EVAL_SET)
    out: set[str] = set()
    for q in eval_set.queries:
        out.add(q.query.strip())
        for p in q.paraphrases:
            out.add(p.strip())
    return {s for s in out if s}


def test_synonyms_not_derived_from_eval_set():
    """随包同义表的**别名**不得逐字来自评测集的 query / paraphrases。

    这是 P4 的核心反过拟合约束。若有人把评测集里的改写抄进词表，这条会红。
    """
    paraphrases = _eval_paraphrases()
    aliases = {alias.strip() for alias, _ in get_synonyms()}
    leaked = aliases & paraphrases
    assert not leaked, (
        "同义词表的别名逐字抄自评测集，构成自证闭环：\n  "
        + "\n  ".join(sorted(leaked))
        + "\n词表来源只允许是产品术语或通用运维口语，不允许是评测集的改写。"
    )


def test_synonym_canonicals_not_eval_answers():
    """规范词也不应直接等于评测集里的 query 原文。

    别名可以指向手册书面语，但如果规范词恰好就是评测集的主查询，等价于
    把「查询 → 正确查询」的映射写死在代码里。
    """
    queries = set()
    if not _EVAL_SET.exists():  # pragma: no cover
        pytest.skip("default eval set not present")
    for q in load_eval_set(_EVAL_SET).queries:
        queries.add(q.query.strip())
    canonicals = {c.strip() for _, c in get_synonyms()}
    leaked = canonicals & queries
    assert not leaked, (
        "同义表的规范词直接等于评测集主查询：\n  " + "\n  ".join(sorted(leaked))
    )


def test_vocab_is_external_data_not_source_code():
    """词表必须是可替换的数据文件，而不是源码里的字面量。"""
    import edr_rag.query as query_mod

    src = Path(query_mod.__file__).read_text(encoding="utf-8")
    # 源码里不应再出现成对的中文字面量词条列表
    assert '("恢复出厂设置"' not in src
    assert '("查看防护状态"' not in src
    # 词表文件存在且可读
    assert (Path(query_mod.__file__).parent / "data" / "synonyms.zh.txt").is_file()


def test_expansion_still_works_on_shipped_vocab():
    """去掉评测集词条后，通用运维口语仍然能扩展（能力没被削没）。"""
    variants = expand_query("怎么看告警")
    assert variants[0] == "怎么看告警"
    # 「怎么看 => 查看」是通用口语条目，应产出变体
    assert len(variants) >= 2


def test_expansion_is_idempotent_and_deterministic():
    a = expand_query("把它打开并查一下日志")
    b = expand_query("把它打开并查一下日志")
    assert a == b
    assert a[0] == "把它打开并查一下日志"
