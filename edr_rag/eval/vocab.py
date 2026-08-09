"""评测集的词表覆盖切片 —— in-vocabulary / OOV。

问题:
  edr_rag/query.py 的 ``SYNONYMS`` 别名表是直接照着本评测集的 ``paraphrases``
  字段写的（该表的 docstring 也写明了）。于是「开启 query expansion 后召回
  提升了多少」这个数字，在被别名表覆盖的查询上衡量的是**查表命中率**，
  而不是泛化能力——表里写了「恢复出厂设置→重置系统」，那条改写当然能召回。

  更要命的是这种污染会往下游复利：eval/tune.py 在同一批查询上网格搜索 RRF
  融合权重，等于在「查表已经保证命中」的样本上调参，选出来的权重未必对真实
  查询更好。

做法（不删数据，只做切片）:
  评测集只有 30 条主查询，摘出去做 held-out 会让样本更没有统计意义；而且
  别名表本身是一个正当特性——你需要一个数字证明它在生效。所以这里保留全部
  数据，只把每条查询打上切片标签，分别汇报:

    in_vocab  命中别名表。其召回含查表贡献，用于回归「别名表是否还在生效」，
              **不可**用于判断泛化能力。
    oov       未命中别名表。这才是泛化能力的判据，也是 tune 默认的选优切片。

关键设计:
  切片标签由 ``SYNONYMS`` **实时推导**，不是写死在评测集数据文件里的字段。
  往别名表里加一条规则，对应查询会自动从 oov 移进 in_vocab，切片语义不会
  随时间腐坏——手工标注做不到这一点。

  判据用「别名是否作为子串出现」而不是「expand_query 是否真的产出了变体」:
  后者受 ``max_variants`` 截断影响，同一条查询会因为表里前面的规则占满名额
  而时而算 in_vocab 时而算 oov。覆盖与否应当只取决于表里有没有这条规则。
"""

from __future__ import annotations

from .. import query as _query
from ..query import normalize_query

#: 切片名。``all`` 为不切片的全集。
SLICE_ALL = "all"
SLICE_IN_VOCAB = "in_vocab"
SLICE_OOV = "oov"

#: 全部合法切片名（供 CLI choices / 校验）。
EVAL_SLICES = (SLICE_ALL, SLICE_IN_VOCAB, SLICE_OOV)


def matched_synonyms(query: str) -> list[str]:
    """返回该查询命中的别名列表（顺序同 SYNONYMS）；未命中为 []。

    比较前先做与检索侧一致的 ``normalize_query``，保证全半角/大小写差异
    不会让覆盖判定和真实检索行为脱节。
    """
    base = normalize_query(query)
    if not base:
        return []
    # 走模块属性而不是 from-import 绑定：别名表在运行时被替换（测试 /
    # 未来做成可配置）时，切片判定要跟着变，否则「实时推导」就是假的。
    return [alias for alias, _ in _query.SYNONYMS if alias in base]


def is_in_vocabulary(query: str) -> bool:
    """该查询是否被 SYNONYMS 别名表覆盖。"""
    return bool(matched_synonyms(query))


def slice_of(query: str) -> str:
    """返回该查询所属切片：SLICE_IN_VOCAB 或 SLICE_OOV。"""
    return SLICE_IN_VOCAB if is_in_vocabulary(query) else SLICE_OOV


def in_slice(query_in_vocab: bool, slice_name: str) -> bool:
    """判断一条（已知 in_vocab 标记的）查询是否属于目标切片。

    Args:
        query_in_vocab: 该查询的 in-vocabulary 标记。
        slice_name: 目标切片名；``all`` 恒为 True，未知切片名同样按 ``all``
            处理（宽松：评测汇报不该因为切片名笔误而变成空表）。
    """
    if slice_name == SLICE_IN_VOCAB:
        return query_in_vocab
    if slice_name == SLICE_OOV:
        return not query_in_vocab
    return True
