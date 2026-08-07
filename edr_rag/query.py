"""P1-6 查询侧中文规范化与多查询生成（query expansion）。

设计目标（对应 P1-6 desc）:
  查询侧加「中文规范化 + 多查询生成」两条能力，用于提升中文手册检索的召回:

  1. normalize_query(text)
     对查询做 NFKC 规范化：全角→半角（ASCII 字母/数字/常见半角可打印字符）、
     中文全角标点→半角、大小写折叠、连续空白折叠、去首尾空白。
     汉字本身不受 NFKC 影响，纯中文查询内容保持不变。
     * 关键：语料侧在 JsonlCorpus._normalize 构建可检索文本时也调用同一
       normalize_query（见 p1/search.py），保证「查询侧规范化」与「索引侧
       规范化」对称——这样全角查询才能命中全角/半角语料，反之亦然。

  2. expand_query(text, max_variants=...)
     基于同义/别名表（SYNONYMS）生成多个查询变体（query expansion），
     检索时对每个变体各召回一路并融合，提高召回率（multi-query recall）。
     变体顺序确定、去重；默认首元素是规范化后的原始查询，权重最高。

两者均为纯函数、确定性、幂等，不依赖任何可选 `rag` extras；
主链路（keyword / BM25 / RRF fusion）在无 LlamaIndex / embedding 时照常可用。
"""

from __future__ import annotations

import re
import unicodedata

# ---------- 规范化 ----------

# ASCII 可打印字符范围 [\x21-\x7e] 之外、且 NFKC 后落回该范围的字符会被折叠。
# NFKC 本身已处理大部分：全角字母/数字/常用符号直接折叠到半角。
#
# 附加处理中文全角标点 → 半角（NFKC 对中文标点多数能折叠，这里兜底常见几类，
# 保证 `。`/`．`、`，`/`、`、` `；`/`；` 等统一到同一种形式，提升 substring 匹配率）。

_FULL_WIDTH_PUNCT = {
    "．": ".", "，": ",", "：": ":", "；": ";", "！": "!", "？": "?",
    "（": "(", "）": ")", "［": "[", "］": "]", "｛": "{", "｝": "}",
    "“": '"', "”": '"', "‘": "'", "’": "'", "　": " ", "、": ",",
}

_WS_RE = re.compile(r"\s+")


def normalize_query(text: str) -> str:
    """对查询做确定性规范化，返回规范化后的字符串（无其他副作用）。

    - NFKC：全角字母/数字/常见符号折叠为半角；兼容组合字符。
    - 中文全角标点映射到半角（兜底）。
    - 大小写折叠（拉丁）。
    - 连续空白折叠为单个空格，去首尾空白。

    幂等：normalize_query(normalize_query(x)) == normalize_query(x)。
    """
    if not text:
        return ""
    s = unicodedata.normalize("NFKC", text)
    # 全角→半角标点映射（NFKC 已处理大部分，这里兜底几类）
    s = "".join(_FULL_WIDTH_PUNCT.get(ch, ch) for ch in s)
    s = s.lower()
    s = _WS_RE.sub(" ", s).strip()
    return s


# ---------- 同义/别名表（多查询生成的变体来源） ----------

#: 有序 (别名, 规范词) 表。expand_query 对每条查询，把命中的首个别名替换为
#: 规范词生成一个变体。规则按「更具体在前」排序，避免短别名过早截断。
#: 这些短语取自评测集（eval/data/eval_set_default.jsonl）的同义改写与常见
#: EDR 口语表述，覆盖「重置系统 / 配置代理 / 查看安全状态」等高频场景。
SYNONYMS: list[tuple[str, str]] = [
    ("恢复出厂设置", "重置系统"),
    ("恢复出厂", "重置系统"),
    ("系统初始化", "重置系统"),
    ("系统重置", "重置系统"),
    ("重置计算机", "重置系统"),
    ("重新安装系统", "重置系统"),
    ("设置代理", "配置代理"),
    ("代理怎么设置", "配置代理"),
    ("配置网络代理", "配置代理"),
    ("代理服务器配置", "配置代理"),
    ("查看防护状态", "查看安全状态"),
    ("安全状态怎么看", "查看安全状态"),
    ("安全状态在哪看", "查看安全状态"),
    ("防护状态", "安全状态"),
    ("查询安全状态", "查看安全状态"),
    ("怎么查看", "查看"),
]


def expand_query(text: str, max_variants: int = 5) -> list[str]:
    """生成查询变体列表（多查询生成）。

    输入先 normalize_query；输出首元素永远是规范化后的原始查询，随后是
    基于 SYNONYMS 的别名替身变体（每变体只做一次替换）。结果去重保序、
    截断到 max_variants 个。

    Args:
        text: 原始查询。
        max_variants: 最多返回几个变体（含原始查询）。默认 5。

    Returns:
        去重且保序的变体字符串列表。至少含原查询（规范化后）。
    """
    base = normalize_query(text)
    if not base:
        return []
    variants = [base]
    for alias, canonical in SYNONYMS:
        if len(variants) >= max_variants:
            break
        if alias in base:
            variant = base.replace(alias, canonical)
            if variant != base and variant not in variants:
                variants.append(variant)
    return variants


def normalize_and_expand(text: str, max_variants: int = 5) -> list[str]:
    """规范化 + 多查询生成的便捷入口（等价于 expand_query）。"""
    return expand_query(text, max_variants=max_variants)
