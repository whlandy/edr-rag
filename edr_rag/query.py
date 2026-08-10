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

import os
import re
import unicodedata
from pathlib import Path
from typing import Optional

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

#: 词表文件路径。可用 EDR_RAG_SYNONYMS 覆盖（客户现场维护自己产品的术语）。
_DEFAULT_SYNONYMS_FILE = Path(__file__).parent / "data" / "synonyms.zh.txt"


def _synonyms_path() -> Path:
    override = os.environ.get("EDR_RAG_SYNONYMS", "").strip()
    return Path(override) if override else _DEFAULT_SYNONYMS_FILE


def load_synonyms(path: Optional[Path] = None) -> list[tuple[str, str]]:
    """从词表文件加载 (别名, 规范词) 列表。

    格式：每行 ``别名 => 规范词``；``#`` 起始为注释，空行忽略。
    文件缺失或不可读时返回空表（查询扩展退化为「只做规范化」，主链路不受影响）。

    返回列表按**别名长度降序**排序 —— 更具体（更长）的别名优先匹配，避免
    短别名把长别名的机会吃掉（例如「查看」不应抢在「怎么查看」之前命中）。
    """
    p = path or _synonyms_path()
    try:
        text = p.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    pairs: list[tuple[str, str]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=>" not in line:
            continue
        alias, _, canonical = line.partition("=>")
        alias, canonical = alias.strip(), canonical.strip()
        if alias and canonical and alias != canonical:
            pairs.append((alias, canonical))
    pairs.sort(key=lambda kv: -len(kv[0]))
    return pairs


#: 进程内缓存的同义表（惰性加载）。
_SYNONYMS_CACHE: Optional[list[tuple[str, str]]] = None


def get_synonyms() -> list[tuple[str, str]]:
    """返回同义表（进程内缓存）。"""
    global _SYNONYMS_CACHE
    if _SYNONYMS_CACHE is None:
        _SYNONYMS_CACHE = load_synonyms()
    return _SYNONYMS_CACHE


def reset_synonyms_cache() -> None:
    """清空同义表缓存（测试 / 热更新词表后调用）。"""
    global _SYNONYMS_CACHE
    _SYNONYMS_CACHE = None


def _legacy_synonyms() -> list[tuple[str, str]]:
    """向后兼容：模块级 SYNONYMS 名称保留，但改为读词表文件。

    P1-6 的实现把 16 条别名硬编码在这里，而这 16 条**逐条对应**评测集
    (eval/data/eval_set_default.jsonl) 里 3 个 action 的 paraphrases，
    再用同一份评测集去调 RRF 权重 —— 指标只是在验证「我们把答案抄进了
    代码」。P4 把词表外置到 data/synonyms.zh.txt，并加结构性测试禁止
    从评测集反抄（见 tests/test_query_expansion.py）。
    """
    return get_synonyms()


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
    for alias, canonical in get_synonyms():
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
