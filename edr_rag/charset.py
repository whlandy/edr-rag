"""Charset-aware text reading for CHM-extracted files.

CHM 手册内的 HTML/.hhc 文件常见 GB2312/GBK 编码（厂商用中文 Windows 的
编辑器产出），且 <meta charset> 声明经常缺失或与实际不符。硬编码 UTF-8
读取会把中文 content / section_path 读成乱码——bytes 层面 UTF-8 解码对
GBK 双字节序列要么抛错要么经 errors="replace" 变 U+FFFD。

读取顺序：
  1. bytes 前缀里的 <meta charset=...> / <meta http-equiv content=...> 声明；
  2. 按声明解码；声明是 utf 类则按 utf-8 严格解码；
  3. 无声明或声明解码失败时，依次尝试 utf-8（严格）→ gb18030
     （GB2312/GBK 的超集，反向兼容两者）→ big5；
  4. 全部失败才退回 utf-8 + replace（与旧行为一致，但不丢元数据来源）。

gb18030 优先于 big5：大陆厂商手册 gb 系远多于繁体 big5；且 gb18030 对
ASCII 与 UTF-8 双字节中文均有判别失败的高置信度（不会把合法 UTF-8 误判）。
"""

from __future__ import annotations

import re
from pathlib import Path

#: <meta charset="gb2312"> / <meta http-equiv="Content-Type" content="...charset=gbk">
_META_CHARSET_RE = re.compile(
    rb"""<meta[^>]+?charset\s*=\s*["']?\s*([A-Za-z0-9_\-]+)""",
    re.IGNORECASE,
)

#: 声明为 utf 系时按严格 UTF-8 走（声明可信度高于试探）。
_UTF_ALIASES = {"utf-8", "utf8", "utf_8", "u8"}

#: 声明解码失败/无声明时的候选序列。
_FALLBACKS = ("utf-8", "gb18030", "big5")


def sniff_meta_charset(data: bytes) -> str | None:
    """从 bytes 前 4KB 提取 <meta charset> 声明（小写化）；无则 None。"""
    m = _META_CHARSET_RE.search(data[:4096])
    return m.group(1).decode("ascii", errors="ignore").lower() if m else None


def decode_text(data: bytes) -> str:
    """按 sniff → 声明 → 候选序列解码；全失败时 utf-8+replace 兜底。"""
    declared = sniff_meta_charset(data)
    if declared:
        candidates = (
            ("utf-8",) if declared in _UTF_ALIASES else (declared,)
        ) + _FALLBACKS
    else:
        candidates = _FALLBACKS
    seen: set[str] = set()
    for enc in candidates:
        if enc in seen:
            continue
        seen.add(enc)
        try:
            return data.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return data.decode("utf-8", errors="replace")


def read_text_charset_aware(path: Path) -> str:
    """读取文本文件，charset 感知（见模块 docstring）。"""
    return decode_text(path.read_bytes())
