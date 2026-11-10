"""中文 CHM 编码/路径修复的回归测试。

三处实测发现的 bug（详见各测试 docstring）：
  1. extract.py   — hh.exe 对含中文的 CHM 路径静默失败（rc=0 无产物），
                    须复制到 ASCII 临时路径重试（Windows 分支；本机非
                    Windows，测 _is_ascii_path 判据 + 辅助函数契约）。
  2. parse_html.py — CHM 内 HTML 为 GB2312/GBK，硬编码 utf-8 读取乱码。
  3. parse_toc.py  — .hhc 同为 GB2312/GBK，section_path 中文乱码。

charset.py 的判定表单独覆盖（声明 sniff、候选序列、兜底）。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from edr_rag.charset import decode_text, sniff_meta_charset  # noqa: E402
from edr_rag.extract import _is_ascii_path  # noqa: E402
from edr_rag.parse_html import parse_html_file  # noqa: E402
from edr_rag.parse_toc import parse_hhc  # noqa: E402


# ---------- extract: ASCII path predicate ----------

def test_ascii_path_accepts_ascii(tmp_path: Path):
    p = tmp_path / "manual.chm"
    p.write_bytes(b"x")
    assert _is_ascii_path(p) is True


def test_ascii_path_rejects_chinese_filename(tmp_path: Path):
    p = tmp_path / "终端安全管理系统.chm"
    p.write_bytes(b"x")
    assert _is_ascii_path(p) is False


def test_ascii_path_rejects_chinese_parent_dir(tmp_path: Path):
    d = tmp_path / "手册目录"
    d.mkdir()
    p = d / "manual.chm"
    p.write_bytes(b"x")
    assert _is_ascii_path(p) is False


# ---------- charset.py 判定表 ----------

def test_decode_utf8_bytes_roundtrip():
    assert decode_text("中文内容".encode("utf-8")) == "中文内容"


def test_decode_gb2312_without_meta():
    # 「病毒查杀」的 GB2312 字节流，无 meta 声明——旧硬编码 utf-8 会乱码。
    data = "病毒查杀与隔离".encode("gb2312")
    assert decode_text(data) == "病毒查杀与隔离"


def test_decode_gbk_extension_chars():
    # 「邝」是 GBK 扩展字（不在 GB2312 内），gb18030 应解出。
    assert decode_text("邝姓用户".encode("gb18030")) == "邝姓用户"


def test_decode_meta_declared_gb2312_with_utf8_bytes_falls_back():
    # 声明 gb2312 但字节流实为 utf-8（厂商声明与实际不符的常见形态）：
    # 声明解码失败后必须落到候选序列解出，而不是输出乱码。
    data = (
        '<html><head><meta charset="gb2312"></head><body>重置系统</body></html>'
    ).encode("utf-8")
    assert "重置系统" in decode_text(data)


def test_decode_meta_declared_utf8_strict():
    data = '<meta charset="UTF-8">配置代理'.encode("utf-8")
    assert decode_text(data) == '<meta charset="UTF-8">配置代理'


def test_sniff_meta_charset_variants():
    assert sniff_meta_charset(b'<meta charset="gb2312">') == "gb2312"
    assert sniff_meta_charset(
        b'<meta http-equiv="Content-Type" content="text/html; charset=GBK">'
    ) == "gbk"
    assert sniff_meta_charset(b"<html><body>no meta</body></html>") is None


def test_decode_undecodable_bytes_becomes_replacement_not_exception():
    # 候选全失败 → 兜底 replace，不抛异常（与旧行为兼容）。
    junk = b"\xff\xfe\x81\x81\xff"
    out = decode_text(junk)
    assert isinstance(out, str)


# ---------- parse_html: GB2312 content ----------

def test_parse_html_gb2312_content_not_mojibake(tmp_path: Path):
    """硬编码 utf-8 时 content 会变成乱码；修复后应正确解出中文。"""
    html = (
        "<html><head><title>配置代理服务器</title></head>"
        "<body><h1>配置代理服务器</h1>"
        "<p>本节介绍如何配置代理，需要网络设置。</p></body></html>"
    ).encode("gb2312")
    p = tmp_path / "page.html"
    p.write_bytes(html)
    page = parse_html_file(p, tmp_path)
    assert page.title == "配置代理服务器"
    assert "本节介绍如何配置代理" in page.content


def test_parse_html_gb2312_with_meta_charset(tmp_path: Path):
    html = (
        '<html><head><meta http-equiv="Content-Type" '
        'content="text/html; charset=gb2312">'
        "<title>查看安全状态</title></head><body>防护状态正常</body></html>"
    ).encode("gb2312")
    p = tmp_path / "page.html"
    p.write_bytes(html)
    page = parse_html_file(p, tmp_path)
    assert page.title == "查看安全状态"
    assert "防护状态正常" in page.content


def test_parse_html_utf8_still_works(tmp_path: Path):
    """修复不得破坏标准 UTF-8 页面。"""
    p = tmp_path / "page.html"
    p.write_text("<html><title>重置系统</title><body>恢复出厂设置</body></html>",
                 encoding="utf-8")
    page = parse_html_file(p, tmp_path)
    assert page.title == "重置系统"
    assert "恢复出厂设置" in page.content


# ---------- parse_toc: GB2312 section_path ----------

def test_parse_hhc_gb2312_section_path_not_mojibake(tmp_path: Path):
    """.hhc 硬编码 utf-8 时 section_path 中文变乱码；修复后应正确解出。"""
    hhc = (
        "<html><body><ul>"
        "<li><object type='text/sitemap'>"
        "<param name='Name' value='网络配置'>"
        "<param name='Local' value='network/proxy.html'>"
        "</object>"
        "<ul><li><object type='text/sitemap'>"
        "<param name='Name' value='配置代理服务器'>"
        "<param name='Local' value='network/proxy/detail.html'>"
        "</object></li></ul></li>"
        "</ul></body></html>"
    ).encode("gb2312")
    p = tmp_path / "toc.hhc"
    p.write_bytes(hhc)
    entries = parse_hhc(p)
    paths = {e.file: e.path for e in entries}
    assert paths["network/proxy/detail.html"] == ["网络配置", "配置代理服务器"]
    assert paths["network/proxy.html"] == ["网络配置"]


def test_parse_hhc_utf8_still_works(tmp_path: Path):
    """修复不得破坏 UTF-8 的 .hhc。"""
    hhc = (
        "<html><body><ul><li><object type='text/sitemap'>"
        "<param name='Name' value='系统维护'>"
        "<param name='Local' value='system/reset.html'>"
        "</object></li></ul></body></html>"
    )
    p = tmp_path / "toc.hhc"
    p.write_text(hhc, encoding="utf-8")
    entries = parse_hhc(p)
    assert entries[0].path == ["系统维护"]
