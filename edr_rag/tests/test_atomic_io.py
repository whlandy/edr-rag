"""(P0-6) tests: atomic artifact write helper (edr_rag.atomic_io).

Verifies the 临时文件 + rename 原子写语义:
  - write_jsonl / atomic_write_text 产生与直接写相同的内容
  - 目标文件被原子替换，且替换期间不出现残缺中间态
  - 写入失败时目标文件保持原状，且临时文件被清理
  - 支持嵌套目录自动创建
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from edr_rag.atomic_io import (
    _TMP_SUFFIX,
    atomically_replace_file,
    atomic_write_text,
    write_jsonl,
)


def test_atomic_write_text_content(tmp_path: Path):
    target = tmp_path / "sub" / "out.txt"  # 目录尚不存在
    atomic_write_text(target, "你好\nworld\n")
    assert target.exists()
    assert target.read_text(encoding="utf-8") == "你好\nworld\n"


def test_atomic_write_text_replaces_existing(tmp_path: Path):
    target = tmp_path / "out.txt"
    atomic_write_text(target, "old")
    atomic_write_text(target, "new")
    assert target.read_text(encoding="utf-8") == "new"


def test_write_jsonl(tmp_path: Path):
    target = tmp_path / "data.jsonl"
    records = [
        {"id": "a", "title": "甲"},
        {"id": "b", "title": "乙"},
    ]
    write_jsonl(target, records)
    lines = target.read_text(encoding="utf-8").strip().splitlines()
    assert [json.loads(l) for l in lines] == records


def test_write_jsonl_object_with_to_jsonl(tmp_path: Path):
    class Rec:
        def __init__(self, i):
            self.i = i

        def to_jsonl(self):
            return {"i": self.i}

    target = tmp_path / "data.jsonl"
    write_jsonl(target, [Rec(1), Rec(2)])
    lines = target.read_text(encoding="utf-8").strip().splitlines()
    assert [json.loads(l) for l in lines] == [{"i": 1}, {"i": 2}]


def test_atomic_write_no_tmp_left_behind(tmp_path: Path):
    target = tmp_path / "out.txt"
    atomic_write_text(target, "x")
    leftovers = [p for p in tmp_path.iterdir() if _TMP_SUFFIX in p.name]
    assert leftovers == []


def test_atomic_write_failure_keeps_old_file_and_cleans_tmp(tmp_path: Path):
    target = tmp_path / "out.txt"
    atomic_write_text(target, "keep-me")

    def boom(f):
        f.write("partial")
        raise RuntimeError("simulated crash")

    with pytest.raises(RuntimeError):
        atomically_replace_file(target, boom)

    # 目标文件保持原状，未被部分写入污染
    assert target.read_text(encoding="utf-8") == "keep-me"
    # 临时文件被清理
    leftovers = [p for p in tmp_path.iterdir() if _TMP_SUFFIX in p.name]
    assert leftovers == []


def test_atomic_write_failure_when_target_missing(tmp_path: Path):
    # 目标原本不存在且写入失败 -> 不残留任何文件
    target = tmp_path / "out.txt"

    def boom(f):
        f.write("x")
        raise ValueError("boom")

    with pytest.raises(ValueError):
        atomically_replace_file(target, boom)
    assert not target.exists()
    assert list(tmp_path.iterdir()) == []
