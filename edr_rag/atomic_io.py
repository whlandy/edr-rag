"""原子文件写入辅助 (P0-6).

产物重建 (索引/JSONL) 采用「临时文件写入 + rename」的原子方式，保证:

  - 目标文件要么是旧内容、要么是新内容，绝不会出现写了一半的残缺文件；
  - 写入中途 (异常/断电) 失败时不污染现有文件；
  - 临时文件与目标文件在同一目录，保证 rename 在同一文件系统内原子完成。

提供了两个入口:
  - atomic_write_text(path, text) : 原子写入 utf-8 文本内容
  - write_jsonl(path, records)    : 把可 JSON 化记录列表写成 JSONL (每行一个对象)

二者内部都走 atomically_replace_file 的「写临时文件 -> fsync -> replace」流程。
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable, TextIO

#: 临时文件后缀。写入完成后 rename 覆盖目标，其后缀保证不会与真实产物混淆。
_TMP_SUFFIX = ".edr-tmp"


def atomically_replace_file(path: Path, write: Callable[[TextIO], None]) -> None:
    """原子替换目标文件。

    Args:
        path: 目标文件路径 (不存在时等价于原子创建)。
        write: 接收一个已打开的临时文件类对象 (TextIO)，调用方负责写入内容。
              函数返回后 (且无异常) 才会把临时文件 rename 覆盖到 `path`。

    Raises:
        任何写入阶段的异常都会原样向上抛，临时文件会被清理，目标文件保持原状。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # 在同一目录创建临时文件，确保 replace 是同一文件系统内的原子 rename。
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=path.name + ".", suffix=_TMP_SUFFIX
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            write(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        # 清理残留临时文件，避免污染目录。
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def atomic_write_text(path: Path, text: str) -> None:
    """以 utf-8 原子写入文本内容到 `path`。"""

    def _write(f: TextIO) -> None:
        f.write(text)

    atomically_replace_file(path, _write)


def write_jsonl(path: Path, records: Iterable[Any]) -> None:
    """把记录列表原子写入 JSONL (每行一个 JSON 对象)。

    记录可以是 dict 或带 to_jsonl() 的对象。返回 None；成功后 `path` 内容即为
    这批记录的完整 JSONL。
    """

    def _write(f: TextIO) -> None:
        for r in records:
            to_jsonl = getattr(r, "to_jsonl", None)
            if callable(to_jsonl):
                obj = to_jsonl()
            else:
                obj = r
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    atomically_replace_file(path, _write)
