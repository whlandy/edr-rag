"""CHM extraction — unpack a .chm file into extracted/ HTML tree.

Platform strategy:
  - Windows: uses `hh.exe -decompile <out> <chm>`
  - Linux:   uses `extract_chmLib <chm> <out>`
  - macOS:   uses `extract_chmLib` if available; otherwise errors with guidance

The extracted directory is preserved (never auto-deleted).
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


class ChmExtractionError(RuntimeError):
    """Raised when CHM extraction fails."""


def _find_extract_chmlib() -> str | None:
    """Locate extract_chmLib binary on PATH (Linux/macOS)."""
    return shutil.which("extract_chmLib")


def _hash_file(path: Path) -> str:
    """Compute sha256 of the CHM file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_ascii_path(p: Path) -> bool:
    """True if every path component is pure ASCII.

    hh.exe (-decompile) 静默拒绝含非 ASCII 字符的路径：退出码 0、无输出、
    无 stderr。CHM 手册文件名常见中文（如「终端安全管理系统.chm」），必须
    先复制到 ASCII 临时路径再交给 hh.exe。
    """
    try:
        str(p).encode("ascii")
    except UnicodeEncodeError:
        return False
    return True


def extract_chm(chm_path: Path, out_dir: Path) -> tuple[str, bool]:
    """Extract a CHM file into out_dir.

    Returns (source_hash, success). Raises ChmExtractionError on failure.
    """
    if not chm_path.is_file():
        raise ChmExtractionError(f"CHM file not found: {chm_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    source_hash = _hash_file(chm_path)

    if sys.platform.startswith("win"):
        # Windows: hh.exe -decompile
        hh = shutil.which("hh.exe")
        if hh is None:
            raise ChmExtractionError("hh.exe not found on PATH (Windows)")
        if _is_ascii_path(chm_path):
            cmd = [hh, "-decompile", str(out_dir), str(chm_path)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0 and any(
                p.is_file() for p in out_dir.rglob("*")
            ):
                logger.info("Extracted %d files to %s",
                            len([p for p in out_dir.rglob("*") if p.is_file()]),
                            out_dir)
                return source_hash, True
            # rc=0 但无产物也是 hh.exe 对非 ASCII 路径的失败形态，落到
            # ASCII 暂存路径重试；真坏档则重试后同样无产物、按失败报出。
            logger.warning(
                "hh.exe produced no output (rc=%d), retrying via ASCII temp path",
                result.returncode,
            )
        with tempfile.TemporaryDirectory(prefix="edr_rag_chm_") as td:
            td_path = Path(td)
            ascii_chm = td_path / "manual.chm"
            shutil.copyfile(chm_path, ascii_chm)
            ascii_out = td_path / "extracted"
            ascii_out.mkdir()
            cmd = [hh, "-decompile", str(ascii_out), str(ascii_chm)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise ChmExtractionError(
                    f"CHM extraction failed (rc={result.returncode}): "
                    f"{result.stderr.strip()}"
                )
            extracted = list(ascii_out.rglob("*"))
            if not extracted:
                raise ChmExtractionError("CHM extraction produced no files")
            for src in extracted:
                rel = src.relative_to(ascii_out)
                dst = out_dir / rel
                if src.is_dir():
                    dst.mkdir(parents=True, exist_ok=True)
                else:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(src, dst)
    elif sys.platform.startswith("darwin") or sys.platform.startswith("linux"):
        bin_path = _find_extract_chmlib()
        if bin_path is None:
            raise ChmExtractionError(
                "extract_chmLib not found on PATH. Install chmlib "
                "(e.g. brew install chmlib / apt install libchm-bin)."
            )
        cmd = [bin_path, str(chm_path), str(out_dir)]
    else:
        raise ChmExtractionError(f"Unsupported platform: {sys.platform}")

    logger.info("Extracting CHM: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise ChmExtractionError(
            f"CHM extraction failed (rc={result.returncode}): {result.stderr.strip()}"
        )

    # Verify output has content
    files = [p for p in out_dir.rglob("*") if p.is_file()]
    if not files:
        raise ChmExtractionError("CHM extraction produced no files")

    logger.info("Extracted %d files to %s", len(files), out_dir)
    return source_hash, True
