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
        cmd = [hh, "-decompile", str(out_dir), str(chm_path)]
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
