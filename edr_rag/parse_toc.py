"""TOC parsing — build section_path hierarchy from the CHM .hhc file.

The .hhc (HTML Help Contents) file is an HTML file with <UL>/<LI>/<OBJECT>
entries that define the manual's table of contents. Each entry typically has:
  <param name="Name" value="...">
  <param name="Local" value="filename.html">

We walk the nested <UL> structure to derive section_path for every page.
If parsing fails, callers (normalize.py) fall back to HTML heading hierarchy.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover
    BeautifulSoup = None


class TocEntry:
    """A single entry in the TOC."""
    __slots__ = ("name", "file", "path")

    def __init__(self, name: str, file: Optional[str], path: list[str]):
        self.name = name
        self.file = file  # relative source file, may be None for group nodes
        self.path = path  # hierarchical section path

    def __repr__(self) -> str:  # pragma: no cover
        return f"TocEntry(name={self.name!r}, file={self.file!r}, path={self.path!r})"


def parse_hhc(hhc_path: Path) -> list[TocEntry]:
    """Parse a .hhc file into a flat list of (name, file, section_path) entries."""
    if BeautifulSoup is None:
        raise RuntimeError("BeautifulSoup (bs4) is required for TOC parsing")

    text = hhc_path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(text, "html.parser")

    entries: list[TocEntry] = []

    def walk(ul, current_path: list[str]):
        for li in ul.find_all("li", recursive=False):
            # Gather params under this li
            name = None
            local = None
            obj = li.find("object")
            if obj:
                for param in obj.find_all("param"):
                    pname = param.get("name")
                    pval = param.get("value")
                    if pname == "Name":
                        name = pval
                    elif pname == "Local":
                        local = pval
            if name is None:
                # Fallback: use first <a> text
                a = li.find("a")
                if a:
                    name = a.get_text(strip=True)
            if name is None:
                name = ""

            new_path = current_path + [name]
            if local:
                entries.append(TocEntry(name=name, file=local, path=new_path))

            # Recurse into nested UL
            child_ul = li.find("ul", recursive=False)
            if child_ul:
                walk(child_ul, new_path)
        # also handle direct li not grouped

    top_ul = None
    # find first UL with LI children
    for ul in soup.find_all("ul"):
        if ul.find("li", recursive=False):
            top_ul = ul
            break
    if top_ul is None:
        logger.warning("No UL found in TOC file %s", hhc_path)
        return []

    walk(top_ul, [])
    return entries


def build_file_to_path_index(entries: list[TocEntry]) -> dict[str, list[str]]:
    """Map relative source file -> section_path (from TOC)."""
    index: dict[str, list[str]] = {}
    for e in entries:
        if e.file:
            # Normalize backslashes to forward slashes
            key = e.file.replace("\\", "/").lstrip("/")
            index[key] = e.path
    return index
