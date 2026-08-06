"""section_path priority contract — resolve a page's section_path with fallback.

P0 priority (LOCKED in schema.py):
  1. hhc hierarchy           (file → TOC entry path)
  2. html heading hierarchy  (page's h1 list)
  3. [title]                 (single-element path from page title)
  4. []                      (empty — orphan page)

This module is a pure function — no side effects, no filesystem I/O beyond
what callers do. Easy to test in isolation.
"""

from __future__ import annotations

from typing import Optional


def resolve_section_path(
    *,
    hhc_index: dict[str, list[str]],
    file: str,
    headings: list[str],
    title: str,
) -> list[str]:
    """Resolve section_path using the P0 priority contract.

    Args:
        hhc_index: map of normalized relative source file (e.g. "network/proxy.html")
                   to section_path from the .hhc. May be empty if .hhc missing/failed.
        file: normalized relative source file of the current page (same key form).
        headings: list of h1-h6 texts from the HTML page (in document order).
        title: page title (from <title> or first h1).

    Returns:
        The section_path list. Never None. Always at least [].
    """
    # Priority 1: hhc hierarchy
    key = file.replace("\\", "/").lstrip("/")
    hhc_path = hhc_index.get(key)
    if hhc_path:
        return list(hhc_path)

    # Priority 2: html heading hierarchy.
    # P0 interpretation: use the full heading list (filtered to non-empty)
    # as a flat section_path. Multi-h1 pages collapse to a flat path;
    # deeper h2/h3 nesting is P1 territory.
    cleaned = [h.strip() for h in headings if h and h.strip()]
    if cleaned:
        return cleaned

    # Priority 3: [title]
    t = (title or "").strip()
    if t:
        return [t]

    # Priority 4: []
    return []


def first_h1(headings: list[str]) -> str:
    """Return the first non-empty heading, or "".

    Pure helper exposed for callers that want a single-string fallback
    (e.g. for semantic id generation in normalize.py — P0 commits to using
    semantic content, not path hashes).
    """
    for h in headings:
        if h and h.strip():
            return h.strip()
    return ""