"""HTML page parsing — extract cleaned text, headings, links, and image references.

Each HTML page from the extracted CHM is parsed to extract:
  - Cleaned text content (tags stripped, but preserving list structure)
  - Headings (h1-h6) as a list for section hierarchy fallback
  - Links (anchor hrefs)
  - Image references (img src, alt)
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

try:
    from bs4 import BeautifulSoup, NavigableString, Tag
except ImportError:  # pragma: no cover
    BeautifulSoup = None

logger = logging.getLogger(__name__)


class ParsedPage:
    """Output of HTML parsing for one page."""
    __slots__ = ("title", "content", "headings", "links", "images")

    def __init__(
        self,
        title: str,
        content: str,
        headings: list[str],
        links: list[dict[str, str]],
        images: list[dict[str, Optional[str]]],
    ):
        self.title = title
        self.content = content
        self.headings = headings
        self.links = links
        self.images = images

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"ParsedPage(title={self.title!r}, content_len={len(self.content)}, "
            f"headings={len(self.headings)}, links={len(self.links)}, images={len(self.images)})"
        )


def _clean_text(element) -> str:
    """Extract text from a BeautifulSoup element, preserving list structure.

    Container tags (html, body, p, div, section, article, blockquote, pre)
    are recursed into. Leaf-ish tags are extracted via get_text().
    """
    lines: list[str] = []

    def _walk(elm):
        for child in elm.children:
            if isinstance(child, NavigableString):
                t = str(child).strip()
                if t:
                    lines.append(t)
            elif isinstance(child, Tag):
                tag = child.name.lower() if child.name else ""
                if tag in ("ul", "ol"):
                    for i, li in enumerate(child.find_all("li", recursive=False), start=1):
                        if tag == "ul":
                            prefix = "  * "
                        else:
                            prefix = f"  {i}. "
                        text = li.get_text(" ", strip=True)
                        if text:
                            lines.append(prefix + text)
                elif tag in ("p", "div", "section", "article", "blockquote",
                             "pre", "body", "html", "td", "th", "tr", "table"):
                    _walk(child)
                elif tag in ("br",):
                    lines.append("")
                elif tag in ("img",):
                    # Skip images in text content — they are captured separately
                    pass
                else:
                    t = child.get_text(" ", strip=True)
                    if t:
                        lines.append(t)

    _walk(element)
    # Normalize whitespace
    return " ".join(l.strip() for l in lines if l.strip())


def _extract_headings(soup) -> list[str]:
    """Extract all heading texts from h1-h6 in document order.

    Document order is preferred over level-then-order because section_path
    fallback uses the heading list as a flat path — reading order is what
    the user perceives as "section titles in this page".
    """
    headings: list[str] = []
    for h in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        text = h.get_text(" ", strip=True)
        if text:
            headings.append(text)
    return headings


def _extract_links(soup, page_path: Path, extracted_dir: Path) -> list[dict[str, str]]:
    """Extract anchor links, resolving relative paths."""
    links: list[dict[str, str]] = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        text = a.get_text(" ", strip=True)
        links.append({"text": text, "href": href})
    return links


def _extract_images(soup, page_path: Path, extracted_dir: Path) -> list[dict[str, Optional[str]]]:
    """Extract img tags with src and alt, producing asset IDs."""
    images: list[dict[str, Optional[str]]] = []
    for img in soup.find_all("img"):
        src = img.get("src", "")
        alt = img.get("alt", "")
        if src:
            # Normalize relative path
            src_norm = src.replace("\\", "/").lstrip("/")
            # Build an asset ID from image path
            asset_id = f"asset.{src_norm.replace('/', '.').replace(' ', '_')}"
            images.append({"src": src_norm, "alt": alt, "asset_id": asset_id})
    return images


def parse_html_file(html_path: Path, extracted_dir: Path) -> ParsedPage:
    """Parse a single HTML file from the extracted CHM.

    Returns ParsedPage with cleaned content, headings, links, and images.
    """
    if BeautifulSoup is None:
        raise RuntimeError("BeautifulSoup (bs4) is required for HTML parsing")

    text = html_path.read_text(encoding="utf-8", errors="replace")

    soup = BeautifulSoup(text, "html.parser")

    # Title from <title> or first h1
    title_tag = soup.find("title")
    title = title_tag.get_text(" ", strip=True) if title_tag else ""
    if not title:
        h1 = soup.find("h1")
        if h1:
            title = h1.get_text(" ", strip=True)

    # Remove script/style elements for cleaning
    for script in soup(["script", "style", "noscript", "meta", "link"]):
        script.decompose()

    content = _clean_text(soup)
    headings = _extract_headings(soup)
    links = _extract_links(soup, html_path, extracted_dir)
    images = _extract_images(soup, html_path, extracted_dir)

    return ParsedPage(title=title, content=content, headings=headings, links=links, images=images)
