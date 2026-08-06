"""P0 CHM ingestion pipeline — CHM → Manual IR.

Converts a CHM product manual into structured, inspectable artifacts:
  - manifest.json        : describes the ingestion run
  - sections.jsonl       : one ManualSection per useful HTML page
  - assets.jsonl         : one AssetRef per image/resource

Deterministic. No embeddings, no vector DB, no LLM extraction required.
"""

from . import extract, parse_html, parse_toc, normalize, schema  # noqa: F401

__all__ = ["extract", "parse_html", "parse_toc", "normalize", "schema"]
