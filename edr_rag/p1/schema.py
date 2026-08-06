"""P1 schemas for procedure extraction and retrieval chunks.

This module lives under tools/edr_rag/p1/ to keep it separate from
P0 schemas while sharing the same extraction infrastructure.

P1 scope:
  - Procedure: structured step-by-step procedures extracted from ManualSection
  - RetrievalChunk: flat text chunks for local search
  - ExtractionMetadata: provenance info for extracted artifacts
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# ---------- Procedure ----------


@dataclass
class ProcedureStep:
    """A single ordered step in a procedure."""

    order: int
    instruction: str
    target: str = ""
    action_hint: str = ""
    source_instruction: str = ""

    def to_jsonl(self) -> dict:
        return asdict(self)


@dataclass
class ProcedureSource:
    """Provenance: which ManualSection produced this procedure."""

    file: str
    section_id: str
    anchor: Optional[str] = None

    def to_jsonl(self) -> dict:
        return asdict(self)


@dataclass
class Procedure:
    """A procedure extracted from a ManualSection.

    A procedure represents step-by-step instructions for a user-facing
    operation (e.g., "configure proxy", "enable firewall").

    Extraction is deterministic (heuristic-based), not LLM-dependent.
    LLM may optionally clean up results in a second pass.
    """

    id: str
    manual_id: str
    product: str
    operation: str
    title: str
    section_path: list[str]
    source: ProcedureSource
    preconditions: list[str] = field(default_factory=list)
    steps: list[ProcedureStep] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    recovery: list[str] = field(default_factory=list)
    confidence: float = 0.0
    status: str = "candidate"

    # Back-compat alias
    @property
    def status_label(self) -> str:
        return self.status

    def to_jsonl(self) -> dict:
        d = asdict(self)
        d["source"] = self.source.to_jsonl()
        d["steps"] = [s.to_jsonl() for s in self.steps]
        return d


# ---------- RetrievalChunk ----------


@dataclass
class ChunkSource:
    """Provenance for a retrieval chunk."""

    section_id: str
    procedure_id: Optional[str] = None
    file: str = ""

    def to_jsonl(self) -> dict:
        return asdict(self)


@dataclass
class RetrievalChunk:
    """A flat text chunk for local JSONL search.

    P1 chunk types:
      - "procedure": compiled from Procedure steps + preconditions + warnings
      - "section": compiled from a ManualSection (content + headings + links)
      - "step": a single ProcedureStep (fine-grained search)
    """

    id: str
    manual_id: str
    type: str  # procedure | section | step
    text: str
    source: ChunkSource
    metadata: dict = field(default_factory=dict)

    def to_jsonl(self) -> dict:
        d = asdict(self)
        d["source"] = self.source.to_jsonl()
        return d


# ---------- ExtractionMetadata ----------


@dataclass
class ExtractionMetadata:
    """Metadata about a P1 extraction run."""

    manual_id: str
    product: str
    generated_at: str
    section_count: int
    procedure_count: int
    chunk_count: int
    extraction_version: str = "p1"

    def to_json(self) -> dict:
        return asdict(self)


# ---------- Helpers ----------


def make_procedure_id(operation: str) -> str:
    """Generate a stable procedure id from an operation name.

    Format: procedure.{product}.{operation}
    Example: procedure.network.configure_proxy
    """
    return f"procedure.{operation.replace(' ', '_').replace('，', '_').strip().lower()}"


def make_chunk_id(chunk_type: str, source_id: str) -> str:
    """Generate a chunk id.

    Format: chunk.{type}.{source_id_segment}
    Example: chunk.procedure.network.configure_proxy
    """
    # Replace dots in source_id to avoid ambiguity
    safe = source_id.replace(".", "_")
    return f"chunk.{chunk_type}.{safe}"
