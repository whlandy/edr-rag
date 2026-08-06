"""P2 schema: ActionCatalogEntry and related dataclasses.

P2 scope (design §6):
  - Convert procedures into action candidates that a planner can reason about.
  - Each action has risk classification, confirmation policy, and precondition mapping.
  - High-risk actions must not become `published` without review.

Action lifecycle:
  candidate -> needs_review -> approved -> published -> deprecated
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class ActionStep:
    """A single step in an action plan, mapped from a ProcedureStep.

    The 'action' field is mapped from procedure step action_hint.
    The 'target' field is the UI element or object the action operates on.
    """

    order: int
    action: str  # open_window, click, select, input_text, check, close, etc.
    target: str
    source_instruction: str = ""
    value_source: str = ""  # "user", "system", or empty

    def to_jsonl(self) -> dict:
        return asdict(self)


@dataclass
class ActionSource:
    """Provenance: links back to the source procedure and manual section."""

    manual_id: str
    procedure_id: str
    file: str

    def to_jsonl(self) -> dict:
        return asdict(self)


@dataclass
class ValidationMetadata:
    """Validation flags for an action candidate."""

    has_steps: bool = False
    has_risk: bool = False
    has_recovery: bool = False
    source_verified: bool = False

    def to_jsonl(self) -> dict:
        return asdict(self)


@dataclass
class ActionCatalogEntry:
    """A reviewable action candidate from a procedure.

    Fields follow the design doc schema (§6):
      action_id: network.configure_proxy
      product: ProductName
      module: network
      title: 配置代理服务器
      status: needs_review (never auto-published for medium/high risk)
      risk: low|medium|high
      requires_confirmation: bool
      preconditions: list[str]
      procedure: list[ActionStep]
      success_criteria: list[str]
      recovery: list[str]
      source: ActionSource
      validation: ValidationMetadata
    """

    action_id: str
    product: str
    module: str
    title: str
    status: str  # candidate | needs_review | approved | published | deprecated
    confidence: float
    risk: str  # low | medium | high
    requires_confirmation: bool
    preconditions: list[str] = field(default_factory=list)
    procedure: list[ActionStep] = field(default_factory=list)
    success_criteria: list[str] = field(default_factory=list)
    recovery: list[str] = field(default_factory=list)
    source: Optional[ActionSource] = None
    validation: Optional[ValidationMetadata] = None

    def to_jsonl(self) -> dict:
        d = asdict(self)
        if self.source:
            d["source"] = self.source.to_jsonl()
        if self.validation:
            d["validation"] = self.validation.to_jsonl()
        d["procedure"] = [s.to_jsonl() for s in self.procedure]
        return d

    def to_yaml_lines(self) -> list[str]:
        """Render as YAML-like lines (no yaml dependency needed)."""
        lines = [f"action_id: {self.action_id}"]
        lines.append(f"product: {self.product}")
        lines.append(f"module: {self.module}")
        lines.append(f"title: {self.title}")
        lines.append(f"status: {self.status}")
        lines.append(f"confidence: {self.confidence}")
        lines.append(f"risk: {self.risk}")
        lines.append(f"requires_confirmation: {str(self.confirmation).lower()}")
        if self.preconditions:
            lines.append("preconditions:")
            for p in self.preconditions:
                lines.append(f"  - {p}")
        if self.procedure:
            lines.append("procedure:")
            for step in self.procedure:
                lines.append(f"  - order: {step.order}")
                lines.append(f"    action: {step.action}")
                lines.append(f"    target: {step.target}")
                if step.source_instruction:
                    lines.append(f"    source_instruction: {step.source_instruction}")
                if step.value_source:
                    lines.append(f"    value_source: {step.value_source}")
        if self.success_criteria:
            lines.append("success_criteria:")
            for sc in self.success_criteria:
                lines.append(f"  - {sc}")
        if self.recovery:
            lines.append("recovery:")
            for r in self.recovery:
                lines.append(f"  - {r}")
        if self.source:
            lines.append("source:")
            lines.append(f"  manual_id: {self.source.manual_id}")
            lines.append(f"  procedure_id: {self.source.procedure_id}")
            lines.append(f"  file: {self.source.file}")
        if self.validation:
            lines.append("validation:")
            lines.append(f"  has_steps: {str(self.validation.has_steps).lower()}")
            lines.append(f"  has_risk: {str(self.validation.has_risk).lower()}")
            lines.append(f"  has_recovery: {str(self.validation.has_recovery).lower()}")
            lines.append(f"  source_verified: {str(self.validation.source_verified).lower()}")
        return lines

    @property
    def confirmation(self) -> bool:
        """Alias for requires_confirmation (used by to_yaml_lines)."""
        return self.requires_confirmation
