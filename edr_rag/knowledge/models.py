"""Workflow compiler knowledge layer: versioned unified model.

design.md §4 envelope contract for every entity:
  schema_version, stable id, revision, content_hash, created_at (tz-aware
  ISO 8601), sources, applicable product_version. Plain stdlib dataclasses +
  JSON, matching edr-rag conventions.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

SCHEMA_VERSION = "0.1"
EVIDENCE_PURPOSES = ("operation", "expected", "parameter")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class ModelError(ValueError):
    """Raised when an entity violates the knowledge-layer contract."""


def content_hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _check_tz_iso8601(value: str, what: str) -> datetime:
    try:
        dt = datetime.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ModelError(f"{what}: not ISO 8601 ({value!r})") from exc
    if dt.tzinfo is None:
        raise ModelError(f"{what}: timestamp must carry a timezone offset ({value!r})")
    return dt


@dataclass
class SourceRevision:
    """One revision of an ingested source document (design.md §4)."""

    source_id: str
    format: str
    uri: str
    source_hash: str
    source_updated_at: str
    collected_at: str
    revision: int = 1
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _check_tz_iso8601(self.source_updated_at, "source_updated_at")
        _check_tz_iso8601(self.collected_at, "collected_at")
        if not self.source_id or not _ID_RE.match(self.source_id):
            raise ModelError(f"source_id invalid: {self.source_id!r}")
        if not self.source_hash:
            raise ModelError("source_hash required")

    @property
    def ref(self) -> str:
        """Stable reference used by EvidenceRef and idempotency keys."""
        return f"{self.source_id}@rev{self.revision}:{self.source_hash[:16]}"

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "SourceRevision":
        allowed = {"source_id", "format", "uri", "source_hash", "source_updated_at", "collected_at", "revision", "schema_version"}
        unknown = set(d) - allowed
        if unknown:
            raise ModelError(f"SourceRevision unknown fields: {sorted(unknown)}")
        missing = {"source_id", "format", "uri", "source_hash", "source_updated_at", "collected_at"} - set(d)
        if missing:
            raise ModelError(f"SourceRevision missing fields: {sorted(missing)}")
        return SourceRevision(**d)


@dataclass
class EvidenceRef:
    """Pointer into a SourceRevision with a quote hash and purpose (§4)."""

    evidence_id: str
    source_ref: str
    locator: str
    quote: str
    purpose: str

    def __post_init__(self) -> None:
        if not _ID_RE.match(self.evidence_id):
            raise ModelError(f"evidence_id invalid: {self.evidence_id!r}")
        if not self.locator:
            raise ModelError(f"{self.evidence_id}: locator required")
        if not self.quote:
            raise ModelError(f"{self.evidence_id}: quote required")
        if self.purpose not in EVIDENCE_PURPOSES:
            raise ModelError(f"{self.evidence_id}: purpose must be one of {EVIDENCE_PURPOSES}, got {self.purpose!r}")

    @property
    def quote_hash(self) -> str:
        return content_hash({"quote": self.quote})

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "EvidenceRef":
        allowed = {"evidence_id", "source_ref", "locator", "quote", "purpose"}
        unknown = set(d) - allowed
        if unknown:
            raise ModelError(f"EvidenceRef unknown fields: {sorted(unknown)}")
        missing = allowed - set(d)
        if missing:
            raise ModelError(f"EvidenceRef missing fields: {sorted(missing)}")
        return EvidenceRef(**d)


@dataclass
class TestCaseRevision:
    """Platform test case original text, verbatim (design.md §4)."""

    case_id: str
    product_version: str
    platform: str
    title: str
    preconditions: list[str] = field(default_factory=list)
    steps: list[dict[str, str]] = field(default_factory=list)  # {"action":…, "expected":…}
    attachments: list[str] = field(default_factory=list)
    raw_text_hash: str = ""
    source_ref: str = ""
    revision: int = 1
    schema_version: str = SCHEMA_VERSION
    created_at: str = ""

    def __post_init__(self) -> None:
        if not _ID_RE.match(self.case_id):
            raise ModelError(f"case_id invalid: {self.case_id!r}")
        if not self.product_version:
            raise ModelError(f"{self.case_id}: product_version required (cross-version silent merge is forbidden)")
        if not self.title:
            raise ModelError(f"{self.case_id}: title required")
        if not self.created_at:
            raise ModelError(f"{self.case_id}: created_at required")
        _check_tz_iso8601(self.created_at, f"{self.case_id}.created_at")
        if not self.raw_text_hash:
            raise ModelError(f"{self.case_id}: raw_text_hash required")
        if not self.source_ref:
            raise ModelError(f"{self.case_id}: source_ref required")
        for i, step in enumerate(self.steps, 1):
            if "action" not in step:
                raise ModelError(f"{self.case_id} step {i}: missing 'action'")
            if "expected" not in step:
                raise ModelError(f"{self.case_id} step {i}: missing 'expected'")
            unknown = set(step) - {"action", "expected", "note"}
            if unknown:
                raise ModelError(f"{self.case_id} step {i}: unknown step fields {sorted(unknown)}")

    @property
    def idempotency_key(self) -> str:
        """来源身份+源版本/hash (design.md §4)."""
        return f"{self.case_id}:{self.source_ref}"
    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "TestCaseRevision":
        allowed = {
            "case_id", "product_version", "platform", "title", "preconditions",
            "steps", "attachments", "raw_text_hash", "source_ref", "revision",
            "schema_version", "created_at",
        }
        unknown = set(d) - allowed
        if unknown:
            raise ModelError(f"TestCaseRevision unknown fields: {sorted(unknown)}")
        missing = {"case_id", "product_version", "platform", "title", "raw_text_hash", "source_ref", "created_at"} - set(d)
        if missing:
            raise ModelError(f"TestCaseRevision missing fields: {sorted(missing)}")
        return TestCaseRevision(**d)


@dataclass
class CaseBundle:
    """Validated import unit: one case revision plus its evidence set."""

    case: TestCaseRevision
    evidence: list[EvidenceRef] = field(default_factory=list)

    def validate_references(self) -> None:
        """Dangling or mis-purposed evidence refs must be rejected (design.md §4, P1 验收)."""
        if not self.evidence:
            raise ModelError(f"{self.case.case_id}: at least one evidence ref required")
        seen: set[str] = set()
        for ev in self.evidence:
            if ev.source_ref != self.case.source_ref:
                raise ModelError(
                    f"{ev.evidence_id}: evidence source_ref {ev.source_ref!r} != case source_ref {self.case.source_ref!r}"
                )
            if ev.evidence_id in seen:
                raise ModelError(f"duplicate evidence_id: {ev.evidence_id}")
            seen.add(ev.evidence_id)
        covered = {ev.purpose for ev in self.evidence}
        if "operation" not in covered:
            raise ModelError(f"{self.case.case_id}: no evidence with purpose 'operation'")
        if "expected" not in covered:
            raise ModelError(f"{self.case.case_id}: no evidence with purpose 'expected'")


@dataclass
class DocumentSectionRevision:
    """知识层对既有 P0 ManualSection 的 revision 适配 (design.md §4)。

    引用而非复制：section_id 指向既有 ManualSection.id；section_content_hash
    绑定该 section 在 source_revision 下的内容 hash，源变更新建 SourceRevision
    后必须重新核对，不得静默沿用旧 hash。document_version 与 case 的
    product_version 走同一显式过滤语义，跨版本不混检。
    """

    section_id: str
    document_version: str
    source_ref: str
    section_content_hash: str
    toc_path: list[str] = field(default_factory=list)
    page_or_chm_path: str = ""
    revision: int = 1
    schema_version: str = SCHEMA_VERSION
    created_at: str = ""

    def __post_init__(self) -> None:
        if not _ID_RE.match(self.section_id):
            raise ModelError(f"section_id invalid: {self.section_id!r}")
        if not self.document_version:
            raise ModelError(f"{self.section_id}: document_version required")
        if not self.created_at:
            raise ModelError(f"{self.section_id}: created_at required")
        _check_tz_iso8601(self.created_at, f"{self.section_id}.created_at")
        if not self.section_content_hash:
            raise ModelError(f"{self.section_id}: section_content_hash required (source binding)")

    @property
    def idempotency_key(self) -> str:
        return f"{self.section_id}:{self.source_ref}"

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "DocumentSectionRevision":
        allowed = {
            "section_id", "document_version", "source_ref", "section_content_hash",
            "toc_path", "page_or_chm_path", "revision", "schema_version", "created_at",
        }
        unknown = set(d) - allowed
        if unknown:
            raise ModelError(f"DocumentSectionRevision unknown fields: {sorted(unknown)}")
        missing = {"section_id", "document_version", "source_ref", "section_content_hash", "created_at"} - set(d)
        if missing:
            raise ModelError(f"DocumentSectionRevision missing fields: {sorted(missing)}")
        return DocumentSectionRevision(**d)


TASK_TARGET_KINDS = ("feature_operation", "configuration", "verification")
TASK_STATUS = ("needs_review", "validated")

# design.md §7: 动作枚举是领域语义；是否可执行由适配器在 P4 确定。
ATOMIC_ACTIONS = (
    "connect", "navigate", "click", "input", "select", "check",
    "clear", "scroll", "wait", "assert", "screenshot",
)
LOCATOR_STATUSES = ("runtime_resolution", "fixed", "needs_resolution")
VERIFICATION_KINDS = ("value_check", "state_check")
# §6.3: 必要导航可标 inference；业务动作不允许推断新增。
NAV_ATOMIC_ACTIONS = ("navigate", "connect")


@dataclass
class ParamValue:
    """Typed parameter binding: exactly one of {"variable": name} or {"literal": value}.

    任意裸字符串不得作为参数（§7 不能把任意字符串替换成命令）；literal 仅接受
    JSON 原生标量，结构化载荷必须经变量声明。
    """

    variable: str = ""
    literal: Any = None
    _sentinel: Any = None

    def __post_init__(self) -> None:
        has_var = bool(self.variable)
        has_lit = self._sentinel is not None
        if has_var == has_lit:
            raise ModelError("ParamValue: exactly one of 'variable' or 'literal' required")
        if has_var and not _ID_RE.match(self.variable):
            raise ModelError(f"ParamValue variable name invalid: {self.variable!r}")
        if has_lit and isinstance(self.literal, (dict, list)):
            raise ModelError("ParamValue literal must be a JSON scalar (结构化载荷必须走变量)")

    @staticmethod
    def from_dict(d: Any) -> "ParamValue":
        if not isinstance(d, dict):
            raise ModelError(f"ParamValue must be an object, got {type(d).__name__}")
        unknown = set(d) - {"variable", "literal"}
        if unknown:
            raise ModelError(f"ParamValue unknown fields: {sorted(unknown)}")
        if "variable" in d:
            return ParamValue(variable=d["variable"])
        if "literal" in d:
            return ParamValue(_sentinel=True, literal=d["literal"])
        raise ModelError("ParamValue: exactly one of 'variable' or 'literal' required")

    def to_dict(self) -> dict[str, Any]:
        if self.variable:
            return {"variable": self.variable}
        return {"literal": self.literal}


@dataclass
class StepVerification:
    kind: str
    expected_variable: str
    evidence_ref: str

    def __post_init__(self) -> None:
        if self.kind not in VERIFICATION_KINDS:
            raise ModelError(f"verification.kind must be one of {VERIFICATION_KINDS}, got {self.kind!r}")
        if not self.expected_variable:
            raise ModelError("verification.expected_variable required")
        if not self.evidence_ref:
            raise ModelError("verification.evidence_ref required (断言必须有证据)")


@dataclass
class StepBinding:
    """Capability binding produced by the Mapping stage (design.md §6.4 / §8).

    绑定必须携带工具版本与工具 schema hash、能力快照 hash（执行前复核 hash，
    变化则拒绝旧绑定）。locator 仅允许「真实观测 + 环境版本」齐备的固化定位
    器；映射阶段无观测证据时恒为 None（runtime_resolution）。
    """

    tool: str
    tool_version: str
    tool_schema_hash: str
    snapshot_hash: str
    bound_at: str
    locator: dict[str, str] | None = None  # {"kind","value","observed_at","environment_version"}

    def __post_init__(self) -> None:
        if not self.tool:
            raise ModelError("binding.tool required")
        if not self.tool_version:
            raise ModelError("binding.tool_version required")
        if not self.tool_schema_hash:
            raise ModelError("binding.tool_schema_hash required (工具 schema hash 绑定)")
        if not self.snapshot_hash:
            raise ModelError("binding.snapshot_hash required (能力快照 hash 绑定)")
        if not self.bound_at:
            raise ModelError("binding.bound_at required")
        _check_tz_iso8601(self.bound_at, "binding.bound_at")
        if self.locator is not None:
            unknown = set(self.locator) - {"kind", "value", "observed_at", "environment_version"}
            if unknown:
                raise ModelError(f"binding.locator unknown fields {sorted(unknown)}")
            for key in ("kind", "value", "observed_at", "environment_version"):
                if not self.locator.get(key):
                    raise ModelError(f"binding.locator.{key} required (固化定位器必须真实观测+环境版本)")
            _check_tz_iso8601(self.locator["observed_at"], "binding.locator.observed_at")

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool, "tool_version": self.tool_version,
            "tool_schema_hash": self.tool_schema_hash, "snapshot_hash": self.snapshot_hash,
            "bound_at": self.bound_at,
            "locator": dict(self.locator) if self.locator else None,
        }

    @staticmethod
    def from_dict(d: Any) -> "StepBinding":
        if not isinstance(d, dict):
            raise ModelError(f"binding must be an object, got {type(d).__name__}")
        unknown = set(d) - {"tool", "tool_version", "tool_schema_hash", "snapshot_hash", "bound_at", "locator"}
        if unknown:
            raise ModelError(f"binding unknown fields: {sorted(unknown)}")
        missing = {"tool", "tool_version", "tool_schema_hash", "snapshot_hash", "bound_at"} - set(d)
        if missing:
            raise ModelError(f"binding missing fields: {sorted(missing)}")
        return StepBinding(
            tool=d["tool"], tool_version=d["tool_version"], tool_schema_hash=d["tool_schema_hash"],
            snapshot_hash=d["snapshot_hash"], bound_at=d["bound_at"],
            locator=dict(d["locator"]) if d.get("locator") else None,
        )


@dataclass
class AtomicIR:
    """One DAG node of the two-level IR (design.md §4 / §7 sketch).

    binding 在 P1 恒为 None；固化 locator 只有在真实观测且带环境版本时才允许
    （§6.4），P1 schema 仅锁定枚举与结构。
    """

    step_id: str
    source_step: int
    action: str
    target: dict[str, str]  # {"semantic_name":…, "locator_status":…}
    parameters: dict[str, ParamValue] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    verification: StepVerification | None = None
    risk: str = ""
    inference: dict[str, str] | None = None  # §6.3: {"basis": …}，仅导航类允许
    binding: "StepBinding | None" = None  # P3 Mapping 产物；P1/P2 恒为 None
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not _ID_RE.match(self.step_id):
            raise ModelError(f"step_id invalid: {self.step_id!r}")
        if not isinstance(self.source_step, int) or self.source_step < 1:
            raise ModelError(f"{self.step_id}: source_step must be a positive int (source case step)")
        if self.action not in ATOMIC_ACTIONS:
            raise ModelError(f"{self.step_id}: action {self.action!r} not in ATOMIC_ACTIONS")
        if self.inference is not None:
            if self.action not in NAV_ATOMIC_ACTIONS:
                raise ModelError(f"{self.step_id}: inference annotation only allowed for navigation actions {NAV_ATOMIC_ACTIONS} (业务动作不可推断新增)")
            unknown_inf = set(self.inference) - {"basis", "rationale"}
            if unknown_inf:
                raise ModelError(f"{self.step_id}: inference unknown fields {sorted(unknown_inf)}")
            if not self.inference.get("basis"):
                raise ModelError(f"{self.step_id}: inference.basis required (推断必须附依据)")
        unknown_target = set(self.target) - {"semantic_name", "locator_status"}
        if unknown_target:
            raise ModelError(f"{self.step_id}: target unknown fields {sorted(unknown_target)}")
        if not self.target.get("semantic_name"):
            raise ModelError(f"{self.step_id}: target.semantic_name required")
        if self.target.get("locator_status") not in LOCATOR_STATUSES:
            raise ModelError(f"{self.step_id}: locator_status must be one of {LOCATOR_STATUSES}")
        if self.binding is not None:
            if not isinstance(self.binding, StepBinding):
                raise ModelError(f"{self.step_id}: binding must be a StepBinding or null (裸 dict 不可信)")
            # §6.4: locator_status=fixed 只有真实观测+环境版本的 locator 才成立
            if self.target.get("locator_status") == "fixed" and self.binding.locator is None:
                raise ModelError(f"{self.step_id}: locator_status=fixed requires binding.locator (无真实观测不得固化)")
        if not self.evidence_refs:
            raise ModelError(f"{self.step_id}: evidence_refs required (来源完整)")

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["target"] = dict(self.target)
        d["parameters"] = {k: v.to_dict() for k, v in self.parameters.items()}
        d["verification"] = self.verification.__dict__ if self.verification else None
        d["inference"] = dict(self.inference) if self.inference else None
        d["binding"] = self.binding.to_dict() if self.binding else None
        return d

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "AtomicIR":
        allowed = {
            "step_id", "source_step", "action", "target", "parameters",
            "depends_on", "evidence_refs", "verification", "risk", "inference",
            "binding", "schema_version",
        }
        unknown = set(d) - allowed
        if unknown:
            raise ModelError(f"AtomicIR unknown fields: {sorted(unknown)}")
        missing = {"step_id", "source_step", "action", "target", "evidence_refs"} - set(d)
        if missing:
            raise ModelError(f"AtomicIR missing fields: {sorted(missing)}")
        params = {k: ParamValue.from_dict(v) for k, v in d.get("parameters", {}).items()}
        verif = StepVerification(**d["verification"]) if d.get("verification") else None
        inference = dict(d["inference"]) if d.get("inference") else None
        binding = StepBinding.from_dict(d["binding"]) if d.get("binding") else None
        s = AtomicIR(
            step_id=d["step_id"], source_step=d["source_step"], action=d["action"],
            target=dict(d["target"]), parameters=params, depends_on=list(d.get("depends_on", [])),
            evidence_refs=list(d["evidence_refs"]), verification=verif,
            risk=d.get("risk", ""), inference=inference, binding=binding,
        )
        expected = {**d, "parameters": {k: v.to_dict() for k, v in params.items()}}
        if s.to_dict() != expected:
            raise ModelError(f"{s.step_id}: roundtrip mismatch")
        return s


def validate_step_graph(steps: list[AtomicIR]) -> list[dict[str, str]]:
    """Deterministic DAG validation (design.md §6.5): field-path diagnostics.

    依赖无环、引用存在、step_id 唯一。返回诊断列表；空列表即通过。
    """
    diags: list[dict[str, str]] = []
    ids = [s.step_id for s in steps]
    dupes = {i for i in ids if ids.count(i) > 1}
    for dup in sorted(dupes):
        diags.append({"path": f"steps[{dup}]", "code": "duplicate_step_id", "message": f"step_id {dup!r} appears more than once"})
    id_set = set(ids)
    for s in steps:
        if s.step_id in dupes:
            continue
        for dep in s.depends_on:
            if dep not in id_set:
                diags.append({"path": f"steps[{s.step_id}].depends_on", "code": "dangling_dependency", "message": f"dependency {dep!r} is not a step in this workflow"})
            elif dep == s.step_id:
                diags.append({"path": f"steps[{s.step_id}].depends_on", "code": "self_dependency", "message": "step depends on itself"})
    if diags:
        return diags
    # Kahn: 依赖图无环
    indegree = {s.step_id: 0 for s in steps}
    dependents: dict[str, list[str]] = {s.step_id: [] for s in steps}
    for s in steps:
        for dep in s.depends_on:
            indegree[s.step_id] += 1
            dependents[dep].append(s.step_id)
    ready = [i for i, deg in indegree.items() if deg == 0]
    seen = 0
    while ready:
        node = ready.pop()
        seen += 1
        for nxt in dependents[node]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                ready.append(nxt)
    if seen != len(steps):
        cyc = sorted(i for i, deg in indegree.items() if deg > 0)
        diags.append({"path": "steps", "code": "dependency_cycle", "message": f"dependency cycle among {cyc}"})
    return diags


WORKFLOW_STATUS = ("draft", "needs_review", "validated", "published")


@dataclass
class ToolCapability:
    """One tool/action entry inside a capability snapshot (design.md §4 / §8).

    从实际 Skill/工具 schema 建：工具名、版本、schema hash（严格核对绑定），
    支持的动作枚举与平台。参数 schema 用 JSON-schema 形状的对象快照。
    """

    tool: str
    tool_version: str
    tool_schema_hash: str
    platform: str
    actions: tuple[str, ...]
    parameter_schema: dict[str, Any] = field(default_factory=dict)
    observed_at: str = ""

    def __post_init__(self) -> None:
        if not self.tool:
            raise ModelError("capability.tool required")
        if not self.tool_version:
            raise ModelError(f"{self.tool}: tool_version required")
        if not self.tool_schema_hash:
            raise ModelError(f"{self.tool}: tool_schema_hash required")
        if not self.platform:
            raise ModelError(f"{self.tool}: platform required")
        if not self.actions:
            raise ModelError(f"{self.tool}: actions required (空能力不得注册)")
        unknown = [a for a in self.actions if a not in ATOMIC_ACTIONS]
        if unknown:
            raise ModelError(f"{self.tool}: unknown actions {unknown} not in ATOMIC_ACTIONS")
        if not self.observed_at:
            raise ModelError(f"{self.tool}: observed_at required (快照记录观察时间)")
        _check_tz_iso8601(self.observed_at, f"{self.tool}.observed_at")

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool, "tool_version": self.tool_version,
            "tool_schema_hash": self.tool_schema_hash, "platform": self.platform,
            "actions": list(self.actions), "parameter_schema": dict(self.parameter_schema),
            "observed_at": self.observed_at,
        }

    @staticmethod
    def from_dict(d: Any) -> "ToolCapability":
        if not isinstance(d, dict):
            raise ModelError(f"capability must be an object, got {type(d).__name__}")
        unknown = set(d) - {"tool", "tool_version", "tool_schema_hash", "platform", "actions", "parameter_schema", "observed_at"}
        if unknown:
            raise ModelError(f"capability unknown fields: {sorted(unknown)}")
        missing = {"tool", "tool_version", "tool_schema_hash", "platform", "actions", "observed_at"} - set(d)
        if missing:
            raise ModelError(f"capability missing fields: {sorted(missing)}")
        return ToolCapability(
            tool=d["tool"], tool_version=d["tool_version"], tool_schema_hash=d["tool_schema_hash"],
            platform=d["platform"], actions=tuple(d["actions"]),
            parameter_schema=dict(d.get("parameter_schema", {})), observed_at=d["observed_at"],
        )


@dataclass
class CapabilitySnapshot:
    """Frozen registry snapshot (design.md §4 / §8).

    从实际 SKILL.md/工具 schema 建快照，记录观察时间与支持证据；content_hash
    是绑定锚 —— 执行前复核 hash，变化则拒绝旧绑定（§8）。快照不可变：修订
    建新 snapshot，不原地改。
    """

    snapshot_id: str
    platform: str
    product_version: str
    capabilities: list[ToolCapability]
    observed_at: str
    notes: str = ""
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not _ID_RE.match(self.snapshot_id):
            raise ModelError(f"snapshot_id invalid: {self.snapshot_id!r}")
        if not self.platform:
            raise ModelError(f"{self.snapshot_id}: platform required")
        if not self.product_version:
            raise ModelError(f"{self.snapshot_id}: product_version required")
        if not self.observed_at:
            raise ModelError(f"{self.snapshot_id}: observed_at required")
        _check_tz_iso8601(self.observed_at, f"{self.snapshot_id}.observed_at")
        tools = [c.tool for c in self.capabilities]
        dupes = sorted({t for t in tools if tools.count(t) > 1})
        if dupes:
            raise ModelError(f"{self.snapshot_id}: duplicate tool entries {dupes}")

    @property
    def content_hash(self) -> str:
        return content_hash({
            "snapshot_id": self.snapshot_id, "platform": self.platform,
            "product_version": self.product_version,
            "capabilities": [c.to_dict() for c in self.capabilities],
            "observed_at": self.observed_at, "schema_version": self.schema_version,
        })

    def capability_for(self, action: str, platform: str) -> ToolCapability | None:
        """确定性映射：平台+动作 → 唯一工具；未声明能力返回 None（unsupported）。"""
        for c in self.capabilities:
            if c.platform == platform and action in c.actions:
                return c
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id, "platform": self.platform,
            "product_version": self.product_version,
            "capabilities": [c.to_dict() for c in self.capabilities],
            "observed_at": self.observed_at, "notes": self.notes,
            "schema_version": self.schema_version,
        }

    @staticmethod
    def from_dict(d: Any) -> "CapabilitySnapshot":
        if not isinstance(d, dict):
            raise ModelError(f"snapshot must be an object, got {type(d).__name__}")
        unknown = set(d) - {"snapshot_id", "platform", "product_version", "capabilities", "observed_at", "notes", "schema_version"}
        if unknown:
            raise ModelError(f"snapshot unknown fields: {sorted(unknown)}")
        missing = {"snapshot_id", "platform", "product_version", "capabilities", "observed_at"} - set(d)
        if missing:
            raise ModelError(f"snapshot missing fields: {sorted(missing)}")
        return CapabilitySnapshot(
            snapshot_id=d["snapshot_id"], platform=d["platform"],
            product_version=d["product_version"],
            capabilities=[ToolCapability.from_dict(c) for c in d["capabilities"]],
            observed_at=d["observed_at"], notes=d.get("notes", ""),
        )


@dataclass
class TaskVariable:
    name: str
    var_type: str  # string | number | boolean
    required: bool = True
    description: str = ""

    _TYPES = ("string", "number", "boolean")

    def __post_init__(self) -> None:
        if not _ID_RE.match(self.name):
            raise ModelError(f"variable name invalid: {self.name!r}")
        if self.var_type not in self._TYPES:
            raise ModelError(f"{self.name}: var_type must be one of {self._TYPES}, got {self.var_type!r}")

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class TaskIR:
    """阶段2 Case Understanding 的输出契约 (design.md §6.2 / §4)。

    从 TestCaseRevision 提取：目标、前置、变量、来源用例引用与证据。不含
    点击协议，不含业务默认值；证据必须绑定同一用例 source_ref。歧义或缺口
    记入 ambiguities，同时 status=needs_review；没有歧义才允许 validated。
    """

    task_id: str
    product_version: str
    platform: str
    goal: str
    source_case_id: str
    source_case_ref: str
    source_evidence_ids: list[str] = field(default_factory=list)
    preconditions: list[str] = field(default_factory=list)
    variables: list[TaskVariable] = field(default_factory=list)
    business_result: str = ""
    cleanup: list[str] = field(default_factory=list)
    ambiguities: list[str] = field(default_factory=list)
    status: str = "needs_review"
    revision: int = 1
    schema_version: str = SCHEMA_VERSION
    created_at: str = ""

    def __post_init__(self) -> None:
        if not _ID_RE.match(self.task_id):
            raise ModelError(f"task_id invalid: {self.task_id!r}")
        if not self.product_version:
            raise ModelError(f"{self.task_id}: product_version required")
        if not self.goal:
            raise ModelError(f"{self.task_id}: goal required")
        if not self.source_case_id or not self.source_case_ref:
            raise ModelError(f"{self.task_id}: source_case_id/source_case_ref required (缺来源即拒绝)")
        if not self.source_evidence_ids:
            raise ModelError(f"{self.task_id}: source_evidence_ids required (断言/操作必须有证据)")
        if self.status not in TASK_STATUS:
            raise ModelError(f"{self.task_id}: status must be one of {TASK_STATUS}, got {self.status!r}")
        if not self.created_at:
            raise ModelError(f"{self.task_id}: created_at required")
        _check_tz_iso8601(self.created_at, f"{self.task_id}.created_at")
        if self.ambiguities and self.status == "validated":
            raise ModelError(f"{self.task_id}: ambiguities present but status=validated (歧义必须 needs_review)")

    @property
    def idempotency_key(self) -> str:
        return f"{self.task_id}:{self.source_case_ref}"

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["variables"] = [v.to_dict() for v in self.variables]
        return d

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "TaskIR":
        allowed = {
            "task_id", "product_version", "platform", "goal", "source_case_id",
            "source_case_ref", "source_evidence_ids", "preconditions", "variables",
            "business_result", "cleanup", "ambiguities", "status", "revision",
            "schema_version", "created_at",
        }
        unknown = set(d) - allowed
        if unknown:
            raise ModelError(f"TaskIR unknown fields: {sorted(unknown)}")
        missing = {
            "task_id", "product_version", "platform", "goal", "source_case_id",
            "source_case_ref", "source_evidence_ids", "created_at",
        } - set(d)
        if missing:
            raise ModelError(f"TaskIR missing fields: {sorted(missing)}")
        variables = [TaskVariable(**v) for v in d.get("variables", [])]
        rest = {k: v for k, v in d.items() if k != "variables"}
        t = TaskIR(**rest)
        t.variables = variables
        return t


@dataclass
class WorkflowRevision:
    """两级 IR 的冻结单元 (design.md §4 / §8)。

    validated 仅指静态契约；published 必须有显式审核记录并只能批准当前
    content_hash，编译器不自授发布（§8 状态边界）。execution_status 独立于
    静态状态。steps 的 DAG 一致性由 validate_steps() 显式校验，构造不隐藏失败。
    """

    workflow_id: str
    product_version: str
    platform: str
    task_id: str
    task_source_ref: str
    steps: list[AtomicIR] = field(default_factory=list)
    declared_variables: list[TaskVariable] = field(default_factory=list)
    capability_snapshot_ref: str = ""  # P4 绑定真实能力快照 hash；P1 允许空
    status: str = "draft"
    execution_status: str = "not_run"
    revision: int = 1
    schema_version: str = SCHEMA_VERSION
    created_at: str = ""

    def __post_init__(self) -> None:
        if not _ID_RE.match(self.workflow_id):
            raise ModelError(f"workflow_id invalid: {self.workflow_id!r}")
        if not self.product_version:
            raise ModelError(f"{self.workflow_id}: product_version required")
        if not self.task_id or not self.task_source_ref:
            raise ModelError(f"{self.workflow_id}: task_id/task_source_ref required (缺来源即拒绝)")
        if self.status not in WORKFLOW_STATUS:
            raise ModelError(f"{self.workflow_id}: status must be one of {WORKFLOW_STATUS}, got {self.status!r}")
        if self.execution_status not in ("not_run", "running", "passed", "failed", "unknown"):
            raise ModelError(f"{self.workflow_id}: invalid execution_status {self.execution_status!r}")
        if self.status == "published":
            raise ModelError(f"{self.workflow_id}: published requires explicit review record (编译器不自授发布)")
        if not self.created_at:
            raise ModelError(f"{self.workflow_id}: created_at required")
        _check_tz_iso8601(self.created_at, f"{self.workflow_id}.created_at")

    def validate_steps(self) -> list[dict[str, str]]:
        """Full structural validation with field-path diagnostics (§6.5)."""
        diags = validate_step_graph(self.steps)
        id_set = {s.step_id for s in self.steps}
        for s in self.steps:
            for er in s.evidence_refs:
                if er not in id_set and not er.startswith("E-"):
                    diags.append({"path": f"steps[{s.step_id}].evidence_refs", "code": "invalid_evidence_ref", "message": f"evidence ref {er!r} must reference an E-* evidence id"})
            if s.verification is not None:
                names = {v.name for v in self.declared_variables}
                if s.verification.expected_variable not in names:
                    diags.append({"path": f"steps[{s.step_id}].verification", "code": "unknown_variable", "message": f"expected_variable {s.verification.expected_variable!r} not declared"})
                if s.verification.evidence_ref not in s.evidence_refs:
                    diags.append({"path": f"steps[{s.step_id}].verification", "code": "dangling_evidence_ref", "message": f"verification evidence {s.verification.evidence_ref!r} not in step evidence_refs"})
            for pv in s.parameters.values():
                if pv.variable and pv.variable not in {v.name for v in self.declared_variables}:
                    diags.append({"path": f"steps[{s.step_id}].parameters", "code": "unknown_variable", "message": f"variable {pv.variable!r} not declared in declared_variables"})
        return diags

    @property
    def content_hash(self) -> str:
        return content_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["steps"] = [s.to_dict() for s in self.steps]
        d["declared_variables"] = [v.to_dict() for v in self.declared_variables]
        return d

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "WorkflowRevision":
        allowed = {
            "workflow_id", "product_version", "platform", "task_id", "task_source_ref",
            "steps", "declared_variables", "capability_snapshot_ref", "status",
            "execution_status", "revision", "schema_version", "created_at",
        }
        unknown = set(d) - allowed
        if unknown:
            raise ModelError(f"WorkflowRevision unknown fields: {sorted(unknown)}")
        missing = {
            "workflow_id", "product_version", "platform", "task_id", "task_source_ref",
            "created_at",
        } - set(d)
        if missing:
            raise ModelError(f"WorkflowRevision missing fields: {sorted(missing)}")
        steps = [AtomicIR.from_dict(s) for s in d.get("steps", [])]
        variables = [TaskVariable(**v) for v in d.get("declared_variables", [])]
        rest = {k: v for k, v in d.items() if k not in ("steps", "declared_variables")}
        w = WorkflowRevision(**rest)
        w.steps = steps
        w.declared_variables = variables
        return w
