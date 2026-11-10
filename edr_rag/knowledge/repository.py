"""Local-file repository for knowledge-layer entities (design.md §4 MVP).

Uses edr_rag.atomic_io for temp-file+rename writes. Idempotency:
import keyed on source identity + hash; duplicates return status="duplicate".
A newer revision of the same source marks the case stale (rebuild required);
stale does not silently inherit prior validation (§8).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from edr_rag.atomic_io import atomic_write_text
from edr_rag.knowledge.models import (
    CapabilitySnapshot,
    CaseBundle,
    DocumentSectionRevision,
    ModelError,
    SourceRevision,
    TaskIR,
    TestCaseRevision,
    TaskVariable,
    WorkflowRevision,
)

CASES_FILE = "cases.jsonl"
SOURCES_FILE = "sources.jsonl"
STALE_FILE = "stale.jsonl"
STALE_MARKS_FILE = "stale_marks.jsonl"
SECTIONS_FILE = "sections.jsonl"
TASKS_FILE = "tasks.jsonl"
WORKFLOWS_FILE = "workflows.jsonl"
SNAPSHOTS_FILE = "snapshots.jsonl"
REVIEWS_FILE = "reviews.jsonl"


class ImportResult:
    def __init__(self, status: str, case_id: str, revision: int, path: Path | None = None):
        self.status = status  # created | duplicate | superseded
        self.case_id = case_id
        self.revision = revision
        self.path = path

    def to_dict(self) -> dict[str, Any]:
        d = {"status": self.status, "case_id": self.case_id, "revision": self.revision}
        if self.path is not None:
            d["path"] = str(self.path)
        return d


class KnowledgeRepository:
    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ---------- persistence helpers ----------

    def _read_jsonl(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        out: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                out.append(json.loads(line))
        return out

    def _append_jsonl(self, path: Path, record: dict[str, Any]) -> None:
        existing = self._read_jsonl(path)
        existing.append(record)
        text = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in existing)
        atomic_write_text(path, text)

    # ---------- sources ----------

    def save_source(self, source: SourceRevision) -> None:
        self._append_jsonl(self.root / SOURCES_FILE, source.to_dict())

    def latest_source(self, source_id: str) -> SourceRevision | None:
        found = [s for s in self._read_jsonl(self.root / SOURCES_FILE) if s["source_id"] == source_id]
        if not found:
            return None
        return SourceRevision.from_dict(max(found, key=lambda s: s["revision"]))

    def list_sources(self, source_id: str | None = None) -> list[SourceRevision]:
        found = [SourceRevision.from_dict(d) for d in self._read_jsonl(self.root / SOURCES_FILE)]
        if source_id is not None:
            found = [s for s in found if s.source_id == source_id]
        return found

    # ---------- cases ----------

    def _load_cases(self) -> list[TestCaseRevision]:
        # records persist evidence alongside the case; strip it before model reparse
        return [
            TestCaseRevision.from_dict({k: v for k, v in d.items() if k != "evidence"})
            for d in self._read_jsonl(self.root / CASES_FILE)
        ]

    def import_case(self, bundle: CaseBundle) -> ImportResult:
        bundle.validate_references()
        existing = self._load_cases()
        key = bundle.case.idempotency_key

        prior_by_key = [c for c in existing if c.idempotency_key == key]
        if prior_by_key:
            return ImportResult("duplicate", bundle.case.case_id, prior_by_key[0].revision)

        priors = [c for c in existing if c.case_id == bundle.case.case_id]
        new_revision = (max(c.revision for c in priors) + 1) if priors else 1

        stale_ids: list[str] = []
        for c in priors:
            stale_ids.append(f"{c.case_id}@rev{c.revision}")

        case = bundle.case
        case.revision = new_revision
        record = case.to_dict()
        record["evidence"] = [ev.to_dict() for ev in bundle.evidence]
        self._append_jsonl(self.root / CASES_FILE, record)

        for sid in stale_ids:
            self._append_jsonl(self.root / STALE_FILE, {"stale": sid, "superseded_by": f"{case.case_id}@rev{new_revision}"})

        # downstream invalidation: tasks/workflows derived from the superseded
        # case revisions must not keep prior validation (§4/§8 证据失效不自动
        # 继承审核). Applied only when priors existed.
        if priors:
            from edr_rag.knowledge.staleness import append_case_supersede_downstream
            append_case_supersede_downstream(
                self,
                bundle.case.case_id,
                sorted({c.source_ref for c in priors}),
                self.root / STALE_MARKS_FILE,
            )
        return ImportResult("created" if not priors else "superseded", case.case_id, new_revision, self.root / CASES_FILE)

    def get_case(self, case_id: str, revision: int | None = None) -> TestCaseRevision | None:
        matches = [c for c in self._load_cases() if c.case_id == case_id]
        if revision is not None:
            matches = [c for c in matches if c.revision == revision]
        if not matches:
            return None
        return max(matches, key=lambda c: c.revision)

    def list_cases(self, product_version: str | None = None) -> list[TestCaseRevision]:
        cases = self._load_cases()
        if product_version is not None:
            # 跨版本默认不混检 (design.md §4): explicit filter only.
            cases = [c for c in cases if c.product_version == product_version]
        return cases

    def evidence_of(self, case: TestCaseRevision) -> list[dict[str, Any]]:
        for d in self._read_jsonl(self.root / CASES_FILE):
            if d.get("case_id") == case.case_id and d.get("revision") == case.revision:
                return d.get("evidence", [])
        raise ModelError(f"case not found in store: {case.case_id}@rev{case.revision}")

    # ---------- document sections ----------

    def save_section(self, section: DocumentSectionRevision) -> None:
        self._append_jsonl(self.root / SECTIONS_FILE, section.to_dict())

    def latest_section(self, section_id: str, document_version: str | None = None) -> DocumentSectionRevision | None:
        found = [
            DocumentSectionRevision.from_dict(d)
            for d in self._read_jsonl(self.root / SECTIONS_FILE)
            if d["section_id"] == section_id
        ]
        if document_version is not None:
            found = [s for s in found if s.document_version == document_version]
        if not found:
            return None
        return max(found, key=lambda s: s.revision)

    def list_sections(
        self,
        section_id: str | None = None,
        source_ref: str | None = None,
        document_version: str | None = None,
    ) -> list[DocumentSectionRevision]:
        """Explicit-filter listing only; no cross-version merge (design.md §4)."""
        found = [DocumentSectionRevision.from_dict(d) for d in self._read_jsonl(self.root / SECTIONS_FILE)]
        if section_id is not None:
            found = [s for s in found if s.section_id == section_id]
        if source_ref is not None:
            found = [s for s in found if s.source_ref == source_ref]
        if document_version is not None:
            found = [s for s in found if s.document_version == document_version]
        # latest revision per (section_id, document_version)
        best: dict[tuple[str, str], DocumentSectionRevision] = {}
        for s in found:
            k = (s.section_id, s.document_version)
            if k not in best or s.revision > best[k].revision:
                best[k] = s
        return [best[k] for k in sorted(best)]

    # ---------- task IRs ----------

    def save_task(self, task: TaskIR) -> TaskIR | None:
        """Persist a TaskIR keyed on case source ref. Same case ref → duplicate (None).

        The task's source_case_ref must exist as an imported case revision
        (dangling task sources rejected) and its evidence ids must all be
        present in that case's stored evidence (悬空证据引用拒绝).
        """
        case = self.get_case(task.source_case_id)
        if case is None or case.source_ref != task.source_case_ref:
            raise ModelError(f"{task.task_id}: source case {task.source_case_id!r} not imported with ref {task.source_case_ref!r}")
        stored_ev_ids = {e["evidence_id"] for e in self.evidence_of(case)}
        dangling = set(task.source_evidence_ids) - stored_ev_ids
        if dangling:
            raise ModelError(f"{task.task_id}: dangling evidence ids {sorted(dangling)}")
        existing = self._read_jsonl(self.root / TASKS_FILE)
        key = task.idempotency_key
        if any(TaskIR.from_dict(d).idempotency_key == key for d in existing):
            return None
        self._append_jsonl(self.root / TASKS_FILE, task.to_dict())
        return task

    def get_task(self, task_id: str, product_version: str | None = None) -> TaskIR | None:
        found = [
            TaskIR.from_dict(d)
            for d in self._read_jsonl(self.root / TASKS_FILE)
            if d["task_id"] == task_id
        ]
        if product_version is not None:
            found = [t for t in found if t.product_version == product_version]
        if not found:
            return None
        return max(found, key=lambda t: t.revision)

    def list_tasks(self, product_version: str | None = None) -> list[TaskIR]:
        tasks = [TaskIR.from_dict(d) for d in self._read_jsonl(self.root / TASKS_FILE)]
        if product_version is not None:
            tasks = [t for t in tasks if t.product_version == product_version]
        return tasks

    # ---------- workflow revisions ----------

    def save_workflow(self, workflow: WorkflowRevision) -> WorkflowRevision | None:
        """Persist a WorkflowRevision keyed on task source ref (idempotent).

        Same (workflow_id, task_source_ref) → duplicate (None). Rejected when:
        the source task was not imported with that ref (缺来源拒绝); steps carry
        structural diagnostics (§6.5 禁止静默删步骤，带诊断返回而非修复); step
        parameters reference variables the Task did not declare (变量类型化绑定).
        """
        task = self.get_task(workflow.task_id)
        if task is None or task.source_case_ref != workflow.task_source_ref:
            raise ModelError(
                f"{workflow.workflow_id}: source task {workflow.task_id!r} not imported with ref {workflow.task_source_ref!r}"
            )
        diags = workflow.validate_steps()
        if diags:
            raise ModelError(f"{workflow.workflow_id}: step validation failed: {diags}")
        task_vars = {v.name: v for v in task.variables}
        undeclared = [
            pv.variable
            for s in workflow.steps
            for pv in s.parameters.values()
            if pv.variable and pv.variable not in task_vars
        ]
        if undeclared:
            raise ModelError(f"{workflow.workflow_id}: step variables not declared by task: {sorted(set(undeclared))}")
        existing = self._read_jsonl(self.root / WORKFLOWS_FILE)
        key = (workflow.workflow_id, workflow.task_source_ref)
        for d in existing:
            w = WorkflowRevision.from_dict(d)
            if (w.workflow_id, w.task_source_ref) == key:
                return None
        self._append_jsonl(self.root / WORKFLOWS_FILE, workflow.to_dict())
        return workflow

    def get_workflow(self, workflow_id: str, product_version: str | None = None) -> WorkflowRevision | None:
        found = [
            WorkflowRevision.from_dict(d)
            for d in self._read_jsonl(self.root / WORKFLOWS_FILE)
            if d["workflow_id"] == workflow_id
        ]
        if product_version is not None:
            found = [w for w in found if w.product_version == product_version]
        if not found:
            return None
        return max(found, key=lambda w: w.revision)

    def list_workflows(self, product_version: str | None = None) -> list[WorkflowRevision]:
        workflows = [WorkflowRevision.from_dict(d) for d in self._read_jsonl(self.root / WORKFLOWS_FILE)]
        if product_version is not None:
            workflows = [w for w in workflows if w.product_version == product_version]
        return workflows

    # ---------- capability snapshots ----------

    def save_snapshot(self, snapshot: CapabilitySnapshot) -> str:
        """Persist a CapabilitySnapshot, versioned by snapshot_id (design.md §4/§8).

        Same snapshot_id with identical content_hash → duplicate, no new record.
        Same snapshot_id with different content_hash → new version appended
        (snapshots are immutable; revisions create a new record, never rewrite).
        Raises when a stored record with the same snapshot_id fails to reparse
        (store corruption is never silently ignored).
        """
        existing = self._read_jsonl(self.root / SNAPSHOTS_FILE)
        same_id: list[CapabilitySnapshot] = []
        for d in existing:
            if d.get("snapshot_id") == snapshot.snapshot_id:
                same_id.append(CapabilitySnapshot.from_dict(d))
        if any(s.content_hash == snapshot.content_hash for s in same_id):
            return "duplicate"
        self._append_jsonl(self.root / SNAPSHOTS_FILE, snapshot.to_dict())
        return "created"

    def get_snapshot(self, snapshot_id: str) -> CapabilitySnapshot | None:
        """Latest stored version for snapshot_id, or None."""
        found = [
            CapabilitySnapshot.from_dict(d)
            for d in self._read_jsonl(self.root / SNAPSHOTS_FILE)
            if d.get("snapshot_id") == snapshot_id
        ]
        if not found:
            return None
        return max(found, key=lambda s: s.observed_at)

    def list_snapshots(self, platform: str | None = None, product_version: str | None = None) -> list[CapabilitySnapshot]:
        """Latest version per snapshot_id; explicit filters only (no cross-version merge)."""
        by_id: dict[str, CapabilitySnapshot] = {}
        for d in self._read_jsonl(self.root / SNAPSHOTS_FILE):
            s = CapabilitySnapshot.from_dict(d)
            cur = by_id.get(s.snapshot_id)
            if cur is None or s.observed_at >= cur.observed_at:
                by_id[s.snapshot_id] = s
        out = list(by_id.values())
        if platform is not None:
            out = [s for s in out if s.platform == platform]
        if product_version is not None:
            out = [s for s in out if s.product_version == product_version]
        return sorted(out, key=lambda s: s.snapshot_id)

    # ---------- review decisions (design.md §4/§9) ----------

    def save_review(self, record: Any) -> Any:
        """Persist a ReviewDecision keyed on (reviewer, decision_id, workflow_hash, pack_hash).

        Same identity tuple with identical content → duplicate, no new record
        (重复提交同一判定不产生重复行). Same identity with different content
        → rejected (判定记录不可被静默覆盖；改判定必须用新 decision_id).
        Any stored line that fails to reparse raises (store corruption is
        never silently ignored).
        """
        from edr_rag.compiler.review_store import save_review_decision
        return save_review_decision(self.root / REVIEWS_FILE, record)

    def list_reviews(self, workflow_hash: str | None = None, pack_hash: str | None = None) -> list[Any]:
        """Explicit-filter listing of stored review decisions (chronological)."""
        from edr_rag.compiler.review_store import list_review_decisions
        return list_review_decisions(self.root / REVIEWS_FILE, workflow_hash=workflow_hash, pack_hash=pack_hash)

    def effective_approval(self, workflow: WorkflowRevision, pack_hash: str) -> dict[str, Any]:
        """§8/§9 发布门禁查询：workflow_hash + pack_hash 是否有有效人工批准。

        只认可存储中 decision=approve 的记录，按 (workflow.content_hash,
        pack_hash) 精确匹配 —— 对象（workflow 或审核包）修改后 hash 漂移，
        旧批准自动不再命中（修改后作废，与 verify_review_decision 同口径）。
        无匹配 → approved=False，unknown 不算通过。
        """
        from edr_rag.compiler.review_store import effective_approval
        return effective_approval(self.root / REVIEWS_FILE, workflow.content_hash, pack_hash)
