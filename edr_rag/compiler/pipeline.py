"""§9 对外入口接线：compile / validate-plan / render-guide。

§9 六入口 → 库函数映射（本期核对结论，已有入口不重复封装）：

- import-source → edr_rag.knowledge.document_ingest.import_document_workdir
- import-case   → edr_rag.knowledge.connector.import_case_fixture
- compile       → 本模块 compile_workflow
- validate-plan → 本模块 validate_plan
- render-guide  → 本模块 render_guide
- export-bundle → edr_rag.compiler.export_bundle.BundleExporter.export

compile_workflow 是 §6 阶段 2–5 的确定性串联（CaseUnderstanding →
Decomposition → Mapping → Validation），不新增门禁、不复制阶段逻辑；
provider 响应只当数据，任一阶段 grounding/schema 失败即中断并返回
failed_stage，不产出半成品 Workflow。validated ≠ 实机 passed：产物一律
静态契约状态，编译器不自授发布（Workflow 禁止 published，§8）。
"""

from __future__ import annotations

from typing import Any

from edr_rag.compiler import (
    CaseUnderstandingStage,
    DecompositionStage,
    MappingStage,
    RenderingStage,
    ValidationStage,
)
from edr_rag.knowledge.models import (
    TASK_STATUS,
    ModelError,
    TestCaseRevision,
    TaskIR,
    TaskVariable,
    WorkflowRevision,
    _check_tz_iso8601,
)

PIPELINE_SCHEMA_VERSION = "0.1"

_STAGE_SUMMARY_KEYS = ("ok", "input_hash", "prompt_version", "schema_version")


def _stage_summary(out: dict[str, Any]) -> dict[str, Any]:
    return {k: out[k] for k in _STAGE_SUMMARY_KEYS if k in out}


def _fail(
    failed_stage: str,
    error: str,
    diagnostics: list[dict[str, str]],
    stages: dict[str, Any],
) -> dict[str, Any]:
    return {
        "ok": False,
        "stage": "compile",
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "workflow": None,
        "workflow_hash": None,
        "task": None,
        "failed_stage": failed_stage,
        "error": error,
        "stages": stages,
        "runtime_ready": None,
        "unbound_variables": [],
        "diagnostics": diagnostics,
    }


def _task_from_extraction(
    extraction: dict[str, Any],
    case: TestCaseRevision,
    repo: Any,
    created_at: str,
) -> TaskIR:
    """extraction → TaskIR 确定性转换（provider 响应只当数据）。

    证据 id 取自 repository 中该用例已导入的证据（用例未导入/无证据在此
    拒绝）；gaps 一律落 ambiguities，status 保守降级 needs_review ——
    provider 自报 validated 不被信任为免检。
    """
    evidence_ids = [str(e["evidence_id"]) for e in repo.evidence_of(case)]
    gaps = [str(g) for g in extraction.get("gaps", [])]
    status = str(extraction.get("status") or "needs_review")
    if status not in TASK_STATUS or gaps:
        status = "needs_review"
    variables = [TaskVariable(**v) for v in extraction.get("variables", [])]
    return TaskIR(
        task_id=str(extraction.get("task_id", "")),
        product_version=case.product_version,
        platform=case.platform,
        goal=str(extraction.get("goal", "")),
        source_case_id=case.case_id,
        source_case_ref=case.source_ref,
        source_evidence_ids=evidence_ids,
        preconditions=[str(p) for p in extraction.get("preconditions", [])],
        variables=variables,
        business_result=str(extraction.get("business_result", "")),
        cleanup=[str(c) for c in extraction.get("cleanup", [])],
        ambiguities=gaps,
        status=status,
        created_at=created_at,
    )


def compile_workflow(
    *,
    repo: Any,
    case: TestCaseRevision,
    understanding_provider: Any,
    decomposition_provider: Any,
    snapshot: Any,
    workflow_id: str,
    compiled_at: str,
) -> dict[str, Any]:
    """§9 compile：TestCaseRevision → TaskIR/AtomicIR + 能力绑定 + 静态门禁。

    任何阶段失败返回 ok=False + failed_stage + diagnostics；不抛异常、
    不产出部分 Workflow。runtime_ready=False 仅表示必填业务变量未绑定
    值（静态 validated 仍成立，§7）；执行前必须绑定且再校验。
    """
    stages: dict[str, Any] = {}
    current = "case_understanding"
    try:
        _check_tz_iso8601(compiled_at, "compile_workflow.compiled_at")

        cu = CaseUnderstandingStage(understanding_provider, repo).run(case)
        stages[current] = _stage_summary(cu)
        if not cu["ok"]:
            return _fail(current, str(cu.get("error") or "case understanding failed"),
                         list(cu.get("diagnostics", [])), stages)
        task = _task_from_extraction(cu["extraction"], case, repo, compiled_at)

        current = "decomposition"
        dec = DecompositionStage(decomposition_provider, case, task).run()
        stages[current] = _stage_summary(dec)
        if not dec["ok"]:
            return _fail(current, str(dec.get("error") or "decomposition failed"),
                         list(dec.get("diagnostics", [])), stages)

        current = "mapping"
        mapping = MappingStage(snapshot, compiled_at).run(dec["steps"], case.platform)
        stages[current] = _stage_summary(mapping)
        if not mapping["ok"]:
            return _fail(current, "capability mapping failed",
                         list(mapping.get("diagnostics", [])), stages)

        current = "validation"
        validation = ValidationStage(compiled_at).run(
            mapping["bound_steps"],
            workflow_id=workflow_id,
            task_id=task.task_id,
            task_source_ref=case.source_ref,
            product_version=case.product_version,
            platform=case.platform,
            declared_variables=task.variables,
            task_variables=task.variables,
            unsupported_steps=mapping["unsupported_steps"],
            capability_snapshot_ref=snapshot.content_hash,
        )
        stages[current] = _stage_summary(validation)
        if not validation["ok"]:
            return _fail(current, "validation failed",
                         list(validation.get("diagnostics", [])), stages)
    except (ModelError, TypeError, KeyError) as exc:
        return _fail(current, f"schema violation: {exc}",
                     [{"path": current, "code": "schema_violation", "message": str(exc)}], stages)

    workflow: WorkflowRevision = validation["workflow"]
    return {
        "ok": True,
        "stage": "compile",
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "workflow": workflow,
        "workflow_hash": workflow.content_hash,
        "task": task,
        "failed_stage": None,
        "error": None,
        "stages": stages,
        "runtime_ready": validation["runtime_ready"],
        "unbound_variables": list(validation["unbound_variables"]),
        "diagnostics": list(validation["diagnostics"]),
    }


def _coerce_workflow(workflow: Any) -> WorkflowRevision:
    if isinstance(workflow, WorkflowRevision):
        return workflow
    if not isinstance(workflow, dict):
        raise ModelError(
            f"workflow must be a WorkflowRevision or its dict, got {type(workflow).__name__}"
        )
    return WorkflowRevision.from_dict(workflow)


def validate_plan(workflow: Any) -> dict[str, Any]:
    """§9 validate-plan：对冻结/外部 WorkflowRevision 做结构复验（确定性）。

    复验 WorkflowRevision.validate_steps 全部结构/引用/变量诊断；非法
    输入拒绝为 schema_violation，不算通过。status 为静态治理状态
    （draft/needs_review/validated），不是实机结果。
    """
    try:
        wf = _coerce_workflow(workflow)
    except (ModelError, TypeError) as exc:
        return {
            "ok": False,
            "stage": "validate_plan",
            "schema_version": PIPELINE_SCHEMA_VERSION,
            "workflow_id": None,
            "workflow_hash": None,
            "status": None,
            "diagnostics": [{"path": "workflow", "code": "schema_violation", "message": str(exc)}],
        }
    diags = wf.validate_steps()
    return {
        "ok": not diags,
        "stage": "validate_plan",
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "workflow_id": wf.workflow_id,
        "workflow_hash": wf.content_hash,
        "status": wf.status,
        "diagnostics": diags,
    }


def render_guide(workflow: Any, *, rendered_at: str) -> dict[str, Any]:
    """§9 render-guide：冻结 Workflow → 同源 Plan/Guide（RenderingStage 透传）。

    入口层拒绝（schema_violation）以 stage="render_guide" 标记，与
    RenderingStage 自身失败（stage="rendering"）可区分。
    """
    try:
        wf = _coerce_workflow(workflow)
        _check_tz_iso8601(rendered_at, "render_guide.rendered_at")
    except (ModelError, TypeError) as exc:
        return {
            "ok": False,
            "stage": "render_guide",
            "schema_version": PIPELINE_SCHEMA_VERSION,
            "workflow_hash": None,
            "error": str(exc),
            "diagnostics": [{"path": "workflow", "code": "schema_violation", "message": str(exc)}],
        }
    return RenderingStage(rendered_at).run(wf)
