"""P5 batch 1: 同源 Markdown 审核包 (design.md §9 / §10 P5).

审核包输入：冻结 WorkflowRevision + RenderingStage 同源渲染产物 + 原用例
（TestCaseRevision）与证据（EvidenceRef dict 列表），可选 ValidationStage 结论
与 CapabilitySnapshot：

- 输出原用例引用、Plan/Guide 逐步对照、来源跳转（evidence → source_ref /
  locator / quote hash）、参数缺口、能力适配、静态/实机状态分层标注；
- 同源完整性：渲染产物必须绑定同一 workflow_hash，Plan/Guide 步骤与冻结
  WorkflowRevision 一一对应，参数/预期不一致即拒绝（审核包不为失步产物背书）；
- 审核包绑定 workflow_hash 与自身 pack_hash：workflow 或包内容修改后
  verify_audit_pack 校验作废（§9 审核记录绑定内容 hash，修改后作废）；
- 静态 validated ≠ 实机 passed：两状态分层输出，不互相推导；编译器不自授
  published/approved，审核结论由人工 ReviewDecision 生成。

确定性、无 LLM；凭据/绑定值不进入审核包（变量只以类型化引用出现）。
"""

from __future__ import annotations

from typing import Any

from edr_rag.knowledge.models import WorkflowRevision, _check_tz_iso8601, content_hash

AUDIT_PACK_SCHEMA_VERSION = "0.1"


def _cell(text: Any) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def _short(h: str, n: int = 16) -> str:
    h = str(h)
    return h if len(h) <= n else h[:n] + "…"


def _param_cell(params: dict[str, Any]) -> str:
    if not params:
        return "（无）"
    parts = []
    for k, v in params.items():
        if v.get("kind") == "variable":
            parts.append(f"{k}=⟨var:{v.get('name')}⟩")
        else:
            parts.append(f"{k}={v.get('value')!r}")
    return "; ".join(parts)


def _ev_dict(e: Any) -> dict[str, Any]:
    return e.to_dict() if hasattr(e, "to_dict") else dict(e)


class AuditPackBuilder:
    """§9 审核包构建器：确定性、无 LLM，从冻结 WorkflowRevision 生成审核视图。

    失败（产物失步/来源错绑/结构破损）不产出审核包 —— 审核视图不能为不可信
    产物背书；缺口（证据不可解析、绑定结论/快照未提供）如实列出，unknown
    不算通过。
    """

    def __init__(self, built_at: str):
        self.built_at = built_at
        _check_tz_iso8601(built_at, "AuditPackBuilder.built_at")

    def run(
        self,
        workflow: Any,
        rendered: Any,
        *,
        case: Any,
        evidence: list[Any],
        validation: dict[str, Any] | None = None,
        snapshot: Any = None,
    ) -> dict[str, Any]:
        diags: list[dict[str, str]] = []

        def fail() -> dict[str, Any]:
            return {
                "ok": False, "stage": "audit_pack",
                "schema_version": AUDIT_PACK_SCHEMA_VERSION,
                "diagnostics": diags,
            }

        if not isinstance(workflow, WorkflowRevision):
            diags.append({"path": "workflow", "code": "not_workflow_revision",
                          "message": "audit pack requires a frozen WorkflowRevision"})
            return fail()

        struct = workflow.validate_steps()
        if struct:
            diags.append({"path": "workflow", "code": "workflow_structure_invalid",
                          "message": f"frozen workflow carries structural diagnostics: {struct}"})

        if not isinstance(rendered, dict) or not rendered.get("ok"):
            diags.append({"path": "rendered", "code": "rendered_not_ok",
                          "message": "audit pack requires ok same-source rendering output"})
        elif rendered.get("workflow_hash") != workflow.content_hash:
            diags.append({"path": "rendered.workflow_hash", "code": "rendered_hash_mismatch",
                          "message": "rendering product was not derived from this frozen workflow (同源被破坏)"})

        plan_steps = rendered.get("plan", {}).get("steps") if isinstance(rendered, dict) else None
        guide_steps = rendered.get("guide", {}).get("steps") if isinstance(rendered, dict) else None
        if not isinstance(plan_steps, list) or not isinstance(guide_steps, list):
            diags.append({"path": "rendered", "code": "rendered_shape_invalid",
                          "message": "rendering output missing plan/guide step lists"})
            plan_steps, guide_steps = [], []
        else:
            wf_ids = [s.step_id for s in workflow.steps]
            if [s.get("step_id") for s in plan_steps] != wf_ids or [s.get("step_id") for s in guide_steps] != wf_ids:
                diags.append({"path": "rendered.plan/guide.steps", "code": "step_correspondence_violation",
                              "message": "Plan/Guide steps must match frozen workflow steps one-to-one in order (§10 P5)"})
            else:
                for ps, gs in zip(plan_steps, guide_steps):
                    if (ps.get("action") != gs.get("action")
                            or ps.get("expected") != gs.get("expected")
                            or ps.get("parameters") != gs.get("parameters")):
                        diags.append({"path": f"guide[{gs.get('step_id')}]", "code": "guide_divergence",
                                      "message": "Guide diverges from Plan (参数/预期/动作不一致 — 同源渲染被篡改)"})

        if case.source_ref != workflow.task_source_ref:
            diags.append({"path": "case.source_ref", "code": "case_source_ref_mismatch",
                          "message": f"case source_ref {case.source_ref!r} != workflow.task_source_ref {workflow.task_source_ref!r}"})
        if case.platform != workflow.platform:
            diags.append({"path": "case.platform", "code": "platform_mismatch",
                          "message": f"case platform {case.platform!r} != workflow platform {workflow.platform!r} (跨平台不混检)"})
        if case.product_version != workflow.product_version:
            diags.append({"path": "case.product_version", "code": "product_version_mismatch",
                          "message": f"case product_version {case.product_version!r} != workflow {workflow.product_version!r} (跨版本不混检)"})

        ev_dicts = [_ev_dict(e) for e in evidence]
        if not ev_dicts:
            diags.append({"path": "evidence", "code": "evidence_missing",
                          "message": "case evidence required (审核包须含原用例证据，缺证据不算通过)"})
        else:
            bad_src = sorted({e["evidence_id"] for e in ev_dicts if e.get("source_ref") != case.source_ref})
            if bad_src:
                diags.append({"path": "evidence", "code": "evidence_source_mismatch",
                              "message": f"evidence not bound to case source_ref {case.source_ref!r}: {bad_src}"})

        if diags:
            return fail()

        # ---------- gaps：如实列出，不阻断审核视图 ----------
        gaps: list[dict[str, str]] = []

        ev_by_id = {e["evidence_id"]: e for e in ev_dicts}
        evidence_rows: list[dict[str, Any]] = []
        for step in workflow.steps:
            for ref in step.evidence_refs:
                ev = ev_by_id.get(ref)
                if ev is None:
                    gaps.append({"path": f"steps[{step.step_id}].evidence_refs",
                                 "code": "unresolved_evidence_ref",
                                 "message": f"evidence {ref!r} 不在原用例证据中（来源不可解析）"})
                    evidence_rows.append({"step_id": step.step_id, "evidence_id": ref,
                                          "source_ref": "", "locator": "", "quote_hash": "",
                                          "purpose": "", "resolved": False})
                else:
                    evidence_rows.append({"step_id": step.step_id, "evidence_id": ref,
                                          "source_ref": ev.get("source_ref", ""),
                                          "locator": ev.get("locator", ""),
                                          "quote_hash": content_hash({"quote": ev.get("quote", "")}),
                                          "purpose": ev.get("purpose", ""), "resolved": True})

        plan_guide_rows: list[dict[str, Any]] = []
        for wfs, ps, gs in zip(workflow.steps, plan_steps, guide_steps):
            plan_guide_rows.append({
                "step_id": wfs.step_id, "source_step": wfs.source_step, "action": wfs.action,
                "plan_target": str(ps.get("target", {}).get("semantic_name", "")),
                "guide_target": str(gs.get("target_name", "")),
                "params": _param_cell(ps.get("parameters", {})),
                "expected": str(ps.get("expected", "")),
            })

        runtime_ready: bool | None = None
        unbound: list[str] = []
        if validation is None:
            gaps.append({"path": "validation", "code": "binding_state_unknown",
                         "message": "未提供 ValidationStage 结论，变量绑定状态未知（unknown 不算通过）"})
        else:
            runtime_ready = bool(validation.get("runtime_ready"))
            unbound = [str(v) for v in validation.get("unbound_variables", [])]

        snap_hash = ""
        if snapshot is None:
            gaps.append({"path": "snapshot", "code": "capability_snapshot_not_provided",
                         "message": "未提供能力快照，绑定 hash 复核未执行（状态未知，不宣称已核对）"})
        else:
            snap_hash = snapshot.content_hash
            if workflow.capability_snapshot_ref and workflow.capability_snapshot_ref != snap_hash:
                gaps.append({"path": "workflow.capability_snapshot_ref", "code": "snapshot_ref_drift",
                             "message": "workflow compiled against a different capability snapshot (能力快照漂移)"})

        binding_rows: list[dict[str, Any]] = []
        for step in workflow.steps:
            b = step.binding
            if b is None:
                gaps.append({"path": f"steps[{step.step_id}].binding", "code": "unbound_step",
                             "message": f"step {step.step_id} 未绑定能力（§7 未绑定计划不得下发）"})
                binding_rows.append({"step_id": step.step_id, "action": step.action, "bound": False,
                                     "tool": "", "tool_version": "", "tool_schema_hash": "",
                                     "snapshot_hash": "", "locator_status": str(step.target.get("locator_status", "")),
                                     "snapshot_drift": False})
            else:
                drift = bool(snap_hash and b.snapshot_hash != snap_hash)
                if drift:
                    gaps.append({"path": f"steps[{step.step_id}].binding.snapshot_hash",
                                 "code": "capability_drift",
                                 "message": f"step {step.step_id} binding snapshot_hash != current snapshot (能力漂移)"})
                binding_rows.append({"step_id": step.step_id, "action": step.action, "bound": True,
                                     "tool": b.tool, "tool_version": b.tool_version,
                                     "tool_schema_hash": b.tool_schema_hash,
                                     "snapshot_hash": b.snapshot_hash,
                                     "locator_status": str(step.target.get("locator_status", "")),
                                     "snapshot_drift": drift})

        variable_rows: list[dict[str, Any]] = []
        for v in workflow.declared_variables:
            if v.name in unbound:
                note = "未绑定值 —— 静态可 validated，runtime_ready=False；执行前必须绑定并再校验（§7）"
            elif validation is None:
                note = "绑定状态未知（未提供 ValidationStage 结论）"
            else:
                note = "已有绑定结论（绑定值不进入审核包）"
            variable_rows.append({"name": v.name, "var_type": v.var_type,
                                  "required": bool(v.required), "note": note})

        case_view = {
            "case_id": case.case_id, "revision": case.revision,
            "source_ref": case.source_ref, "title": case.title,
            "preconditions": list(case.preconditions),
            "steps": [dict(s) for s in case.steps],
        }
        payload: dict[str, Any] = {
            "schema_version": AUDIT_PACK_SCHEMA_VERSION,
            "workflow_hash": workflow.content_hash,
            "workflow_id": workflow.workflow_id, "revision": workflow.revision,
            "product_version": workflow.product_version, "platform": workflow.platform,
            "task_id": workflow.task_id, "task_source_ref": workflow.task_source_ref,
            "built_at": self.built_at,
            "static_status": workflow.status,
            "execution_status": workflow.execution_status,
            "runtime_ready": runtime_ready,
            "unbound_variables": unbound,
            "approval": "pending_human_review",
            "complete": not gaps,
            "gaps": gaps,
            "case_view": case_view,
            "plan_guide_rows": plan_guide_rows,
            "evidence_rows": evidence_rows,
            "variable_rows": variable_rows,
            "binding_rows": binding_rows,
            "capability_snapshot_hash": snap_hash,
        }
        payload["markdown"] = self._markdown(
            workflow, case_view, evidence_rows, plan_guide_rows,
            variable_rows, binding_rows, gaps, runtime_ready, snap_hash,
        )
        pack_hash = content_hash(payload)
        return {
            "ok": True, "stage": "audit_pack",
            "schema_version": AUDIT_PACK_SCHEMA_VERSION,
            "built_at": self.built_at,
            "workflow_hash": workflow.content_hash,
            "pack_hash": pack_hash,
            "payload": payload,
            "markdown": payload["markdown"],
            "gaps": gaps,
            "complete": not gaps,
            "diagnostics": [],
        }

    # ---------- markdown view ----------

    def _markdown(
        self,
        workflow: Any,
        case_view: dict[str, Any],
        evidence_rows: list[dict[str, Any]],
        plan_guide_rows: list[dict[str, Any]],
        variable_rows: list[dict[str, Any]],
        binding_rows: list[dict[str, Any]],
        gaps: list[dict[str, str]],
        runtime_ready: bool | None,
        snap_hash: str,
    ) -> str:
        L: list[str] = []
        L.append(f"# 审核包 {workflow.workflow_id}@rev{workflow.revision}")
        L.append("")
        L.append(f"- 生成时间: {self.built_at}")
        L.append(f"- schema_version: {AUDIT_PACK_SCHEMA_VERSION}")
        L.append(f"- 产品/平台: {workflow.product_version} / {workflow.platform}")
        L.append(f"- workflow_hash: `{workflow.content_hash}`（审核记录必须绑定此 hash；workflow 修改后本包作废）")
        L.append("- 本包内容 hash（pack_hash）随包记录；修改本包任一内容即作废。")
        L.append("")
        L.append("## 1. 哈希绑定与状态分层")
        L.append("")
        L.append(f"- 静态状态: `{workflow.status}` —— 仅指静态契约校验通过（§8）")
        L.append(f"- 实机状态: `{workflow.execution_status}` —— 与静态状态独立记录")
        if runtime_ready is None:
            rr = "未知（未提供绑定结论；unknown 不算通过）"
        elif runtime_ready:
            rr = "true"
        else:
            rr = "false —— 存在未绑定必填变量，执行前必须绑定并再校验（§7）"
        L.append(f"- runtime_ready: {rr}")
        L.append("- 发布/批准: 无 —— 编译器不自授 published/approved；批准需人工 ReviewDecision 绑定当前 workflow_hash。")
        L.append("")
        L.append("> 静态 validated ≠ 实机 passed：静态校验不证明实机行为；实机结论需执行证据，unknown 不算通过。")
        L.append("")
        L.append("## 2. 原用例（原文引用）")
        L.append("")
        L.append(f"- case_id: `{case_view['case_id']}` @rev{case_view['revision']}（source_ref: `{case_view['source_ref']}`）")
        L.append(f"- 标题: {_cell(case_view['title'])}")
        L.append("")
        L.append("### 前置条件")
        L.append("")
        if case_view["preconditions"]:
            for p in case_view["preconditions"]:
                L.append(f"- {_cell(p)}")
        else:
            L.append("-（无）")
        L.append("")
        L.append("### 步骤原文")
        L.append("")
        L.append("| # | 操作 | 预期 |")
        L.append("|---|------|------|")
        for i, s in enumerate(case_view["steps"], 1):
            L.append(f"| {i} | {_cell(s.get('action', ''))} | {_cell(s.get('expected', ''))} |")
        L.append("")
        L.append("## 3. Plan / Guide 逐步对照（同源 workflow_hash）")
        L.append("")
        L.append("| step | source_step | 动作 | Plan 目标 | Guide 目标 | 参数 | 预期（Plan=Guide） |")
        L.append("|------|-------------|------|-----------|-----------|------|--------------------|")
        for r in plan_guide_rows:
            L.append(f"| {r['step_id']} | {r['source_step']} | {r['action']} | {_cell(r['plan_target'])} "
                     f"| {_cell(r['guide_target'])} | {_cell(r['params'])} | {_cell(r['expected'])} |")
        L.append("")
        L.append("## 4. 来源跳转")
        L.append("")
        L.append("| step | evidence_id | source_ref | locator | quote_hash | 用途 |")
        L.append("|------|-------------|------------|---------|------------|------|")
        for r in evidence_rows:
            if r["resolved"]:
                L.append(f"| {r['step_id']} | {r['evidence_id']} | `{r['source_ref']}` | {_cell(r['locator'])} "
                         f"| `{_short(r['quote_hash'])}` | {r['purpose']} |")
            else:
                L.append(f"| {r['step_id']} | {r['evidence_id']} | —— 来源不可解析 —— | | | |")
        L.append("")
        L.append("## 5. 参数缺口（变量类型化绑定；绑定值不进入审核包）")
        L.append("")
        L.append("| 变量 | 类型 | 必填 | 状态 |")
        L.append("|------|------|------|------|")
        for r in variable_rows:
            L.append(f"| `{r['name']}` | {r['var_type']} | {'是' if r['required'] else '否'} | {_cell(r['note'])} |")
        L.append("")
        L.append("## 6. 能力适配")
        L.append("")
        L.append("| step | 动作 | 绑定工具 | 工具版本 | schema hash | snapshot hash | locator | 备注 |")
        L.append("|------|------|----------|----------|-------------|---------------|---------|------|")
        for r in binding_rows:
            if r["bound"]:
                note = "快照漂移" if r["snapshot_drift"] else ""
                L.append(f"| {r['step_id']} | {r['action']} | {_cell(r['tool'])} | {_cell(r['tool_version'])} "
                         f"| `{_short(r['tool_schema_hash'])}` | `{_short(r['snapshot_hash'])}` "
                         f"| {r['locator_status']} | {note} |")
            else:
                L.append(f"| {r['step_id']} | {r['action']} | （未绑定） | | | | {r['locator_status']} "
                         f"| 未绑定计划不得下发（§7） |")
        L.append("")
        if snap_hash:
            L.append(f"- 绑定已对照当前能力快照 `{_short(snap_hash)}`；漂移见备注行。")
        else:
            L.append("- 快照未提供：绑定 hash 复核未执行（状态未知，不宣称已核对）。")
        L.append("")
        L.append("## 7. 审核结论占位与缺口")
        L.append("")
        L.append("- 编译器不生成审核结论；批准/拒绝由人工 ReviewDecision 记录（需绑定 workflow_hash 与 pack_hash）。")
        L.append("")
        if gaps:
            L.append(f"gaps 共 {len(gaps)} 项：")
            L.append("")
            for g in gaps:
                L.append(f"- [{g['code']}] {_cell(g['message'])}")
        else:
            L.append("gaps 共 0 项 —— 来源全部可解析，绑定结论与快照核对已提供。")
        L.append("")
        return "\n".join(L)


def verify_audit_pack(pack: dict[str, Any], workflow: WorkflowRevision) -> dict[str, Any]:
    """§9 审核记录绑定内容 hash：workflow 或包内容修改后校验作废。"""
    payload = pack.get("payload") if isinstance(pack, dict) else None
    if not isinstance(payload, dict) or not isinstance(pack.get("pack_hash"), str):
        return {"ok": False, "valid": False, "diagnostics": [
            {"path": "pack", "code": "malformed_pack",
             "message": "audit pack envelope missing payload/pack_hash"}]}
    diags: list[dict[str, str]] = []
    if content_hash(payload) != pack["pack_hash"]:
        diags.append({"path": "pack.pack_hash", "code": "pack_hash_mismatch",
                      "message": "audit pack content changed after build (修改后作废)"})
    if payload.get("workflow_hash") != workflow.content_hash:
        diags.append({"path": "pack.payload.workflow_hash", "code": "pack_workflow_hash_mismatch",
                      "message": "workflow changed after pack was built (审核记录绑定 hash 作废)"})
    return {"ok": not diags, "valid": not diags, "diagnostics": diags}
