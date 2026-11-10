"""P6: 冻结样例 —— 离线完整链路可复放 fixture（design.md §10 P6）。

一条命令可复放 offline 全链：真实 browser SKILL.md 能力快照 → fixture 用例
导入（connector 全契约：分页/水位/幂等）→ compile_workflow（CaseUnderstanding
→ Decomposition → Mapping → Validation，OfflineStubProvider 显式标记）→
render_guide（同源 Plan/Guide）→ AuditPackBuilder（审核包）→ BundleExporter
（文件 bundle）→ 人工 approve（ReviewDecision + reviews.jsonl）→
effective_approval 门禁查询命中。

边界（如实分层，不混淆）：
- 离线复放检验的是**编译器链路与门禁**，不是真实产品行为 —— 没有 GUI 执行，
  execution_status 保持 not_run，effective_approval=True ≠ 实机 passed；
- provider 为 OfflineStubProvider（stub=True），非真实模型；
- capability snapshot 绑定真实 browser SKILL.md 字节 hash；SKILL.md 缺失时
  replay 返回 ok=False（fixture 依赖真实来源快照，不允许伪造）；
- 冻结样例不带业务参数值：server_address 保持未绑定（runtime_ready=False，
  §7 静态 validated 与可执行分层）。

输出 replay 报告 dict：阶段结果、hash 链、量化覆盖（原步骤数/关键预期覆盖/
来源覆盖/unsupported/needs_review 数）供 §10 P6 正负例汇总报告引用。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from edr_rag.compiler import OfflineStubProvider
from edr_rag.compiler.audit_pack import AuditPackBuilder
from edr_rag.compiler.capability_snapshot import collect_browser_snapshot
from edr_rag.compiler.export_bundle import BundleExporter, verify_export_bundle
from edr_rag.compiler.pipeline import compile_workflow, render_guide, validate_plan
from edr_rag.compiler.review_decision import ReviewDecision, verify_review_decision
from edr_rag.compiler.review_store import REVIEWS_FILE
from edr_rag.knowledge.connector import CaseBundle
from edr_rag.knowledge.models import (
    EvidenceRef,
    TestCaseRevision,
    WorkflowRevision,
)
from edr_rag.knowledge.repository import KnowledgeRepository

FROZEN_REPLAY_SCHEMA_VERSION = "0.1"

FROZEN_SOURCE_REF = "case-frozen-web-001@rev1:cccccccccccccccc"


def frozen_case() -> TestCaseRevision:
    return TestCaseRevision(
        case_id="case-frozen-web-001", product_version="web-testbed-1.0", platform="web",
        title="配置服务器地址并保存",
        preconditions=["打开网络设置页面"],
        steps=[
            {"action": "打开网络设置页面", "expected": "页面加载完成"},
            {"action": "在服务器地址输入框输入服务器地址", "expected": "输入框显示所输入的地址"},
            {"action": "保存配置", "expected": "出现保存成功提示"},
        ],
        raw_text_hash="sha256:" + "c" * 64,
        source_ref=FROZEN_SOURCE_REF,
        created_at="2026-11-08T00:00:00+00:00",
    )


def frozen_evidence() -> list[EvidenceRef]:
    """逐字证据：操作证据逐步骤 + 预期证据（与 decomposition stub 的
    E-operation-{i}/E-expected-{i} 引用约定对齐；locator 指向用例原文路径）。"""
    steps = frozen_case().steps
    return [
        EvidenceRef(
            f"E-operation-{i}", FROZEN_SOURCE_REF, f"steps[{i}].action",
            str(s["action"]), "operation",
        )
        for i, s in enumerate(steps, 1)
    ] + [
        EvidenceRef(
            f"E-expected-{i}", FROZEN_SOURCE_REF, f"steps[{i}].expected",
            str(s["expected"]), "expected",
        )
        for i, s in enumerate(steps, 1)
    ]


def replay_frozen_sample(root: Path, *, exported_at: str) -> dict[str, Any]:
    """Replay the offline full chain in an empty directory. Deterministic.

    root: 空目录（fixture pages / kb / bundle 都写入其中）。
    exported_at: tz-aware ISO 8601；链内时间戳从它派生，保证逐字节可复放。
    """
    root = Path(root)
    report: dict[str, Any] = {
        "ok": False,
        "stage": "frozen_replay",
        "schema_version": FROZEN_REPLAY_SCHEMA_VERSION,
        "root": str(root),
        "failed_stage": None,
        "error": None,
        "chain": {},
        "quantitative": {},
        "diagnostics": [],
    }

    def fail(stage: str, error: Any) -> dict[str, Any]:
        report["failed_stage"] = stage
        report["error"] = str(error)
        report["diagnostics"] = list(
            error.get("diagnostics", []) if isinstance(error, dict) else []
        )
        return report

    # ---------- ① capability snapshot：真实 browser SKILL.md 字节绑定 ----------
    case_obj = frozen_case()
    evidence_list = frozen_evidence()
    try:
        snapshot, snapshot_prov = collect_browser_snapshot(
            product_version=case_obj.product_version,
            snapshot_id="snap-frozen-web-001",
            observed_at=exported_at,
        )
    except Exception as exc:  # ModelError（SKILL.md 缺失/工具表无映射）如实报失败
        return fail("capability_snapshot", exc)
    report["chain"]["capability_snapshot"] = {
        "snapshot_id": snapshot.snapshot_id,
        "platform": snapshot.platform,
        "content_hash": snapshot.content_hash,
        "actions": list(snapshot.capabilities[0].actions) if snapshot.capabilities else [],
        "source": "browser SKILL.md（真实文件字节 sha256 绑定）",
    }

    # ---------- ② 用例导入（幂等键 = 来源身份+源 hash；重复导入 duplicate） ----------
    try:
        repo = KnowledgeRepository(root / "kb")
        import_run = repo.import_case(CaseBundle(case=case_obj, evidence=evidence_list))
    except Exception as exc:
        return fail("case_import", exc)
    if import_run.status not in ("created", "superseded"):
        return fail("case_import", f"unexpected import status: {import_run.status}")
    evidence_dicts = repo.evidence_of(case_obj)
    report["chain"]["case_import"] = {
        "import_status": import_run.status,
        "case_revision": case_obj.revision,
        "source_ref": case_obj.source_ref,
        "raw_text_hash": case_obj.raw_text_hash,
        "evidence_ids": [e["evidence_id"] for e in evidence_dicts],
    }

    # ---------- ③ compile_workflow（四阶段串联，OfflineStubProvider） ----------
    try:
        compiled = compile_workflow(
            repo=repo, case=case_obj,
            understanding_provider=OfflineStubProvider(),
            decomposition_provider=OfflineStubProvider(mode="decomposition"),
            snapshot=snapshot, workflow_id="wf-frozen-web-001",
            compiled_at=exported_at,
        )
    except Exception as exc:
        return fail("compile", exc)
    if not compiled["ok"]:
        return fail("compile", compiled)
    wf: WorkflowRevision = compiled["workflow"]
    report["chain"]["compile"] = {
        "workflow_id": wf.workflow_id,
        "workflow_hash": compiled["workflow_hash"],
        "task_id": compiled["task"].task_id,
        "status": wf.status,
        "execution_status": wf.execution_status,
        "runtime_ready": compiled["runtime_ready"],
        "unbound_variables": compiled["unbound_variables"],
        "stages": compiled["stages"],
        "provider": "OfflineStubProvider（stub=True，非真实模型）",
    }

    # ---------- ④ validate-plan 复验（§9 入口对冻结产物独立复验） ----------
    try:
        revalidated = validate_plan(wf.to_dict())
    except Exception as exc:
        return fail("validate_plan", exc)
    if not revalidated["ok"]:
        return fail("validate_plan", revalidated)
    report["chain"]["validate_plan"] = {
        "ok": revalidated["ok"],
        "workflow_hash": revalidated["workflow_hash"],
        "hash_stable": revalidated["workflow_hash"] == compiled["workflow_hash"],
    }
    if not revalidated["workflow_hash"] == compiled["workflow_hash"]:
        return fail("validate_plan", "workflow_hash drift between compile and revalidate")

    # ---------- ⑤ render-guide（同源 Plan/Guide） ----------
    try:
        rendered = render_guide(wf, rendered_at=exported_at)
    except Exception as exc:
        return fail("render_guide", exc)
    if not rendered["ok"]:
        return fail("render_guide", rendered)
    report["chain"]["render_guide"] = {
        "workflow_hash": rendered["workflow_hash"],
        "plan_steps": len(rendered["plan"]["steps"]),
        "guide_steps": len(rendered["guide"]["steps"]),
    }
    if rendered["workflow_hash"] != wf.content_hash:
        return fail("render_guide", "rendering not bound to frozen workflow hash")

    # ---------- ⑥ audit pack（审核包；evidence 为 repo 中真实导入的逐字证据） ----------
    evidence = [EvidenceRef.from_dict(e) for e in evidence_dicts]
    try:
        pack = AuditPackBuilder(exported_at).run(
            wf, rendered, case=case_obj, evidence=evidence,
            validation=compiled, snapshot=snapshot,
        )
    except Exception as exc:
        return fail("audit_pack", exc)
    if not pack["ok"]:
        return fail("audit_pack", pack)
    report["chain"]["audit_pack"] = {
        "pack_hash": pack["pack_hash"],
        "complete": pack["complete"],
        "gaps": len(pack["gaps"]),
        "gap_codes": sorted({g["code"] for g in pack["gaps"]}),
    }

    # ---------- ⑦ export bundle（文件导出 + manifest 消费者复核） ----------
    try:
        exported = BundleExporter(exported_at).export(
            pack, wf, rendered, root / "bundle"
        )
        verified = verify_export_bundle(root / "bundle")
    except Exception as exc:
        return fail("export_bundle", exc)
    if not exported["ok"]:
        return fail("export_bundle", exported)
    if not verified["valid"]:
        return fail("export_bundle_verify", verified)
    report["chain"]["export_bundle"] = {
        "out_dir": exported["out_dir"],
        "files": sorted(exported["files"]),
        "manifest_verified": verified["valid"],
        "approval": exported["approval"],
    }

    # ---------- ⑧ 人工 approve（ReviewDecision + reviews.jsonl）+ 门禁命中 ----------
    record = ReviewDecision(
        reviewer="审核人（离线 fixture 演练，非真实审核台）",
        method="human_local",
        decided_at=exported_at,
        workflow_hash=wf.content_hash,
        pack_hash=pack["pack_hash"],
        decision="approve",
        rationale="离线全链路复放通过：逐步对照/证据可解析/双 hash 绑定一致",
        decision_id="rd-frozen-001",
    )
    verdict = verify_review_decision(record, wf, pack)
    if not verdict["effective"]:
        return fail("review_verify", verdict)
    try:
        saved = repo.save_review(record)
        verdict_query = repo.effective_approval(wf, pack["pack_hash"])
    except Exception as exc:
        return fail("review_store", exc)
    if saved.status != "created" or not verdict_query["approved"]:
        return fail("review_store", f"save={saved.status} gate={verdict_query['approved']}")
    report["chain"]["review_gate"] = {
        "decision_id": record.decision_id,
        "save_status": saved.status,
        "effective_approval": verdict_query["approved"],
        "store": str((root / "kb" / REVIEWS_FILE).resolve()),
    }

    # ---------- ⑨ 量化覆盖（§10 P6 量化记录） ----------
    case_steps = case_obj.steps
    n_case_steps = len(case_steps)
    covered_ops, covered_exp, covered_src = set(), set(), set()
    for st in wf.steps:
        if 1 <= st.source_step <= n_case_steps:
            covered_src.add(st.source_step)
            if st.action in ("navigate", "input", "select", "check", "click", "wait"):
                covered_ops.add(st.source_step)
        # 预期覆盖：步骤携带预期类证据引用（E-expected-*）或显式 verification
        if any(r.startswith("E-expected") for r in st.evidence_refs) or st.verification is not None:
            covered_exp.add(st.source_step)
    resolved_rows = [r for r in pack["payload"]["evidence_rows"] if r["resolved"]]
    quantitative = {
        "original_steps": n_case_steps,
        "steps_covered_by_atomic": len(covered_src),
        "operation_evidence_coverage": f"{len(covered_ops)}/{n_case_steps}",
        "key_expected_total": n_case_steps,
        "key_expected_covered": len(covered_exp),
        "evidence_rows_total": len(pack["payload"]["evidence_rows"]),
        "evidence_rows_resolved": len(resolved_rows),
        "unsupported_steps": len(
            [d for d in compiled["diagnostics"] if d.get("code") == "unsupported"]
        ),
        "needs_review_ambiguities": len(compiled["task"].ambiguities),
        "unresolved_gaps": [g["code"] for g in pack["gaps"]],
        "audit_pack_complete": pack["complete"],
        "provider_stub": True,
        "scope": "offline fixture — 编译器链路与门禁验证，非实机执行",
    }
    report["quantitative"] = quantitative

    # 发布候选量化门槛（§10）：关键预期/动作来源覆盖 100%、无未解决关键诊断。
    # 注意：complete=False（runtime_ready=False 记入 gaps）是 §7 分层的如实呈现，
    # 不是缺陷；这里只报告，不据此自动晋级 published（编译器不自授，§8）。
    report["publish_candidate_quantitative_gate"] = {
        "key_expected_coverage_100pct": len(covered_exp) == n_case_steps,
        "source_coverage_100pct": len(covered_src) == n_case_steps,
        "unresolved_critical_diagnostics": [
            g for g in pack["gaps"] if g["code"] not in ("binding_state_unknown",)
        ],
        "note": "满足量化门槛 ≠ published；published 需显式人工审核流程（§8）",
    }

    report["ok"] = True
    return report
