"""P5 batch 2: export-bundle 文件导出（design.md §9 / §10 P5）。

输入：ok 的 AuditPackBuilder 产物（同源完整性已由 builder 拒绝失步/篡改
产物）+ 冻结 WorkflowRevision + RenderingStage 产物。导出目录包含：

- workflow.json       —— 冻结 WorkflowRevision.to_dict()（含 steps/变量/
                        capability_snapshot_ref/status/execution_status）；
- plan.json / guide.json —— 同源渲染产物（同一 workflow_hash）；
- audit-pack.md       —— 审核视图（静态 validated ≠ 实机 passed 分层、
                        gaps 如实列出、approval=pending_human_review）；
- manifest.json       —— pack_hash + 各文件 sha256 + workflow_hash 清单。

确定性、无 LLM。目录写入：先在目标目录下写全部临时文件、fsync，最后统一
rename 到最终名 —— 任一文件失败则已写最终名的文件保持原状、失败文件的
临时残片清理，manifest 最后落盘（manifest 缺失即 bundle 不完整，消费者
必须拒绝）。

导出不改变任何治理状态：approval 仍是 pending_human_review；bundle 的
存在不是发布（§8 编译器不自授 published/approved）。
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from edr_rag.knowledge.models import WorkflowRevision, _check_tz_iso8601

EXPORT_BUNDLE_SCHEMA_VERSION = "0.1"

BUNDLE_FILES = ("workflow.json", "plan.json", "guide.json", "audit-pack.md")


def _fail(stage: str, diags: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "ok": False, "stage": stage,
        "schema_version": EXPORT_BUNDLE_SCHEMA_VERSION,
        "diagnostics": diags,
    }


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


class BundleExporter:
    """§9 export-bundle：从 ok 的审核包生成可存档/可人工审阅的文件集。

    前置校验（拒绝导出，不产出半可信 bundle）：
    - pack 必须是 AuditPackBuilder ok 产物，且 pack_hash 与内容一致
      （verify_audit_pack 同规则 —— 包被改过/损坏不得存档）；
    - workflow 冻结 hash 必须与 pack.workflow_hash 一致；
    - 渲染产物 plan/guide 必须绑定同一 workflow_hash。
    """

    def __init__(self, exported_at: str):
        self.exported_at = exported_at
        _check_tz_iso8601(exported_at, "BundleExporter.exported_at")

    def export(
        self,
        pack: dict[str, Any],
        workflow: WorkflowRevision,
        rendered: dict[str, Any],
        out_dir: Path | str,
    ) -> dict[str, Any]:
        diags: list[dict[str, str]] = []

        # ---------- pack envelope 校验（与 verify_audit_pack 同规则） ----------
        payload = pack.get("payload") if isinstance(pack, dict) else None
        if not isinstance(payload, dict) or not isinstance(pack.get("pack_hash"), str):
            return _fail("export_bundle", [{
                "path": "pack", "code": "malformed_pack",
                "message": "export requires an AuditPackBuilder result (payload/pack_hash missing)"}])
        if not pack.get("ok"):
            return _fail("export_bundle", [{
                "path": "pack", "code": "pack_not_ok",
                "message": "pack builder reported failure — 失败审核包不得导出"}])
        if not isinstance(workflow, WorkflowRevision):
            return _fail("export_bundle", [{
                "path": "workflow", "code": "not_workflow_revision",
                "message": "export requires a frozen WorkflowRevision"}])

        # pack 内容完整性：与 verify_audit_pack 相同的 hash 复核
        from edr_rag.compiler.audit_pack import verify_audit_pack
        verdict = verify_audit_pack(pack, workflow)
        if not verdict["valid"]:
            return _fail("export_bundle", verdict["diagnostics"])

        if not isinstance(rendered, dict) or not rendered.get("ok"):
            return _fail("export_bundle", [{
                "path": "rendered", "code": "rendered_not_ok",
                "message": "export requires ok same-source rendering output"}])
        if rendered.get("workflow_hash") != workflow.content_hash:
            return _fail("export_bundle", [{
                "path": "rendered.workflow_hash", "code": "rendered_hash_mismatch",
                "message": "rendering product not derived from this frozen workflow (同源被破坏)"}])

        # ---------- 内容准备 ----------
        files: dict[str, str] = {
            "workflow.json": json.dumps(workflow.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            "plan.json": json.dumps(rendered["plan"], ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            "guide.json": json.dumps(rendered["guide"], ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            "audit-pack.md": pack["markdown"],
        }

        manifest: dict[str, Any] = {
            "schema_version": EXPORT_BUNDLE_SCHEMA_VERSION,
            "kind": "workflow-audit-bundle",
            "exported_at": self.exported_at,
            "workflow_id": workflow.workflow_id,
            "workflow_revision": workflow.revision,
            "workflow_hash": workflow.content_hash,
            "pack_hash": pack["pack_hash"],
            "approval": "pending_human_review",
            "files": {},
        }

        out_dir = Path(out_dir)
        if out_dir.exists():
            return _fail("export_bundle", [{
                "path": "out_dir", "code": "out_dir_exists",
                "message": f"target directory already exists: {out_dir} (不覆盖既有 bundle)"}])
        out_dir.mkdir(parents=True)

        # ---------- 原子落盘：全部写临时、最后统一 rename ----------
        staged: dict[str, Path] = {}
        try:
            for name, text in files.items():
                fd, tmp = tempfile.mkstemp(dir=str(out_dir), prefix=name + ".", suffix=".tmp")
                tmp_path = Path(tmp)
                staged[name] = tmp_path
                with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
                    f.write(text)
                    f.flush()
                    os.fsync(f.fileno())
            # manifest 内容依赖全部文件的实际 sha256 —— sha256 按暂存文件
            # 读回，manifest 最后 rename 落盘。
            file_hashes = {name: _sha256_file(tmp_path) for name, tmp_path in staged.items()}
            manifest["files"] = file_hashes
            manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"

            for name, tmp_path in staged.items():
                os.replace(tmp_path, out_dir / name)
            fd, tmp = tempfile.mkstemp(dir=str(out_dir), prefix="manifest.json.", suffix=".tmp")
            tmp_path = Path(tmp)
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
                f.write(manifest_text)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, out_dir / "manifest.json")
        except BaseException:
            for tmp_path in staged.values():
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
            raise

        return {
            "ok": True, "stage": "export_bundle",
            "schema_version": EXPORT_BUNDLE_SCHEMA_VERSION,
            "exported_at": self.exported_at,
            "out_dir": str(out_dir),
            "workflow_id": workflow.workflow_id,
            "workflow_hash": workflow.content_hash,
            "pack_hash": pack["pack_hash"],
            "files": {name: str(out_dir / name) for name in BUNDLE_FILES},
            "manifest": manifest,
            "approval": "pending_human_review",
            "diagnostics": [],
        }


def verify_export_bundle(out_dir: Path | str) -> dict[str, Any]:
    """消费者侧复核：manifest 存在且各文件 sha256 与清单一致才可信。

    manifest 缺失 / 文件缺失 / hash 不符 → not valid（bundle 不完整或被改）。
    返回 workflow_id/workflow_hash/pack_hash 供审核记录绑定。
    """
    out_dir = Path(out_dir)
    diags: list[dict[str, str]] = []
    mpath = out_dir / "manifest.json"
    if not mpath.is_file():
        return {"ok": False, "valid": False, "diagnostics": [
            {"path": "manifest.json", "code": "manifest_missing",
             "message": "bundle without manifest is incomplete — 消费者必须拒绝"}]}
    try:
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"ok": False, "valid": False, "diagnostics": [
            {"path": "manifest.json", "code": "manifest_unreadable",
             "message": f"manifest unreadable: {exc}"}]}
    if manifest.get("kind") != "workflow-audit-bundle":
        return {"ok": False, "valid": False, "diagnostics": [
            {"path": "manifest.kind", "code": "manifest_kind_unknown",
             "message": f"not a workflow-audit-bundle manifest: {manifest.get('kind')!r}"}]}

    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        return {"ok": False, "valid": False, "diagnostics": [
            {"path": "manifest.files", "code": "manifest_files_invalid",
             "message": "manifest missing files hash map"}]}
    for name, expected in sorted(files.items()):
        fpath = out_dir / name
        if not fpath.is_file():
            diags.append({"path": f"files[{name}]", "code": "bundle_file_missing",
                          "message": f"file listed in manifest is absent: {name}"})
            continue
        if name not in BUNDLE_FILES:
            diags.append({"path": f"files[{name}]", "code": "bundle_file_name_unexpected",
                          "message": f"unexpected file kind: {name}"})
            continue
        actual = _sha256_file(fpath)
        if actual != expected:
            diags.append({"path": f"files[{name}]", "code": "bundle_file_hash_mismatch",
                          "message": f"file content changed after export (修改后作废): {name}"})
    return {"ok": not diags, "valid": not diags, "manifest": manifest, "diagnostics": diags}
