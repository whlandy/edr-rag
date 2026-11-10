"""P5 batch 3: ReviewDecision 人工审核记录模型 (design.md §4 / §8 / §9).

人工审核记录：审核人 / 方式 / 时间 / 对象 hash（workflow_hash + pack_hash）/
决策 / 理由。关键契约：

- 批准只绑定当前 workflow_hash + pack_hash；对象（workflow 或审核包）修改后
  verify_review_decision 判作废 —— hash 不再匹配即批准失效，需要重新审核；
- 编译器不自授：ReviewDecision 只能由人工输入构造（reviewer/method 必填），
  编译器内部不提供自动批准路径；决策枚举 approve/reject/needs_changes。
- 拒绝/需修改同样绑定对象 hash —— 记录的是"对该具体内容的判定"，内容变了
  旧结论不能沿用。
- 静态 validated ≠ 实机 passed：approve 指人工接受当前静态产物，不宣称
  实机行为；execution_status 独立，审核记录不写执行结论。
"""

from __future__ import annotations

from typing import Any

from edr_rag.knowledge.models import (
    ModelError,
    WorkflowRevision,
    _check_tz_iso8601,
    content_hash,
)

REVIEW_SCHEMA_VERSION = "0.1"
REVIEW_DECISIONS = ("approve", "reject", "needs_changes")
REVIEW_METHODS = ("human_local", "human_remote")


def review_payload_hash(payload: dict[str, Any]) -> str:
    """对审核包 payload 的 hash（pack_hash 的独立重算口径，供记录绑定）。"""
    return content_hash(payload)


class ReviewDecision:
    """§4/§8 审核记录：绑定内容 hash 的不可变人工判定。

    构造即校验时间格式、决策/方法枚举、hash 非空；不提供"修改后仍有效"的
    逃生门 —— 想改判定就构造新记录（新 decision_id）。
    """

    def __init__(
        self,
        *,
        reviewer: str,
        method: str,
        decided_at: str,
        workflow_hash: str,
        pack_hash: str,
        decision: str,
        rationale: str,
        decision_id: str,
        schema_version: str = REVIEW_SCHEMA_VERSION,
    ):
        if not reviewer or not str(reviewer).strip():
            raise ModelError("reviewer required (审核记录必须有人)")
        if method not in REVIEW_METHODS:
            raise ModelError(f"method must be one of {REVIEW_METHODS}, got {method!r}")
        _check_tz_iso8601(decided_at, "decided_at")
        for name, h in (("workflow_hash", workflow_hash), ("pack_hash", pack_hash)):
            if not isinstance(h, str) or not h:
                raise ModelError(f"{name} required (审核记录必须绑定对象 hash)")
        if decision not in REVIEW_DECISIONS:
            raise ModelError(f"decision must be one of {REVIEW_DECISIONS}, got {decision!r}")
        if not rationale or not str(rationale).strip():
            raise ModelError("rationale required (拒绝/批准都必须给出理由)")
        if not decision_id or not decision_id.strip():
            raise ModelError("decision_id required")
        self.reviewer = str(reviewer)
        self.method = method
        self.decided_at = decided_at
        self.workflow_hash = workflow_hash
        self.pack_hash = pack_hash
        self.decision = decision
        self.rationale = str(rationale)
        self.decision_id = decision_id
        self.schema_version = schema_version

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "ReviewDecision":
        allowed = {
            "reviewer", "method", "decided_at", "workflow_hash", "pack_hash",
            "decision", "rationale", "decision_id", "schema_version",
        }
        unknown = set(d) - allowed
        if unknown:
            raise ModelError(f"ReviewDecision unknown fields: {sorted(unknown)}")
        missing = allowed - set(d)
        if missing:
            raise ModelError(f"ReviewDecision missing fields: {sorted(missing)}")
        return ReviewDecision(**d)


def verify_review_decision(
    record: ReviewDecision,
    workflow: WorkflowRevision,
    pack: dict[str, Any] | None,
) -> dict[str, Any]:
    """对象修改后作废：workflow_hash 与 pack_hash 逐一复核。

    - workflow.content_hash != record.workflow_hash → 作废（workflow 变了）；
    - pack_hash 用 verify_audit_pack 同规则复核：包内容 hash 失配或包声明的
      workflow_hash 与当前 workflow 不一致 → 作废（包被改 / 不同源）；
    - pack=None 表示包未随记录提供 —— 无法核对 pack_hash，判 pending_pack
      （unknown 不算通过：不能只凭 workflow hash 就宣布批准有效）。
    """
    diags: list[dict[str, str]] = []

    if workflow.content_hash != record.workflow_hash:
        diags.append({"path": "workflow.content_hash", "code": "workflow_hash_drift",
                      "message": "workflow changed after review decision (对象修改后作废)"})

    if pack is None:
        diags.append({"path": "pack", "code": "pack_not_provided",
                      "message": "audit pack not provided; pack_hash 未复核（unknown 不算通过）"})
    else:
        payload = pack.get("payload") if isinstance(pack, dict) else None
        if not isinstance(payload, dict) or not isinstance(pack.get("pack_hash"), str):
            diags.append({"path": "pack", "code": "malformed_pack",
                          "message": "audit pack envelope missing payload/pack_hash"})
        else:
            if content_hash(payload) != pack["pack_hash"]:
                diags.append({"path": "pack.pack_hash", "code": "pack_hash_mismatch",
                              "message": "audit pack content changed after review (包修改后作废)"})
            if record.pack_hash != pack["pack_hash"]:
                diags.append({"path": "record.pack_hash", "code": "record_pack_hash_mismatch",
                              "message": "record was issued against a different pack (判定对象不匹配)"})
            if payload.get("workflow_hash") != record.workflow_hash:
                diags.append({"path": "pack.payload.workflow_hash", "code": "pack_workflow_hash_mismatch",
                              "message": "pack declares a different workflow than the record binds"})

    valid = not diags
    return {
        "ok": True, "stage": "review_verify",
        "schema_version": REVIEW_SCHEMA_VERSION,
        "valid": valid,
        "decision": record.decision,
        "effective": valid and record.decision == "approve",
        "diagnostics": diags,
    }
