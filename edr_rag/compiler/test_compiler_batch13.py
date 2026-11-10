"""P5 第五批次：ReviewDecision Repository 持久化（design.md §4 / §8 / §9）。

reviews.jsonl 存储契约：
- 正例：真实 SKILL.md 快照全链路 → 审核包 → 人工 approve → save_review 创建；
  同记录重复提交 → duplicate（无重复行）；list_reviews 按 hash 过滤；
  effective_approval 对 (workflow_hash, pack_hash) 精确命中 approved=True；
  多条不同 decision_id 记录共存。
- 负例：
  - 同 identity（reviewer+decision_id+双 hash）不同内容 → conflict 拒绝
    （批准记录不可被静默覆盖）；
  - 审核包内容被篡改后 pack_hash 漂移 → effective_approval 不命中
    （修改后作废，unknown 不算通过）；
  - workflow 重新编译（hash 漂移）→ 旧批准对新 hash 不命中；
  - 存储行 JSON 损坏 / 字段破损 → StoreCorruption 抛出（损坏可检出，
    门禁不失明）；文件缺行内对象/非 dict → 同样拒绝；
  - 空存储 → approved=False（无批准不是批准）。

全部为离线确定性验证；无实机执行。effective_approval=True ≠ 实机 passed；
published 迁移仍需显式审核流程（编译器不自授，§8）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from edr_rag.compiler.audit_pack import AuditPackBuilder
from edr_rag.compiler.capability_snapshot import collect_browser_snapshot
from edr_rag.compiler.review_decision import ReviewDecision
from edr_rag.compiler.review_store import (
    REVIEWS_FILE,
    StoreCorruption,
    effective_approval,
    save_review_decision,
)
from edr_rag.compiler import MappingStage, RenderingStage, ValidationStage
from edr_rag.knowledge.models import (
    AtomicIR,
    EvidenceRef,
    ModelError,
    ParamValue,
    StepVerification,
    TaskVariable,
    TestCaseRevision,
)

BROWSER_SKILL = Path("/Users/edr-test/.hermes/skills/browser-automation/SKILL.md")
browser_skill = pytest.mark.skipif(
    not BROWSER_SKILL.is_file(), reason="browser SKILL.md not present"
)

T0 = "2026-11-08T08:00:00+00:00"
T1 = "2026-11-08T08:00:01+00:00"
T2 = "2026-11-08T08:00:02+00:00"
T3 = "2026-11-08T08:00:03+00:00"

PV = "web-testbed-1.0"
CASE_REF = "case-rstore-e2e@rev1:aaaaaaaaaaaaaaaa"


def rstore_case() -> TestCaseRevision:
    return TestCaseRevision(
        case_id="case-rstore-e2e", product_version=PV, platform="web",
        title="配置服务器地址并保存",
        preconditions=["打开网络设置页面"],
        steps=[
            {"action": "打开网络设置页面", "expected": "页面加载完成"},
            {"action": "在服务器地址输入框输入服务器地址", "expected": "输入框显示所输入的地址"},
            {"action": "保存配置", "expected": "出现保存成功提示"},
        ],
        raw_text_hash="sha256:" + "5" * 64,
        source_ref=CASE_REF,
        created_at=T0,
    )


def rstore_evidence() -> list[EvidenceRef]:
    return [
        EvidenceRef("E-operation-1", CASE_REF, "steps[1].action", "打开网络设置页面", "operation"),
        EvidenceRef("E-operation-2", CASE_REF, "steps[2].action", "在服务器地址输入框输入服务器地址", "operation"),
        EvidenceRef("E-expected-3", CASE_REF, "steps[3].expected", "出现保存成功提示", "expected"),
    ]


def rstore_steps() -> list[AtomicIR]:
    return [
        AtomicIR(step_id="S1", source_step=1, action="navigate",
                 target={"semantic_name": "网络设置页面", "locator_status": "runtime_resolution"},
                 depends_on=[], evidence_refs=["E-operation-1"],
                 inference={"basis": "用例前置：打开网络设置页面（导航动作，来源用例原文）"}),
        AtomicIR(step_id="S2", source_step=2, action="input",
                 target={"semantic_name": "服务器地址输入框", "locator_status": "runtime_resolution"},
                 parameters={"value": ParamValue(variable="server_address")},
                 depends_on=["S1"], evidence_refs=["E-operation-2"]),
        AtomicIR(step_id="S3", source_step=3, action="assert",
                 target={"semantic_name": "保存成功提示", "locator_status": "runtime_resolution"},
                 depends_on=["S2"], evidence_refs=["E-expected-3"],
                 verification=StepVerification("state_check", "server_address", "E-expected-3")),
    ]


def rstore_vars() -> list[TaskVariable]:
    return [TaskVariable(name="server_address", var_type="string", required=True,
                         description="目标服务器地址（用例参数）")]


def build_rstore_pack(workflow_id: str):
    snap, _prov = collect_browser_snapshot(product_version=PV, snapshot_id=f"rst-{workflow_id}",
                                           observed_at=T0)
    mapping = MappingStage(snap, T1).run(rstore_steps(), "web")
    assert mapping["ok"] is True, mapping["diagnostics"]
    validation = ValidationStage(T2).run(
        mapping["bound_steps"],
        workflow_id=workflow_id, task_id=f"task-{workflow_id}", task_source_ref=CASE_REF,
        product_version=PV, platform="web", declared_variables=rstore_vars(),
        task_variables=rstore_vars(), capability_snapshot_ref=snap.content_hash,
    )
    assert validation["ok"] is True, validation["diagnostics"]
    wf = validation["workflow"]
    rendered = RenderingStage(T2).run(wf)
    assert rendered["ok"] is True, rendered["diagnostics"]
    pack = AuditPackBuilder(T2).run(
        wf, rendered, case=rstore_case(), evidence=rstore_evidence(),
        validation=validation, snapshot=snap,
    )
    assert pack["ok"] is True, pack["diagnostics"]
    return wf, pack


def make_record(wf, pack, decision_id: str = "rsd-001", decision: str = "approve",
                reviewer: str = "张工（QA 负责人）", rationale: str = "逐步对照通过") -> ReviewDecision:
    return ReviewDecision(
        reviewer=reviewer, method="human_local", decided_at=T3,
        workflow_hash=wf.content_hash, pack_hash=pack["pack_hash"],
        decision=decision, rationale=rationale, decision_id=decision_id,
    )


@pytest.fixture
def built(tmp_path):
    wf, pack = build_rstore_pack("wf-rstore-e2e")
    path = tmp_path / "kb" / REVIEWS_FILE
    return wf, pack, path


@browser_skill
class TestReviewStorePositive:
    def test_save_created_then_duplicate_idempotent(self, built):
        wf, pack, path = built
        rec = make_record(wf, pack)
        r1 = save_review_decision(path, rec)
        assert r1.status == "created"
        assert path.exists()
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        # 同记录重复提交：duplicate，不产生重复行
        r2 = save_review_decision(path, rec)
        assert r2.status == "duplicate"
        assert len(path.read_text(encoding="utf-8").splitlines()) == 1
        # from_dict 往返一致（存储行可复析）
        assert ReviewDecision.from_dict(json.loads(lines[0])).to_dict() == rec.to_dict()

    def test_multiple_decisions_coexist_and_filter(self, built):
        wf, pack, path = built
        save_review_decision(path, make_record(wf, pack, decision_id="rsd-a"))
        save_review_decision(path, make_record(wf, pack, decision_id="rsd-b",
                                               reviewer="李工", decision="reject",
                                               rationale="清理步骤缺失"))
        eff = effective_approval(path, wf.content_hash, pack["pack_hash"])
        assert eff["approved"] is True  # 一条 approve 即有效
        assert eff["approving_reviewers"] == ["张工（QA 负责人）"]
        assert eff["matched_records"] == 2
        # 按 pack_hash 过滤：不相关 hash 无记录
        assert effective_approval(path, wf.content_hash, "f" * 64)["approved"] is False
        assert effective_approval(path, "e" * 64, pack["pack_hash"])["matched_records"] == 0

    def test_repository_methods_roundtrip(self, built, tmp_path):
        wf, pack, path = built
        from edr_rag.knowledge.repository import KnowledgeRepository
        repo = KnowledgeRepository(tmp_path / "kb")
        rec = make_record(wf, pack)
        assert repo.save_review(rec).status == "created"
        assert repo.save_review(rec).status == "duplicate"
        assert [r.decision_id for r in repo.list_reviews()] == [rec.decision_id]
        assert repo.list_reviews(workflow_hash=wf.content_hash) != []
        assert repo.list_reviews(pack_hash="d" * 64) == []
        verdict = repo.effective_approval(wf, pack["pack_hash"])
        assert verdict["ok"] is True and verdict["approved"] is True


@browser_skill
class TestReviewStoreNegative:
    def test_conflicting_rewrite_rejected(self, built):
        wf, pack, path = built
        save_review_decision(path, make_record(wf, pack, decision_id="rsd-c",
                                               rationale="原始批准理由"))
        # 同 identity（reviewer/decision_id/双 hash）但内容不同 → 拒绝
        with pytest.raises(ModelError, match="conflict"):
            save_review_decision(path, make_record(wf, pack, decision_id="rsd-c",
                                                   rationale="事后改口的新理由"))
        # 原记录未被覆盖
        stored = [ReviewDecision.from_dict(json.loads(l))
                  for l in path.read_text(encoding="utf-8").splitlines()]
        assert len(stored) == 1 and stored[0].rationale == "原始批准理由"

    def test_pack_tamper_breaks_gate_hit(self, built):
        wf, pack, path = built
        save_review_decision(path, make_record(wf, pack))
        tampered = json.loads(json.dumps(pack))
        tampered["payload"]["markdown"] = tampered["payload"]["markdown"] + "\n被篡改"
        # 门禁查询键是 pack_hash：envelope 未重算时原 hash 仍命中记录本身，
        # 内容篡改由 verify_review_decision 的 pack_hash_mismatch 检出（batch11 已验）；
        # 这里验证作废路径 —— 重算篡改后 payload 的 pack_hash，旧批准对新对象不命中
        from edr_rag.knowledge.models import content_hash
        new_hash = content_hash(tampered["payload"])
        assert new_hash != pack["pack_hash"]
        assert effective_approval(path, wf.content_hash, new_hash)["approved"] is False
        # 原始对象上的批准仍有效（未被篡改影响）
        assert effective_approval(path, wf.content_hash, pack["pack_hash"])["approved"] is True

    def test_recompiled_workflow_hash_misses_old_approval(self, built):
        wf, pack, path = built
        save_review_decision(path, make_record(wf, pack))
        wf2, pack2 = build_rstore_pack("wf-rstore-recompile")
        assert wf2.content_hash != wf.content_hash
        eff = effective_approval(path, wf2.content_hash, pack2["pack_hash"])
        assert eff["approved"] is False and eff["matched_records"] == 0
        # 旧对象批准仍在，但只对旧 hash 有效
        assert effective_approval(path, wf.content_hash, pack["pack_hash"])["approved"] is True

    def test_unparseable_line_raises_corruption(self, built):
        wf, pack, path = built
        save_review_decision(path, make_record(wf, pack))
        path.write_text(path.read_text(encoding="utf-8") + "{not json}\n", encoding="utf-8")
        with pytest.raises(StoreCorruption):
            effective_approval(path, wf.content_hash, pack["pack_hash"])

    def test_broken_record_line_raises_corruption(self, built):
        wf, pack, path = built
        save_review_decision(path, make_record(wf, pack))
        broken = json.dumps({"reviewer": "x"})  # 缺大量必填字段
        path.write_text(path.read_text(encoding="utf-8") + broken + "\n", encoding="utf-8")
        with pytest.raises(StoreCorruption):
            effective_approval(path, wf.content_hash, pack["pack_hash"])
        with pytest.raises(StoreCorruption):
            save_review_decision(path, make_record(wf, pack, decision_id="rsd-new"))

    def test_non_object_line_raises_corruption(self, built):
        wf, pack, path = built
        save_review_decision(path, make_record(wf, pack))
        path.write_text(path.read_text(encoding="utf-8") + "[1,2,3]\n", encoding="utf-8")
        with pytest.raises(StoreCorruption):
            effective_approval(path, wf.content_hash, pack["pack_hash"])

    def test_empty_store_is_not_approval(self, built):
        wf, pack, path = built
        eff = effective_approval(path, wf.content_hash, pack["pack_hash"])
        assert eff["ok"] is True and eff["approved"] is False
        assert eff["matched_records"] == 0
