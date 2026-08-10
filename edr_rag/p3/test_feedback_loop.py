"""P4 Phase H —— 反馈闭环：分类、补丁生成、审核门控、真正应用。

对应成熟度评估里的严重问题 11：``classify_failure`` 是 4 条 if-elif 字符串
匹配；``propose_patch`` 只是把调用方自己传进来的 patch 原样返回并改个状态。
既不生成也不应用，反馈回路在这里断了 —— 捕获的失败永远变不成对目录的改进。
"""

from __future__ import annotations

import json

import pytest

from edr_rag.p3.classify import RULES, classify, validate_rules
from edr_rag.p3.feedback import FeedbackEvidence, FeedbackItem, SuggestedPatch
from edr_rag.p3.patch import PATCHABLE_FIELDS, apply, propose
from edr_rag.p3.server import EdrRagAgent


def _item(**over):
    base = dict(
        feedback_id="fb1",
        manual_id="m",
        action_id="network.configure_proxy",
        failure_type="",
        expected="",
        observed="",
    )
    base.update(over)
    evidence = base.pop("evidence", None)
    item = FeedbackItem(**base)
    if evidence:
        item.evidence = FeedbackEvidence(**evidence)
    return item


@pytest.fixture
def workdir(tmp_path):
    rows = [
        {"action_id": "network.configure_proxy", "title": "配置代理服务器",
         "risk": "low", "status": "ready", "confidence": 0.9,
         "selector_synonyms": ["代理设置"]},
        {"action_id": "system.reset", "title": "重置系统",
         "risk": "high", "status": "ready", "confidence": 0.9},
    ]
    (tmp_path / "actions.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
    return tmp_path


def _actions(workdir):
    return {
        r["action_id"]: r
        for r in (
            json.loads(ln)
            for ln in (workdir / "actions.jsonl").read_text(encoding="utf-8").splitlines()
            if ln.strip()
        )
    }


# ---------- 分类 ----------


def test_rule_table_is_valid():
    assert validate_rules() == []
    assert len(RULES) >= 10


@pytest.mark.parametrize("message,expected_type", [
    ("Button 'Apply' not found on screen", "button_text_not_found"),
    ("按钮「应用」未找到", "button_text_not_found"),
    ("ElementNotFound: selector #proxy", "missing_ui_label"),
    ("Operation timed out after 30s", "timeout"),
    ("操作超时", "timeout"),
    ("ConnectionError: connection refused", "network_error"),
    ("网络连接失败", "network_error"),
    ("Unexpected dialog appeared", "unexpected_state"),
    ("pywinauto.findwindows raised", "ui_error"),
    ("success criteria not met", "failed_success_criteria"),
    ("no matching action in catalog", "action_not_found"),
    ("confidence too low", "low_confidence_retrieval"),
])
def test_classification_covers_failure_types(message, expected_type):
    result = classify(_item(evidence={"error_message": message}))
    assert result.matched is True
    assert result.failure_type == expected_type, (
        f"{message!r} -> {result.failure_type} (expected {expected_type})"
    )


def test_no_word_boundary_false_positive():
    """旧实现 `"ui" in msg` 会把 build/guide/require 误判成 UI 错误。"""
    result = classify(_item(evidence={
        "error_message": "Failed to build the guide; requirements not satisfied"
    }))
    assert result.failure_type != "ui_error"


def test_more_specific_rule_wins():
    """同时提到按钮与元素时，更具体的按钮规则应当胜出。"""
    result = classify(_item(evidence={
        "error_message": "element lookup failed: button 'OK' not found"
    }))
    assert result.failure_type == "button_text_not_found"


def test_classifies_from_expected_observed_without_exception():
    """很多失败没有异常消息——线索全在 expected/observed 里。旧实现只看
    error_message，这类失败永远分不出来。"""
    result = classify(_item(
        expected="成功准则应满足", observed="校验失败，界面未变化"
    ))
    assert result.matched is True
    assert result.failure_type == "failed_success_criteria"


def test_unmatched_keeps_original_type():
    """没有规则命中时保留原有类型，而不是硬塞一个错的。"""
    result = classify(_item(failure_type="user_correction",
                            evidence={"error_message": "完全无关的一句话"}))
    assert result.matched is False
    assert result.failure_type == "user_correction"


def test_empty_evidence_is_safe():
    result = classify(_item())
    assert result.matched is False


# ---------- 补丁生成 ----------


def test_proposes_selector_synonym_for_button_mismatch():
    patch = propose(_item(
        failure_type="button_text_not_found",
        expected="应用", observed="确定",
    ))
    assert patch is not None
    assert patch.patch_type == "selector_synonyms"
    assert patch.new_value == "确定"


def test_proposes_risk_escalation_for_unexpected_state():
    patch = propose(_item(failure_type="unexpected_state"))
    assert patch.field == "risk" and patch.new_value == "high"


def test_proposes_needs_review_for_failed_criteria():
    patch = propose(_item(failure_type="failed_success_criteria"))
    assert patch.field == "status" and patch.new_value == "needs_review"


def test_explicit_patch_wins_over_heuristic():
    """人给的补丁比启发式规则可靠。"""
    explicit = SuggestedPatch(patch_type="action_risk", field="risk",
                              old_value="low", new_value="medium")
    patch = propose(_item(failure_type="unexpected_state", suggested_patch=explicit))
    assert patch is explicit


def test_retrieval_failures_yield_no_patch():
    """找不到动作不是目录的错，补丁改不了它。"""
    for ftype in ("action_not_found", "procedure_not_found", "low_confidence_retrieval"):
        assert propose(_item(failure_type=ftype)) is None


# ---------- 应用 ----------


def test_apply_writes_to_catalog(workdir):
    result = apply(workdir, _item(
        failure_type="button_text_not_found", expected="应用", observed="确定",
    ))
    assert result.ok, result.message
    synonyms = _actions(workdir)["network.configure_proxy"]["selector_synonyms"]
    assert "确定" in synonyms
    # 列表字段是追加而非替换——反馈是增量信息，不该抹掉手册原文
    assert "代理设置" in synonyms


def test_apply_captures_rollbackable_version(workdir):
    result = apply(workdir, _item(
        failure_type="button_text_not_found", expected="应用", observed="确定",
    ))
    assert result.version_id is not None
    assert (workdir / "index_versions.json").is_file()


def test_high_risk_action_requires_confirmation(workdir):
    """动作目录驱动客户终端上的点击——高风险条目不许无确认改。"""
    item = _item(action_id="system.reset", failure_type="unexpected_state")
    blocked = apply(workdir, item)
    assert blocked.ok is False
    assert blocked.requires_confirmation is True
    assert _actions(workdir)["system.reset"].get("risk") == "high"

    allowed = apply(workdir, item, confirm=True)
    assert allowed.ok is True


def test_risk_cannot_be_lowered(workdir):
    """允许降级等于让执行侧自己绕过 P2-5 的高危确认门控。"""
    patch = SuggestedPatch(patch_type="action_risk", field="risk",
                           old_value="high", new_value="low")
    result = apply(workdir, _item(action_id="system.reset"), patch, confirm=True)
    assert result.ok is False
    assert "lower risk" in result.message
    assert _actions(workdir)["system.reset"]["risk"] == "high"


def test_risk_can_be_escalated(workdir):
    patch = SuggestedPatch(patch_type="action_risk", field="risk",
                           old_value="low", new_value="high")
    result = apply(workdir, _item(), patch)
    assert result.ok is True
    assert _actions(workdir)["network.configure_proxy"]["risk"] == "high"


def test_non_patchable_field_rejected(workdir):
    """反馈来自被自动化驱动的执行环境；能改任意字段就等于能改写自己的权限约束。"""
    for bad_field in ("tenant", "device", "action_id", "confidence"):
        patch = SuggestedPatch(patch_type="x", field=bad_field, new_value="evil")
        result = apply(workdir, _item(), patch)
        assert result.ok is False, f"字段 {bad_field} 不该可被补丁修改"
        assert "not patchable" in result.message
    assert "tenant" not in PATCHABLE_FIELDS


def test_unknown_action_is_rejected(workdir):
    result = apply(workdir, _item(action_id="no.such.action",
                                  failure_type="unexpected_state"))
    assert result.ok is False and "not in catalog" in result.message


def test_missing_action_id_is_rejected(workdir):
    result = apply(workdir, _item(action_id="", failure_type="unexpected_state"))
    assert result.ok is False


def test_empty_new_value_rejected(workdir):
    patch = SuggestedPatch(patch_type="x", field="status", new_value="")
    assert apply(workdir, _item(), patch).ok is False


# ---------- 端到端状态机 ----------


def test_full_loop_capture_classify_propose_apply(workdir):
    agent = EdrRagAgent(str(workdir), trace=False)

    fb_id = agent.capture_feedback({
        "manual_id": "m",
        "action_id": "network.configure_proxy",
        "expected": "应用",
        "observed": "确定",
        "evidence": {"error_message": "Button '应用' not found"},
    })
    assert fb_id

    classified = agent.classify_failure(fb_id)
    assert classified["ok"] is True
    assert classified["failure_type"] == "button_text_not_found"
    assert classified["matched"] is True

    patch = agent.propose_patch(fb_id)
    assert patch is not None
    assert patch["patch_type"] == "selector_synonyms"

    applied = agent.apply_patch(fb_id)
    assert applied["ok"] is True, applied

    # 目录真的被改了
    assert "确定" in _actions(workdir)["network.configure_proxy"]["selector_synonyms"]
    # 状态推进到 patched
    item = agent._find_feedback(fb_id)
    assert item.status == "patched"


def test_apply_requires_prior_proposal(workdir):
    """不许跳过提议直接应用 —— 审核门控就是靠这个状态推进保证的。"""
    agent = EdrRagAgent(str(workdir), trace=False)
    fb_id = agent.capture_feedback({
        "manual_id": "m", "action_id": "network.configure_proxy",
        "expected": "应用", "observed": "确定",
    })
    result = agent.apply_patch(fb_id)
    assert result["ok"] is False
    assert "propose_patch first" in result["error"]


def test_reject_closes_the_branch(workdir):
    agent = EdrRagAgent(str(workdir), trace=False)
    fb_id = agent.capture_feedback({
        "manual_id": "m", "action_id": "network.configure_proxy",
        "expected": "应用", "observed": "确定",
        "evidence": {"error_message": "Button '应用' not found"},
    })
    agent.classify_failure(fb_id)
    agent.propose_patch(fb_id)
    rejected = agent.reject_patch(fb_id, reason="界面改版属预期，不改目录")
    assert rejected["ok"] is True

    item = agent._find_feedback(fb_id)
    assert item.status == "patch_rejected"
    assert "界面改版" in item.review_note
    # 驳回后不能再应用
    assert agent.apply_patch(fb_id)["ok"] is False


def test_classify_updates_both_fields_atomically(workdir):
    """P3 分两次非原子重写文件（先 status 再 failure_type）；现在一次写完。"""
    agent = EdrRagAgent(str(workdir), trace=False)
    fb_id = agent.capture_feedback({
        "manual_id": "m", "action_id": "network.configure_proxy",
        "evidence": {"error_message": "Operation timed out"},
    })
    agent.classify_failure(fb_id)
    item = agent._find_feedback(fb_id)
    assert item.failure_type == "timeout"
    assert item.status == "needs_review"


def test_unknown_feedback_id_handled(workdir):
    agent = EdrRagAgent(str(workdir), trace=False)
    assert agent.classify_failure("nope")["ok"] is False
    assert agent.propose_patch("nope") is None
    assert agent.apply_patch("nope")["ok"] is False
    assert agent.reject_patch("nope")["ok"] is False
