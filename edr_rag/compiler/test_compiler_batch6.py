"""P4 第一批次：CapabilitySnapshot 真实采集测试（design.md §4 / §8）。

来源为真实只读仓库/文件（edr-wd action_catalog、browser-automation SKILL.md），
另用合成 fixture 目录验证负例与漂移，不把采集失败静默降级。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from edr_rag.compiler.capability_snapshot import (
    BROWSER_ACTION_MAP,
    EDW_ACTION_MAP,
    _edw_catalog_payload,
    collect_browser_snapshot,
    collect_edr_wd_snapshot,
)
from edr_rag.compiler import MappingStage
from edr_rag.knowledge.models import (
    ATOMIC_ACTIONS,
    AtomicIR,
    CapabilitySnapshot,
    ModelError,
)

EDR_WD_REPO = Path("/Users/edr-test/ai-projects/edr-wd")
BROWSER_SKILL = Path("/Users/edr-test/.hermes/skills/browser-automation/SKILL.md")

edr_wd_repo = pytest.mark.skipif(
    not EDR_WD_REPO.is_dir(), reason="edr-wd repo not present"
)
browser_skill = pytest.mark.skipif(
    not BROWSER_SKILL.is_file(), reason="browser SKILL.md not present"
)


def _make_fake_edr_wd(tmp_path: Path, tools: list[dict] | None = None) -> Path:
    """合成最小 edr-wd 形状目录（仅用于负例/漂移验证，非真实采集）。"""
    (tmp_path / "target" / "action_catalog").mkdir(parents=True)
    (tmp_path / "target" / "__init__.py").write_text("")
    (tmp_path / "target" / "action_catalog" / "__init__.py").write_text("")
    actions = tools if tools is not None else [
        {
            "action_id": "session.connect", "action_code": "A001",
            "tool_name": "connect", "description": "Connect to a GUI application.",
            "category": "session", "input_schema": {"type": "object"},
            "result_schema": {"type": "object"},
            "backends": ["windows_pywinauto"], "side_effect": "session_mutation",
            "risk": "low", "execution_provider": "backend",
        },
    ]
    code = (
        "ACTIONS_V1 = (\n"
        + "".join(
            "    SimpleNamespace(**json.loads(%r)),\n" % json.dumps(a) for a in actions
        )
        + ")\n"
        "CATALOG_VERSION = '9.9.9-fake'\n"
    )
    (tmp_path / "target" / "action_catalog" / "actions_v1.py").write_text(
        "import json\nfrom types import SimpleNamespace\n" + code
    )
    # digest 机制：按 edr-wd 原样最小实现（canonical sha256 over sorted actions）
    (tmp_path / "target" / "action_catalog" / "digest.py").write_text(
        "import hashlib, json\n"
        "def catalog_digest(specs=None):\n"
        "    from .actions_v1 import ACTIONS_V1\n"
        "    payload = {'actions': [\n"
        "        {'action_id': s.action_id, 'tool_name': s.tool_name}\n"
        "        for s in sorted(ACTIONS_V1, key=lambda x: x.action_id)]}\n"
        "    b = json.dumps(payload, sort_keys=True, separators=(',', ':'),"
        " ensure_ascii=False).encode('utf-8')\n"
        "    return 'sha256:' + hashlib.sha256(b).hexdigest()\n"
    )
    (tmp_path / "target" / "action_catalog" / "registry.py").write_text(
        "CATALOG_VERSION = '9.9.9-fake'\n"
    )
    return tmp_path


class TestCollectEdrWdSnapshot:
    @edr_wd_repo
    def test_real_catalog_snapshot(self):
        snap, prov = collect_edr_wd_snapshot(
            product_version="5.3.0-testbed", snapshot_id="edr-wd-p4-live"
        )
        assert isinstance(snap, CapabilitySnapshot)
        assert snap.platform == "windows"
        assert snap.capabilities[0].tool.startswith("edr-wd:")
        assert prov["catalog_version"].count(".") == 2
        assert prov["catalog_digest"].startswith("sha256:")
        # 映射后的动作必须是合法 AtomicIR 语义动作子集
        assert set(snap.capabilities[0].actions) <= set(ATOMIC_ACTIONS)
        assert "connect" in snap.capabilities[0].actions
        assert "click" in snap.capabilities[0].actions

    @edr_wd_repo
    def test_hash_binding_deterministic(self):
        """同一次观察内重复采集 → 相同 tool_schema_hash 与 content_hash。"""
        s1, _ = collect_edr_wd_snapshot(
            product_version="5.3.0", snapshot_id="p4-a", observed_at="2026-11-08T00:00:00+00:00"
        )
        s2, _ = collect_edr_wd_snapshot(
            product_version="5.3.0", snapshot_id="p4-a", observed_at="2026-11-08T00:00:00+00:00"
        )
        assert s1.content_hash == s2.content_hash
        assert s1.capabilities[0].tool_schema_hash == s2.capabilities[0].tool_schema_hash

    @edr_wd_repo
    def test_macos_backend_maps_to_macos_platform(self):
        snap, prov = collect_edr_wd_snapshot(
            product_version="5.3.0", snapshot_id="p4-mac",
            platform_backend="macos_accessibility",
        )
        assert snap.platform == "macos"
        assert prov["platform_backend"] == "macos_accessibility"

    @edr_wd_repo
    def test_drift_detection_via_mapping_stage(self, tmp_path):
        """§8：绑定后目录变化（schema hash 不同）→ MappingStage 拒绝旧快照门禁可检出。

        用真实快照绑定，再用合成目录（不同 digest）构造新快照，二者 hash 必不同。
        """
        s_real, _ = collect_edr_wd_snapshot(
            product_version="5.3.0", snapshot_id="p4-real",
            observed_at="2026-11-08T00:00:00+00:00",
        )
        fake = _make_fake_edr_wd(tmp_path / "fake_repo")
        s_fake, _ = collect_edr_wd_snapshot(
            product_version="5.3.0", snapshot_id="p4-fake",
            observed_at="2026-11-08T00:00:00+00:00", repo_root=fake,
        )
        assert s_real.capabilities[0].tool_schema_hash != s_fake.capabilities[0].tool_schema_hash
        # content_hash 锚不同 → 旧绑定不可复用
        assert s_real.content_hash != s_fake.content_hash

    def test_missing_repo_raises_not_silent(self, tmp_path):
        with pytest.raises(ModelError, match="not found"):
            collect_edr_wd_snapshot(
                product_version="5.3.0", snapshot_id="p4-missing",
                repo_root=tmp_path / "nope",
            )

    def test_unmapped_tools_listed_in_provenance(self):
        _, prov = collect_edr_wd_snapshot(
            product_version="5.3.0", snapshot_id="p4-prov",
        ) if EDR_WD_REPO.is_dir() else ({}, {})
        if not prov:
            pytest.skip("edr-wd repo not present")
        # 目录真实存在但未映射进 snapshot 的工具必须可审计（不虚构声明）
        assert "dump_tree" in prov["unmapped_tools"]
        assert "find_control" in prov["unmapped_tools"]

    def test_unknown_backend_rejected(self):
        with pytest.raises(ModelError, match="platform_backend"):
            collect_edr_wd_snapshot(
                product_version="5.3.0", snapshot_id="p4-bad",
                platform_backend="linux_x11",
            )


class TestCollectBrowserSnapshot:
    @browser_skill
    def test_real_skill_md_snapshot(self):
        snap, prov = collect_browser_snapshot(
            product_version="web-testbed", snapshot_id="web-p4-live"
        )
        assert snap.platform == "web"
        assert "navigate" in snap.capabilities[0].actions
        assert "click" in snap.capabilities[0].actions
        assert snap.capabilities[0].tool_schema_hash.startswith("sha256:")
        assert prov["file_sha256"].startswith("sha256:")

    @browser_skill
    def test_skill_md_hash_is_content_bound(self):
        s1, _ = collect_browser_snapshot(
            product_version="web", snapshot_id="w-a",
            observed_at="2026-11-08T00:00:00+00:00",
        )
        s2, _ = collect_browser_snapshot(
            product_version="web", snapshot_id="w-a",
            observed_at="2026-11-08T00:00:00+00:00",
        )
        assert s1.content_hash == s2.content_hash

    @browser_skill
    def test_drift_when_skill_md_changes(self, tmp_path):
        original = BROWSER_SKILL.read_text()
        tampered = original.replace("`browser_click`", "`browser_click_v2`")
        fake = tmp_path / "SKILL.md"
        fake.write_text(tampered)
        s1, p1 = collect_browser_snapshot(
            product_version="web", snapshot_id="w-orig",
            observed_at="2026-11-08T00:00:00+00:00",
        )
        s2, p2 = collect_browser_snapshot(
            product_version="web", snapshot_id="w-tamper",
            observed_at="2026-11-08T00:00:00+00:00", skill_md_path=fake,
        )
        assert s1.content_hash != s2.content_hash
        # tampered tool 名不再映射 → actions 收缩，漂移可检出
        assert "click" not in s2.capabilities[0].actions
        assert "click" in s1.capabilities[0].actions

    def test_missing_skill_md_raises(self, tmp_path):
        with pytest.raises(ModelError, match="not found"):
            collect_browser_snapshot(
                product_version="web", snapshot_id="w-missing",
                skill_md_path=tmp_path / "absent.md",
            )


class TestMappingWiring:
    @edr_wd_repo
    def test_real_snapshot_binds_atomic_steps(self):
        """端到端：真实快照 → MappingStage 确定性绑定（§6.4）。"""
        snap, _ = collect_edr_wd_snapshot(
            product_version="5.3.0", snapshot_id="p4-e2e",
            observed_at="2026-11-08T00:00:00+00:00",
        )
        steps = [
            AtomicIR(step_id="S1", source_step=1, action="connect",
                     target={"semantic_name": "HiSec 主窗口",
                             "locator_status": "runtime_resolution"},
                     parameters={}, depends_on=[], evidence_refs=["E1"]),
            AtomicIR(step_id="S2", source_step=2, action="click",
                     target={"semantic_name": "设置按钮",
                             "locator_status": "runtime_resolution"},
                     parameters={}, depends_on=["S1"], evidence_refs=["E2"]),
        ]
        result = MappingStage(snap, "2026-11-08T00:00:01+00:00").run(steps, "windows")
        assert result["ok"] is True
        assert result["status"] == "bound"
        assert len(result["bound_steps"]) == 2
        binding = result["bound_steps"][1].binding
        assert binding.tool.startswith("edr-wd:")
        assert binding.snapshot_hash == snap.content_hash

    @edr_wd_repo
    def test_unmapped_atomic_action_unsupported(self):
        """clear 在 edr-wd 目录无对应声明 → unsupported，不静默绑定。"""
        assert "clear" not in (collect_edr_wd_snapshot(
            product_version="5.3.0", snapshot_id="p4-u",
            observed_at="2026-11-08T00:00:00+00:00",
        )[0].capabilities[0].actions)
        snap, _ = collect_edr_wd_snapshot(
            product_version="5.3.0", snapshot_id="p4-u2",
            observed_at="2026-11-08T00:00:00+00:00",
        )
        steps = [AtomicIR(step_id="S1", source_step=1, action="clear",
                          target={"semantic_name": "输入框",
                                  "locator_status": "runtime_resolution"},
                          parameters={}, depends_on=[], evidence_refs=["E1"])]
        result = MappingStage(snap, "2026-11-08T00:00:01+00:00").run(steps, "windows")
        assert result["status"] == "unsupported"
        assert result["unsupported_steps"] == ["S1"]
        assert any(d["code"] == "unsupported" for d in result["diagnostics"])

    @edr_wd_repo
    def test_platform_mismatch_rejected_with_real_snapshot(self):
        snap, _ = collect_edr_wd_snapshot(
            product_version="5.3.0", snapshot_id="p4-pm",
            observed_at="2026-11-08T00:00:00+00:00",
        )
        steps = [AtomicIR(step_id="S1", source_step=1, action="click",
                          target={"semantic_name": "x",
                                  "locator_status": "runtime_resolution"},
                          parameters={}, depends_on=[], evidence_refs=["E1"])]
        result = MappingStage(snap, "2026-11-08T00:00:01+00:00").run(steps, "web")
        assert result["ok"] is False
        assert any(d["code"] == "platform_mismatch" for d in result["diagnostics"])


class TestCatalogPayloadNegative:
    def test_payload_digest_required(self, tmp_path):
        repo = _make_fake_edr_wd(tmp_path)
        payload = _edw_catalog_payload(repo)
        assert payload["catalog_digest"].startswith("sha256:")
        assert payload["actions"][0]["tool_name"] == "connect"


def test_action_maps_only_declared_semantics():
    """映射表动作必须在 AtomicIR 域内（不虚构枚举值）。"""
    for tool, acts in EDW_ACTION_MAP.items():
        for act in acts:
            assert act in ATOMIC_ACTIONS, (tool, act)
    for tool, acts in BROWSER_ACTION_MAP.items():
        for act in acts:
            assert act in ATOMIC_ACTIONS, (tool, act)
