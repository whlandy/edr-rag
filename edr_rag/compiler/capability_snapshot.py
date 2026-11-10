"""CapabilitySnapshot 真实采集雏形（design.md §4 / §8，P4 第一批次）。

从真实只读来源构造 CapabilitySnapshot（schema hash 绑定、观察时间记录），
编译侧产物，来源仓库只读：

- edr-wd：`target/action_catalog/`（ACTIONS_V1 + CATALOG_VERSION +
  catalog_digest，机器可读单一事实源）→ windows 平台 ToolCapability；
  action_code（A001…）与 AtomicIR 语义动作的对应在 EDW_ACTION_MAP 声明，
  仅映射目录描述中真实声明的能力，不虚构。
- browser-use（本机 browser-automation SKILL.md 工具表）→ web 平台
  ToolCapability；SKILL.md 以原始字节 sha256 绑定。

采集输出为 CapabilitySnapshot（不可变，content_hash 为绑定锚）。任何
来源缺失/解析失败/映射外工具一律 raise，不静默降级（unknown 不算通过）。
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from edr_rag.knowledge.models import CapabilitySnapshot, ModelError, ToolCapability

EDR_WD_REPO = Path("/Users/edr-test/ai-projects/edr-wd")
BROWSER_SKILL_MD = Path("/Users/edr-test/.hermes/skills/browser-automation/SKILL.md")

# edr-wd action_catalog 实际声明能力 → AtomicIR 语义动作（§6.4 域枚举）。
# 每一项依据 ACTIONS_V1 的 tool_name/description 真实存在；未列出 = 未声明。
EDW_ACTION_MAP: dict[str, tuple[str, ...]] = {
    "connect": ("connect",),
    "activate_app": ("connect",),          # bring-to-foreground，目标定位语义
    "activate_edr": ("connect",),          # HiSec 专用 bring-to-foreground（同 activate_app 类）
    "click": ("click",),
    "click_target": ("click",),
    "click_at": ("click",),
    "click_window_at": ("click",),
    "double_click_at": ("click",),
    "right_click_at": ("click",),
    "middle_click_at": ("click",),
    "type_text": ("input",),
    "select": ("select",),
    "scroll": ("scroll",),
    "wait_window": ("wait",),
    "is_window_open": ("assert",),
    "get_window_lock": ("assert",),
    "verify_window_lock": ("assert",),
    "screenshot": ("screenshot",),
    "get_text": ("assert",),               # 只读文本观测 → 断言取证语义
}

# browser SKILL.md 工具表（浏览器目标 → AtomicIR web 语义动作）
BROWSER_ACTION_MAP: dict[str, tuple[str, ...]] = {
    "browser_navigate": ("navigate",),
    "browser_click": ("click",),
    "browser_type": ("input",),
    "browser_select_option": ("select",),
    "browser_get_text": ("assert",),
    "browser_snapshot": ("assert",),
    "browser_press": ("input",),
}

_TOOL_ROW_RE = re.compile(r"^\|\s*`([a-z_]+)`\s*\|")


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _edw_catalog_payload(repo_root: Path = EDR_WD_REPO) -> dict[str, Any]:
    """从真实 edr-wd 仓库经其自身 digest 机制导出目录 JSON（只读，子进程 -I）。"""
    if not repo_root.is_dir():
        raise ModelError(f"edr-wd repo not found: {repo_root}")
    code = (
        "import sys, json;"
        f"sys.path.insert(0, {str(repo_root)!r});"
        "from target.action_catalog.actions_v1 import ACTIONS_V1;"
        "from target.action_catalog.digest import catalog_digest;"
        "from target.action_catalog.registry import CATALOG_VERSION;"
        "print(json.dumps({'catalog_version': CATALOG_VERSION,"
        "'catalog_digest': catalog_digest(ACTIONS_V1),"
        "'actions': [{'action_id': s.action_id, 'action_code': s.action_code,"
        "'tool_name': s.tool_name, 'description': s.description,"
        "'category': s.category, 'input_schema': dict(s.input_schema),"
        "'result_schema': dict(s.result_schema), 'backends': list(s.backends),"
        "'side_effect': s.side_effect, 'risk': s.risk,"
        "'execution_provider': s.execution_provider} for s in ACTIONS_V1]}"
        ", sort_keys=True))"
    )
    proc = subprocess.run(
        [sys.executable, "-I", "-c", code],
        capture_output=True, text=True, timeout=60,
    )
    if proc.returncode != 0:
        raise ModelError(f"edr-wd catalog export failed: {proc.stderr.strip()[:300]}")
    try:
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as exc:
        raise ModelError(f"edr-wd catalog export unparsable: {exc}") from exc
    if payload.get("catalog_digest", "").startswith("sha256:") and payload.get("actions"):
        return payload
    raise ModelError("edr-wd catalog export missing digest or actions")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect_edr_wd_snapshot(
    *,
    product_version: str,
    snapshot_id: str,
    platform_backend: str = "windows_pywinauto",
    observed_at: str | None = None,
    repo_root: Path = EDR_WD_REPO,
) -> tuple[CapabilitySnapshot, dict[str, Any]]:
    """真实 edr-wd 目录 → windows CapabilitySnapshot。

    返回 (snapshot, provenance)；provenance 记录 catalog_version、catalog
    digest、backend 过滤、动作映射来源，用于审核包。未映射工具保留在
    provenance（未声明能力不进入 snapshot，可被 unsupported 门禁检出）。
    """
    if platform_backend not in ("windows_pywinauto", "macos_accessibility"):
        raise ModelError(f"unknown platform_backend: {platform_backend!r}")
    platform = "windows" if platform_backend == "windows_pywinauto" else "macos"
    when = observed_at or _now_iso()
    payload = _edw_catalog_payload(repo_root)

    backend_tools = {
        a["tool_name"]: a for a in payload["actions"] if platform_backend in a["backends"]
    }
    mapped_actions: list[str] = []
    unmapped: list[str] = sorted(t for t in backend_tools if t not in EDW_ACTION_MAP)
    for tool, acts in EDW_ACTION_MAP.items():
        if tool not in backend_tools:
            continue
        for act in acts:
            if act not in mapped_actions:
                mapped_actions.append(act)
    if not mapped_actions:
        raise ModelError("edr-wd backend has no tools mapped to AtomicIR actions")

    # tool_schema_hash：目录内容指纹（edr-wd 自身 canonical digest，§8 严格核对）
    cap = ToolCapability(
        tool=f"edr-wd:{payload['catalog_version']}:{platform_backend}",
        tool_version=payload["catalog_version"],
        tool_schema_hash=payload["catalog_digest"],
        platform=platform,
        actions=tuple(sorted(mapped_actions)),
        parameter_schema={
            "source": "target/action_catalog/actions_v1.py",
            "catalog_digest": payload["catalog_digest"],
            "backend": platform_backend,
            "tool_count": len(backend_tools),
            "unmapped_tools": unmapped,
        },
        observed_at=when,
    )
    provenance = {
        "source": "edr-wd target/action_catalog",
        "repo": str(repo_root),
        "catalog_version": payload["catalog_version"],
        "catalog_digest": payload["catalog_digest"],
        "platform_backend": platform_backend,
        "mapped_tools": sorted(t for t in backend_tools if t in EDW_ACTION_MAP),
        "unmapped_tools": unmapped,
        "action_map": "edr_rag.compiler.capability_snapshot.EDW_ACTION_MAP",
    }
    snapshot = CapabilitySnapshot(
        snapshot_id=snapshot_id, platform=platform,
        product_version=product_version, capabilities=[cap],
        observed_at=when,
        notes=f"collected from edr-wd catalog {payload['catalog_version']}",
    )
    return snapshot, provenance


def collect_browser_snapshot(
    *,
    product_version: str,
    snapshot_id: str,
    observed_at: str | None = None,
    skill_md_path: Path = BROWSER_SKILL_MD,
) -> tuple[CapabilitySnapshot, dict[str, Any]]:
    """browser-automation SKILL.md 工具表 → web CapabilitySnapshot。

    SKILL.md 原始字节 sha256 绑定（tool_schema_hash），观察时间独立记录；
    表中工具缺失/结构变化时 raise，不猜。
    """
    if not skill_md_path.is_file():
        raise ModelError(f"browser SKILL.md not found: {skill_md_path}")
    when = observed_at or _now_iso()
    raw = skill_md_path.read_bytes()
    text = raw.decode("utf-8")

    tools: list[str] = []
    for line in text.splitlines():
        m = _TOOL_ROW_RE.match(line)
        if m:
            tools.append(m.group(1))
    tools = sorted(set(tools))
    known = [t for t in tools if t in BROWSER_ACTION_MAP]
    if not known:
        raise ModelError(f"browser SKILL.md tool table has no mapped tools: {tools}")

    actions: list[str] = []
    for t in known:
        for act in BROWSER_ACTION_MAP[t]:
            if act not in actions:
                actions.append(act)

    file_hash = "sha256:" + _sha256_file(skill_md_path)
    cap = ToolCapability(
        tool="browser-use:playwright-mcp",
        tool_version="SKILL.md@" + file_hash[:19],
        tool_schema_hash=file_hash,
        platform="web",
        actions=tuple(sorted(actions)),
        parameter_schema={
            "source": "SKILL.md tool table",
            "tools_declared": known,
            "tools_unmapped": sorted(t for t in tools if t not in BROWSER_ACTION_MAP),
        },
        observed_at=when,
    )
    provenance = {
        "source": "browser-automation SKILL.md tool table",
        "path": str(skill_md_path),
        "file_sha256": file_hash,
        "tools_declared": known,
        "tools_unmapped": sorted(t for t in tools if t not in BROWSER_ACTION_MAP),
        "action_map": "edr_rag.compiler.capability_snapshot.BROWSER_ACTION_MAP",
    }
    snapshot = CapabilitySnapshot(
        snapshot_id=snapshot_id, platform="web",
        product_version=product_version, capabilities=[cap],
        observed_at=when,
        notes="collected from browser-automation SKILL.md",
    )
    return snapshot, provenance
