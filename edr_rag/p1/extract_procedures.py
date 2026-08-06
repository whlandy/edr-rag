"""P1 procedure extraction — deterministic, heuristic-based.

Input:  normalized/sections.jsonl (P0 ManualSection records, read as dicts)
Output: normalized/procedures.jsonl + normalized/chunks.jsonl

Strategy (from design §5):
  HTML structure -> headings/lists/tables -> candidate Procedure

This module does NOT call an LLM. It applies deterministic heuristics:
  - Section looks like a procedure if its title/path/text contains action
    verbs or heading markers (操作步骤, 步骤, 配置, 添加, 删除, 启用/禁用,
    重置, 导出, 新建 ...), or if the body contains ordered lists (numbered steps).
  - Ordered lists (<ol>) that survived HTML cleanup as "N. step" become steps.
  - Keyword blocks (注意, 警告, 重要, 前置条件, 恢复) map to warnings /
    preconditions / recovery.

The output is intended to be REVIEWABLE, not perfect. All procedures are
tagged `status: candidate` by default; a person or LLM pass can promote them.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

from .schema import (
    ChunkSource,
    Procedure,
    ProcedureSource,
    ProcedureStep,
    RetrievalChunk,
    make_chunk_id,
    make_procedure_id,
)

logger = logging.getLogger(__name__)

# ---------- Heuristic keyword sets (configurable) ----------

# Verbs that suggest a section is a procedure
PROCEDURE_VERBS = {
    "配置", "添加", "删除", "启用", "禁用", "重置", "导出", "导入",
    "新建", "创建", "修改", "设置", "安装", "卸载", "启动", "停止",
    "重启", "恢复", "备份", "更新", "升级", "连接", "断开", "注册",
    "取消", "保存", "提交", "下载", "上传",
}

# Heading markers that strongly indicate a procedure
PROCEDURE_HEADINGS = {
    "操作步骤", "操作说明", "步骤", "配置步骤", "配置方法",
    "安装步骤", "使用方法", "操作流程", "常见操作",
}

# Markers for warnings / important notes
WARNING_MARKERS = {"注意", "警告", "重要", "提示", "小心", "危险"}

# Markers for preconditions
PRECONDITION_MARKERS = {"前置条件", "前提条件", "前置", "前提", "准备", "要求"}

# Markers for recovery
RECOVERY_MARKERS = {"恢复", "还原", "回滚", "如果失败", "如果连接失败", "故障排除", "排错"}

# Numbered list detection: lines like "1. ...", "1) ...", "Step 1 ...", "步骤1 ..."
_STEP_RE = re.compile(
    r"^\s*(?:"
    r"(\d+)[.、)．]\s*"              # 1. / 1) / 1、/ 1．
    r"|步骤\s*(\d+)\s*[:：]?\s*"     # 步骤1:
    r"|step\s*(\d+)\s*[:：.]?\s*"    # Step 1:
    r")",
    re.IGNORECASE,
)

# Line that looks like a standalone content line (has >=2 CJK chars or word)
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


# ---------- Detection ----------

def _looks_like_procedure(title: str, section_path: list[str], content: str) -> bool:
    """Heuristic: could this section be a procedure?"""
    # 1) Heading marker in title or any path component
    joined = " ".join(section_path + [title])
    for h in PROCEDURE_HEADINGS:
        if h in joined:
            return True

    # 2) Action verb in title or last path component
    last_path = section_path[-1] if section_path else title
    for v in PROCEDURE_VERBS:
        if v in title or v in last_path:
            return True

    # 3) Section body contains at least one numbered-list step AND starts
    #    with an imperative-ish action line
    lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
    steps = [ln for ln in lines if _STEP_RE.match(ln)]
    if len(steps) >= 2:
        return True

    return False


# ---------- Extraction ----------

def _extract_action_verb(text: str) -> str:
    """Extract a coarse action verb from a step instruction."""
    for v in ("点击", "选择", "输入", "打开", "关闭", "确认", "保存", "导出",
              "导入", "启用", "禁用", "添加", "删除", "设置", "配置", "新建",
              "启动", "停止", "重启", "跳转", "进入", "勾选", "取消"):
        if v in text:
            return v
    return ""


def _extract_target(text: str, verb: str) -> str:
    """A naive target extraction: text right after the verb."""
    if not verb:
        return ""
    idx = text.find(verb)
    if idx == -1:
        return ""
    rest = text[idx + len(verb):].lstrip(" ")
    # take up to first separator or CJK sentence end
    end = len(rest)
    for sep in ("，", "。", "；", ",", ".", "；", " "):
        pos = rest.find(sep)
        if pos != -1 and pos < end:
            end = pos
    return rest[:end].strip()


def _split_content_blocks(content: str) -> tuple[list[str], list[str], list[str], list[str]]:
    """Split section content into (steps, warnings, preconditions, recovery).

    Handles two content shapes:
      (a) line-oriented: steps on their own lines ("1. 打开网络设置")
      (b) flattened: single-line space-joined content where numbered
          steps appear inline ("... 1. 打开 ... 2. 选择 ...")

    The P0 `_clean_text` produces flattened content (space-joined), so we
    must also handle a regex over the whole string. This is heuristic and
    conservative — a malformed flattening yields fewer steps, never wrong ones.
    """
    steps: list[str] = []
    warnings: list[str] = []
    preconditions: list[str] = []
    recovery: list[str] = []

    lines = [ln.strip() for ln in content.splitlines() if ln.strip()]

    # --- Pass 1: line-oriented detection ---
    current_block: Optional[list] = None

    # Also handle the flattened case: walk sentence-like fragments.
    # If the whole content is one line but contains "N. " step markers,
    # split it into fragments on those markers so the loop sees them.
    # Use an UNANCHORED search here — the flattened content is a single
    # line with steps inline, so "^N." won't match mid-string.
    _INLINE_STEP = re.compile(r"\d+[.、)．]\s")
    if len(lines) == 1 and _INLINE_STEP.search(lines[0]):
        # Rebuild as pseudo-lines split on step boundaries
        pseudo_lines: list[str] = []
        frags = re.split(r"(?=\d+[.、)．]\s)", lines[0])
        for fr in frags:
            fr = fr.strip()
            if fr:
                pseudo_lines.append(fr)
        lines = pseudo_lines

    for ln in lines:
        # Detect numbered-step lines
        m = _STEP_RE.match(ln)
        if m:
            if current_block is not None and current_block is not steps:
                current_block = None
            current_block = steps
            step_text = _STEP_RE.sub("", ln).strip(" :：.-—–")
            if step_text and len(step_text) >= 2:
                steps.append(step_text)
            continue

        # Detect warning/important/note heading lines
        matched_meta = False
        for w in WARNING_MARKERS:
            if ln.startswith(w) or f"{w}：" in ln[:8]:
                current_block = warnings
                wtext = ln[len(w):].lstrip(" ：:，。")
                if wtext:
                    warnings.append(wtext)
                matched_meta = True
                break
        if matched_meta:
            continue

        # Detect precondition lines
        for p in PRECONDITION_MARKERS:
            if ln.startswith(p) or f"{p}：" in ln[:8]:
                current_block = preconditions
                ptext = ln[len(p):].lstrip(" ：:，。")
                if ptext:
                    preconditions.append(ptext)
                matched_meta = True
                break
        if matched_meta:
            continue

        # Detect recovery lines
        for r in RECOVERY_MARKERS:
            if r in ln:
                current_block = recovery
                recovery.append(ln)
                matched_meta = True
                break
        if matched_meta:
            continue

        # Continuation of current block
        if current_block is not None and current_block is not steps:
            current_block.append(ln)

    return steps, warnings, preconditions, recovery


# ---------- Procedure building ----------

def _build_operation(section_path: list[str], title: str) -> str:
    """Build a snake_case operation name from semantic identity."""
    parts = [p for p in (section_path + [title]) if p]
    if not parts:
        return "unknown"
    # take title (or last path component) as the operation root
    root = parts[-1]
    # strip common suffixes
    root = re.sub(r"(配置|设置|操作|方法|步骤|说明)$", "", root)
    return root.strip()


def extract_procedures_from_section(
    section: dict,
    *,
    product: str | None = None,
) -> list[Procedure]:
    """Extract 0..N Procedures from one ManualSection dict.

    Args:
        section: a P0 ManualSection as a dict (from sections.jsonl).
        product: optional product name override; defaults to the section's.

    Returns:
        List of candidate Procedures. Empty if the section does not look like
        a procedure.
    """
    manual_id = section.get("manual_id", "")
    product = product or section.get("product", "")
    title = section.get("title", "")
    section_path = section.get("section_path", [])
    content = section.get("content", "")
    source_ref = section.get("source_ref", {})
    rel_file = source_ref.get("relative_path", "")

    if not _looks_like_procedure(title, section_path, content):
        return []

    steps_raw, warnings, preconditions, recovery = _split_content_blocks(content)

    # If no steps were found but section is marked as procedure, try to
    # extract steps from unstructured sentence boundaries (naive).
    if not steps_raw:
        sentences = _split_sentences(content)
        steps_raw = [s for s in sentences if _looks_like_step_sentence(s)][:20]

    # Guard: a procedure with zero steps is not useful
    if not steps_raw:
        logger.debug("Section %s marked as procedure but no steps found — skipping",
                     section.get("id"))
        return []

    procedure_steps = [
        ProcedureStep(
            order=i,
            instruction=raw,
            target=_extract_target(raw, _extract_action_verb(raw)),
            action_hint=_map_action_hint(raw),
            source_instruction=raw,
        )
        for i, raw in enumerate(steps_raw, start=1)
    ]

    operation = _build_operation(section_path, title)
    procedure_id = make_procedure_id(f"{'.'.join(_slug_path(section_path))}.{operation}")

    # Confidence heuristics
    confidence = _compute_confidence(section, procedure_steps, warnings, preconditions)

    proc = Procedure(
        id=procedure_id,
        manual_id=manual_id,
        product=product,
        operation=operation,
        title=title,
        section_path=section_path,
        source=ProcedureSource(
            file=rel_file,
            section_id=section.get("id", ""),
            anchor=source_ref.get("anchor"),
        ),
        preconditions=preconditions,
        steps=procedure_steps,
        warnings=warnings,
        recovery=recovery,
        confidence=confidence,
        status="candidate",
    )
    return [proc]


def _slug_path(path: list[str]) -> list[str]:
    """Convert section_path components to slug form (lowercase, no spaces)."""
    out = []
    for p in path:
        s = re.sub(r"\s+", "", p).lower()
        if s:
            out.append(s)
    return out


def _compute_confidence(
    section: dict,
    steps: list[ProcedureStep],
    warnings: list[str],
    preconditions: list[str],
) -> float:
    """Heuristic confidence: 0.5-0.95."""
    c = 0.6
    # more steps = higher confidence (stronger procedure signal)
    c += min(len(steps) * 0.05, 0.2)
    if warnings:
        c += 0.05
    if preconditions:
        c += 0.05
    if section.get("headings"):
        c += 0.05
    return round(min(c, 0.95), 2)


# ---------- Step sentence heuristics ----------

def _split_sentences(content: str) -> list[str]:
    """Split content into sentences on CJK / ASCII sentence boundaries."""
    parts = re.split(r"(?<=[。！？!?；;])\s*", content)
    return [p.strip() for p in parts if len(p.strip()) >= 4]


def _looks_like_step_sentence(s: str) -> bool:
    """Heuristic: a sentence is a step if it starts with an action verb."""
    for v in ("点击", "选择", "输入", "打开", "关闭", "确认", "保存",
              "导出", "导入", "启用", "禁用", "添加", "删除", "设置",
              "配置", "新建", "启动", "停止", "重启", "进入", "勾选"):
        if s.startswith(v):
            return True
    return False


# ---------- Action hint mapping ----------

_ACTION_HINT_MAP = [
    ("点击", "click"),
    ("选择", "select"),
    ("输入", "input_text"),
    ("勾选", "check"),
    ("打开", "open_window"),
    ("关闭", "close"),
    ("启动", "start"),
    ("停止", "stop"),
    ("重启", "restart"),
    ("启用", "enable"),
    ("禁用", "disable"),
    ("添加", "add"),
    ("删除", "delete"),
    ("确认", "confirm"),
    ("保存", "save"),
    ("导出", "export"),
    ("导入", "import"),
    ("取消", "cancel"),
    ("进入", "navigate"),
]


def _map_action_hint(instruction: str) -> str:
    for verb, hint in _ACTION_HINT_MAP:
        if verb in instruction:
            return hint
    return ""


# ---------- Chunk building ----------

def build_chunks(
    section: dict,
    procedures: list[Procedure],
) -> list[RetrievalChunk]:
    """Build retrieval chunks from a section and its extracted procedures.

    Produces:
      - one "section" chunk from the section itself
      - one "procedure" chunk per procedure
      - one "step" chunk per procedure step (fine-grained)
    """
    chunks: list[RetrievalChunk] = []
    manual_id = section.get("manual_id", "")
    section_id = section.get("id", "")
    rel_file = section.get("source_ref", {}).get("relative_path", "")

    # Section chunk
    section_text = section.get("content", "")
    if section_text.strip():
        chunks.append(
            RetrievalChunk(
                id=make_chunk_id("section", section_id),
                manual_id=manual_id,
                type="section",
                text=section_text,
                source=ChunkSource(section_id=section_id, file=rel_file),
                metadata={
                    "product": section.get("product", ""),
                    "title": section.get("title", ""),
                },
            )
        )

    for proc in procedures:
        # Procedure chunk
        proc_text = _procedure_to_text(proc)
        chunks.append(
            RetrievalChunk(
                id=make_chunk_id("procedure", proc.id.split(".", 1)[-1]),
                manual_id=manual_id,
                type="procedure",
                text=proc_text,
                source=ChunkSource(
                    section_id=section_id,
                    procedure_id=proc.id,
                    file=rel_file,
                ),
                metadata={
                    "product": proc.product,
                    "title": proc.title,
                    "status": proc.status,
                },
            )
        )
        # Step chunks
        for step in proc.steps:
            chunks.append(
                RetrievalChunk(
                    id=make_chunk_id(
                        "step", f"{proc.id.split('.',1)[-1]}.s{step.order}"
                    ),
                    manual_id=manual_id,
                    type="step",
                    text=step.instruction,
                    source=ChunkSource(
                        section_id=section_id,
                        procedure_id=proc.id,
                        file=rel_file,
                    ),
                    metadata={
                        "product": proc.product,
                        "order": step.order,
                        "action_hint": step.action_hint,
                    },
                )
            )

    return chunks


def _procedure_to_text(proc: Procedure) -> str:
    """Render a procedure as flat searchable text."""
    lines = [proc.title]
    if proc.preconditions:
        lines.append("前置条件: " + "；".join(proc.preconditions))
    for step in proc.steps:
        lines.append(f"{step.order}. {step.instruction}")
    if proc.warnings:
        lines.append("注意: " + "；".join(proc.warnings))
    if proc.recovery:
        lines.append("恢复: " + "；".join(proc.recovery))
    return "\n".join(lines)


# ---------- File-level orchestrators ----------

def extract_procedures_from_file(
    sections_path: Path,
    *,
    product: str | None = None,
) -> tuple[list[Procedure], list[RetrievalChunk]]:
    """Run P1 extraction over a sections.jsonl file.

    Returns (procedures, chunks). Does NOT write any files.
    """
    import json

    procedures: list[Procedure] = []
    chunks: list[RetrievalChunk] = []

    with Path(sections_path).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            section = json.loads(line)
            procs = extract_procedures_from_section(section, product=product)
            procedures.extend(procs)
            chunks.extend(build_chunks(section, procs))

    return procedures, chunks


def write_p1_artifacts(
    workdir: Path,
    procedures: list[Procedure],
    chunks: list[RetrievalChunk],
    sections: list[dict],
    *,
    manual_id: str,
    product: str,
) -> Path:
    """Write procedures.jsonl, chunks.jsonl, and p1_metadata.json under workdir.

    Returns the path to p1_metadata.json.
    """
    import json
    from datetime import datetime, timezone

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    with (workdir / "procedures.jsonl").open("w", encoding="utf-8") as f:
        for p in procedures:
            f.write(json.dumps(p.to_jsonl(), ensure_ascii=False) + "\n")

    with (workdir / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.to_jsonl(), ensure_ascii=False) + "\n")

    meta = {
        "manual_id": manual_id,
        "product": product,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "section_count": len(sections),
        "procedure_count": len(procedures),
        "chunk_count": len(chunks),
        "extraction_version": "p1",
    }
    meta_path = workdir / "p1_metadata.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta_path
