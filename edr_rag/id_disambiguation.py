"""ingest 层 ID 碰撞根因消歧：碰撞组范围限定的身份后缀。

设计依据（doc/UNIFIED-KNOWLEDGE-BASE-DESIGN.md §5）：
  EvidenceUnit 身份 = 来源 + 定位；同 ID 异内容跨文件变体共享一个 ID
  违反该契约（同一 ID 语义下藏多个不同事实面）。

方案（决策 D-20261112-1824-1，范围限定的 A 方案）：
  - ingest 写盘前对 sections 做碰撞组分析：
      同 ID + 语义内容（除 source_ref/links/images 外逐字段）相同 → 保留原 ID
        （读取侧 _dedupe_identical_id_rows 已收敛此类，等价无信息损失）
      同 ID + 语义内容不同 → 每个成员 ID 追加 "@<fileslug>" 消歧后缀
        （fileslug = source_ref.relative_path 去扩展名后 slug 化）
  - 未碰撞 ID（anquan 1444/1474、duli 1770/1788）零变化。
  - 下游 procedure/chunk/node ID 由 section ID 派生，自动级联消歧。
  - [1617] 读取层 #dupN 消歧保留为纵深防御（构造期上游不经读取层）。

前置确认（本轮程序化核验）：
  - 碰撞全部为跨文件（same-file 重复 0/17/11 组）。
  - 评测集 gold 仅 a-03/a-04 引用碰撞 ID（manual.section.登录_管理面…），
    source_file=zh-cn_topic_0180928815.html —— ID 变更后需同步评测集 gold，
    属下轮重建批次范围（含全量回归），本轮只交付 ingest 层实现与离线快照验证。

锁定测试 test_semantic_id_not_derived_from_path 语义不变：消歧只作用于
「同批内同 ID 异内容」碰撞组，两个不同 section_path 仍得不同 ID；
单文件语义 ID 与既有格式逐字节相同。
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Optional

# 与 knowledge/document_ingest.py 的等价口径一致：身份等价性忽略定位字段
_EQUIV_EXCLUDED_KEYS = frozenset({"source_ref", "links", "images"})


def _equivalence_dict(section: dict) -> dict:
    return {k: v for k, v in section.items() if k not in _EQUIV_EXCLUDED_KEYS}


def _equivalence_key(section: dict) -> str:
    return json.dumps(_equivalence_dict(section), sort_keys=True, ensure_ascii=False)


def _file_slug(relative_path: str) -> str:
    """source_ref.relative_path → 确定性短 slug（无扩展名）。"""
    name = relative_path.replace("\\", "/").rsplit("/", 1)[-1]
    if "." in name:
        name = name.rsplit(".", 1)[0]
    return name


def disambiguate_collision_ids(sections: list[dict]) -> list[dict]:
    """对同 ID 异内容的碰撞组成员追加 @<fileslug> 后缀。

    - 同 ID 等价组（含去 source_ref 等价）：ID 不变。
    - 同 ID 异内容组：全部成员（含首现）都加后缀——组内无「规范代表」
      可信手选，行序不可作为身份依据（[1105] 实证：哪个变体存活取决于
      行序，属缺陷形状本身）。
    - 唯一 ID：零变化。
    - 同批内 fileslug 再碰撞（同名不同子目录）：序号按「完整来源定位
      （relative_path）字典序」确定性分配，不随输入行序变化——来源→ID
      映射是输入顺序无关的纯函数。

    输入 sections 不被修改（返回新 list、新 dict）。
    """
    if not sections:
        return []

    # 1. 碰撞组分析：id → [(index, equiv_key)]
    by_id: dict[str, list[tuple[int, str]]] = {}
    for i, sec in enumerate(sections):
        sid = sec.get("id", "")
        by_id.setdefault(sid, []).append((i, _equivalence_key(sec)))

    # 2. 对每个异内容碰撞组计算成员 → 新 ID
    rename: dict[int, str] = {}
    for sid, members in by_id.items():
        if len(members) <= 1:
            continue
        distinct = {k for _, k in members}
        if len(distinct) <= 1:
            continue  # 等价组：读取侧去重已收敛，ID 不变
        # fileslug 再碰撞时按完整 relative_path 字典序定序：同名后缀
        # 分配与输入顺序解耦（同一来源集合 → 同一来源→ID 映射）。
        ordered = sorted(
            members,
            key=lambda m: (
                sections[m[0]].get("source_ref", {}).get("relative_path", ""),
                m[0],
            ),
        )
        slug_seen: dict[str, int] = {}
        for idx, _key in ordered:
            rel = sections[idx].get("source_ref", {}).get("relative_path", "")
            slug = _file_slug(rel)
            seen = slug_seen.get(slug, 0)
            slug_seen[slug] = seen + 1
            if seen:
                slug = f"{slug}.{seen + 1}"
            rename[idx] = f"{sid}@{slug}"

    if not rename:
        return list(sections)

    out: list[dict] = []
    for i, sec in enumerate(sections):
        if i in rename:
            new_sec = copy.deepcopy(sec)
            new_sec["id"] = rename[i]
            # 消歧来源显式记录：ID 派生依据不可隐式（§5 provenance）
            new_sec.setdefault("metadata", {})["id_disambiguation"] = {
                "original_id": sec["id"],
                "reason": "same_id_different_content",
                "source_file": sec.get("source_ref", {}).get("relative_path", ""),
            }
            out.append(new_sec)
        else:
            out.append(sec)
    return out


def disambiguate_sections_file(sections_path: Path) -> Optional[int]:
    """对已写盘的 sections.jsonl 原位消歧（重建批次使用）。

    Returns: 重命名行数；文件不存在返回 None。原子写，失败不污染。
    """
    path = Path(sections_path)
    if not path.exists():
        return None
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    renamed_rows = disambiguate_collision_ids(rows)
    renamed = sum(
        1 for a, b in zip(rows, renamed_rows) if a.get("id") != b.get("id")
    )
    if renamed:
        from edr_rag.atomic_io import write_jsonl

        write_jsonl(path, renamed_rows)
    return renamed


# ---------- P1 级联：procedure id 二次碰撞 ----------

# procedure id 由 section_path+title 派生（非 section id），存在两类
# section 消歧覆盖不到的碰撞：
#   1. 源 section id 带 @后缀 → proc id 继承（extract 侧级联）。
#   2. 不同 section id 经 slug 折叠出同一 proc id（实证：duli
#      「登录 业务面」空格被 _slug_path 删除 vs「登录_业务面」下划线
#      保留 → section id 不同，proc id 相同）。
# 规则与 section 消歧同构：同派生 id 组内，(source.section_id, 语义字段)
# 等价的成员保留原 id（读取侧去重收敛）；不等价的全部成员加
# @<fileslug(source.file)>。

_PROC_EQUIV_EXCLUDED = frozenset({"source", "id"})


def _procedure_equivalence_key(proc: dict) -> str:
    body = {k: v for k, v in proc.items() if k not in _PROC_EQUIV_EXCLUDED}
    src = proc.get("source") or {}
    return json.dumps(
        {"body": body, "section_id": src.get("section_id", "")},
        sort_keys=True, ensure_ascii=False,
    )


def disambiguate_procedure_ids(procedures: list[dict]) -> list[dict]:
    """对同派生 id 异 (section_id, 语义内容) 的 procedure 加 @<fileslug>。

    输入不被修改；返回新 list。等价组、唯一 id 零变化。
    fileslug 再碰撞时序号按完整 source.file 字典序确定性分配（与输入
    顺序解耦，与 sections 侧同规则）。
    """
    if not procedures:
        return []

    by_id: dict[str, list[tuple[int, str]]] = {}
    for i, proc in enumerate(procedures):
        pid = proc.get("id", "")
        by_id.setdefault(pid, []).append((i, _procedure_equivalence_key(proc)))

    rename: dict[int, str] = {}
    for pid, members in by_id.items():
        if len(members) <= 1:
            continue
        if len({k for _, k in members}) <= 1:
            continue
        ordered = sorted(
            members,
            key=lambda m: (
                (procedures[m[0]].get("source") or {}).get("file", ""),
                m[0],
            ),
        )
        slug_seen: dict[str, int] = {}
        for idx, _key in ordered:
            rel = (procedures[idx].get("source") or {}).get("file", "")
            slug = _file_slug(rel)
            seen = slug_seen.get(slug, 0)
            slug_seen[slug] = seen + 1
            if seen:
                slug = f"{slug}.{seen + 1}"
            rename[idx] = f"{pid}@{slug}"

    if not rename:
        return list(procedures)

    out: list[dict] = []
    for i, proc in enumerate(procedures):
        if i in rename:
            new_proc = copy.deepcopy(proc)
            new_proc["id"] = rename[i]
            new_proc.setdefault("metadata", {})["id_disambiguation"] = {
                "original_id": proc["id"],
                "reason": "same_derived_id_different_source",
            }
            out.append(new_proc)
        else:
            out.append(proc)
    return out
