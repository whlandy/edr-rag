"""索引版本灰度切换 / 回滚 (P3-5).

在 P0-5 的「内容 hash 过期检测」(index_manifest) 与 P3-1 的增量索引
(VectorIndex.index_records_incremental) 之上，提供**版本化**的索引管理：

  - 每次构建 (全量或增量) 后，为某个 corpus 捕获一个版本快照，记录:
      version_id    : 版本号 (自增整数)。
      corpus        : 所属语料 (chunks / procedures / actions ...)。
      source_hash   : 当时的源头 hash (源自 P0 manifest 或调用方传入)。
      records_hash  : 整批语料的指纹 (对全部记录内容做 sha1)。
      fingerprints  : id -> 内容指纹 的映射，供下一轮增量检索变更部分使用。
      built_at      : 构建时间 (UTC)。
      active        : 当前是否激活。

  所有版本集中持久化到 <workdir>/index_versions.json (原子写入)。同一
  corpus 可有多个历史版本；新版本捕获后即成为 active，但旧版本仍保留在
  注册表中，因此可以随时「灰度切换」——回滚到任一历史版本 (activate)，
  让检索/消费侧切回旧索引，实现故障时的安全回退。

  与 index_manifest 的关系：index_manifest 记录的是产物的内容 hash 是否
  过期 (read 时校验)；index_versions 记录的是**版本历史**，用于「切到哪个
  版本」的决定。二者互补——manifest 告诉你「要不要重建」，这里告诉你
  「重建后的版本能否回退」。

本模块只依赖 stdlib + edr_rag.atomic_io，主链路 (keyword/BM25) 不依赖它；
缺版本文件时各函数返回空/None，绝不抛异常。
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .atomic_io import atomically_replace_file

logger = logging.getLogger(__name__)

#: 版本注册表文件名。
_VERSIONS_FILE = "index_versions.json"

#: 版本注册表格式版本。
VERSIONS_SCHEMA_VERSION = 1


@dataclass
class IndexVersion:
    """一个 corpus 的一次索引版本快照。"""

    version_id: int
    corpus: str
    built_at: str                       # UTC ISO-8601
    source_hash: str = ""               # 构建时的源头 hash
    records_hash: str = ""              # 整批语料内容指纹 (sha1)
    active: bool = True
    fingerprints: dict[str, str] = field(default_factory=dict)  # id -> 内容指纹

    def to_json(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_json(d: dict) -> "IndexVersion":
        return IndexVersion(
            version_id=int(d.get("version_id", 0)),
            corpus=d.get("corpus", ""),
            built_at=d.get("built_at", ""),
            source_hash=d.get("source_hash", ""),
            records_hash=d.get("records_hash", ""),
            active=bool(d.get("active", False)),
            fingerprints=d.get("fingerprints", {}) or {},
        )


# ---------- 文件读写 ----------

def versions_path(workdir: Path) -> Path:
    return Path(workdir) / _VERSIONS_FILE


def load_versions(workdir: Path) -> dict[str, list[IndexVersion]]:
    """读取版本注册表：{corpus: [IndexVersion, ...]} (版本号升序)。

    文件不存在 / 解析失败时返回空 dict；坏文件仅记 warning，不抛异常。
    """
    p = versions_path(workdir)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, ValueError) as e:
        logger.warning("index versions %s unreadable: %s", p, e)
        return {}
    out: dict[str, list[IndexVersion]] = {}
    for corpus, versions in (data.get("versions") or {}).items():
        out[corpus] = [
            IndexVersion.from_json(v)
            for v in versions
            if isinstance(v, dict)
        ]
    return out


def save_versions(workdir: Path, registry: dict[str, list[IndexVersion]]) -> Path:
    """原子写回版本注册表，返回其路径。"""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": VERSIONS_SCHEMA_VERSION,
        "versions": {
            corpus: [v.to_json() for v in versions]
            for corpus, versions in registry.items()
        },
    }

    def _write(f):
        f.write(json.dumps(payload, ensure_ascii=False, indent=2))

    p = versions_path(workdir)
    atomically_replace_file(p, _write)
    return p


# ---------- 指纹工具 ----------

def compute_records_hash(records: list[dict], text_field: str = "_searchable") -> str:
    """对整批记录的可检索文本计算指纹 (sha1)，作为整批语料的版本指纹。

    与逐条指纹 (index_versioning 用 _record_fingerprint 的语义) 对齐，
    文本取 _searchable / text / content 中的首个非空字段。
    """
    h = hashlib.sha1()
    texts = []
    for r in records:
        texts.append(
            r.get(text_field)
            or r.get("text")
            or r.get("content")
            or ""
        )
    # 先排序再拼接：整批指纹与记录顺序无关 (语料重排不算内容变化)。
    for text in sorted(texts):
        h.update(text.encode("utf-8", "ignore"))
    return h.hexdigest()


def _record_fingerprint(r: dict, text_field: str) -> str:
    text = r.get(text_field) or r.get("text") or r.get("content") or ""
    return hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()


def compute_fingerprints(
    records: list[dict],
    id_field: str = "id",
    text_field: str = "_searchable",
) -> dict[str, str]:
    """为一批记录构建 {id: 内容指纹} 映射 (供增量索引 + 版本快照共用)。"""
    out: dict[str, str] = {}
    for i, r in enumerate(records):
        rec_id = str(r.get(id_field) or f"rec_{i}")
        out[rec_id] = _record_fingerprint(r, text_field)
    return out


# ---------- 变更检测 ----------

@dataclass
class RecordsDiff:
    """两轮语料之间的增量差异 (P3-5 的「增量更新检测」核心)。"""

    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)

    @property
    def total_changed(self) -> int:
        return len(self.added) + len(self.updated) + len(self.removed)


def diff_records(
    current: dict[str, str],
    known: dict[str, str],
) -> RecordsDiff:
    """比较两轮 {id: 内容指纹}，返回增量差异。

    Args:
        current: 当前语料的 id->指纹。
        known: 上一轮/上一版本的 id->指纹。

    Returns:
        RecordsDiff——added/updated/removed/unchanged 四类 id 列表。
    """
    diff = RecordsDiff()
    for rid, fp in current.items():
        if rid not in known:
            diff.added.append(rid)
        elif known[rid] != fp:
            diff.updated.append(rid)
        else:
            diff.unchanged.append(rid)
    diff.removed = [rid for rid in known if rid not in current]
    return diff


# ---------- 版本捕获 / 切换 / 回滚 ----------

def _load_or_empty(registry: dict[str, list[IndexVersion]], corpus: str) -> list[IndexVersion]:
    return registry.setdefault(corpus, [])


def _next_version_id(versions: list[IndexVersion]) -> int:
    return max((v.version_id for v in versions), default=0) + 1


def current_version(
    registry: dict[str, list[IndexVersion]],
    corpus: str,
) -> Optional[IndexVersion]:
    """返回某 corpus 当前激活的版本；无则返回 None。"""
    versions = registry.get(corpus) or []
    for v in versions:
        if v.active:
            return v
    return None


def previous_version(
    registry: dict[str, list[IndexVersion]],
    corpus: str,
) -> Optional[IndexVersion]:
    """返回当前激活版本之前的最近一个历史版本 (用于回滚目标)。"""
    versions = registry.get(corpus) or []
    if len(versions) < 2:
        return None
    # 最后一个 (最新) 通常是 active；回滚目标取其前一个非当前版本。
    active_id = None
    for v in versions:
        if v.active:
            active_id = v.version_id
            break
    for v in reversed(versions):
        if v.version_id != active_id:
            return v
    return None


def capture_version(
    workdir: Path,
    *,
    corpus: str,
    records: Optional[list[dict]] = None,
    fingerprints: Optional[dict[str, str]] = None,
    source_hash: str = "",
    id_field: str = "id",
    text_field: str = "_searchable",
    keep_history: bool = True,
) -> tuple[IndexVersion, Path]:
    """为某 corpus 捕获一个新版本并置为 active，返回 (version, 注册表路径)。

    新版本捕获后成为当前 active；旧版本默认保留在历史中 (keep_history=True)，
    供灰度切换/回滚使用。捕获前会对该 corpus 的旧版本全部置非 active。

    Args:
        records: 当前语料记录列表 (用于计算指纹)；与 fingerprints 二选一。
        fingerprints: 直接提供的 id->指纹映射；未提供时由 records 计算。
        source_hash: 构建时的源头 hash。
        keep_history: False 时丢弃该 corpus 的旧版本 (只留最新)，节省空间；
            缺省保留历史以支持回滚。

    Returns:
        (新 IndexVersion, 注册表文件路径)。
    """
    if fingerprints is None:
        fingerprints = (
            compute_fingerprints(records or [], id_field=id_field, text_field=text_field)
            if records is not None
            else {}
        )
    records_hash = (
        compute_records_hash(records or [], text_field=text_field)
        if records is not None
        else ""
    )

    registry = load_versions(workdir)
    versions = _load_or_empty(registry, corpus)

    new_ver = IndexVersion(
        version_id=_next_version_id(versions),
        corpus=corpus,
        built_at=datetime.now(timezone.utc).isoformat(),
        source_hash=source_hash,
        records_hash=records_hash,
        active=True,
        fingerprints=fingerprints,
    )

    if keep_history:
        for v in versions:
            v.active = False
        versions.append(new_ver)
    else:
        versions = [new_ver]
    registry[corpus] = versions

    p = save_versions(workdir, registry)
    return new_ver, p


def activate_version(
    workdir: Path,
    *,
    corpus: str,
    version_id: int,
) -> Optional[IndexVersion]:
    """把某 corpus 切换到指定版本 (激活它，其余置非 active)。

    用于「灰度切换/回滚」：回滚到历史版本即调用 activate_version(旧 id)。
    版本不存在时返回 None 且不修改注册表。

    Returns:
        激活后的版本；未找到目标版本时返回 None。
    """
    registry = load_versions(workdir)
    versions = registry.get(corpus)
    if not versions:
        return None
    target = next((v for v in versions if v.version_id == version_id), None)
    if target is None:
        return None
    for v in versions:
        v.active = (v.version_id == version_id)
    save_versions(workdir, registry)
    return target


def rollback(
    workdir: Path,
    *,
    corpus: str,
    to_version: Optional[int] = None,
) -> Optional[IndexVersion]:
    """回滚某 corpus 到前一版本 (或指定版本)。

    与 activate_version 语义等价；to_version 缺省时选「当前激活版本之前
    的最近历史版本」(previous_version)。没有可回滚的历史版本或无任何版本时
    返回 None。

    Returns:
        回滚后激活的版本；无历史版本/无版本时返回 None。
    """
    if to_version is not None:
        return activate_version(workdir, corpus=corpus, version_id=to_version)
    registry = load_versions(workdir)
    prev = previous_version(registry, corpus)
    if prev is None:
        return None
    return activate_version(workdir, corpus=corpus, version_id=prev.version_id)
