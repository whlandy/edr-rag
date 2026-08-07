"""索引 manifest 版本/hash/过期检测 (P0-5).

为 P0/P1/P2 生成的 JSONL 产物 (sections.jsonl / procedures.jsonl /
chunks.jsonl / actions.jsonl …) 维护一份统一的"索引 manifest"
(index_manifest.json)，记录:

  - schema_version          : 索引 manifest 自身的格式版本
  - chain                   : 所属链路 (p0 / p1 / p2)
  - built_at                : 生成时间 (UTC ISO-8601)
  - source_hash             : 上游源头 hash (源自 P0 manifest 的
                              source_hash，即源 CHM 的 hash)
  - artifacts[<filename>]   : 各 JSONL 产物的内容 sha256 + 行数

并提供一个校验工具 validate_index_manifest：读取时如果发现
  - 某产物的当前内容 hash 与记录不符  -> 内容过期 (STALE_CONTENT)
  - source_hash 与当前源头不符        -> 源头变化 (SOURCE_MISMATCH)
  - 源头为空 (sections.jsonl 不存在/为空) 但下游索引产物仍有内容
                                     -> 孤儿残留 (ORPHAN_INDEX)
  - schema_version 不匹配             -> 需要重建 (SCHEMA_MISMATCH)

语义分层 (P0-5):
  - warning 级 (可继续读，但应提示)：内容过期、源头变化
  - refuse 级 (拒绝读取)：schema 不匹配、孤儿残留 —— 前者无法安全读取，
    后者意味着索引与源头脱节，读出来也是错的。

主链路 (keyword/BM25 检索) 不依赖本模块；本模块只在生成/校验入口被显式
调用，缺索引时给出 NO_INDEX 结果而不抛异常。
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

#: 索引 manifest 当前格式版本。凡改动 artifacts 的 key 结构 / 校验语义，
#: 都应递增此号，旧版本产物在读取时触发 SCHEMA_MISMATCH (拒绝读取)。
INDEX_MANIFEST_SCHEMA_VERSION = 1

#: 与 P0 manifest.json 交互的 key (P0 normalize.Manifest 使用)。
_MANIFEST_FILE = "manifest.json"


@dataclass
class ArtifactHash:
    """单个 JSONL 产物的内容指纹。"""

    sha256: str
    line_count: int


@dataclass
class IndexManifest:
    """索引 manifest：描述一次某链路的产物生成。"""

    schema_version: int
    chain: str                      # p0 / p1 / p2
    built_at: str                   # UTC ISO-8601
    source_hash: str                # 上游源头 hash (通常来自 P0 manifest)
    artifacts: dict[str, ArtifactHash] = field(default_factory=dict)  # filename -> hash

    def to_json(self) -> dict:
        d = asdict(self)
        d["artifacts"] = {
            name: {"sha256": a.sha256, "line_count": a.line_count}
            for name, a in self.artifacts.items()
        }
        return d

    @staticmethod
    def from_json(d: dict) -> "IndexManifest":
        artifacts = {}
        for name, a in (d.get("artifacts") or {}).items():
            artifacts[name] = ArtifactHash(
                sha256=a.get("sha256", ""),
                line_count=int(a.get("line_count", 0)),
            )
        return IndexManifest(
            schema_version=int(d.get("schema_version", 0)),
            chain=d.get("chain", ""),
            built_at=d.get("built_at", ""),
            source_hash=d.get("source_hash", ""),
            artifacts=artifacts,
        )


# ---------- 校验结果 ----------

#: 校验结论分级
VERDICT_OK = "OK"                        # 索引与新产物一致
VERDICT_NO_INDEX = "NO_INDEX"            # 未找到索引 manifest
VERDICT_STALE_CONTENT = "STALE_CONTENT"  # 某产物内容已变化 (过期，warning)
VERDICT_SOURCE_MISMATCH = "SOURCE_MISMATCH"  # 源头 hash 变化 (warning)
VERDICT_ORPHAN_INDEX = "ORPHAN_INDEX"    # 源为空但旧索引残留 (refuse)
VERDICT_SCHEMA_MISMATCH = "SCHEMA_MISMATCH"  # schema 版本不匹配 (refuse)

#: refuse 级结论：读取时应拒绝
_REFUSE_VERDICTS = {VERDICT_ORPHAN_INDEX, VERDICT_SCHEMA_MISMATCH}


@dataclass
class ValidationResult:
    """一次索引校验的结果。"""

    verdict: str
    index_path: Optional[Path] = None
    stale_files: list[str] = field(default_factory=list)   # 内容已变化的产物
    message: str = ""

    @property
    def ok(self) -> bool:
        return self.verdict == VERDICT_OK

    @property
    def refuse(self) -> bool:
        return self.verdict in _REFUSE_VERDICTS


# ---------- Hash 工具 ----------

def compute_file_hash(path: Path) -> tuple[str, int]:
    """返回 JSONL 文件内容的 (sha256 hex, 非空行数)。

    sha256 基于文件原始字节 (仅去除行尾空白以容忍换行风格差异)，
    行数为非空 JSON 行数。
    """
    h = hashlib.sha256()
    line_count = 0
    with Path(path).open("rb") as f:
        for raw_line in f:
            chunk = raw_line.rstrip(b"\r\n")
            if chunk.strip():
                line_count += 1
            h.update(chunk)
    return h.hexdigest(), line_count


def compute_artifacts_hashes(workdir, filenames: list[str]) -> dict[str, ArtifactHash]:
    """对指定文件名计算内容指纹；文件不存在或为空时记空表项。"""
    workdir = Path(workdir)
    out: dict[str, ArtifactHash] = {}
    for name in filenames:
        p = workdir / name
        if not p.exists():
            out[name] = ArtifactHash(sha256="", line_count=0)
            continue
        sha, lines = compute_file_hash(p)
        out[name] = ArtifactHash(sha256=sha, line_count=lines)
    return out


# ---------- 读取/写入 ----------

def index_path(workdir: Path) -> Path:
    return Path(workdir) / "index_manifest.json"


def load_source_hash(workdir: Path) -> str:
    """从 P0 manifest.json 读取 source_hash；缺失时返回空串。"""
    p = Path(workdir) / _MANIFEST_FILE
    if not p.exists():
        return ""
    try:
        manifest = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return ""
    return str(manifest.get("source_hash", "") or "")


def write_index_manifest(
    workdir: Path,
    *,
    chain: str,
    artifacts: dict[str, ArtifactHash],
    source_hash: str = "",
    built_at: Optional[str] = None,
    schema_version: int = INDEX_MANIFEST_SCHEMA_VERSION,
) -> Path:
    """原子写入 index_manifest.json 到 workdir，返回其路径。"""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    if not built_at:
        built_at = datetime.now(timezone.utc).isoformat()
    im = IndexManifest(
        schema_version=schema_version,
        chain=chain,
        built_at=built_at,
        source_hash=source_hash,
        artifacts=dict(artifacts),
    )
    out = workdir / "index_manifest.json"
    tmp = workdir / "index_manifest.json.tmp"
    tmp.write_text(
        json.dumps(im.to_json(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(out)
    return out


def load_index_manifest(workdir: Path) -> Optional[IndexManifest]:
    """读取并解析 index_manifest.json；不存在或解析失败返回 None。"""
    p = index_path(workdir)
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return IndexManifest.from_json(d)
    except (json.JSONDecodeError, OSError, ValueError, TypeError) as e:  # pragma: no cover
        logger.warning("index manifest %s unreadable: %s", p, e)
        return None


# ---------- 校验 ----------

def validate_index_manifest(
    workdir: Path,
    *,
    chain: Optional[str] = None,
    source_hash: Optional[str] = None,
    artifacts: Optional[list[str]] = None,
) -> ValidationResult:
    """校验 workdir 下的索引 manifest 与新产物是否一致。

    Args:
        workdir: 产物目录。
        chain: 期望链路 (p1/p2)。None 表示只校验通用一致性。
        source_hash: 期望的源头 hash；None 则用 P0 manifest 现存的 source_hash。
        artifacts: 需要参与内容 hash 校验的产物文件名；None 则用索引中已有的
            全部 artifacts。

    Returns:
        ValidationResult。verdict 语义见模块 docstring：
          OK / NO_INDEX / STALE_CONTENT / SOURCE_MISMATCH /
          ORPHAN_INDEX / SCHEMA_MISMATCH
    """
    workdir = Path(workdir)
    ip = index_path(workdir)
    im = load_index_manifest(workdir)

    if im is None:
        return ValidationResult(
            VERDICT_NO_INDEX, index_path=ip,
            message=f"未找到索引 manifest: {ip} (先运行 generate 生成)",
        )

    # schema 版本不匹配 -> 拒绝读取
    if im.schema_version != INDEX_MANIFEST_SCHEMA_VERSION:
        return ValidationResult(
            VERDICT_SCHEMA_MISMATCH, index_path=ip,
            message=(
                f"索引 manifest schema_version={im.schema_version} 与当前支持 "
                f"{INDEX_MANIFEST_SCHEMA_VERSION} 不符，拒绝读取，请重新 generate"
            ),
        )

    # 链路不匹配
    if chain is not None and im.chain != chain:
        return ValidationResult(
            VERDICT_SCHEMA_MISMATCH, index_path=ip,
            message=f"索引链路={im.chain!r} 与期望 {chain!r} 不符，拒绝读取",
        )

    # 源头 hash 变化 -> warning (可能只是重新 ingest，产物需重建)
    if source_hash is None:
        source_hash = load_source_hash(workdir)
    if source_hash and im.source_hash and im.source_hash != source_hash:
        return ValidationResult(
            VERDICT_SOURCE_MISMATCH, index_path=ip,
            message=(
                f"源 CHM hash 已变化 (索引={im.source_hash}, 当前={source_hash})，"
                f"下游索引已过期，建议重新生成"
            ),
        )

    # 孤儿残留：源为空但下游索引仍有内容 -> 拒绝读取
    if _source_empty(workdir):
        active = [n for n, a in im.artifacts.items() if a.line_count > 0]
        if active:
            return ValidationResult(
                VERDICT_ORPHAN_INDEX, index_path=ip,
                stale_files=active,
                message=(
                    f"源头为空 (sections.jsonl 不存在/无内容)，但索引产物 "
                    f"{active} 仍有 {len(active)} 份残留，索引与源头脱节，"
                    f"拒绝读取，请重新 ingest 后再 generate"
                ),
            )

    # 内容 hash 对比 -> 变化为 STALE_CONTENT (warning)
    names = artifacts if artifacts is not None else list(im.artifacts.keys())
    stale: list[str] = []
    current = compute_artifacts_hashes(workdir, names)
    for name in names:
        recorded = im.artifacts.get(name)
        if recorded is None:
            # 索引未跟踪该产物：仅当它非空时视为异常
            if current.get(name) and current[name].line_count > 0:
                stale.append(f"{name} (未跟踪)")
            continue
        cur = current.get(name)
        if cur is None or cur.sha256 != recorded.sha256:
            stale.append(name)

    if stale:
        return ValidationResult(
            VERDICT_STALE_CONTENT, index_path=ip, stale_files=stale,
            message=(
                f"以下产物内容已变化，索引已过期(建议重新 generate): {stale}"
            ),
        )

    return ValidationResult(
        VERDICT_OK, index_path=ip,
        message=f"索引 manifest 校验通过 ({ip})",
    )


def _source_empty(workdir: Path) -> bool:
    """源头 (sections.jsonl) 是否为"空"（不存在或无非空行）。"""
    p = Path(workdir) / "sections.jsonl"
    if not p.exists():
        return True
    try:
        for line in p.open(encoding="utf-8"):
            if line.strip():
                return False
    except OSError:  # pragma: no cover
        return True
    return True


# ---------- 便捷：生成时一键维护 ----------

def maintain_index(
    workdir: Path,
    *,
    chain: str,
    artifacts: list[str],
    source_hash: Optional[str] = None,
) -> Path:
    """生成后调用：计算各产物 hash 并写入 index_manifest.json。

    source_hash 缺省从 P0 manifest.json 读取。
    """
    if source_hash is None:
        source_hash = load_source_hash(workdir)
    hashes = compute_artifacts_hashes(workdir, artifacts)
    return write_index_manifest(
        workdir, chain=chain, artifacts=hashes, source_hash=source_hash
    )
