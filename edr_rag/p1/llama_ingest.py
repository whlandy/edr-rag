"""P1 LlamaIndex 摄取管线（可选依赖 `rag` extras）。 [P1-1]

目标：
  - 在「结构化切分之后的 JSONL 产物」（chunks.jsonl / sections.jsonl）之上，
    提供一条基于 LlamaIndex Ingestion Pipeline 的可选摄取管线，用于：
      1. 节点转换（node transforms）：文本切分 / 清理 / 元数据注入；
      2. 结构化切分（structured splitting）：以既有 chunk 边界为基底做切分；
      3. 持久化缓存（persistence cache）：把转换后的节点序列化到磁盘，避免
         重复摄取重复计算；
      4. 重复文档检测（duplicate detection）：按内容 hash + 文档 id 去重。

优雅降级：
  llama-index-core / embeddings / LLM 均为可选（`rag` extras）。当它们缺失时：
    - 本模块仍可正常导入与调用，不会让主链路报错；
    - `run_ingestion_pipeline` 走纯标准库的降级路径（结构化切分 + 持久化 +
      去重均可用实现一致），并把 mode 置为 'fallback'；
    - 检索侧继续使用既有的 keyword / BM25（见 edr_rag/p1/search.py 与
      edr_rag/retrieval.py 的 vector_search 降级）。

本模块刻意不把 LlamaIndex 相关的 import 放在模块顶层——所有可选依赖都在
函数内部惰性导入，保证缺依赖时 import edr_rag.p1.llama_ingest 本身不失败。
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

#: 持久化缓存文件名（节点序列化）。
_NODES_CACHE_FILE = "llama_nodes.json"
#: 摄取报告文件名。
_REPORT_FILE = "llama_ingest_report.json"


class LlamaIndexUnavailable(RuntimeError):
    """当 llama-index-core 不可导入时抛出。

    调用方可用 except 捕获后走降级路径；主链路永不因缺依赖而崩溃。
    """


def llama_index_available() -> bool:
    """返回可选的 llama-index-core 是否可导入。"""
    try:
        import llama_index.core  # noqa: F401

        return True
    except Exception:  # pragma: no cover - optional deps absent
        return False


# ---------- 内容指纹 / 去重 ----------

def _content_fingerprint(text: str) -> str:
    """对一条记录的可检索文本计算稳定指纹（sha256 前 16 位）。"""
    return hashlib.sha256((text or "").encode("utf-8", "ignore")).hexdigest()[:16]


def _record_text(r: dict) -> str:
    """提取一条 JSONL 记录的可检索文本（与 JsonlCorpus._normalize 对齐的字段）。"""
    text = r.get("text") or r.get("content") or ""
    title = r.get("title") or ""
    section_path = r.get("section_path") or []
    paths = " ".join(section_path) if isinstance(section_path, list) else ""
    return " ".join([title, paths, text])


def _record_key(r: dict, corpus_type: str) -> str:
    """提取记录主键：actions 用 action_id，其余用 id。"""
    if corpus_type == "actions":
        return str(r.get("action_id") or r.get("id") or "")
    return str(r.get("id") or "")


# ---------- 输入读取 ----------

def read_jsonl(path: Path) -> list[dict]:
    """读取一个 JSONL 文件为 dict 列表；文件缺失时返回空列表。"""
    path = Path(path)
    if not path.exists():
        logger.warning("ingest source not found: %s", path)
        return []
    out: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


# ---------- 结构化切分（降级路径与基底） ----------

@dataclass
class SplitNode:
    """一次摄取产生的一个节点（与可选依赖无关的中间表示）。

    - id: 稳定 id（源记录主键，按类型加前缀）
    - text: 节点文本
    - metadata: 随节点保留的元数据（manual_id/type/source_file/... ）
    - fingerprint: 内容指纹（用于去重）
    """

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    fingerprint: str = ""

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata,
            "fingerprint": self.fingerprint,
        }

    @classmethod
    def from_json(cls, d: dict) -> "SplitNode":
        return cls(
            id=d.get("id", ""),
            text=d.get("text", ""),
            metadata=d.get("metadata", {}) or {},
            fingerprint=d.get("fingerprint", ""),
        )


def structured_split_records(
    records: list[dict],
    corpus_type: str,
    *,
    chunk_size: int = 512,
    overlap: int = 0,
) -> list[SplitNode]:
    """对 JSONL 记录做结构化切分，产出节点列表（纯标准库实现）。

    切分策略（与 P1 既有 chunk 边界对齐）：
      - 每一步/每一条已是「结构化单元」，其本身即一个节点；
      - 若文本超过 chunk_size，再用字符窗口进一步切分并保留 overlap，
        以控制单节点体量（用于后续向量化/检索的粒度）。

    说明：结构化切分优先依赖既有 P1 的 chunk/step 粒度；这里的再切分仅作为
    超长文本的兜底，保证节点体量有上界。
    """
    nodes: list[SplitNode] = []

    def _emit(prefix: str, key: str, text: str, meta: dict):
        text = (text or "").strip()
        if not text:
            return
        if len(text) <= chunk_size or overlap < 0 or chunk_size <= 0:
            node_id = f"{prefix}.{key}"
            nodes.append(
                SplitNode(
                    id=node_id,
                    text=text,
                    metadata=_clean_meta(meta),
                    fingerprint=_content_fingerprint(text),
                )
            )
            return
        # 字符窗口再切分
        step = max(chunk_size - overlap, 1)
        for i in range(0, len(text), step):
            piece = text[i:i + chunk_size].strip()
            if not piece:
                continue
            node_id = f"{prefix}.{key}.{i // step}"
            nodes.append(
                SplitNode(
                    id=node_id,
                    text=piece,
                    metadata=_clean_meta(meta),
                    fingerprint=_content_fingerprint(piece),
                )
            )

    for r in records:
        key = _record_key(r, corpus_type)
        text = _record_text(r)
        meta = _record_metadata(r, corpus_type)
        prefix = f"{corpus_type if corpus_type != 'actions' else 'action'}"
        _emit(prefix, key, text, meta)

    return nodes


def _clean_meta(meta: dict) -> dict[str, Any]:
    """把元数据规整成 JSON 可序列化且便于检索的 dict。"""
    out: dict[str, Any] = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        elif isinstance(v, (list, tuple)):
            out[k] = [str(x) for x in v]
        elif isinstance(v, dict):
            out[k] = json.dumps(v, ensure_ascii=False)
        else:
            out[k] = str(v)
    return out


def _record_metadata(r: dict, corpus_type: str) -> dict[str, Any]:
    """抽取记录元数据用于节点；与 JsonlCorpus 检索文本折叠逻辑对应。"""
    meta: dict[str, Any] = {
        "id": r.get("id", ""),
        "type": r.get("type", corpus_type),
        "manual_id": r.get("manual_id", ""),
        "title": r.get("title", ""),
        "product": r.get("product", ""),
    }
    if corpus_type == "actions":
        meta["action_id"] = r.get("action_id", "")
        meta["risk"] = r.get("risk", "")
        meta["status"] = r.get("status", "")
    src = r.get("source") or {}
    if isinstance(src, dict):
        meta["source_file"] = src.get("file", "")
    else:
        meta["source_file"] = r.get("source_file", "")
    ch_meta = r.get("metadata")
    if isinstance(ch_meta, dict):
        for k in ("section_path", "module", "operation", "order", "version"):
            if k in ch_meta:
                meta[k] = ch_meta[k]
    return meta


# ---------- 持久化缓存 / 去重 ----------

def _seen_set_from_cache(nodes_cache_path: Path) -> set[str]:
    """读取持久化缓存，返回其中已存在的指纹 + id 集合（用于去重）。"""
    seen: set[str] = set()
    if not nodes_cache_path.exists():
        return seen
    try:
        data = json.loads(nodes_cache_path.read_text(encoding="utf-8"))
        for n in data.get("nodes", []):
            fp = n.get("fingerprint")
            if fp:
                seen.add(f"fp:{fp}")
            if n.get("id"):
                seen.add(f"id:{n['id']}")
    except Exception:
        logger.warning("ingest cache unreadable; rebuilding: %s", nodes_cache_path)
    return seen


def dedupe_nodes(
    nodes: list[SplitNode],
    *,
    seen: set[str],
    fingerprint_key: bool = True,
    id_key: bool = True,
) -> list[SplitNode]:
    """按指纹和/或 id 去重，返回新增节点。

    - fingerprint 相等 -> 内容相同，丢弃；
    - id 相等 -> 同源记录重复，丢弃。
    """
    out: list[SplitNode] = []
    local: set[str] = set()
    for n in nodes:
        fp_key = f"fp:{n.fingerprint}" if (fingerprint_key and n.fingerprint) else None
        id_key_s = f"id:{n.id}" if (id_key and n.id) else None
        if fp_key and (fp_key in seen or fp_key in local):
            continue
        if id_key_s and (id_key_s in seen or id_key_s in local):
            continue
        if fp_key:
            local.add(fp_key)
        if id_key_s:
            local.add(id_key_s)
        out.append(n)
    return out


# ---------- LlamaIndex 摄取管线（可选） ----------

def _import_llama():
    """惰性导入 llama_index.core；缺失时抛 LlamaIndexUnavailable。"""
    try:
        import llama_index.core  # noqa: F401
        from llama_index.core.ingestion import IngestionPipeline  # noqa: F401
        from llama_index.core.node_parser import SentenceSplitter  # noqa: F401
        from llama_index.core.schema import Document  # noqa: F401

        return {
            "core": llama_index.core,
            "IngestionPipeline": IngestionPipeline,
            "SentenceSplitter": SentenceSplitter,
            "Document": Document,
        }
    except Exception as e:  # pragma: no cover - optional deps absent
        raise LlamaIndexUnavailable(
            "llama-index-core not importable; run `pip install -e '.[rag]'` "
            "to enable the LlamaIndex ingestion pipeline"
        ) from e


def build_llama_documents(
    records: list[dict],
    corpus_type: str,
) -> list[Any]:
    """把 JSONL 记录转成 LlamaIndex Document 列表（需 llama-index-core）。"""
    llama = _import_llama()
    Document = llama["Document"]
    docs: list[Any] = []
    for r in records:
        key = _record_key(r, corpus_type)
        text = _record_text(r)
        meta = _record_metadata(r, corpus_type)
        if not text.strip():
            continue
        meta["node_id"] = f"{corpus_type}.{key}"
        # 用既有结构化边界作为元数据；Document 保留原始全文。
        docs.append(
            Document(text=text, metadata=meta, id_=f"{corpus_type}.{key}")
        )
    return docs


def run_llama_pipeline(
    records: list[dict],
    corpus_type: str,
    *,
    chunk_size: int = 512,
    overlap: int = 0,
) -> list[dict]:
    """用 LlamaIndex IngestionPipeline 摄取记录，返回节点 JSON 列表。

    在 LlamaIndex 侧做「节点转换 + 结构化切分」：
      - 把每条记录做成 Document；
      - 用 SentenceSplitter 对超长文本做结构化切分（chunk/overlap）；
      - pipeline.run(documents) 产出节点列表。

    缺依赖（llama-index-core 不可用）或切分失败时抛 LlamaIndexUnavailable，
    由调用方决定是否降级到纯标准库路径。
    """
    llama = _import_llama()
    IngestionPipeline = llama["IngestionPipeline"]
    SentenceSplitter = llama["SentenceSplitter"]
    Document = llama["Document"]

    docs = build_llama_documents(records, corpus_type)
    if not docs:
        return []

    transform = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    pipeline = IngestionPipeline(transformations=[transform])
    nodes = pipeline.run(documents=docs)

    out: list[dict] = []
    for n in nodes:
        text = getattr(n, "text", "") or (n.get_content() if hasattr(n, "get_content") else "")
        node_id = getattr(n, "node_id", "") or ""
        meta = dict(getattr(n, "metadata", {}) or {})
        out.append(
            {
                "id": node_id or text[:16],
                "text": text,
                "metadata": _clean_meta(meta),
                "fingerprint": _content_fingerprint(text),
            }
        )
    return out


# ---------- 编排入口 ----------

@dataclass
class IngestResult:
    """一次摄取的结果报告。"""

    mode: str  # 'llama' | 'fallback'
    corpus_type: str
    source_records: int
    produced_nodes: int
    new_nodes: int
    duplicates_skipped: int
    cache_path: str
    report_path: str
    error: Optional[str] = None

    def to_json(self) -> dict:
        return {
            "mode": self.mode,
            "corpus_type": self.corpus_type,
            "source_records": self.source_records,
            "produced_nodes": self.produced_nodes,
            "new_nodes": self.new_nodes,
            "duplicates_skipped": self.duplicates_skipped,
            "cache_path": self.cache_path,
            "report_path": self.report_path,
            "error": self.error,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }


def run_ingestion_pipeline(
    workdir: str | Path,
    corpus_type: str = "chunks",
    *,
    chunk_size: int = 512,
    overlap: int = 0,
    force_llama: bool = False,
    persist: bool = True,
) -> IngestResult:
    """运行一次 LlamaIndex 摄取（可选依赖），并优雅降级。

    流程：
      1. 从 <workdir>/<corpus>.jsonl 读记录（chunks/procedures/sections/actions）。
      2. 读持久化缓存 -> 已知指纹/id 集合。
      3. 生成节点：
           - 若 llama-index-core 可用且未强制降级 -> 走 LlamaIndex pipeline；
           - 否则走纯标准库结构化切分（fallback）。
      4. 去重（指纹 + id）。
      5. 持久化：把新增节点并入缓存写回 <workdir>/llama_nodes.json，
         并写摄取报告 <workdir>/llama_ingest_report.json。

    Args:
        workdir: 产物工作目录。
        corpus_type: 要摄取的语料类型（chunks/procedures/sections/actions）。
        chunk_size: 节点最大文本长度（字符）。
        overlap: 再切分时的重叠长度。
        force_llama: True 时要求走 LlamaIndex；此时缺依赖会抛
            LlamaIndexUnavailable 而非静默降级。
        persist: 是否写持久化缓存与报告。

    Returns:
        IngestResult 报告。缺依赖且非 force_llama 时 mode='fallback'。
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    cache_path = workdir / _NODES_CACHE_FILE
    report_path = workdir / _REPORT_FILE

    records = read_jsonl(workdir / f"{corpus_type}.jsonl")

    # 结构化切分：先产出基地节点（无论是否用 llama，都先算一遍文本边界）
    base_nodes = structured_split_records(
        records, corpus_type, chunk_size=chunk_size, overlap=overlap
    )

    # 决定使用哪种切分实现
    use_llama = (not force_llama and llama_index_available()) or force_llama
    mode = "llama" if use_llama else "fallback"
    error: Optional[str] = None
    produced_nodes: list[dict] = []

    if use_llama:
        try:
            produced_nodes = run_llama_pipeline(
                records, corpus_type, chunk_size=chunk_size, overlap=overlap
            )
        except LlamaIndexUnavailable as e:
            error = str(e)
            produced_nodes = []
            if force_llama:
                raise
            # 缺依赖：回退到标准库结构化切分，mode 置 fallback
            mode = "fallback"
        except Exception as e:  # pragma: no cover - optional path only
            logger.warning("llama ingestion failed (%s); falling back", e)
            error = f"{type(e).__name__}: {e}"
            produced_nodes = []
            if force_llama:
                raise
            mode = "fallback"

    # 无论哪种 mode，都确保有节点可产出（fallback 用标准库节点）
    if not produced_nodes:
        produced_nodes = [n.to_json() for n in base_nodes]

    # 持久化缓存 -> 去重 -> 并入缓存
    seen = _seen_set_from_cache(cache_path)
    nodes_objs = [SplitNode.from_json(d) for d in produced_nodes]
    new_nodes = dedupe_nodes(nodes_objs, seen=seen)

    if persist:
        # 读取旧缓存节点并追加新增节点，再原子写回
        old_nodes: list[dict] = []
        if cache_path.exists():
            try:
                old_nodes = json.loads(cache_path.read_text(encoding="utf-8")).get(
                    "nodes", []
                )
            except Exception:
                old_nodes = []
        merged = old_nodes + [n.to_json() for n in new_nodes]
        cache_payload = {
            "corpus_type": corpus_type,
            "node_count": len(merged),
            "nodes": merged,
        }
        _atomic_write_json(cache_path, cache_payload)
        _atomic_write_json(
            report_path,
            IngestResult(
                mode=mode,
                corpus_type=corpus_type,
                source_records=len(records),
                produced_nodes=len(produced_nodes),
                new_nodes=len(new_nodes),
                duplicates_skipped=len(produced_nodes) - len(new_nodes),
                cache_path=str(cache_path),
                report_path=str(report_path),
                error=error,
            ).to_json(),
        )
    else:
        logger.warning("persist=False: cache not updated")

    return IngestResult(
        mode=mode,
        corpus_type=corpus_type,
        source_records=len(records),
        produced_nodes=len(produced_nodes),
        new_nodes=len(new_nodes),
        duplicates_skipped=len(produced_nodes) - len(new_nodes),
        cache_path=str(cache_path),
        report_path=str(report_path),
        error=error,
    )


def _atomic_write_json(path: Path, payload: dict) -> None:
    """原子写 JSON（临时文件 + rename），重建中途失败不污染现有产物。"""
    from ..atomic_io import atomic_write_text

    atomic_write_text(
        path, json.dumps(payload, ensure_ascii=False, indent=2)
    )


def load_cached_nodes(workdir: str | Path) -> list[dict]:
    """读取持久化的节点缓存（无则返回空列表）。"""
    path = Path(workdir) / _NODES_CACHE_FILE
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("nodes", [])
    except Exception:
        logger.warning("cannot read node cache: %s", path)
        return []


def ingest_health(workdir: str | Path) -> dict:
    """报告当前摄取栈可用性（供 CLI/诊断）。"""
    return {
        "llama_index_available": llama_index_available(),
        "cache_exists": (Path(workdir) / _NODES_CACHE_FILE).exists(),
        "cached_nodes": len(load_cached_nodes(workdir)),
    }
