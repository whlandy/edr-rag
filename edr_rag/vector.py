"""Optional LlamaIndex vector and hybrid retrieval over EDR JSONL artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

DEFAULT_EMBED_MODEL = "BAAI/bge-small-zh-v1.5"
INDEX_DIR_NAME = ".llamaindex"
ARTIFACT_FILES = {
    "section": "sections.jsonl",
    "procedure": "procedures.jsonl",
    "step": "chunks.jsonl",
    "action": "actions.jsonl",
}


class LlamaIndexUnavailableError(RuntimeError):
    """Raised when the optional LlamaIndex dependencies are not installed."""


def _imports():
    try:
        from llama_index.core import Document, StorageContext, VectorStoreIndex, load_index_from_storage
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    except ImportError as exc:
        raise LlamaIndexUnavailableError(
            "LlamaIndex support is not installed. Run: pip install -e '.[rag]'"
        ) from exc
    return Document, StorageContext, VectorStoreIndex, load_index_from_storage, HuggingFaceEmbedding


def _record_text(record: dict) -> str:
    """Turn any P0/P1/P2 record into embedding-friendly text."""
    parts: list[str] = []
    for key in ("title", "module", "text", "content", "summary"):
        value = record.get(key)
        if value:
            parts.append(str(value))
    parts.extend(str(value) for value in record.get("section_path", []) if value)
    parts.extend(str(value) for value in record.get("preconditions", []) if value)
    for step in record.get("procedure", []):
        if isinstance(step, dict):
            parts.extend(
                str(step[key]) for key in ("instruction", "source_instruction", "target")
                if step.get(key)
            )
    return "\n".join(parts)


def _record_id(record: dict, artifact_type: str, position: int) -> str:
    return str(record.get("id") or record.get("action_id") or f"{artifact_type}.{position}")


def _read_records(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class LlamaIndexStore:
    """Build and query one persisted vector index per artifact type."""

    def __init__(
        self,
        workdir: str | Path,
        *,
        embed_model: str = DEFAULT_EMBED_MODEL,
        index_dir: str | Path | None = None,
    ):
        self.workdir = Path(workdir)
        self.embed_model_name = embed_model
        self.index_dir = Path(index_dir) if index_dir else self.workdir / INDEX_DIR_NAME
        self._embed_model_instance = None

    def _embed_model(self):
        if self._embed_model_instance is None:
            *_, HuggingFaceEmbedding = _imports()
            self._embed_model_instance = HuggingFaceEmbedding(model_name=self.embed_model_name)
        return self._embed_model_instance

    def build(self, artifact_types: Iterable[str], *, show_progress: bool = False) -> dict[str, int]:
        Document, _, VectorStoreIndex, _, _ = _imports()
        counts: dict[str, int] = {}
        for artifact_type in artifact_types:
            if artifact_type not in ARTIFACT_FILES:
                raise ValueError(f"Unsupported artifact type: {artifact_type}")
            source = self.workdir / ARTIFACT_FILES[artifact_type]
            if not source.exists():
                raise FileNotFoundError(f"Missing artifact: {source}")
            records = _read_records(source)
            documents = []
            for position, record in enumerate(records):
                record_id = _record_id(record, artifact_type, position)
                metadata = {
                    "record_id": record_id,
                    "artifact_type": artifact_type,
                    "title": str(record.get("title", "")),
                    "source_file": str(
                        (record.get("source") or {}).get("file")
                        or (record.get("source_ref") or {}).get("relative_path", "")
                    ),
                }
                documents.append(Document(text=_record_text(record), id_=record_id, metadata=metadata))
            if not documents:
                counts[artifact_type] = 0
                continue
            index = VectorStoreIndex.from_documents(
                documents,
                embed_model=self._embed_model(),
                show_progress=show_progress,
            )
            destination = self.index_dir / artifact_type
            destination.mkdir(parents=True, exist_ok=True)
            index.storage_context.persist(persist_dir=str(destination))
            counts[artifact_type] = len(documents)
        manifest = {
            "embed_model": self.embed_model_name,
            "counts": counts,
        }
        self.index_dir.mkdir(parents=True, exist_ok=True)
        (self.index_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return counts

    def search(self, query: str, artifact_type: str, *, limit: int = 5) -> list[dict]:
        _, StorageContext, _, load_index_from_storage, _ = _imports()
        persist_dir = self.index_dir / artifact_type
        if not persist_dir.exists():
            raise FileNotFoundError(
                f"Vector index not found: {persist_dir}. Run `edr-rag index` first."
            )
        storage = StorageContext.from_defaults(persist_dir=str(persist_dir))
        index = load_index_from_storage(storage, embed_model=self._embed_model())
        nodes = index.as_retriever(similarity_top_k=limit).retrieve(query)
        results = []
        for item in nodes:
            metadata = item.node.metadata or {}
            results.append({
                "id": metadata.get("record_id", item.node.node_id),
                "type": metadata.get("artifact_type", artifact_type),
                "score": round(float(item.score or 0.0), 6),
                "title": metadata.get("title", ""),
                "source": {"file": metadata.get("source_file", "")},
                "snippet": item.node.get_content(metadata_mode="none")[:120],
            })
        return results


def reciprocal_rank_fusion(
    lexical: list[dict], semantic: list[dict], *, limit: int = 5, k: int = 60
) -> list[dict]:
    """Fuse lexical and semantic rankings using deterministic RRF."""
    by_id: dict[str, dict] = {}
    scores: dict[str, float] = {}
    for result_list in (lexical, semantic):
        for rank, result in enumerate(result_list, start=1):
            record_id = str(result.get("id", ""))
            if not record_id:
                continue
            by_id.setdefault(record_id, result.copy())
            scores[record_id] = scores.get(record_id, 0.0) + 1.0 / (k + rank)
    ranked = sorted(scores, key=lambda record_id: (-scores[record_id], record_id))
    output = []
    for record_id in ranked[:limit]:
        result = by_id[record_id]
        result["score"] = round(scores[record_id], 6)
        result["scoring"] = "hybrid_rrf"
        output.append(result)
    return output
