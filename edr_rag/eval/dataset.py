"""P1-5 评测集：schema / 加载 / 校验。

评测集是「查询 + 标注」的集合，支持 JSON (list of dict) 与 JSONL (每行一个
dict) 两种格式。每条查询 (EvalQuery) 包含:

字段:
  qid            必填, 唯一查询 id
  query          必填, 检索主查询
  corpus_type    必填, 检索语料 (actions/sections/procedures/chunks/nodes)
  relevant       必填(list), 相关文档 id 集合 (可用于标注为空 list 的纯元数据用例)
  paraphrases    可选(list), 同义改写 (计入评测集规模, runner 可选用)
  top_k          可选(int), 评估截断 K (缺省 5)
  filter         可选(dict), metadata 过滤条件 (透传给 RetrievalService)
  filter_expected 可选(list), 过滤后应出现的 id 集合

runner 默认对 `query` 评测；若 ``use_paraphrases`` 打开则把每条改写也当
一条独立查询评估 (用于放大评测规模)。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class EvalQuery:
    """单条评测查询及其标注。"""

    qid: str
    query: str
    corpus_type: str
    relevant: list[str] = field(default_factory=list)
    paraphrases: list[str] = field(default_factory=list)
    top_k: int = 5
    filter: Optional[dict[str, Any]] = None
    filter_expected: list[str] = field(default_factory=list)

    @staticmethod
    def from_dict(d: dict) -> "EvalQuery":
        """从 dict 构造，缺失字段用缺省值兜底。"""
        return EvalQuery(
            qid=str(d.get("qid", "")),
            query=str(d.get("query", "")),
            corpus_type=str(d.get("corpus_type", "actions")),
            relevant=list(d.get("relevant", []) or []),
            paraphrases=list(d.get("paraphrases", []) or []),
            top_k=int(d.get("top_k", 5) or 5),
            filter=d.get("filter") if d.get("filter") else None,
            filter_expected=list(d.get("filter_expected", []) or []),
        )

    def effective_queries(self, use_paraphrases: bool = False) -> list[tuple[str, str]]:
        """返回 (qid, query) 列表：默认仅主查询；开启改写则含每条改写。

        改写条目的 qid 附 ``#p<i>`` 后缀避免与主查询混淆。
        """
        items: list[tuple[str, str]] = [(self.qid, self.query)]
        if use_paraphrases:
            for i, p in enumerate(self.paraphrases, start=1):
                if p.strip():
                    items.append((f"{self.qid}#p{i}", p))
        return items


@dataclass
class EvalSet:
    """评测集：一组 EvalQuery + 元信息。

    ``paraphrase_total`` 统计全部改写条目数，用于体现评测集规模
    (同义改写也算一份有效查询)。
    """

    queries: list[EvalQuery]
    name: str = ""

    @property
    def paraphrase_total(self) -> int:
        return sum(len(q.paraphrases) for q in self.queries)

    @property
    def variant_total(self) -> int:
        """主查询 + 改写 = 全部可评测查询变体数。"""
        return len(self.queries) + self.paraphrase_total

    def validate(self) -> list[str]:
        """校验一致性，返回错误信息列表 (空 = 通过)。"""
        errors: list[str] = []
        seen: set[str] = set()
        allowed_corpora = {"actions", "sections", "procedures", "chunks", "nodes"}
        for q in self.queries:
            if not q.qid:
                errors.append("query missing qid")
            if q.qid in seen:
                errors.append(f"duplicate qid: {q.qid}")
            seen.add(q.qid)
            if not q.query.strip():
                errors.append(f"[{q.qid}] empty query")
            if q.corpus_type not in allowed_corpora:
                errors.append(f"[{q.qid}] unknown corpus_type '{q.corpus_type}'")
            if q.top_k < 1:
                errors.append(f"[{q.qid}] top_k must be >= 1")
            if q.filter and q.filter_expected and not isinstance(q.filter, dict):
                errors.append(f"[{q.qid}] filter must be a dict")
        return errors


def _load_json(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    data = json.loads(text)
    if isinstance(data, dict):
        # 允许 {"name": ..., "queries": [...]} 的 JSON 容器格式。
        if "queries" in data:
            return list(data["queries"])
        return []
    if isinstance(data, list):
        return data
    raise ValueError(f"unsupported JSON top-level type: {type(data).__name__}")


def _load_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def load_eval_set(path: str | Path) -> EvalSet:
    """加载评测集（JSON 或 JSONL），返回 EvalSet。

    Args:
        path: 评测集文件路径 (.json 或 .jsonl)。

    Raises:
        FileNotFoundError: 文件不存在。
        ValueError: 格式不支持 / 无法解析 / 数据畸形。
        EvalSetValidationError: 校验失败 (见 validate)。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"eval set not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".json":
        raw = _load_json(path)
    elif suffix == ".jsonl":
        raw = _load_jsonl(path)
    else:
        raise ValueError(f"unsupported eval set format '{suffix}'; use .json or .jsonl")

    queries = [EvalQuery.from_dict(d) for d in raw if isinstance(d, dict)]
    name = path.stem
    # JSON 容器格式可携带 name
    if suffix == ".json":
        try:
            top = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(top, dict) and top.get("name"):
                name = str(top["name"])
        except Exception:  # pragma: no cover - already parsed above
            pass

    evalset = EvalSet(queries=queries, name=name)
    errors = evalset.validate()
    if errors:
        raise EvalSetValidationError("\n".join(errors[:20]))
    return evalset


class EvalSetValidationError(ValueError):
    """评测集数据不合法。"""
