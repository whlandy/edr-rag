"""统一 RetrievalService — 收敛所有检索入口 (P0-2).

设计目标:
  - 新建 edr_rag/retrieval.py，提供统一的 RetrievalService，作为纯字 API
    (EdrRagAPI)、CLI (p2/cli.py)、MCP (EdrRagAgent) 的唯一检索入口。
  - 禁止调用方各自直接 new JsonlCorpus 或直接读 JSONL；全部通过
    RetrievalService.search(...) 走同一套 corpora + 打分 + 过滤逻辑。

支持的 corpus_type:
  - sections   -> sections.jsonl   (P0 章节/手册正文)
  - procedures -> procedures.jsonl (P1 流程)
  - actions    -> actions.jsonl    (P2 动作目录，返回富字段)
  - chunks     -> chunks.jsonl     (P1 步骤分片)
  - images     -> ocr.jsonl        (P3-4 图片 OCR 文本/截图检索)

搜索策略 strategy: 'keyword' | 'bm25' | 'vector' | 'hybrid'
  - keyword: 子串/词元重叠打分，动作对 title/action_id 加权
  - bm25:    轻量 BM25
  - vector:  可选 RAG extras 的向量检索；缺依赖时优雅降级到 bm25
             (主链路始终可用 keyword/BM25)
  - hybrid:  多路召回 -> RRF 融合 -> (可选) cross-encoder rerank -> top K
             (P1-3)。召回侧融合 keyword + bm25 (+ 有 embedder 时叠加 vector)，
             融合后按需做 bge-reranker 精排；reranker 为可选依赖，缺依赖或
             通过 rerank=False / 环境变量 EDR_RAG_RERANK=0 关闭时静默跳过。

planner_retrieve 保持既有的 action -> procedure -> manual 优先级，
并把调用方传入的 strategy 透传给每一层检索。
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Optional

from .p1.search import JsonlCorpus
from .rerank import rrf_fuse, rerank, rerank_enabled as _rerank_default
from .security import ACLPolicy, acl_allows

# 允许的 corpus_type -> 对应 JSONL 文件名
_CORPUS_FILES: dict[str, str] = {
    "sections": "sections.jsonl",
    "procedures": "procedures.jsonl",
    "actions": "actions.jsonl",
    "chunks": "chunks.jsonl",
    "nodes": "nodes.jsonl",  # P1-2 结构化节点（P1-4 上下文补全用）
    "images": "ocr.jsonl",   # P3-4 图片 OCR 文本（截图文字检索）
}

_ALLOWED_STRATEGIES = ("keyword", "bm25", "vector", "hybrid")
# hybrid 为未来融合策略占位：当前实现落到 bm25（对齐 vector 缺依赖时的降级行为），
# 后续接入 keyword+vector 融合时只需改进 _resolve_strategy。
_DEFAULT_STRATEGY = "keyword"

#: 缓存签名类型：用文件 mtime(纳秒)+size 做廉价失效判断，避免每次调用重读 JSONL。
#: mtime/size 任一变化即视为文件已重建，触发重载并重建内存索引。
_CacheSig = tuple[int, int] | None  # (st_mtime_ns, st_size)；文件缺失为 None


def _file_signature(path: Path) -> _CacheSig:
    """返回文件 (mtime_ns, size) 签名；文件不存在时返回 None。"""
    try:
        st = path.stat()
    except FileNotFoundError:
        return None
    return (st.st_mtime_ns, st.st_size)


def _read_jsonl(path: Path) -> list[dict]:
    """读取 JSONL 文件为 dict 列表；文件缺失 / 空 / 坏行时安全跳过。"""
    if not path.exists():
        return []
    out: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:  # pragma: no cover - defensive
                    logging.getLogger(__name__).warning(
                        "skip unparseable line in %s", path
                    )
    return out


def _node_objects(records: list[dict]):
    """把 nodes.jsonl 的 dict 记录还原成 StructuredNode 对象列表。"""
    from .p1.nodes import StructuredNode

    return [StructuredNode.from_json(r) for r in records]


class RetrievalService:
    """统一检索服务。

    由 workdir 懒加载各 corpus 并缓存；对外仅暴露 search / get /
    planner_retrieve 三个检索入口。
    """

    def __init__(self, workdir: str | Path):
        self.workdir = Path(workdir)
        # 每个 corpus_type 缓存 None(未加载/文件缺失) 或 (签名, corpus)。
        self._corpora: dict[str, tuple[_CacheSig, JsonlCorpus] | None] = {
            "sections": None,
            "procedures": None,
            "actions": None,
            "chunks": None,
            "nodes": None,
            "images": None,
        }
        # P1-4: 结构化节点缓存（nodes.jsonl），供上下文补全使用；
        # 同样以 (mtime_ns, size) 签名做失效判断。
        self._nodes: tuple[_CacheSig, list[dict]] | None = None
        # P1-6: hybrid 各路召回的 RRF 融合权重，顺序固定为
        # [keyword, bm25, vector(如可用)]；None 表示等权。
        # 由评测集调节 (edr_rag/eval/tune.py) 或直接 set_fusion_weights 设置。
        self.fusion_weights: list[float] | None = None
        # P4: (corpus_type, 文件签名, acl 指纹) -> 允许行号集合。
        self._acl_cache: dict[tuple, set[int]] = {}

    def set_fusion_weights(self, weights: list[float] | None) -> None:
        """设置 hybrid 的 RRF 融合权重（顺序 keyword/bm25/vector）。

        Args:
            weights: 非负权重列表，长度 2 或 3（有 vector 时为 3）；
                None 恢复等权。长度超出实际召回路数时仅取前面可用项。
        """
        if weights is not None and any(w < 0 for w in weights):
            raise ValueError("fusion weights must be non-negative")
        self.fusion_weights = weights

    # ---------- Corpus 懒加载 ----------

    def _corpus(self, corpus_type: str) -> Optional[JsonlCorpus]:
        """按 corpus_type 返回缓存 corpus；文件不存在或为空时返回 None。

        P0-6: 以文件 (mtime_ns, size) 签名做失效判断——同一服务实例多次检索
        时不重复读文件/重建索引；一旦 JSONL 被重建 (例如新跑一遍 ingest)，
        签名变化即自动重载，无需重建整个服务。
        """
        if corpus_type not in _CORPUS_FILES:
            raise ValueError(
                f"unknown corpus_type '{corpus_type}'; "
                f"expected one of {sorted(_CORPUS_FILES)}"
            )
        path = self.workdir / _CORPUS_FILES[corpus_type]
        sig = _file_signature(path)

        entry = self._corpora.get(corpus_type)
        if entry is not None and entry[0] == sig:
            # 命中：签名未变，直接复用内存中的 corpus (含已建好的索引)。
            return entry[1]

        if sig is None:
            # 文件缺失 (或已删除)：不缓存 corpus，返回 None。
            self._corpora[corpus_type] = None
            return None

        # 首次加载或签名变化 -> 重建 corpus 及其内存索引。
        corpus = JsonlCorpus.from_file(path)
        self._corpora[corpus_type] = (sig, corpus)
        return corpus

    def representative_map(self, corpus_type: str) -> dict[str, str]:
        """返回该语料内容级去重的 id→代表 映射（只读副本）。

        供 synthesize 层引用归一与 eval gold 对齐共用（[0440] 结论三 /
        [0512]：三处消费同一份映射，不各自维护）。corpus 未加载/文件缺失
        时返回空 dict。
        """
        corpus = self._corpus(corpus_type)
        if corpus is None:
            return {}
        return dict(corpus._rep_of)

    # ---------- ACL 谓词下推 (P4) ----------

    def _acl_fingerprint(self, acl: ACLPolicy) -> tuple:
        """把 ACLPolicy 归一成可哈希的缓存键。"""
        return (
            tuple(sorted(map(str, acl.tenants))) if acl.tenants else None,
            tuple(sorted(map(str, acl.devices))) if acl.devices else None,
            bool(acl.restrict_unlabeled),
        )

    def _acl_allowed_rows(
        self,
        corpus_type: str,
        acl: Optional[ACLPolicy],
    ) -> Optional[set[int]]:
        """返回该策略下允许可见的语料行号集合；不受限时返回 None。

        结果按 (corpus 文件签名, acl 指纹) 缓存 —— 同一个调用方连续查询
        不必反复遍历整个语料判权限。语料文件变化时签名变，缓存自动失效。

        Returns:
            允许的行号集合；None 表示不限制（调用方据此跳过全部过滤逻辑）。
            空集合表示「一条都看不到」，调用方应直接短路。
        """
        if acl is None or acl.unrestricted:
            return None
        corpus = self._corpus(corpus_type)
        if corpus is None:
            return set()
        sig = _file_signature(self.workdir / _CORPUS_FILES[corpus_type])
        key = (corpus_type, sig, self._acl_fingerprint(acl))
        cached = self._acl_cache.get(key)
        if cached is not None:
            return cached
        allowed = {
            i for i, rec in enumerate(corpus.records) if acl.allows(rec)
        }
        # 只保留最近若干份，避免多租户场景把缓存撑爆。
        if len(self._acl_cache) >= 32:
            self._acl_cache.clear()
        self._acl_cache[key] = allowed
        return allowed

    def _allowed_node_ids(self, acl: Optional[ACLPolicy]) -> Optional[set[str]]:
        """返回 ACL 允许可见的结构化节点 id 集合；不受限时 None (P4)。"""
        if acl is None or acl.unrestricted:
            return None
        return {
            str(n.get("id"))
            for n in self._load_nodes()
            if n.get("id") and acl.allows(n)
        }

    @staticmethod
    def _scrub_node_refs(nodes: list[dict], allowed_ids: set[str]) -> list[dict]:
        """过滤越权节点，并抹掉指向它们的**悬空引用** (P4)。

        只过滤节点本身是不够的：一个被授权的父节点，它的 ``children`` 数组
        里仍然列着越权子节点的 id；``prev_id`` / ``next_id`` 同理。内容虽然
        没泄露，但 **id 本身就是信息** —— 它告诉调用方「这里还存在一个你看
        不到的步骤」，暴露了受限文档的存在性与数量。

        所以过滤之后要把所有跨界引用一并清掉，让受限视图看起来是自洽的。
        """
        out: list[dict] = []
        for n in nodes:
            if str(n.get("id")) not in allowed_ids:
                continue
            n2 = dict(n)
            for key in ("parent_id", "prev_id", "next_id"):
                if n2.get(key) and str(n2[key]) not in allowed_ids:
                    n2[key] = ""
            children = n2.get("children")
            if isinstance(children, list):
                n2["children"] = [
                    c for c in children if str(c) in allowed_ids
                ]
            out.append(n2)
        return out

    # ---------- 搜索 ----------

    def search(
        self,
        *args,
        include_full_text: bool = False,
        **kwargs,
    ) -> list[dict]:
        """统一检索入口（对外）。

        这是一层薄包装：真正的检索在 ``_search_impl``，这里只负责在返回前
        剥掉内部字段（以 ``_`` 开头，如 ``_rerank_text``）。内部字段是检索
        链路自己用的（例如喂给 cross-encoder 的完整正文），不该出现在
        API / MCP / CLI 的响应里。

        include_full_text=True 供喂给 LLM 的合成链路使用：``_rerank_text``
        是 1200 字完整正文，LLM 只看 120 字 snippet 会把「证据充分」误判为
        「证据不足」（实测阳性对照全误拒）。开启时按 id 回填正文并保留该
        字段；默认 False 时行为与之前逐字节一致。
        """
        results = self._search_impl(*args, **kwargs)
        if include_full_text:
            corpus_type = (
                args[1] if len(args) >= 2 else kwargs.get("corpus_type", "actions")
            )
            return self._attach_full_text(corpus_type, results)
        return _strip_internal_fields(results)

    def _attach_full_text(
        self, corpus_type: str, results: list[dict]
    ) -> list[dict]:
        """给检索结果回填 ``_rerank_text``（完整正文字段，供 LLM 上下文）。

        动作富字段与 generic 薄形状都不携带正文，这里按 id 解析回语料原始
        记录后统一用 ``JsonlCorpus._rerank_text`` 生成（与 rerank 打分用的
        是同一字段同一规范化，上限 RERANK_TEXT_MAX）。解析不到原始记录的
        结果原样保留。
        """
        corpus = self._corpus(corpus_type)
        if corpus is None or not results:
            return results
        by_id: dict[str, dict] = {}
        for rec in corpus.records:
            for key in (rec.get("_doc_id"), rec.get("id"), rec.get("action_id")):
                if key:
                    by_id.setdefault(str(key), rec)
        out: list[dict] = []
        for r in results:
            rid = r.get("id") or r.get("action_id")
            rec = by_id.get(str(rid)) if rid is not None else None
            if rec is None:
                out.append(r)
                continue
            r2 = dict(r)
            r2["_rerank_text"] = corpus._rerank_text(rec)
            out.append(r2)
        return out

    def _search_impl(
        self,
        query: str,
        corpus_type: str = "actions",
        strategy: str = _DEFAULT_STRATEGY,
        filters: Optional[dict[str, Any]] = None,
        top_k: int = 5,
        recall_budget: int = 4,
        rerank: Optional[bool] = None,
        with_context: bool = False,
        adjacency: int = 1,
        query_expansion: bool = True,
        acl: Optional[ACLPolicy] = None,
    ) -> list[dict]:
        """统一检索入口。

        Args:
            query: 检索词 (支持中文)。
            corpus_type: 'sections' | 'procedures' | 'actions' | 'chunks'。
            strategy: 'keyword' | 'bm25' | 'vector' | 'hybrid'；vector 缺依赖
                时降级 bm25；hybrid 走「多路召回 -> RRF 融合 -> (可选)
                cross-encoder rerank -> top K」(P1-3)。
            filters: dict，支持 'risk'/'status'（动作过滤）与 'id'（精确获取）。
            top_k: 返回条数上限。
            recall_budget: 动作检索的召回预算倍数 (P0-4)。
            rerank: cross-encoder 精排开关 (P1-3)；None 跟随全局默认。
            with_context: (P1-4) 结构化节点 (nodes) 的上下文补全开关。
            adjacency: (P1-4) 上下文补全相邻兄弟个数。
            query_expansion: (P1-6) 是否开启查询侧中文规范化 + 多查询生成。
                开启（默认）时先 normalize_query，再基于 SYNONYMS 生成多个
                查询变体做 multi-query 召回并 RRF 融合，提升召回；变体只有
                一个时行为与关闭完全一致（无额外开销）。关闭则仅规范化。
            acl: (P3-3) 文档级 ACL 过滤策略 (edr_rag.security.ACLPolicy)。
                None 或未受限策略时不做过滤（向后兼容主链路）；否则对
                返回结果按设备/租户 metadata 过滤。

        Returns:
            结果 dict 列表。actions 返回富字段；其余类型返回 JsonlCorpus
            标准结果 (id/type/score/title/source/snippet)。
        """
        filters = filters or {}
        top_k = max(int(top_k), 0)
        requested_top_k = top_k
        recall_budget = max(int(recall_budget), 1)
        # P4: 每次读环境变量而非 import 时求值一次，支持运行时调整。
        rerank_enabled = _rerank_default() if rerank is None else bool(rerank)

        # P4 (严重问题 6): ACL 谓词下推。
        #
        # P3-3 的做法是「先按全语料召回、再过滤」——受限查询会把 top_k 撑到
        # len(corpus.records)，于是整个语料都要打分，hybrid 下还要整个语料过一遍
        # cross-encoder 精排。受限租户的查询延迟随语料线性增长，正好惩罚了权限
        # 最小的那批用户。
        #
        # 现在先算出该策略下**允许可见的行号集合**（每个 (corpus, acl) 只算一次
        # 并缓存），把它下推给各路召回，越权文档根本不参与打分与精排。
        # top_k 保持调用方请求的值。
        acl_restricted = acl is not None and not acl.unrestricted
        allowed = self._acl_allowed_rows(corpus_type, acl) if acl_restricted else None
        if allowed is not None and not allowed:
            # 该策略下没有任何可见文档，直接短路。
            return []

        # 上下文补全现在是 ACL 感知的：只把允许可见的节点喂给 assemble_context，
        # 于是父/兄弟/窗口都只可能引用到已授权的节点。P3-3 是直接关掉补全
        # （受限用户功能降级），P4 不再需要那样。
        safe_with_context = with_context

        # 精确 id 获取模式：不做全文打分，直接按主键取。
        if filters.get("id") is not None:
            rec = self.get(corpus_type, filters["id"])
            if rec is None:
                return []
            result = (
                self._record_to_action_result(rec)
                if corpus_type == "actions"
                else dict(rec)
            )
            # P3-3: 精确获取同样受 ACL 约束——越权的主键不得泄露内容。
            # 在原始 record 上判定（富字段结果已丢失 device/tenant metadata）。
            if not acl_allows(acl, rec):
                return []
            return [result]

        if strategy not in _ALLOWED_STRATEGIES:
            raise ValueError(
                f"unknown strategy '{strategy}'; expected one of {_ALLOWED_STRATEGIES}"
            )

        strategy = self._resolve_strategy(strategy)

        # P1-6: 查询侧规范化 + 多查询生成（query expansion）。
        from .query import expand_query, normalize_query

        variants = (
            expand_query(query) if query_expansion else [normalize_query(query)]
        )
        if len(variants) <= 1:
            return self._filter_acl_corpus(
                corpus_type,
                self._search_single(
                    variants[0] if variants else query,
                    corpus_type, strategy, filters,
                    top_k, recall_budget, rerank_enabled,
                    safe_with_context, adjacency, allowed, acl,
                ),
                acl,
            )[:requested_top_k]
        # 多查询：逐变体召回一路，按变体位置降权 RRF 融合，再截断 top_k。
        variant_weights = [
            1.0 / (1.0 + 0.3 * i) for i in range(len(variants))
        ]
        per_variant: list[list[dict]] = []
        for v in variants:
            res = self._search_single(
                v, corpus_type, strategy, filters,
                top_k, recall_budget, rerank_enabled,
                False, adjacency, allowed, acl,
            )
            if res:
                per_variant.append(res)
        if not per_variant:
            return []
        fused = rrf_fuse(per_variant, weights=variant_weights)
        if safe_with_context and corpus_type == "nodes" and fused:
            fused = self._enrich_with_context(fused, adjacency=adjacency, acl=acl)
        # P3-3: 多查询融合后同样应用文档级 ACL 过滤。
        return self._filter_acl_corpus(corpus_type, fused, acl)[:requested_top_k]

    def _filter_acl_corpus(
        self,
        corpus_type: str,
        results: list[dict],
        acl: Optional[ACLPolicy],
    ) -> list[dict]:
        """对已生成的检索结果做文档级 ACL 过滤（P3-3）。

        结果形状（thin `_to_result` / 动作富字段）可能已丢失 device/tenant/
        metadata 字段，因此这里把每个结果解析回它在语料中的**原始记录**，
        再以原始记录判定 ACL，保证对 actions/sections/procedures/chunks
        四类语料、keyword/bm25/vector/hybrid 全部策略都一致生效。
        """
        if acl is None or acl.unrestricted:
            return results
        corpus = self._corpus(corpus_type)
        if corpus is None or not results:
            return []
        # 建立 id -> 原始记录 的映射（_doc_id / id / action_id 三种键）。
        by_id: dict[str, dict] = {}
        for rec in corpus.records:
            for key in (rec.get("_doc_id"), rec.get("id"), rec.get("action_id")):
                if key:
                    by_id[str(key)] = rec
        out: list[dict] = []
        for r in results:
            rid = r.get("id") or r.get("action_id")
            if rid is None:
                continue
            rec = by_id.get(str(rid))
            if rec is not None and acl_allows(acl, rec):
                out.append(r)
        return out

    def _search_single(
        self,
        query: str,
        corpus_type: str,
        strategy: str,
        filters: dict[str, Any],
        top_k: int,
        recall_budget: int,
        rerank_enabled: bool,
        with_context: bool,
        adjacency: int,
        allowed: Optional[set[int]] = None,
        acl: Optional[ACLPolicy] = None,
    ) -> list[dict]:
        """对单个查询执行检索（keyword/bm25/vector/hybrid），供 search 分发。

        ``allowed`` 是 ACL 允许的行号集合 (P4 谓词下推)；None 表示不限制。
        """

        if strategy == "hybrid":
            results = self._search_hybrid(
                query,
                corpus_type,
                filters=filters,
                top_k=top_k,
                recall_budget=recall_budget,
                rerank_enabled=rerank_enabled,
                allowed=allowed,
            )
            if with_context and corpus_type == "nodes" and results:
                results = self._enrich_with_context(
                    results, adjacency=adjacency, acl=acl
                )
            return results

        if corpus_type == "actions":
            results = self._search_actions(
                query,
                strategy=strategy,
                filters=filters,
                top_k=top_k,
                recall_budget=recall_budget,
                allowed=allowed,
            )
        else:
            corpus = self._corpus(corpus_type)
            if corpus is None:
                return []
            results = self._search_generic(corpus, query, strategy, top_k, allowed)
            results = _apply_filter(results, filters)

        # P1-4: 结构化节点语料的上下文补全（仅 nodes 语料、且 with_context=True）
        if with_context and corpus_type == "nodes" and results:
            results = self._enrich_with_context(results, adjacency=adjacency, acl=acl)

        return results

    @staticmethod
    def _resolve_strategy(strategy: str) -> str:
        """把可能降级的策略解析为实际执行的检索策略。

        - 'vector' 的缺依赖降级由底层 JsonlCorpus.vector_search 处理。
        - 其余 ('keyword'/'bm25'/'hybrid') 原样返回；hybrid 由本服务
          的 `_search_hybrid` 执行完整的融合 + rerank 流程 (P1-3)。
        """
        return strategy

    def _search_actions(
        self,
        query: str,
        strategy: str,
        filters: dict[str, Any],
        top_k: int,
        recall_budget: int = 4,
        allowed: Optional[set[int]] = None,
    ) -> list[dict]:
        """动作目录检索，输出富字段结果。

        P0-4 召回预算逻辑:
          先按扩大后的候选预算 (top_k * recall_budget) 打分排序，再对候选
          池应用 metadata 过滤 (risk/status)，避免「在打分循环内过滤」导致
          结果不足或漏召回。若过滤后条数仍不足 top_k，且候选池尚未覆盖整个
          语料，则循环倍增预算继续打分补召回，直到满足或耗尽语料为止。

        keyword 策略对 title/action_id 加权，与历史 EdrRagAPI.action_search
        行为保持一致。
        """
        corpus = self._corpus("actions")
        if corpus is None:
            return []
        total = len(corpus.records) if allowed is None else len(allowed)
        budget = max(top_k * recall_budget, top_k)

        while True:
            scored = self._score_actions(corpus, query, strategy, budget, allowed)
            filtered = _apply_filter(scored, filters, action_shape=True)
            if len(filtered) >= top_k or budget >= total:
                break
            budget = min(total, max(budget * 2, budget + 1))

        return filtered[:top_k]

    # ---------- hybrid (P1-3): 多路召回 -> RRF 融合 -> rerank -> top K ----------

    def _search_hybrid(
        self,
        query: str,
        corpus_type: str,
        filters: dict[str, Any],
        top_k: int,
        recall_budget: int,
        rerank_enabled: bool,
        allowed: Optional[set[int]] = None,
    ) -> list[dict]:
        """hybrid 完整流程。

        多路召回 (keyword + bm25，有 embedder 时叠加 vector) 各自打分，
        用 RRF 融合成单一排名，然后 (可选) 做 cross-encoder rerank 精排，
        最后应用 metadata 过滤并取 top K。

        actions 复用 P0-4 的召回预算循环：融合后按预算扩候选池、过滤、
        不足则倍增预算，直到满足 top_k 或耗尽语料；rerank 在过滤后的
        最终候选池上执行一次 (避免每个循环迭代都跑昂贵的精排)。
        """
        corpus = self._corpus(corpus_type)
        if corpus is None:
            return []

        if corpus_type == "actions":
            return self._search_actions_hybrid(
                corpus, query, filters, top_k, recall_budget, rerank_enabled, allowed
            )
        return self._search_generic_hybrid(
            corpus, query, filters, top_k, rerank_enabled, allowed
        )

    def _embedder_available(self) -> bool:
        """embedding 是否可用 (用于决定 hybrid 是否叠加 vector 一路召回)。

        1433 修复：`_optional_embedder` 现含真嵌入探针——import 可用但模型
        加载/推理失败时返回 None，此处随之判 False，与 vector 路实际行为一致。
        """
        from .p1.search import _optional_embedder

        return _optional_embedder() is not None

    @staticmethod
    def embedder_status() -> str:
        """嵌入器通道状态 ('active' | 'failed' | 'absent')。

        评测/服务应在输出头部声明，使归档数值自带模式证据。
        """
        from .p1.search import embedder_status

        return embedder_status()

    def _search_generic_hybrid(
        self,
        corpus: JsonlCorpus,
        query: str,
        filters: dict[str, Any],
        top_k: int,
        rerank_enabled: bool,
        allowed: Optional[set[int]] = None,
    ) -> list[dict]:
        """generic 语料 (sections/procedures/chunks) 的融合检索。"""
        # 每路召回带上限 (top_k * 固定倍数)，保证融合候选池有足够区分度。
        recall_cap = max(top_k * 4, top_k)

        lists: list[list[dict]] = []
        kw = corpus.keyword_search(query, limit=recall_cap, allowed=allowed)
        if kw:
            lists.append(kw)
        bm = corpus.bm25_search(query, limit=recall_cap, allowed=allowed)
        if bm:
            lists.append(bm)
        # 有嵌入器时叠加 vector 一路。fallback='empty'：运行中失败返回空，
        # 绝不把 BM25 结果伪装成向量路重复计入融合（1433 事故：
        # keyword+BM25+BM25 改变排序且无观测痕迹）。缺依赖/探针失败时
        # _optional_embedder() 返回 None，vector 路整体不参与。
        if self._embedder_available():
            vec = corpus.vector_search(
                query, limit=recall_cap, allowed=allowed, fallback="empty"
            )
            if vec:
                lists.append(vec)

        if not lists:
            return []
        fused = rrf_fuse(lists, weights=self.fusion_weights)
        fused = _apply_filter(fused, filters)
        reranked, _ = rerank(query, fused, enabled=rerank_enabled)
        return reranked[:top_k]

    def _search_actions_hybrid(
        self,
        corpus: JsonlCorpus,
        query: str,
        filters: dict[str, Any],
        top_k: int,
        recall_budget: int,
        rerank_enabled: bool,
        allowed: Optional[set[int]] = None,
    ) -> list[dict]:
        """actions 的融合检索，保留 P0-4 召回预算 + metadata 过滤循环。"""
        total = len(corpus.records) if allowed is None else len(allowed)
        budget = max(top_k * recall_budget, top_k)

        filtered: list[dict] = []
        while True:
            fused = self._hybrid_score_actions(corpus, query, budget, allowed)
            filtered = _apply_filter(fused, filters, action_shape=True)
            if len(filtered) >= top_k or budget >= total:
                break
            budget = min(total, max(budget * 2, budget + 1))

        reranked, _ = rerank(query, filtered, enabled=rerank_enabled)
        return reranked[:top_k]

    def _hybrid_score_actions(
        self,
        corpus: JsonlCorpus,
        query: str,
        budget: int,
        allowed: Optional[set[int]] = None,
    ) -> list[dict]:
        """对动作语料做多路召回并 RRF 融合，返回前 `budget` 个候选富字段结果。

        召回侧: keyword (富字段) + bm25 (+ 有 embedder 时的 vector)。
        用动作富字段结果去融合 (id = action_id)，保证后续 metadata 过滤
        有 risk/status 可用。
        """
        lists: list[list[dict]] = []

        kw = self._keyword_actions(corpus, query, budget, allowed)
        if kw:
            lists.append(kw)

        bm_pairs = self._bm25_rank(corpus, query, allowed)[:budget]
        if bm_pairs:
            lists.append([self._action_result(corpus, i, s) for s, i in bm_pairs])

        # 仅为避免重复 bm25 (vector 缺依赖降级) 才只叠加一路；embedder 可用
        # 时 vector 提供独立的语义信号。
        if self._embedder_available():
            vec_pairs = self._vector_rank(corpus, query, allowed=allowed)[:budget]
            if vec_pairs:
                lists.append([self._action_result(corpus, i, s) for s, i in vec_pairs])

        if not lists:
            return []
        return rrf_fuse(lists, weights=self.fusion_weights)

    def _score_actions(
        self,
        corpus: JsonlCorpus,
        query: str,
        strategy: str,
        budget: int,
        allowed: Optional[set[int]] = None,
    ) -> list[dict]:
        """对语料打分排序，返回前 `budget` 个命中的动作富字段结果。

        bm25/vector 底层通用结果缺少 risk/status，动作的 metadata 过滤
        (risk/status) 依赖富字段，因此这里统一用动作富字段重打分排序，
        保证所有策略都能做一致的 metadata 过滤 (P0-4)。
        """
        if strategy == "bm25":
            pairs = self._bm25_rank(corpus, query, allowed)
        elif strategy == "vector":
            pairs = self._vector_rank(corpus, query, allowed=allowed)
        else:  # keyword
            return self._keyword_actions(corpus, query, budget, allowed)
        return [self._action_result(corpus, i, s) for s, i in pairs[:budget]]

    def _bm25_rank(
        self,
        corpus: JsonlCorpus,
        query: str,
        allowed: Optional[set[int]] = None,
    ) -> list[tuple[float, int]]:
        """轻量 BM25 打分，返回 (score, doc_idx) 降序列表 (动作富字段用)。

        ``allowed`` 限定参与打分的行号 (ACL 谓词下推，P4)。
        """
        from .p1.search import tokenize

        q_tokens = set(tokenize(query))
        if not q_tokens:
            return []
        n = len(corpus.records)
        idf_cache = {}
        for t in q_tokens:
            df = corpus._df.get(t, 0)
            idf_cache[t] = 0.0 if df == 0 else math.log((n - df + 0.5) / (df + 0.5) + 1)
        scores: dict[int, float] = {}
        for t in q_tokens:
            idf = idf_cache[t]
            if idf == 0:
                continue
            for doc_idx, tf in corpus._postings.get(t, []):
                if allowed is not None and doc_idx not in allowed:
                    continue
                dl = corpus._doc_len[doc_idx]
                tf_norm = (
                    (tf * (1.5 + 1)) / (tf + 1.5 * (1 - 0.75 + 0.75 * (dl / corpus._avgdl)))
                    if corpus._avgdl
                    else 0.0
                )
                scores[doc_idx] = scores.get(doc_idx, 0.0) + idf * tf_norm
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return [(s, i) for i, s in ranked]

    def _vector_rank(
        self,
        corpus: JsonlCorpus,
        query: str,
        limit: Optional[int] = None,
        vector_filter: Optional[dict[str, Any]] = None,
        allowed: Optional[set[int]] = None,
    ) -> list[tuple[float, int]]:
        """向量打分 (缺依赖时降级 BM25) 的 (score, doc_idx) 降序列表。

        P4: 改为复用与 ``JsonlCorpus.vector_search`` 相同的持久化向量索引。
        旧实现对**每条记录**调 ``_cached_vec`` 做 O(N) Python 循环余弦，且
        与 corpus.vector_search 各建一套缓存；现在两条路径共享同一个
        namespace 的向量存储，嵌入只做一次、落盘复用。

        Args:
            limit: 返回条数上限；None 表示取回整个语料的排序 (调用方自行截断)。
            vector_filter: metadata 谓词，在打分前下推到存储层 (ACL)。
        """
        from .p1.search import _get_shared_index, _optional_embedder

        embed_fn = _optional_embedder()
        if embed_fn is None:
            return self._bm25_rank(corpus, query, allowed)
        top_k = limit if limit is not None else len(corpus.records)
        if allowed is not None:
            # 受限时多召回一些，抵消随后按行号过滤造成的损耗。
            top_k = max(top_k, min(len(corpus.records), len(allowed) * 4))
        if top_k <= 0:
            return []
        try:
            index = _get_shared_index(
                embed_fn,
                namespace=corpus.namespace or "default",
                directory=corpus.vector_dir if corpus.namespace else None,
            )
            indexed = index.index_records(
                corpus.records, text_field="_searchable", id_field="_doc_id"
            )
            if indexed and corpus.vector_dir is not None:
                index.persist()
            hits = index.search(query, top_k=top_k, filter=vector_filter)
            if allowed is not None:
                hits = [(sc, i) for sc, i in hits if i in allowed]
            return hits
        except Exception:
            logging.getLogger(__name__).warning(
                "vector_score failed; falling back to BM25", exc_info=True
            )
            return self._bm25_rank(corpus, query, allowed)

    def _keyword_actions(
        self,
        corpus: JsonlCorpus,
        query: str,
        top_k: int,
        allowed: Optional[set[int]] = None,
    ) -> list[dict]:
        """动作专用 keyword 打分：对 action_id/title 加权。

        ``allowed`` 限定参与打分的行号 (ACL 谓词下推，P4)。
        """
        from .p1.search import tokenize

        q_tokens = set(tokenize(query))
        if not q_tokens:
            return []

        # 需要访问每个记录的搜索文本。直接复用 JsonlCorpus 内部字段。
        scored: list[tuple[float, int]] = []
        for i, r in enumerate(corpus.records):
            if allowed is not None and i not in allowed:
                continue
            field = r["_searchable"]
            title = r.get("_title", "")
            action_id = r.get("action_id", "")
            score = 0.0
            for t in q_tokens:
                if t in field:
                    weight = 3.0 if (t in title or t in action_id) else 1.0
                    score += field.count(t) * weight
            if score > 0:
                scored.append((score, i))
        scored.sort(key=lambda x: -x[0])
        return [self._action_result(corpus, i, s) for s, i in scored[:top_k]]

    def _action_result(self, corpus: JsonlCorpus, doc_idx: int, score: float) -> dict:
        r = corpus.records[doc_idx]
        result = self._record_to_action_result(r)
        result["score"] = round(score, 3)
        return result

    @staticmethod
    def _record_to_action_result(r: dict) -> dict:
        """把动作原始 record 转成统一富字段结果 (id = action_id)。"""
        snippet = r.get("title", "")
        pre = r.get("preconditions", [])
        if pre:
            snippet += " | 前置条件: " + "；".join(pre)
        return {
            "id": r.get("action_id", ""),
            "title": r.get("title", ""),
            "risk": r.get("risk", ""),
            "status": r.get("status", ""),
            "confidence": r.get("confidence", 0.0),
            "score": 0.0,
            "snippet": snippet,
        }

    def _search_generic(
        self,
        corpus: JsonlCorpus,
        query: str,
        strategy: str,
        top_k: int,
        allowed: Optional[set[int]] = None,
    ) -> list[dict]:
        if strategy == "bm25":
            return corpus.bm25_search(query, limit=top_k, allowed=allowed)
        if strategy == "vector":
            return corpus.vector_search(query, limit=top_k, allowed=allowed)
        return corpus.keyword_search(query, limit=top_k, allowed=allowed)

    # ---------- 精确获取 ----------

    def get(
        self,
        corpus_type: str,
        key: str,
        key_field: Optional[str] = None,
    ) -> Optional[dict]:
        """按主键精确获取一条记录 (actions 用 action_id，其余用 id)。

        Args:
            corpus_type: 同上。
            key: 主键值。
            key_field: 显式指定匹配字段；缺省 actions->action_id，其余->id。
        """
        corpus = self._corpus(corpus_type)
        if corpus is None:
            return None

        if key_field is None:
            key_field = "action_id" if corpus_type == "actions" else "id"

        target = key
        # 兼容 'action.' / 'procedure.' 前缀的 action_id 查询。
        if corpus_type == "actions":
            for prefix in ("action.", "procedure."):
                if target.startswith(prefix):
                    target = target[len(prefix):]
                    break

        for r in corpus.records:
            if r.get(key_field, "") == target:
                # 剥掉 JsonlCorpus 注入的内部检索字段，仅返回原始记录。
                return {k: v for k, v in r.items() if not k.startswith("_")}
        return None

    def list_actions(
        self,
        product: Optional[str] = None,
        module: Optional[str] = None,
        status: Optional[str] = None,
        acl: Optional[ACLPolicy] = None,
    ) -> list[dict]:
        """按条件枚举 action catalog（不做关键词匹配，design §3.3）。

        与 search 的区别：search 按相关性排序取 top-k，会漏掉低分条目；
        这里是无查询的全量枚举 + 过滤，供 agent 先看「目录里有什么」。

        Args:
            product: 按产品过滤；None 不过滤。
            module: 按模块过滤；None 不过滤。
            status: 按生命周期状态过滤 (candidate/needs_review/approved/
                published/deprecated)；None 不过滤。
            acl: ACL 策略；受限调用方只能枚举到有权见的记录。
        """
        corpus = self._corpus("actions")
        if corpus is None:
            return []
        allowed = self._acl_allowed_rows("actions", acl)
        out: list[dict] = []
        for i, r in enumerate(corpus.records):
            if allowed is not None and i not in allowed:
                continue
            if product is not None and r.get("product", "") != product:
                continue
            if module is not None and r.get("module", "") != module:
                continue
            if status is not None and r.get("status", "") != status:
                continue
            out.append({k: v for k, v in r.items() if not k.startswith("_")})
        return out

    # ---------- P1-4: 父子节点与相邻步骤上下文补全 ----------

    def _load_nodes(self) -> list[dict]:
        """懒加载结构化节点（nodes.jsonl），以 (mtime_ns, size) 签名做失效判断。

        文件缺失 / 空时返回 []，主链路不报错（上下文补全可选使用）。
        """
        path = self.workdir / "nodes.jsonl"
        sig = _file_signature(path)
        if self._nodes is not None and self._nodes[0] == sig:
            return self._nodes[1]
        if sig is None:
            self._nodes = None
            return []
        nodes = _read_jsonl(path)
        # 缺陷④b：此路径绕过 JsonlCorpus 防御层（P1-4 上下文补全消费
        # node.text / metadata.recovery），须同口径剥离 DOCTYPE 泄漏。
        nodes = JsonlCorpus._strip_doctype_leak(nodes)
        self._nodes = (sig, nodes)
        return nodes

    def context_for(
        self,
        node_id: str,
        *,
        adjacency: int = 1,
        include_section: bool = True,
        acl: Optional[ACLPolicy] = None,
    ) -> Optional[dict]:
        """为命中节点组装「父子节点与相邻步骤上下文补全」(P1-4)。

        基于 nodes.jsonl 里 P1-2 的结构化节点关系（parent_id / prev_id /
        next_id / children），返回包含所属 procedure 的 preconditions /
        warnings / recovery / expected_result、相邻 steps、祖先 section 的
        结构化上下文。nodes.jsonl 缺失或 node_id 不存在时返回 None。

        Args:
            node_id: 命中节点的 id（nodes.jsonl 中的 id，如 node.step.x.s1）。
            adjacency: 相邻兄弟个数（每侧）。
            include_section: 是否附带祖先 section。

        Returns:
            assemble_context 的结构化 dict；缺 nodes.jsonl 或未知 id 时 None。
        """
        nodes = self._load_nodes()
        if not nodes:
            return None
        # P4: 越权节点不进上下文，且越权的 node_id 本身不得泄露内容。
        allowed_ids = self._allowed_node_ids(acl)
        if allowed_ids is not None:
            if str(node_id) not in allowed_ids:
                return None
            nodes = self._scrub_node_refs(nodes, allowed_ids)
        from .p1.context import assemble_context

        snodes = _node_objects(nodes)
        ctx = assemble_context(
            node_id, snodes,
            adjacency=adjacency,
            include_section=include_section,
        )
        # 未知 node_id：assemble_context 返回非抛错的空形状，这里收敛为 None
        if ctx.get("hit") is None:
            return None
        return ctx

    def _enrich_with_context(
        self,
        results: list[dict],
        *,
        adjacency: int = 1,
        acl: Optional[ACLPolicy] = None,
    ) -> list[dict]:
        """给结构化节点检索结果附加 `context` 键（P1-4 上下文补全）。

        每个结果按 id 命中节点，调用 assemble_context（结构化 dict）并渲染
        出文本；节点加载失败或某条结果无法命中时，该条结果保持原样。

        P4: **ACL 感知**。只把该策略允许可见的节点喂给 assemble_context ——
        于是父节点、相邻兄弟、上下文窗口都只可能引用到已授权节点，不需要
        事后再擦一遍。P3-3 的做法是对受限查询直接关掉补全（受限用户功能
        降级），那既伤体验又没必要。
        """
        nodes = self._load_nodes()
        if not nodes:
            return results
        allowed_ids = self._allowed_node_ids(acl)
        if allowed_ids is not None:
            nodes = self._scrub_node_refs(nodes, allowed_ids)
            if not nodes:
                return results
        from .p1.context import assemble_context, render_context_text

        snodes = _node_objects(nodes)
        by_id = {n.id: n for n in snodes}

        enriched = []
        for r in results:
            rid = r.get("id", "")
            if rid in by_id:
                ctx = assemble_context(
                    rid, snodes,
                    adjacency=adjacency,
                    include_section=True,
                )
                r = dict(r)
                r["context"] = ctx
                r["context_text"] = render_context_text(ctx)
            enriched.append(r)
        return enriched

    # ---------- Planner 编排 (action -> procedure -> manual) ----------

    def planner_retrieve(
        self,
        user_request: str,
        limit: int = 5,
        strategy: str = _DEFAULT_STRATEGY,
        rerank: Optional[bool] = None,
        acl: Optional[ACLPolicy] = None,
        include_full_text: bool = False,
    ) -> dict:
        """完整的 planner 检索流程，保持既有优先级:

          1. 先检索动作目录 (actions)
          2. 无动作结果 -> 检索流程 (procedures)
          3. 仍无结果 -> 检索手册章节 (sections)

        传入的 strategy 会被透传给每一层检索，保证 Planner、API、CLI、
        MCP 共享同一套可配置检索策略。`rerank` 同样透传 (P1-3)，详见
        `search`。`acl` (P3-3) 透传给每一层做文档级过滤。
        `include_full_text` (见 `search`) 透传：合成链路需要完整正文。

        Returns:
            dict: {query, actions, [procedures], [sections]}
        """
        actions = self.search(
            user_request, corpus_type="actions", top_k=limit,
            strategy=strategy, rerank=rerank, acl=acl,
            include_full_text=include_full_text,
        )
        result: dict[str, Any] = {"query": user_request, "actions": actions}

        if not actions:
            procedures = self.search(
                user_request, corpus_type="procedures", top_k=limit,
                strategy=strategy, rerank=rerank, acl=acl,
                include_full_text=include_full_text,
            )
            result["procedures"] = procedures
            if not procedures:
                sections = self.search(
                    user_request, corpus_type="sections", top_k=limit,
                    strategy=strategy, rerank=rerank, acl=acl,
                    include_full_text=include_full_text,
                )
                result["sections"] = sections
        else:
            result["procedures"] = []
            result["sections"] = []

        return result


def _strip_internal_fields(results: list[dict]) -> list[dict]:
    """剥掉检索结果里的内部字段（``_`` 前缀），返回对外形状 (P4)。

    目前只有 ``_rerank_text``（喂给 cross-encoder 的完整正文）。它必须
    存在于检索链路内部、但不能出现在响应里 —— 否则每条结果都要多带
    上千字符的正文，MCP / HTTP 响应体会膨胀一个数量级。
    """
    out: list[dict] = []
    for r in results:
        if any(k.startswith("_") for k in r):
            out.append({k: v for k, v in r.items() if not k.startswith("_")})
        else:
            out.append(r)
    return out


# ---------- 过滤辅助 ----------

def _apply_filter(
    results: list[dict],
    filters: dict[str, Any],
    action_shape: bool = False,
) -> list[dict]:
    """对结果应用健康过滤 (risk/status)。

    action_shape=True 时从富字段读取 risk/status；否则动作也回落到原始
    record 字段 (generic 结果没有 risk/status 键，直接跳过)。
    """
    risk = filters.get("risk")
    status = filters.get("status")
    if not risk and not status:
        return results

    out = []
    for r in results:
        if risk and r.get("risk") != risk:
            continue
        if status and r.get("status") != status:
            continue
        out.append(r)
    return out
