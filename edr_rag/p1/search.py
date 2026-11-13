"""P1 local JSONL search — keyword + BM25.

Runs over the artifacts written by P0 (sections.jsonl) and P1
(procedures.jsonl, chunks.jsonl). No vector DB. Deterministic.

Search strategies (design §5):
  - keyword: substring / token overlap on text + title + path
  - bm25: lightweight BM25 over the corpus (used when --scoring bm25)
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import threading
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Callable, Optional

from edr_rag.query import normalize_query

logger = logging.getLogger(__name__)

# ---------- Tokenization (CJK-aware) ----------

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]+")
_WORD_RE = re.compile(r"[A-Za-z0-9_]+")
_SPLIT_RE = re.compile(r"[^\u4e00-\u9fffA-Za-z0-9_]+")


#: P4 Lock 4 —— 是否产出 CJK bigram。可用 EDR_RAG_CJK_BIGRAM=0 关闭以回到
#: P1 的纯 unigram 行为 (仅用于对照实验，不建议生产关闭)。
CJK_BIGRAM = os.environ.get("EDR_RAG_CJK_BIGRAM", "1") != "0"

#: jieba 词典分词探测结果 (模块级单例，语义同 _optional_embedder)。
_jieba_checked = False
_jieba_cut = None


def _optional_jieba():
    """返回 jieba 的切词函数；未安装时返回 None (bigram 已能兜住主要收益)。"""
    global _jieba_checked, _jieba_cut
    if _jieba_checked:
        return _jieba_cut
    try:
        import jieba  # type: ignore

        # search 模式产出更细粒度的重叠切分，利于召回。
        _jieba_cut = jieba.lcut_for_search
    except Exception:  # pragma: no cover - optional dep absent
        _jieba_cut = None
    _jieba_checked = True
    return _jieba_cut


def reset_tokenizer_cache() -> None:
    """重置 jieba 探测缓存 (测试用)。"""
    global _jieba_checked, _jieba_cut
    _jieba_checked = False
    _jieba_cut = None


def tokenize(text: str) -> list[str]:
    """把文本切成检索 token：拉丁词整词 + CJK unigram + CJK bigram。

    P1-6: 先做与查询侧一致的 NFKC 规范化（全半角/大小写），保证查询 token
    与文档 token 对称，全角查询也能命中半角语料。

    P4 Lock 4 —— 加 bigram。纯 unigram 的中文 BM25 有个经典缺陷：词序信息
    完全丢失，「安全」与「全安」的 token 集合完全相同因而同分，「网络代理」
    与「代理网络」也无法区分。bigram 是**纯 stdlib** 就能拿到大部分收益的
    方案，不给内网离线部署增加任何依赖。

    bigram 只在**连续 CJK 片段内部**生成，不跨标点/拉丁字符组合，避免
    「代理。配置」被拼出无意义的「理配」。

    jieba 可用时额外叠加词典切分的长词（>2 字），进一步提升领域词精度；
    不可用时静默跳过。

    Returns:
        token 列表，顺序为 [拉丁词..., CJK unigram..., CJK bigram..., jieba 长词...]。
        保留重复项 —— BM25 需要词频。
    """
    text = normalize_query(text)
    tokens: list[str] = []
    # 拉丁词 + 数字整词
    for w in _WORD_RE.findall(text):
        tokens.append(w)
    # CJK 单字 unigram（保底召回）
    for m in _CJK_RE.finditer(text):
        tokens.append(m.group(0))
    # CJK bigram（词序/搭配精度），仅在连续汉字片段内部滑窗
    if CJK_BIGRAM:
        for run in _CJK_RUN_RE.findall(text):
            for i in range(len(run) - 1):
                tokens.append(run[i:i + 2])
    # 可选：jieba 词典切分，补充 2 字以上的领域词
    cut = _optional_jieba()
    if cut is not None:
        try:
            for w in cut(text):
                w = w.strip()
                # 单字已由 unigram 覆盖、2 字已由 bigram 覆盖，这里只补更长的词
                if len(w) > 2 and _CJK_RE.search(w):
                    tokens.append(w)
        except Exception:  # pragma: no cover - optional path
            logger.debug("jieba tokenization failed; using bigram tokens only")
    return tokens


# ---------- Optional embedding helpers (graceful degradation) ----------

#: 嵌入模型名。可用 EDR_RAG_EMBED_MODEL 覆盖 (离线部署指向本地模型目录)。
EMBED_MODEL = os.environ.get("EDR_RAG_EMBED_MODEL", "BAAI/bge-small-zh-v1.5")

#: 文档向量缓存上限。旧实现是无界 dict —— 长跑进程上是内存泄漏。
_EMBED_CACHE_MAXSIZE = int(os.environ.get("EDR_RAG_EMBED_CACHE_SIZE", "20000"))
_EMBEDDER_CACHE: "OrderedDict[str, list[float]]" = OrderedDict()
_EMBED_CACHE_LOCK = threading.Lock()

#: P4 关键修复 —— 模块级单例。
#:
#: 旧实现把 ``@lru_cache`` 定义在 ``_optional_embedder()`` **函数体内部**，
#: 于是每次调用都生成全新闭包 + 全新缓存，BGE 模型被反复从磁盘装载。实测
#: 每次 ~9–10 秒，而 retrieval.py 每个查询要调 1–2 次。这是 P4 的头号性能
#: 阻塞项 (见 P4-HARDENING-DESIGN.md §4 Lock 2)。
#:
#: 契约：同一进程内 ``_optional_embedder()`` 必须返回**同一个函数对象**。
#: 由 tests/test_embedder_singleton.py 锁死。
_embedder_checked = False
_embedder_fn: Optional[Callable[[str], list[float]]] = None
_embedder_model = None
_embedder_lock = threading.Lock()

#: 嵌入器真可用性探测（20261112-1433 事故修复）：import 可用不代表模型
#: 可加载/可推理。首次使用前试嵌入一个固定探针文本，失败即判不可用
#: （进程内缓存判定结果）。embedder_status() 报告三态：
#:   active  探针嵌入成功，vector 路真实可用
#:   failed  import 可用但探针失败（模型加载/推理不可用）——不可用
#:   absent  依赖不可导入——不可用
_PROBE_TEXT = "向量通道探针"
_probe_done = False
_probe_ok: Optional[bool] = None


def embedder_status() -> str:
    """返回嵌入器通道状态：'active' | 'failed' | 'absent'。

    触发探测（进程内仅一次）。评测/检索链应在输出头部声明该状态，
    使归档数值自带模式证据（keyword+bm25+vector 三路 / 两路）。
    """
    fn = _optional_embedder()
    if fn is None:
        # 区分 failed（import 过但探针失败）与 absent（依赖缺失）
        if _probe_done and _probe_ok is False:
            return "failed"
        if _embedder_checked and _embedder_fn is None:
            return "absent"
        # 未探测过时探测一次
        _probe_embedder()
        if _probe_done and _probe_ok is False:
            return "failed"
        return "absent"
    # fn 非 None：确保探针已真实运行过（1433 教训——"看似可用"必须验证）
    if not _probe_done:
        if not _probe_embedder():
            return "failed"
    elif _probe_ok is False:
        return "failed"
    return "active"


def _probe_embedder() -> bool:
    """真嵌入探针文本，判定并缓存嵌入器可用性。幂等。"""
    global _probe_done, _probe_ok
    with _embedder_lock:
        if _probe_done:
            return bool(_probe_ok)
        _probe_done = True
        _probe_ok = False
        fn = _embedder_fn
        if fn is None:
            # import 探测尚未运行时先跑一遍
            fn = _optional_embedder()
        if fn is None:
            return False
        try:
            vec = fn(_PROBE_TEXT)
            _probe_ok = isinstance(vec, list) and len(vec) > 0
        except Exception:
            logger.error(
                "embedder probe failed: import 可用但模型加载/推理失败，"
                "vector 路判定为不可用（不静默降级伪装）",
                exc_info=True,
            )
            _probe_ok = False
        return bool(_probe_ok)


def _load_embed_model():
    """惰性构造嵌入模型 (进程内仅一次)。"""
    global _embedder_model
    if _embedder_model is None:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # type: ignore

        _embedder_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
    return _embedder_model


def _optional_embedder() -> Optional[Callable[[str], list[float]]]:
    """Return an embedding function if the optional `rag` extras are present.

    Uses LlamaIndex's HuggingFaceEmbedding when importable; otherwise returns
    None and callers (vector_search) fall back to BM25. Never raises on a
    missing optional dependency — the main chain stays on keyword/BM25.

    P4: 探测结果与模型实例均为模块级单例，重复调用零开销。

    1433 修复：返回非 None 前先做一次真嵌入探针；探针失败（模型加载/
    推理不可用）同样返回 None——import 可用性不再是"向量路可用"的判据。
    探针失败原因经 embedder_status() 以 'failed' 状态与 error 日志暴露。
    """
    global _embedder_checked, _embedder_fn
    if _embedder_checked:
        # 1433 修复：探针已判失败时缓存值不再可信，返回 None。
        if _probe_done and _probe_ok is False:
            return None
        return _embedder_fn
    with _embedder_lock:
        if _embedder_checked:  # 双检锁：等锁期间可能已被别的线程完成探测
            return _embedder_fn
        try:
            # 只探测可导入性，模型本身留到首次真正嵌入时再构造，
            # 避免「只是问一句 embedder 在不在」就付出装载成本。
            import llama_index.embeddings.huggingface  # type: ignore  # noqa: F401

            def _embed(text: str) -> list[float]:
                vec = _load_embed_model().get_text_embedding(text)
                return list(vec)

            # 供 vectorstore._embedder_sig 识别模型：换嵌入模型时内容签名
            # 随之失效，持久化短路不会服务旧模型的陈旧向量。
            _embed.model_id = f"hf:{EMBED_MODEL}"  # type: ignore[attr-defined]
            _embedder_fn = _embed
        except Exception:  # pragma: no cover - optional deps absent
            logger.debug("optional RAG embedder unavailable; using BM25 fallback")
            _embedder_fn = None
        _embedder_checked = True
    # 1433 修复：import 探测通过后立即做真嵌入探针。探针失败（模型加载/
    # 推理不可用）把嵌入器判为不可用——调用方拿到 None，vector_search 走
    # "缺依赖"分支，hybrid 不再把降级 BM25 伪装成向量路。
    if _embedder_fn is not None and not _probe_embedder():
        _embedder_fn = None
    return _embedder_fn


def reset_embedder_cache() -> None:
    """重置嵌入探测与向量缓存 (测试用；生产不应调用)。"""
    global _embedder_checked, _embedder_fn, _embedder_model
    global _probe_done, _probe_ok
    with _embedder_lock:
        _embedder_checked = False
        _embedder_fn = None
        _embedder_model = None
        _probe_done = False
        _probe_ok = None
    with _EMBED_CACHE_LOCK:
        _EMBEDDER_CACHE.clear()


def _cached_vec(embed_fn: Callable[[str], list[float]], text: str) -> Optional[list[float]]:
    """Cache per-document embedding vectors (LRU, keyed by content hash)."""
    key = hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()
    with _EMBED_CACHE_LOCK:
        vec = _EMBEDDER_CACHE.get(key)
        if vec is not None:
            _EMBEDDER_CACHE.move_to_end(key)
            return vec
    try:
        vec = embed_fn(text)
    except Exception:  # pragma: no cover - optional path only
        return None
    if vec:
        with _EMBED_CACHE_LOCK:
            _EMBEDDER_CACHE[key] = vec
            _EMBEDDER_CACHE.move_to_end(key)
            while len(_EMBEDDER_CACHE) > _EMBED_CACHE_MAXSIZE:
                _EMBEDDER_CACHE.popitem(last=False)
    return vec


def _cosine(a: list[float], b: list[float]) -> float:
    import math as _m

    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = _m.sqrt(sum(x * x for x in a))
    nb = _m.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


#: 进程内共享的 VectorIndex，**按 (namespace, 持久化目录) 缓存** (P4)。
#:
#: P3-1 的实现按 ``id(store)`` 缓存，而 ``get_vector_store()`` 全进程只返回
#: 一个 store —— 于是所有语料共用同一个 VectorIndex。``index_records`` 按整批
#: 内容签名短路，一换语料签名就不匹配，触发 ``delete(全部) + 全量重嵌``。
#: ``planner_retrieve`` 逐层走 actions → procedures → sections，三层会互相
#: 清空对方的向量，vector 策略下每层都要重嵌一次整个语料。
#:
#: P4 让每个语料拿到独立 namespace 的 store 与独立 VectorIndex。
_INDEX_CACHE: dict[tuple[str, str], Any] = {}
_INDEX_CACHE_LOCK = threading.Lock()


def _get_shared_index(
    embed_fn: Callable[[str], list[float]],
    namespace: str = "default",
    directory: Optional[Path] = None,
):
    """返回该 namespace 的进程内共享 VectorIndex，惰性创建。

    Args:
        embed_fn: 嵌入函数。
        namespace: 语料名 —— 隔离边界，不同语料必须不同。
        directory: 持久化目录；None 表示纯内存 (无 workdir 的调用方)。
    """
    from edr_rag.vectorstore import VectorIndex, get_vector_store

    dir_key = str(directory) if directory is not None else ""
    key = (namespace, dir_key)
    with _INDEX_CACHE_LOCK:
        idx = _INDEX_CACHE.get(key)
        if idx is None:
            cfg: dict[str, Any] = {}
            if directory is not None:
                cfg["directory"] = str(directory)
            else:
                # 无持久化目录时退回纯进程内实现，避免在 CWD 里乱写文件。
                cfg["backend"] = "simple"
            store = get_vector_store(namespace=namespace, config=cfg)
            idx = VectorIndex(store, embed_fn)
            _INDEX_CACHE[key] = idx
        return idx


def reset_index_cache() -> None:
    """清空 VectorIndex 缓存 (测试用；生产不应调用)。"""
    with _INDEX_CACHE_LOCK:
        _INDEX_CACHE.clear()


# ---------- Corpus ----------

#: 缺陷④ [0652] 读取侧防御：泄漏的 DOCTYPE 声明。根因在
#: parse_html._walk（Doctype 子类漏过 NavigableString 判定，已修），此处
#: 兜底存量产物（两手册 sections 100% 前缀泄漏 + chunks/nodes 前缀型与
#: 跨节拼接的中部型「恢复: PUBLIC …」）。签名取「PUBLIC "//DTD…" "URL"」
#: 双引号对——DTD 标识符必带 //DTD；前导 html 可缺（中部型）。
#: 实测两手册四语料 100% 覆盖、0 误伤（//DTD 签名正文不出现）。
_LEAKED_DOCTYPE_RE = re.compile(
    r'(?:\bhtml\s+)?\bPUBLIC\s+"[^"]*//DTD[^"]*"\s+"[^"]*"\s*',
    re.IGNORECASE,
)


class JsonlCorpus:
    """A searchable corpus over one or more JSONL artifact files.

    P4: 语料携带 ``namespace`` 与 ``vector_dir``，供向量层做隔离与持久化。
    ``namespace`` 由来源文件名推导 (sections.jsonl -> 'sections')；从裸记录
    构造时为 None，此时向量检索走纯内存、不落盘 (测试与轻量调用方)。
    """

    def __init__(
        self,
        records: list[dict],
        namespace: Optional[str] = None,
        vector_dir: Optional[Path] = None,
    ):
        # 内容级去重的 id→代表 映射；仅 from_file 路径填充。
        self._rep_of: dict[str, str] = {}
        # Normalize each record into a searchable doc
        self.records = []
        for r in records:
            self.records.append(
                self._normalize(self._strip_doctype_leak(r))
            )
        # 同 ID 异内容身份契约（1433 审核漏洞 #2）：检索等价去重后仍同 ID
        # 的行是**实质不同的变体**（实测 duli「登录_管理面」三变体浏览器
        # 要求不同）。若共享 _doc_id，向量 upsert 后写覆盖只剩最后一变体、
        # 增量索引按 ID 收敛会以最后变体文本错位重嵌入。给非首现变体分配
        # 带序号的内部 _doc_id（upsert 键唯一化）；公开 id 字段保持原值
        # （gold/引用/get() 语义不变），_to_result 公开口输出公开 id。
        # 必须在 _normalize 之后运行（_normalize 会重写 _doc_id = id）。
        self._disambiguate_duplicate_ids(self.records)

        self.namespace = namespace
        self.vector_dir = vector_dir

        # Build BM25 index
        self._df: Counter = Counter()          # term -> #docs containing
        self._postings: dict[str, list[tuple[int, int]]] = {}  # term -> [(doc_idx, tf)]
        self._doc_len: list[int] = []
        self._avgdl = 0.0
        self._build_bm25()

    @staticmethod
    def _dedupe_identical_id_rows(records: list[dict]) -> tuple[list[dict], int]:
        """同 ID 且检索语义完全一致的重复行只保留首次出现 (缺陷②最小缓解)。

        背景：ingest 按「导航路径」拼 section ID，同一导航路径在多个 topic
        HTML 反复出现导致跨文件 ID 碰撞（anquan 17 组 / duli 11 组重复）。
        其中「除 source_ref/links 外逐字段相同」的组在检索上是等价副本：
        keyword/bm25 重复计分挤占 top-k，vector upsert 按 ID 覆盖只剩一行。
        本函数在**读取侧**去重这类组（不改磁盘产物、不改 ID）；
        「内容确实不同」的同 ID 组保持原样（哪个变体可见属 ingest 缺陷，
        需另行决策），provenance 取保留行的 source_ref。

        等价判据：id + 参与检索/展示/上下文组装的全部字段相同
        (title/headings/section_path/content/metadata 及其余非 _ 字段)，
        仅 links/images/source_ref 允许不同——三者均不进入 _searchable
        (见 _normalize)，且检索链路无消费方。
        """
        if not records:
            return records, 0
        seen: dict[str, str] = {}
        out: list[dict] = []
        dropped = 0
        for r in records:
            rid = r.get("id") or ""
            payload = json.dumps(
                {k: v for k, v in r.items() if k not in ("source_ref", "links", "images")},
                sort_keys=True,
                ensure_ascii=False,
            )
            key = f"{rid}\x00{hashlib.sha1(payload.encode('utf-8')).hexdigest()}"
            if rid and key in seen:
                dropped += 1
                continue
            if rid:
                seen[key] = rid
            out.append(r)
        return out, dropped

    # 内容级去重的分组阈值：短内容（导航锚点/样板节）不参与分组，
    # 避免把合法短节误并（[0310] 可行性评估确定）。
    _CONTENT_DEDUP_MIN_LEN = 200

    @staticmethod
    def _dedupe_content_rows(
        records: list[dict],
    ) -> tuple[list[dict], dict[str, str]]:
        """不同 ID 但 content 逐字节相同的行只保留首现行为代表（读取侧）。

        背景：两本手册存在大面积同内容多 ID 节（anquan 66 组 / duli 212 组，
        [0258]/[0310] 已量化）：双配置树镜像 + 跨角色视角重复。相同内容 →
        相同向量 → 检索语义并列，挤占 top-k（duli vector 9 行因此 gold 出
        top-5）。本函数在读取侧移除非代表行（磁盘产物不改、ID 不改），
        组内差异经实证仅 section_path（导航树位置），title/metadata/headings
        逐字段一致（anquan/duli 66+212 组全部如此，实机核验）。

        前置要求：先执行 _dedupe_identical_id_rows（同 ID 等价行已收敛），
        否则同 ID 组内可能被本函数按内容错误并组。

        返回 (保留行列表, id→代表 id 映射)。映射必须通过
        ``representative_id`` 公共查询口消费——eval gold 重映射、引用渲染
        （synthesize 层）、nodes 语料重建后的评测对齐共用同一份映射，
        不各自维护（[0440] 结论三）。
        """
        if not records:
            return records, {}
        seen: dict[str, str] = {}
        out: list[dict] = []
        rep_of: dict[str, str] = {}
        for r in records:
            # 正文在 content（sections/procedures）或 text（chunks/nodes），
            # 与 _normalize 的取值口径一致；缺陷③ [0652]：此前只读 content，
            # chunks 语料静默不参与内容级去重。
            content = r.get("content") or r.get("text") or ""
            if len(content) < JsonlCorpus._CONTENT_DEDUP_MIN_LEN:
                out.append(r)
                continue
            # D-0659-1 方案 B：键升级为 content|text + title + operation。
            # chunks 组内 title/operation 可不同（[0659] 实测 31 组，含跨功能
            # 同文组），title/operation 参与 _searchable 打分 → 仅内容键会
            # 并组检索不等价的行。sections 组内差异仅 section_path（[0512]
            # 逐字段核验），title 一致时行为不变。
            meta = r.get("metadata") or {}
            title = r.get("title") or meta.get("title") or ""
            operation = r.get("operation") or meta.get("operation") or ""
            # Equal text does not imply equal applicability or authorization.
            # Keep metadata and top-level scope in the key, otherwise a copy
            # visible to one tenant/version can erase another tenant's copy.
            scope = {k: r.get(k) for k in (
                "tenant", "device", "manual_id", "product", "platform",
                "version", "product_version", "document_version",
            )}
            key = hashlib.sha256(json.dumps(
                [content, title, operation, meta, scope],
                ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            ).encode("utf-8")).hexdigest()
            rep = seen.get(key)
            if rep is not None:
                rep_of[str(r["id"])] = rep
                continue
            seen[key] = str(r["id"])
            out.append(r)
        return out, rep_of

    @classmethod
    def from_file(cls, path: Path) -> "JsonlCorpus":
        """从 JSONL 构造语料，并由文件名/位置推导向量 namespace 与持久化目录。

        读取侧两级去重（磁盘 JSONL 本身不被修改）：
        1. 同 ID 且除 source_ref/links/images 外逐字段相同的重复行
           （ingest ID 碰撞最小缓解，见 _dedupe_identical_id_rows）；
        2. 不同 ID 但 content 逐字节相同的行收敛为首现代表
           （见 _dedupe_content_rows）。
        """
        path = Path(path)
        records = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        # 缺陷④：先剥离 DOCTYPE 泄漏再去重（去重键含 content|text）。
        records = [cls._strip_doctype_leak(r) for r in records]
        records, _dropped = cls._dedupe_identical_id_rows(records)
        records, rep_of = cls._dedupe_content_rows(records)
        # sections.jsonl -> 'sections'；ocr.jsonl -> 'ocr'。
        namespace = path.name[: -len(".jsonl")] if path.name.endswith(".jsonl") else path.stem
        import os as _os

        from edr_rag.vectorstore import vector_dir_for_workdir

        env_dir = _os.environ.get("EDR_RAG_VECTOR_DIR", "").strip()
        # 1433 修复：评测隔离——EDR_RAG_VECTOR_DIR 设置时向量持久化重定向，
        # 查询触发的自动重建不再写入正式 workdir 的 vector_index。
        vector_dir = Path(env_dir) if env_dir else vector_dir_for_workdir(path.parent)

        corpus = JsonlCorpus(
            records,
            namespace=namespace or None,
            vector_dir=vector_dir,
        )
        corpus._rep_of = rep_of
        return corpus

    @staticmethod
    def _disambiguate_duplicate_ids(records: list[dict]) -> list[dict]:
        """给同 ID 异内容变体的非首现行分配唯一内部 _doc_id（读取侧）。

        首现行 _doc_id = 公开 id（规范代表）；第 2..n 个变体为
        ``<id>#dup2``、``<id>#dup3``…（仅作向量 upsert 键与内部映射，
        公开 id 字段不改——gold 对齐、引用定位、get() 主键查询全部仍走
        原始 ID）。_normalize 在 from_file 内先于本函数运行时已写过
        _doc_id，这里直接覆写；裸构造路径无重复时零改写。
        """
        if not records:
            return records
        seen: dict[str, int] = {}
        for r in records:
            rid = str(r.get("id") or "")
            if not rid:
                continue
            n = seen.get(rid, 0) + 1
            seen[rid] = n
            if n > 1:
                r["_doc_id"] = f"{rid}#dup{n}"
        return records

    def representative_id(self, record_id: str) -> str:
        """返回该记录 ID 在内容级去重后的代表 ID。

        语料中已存在的 ID 原样返回；被去重收敛的 ID 返回其组代表。
        消费方（eval gold 重映射、引用渲染、nodes 评测对齐）必须经此
        口查询，不得各自维护映射（[0440] 结论三）。
        """
        return self._rep_of.get(record_id, record_id)

    @classmethod
    def _strip_doctype_leak(cls, r: dict) -> dict:
        """剥离泄漏的 DOCTYPE 声明（缺陷④，读取侧）。

        递归覆盖：前缀型整段删除；chunks/nodes 跨节拼接的中部型
        （「恢复: PUBLIC …」）按同一签名删除。顶层 content/text 之外，
        metadata.recovery 等嵌套字符串字段同样泄漏（缺陷④b，anquan
        chunks 276 / nodes 304 行），签名 //DTD 已验证 0 误伤，与字段
        位置无关。必须发生在内容级去重之前——去重键为 content|text，
        剥离先行才能让仅泄漏空白差异的孪生行并组。
        """
        return cls._strip_leak_value(r)

    @classmethod
    def _strip_leak_value(cls, v):
        if isinstance(v, dict):
            return {k: cls._strip_leak_value(x) for k, x in v.items()}
        if isinstance(v, list):
            return [cls._strip_leak_value(x) for x in v]
        if isinstance(v, str) and "PUBLIC" in v:
            v2 = _LEAKED_DOCTYPE_RE.sub(" ", v)
            if v2 != v:
                return re.sub(r"^\s+", "", v2)
        return v

    @staticmethod
    def _normalize(r: dict) -> dict:
        """Merge title/path/text into a searchable 'searchable' field."""
        text = r.get("text") or r.get("content") or ""
        title = r.get("title") or ""
        section_path = r.get("section_path") or []
        paths = " ".join(section_path) if section_path else ""
        source = r.get("source") or {}
        file = source.get("file") or r.get("source_file") or ""

        # Chunk records keep module/section context inside `metadata`
        # (built by P1 build_chunks). Fold that in so queries relying on
        # module/module-path context can still match.
        meta = r.get("metadata") or {}
        meta_paths = meta.get("section_path") or []
        if isinstance(meta_paths, str):
            meta_paths = [meta_paths]
        meta_title = meta.get("title") or ""
        module = meta.get("module") or ""
        version = meta.get("version") or ""
        expected_result = meta.get("expected_result") or ""
        folded_meta = " ".join(
            [
                meta_title,
                module,
                version,
                expected_result,
                " ".join(meta_paths),
                meta.get("operation") or "",
            ]
        )

        r["_searchable"] = normalize_query(" ".join([title, paths, file, folded_meta, text]))
        r["_title"] = title or meta_title
        r["_doc_id"] = r.get("id") or ""
        return r

    def _build_bm25(self):
        for i, r in enumerate(self.records):
            toks = tokenize(r["_searchable"])
            self._doc_len.append(len(toks))
            counts = Counter(toks)
            for term, tf in counts.items():
                self._df[term] += 1
                self._postings.setdefault(term, []).append((i, tf))
        total = sum(self._doc_len)
        self._avgdl = total / len(self._doc_len) if self._doc_len else 0.0

    def keyword_search(
        self,
        query: str,
        limit: int = 5,
        allowed: Optional[set[int]] = None,
    ) -> list[dict]:
        """Simple substring / token-overlap scoring.

        Args:
            allowed: 允许参与打分的行号集合 (ACL 谓词下推，P4)。None = 不限制。
                越权文档在**打分前**就被排除，而不是打完分再过滤。
        """
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        scores: list[tuple[float, int]] = []
        for i, r in enumerate(self.records):
            if allowed is not None and i not in allowed:
                continue
            field = r["_searchable"]
            score = 0.0
            for t in q_tokens:
                if t in field:
                    # title+path matches weighted higher
                    weight = 3.0 if (t in r["_title"] or t in r.get("_searchable", "")[:200]) else 1.0
                    score += field.count(t) * weight
            if score > 0:
                scores.append((score, i))
        scores.sort(key=lambda x: -x[0])
        return [self._to_result(i, s) for s, i in scores[:limit]]

    def bm25_search(
        self,
        query: str,
        limit: int = 5,
        k1: float = 1.5,
        b: float = 0.75,
        allowed: Optional[set[int]] = None,
    ) -> list[dict]:
        """BM25 打分。``allowed`` 为 ACL 允许的行号集合 (谓词下推，P4)。"""
        q_tokens = set(tokenize(query))
        if not q_tokens:
            return []
        n = len(self.records)
        idf_cache = {}
        for t in q_tokens:
            df = self._df.get(t, 0)
            idf_cache[t] = 0.0 if df == 0 else math.log((n - df + 0.5) / (df + 0.5) + 1)
        scores: dict[int, float] = {}
        for t in q_tokens:
            idf = idf_cache[t]
            if idf == 0:
                continue
            for doc_idx, tf in self._postings.get(t, []):
                if allowed is not None and doc_idx not in allowed:
                    continue
                dl = self._doc_len[doc_idx]
                tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (dl / self._avgdl))) if self._avgdl else 0.0
                scores[doc_idx] = scores.get(doc_idx, 0.0) + idf * tf_norm
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return [self._to_result(i, s) for i, s in ranked[:limit]]

    def vector_search(
        self,
        query: str,
        limit: int = 5,
        filter: Optional[dict[str, Any]] = None,
        allowed: Optional[set[int]] = None,
        fallback: str = "bm25",
    ) -> list[dict]:
        """Embedding-based retrieval via the pluggable vector store (P3-1).

        Uses LlamaIndex + a BGE embedder when the optional `rag` extras are
        installed, indexing this corpus's records into the configured vector
        backend (default: PersistentVectorStore, on-disk exact cosine). If the
        optional deps are missing, gracefully degrades to BM25 so the main
        retrieval chain never errors. The configured backend is determined by
        EDR_RAG_VECTOR_BACKEND (see edr_rag/vectorstore).

        Args:
            filter: 可选 metadata 谓词，在**打分之前**下推到存储层
                (P4 ACL 谓词下推的基础)。值为 list/set/tuple 时语义是
                「属于集合」，标量时是精确相等。
            fallback: 运行中失败/缺依赖时的行为。
                'bm25'（默认，公开契约）：单策略 vector 调用降级 BM25；
                'empty'（hybrid 内部用）：返回 []，调用方不得把其它路的
                结果伪装成向量路重复计入融合（1433 事故：keyword+BM25+BM25）。
        """
        embed_fn = _optional_embedder()
        if embed_fn is None:
            # Degrade to BM25 when LlamaIndex / embedding model is absent —
            # main chain stays green without the optional deps.
            if fallback == "empty":
                return []
            return self.bm25_search(query, limit=limit, allowed=allowed)

        try:
            index = _get_shared_index(
                embed_fn,
                namespace=self.namespace or "default",
                directory=self.vector_dir if self.namespace else None,
            )
            indexed = index.index_records(
                self.records,
                text_field="_searchable",
                id_field="_doc_id",
            )
            # 只有真正发生写入时才落盘 —— index_records 的签名短路命中时
            # (常见路径) 不做任何 IO。
            if indexed and self.vector_dir is not None:
                index.persist()
            # ACL 受限时多召回一些，抵消随后按行号过滤造成的损耗。
            fetch = limit if allowed is None else max(limit * 4, limit)
            hits = index.search(query, top_k=fetch, filter=filter)
            if allowed is not None:
                hits = [(s_, i) for s_, i in hits if i in allowed]
            return [self._to_result(doc_idx, score) for score, doc_idx in hits[:limit]]
        except Exception:  # pragma: no cover - optional path only
            # 1433 修复：失败必须可观测（升级为 error），且 hybrid 语义
            # （fallback='empty'）下返回空——降级 BM25 的结果与真 BM25 路
            # 完全相同，混入融合等于 keyword+BM25+BM25（已实证改变排序）。
            logger.error("vector_search failed (fallback=%s)", fallback, exc_info=True)
            if fallback == "empty":
                return []
            return self.bm25_search(query, limit=limit, allowed=allowed)

    #: 注入给 reranker 的正文上限。比 snippet 的 120 字长一个数量级，
    #: 让 cross-encoder 有足够上下文判相关性 (P4 严重问题 7)。
    RERANK_TEXT_MAX = 1200

    def _to_result(self, doc_idx: int, score: float) -> dict:
        r = self.records[doc_idx]
        return {
            # 公开 id 用原始记录 id（同 ID 异内容变体共享公开 ID，
            # _doc_id 是内部唯一 upsert 键——#dupN 后缀不外泄）。
            "id": r.get("id") or r["_doc_id"],
            "type": r.get("type", "section"),
            "score": round(score, 3),
            "title": r["_title"],
            "source": r.get("source") or {
                "file": r.get("source_file", ""),
            },
            "snippet": self._snippet(r),
            # 内部字段：仅供 rerank 打分，由 RetrievalService 在返回前剥掉。
            # snippet 是给人看的 120 字摘要，不适合喂 cross-encoder。
            "_rerank_text": self._rerank_text(r),
        }

    def _rerank_text(self, r: dict) -> str:
        """给 reranker 用的较完整正文（截断到 RERANK_TEXT_MAX）。"""
        text = r.get("text") or r.get("content") or r.get("_searchable") or ""
        return re.sub(r"\s+", " ", str(text)).strip()[: self.RERANK_TEXT_MAX]

    def _snippet(self, r: dict, max_len: int = 120) -> str:
        text = r.get("text") or r.get("content") or ""
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) <= max_len:
            return text
        return text[:max_len] + "..."
