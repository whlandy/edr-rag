# P4 — 商用化硬化 — Design (R0)

| Field | Value |
|-------|-------|
| Phase | P4 (P0–P3 之后的硬化阶段) |
| Status | R0 (design gate pending review) |
| Reviewer | Hermes (under edr-test direction) |
| Target branch | `origin/main` |
| Working branch | `feature/p4-hardening` |
| 前置评审 | 2026-08-10 edr-rag 成熟度评估（阻塞级 4 项 + 严重级 8 项） |

---

## 1. Problem Statement

P0–P3 交付了一套结构清晰的手册知识管线，但成熟度评估发现三类问题使它无法商用：
**（a）有些能力是空的**——五个外部向量后端全是 `NotImplementedError` 骨架，生成侧没有任何
LLM 接入；**（b）有些能力是假的**——查询扩展的同义词表逐条来自 30 行评测集，再用同一评测集
调 RRF 权重，构成自证闭环；**（c）有些能力有但不可用于生产**——embedding 模型每次查询重新
装载（实测 ~9–10 秒），向量检索是 O(N) 暴力循环且不持久化，ACL 走「全语料召回后过滤」。

本阶段的目标不是加功能，而是把**阻塞级 4 项 + 严重级 8 项**全部落地，使系统达到「可在客户
内网离线环境部署、指标可信、有运维面」的状态。

---

## 2. Scope

### 2.1 In scope

* **A** 检索内核：embedder 缓存修复、向量存储持久化 + ANN、跨语料索引隔离
* **B** 中文分词：unigram → unigram+bigram，可选词典分词
* **C** 评测可信度：去除评测集反向硬编码，调参加 holdout 切分
* **D** 生成侧：LLM provider 抽象 + OpenAI 兼容端点客户端（自托管）
* **E** 答案质量：置信度与检索得分挂钩、rerank 修复、冲突检测扩展
* **F** 安全与权限：ACL 谓词下推、注入防护全路径覆盖
* **G** 服务化与运维：HTTP API + 鉴权 + 限流、并发安全、CI、可观测性轮转
* **H** 反馈闭环：失败分类规则化、patch 真正生成与应用

### 2.2 Out of scope（明确延后，并注明解锁条件）

| 延后项 | 原因 | 解锁条件 |
|---|---|---|
| 评测集扩充到 300+ 条真实查询 | 需要真实 CHM 语料 + 领域专家标注，无法在代码层凭空生成 | 客户提供手册 + 标注人力 |
| 多进程 / 横向扩展的向量服务 | 当前语料规模（单产品手册）用不上；见 §4 Lock 2 | 语料超过 ~50 万 chunk |
| 图像 embedding（CLIP 类）真多模态检索 | 当前图文复合通过 OCR→文本统一到一个向量空间已足够；见 §4 Lock 3 | OCR 召回率实测不达标时 |
| 手册格式扩展（PDF / Word / Confluence） | 与本阶段的「修复既有能力」目标正交 | 产品明确要接第二种手册源 |

---

## 3. Architecture

### 3.1 部署形态（本阶段锁定：内网离线）

```
客户内网（无外网）
├── edr-rag 服务进程
│   ├── HTTP API (stdlib http.server, 无 fastapi/uvicorn 依赖)
│   ├── RetrievalService  ── PersistentVectorStore (本地文件 + 可选 hnswlib)
│   └── LLMClient ──────────► 自托管 OpenAI 兼容端点 (vLLM / Ollama / 内网推理服务)
└── workdir/
    ├── *.jsonl              (P0–P3 产物)
    ├── vector_index/        (新增：持久化向量，进程重启不重嵌)
    └── trace.jsonl.*        (新增：轮转)
```

**交付物是一个 pip 包 + 模型文件 + workdir**，不引入任何需要客户运维的中间件。

### 3.2 数据流（检索链路，修复后）

```
query
  │
  ├─► normalize + expand (外置词表，不再来自评测集)
  │
  ├─► 多路召回 ──┬─ keyword (bigram BM25)
  │              ├─ bm25    (bigram)
  │              └─ vector  (PersistentVectorStore, ACL 谓词下推)
  │
  ├─► RRF 融合 (权重来自 holdout 调参)
  │
  ├─► cross-encoder rerank (全文，非 snippet)
  │
  ├─► ACL 二次校验 (纵深防御)
  │
  └─► synthesize ──┬─ LLM 路径 (OpenAI 兼容端点 + 注入隔离)
                   └─ 组装路径 (降级，同样过注入清洗)
```

### 3.3 Boundary rules

**检索层 → 向量存储**：跨边界的是 `(namespace, VectorRecord[])`。边界上**必须**带
namespace（语料名）——这是修复跨语料冲刷的核心契约。**禁止**任何调用方共用同一
namespace 写入不同语料。

**检索结果 → LLM**：跨边界的是清洗后的文本。边界上**必须**执行 `sanitize_document_text`
+ 数据围栏包裹，**无论是否走 LLM 路径**（旧实现只在 LLM 路径清洗，组装路径裸奔）。
**禁止**把文档内容拼进 system 角色。

**HTTP 层 → 服务层**：跨边界的是已鉴权、已限流的请求。边界上**必须**完成 API key 校验与
配额扣减，**禁止**把调用方传入的 `workdir` 直接落到文件系统操作（路径穿越）。

---

## 4. Design Decisions

### Lock 1 — 向量后端选型：本地持久化 + 可选 ANN，删除五个骨架

**决策**：实现 `PersistentVectorStore`（本地文件持久化 + numpy 向量化**精确**检索 +
metadata 谓词下推），删除 `backends.py` 中 ES / OpenSearch / Qdrant / Milvus / pgvector
五个空壳。`VectorStore` 抽象接口保留。

**不引入 hnswlib**：初稿曾写「可选 hnswlib ANN」，实现时否掉了。理由是——在目标语料规模
下 ANN 没有收益（见下表），而一个只在某些环境装得上、且删除操作会让索引退化的可选 ANN 分支，
本质上就是本文档正在删除的那五个空壳的翻版。精确检索还额外保证了**没有召回损失**。
若语料规模真的涨到需要 ANN，那时补一个实现，而不是现在提前留一个半成品。

**理由**（针对「内网离线 + 产品不复杂 + 图文复合」）：

| 因素 | 推导 |
|---|---|
| 内网离线 | 每多一个外部服务，就是客户现场要装、要备份、要升级、要排障的一个组件。EDR 产品的交付窗口通常很紧，一个 Qdrant 容器可能就卡在客户的安全审批上 |
| 产品不复杂 | 单产品手册量级约 1k–5 万 chunk。numpy 向量化余弦在 5 万 × 512 维上是**毫秒级**，HNSW 的收益在这个规模上基本为零 |
| 图文复合 | 关键在于**统一向量空间 + 模态标签**，不在存储层。OCR 文本与正文 chunk 进同一个 store、用 `modality` metadata 区分，本地存储完全够用 |
| ACL 过滤 | 小规模下「先按 metadata 预过滤再打分」比向量库的 filtered-HNSW **召回率更高**——后者在高选择性过滤时会漏召回，这是已知的工程陷阱 |

**为什么不是继续摊五个骨架**：五个 `NotImplementedError` 提供的是「支持五种后端」的**错觉**，
任何一个被配置上去都会在第一次查询时崩掉然后静默回退。一个真的比五个假的有价值。

**升级路径**：`VectorStore` ABC 保留不动。语料规模真的涨到需要外部向量库时，新增一个实现即可，
检索层无需改动。这是抽象层存在的意义，不是提前写五个空壳。

### Lock 2 — embedder 单例：模块级，而非函数内闭包

**根因**：`_optional_embedder()` 内部定义 `@lru_cache`，每次调用生成全新闭包 + 全新缓存
→ BGE 模型从磁盘重载。实测 9952ms / 8770ms / 10368ms。

**决策**：模型实例提到模块级单例，探测结果（可用/不可用）也缓存。契约：
`_optional_embedder()` 在同一进程内**必须**返回同一个函数对象。加回归测试锁死这一点。

### Lock 3 — 图文复合的处理层次

**决策**：图像不做独立的 image embedding，而是 OCR → 文本 → 与正文进**同一个向量空间**，
用 `modality: "image" | "text"` metadata 区分，检索时可按模态过滤或混排。

**理由**：产品的图是操作截图（按钮、菜单、对话框），其检索价值几乎全部在**图中的文字**上，
而不在视觉语义上。CLIP 类图像 embedding 对「界面截图」的区分度反而低于 OCR 文本。
若后续实测 OCR 召回不达标，再引入图像 embedding（见 §2.2 解锁条件）。

**同时修复**：OCR 测试当前在缺 tesseract 时硬 `assert` 失败（6 个）。改为 `skipif`，
并补一个不依赖任何 OCR 引擎的纯逻辑测试，保证该路径在 CI 上有覆盖。

### Lock 4 — 中文分词：bigram 保底，jieba 可选

**决策**：`tokenize()` 对 CJK 同时产出 unigram 与 bigram；jieba 可用时额外叠加词典切分结果。

**理由**：纯 unigram 使「安全」与「全安」同分，是中文 BM25 的经典缺陷。bigram 是**纯 stdlib**
就能拿到大部分收益的方案，且不引入离线部署的额外依赖。jieba 作为可选增强。

**破坏性**：这会改变 BM25 索引与所有分数。契约：`tokenize()` 的输出顺序为
`[latin..., cjk_unigram..., cjk_bigram...]`，既有断言 token 集合的测试需同步更新。

### Lock 5 — 置信度语义变更

**根因**：`_compute_confidence()` 取命中条目 `confidence` 静态字段的算术平均，与检索质量完全
无关；而拒答阈值 0.4 建在这个数上。

**决策**：置信度改为三因子加权 —— 抽取置信度（静态）、检索得分（归一化后的 top-1）、
证据量（命中条数的饱和函数）。旧字段保留为 `extraction_confidence` 以便审计。

**破坏性**：拒答行为会变。契约：阈值随之重标定，并在评测集上验证误拒答率不上升。

---

## 5. 阶段拆分（每阶段 = 一个原子 commit + 一份报告）

| Phase | 概念 | 对应问题 | 主要文件 |
|---|---|---|---|
| A | 检索内核 | 阻塞 1、2 + embedder bug | `p1/search.py`, `vectorstore/*` |
| B | 中文分词 | 阻塞 4 | `p1/search.py` |
| C | 评测可信度 | 阻塞 3 | `query.py`, `eval/tune.py` |
| D | 生成侧 | 头号结论「没有 G」 | `llm/*`（新）, `p2/synthesize.py` |
| E | 答案质量 | 严重 5、7、9 | `p2/synthesize.py`, `rerank.py` |
| F | 安全与权限 | 严重 6、8 | `retrieval.py`, `security.py` |
| G | 服务化与运维 | 严重 10、12 | `service/*`（新）, `p3/tracing.py`, CI |
| H | 反馈闭环 | 严重 11 | `p3/feedback.py`, `p3/server.py` |

**阶段间约束**：不允许跨阶段修补（Phase E 不得修 Phase A 遗漏的问题）。每阶段结束时
全量测试必须通过，且新增测试数 > 0。

---

## 6. Acceptance Criteria

| # | 可观测的验收标准 |
|---|---|
| AC-1 | 同一进程内连续两次 embedding 调用，第二次耗时 < 100ms（当前 ~10000ms） |
| AC-2 | 进程重启后首次向量检索不触发重新嵌入（读持久化文件） |
| AC-3 | `planner_retrieve` 走 vector 策略时，三层语料各自命中独立 namespace，不互相清空 |
| AC-4 | `backends.py` 中不存在任何 `NotImplementedError` |
| AC-5 | `tokenize("安全")` 与 `tokenize("全安")` 产出不同 token 集合 |
| AC-6 | `query.py` 中不存在来自 `eval_set_default.jsonl` 的硬编码同义词 |
| AC-7 | `tune_rrf_weights` 在 holdout 集上报告分数，且 train/test 不重叠 |
| AC-8 | 存在可运行的 OpenAI 兼容端点客户端，端点不可达时降级到组装路径且不抛异常 |
| AC-9 | ACL 受限查询不再对全语料打分（受限与非受限查询的打分条数量级相同） |
| AC-10 | 组装路径（无 LLM）同样执行注入清洗 |
| AC-11 | HTTP API 无有效 API key 时返回 401；超配额返回 429 |
| AC-12 | 全量测试通过；OCR 测试在缺引擎时 skip 而非 fail |
| AC-13 | 存在 CI workflow，在干净环境（无可选依赖）下可通过 |

---

## 7. 风险与残留

| 风险 | 缓解 |
|---|---|
| 分词变更导致既有检索测试大面积失败 | Phase B 单独成 commit，逐个核对失败是「预期的分数变化」还是「真回归」 |
| 置信度语义变更影响拒答行为 | 在评测集上对比变更前后的拒答率 / 误拒答率，写进 Phase E 报告 |
| stdlib HTTP server 的性能上限 | 内网低 QPS 场景足够；接口层与业务层分离，需要时换 uvicorn 只改一个文件 |
| 评测集仍只有 30 条 | 本阶段无法解决（§2.2）。但通过 holdout 切分，至少让「调参过拟合」变得可见 |

---

## 8. Review Stop Point

**Status: R0 — 待 edr-test 评审。**

按 review protocol，本文档需要 `APPROVED` 或 `APPROVED WITH MINOR LOCKS` 后方可进入实现阶段。
用户已明确指示「写计划然后把阻塞和严重问题都改好」，因此实现将紧随本文档推进；
每个 Phase 仍按 A/B/C 拍板选项单独提交报告，评审边界不丢失。
