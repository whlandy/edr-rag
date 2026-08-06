# EDR-RAG 优化设计文档 V2

> 状态：Draft for implementation  
> 适用代码版本：`edr-rag 0.1.x`  
> 更新时间：2026-08-06  
> 上一版：[EDR-RAG-DESIGN.md](./EDR-RAG-DESIGN.md)

## 1. 文档目的

本文档定义 `edr-rag` 下一阶段的目标架构、数据契约、检索策略、运行边界和验收标准。

V2 不重新设计已经稳定的 CHM 解析流程，而是在当前实现之上重点解决以下问题：

1. 将关键词、BM25 和 LlamaIndex 向量检索统一为一个可配置的检索层。
2. 保持 Action 优先、Procedure 次之、Manual Section 兜底的业务语义。
3. 让索引具备版本、来源校验、可重建和可观测能力。
4. 让 MCP、CLI 和 Python API 使用同一套检索行为。
5. 保证 RAG 只提供知识和计划依据，不越权执行 EDR 操作。

本文档中的状态标记：

- `已实现`：当前仓库已有可运行代码。
- `部分实现`：已有基础能力，但尚未达到生产标准。
- `待实现`：目标设计，需进入后续迭代。

## 2. 系统定位

`edr-rag` 是 EDR Agent 的“产品手册知识与操作规划层”，不是 GUI 自动化执行器。

它负责：

- 将 CHM 手册转换为结构化、可追溯的数据。
- 检索说明、操作流程和 Action 候选。
- 返回风险、前置条件、成功标准和原始出处。
- 接收执行反馈，为人工审核和重新索引提供依据。

它不负责：

- 点击、输入、PowerShell 执行等终端操作。
- 绕过用户确认执行中高风险动作。
- 根据模型猜测手册中不存在的操作步骤。
- 自动发布未经审核的高风险 Action。

责任边界：

```text
edr-rag       知道“应该做什么、依据是什么、风险是什么”
edr-wd        负责“如何在目标 Windows 环境执行”
Agent/Skill   负责编排检索、确认、执行、验证和反馈
```

## 3. 设计原则

### 3.1 结构化数据优先

CHM 是带目录、HTML、链接和图片的知识包，不应被直接当作一个长文本文件。系统必须先建立稳定的 Manual IR，再构建 Procedure、Action 和向量索引。

### 3.2 可追溯优先于生成能力

每个检索结果、Procedure 和 Action 必须能追溯到：

- `manual_id`
- section/procedure/action ID
- CHM 内相对路径
- 源文档摘要或正文片段
- 数据和索引版本

### 3.3 混合检索优先

默认检索策略应为 Hybrid，而不是纯向量检索：

- BM25 擅长产品名、菜单名、错误码、命令和精确术语。
- Embedding 擅长同义表达和自然语言意图。
- RRF 融合两类排名，降低单一模型或单一打分体系造成的偏差。

### 3.4 索引是派生物

JSONL 是事实来源，向量索引可以随时删除和重建。不得只在向量数据库中保存唯一数据副本。

### 3.5 安全能力独立于召回分数

高相关不等于可执行。Action 是否可执行必须额外检查：

- `status`
- `risk`
- `requires_confirmation`
- `preconditions`
- `success_criteria`
- `source_verified`

## 4. 当前实现基线

| 能力 | 状态 | 当前实现 |
|---|---|---|
| CHM 解包 | 已实现 | Windows `hh.exe`；macOS/Linux `extract_chmLib` |
| TOC/HTML 解析 | 已实现 | `.hhc` + BeautifulSoup |
| Manual Section | 已实现 | `sections.jsonl` |
| 资源登记 | 已实现 | `assets.jsonl` |
| Procedure 抽取 | 已实现 | 规则抽取，输出 `procedures.jsonl`、`chunks.jsonl` |
| Action Catalog | 已实现 | `actions.jsonl`、`actions.yaml` |
| 关键词/BM25 | 已实现 | 本地 `JsonlCorpus` |
| LlamaIndex | 已实现 | 本地 `VectorStoreIndex`，按类型持久化 |
| 中文 Embedding | 已实现 | 默认 `BAAI/bge-small-zh-v1.5` |
| Hybrid 检索 | 已实现 | BM25 + Vector + RRF |
| Python API | 部分实现 | P2 已支持部分 vector/hybrid 路径 |
| MCP Server | 部分实现 | 已暴露工具，但搜索仍主要走关键词路径 |
| Feedback | 部分实现 | JSONL 存储和启发式分类 |
| 自动 Re-index | 待实现 | 当前仅返回操作提示 |
| 索引版本校验 | 待实现 | 当前 manifest 信息不足 |
| 并发安全 | 待实现 | JSONL 更新仍为整文件重写 |

## 5. 目标架构

```text
                         ┌──────────────────────────┐
                         │       CHM Manual         │
                         └────────────┬─────────────┘
                                      │
                         ┌────────────▼─────────────┐
                         │ P0 Ingestion             │
                         │ extract / TOC / HTML     │
                         └────────────┬─────────────┘
                                      │
                   sections.jsonl / assets.jsonl / manifest.json
                                      │
                         ┌────────────▼─────────────┐
                         │ P1 Knowledge Modeling    │
                         │ procedures / chunks      │
                         └────────────┬─────────────┘
                                      │
                         ┌────────────▼─────────────┐
                         │ P2 Action Modeling       │
                         │ risk / confirmation      │
                         └────────────┬─────────────┘
                                      │
              ┌───────────────────────┴───────────────────────┐
              │                                               │
    ┌─────────▼──────────┐                         ┌──────────▼─────────┐
    │ Lexical Index      │                         │ LlamaIndex Vector │
    │ keyword / BM25     │                         │ local embeddings  │
    └─────────┬──────────┘                         └──────────┬─────────┘
              └───────────────────────┬───────────────────────┘
                                      │
                         ┌────────────▼─────────────┐
                         │ Retrieval Service       │
                         │ filter / RRF / hydrate  │
                         └───────┬──────────┬───────┘
                                 │          │
                         ┌───────▼───┐  ┌──▼────────┐
                         │ CLI / API │  │ MCP Server│
                         └───────┬───┘  └──┬────────┘
                                 └────┬─────┘
                                      │
                         ┌────────────▼─────────────┐
                         │ Agent Planner           │
                         │ action-first + safety   │
                         └────────────┬─────────────┘
                                      │ approved plan only
                         ┌────────────▼─────────────┐
                         │ edr-wd Executor         │
                         └────────────┬─────────────┘
                                      │
                         ┌────────────▼─────────────┐
                         │ Feedback / Review Loop  │
                         └──────────────────────────┘
```

## 6. 工作目录与产物

建议统一采用扁平的业务产物目录，并将派生索引放入隐藏目录：

```text
WORKDIR/
├── input/
│   └── product-manual.chm
├── extracted/
├── manifest.json
├── sections.jsonl
├── assets.jsonl
├── procedures.jsonl
├── chunks.jsonl
├── p1_metadata.json
├── actions.jsonl
├── actions.yaml
├── feedback.jsonl
├── .llamaindex/
│   ├── manifest.json
│   ├── section/
│   ├── procedure/
│   ├── step/
│   └── action/
└── logs/
```

约束：

- `extracted/` 每次导入必须由新目录原子替换，不能残留旧 CHM 文件。
- `.llamaindex/` 不进入 Git。
- 原始 CHM、解压文件和模型缓存不进入 Git。
- JSONL 写入必须使用 UTF-8，一行一个完整 JSON 对象。

## 7. 核心数据契约

### 7.1 Section ID

格式：

```text
manual.<manual_slug>.section.<semantic_path>
```

示例：

```text
manual.product_manual.section.network.proxy
```

规则：

1. `manual_slug` 来源于 `manual_id`。
2. `semantic_path` 优先来源于 TOC，再回退到标题。
3. 无 TOC、无标题时使用正文内容摘要构建 `orphan_<digest>`。
4. ID 不得依赖 CHM 内部文件路径。
5. 同一批次出现重复 ID 时必须中止，不能静默覆盖。

### 7.2 检索统一返回结构

所有搜索入口最终统一为：

```json
{
  "id": "manual.product_manual.section.network.proxy",
  "type": "section",
  "title": "配置代理服务器",
  "snippet": "打开网络设置……",
  "source": {
    "file": "network/proxy.html",
    "manual_id": "product-manual"
  },
  "score": 0.032787,
  "scoring": "hybrid_rrf",
  "rank_signals": {
    "bm25_rank": 1,
    "vector_rank": 2
  },
  "index_version": "sha256:..."
}
```

说明：

- Hybrid 的 `score` 是融合分数，不能与 BM25 或余弦相似度直接比较。
- Action 搜索结果必须额外返回 `risk`、`status`、`confidence` 和 `requires_confirmation`。
- API 和 MCP 不应返回 LlamaIndex 内部 Node 对象。

### 7.3 索引 Manifest

目标 `.llamaindex/manifest.json`：

```json
{
  "schema_version": 1,
  "created_at": "2026-08-06T12:00:00Z",
  "embed_provider": "huggingface",
  "embed_model": "BAAI/bge-small-zh-v1.5",
  "embedding_dimension": 512,
  "llama_index_version": "0.14.23",
  "source_artifacts": {
    "section": {"file": "sections.jsonl", "sha256": "...", "count": 128},
    "procedure": {"file": "procedures.jsonl", "sha256": "...", "count": 32},
    "step": {"file": "chunks.jsonl", "sha256": "...", "count": 118},
    "action": {"file": "actions.jsonl", "sha256": "...", "count": 27}
  }
}
```

查询前必须校验：

- 索引存在。
- 查询使用的 embedding 模型与建索引模型一致。
- 源 JSONL 哈希未变化。
- schema 版本受当前程序支持。

校验失败时返回明确的 `index_stale` 或 `model_mismatch`，不得继续返回可能过期的结果。

## 8. LlamaIndex 设计

### 8.1 依赖策略

LlamaIndex 保持为可选依赖：

```bash
python -m pip install -e '.[rag]'
```

基础安装必须仍可运行：

- CHM 导入
- JSONL 生成
- Keyword/BM25 搜索
- Action 查询

### 8.2 索引粒度

每种业务对象使用独立索引：

| 类型 | 输入文件 | 主要用途 |
|---|---|---|
| `section` | `sections.jsonl` | 原理说明、故障排查、背景知识 |
| `procedure` | `procedures.jsonl` | 完整操作流程 |
| `step` | `chunks.jsonl` | 精确步骤定位 |
| `action` | `actions.jsonl` | Agent 计划和执行候选 |

独立索引优于单一大索引，因为它可以：

- 保持 Action-first 的业务优先级。
- 避免依赖向量 metadata filter 的具体实现差异。
- 允许某一类产物单独重建。
- 为不同类型配置不同 top-k、阈值和评估集。

### 8.3 Document 映射

向量正文应包含有助于语义匹配的字段：

```text
Section:
  title + section_path + content

Procedure:
  title + operation + preconditions + steps + warnings + recovery

Step:
  title/context + text + source procedure

Action:
  title + module + preconditions + source_instruction + target
```

metadata 仅保存过滤、显示和回查所需的小字段。完整业务对象仍从 JSONL 按 ID 获取。

### 8.4 Embedding 模型

当前默认：

```text
BAAI/bge-small-zh-v1.5
```

选择原因：

- 中文语义检索效果优于纯英文小模型。
- 可本地运行，不要求外部 API Key。
- 体量适合开发机和离线 EDR 环境。

后续可配置：

- `BAAI/bge-m3`：多语言和更强检索质量，资源消耗更高。
- OpenAI Embedding：适合联网部署，但必须单独处理密钥、费用和数据出境策略。
- 企业内部 Embedding 服务：通过适配器接入，不改变检索上层接口。

建索引和查询必须使用同一模型。

## 9. 检索与排序

### 9.1 策略

支持四种策略：

```text
keyword  子串/Token 匹配，调试和精确查找
bm25     默认词法召回
vector   纯语义召回
hybrid   BM25 + Vector + RRF，生产默认
```

### 9.2 Hybrid 流程

```text
query
  ├── normalize/tokenize ──> BM25 top 2K
  └── embedding ───────────> Vector top 2K
                                  │
                                  v
                    Reciprocal Rank Fusion
                                  │
                                  v
                   metadata filter + hydrate
                                  │
                                  v
                              final top K
```

RRF：

```text
score(d) = Σ 1 / (k + rank_i(d))
```

默认：

- `k = 60`
- 每路召回 `2 * final_limit`
- 相同 RRF 分数按稳定 ID 排序，保证测试确定性

### 9.3 Action-first Planner Retrieval

对操作型请求：

1. 使用 Hybrid 搜索 Action。
2. 对候选执行 risk/status/product/module 过滤。
3. 取得完整 Action 并验证来源与前置条件。
4. Action 无合格结果时搜索 Procedure。
5. Procedure 无结果时搜索 Section。
6. 解释型问题可以直接搜索 Section，但不得据此绕开 Action 风险策略。

不得仅以“搜索结果非空”判断检索成功。目标实现应增加：

- 最低相关性门槛。
- top-1 与 top-2 差距判断。
- `low_confidence_retrieval` 反馈。
- 无已审核 Action 时的降级提示。

## 10. API、CLI 与 MCP

### 10.1 统一 RetrievalService

下一步应抽取统一服务，避免 CLI、P2 API 和 MCP 分别实现搜索：

```python
class RetrievalService:
    def search(
        self,
        query: str,
        artifact_type: str,
        strategy: str = "hybrid",
        limit: int = 5,
        filters: dict | None = None,
    ) -> list[SearchResult]: ...
```

CLI、Python API 和 MCP 只负责参数校验与结果序列化。

### 10.2 CLI

建索引：

```bash
edr-rag index --workdir WORKDIR --type all --show-progress
```

搜索：

```bash
edr-rag search \
  --workdir WORKDIR \
  --type section \
  --strategy hybrid \
  --query "如何设置网络代理" \
  --limit 5
```

目标新增：

```bash
edr-rag index-status --workdir WORKDIR
edr-rag reindex --workdir WORKDIR --changed-only
edr-rag evaluate --dataset retrieval-eval.jsonl
```

### 10.3 MCP Tools V2

建议统一为：

```text
search_knowledge(query, type, strategy="hybrid", filters={}, limit=5)
search_actions(query, strategy="hybrid", risk=None, status=None, limit=5)
get_action(action_id)
get_procedure(procedure_id)
capture_feedback(...)
get_index_status()
reindex(manual_id, changed_only=true)
```

MCP 响应必须包含：

- `ok`
- `results` 或 `error`
- `strategy`
- `index_version`
- `degraded`：是否降级到 BM25

如果向量索引不可用，推荐默认降级为 BM25，并明确返回：

```json
{
  "ok": true,
  "strategy": "bm25",
  "degraded": true,
  "degraded_reason": "vector_index_missing"
}
```

## 11. 索引生命周期

### 11.1 构建

```text
validate JSONL
  -> calculate source hashes
  -> build in staging directory
  -> run smoke queries
  -> write manifest
  -> atomically replace active index
```

索引构建失败时必须保留上一个可用索引。

### 11.2 失效条件

以下任一变化都需要重建对应索引：

- JSONL SHA-256 变化。
- embedding 模型或维度变化。
- Document 文本映射规则变化。
- LlamaIndex 存储 schema 不兼容。
- tokenizer/normalizer 的语义发生变化。

### 11.3 增量策略

第一阶段继续全量重建，简单且可靠。数据规模达到以下任一条件后再实现增量：

- 单类文档超过 50,000。
- 全量构建超过 10 分钟。
- 模型下载之外的索引文件超过 2 GB。

## 12. Feedback 与持续优化

Feedback 生命周期：

```text
captured -> needs_review -> patch_proposed -> patched -> reindexed -> closed
```

优化要求：

1. `feedback_id` 使用 UUID/ULID，不能使用秒级时间戳。
2. 写入和状态更新必须具备进程级锁或迁移到 SQLite。
3. patch 只能提出建议，不能直接覆盖 Action Catalog。
4. 只有审核通过的 patch 才能触发 JSONL 更新和重建索引。
5. 重建完成后记录新旧索引版本以及受影响对象 ID。

推荐短期将 FeedbackStore 迁移为 SQLite：

```text
feedback
  id PRIMARY KEY
  manual_id
  action_id
  procedure_id
  failure_type
  payload_json
  status
  created_at
  updated_at
```

JSONL 可以继续作为导入导出格式，而不再承担并发事务存储。

## 13. 安全设计

### 13.1 内容安全

- 拒绝逃逸 `extracted/` 的本地路径。
- 外部 URL 不作为本地资源读取。
- HTML 中的脚本只作为资源登记，不执行。
- CHM 解包进程应配置超时、最大文件数和最大总大小。
- 对异常压缩比、符号链接和路径穿越进行审计。

### 13.2 Agent 安全

- 检索结果不得直接触发执行。
- `medium/high` Action 必须确认。
- `candidate/needs_review` Action 默认不可自动执行。
- 缺少 source、risk 或 success criteria 的 Action 不可发布。
- RAG 输出中的文档文本属于不可信内容，不能覆盖系统级安全策略。

### 13.3 数据与模型

- 本地 embedding 是默认部署模式。
- 使用云 embedding 前必须明确数据出境和日志策略。
- 日志不得记录密码、Token、用户输入的敏感字段或完整屏幕内容。

## 14. 可观测性

每次查询建议记录结构化事件：

```json
{
  "request_id": "...",
  "query_hash": "sha256:...",
  "artifact_type": "action",
  "requested_strategy": "hybrid",
  "actual_strategy": "hybrid",
  "latency_ms": 83,
  "bm25_candidates": 10,
  "vector_candidates": 10,
  "returned": 5,
  "top_ids": ["network.configure_proxy"],
  "index_version": "sha256:...",
  "degraded": false
}
```

默认不记录原始 query；调试环境可显式开启。

建议指标：

- 查询延迟 P50/P95/P99。
- Hybrid 降级率。
- 无结果率和低置信率。
- Action 命中率。
- 用户纠正率。
- 索引构建耗时和失败率。
- 按失败类型统计的执行反馈。

## 15. 测试与评估

### 15.1 单元测试

- JSONL 到 Document 的字段映射。
- 中文与英文大小写检索。
- RRF 排序和稳定 tie-break。
- metadata 过滤。
- ID 唯一性。
- 索引 manifest 校验。
- 模型不匹配和索引缺失错误。

### 15.2 集成测试

- 小型中文 fixture 的真实建索引、持久化、重载和查询。
- CHM → Section → Procedure → Action → Index 全链路。
- 重建失败时旧索引仍可查询。
- MCP vector/hybrid 调用与 CLI/API 结果一致。

### 15.3 Retrieval 评估集

建立人工标注的 `retrieval-eval.jsonl`：

```json
{
  "query": "怎么给终端设置上网代理",
  "type": "action",
  "relevant_ids": ["network.configure_proxy"],
  "must_rank_top_k": 3,
  "tags": ["semantic", "synonym", "zh-CN"]
}
```

至少覆盖：

- 菜单名和功能名精确查询。
- 同义表达。
- 中英文混合。
- 错误码。
- 相似但风险不同的 Action。
- 应无结果的问题。

上线门槛建议：

| 指标 | 目标 |
|---|---:|
| Action Recall@5 | ≥ 0.95 |
| Procedure Recall@5 | ≥ 0.90 |
| MRR@10 | ≥ 0.80 |
| 错误高风险 Action Top-1 | 0 |
| Hybrid P95（索引已加载） | < 500 ms |

## 16. 性能策略

当前 `LlamaIndexStore` 每次 CLI 查询会重新加载 embedding 模型，CLI 场景可接受，但 MCP 长驻服务必须：

- 进程启动时加载一次 embedding 模型。
- 缓存已加载的各类型 index。
- 索引切换时使用版本检测和原子刷新。
- 限制并发 embedding 查询数量。
- 对相同 query + filter + index_version 使用短期缓存。

建议目标：

```text
冷启动：允许数秒
热查询：P95 < 500 ms
索引刷新：不中断现有查询
```

## 17. 异常与降级

| 场景 | 行为 |
|---|---|
| 未安装 LlamaIndex | BM25 可用；vector 明确报依赖缺失 |
| 模型未下载且无网络 | 降级 BM25；返回 degraded 原因 |
| 向量索引不存在 | 提示执行 index；MCP 默认降级 BM25 |
| 索引与 JSONL 哈希不符 | 标记 stale；不使用旧向量结果 |
| 某类 JSONL 不存在 | `--type all` 跳过；显式指定时失败 |
| JSONL 某行损坏 | 构建失败并报告文件与行号 |
| 索引重建失败 | 保留旧索引，记录失败事件 |
| embedding 查询超时 | 返回 BM25 结果并标记 degraded |

## 18. 分阶段实施计划

### Phase A：统一检索层

目标：消除 CLI、API、MCP 的搜索实现分叉。

- 新建 `RetrievalService` 和统一 `SearchResult`。
- 把 Keyword、BM25、Vector、Hybrid 接到同一入口。
- MCP 增加 `strategy` 参数并默认 Hybrid。
- 缺失向量索引时自动降级 BM25。
- 补齐 Action 结果 hydration 和 filter。

验收：同一 query/type/strategy 在 CLI、API、MCP 返回相同 ID 顺序。

### Phase B：可靠索引生命周期

- manifest 增加 hash、版本、维度和构建时间。
- 构建使用 staging + atomic replace。
- 查询前校验 freshness 和 model compatibility。
- 增加 `index-status` 与 `reindex`。

验收：修改 JSONL 后系统能识别 stale index；失败重建不影响旧索引。

### Phase C：检索质量评估

- 建立至少 100 条真实查询评估集。
- 对比 BM25、Vector、Hybrid。
- 调整候选数、RRF 参数和 Action 过滤规则。
- 增加 low-confidence 判定。

验收：达到第 15 节指标，且 Hybrid 显著优于任一单路检索。

### Phase D：反馈与运维

- FeedbackStore 迁移 SQLite。
- 实现审核后 patch 应用。
- 完成自动重建及索引热切换。
- 增加结构化日志和基础指标。

验收：反馈从 capture 到 reindexed 可闭环追踪，且无未审核自动发布。

## 19. 明确不做的事项

当前阶段不引入：

- 复杂的多 Agent RAG 编排框架。
- 仅为了使用框架而替换现有确定性解析器。
- 大规模外部向量数据库。
- RAG 直接执行 GUI 或 PowerShell。
- 无来源约束的 LLM 自动生成 Action。
- 未建立评估集前的复杂 reranker。

当本地 SimpleVectorStore 达到容量或并发瓶颈后，再评估 Qdrant、Milvus、pgvector 或企业已有向量服务。

## 20. 完成定义

V2 达到可发布状态需同时满足：

- P0/P1/P2 产物通过 schema 和唯一性校验。
- Hybrid 成为 API 和 MCP 的默认策略。
- 所有结果包含稳定 ID 和 source citation。
- 索引能够检测 stale/model mismatch。
- 缺少向量能力时可控降级 BM25。
- Action-first、安全过滤和确认策略不可被检索分数绕过。
- 真实中文评估集达到质量门槛。
- 索引构建、查询和反馈路径具备结构化日志。
- 完整测试在离线基础安装和 `[rag]` 安装两种环境均通过。

---

本设计的核心结论是：保留现有结构化 CHM 流水线，以 LlamaIndex 增强语义召回，用 BM25 保证精确术语召回，通过统一 RetrievalService、索引版本校验和 Action 安全策略将其收敛为可维护、可评估、可安全运行的 EDR 知识系统。
