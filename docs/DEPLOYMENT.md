# edr-rag 部署指南（内网离线）

面向的部署形态：**客户内网、无外网**。交付物是一个 pip 包 + 模型文件 + workdir，
不要求现场运维任何中间件（无 Qdrant / ES / Postgres / Redis）。

---

## 1. 安装

```bash
# 最小安装：keyword / BM25 检索 + 组装式答案。纯 stdlib 依赖，永远可用。
pip install edr-rag

# 完整能力：向量检索 + cross-encoder 精排 + LlamaIndex 摄取 + OCR
pip install "edr-rag[rag,ocr]"
```

离线环境请先在有网机器上 `pip download` 到本地目录，再 `pip install --no-index
--find-links=./wheels`。

**模型文件需要预先下载**。默认嵌入模型是 `BAAI/bge-small-zh-v1.5`；离线机器上把它
放到本地目录后指向它：

```bash
export EDR_RAG_EMBED_MODEL=/opt/models/bge-small-zh-v1.5
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

没有模型文件也能跑 —— 向量检索会自动降级到 BM25，主链路不受影响。

---

## 2. 构建产物

```bash
edr-rag ingest --chm ./input/product-manual.chm \
  --manual-id product-manual --product ProductName --workdir ~/edr-rag-work

edr-rag-p1 extract-procedures --workdir ~/edr-rag-work
edr-rag-p2 build-actions      --workdir ~/edr-rag-work
edr-rag validate              --workdir ~/edr-rag-work
```

workdir 结构：

```text
~/edr-rag-work/
├── sections.jsonl / procedures.jsonl / actions.jsonl / chunks.jsonl / nodes.jsonl
├── ocr.jsonl            图片 OCR 文本（装了 ocr extras 时）
├── vector_index/        持久化向量，进程重启不重嵌（P4）
├── index_versions.json  索引版本与快照，支持灰度切换/回滚
└── trace.jsonl[.1..N]   结构化可观测事件，自动轮转（P4）
```

---

## 3. 启动 HTTP 服务

**先配 API key，否则服务拒绝启动。** 一个没有鉴权的内网手册服务等于把整本手册
对全网开放；静默放行是比起不来更糟的默认。

最简（单 key，无 ACL、无限流）：

```bash
export EDR_RAG_API_KEY="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
edr-rag-serve --workdir ~/edr-rag-work --host 0.0.0.0 --port 8080
```

多调用方（按 key 绑定租户 ACL 与限流）——`keys.json`：

```json
{
  "keys": [
    {"key": "…", "id": "soc-console", "tenants": ["tenant-a"], "rate_limit": 20,
     "description": "SOC 值班台"},
    {"key": "…", "id": "ticket-bot", "tenants": ["tenant-a", "tenant-b"],
     "rate_limit": 5, "description": "工单机器人"}
  ]
}
```

```bash
export EDR_RAG_API_KEYS_FILE=/etc/edr-rag/keys.json
edr-rag-serve --workdir ~/edr-rag-work --host 0.0.0.0 --port 8080
```

**key 决定能检索到什么** —— 文档级 ACL 绑定在调用方身份上，而不是由调用方自己
声明自己是谁。

### 端点

| 方法 | 路径 | 鉴权 | 说明 |
|---|---|---|---|
| GET | `/healthz` | 否 | 存活探针 |
| GET | `/readyz` | 否 | 就绪探针；无语料时返回 503 |
| POST | `/v1/search` | 是 | 检索 |
| POST | `/v1/ask` | 是 | 结构化答案（AskAnswer） |
| GET | `/v1/actions/{id}` | 是 | 按 action_id 精确获取 |
| POST | `/v1/feedback` | 是 | 执行反馈写入 |

```bash
curl -s -H "Authorization: Bearer $EDR_RAG_API_KEY" \
     -H 'Content-Type: application/json' \
     -d '{"query":"如何配置代理服务器","strategy":"hybrid"}' \
     http://127.0.0.1:8080/v1/ask | python3 -m json.tool
```

---

## 4. 接入自托管 LLM（可选）

不配置就走确定性组装路径，行为与 P2 完全一致。配置后答案由模型生成，
端点不可达时**自动退回组装**，检索始终可用。

```bash
export EDR_RAG_LLM_BASE_URL=http://10.0.0.8:8000/v1   # vLLM / Ollama / TGI / Xinference
export EDR_RAG_LLM_MODEL=Qwen2.5-14B-Instruct
export EDR_RAG_LLM_API_KEY=                            # 自托管常常不校验，可留空
export EDR_RAG_LLM_TIMEOUT=30
export EDR_RAG_LLM_RETRIES=1
export EDR_RAG_LLM_MAX_TOKENS=1024
```

安全边界：检索到的手册内容**只进 user 消息**并包在不可信数据围栏里，system
消息永远是常量指令。被投毒的手册可能污染答案内容，但拿不到指令权。

---

## 5. 环境变量一览

### 检索

| 变量 | 默认 | 说明 |
|---|---|---|
| `EDR_RAG_EMBED_MODEL` | `BAAI/bge-small-zh-v1.5` | 嵌入模型；离线指向本地目录 |
| `EDR_RAG_EMBED_CACHE_SIZE` | `20000` | 文档向量 LRU 上限 |
| `EDR_RAG_VECTOR_BACKEND` | `persistent` | `persistent` \| `simple`(纯内存) |
| `EDR_RAG_VECTOR_DIR` | `<workdir>/vector_index` | 向量持久化目录 |
| `EDR_RAG_RERANK` | 启用 | `0` 关闭 cross-encoder 精排（运行时可改） |
| `EDR_RAG_RERANK_TEXT_MAX` | `1200` | 送进精排的正文字符上限 |
| `EDR_RAG_CJK_BIGRAM` | `1` | `0` 回退到纯 unigram 分词（仅对照实验） |
| `EDR_RAG_SYNONYMS` | 随包词表 | 自定义同义词表路径 |

### 服务

| 变量 | 默认 | 说明 |
|---|---|---|
| `EDR_RAG_API_KEY` | — | 单 key 模式 |
| `EDR_RAG_API_KEYS_FILE` | — | 多 key + ACL + 限流 |
| `EDR_RAG_RATE_LIMIT` | `0`(不限) | 单 key 模式下的每秒请求数 |

### 可观测性

| 变量 | 默认 | 说明 |
|---|---|---|
| `EDR_RAG_TRACE` | `1` | `0` 关闭结构化追踪 |
| `EDR_RAG_TRACE_MAX_BYTES` | `64MB` | 单文件上限，超过轮转 |
| `EDR_RAG_TRACE_BACKUPS` | `5` | 保留的历史文件数 |
| `EDR_RAG_TRACE_BUFFER` | `1` | 攒够多少条落盘；调大降低 IO，代价是崩溃时丢观测数据 |

### OCR

| 变量 | 默认 | 说明 |
|---|---|---|
| `EDR_RAG_OCR` | 启用 | `0` 关闭图片 OCR |

---

## 6. 索引更新与回滚

```bash
edr-rag index list     --workdir ~/edr-rag-work
edr-rag index capture  --workdir ~/edr-rag-work --corpus actions
edr-rag index rollback --workdir ~/edr-rag-work --corpus actions
```

`capture` 会连同语料快照一起存档，`rollback` 恢复快照文件 —— 不是只翻一个
没有运行时效果的 active 标记。回滚后进程内缓存按文件签名自动失效，向量索引
惰性重建。

---

## 7. 运行前自检

```bash
# 产物完整性
edr-rag validate --workdir ~/edr-rag-work

# 检索质量（评测集需按实际手册标注，随包的 30 条只是示例）
edr-rag-eval compare --workdir ~/edr-rag-work --data your-eval-set.jsonl

# 融合权重调参（train/holdout 切分，会同时报过拟合间隙与调参净收益）
edr-rag-eval tune --workdir ~/edr-rag-work --data your-eval-set.jsonl
```

> **随包评测集只有 30 条、覆盖 3 个 action，只能当格式示例。** 上线前必须用真实
> 手册与真实查询重新标注一份，否则所有指标都没有预测力。
