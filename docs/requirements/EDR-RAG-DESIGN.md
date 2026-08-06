# CHM-to-Agent Manual Knowledge System Design

## 0. Purpose

This document defines a phased implementation plan for turning CHM product manuals into agent-usable knowledge, procedures, and action schemas.

The system should not treat CHM files as PDF-like text blobs. A CHM manual is already a structured HTML help package. The implementation should preserve that structure and convert it into:

- searchable manual sections
- extracted procedures
- warnings, preconditions, risks, and recovery notes
- image and screenshot references
- action catalog candidates for GUI agent planning

The design is organized by implementation priority:

```text
P0: CHM extraction and structured manual IR
P1: Procedure extraction and retrieval-ready knowledge chunks
P2: Action catalog generation and planner integration
P3: Execution feedback loop and continuous improvement
```

Hermes should implement each P level in order. Do not jump directly to vector DB or action planning before P0/P1 artifacts are stable.

## 1. Temporary File Location

Use a desktop workspace for temporary files:

```text
~/Desktop/edr-chm-rag-work/
```

This is intentionally outside the git repo because extracted CHM files can be large, vendor-owned, and noisy.

Recommended directory structure:

```text
~/Desktop/edr-chm-rag-work/
├── input/
│   └── product-manual.chm
├── extracted/
│   └── product-manual/
│       ├── toc.hhc
│       ├── index.hhk
│       ├── *.html
│       ├── images/
│       └── css/
├── normalized/
│   ├── manifest.json
│   ├── sections.jsonl
│   ├── procedures.jsonl
│   ├── warnings.jsonl
│   └── assets.jsonl
├── action_catalog/
│   ├── actions.yaml
│   └── actions.jsonl
├── vector_index/
│   └── qdrant_snapshot/
└── logs/
    ├── ingestion.log
    ├── extraction.log
    ├── validation.log
    └── feedback.log
```

Make the path configurable:

```text
CHM_RAG_WORKDIR=~/Desktop/edr-chm-rag-work
```

Versioned outputs can later be copied into the project:

```text
references/manuals/
references/action_catalog/
```

## 2. Target Architecture

```text
CHM manual
  |
  v
P0 extraction + Manual IR
  |
  v
P1 procedure extraction + knowledge chunks
  |
  v
P2 action catalog + planner integration
  |
  v
P3 execution feedback + catalog/RAG updates
```

The core principle:

```text
Manual text explains.
Manual IR structures.
Procedures guide.
Action catalog plans.
Executor acts.
Feedback improves.
```

## 3. RAG Usage Model

RAG should be implemented as an independent manual knowledge capability, not as a skill by itself.

Recommended layering:

```text
CHM ingestion pipeline
  -> produces structured artifacts

EDR RAG library/service
  -> searches sections, procedures, chunks, and action catalog

MCP interface
  -> exposes manual search tools to agents

Skill / agent instruction
  -> tells the agent when and how to use the EDR RAG tools

edr-wd MCP
  -> executes GUI and PowerShell actions on the Windows target
```

Clear responsibility split:

```text
edr-rag:
  knows what the manual says and which procedure/action should be used

edr-wd:
  controls the Windows GUI, clicks controls, reads windows, runs PowerShell

skill:
  teaches the agent the workflow:
    search action catalog first
    fallback to procedures
    require confirmation for risky actions
    execute through edr-wd
```

The RAG layer should not directly click GUI controls. It should return structured knowledge and action candidates. The executor layer should perform actions.

### 3.1 Recommended Agent Flow

For an operational user request:

```text
User request
  |
  v
Agent
  |
  | 1. action_search(query)
  v
edr-rag MCP
  |
  | returns action candidates with risk/source/procedure
  v
Agent planner
  |
  | 2. if no action found, procedure_search(query)
  v
edr-rag MCP
  |
  | returns procedure steps and citations
  v
Agent planner
  |
  | 3. build plan + check preconditions/risk
  v
Confirmation gate
  |
  | 4. execute approved plan
  v
edr-wd MCP
  |
  | 5. sense/verify result
  v
feedback log
```

Example:

```text
User: 帮我配置代理

Agent:
  1. action_search("配置代理")
  2. get_action("network.configure_proxy")
  3. inspect risk/preconditions/source
  4. ask for confirmation if needed
  5. use edr-wd MCP:
       connect
       dump_tree
       click_target
       type_text
       screenshot / wait_window
```

### 3.2 RAG as CLI First, MCP Later

Implementation should not start by building a complex MCP server. Start with a deterministic CLI and JSONL artifacts.

Evolution path:

```text
P0:
  no RAG server
  only ingestion CLI and JSONL outputs

P1:
  local search CLI over sections/procedures/chunks
  no vector DB required

P2:
  edr-rag Python library
  optional vector index
  action search and procedure search APIs

P3:
  expose edr-rag as MCP tools
  integrate feedback and re-indexing
```

This avoids mixing ingestion, retrieval, and agent runtime too early.

### 3.3 Future edr-rag MCP Tools

When the search APIs are stable, expose them through a separate MCP server.

Recommended MCP server name:

```text
edr-rag
```

Recommended tools:

```text
manual_search(query, product=None, module=None, type=None, limit=5)
procedure_search(query, product=None, module=None, limit=5)
get_procedure(procedure_id)
action_search(query, product=None, module=None, risk=None, limit=5)
get_action(action_id)
list_actions(product=None, module=None, status=None)
submit_feedback(action_id=None, procedure_id=None, failure_type=None, evidence=None)
```

Tool responsibilities:

```text
manual_search:
  general manual section search

procedure_search:
  retrieve step-by-step procedures

action_search:
  retrieve executable/reviewable action catalog entries

get_action:
  return full action schema with risk, confirmation policy, preconditions, source, and procedure

submit_feedback:
  record execution failures or human corrections for P3 improvement
```

The edr-rag MCP should not expose GUI tools. GUI tools remain in edr-wd MCP.

### 3.4 Skill Role

A skill should be a thin instruction layer, not the knowledge database.

The skill should say:

```text
When the user asks how to operate the product:
  1. call action_search first
  2. if no action is found, call procedure_search
  3. cite manual source
  4. check risk and preconditions
  5. ask confirmation for medium/high risk
  6. execute only through edr-wd MCP
  7. verify result
  8. submit feedback if execution fails
```

The skill should not contain the full manual. It should only contain the workflow for using edr-rag and edr-wd together.

## 4. P0: CHM Extraction and Manual IR

### Goal

Extract a CHM file into structured, inspectable artifacts without losing the table of contents, source paths, images, or document hierarchy.

P0 does not need embeddings, vector DB, or LLM extraction. It should be deterministic and easy to debug.

### Scope

Implement:

- CHM extraction
- TOC parsing
- HTML cleanup
- image asset registry
- `ManualSection` JSONL output
- ingestion manifest

### Non-goals

Do not implement in P0:

- vector DB
- action catalog
- planner integration
- automated GUI execution
- LLM-based procedure extraction

### Extraction Commands

Windows:

```cmd
hh.exe -decompile "%USERPROFILE%\Desktop\edr-chm-rag-work\extracted\product-manual" "%USERPROFILE%\Desktop\edr-chm-rag-work\input\product-manual.chm"
```

Linux:

```bash
extract_chmLib manual.chm ~/Desktop/edr-chm-rag-work/extracted/product-manual
```

### Proposed Module Layout

Create a new package or scripts under the repo:

```text
tools/chm_ingest/
├── __init__.py
├── extract.py
├── parse_toc.py
├── parse_html.py
├── normalize.py
├── schema.py
└── cli.py
```

If the project prefers fewer files initially, start with:

```text
tools/chm_ingest.py
```

and split later.

### P0 Schema: Manifest

Write:

```text
normalized/manifest.json
```

Example:

```json
{
  "manual_id": "product-manual",
  "source_chm": "~/Desktop/edr-chm-rag-work/input/product-manual.chm",
  "source_hash": "sha256:...",
  "extracted_dir": "~/Desktop/edr-chm-rag-work/extracted/product-manual",
  "generated_at": "2026-08-05T12:00:00+08:00",
  "section_count": 128,
  "asset_count": 42,
  "parser_version": "p0"
}
```

### P0 Schema: ManualSection

Write one JSON object per line:

```text
normalized/sections.jsonl
```

Schema:

```json
{
  "id": "manual.section.network.configure_proxy",
  "manual_id": "product-manual",
  "product": "ProductName",
  "title": "Configure Proxy",
  "section_path": ["Network", "Proxy", "Configure Proxy"],
  "source_file": "network/proxy.html",
  "anchor": null,
  "content": "Cleaned text content...",
  "headings": ["Configure Proxy"],
  "links": [
    {"text": "Troubleshooting", "href": "troubleshooting.html"}
  ],
  "images": [
    {"src": "images/proxy.png", "alt": "", "asset_id": "asset.images.proxy_png"}
  ],
  "metadata": {
    "language": "zh-CN",
    "module": "network"
  }
}
```

### P0 Schema: AssetRef

Write:

```text
normalized/assets.jsonl
```

Schema:

```json
{
  "id": "asset.images.proxy_png",
  "manual_id": "product-manual",
  "type": "image",
  "path": "images/proxy.png",
  "absolute_path": "~/Desktop/edr-chm-rag-work/extracted/product-manual/images/proxy.png",
  "referenced_by": ["network/proxy.html"],
  "description": null
}
```

### P0 Implementation Notes

- Use BeautifulSoup for HTML parsing.
- Preserve original `source_file`.
- Normalize whitespace but do not discard list structure.
- Convert relative image paths to stable asset IDs.
- Use the CHM TOC file (`*.hhc`) to build `section_path`.
- If TOC parsing fails, fallback to HTML heading hierarchy.
- Do not delete extracted files automatically.

### P0 CLI

Recommended command:

```bash
python -m tools.chm_ingest.cli ingest \
  --chm ~/Desktop/edr-chm-rag-work/input/product-manual.chm \
  --manual-id product-manual \
  --product ProductName \
  --workdir ~/Desktop/edr-chm-rag-work
```

### P0 Acceptance Criteria

P0 is complete when:

- CHM extraction succeeds
- `manifest.json` exists
- `sections.jsonl` exists and has one record per useful HTML page/section
- `assets.jsonl` exists and image paths resolve
- each section has `id`, `title`, `section_path`, `source_file`, and `content`
- no vector DB is required to inspect the result

## 5. P1: Procedure Extraction and Retrieval Chunks

### Goal

Convert manual sections into structured procedures and retrieval-ready knowledge chunks.

P1 should make the manual useful for answering “how do I do X?” and for finding step-by-step SOPs.

### Scope

Implement:

- procedure detection
- step extraction
- warning extraction
- precondition extraction
- recovery extraction
- retrieval chunk generation with metadata
- local JSONL search CLI

### Non-goals

Do not implement in P1:

- GUI execution
- action catalog publication
- confirmation gates
- automatic selector generation

### Procedure Detection Heuristics

Detect procedure-like sections using:

- ordered lists
- numbered paragraphs
- headings like 操作步骤, 配置, 添加, 删除, 启用, 禁用, 重置, 导出
- blocks starting with Step, 步骤, 注意, 警告, 重要
- verbs such as 点击, 选择, 输入, 打开, 关闭, 确认, 保存, 导出

### P1 Schema: Procedure

Write:

```text
normalized/procedures.jsonl
```

Schema:

```json
{
  "id": "procedure.network.configure_proxy",
  "manual_id": "product-manual",
  "product": "ProductName",
  "operation": "configure_proxy",
  "title": "配置代理服务器",
  "section_path": ["网络配置", "代理服务器"],
  "source": {
    "file": "network/proxy.html",
    "section_id": "manual.section.network.configure_proxy",
    "anchor": null
  },
  "preconditions": [
    "administrator_required",
    "network_available"
  ],
  "steps": [
    {
      "order": 1,
      "instruction": "打开系统设置",
      "target": "设置",
      "action_hint": "open_window"
    },
    {
      "order": 2,
      "instruction": "点击网络配置",
      "target": "网络配置",
      "action_hint": "click"
    }
  ],
  "warnings": [
    "修改代理可能影响网络连接"
  ],
  "recovery": [
    "如果连接失败，恢复为默认代理设置"
  ],
  "confidence": 0.76,
  "status": "candidate"
}
```

### P1 Schema: RetrievalChunk

Write:

```text
normalized/chunks.jsonl
```

Schema:

```json
{
  "id": "chunk.procedure.network.configure_proxy",
  "manual_id": "product-manual",
  "type": "procedure",
  "text": "配置代理服务器\n前置条件: ...\n步骤: ...",
  "source": {
    "section_id": "manual.section.network.configure_proxy",
    "procedure_id": "procedure.network.configure_proxy",
    "file": "network/proxy.html"
  },
  "metadata": {
    "product": "ProductName",
    "module": "network",
    "type": "procedure",
    "risk": "unknown",
    "language": "zh-CN"
  }
}
```

### P1 Extraction Strategy

Use deterministic extraction first:

```text
HTML structure -> headings/lists/tables -> candidate Procedure
```

Optionally use LLM only as a second pass:

```text
ManualSection + deterministic candidate -> LLM cleanup -> Procedure
```

If using LLM, require source preservation. The LLM must not invent steps.

### P1 Local Search

Implement a simple local search before introducing vector DB.

Recommended command:

```bash
python -m tools.chm_ingest.cli search \
  --query "配置代理" \
  --workdir ~/Desktop/edr-chm-rag-work \
  --type procedure \
  --limit 5
```

P1 search can use:

```text
keyword match
BM25
simple title/path boost
procedure type filter
```

Return:

```json
{
  "ok": true,
  "query": "配置代理",
  "results": [
    {
      "id": "procedure.network.configure_proxy",
      "type": "procedure",
      "score": 12.4,
      "title": "配置代理服务器",
      "source": {"file": "network/proxy.html"},
      "snippet": "打开系统设置..."
    }
  ]
}
```

### P1 Acceptance Criteria

P1 is complete when:

- `procedures.jsonl` exists
- every procedure has at least one source citation
- every procedure has ordered steps
- warnings and preconditions are captured when present
- `chunks.jsonl` exists and contains structured chunks
- simple keyword or semantic search can find relevant procedures

## 6. P2: Action Catalog and Planner Integration

### Goal

Convert procedures into action candidates that a planner can reason about.

P2 does not need perfect GUI selectors. It should produce safe, reviewable action schemas with risk and confirmation metadata.

### Scope

Implement:

- action candidate generation
- risk classification
- confirmation policy
- precondition mapping
- success criteria
- recovery mapping
- planner retrieval flow
- edr-rag Python API
- optional vector index

### Action Lifecycle

Use explicit statuses:

```text
candidate -> needs_review -> approved -> published -> deprecated
```

High-risk actions must not become `published` without review.

### P2 Schema: Action Catalog Entry

Write:

```text
action_catalog/actions.yaml
action_catalog/actions.jsonl
```

Schema:

```yaml
action_id: network.configure_proxy
product: ProductName
module: network
title: 配置代理服务器
status: needs_review
confidence: 0.72
risk: medium
requires_confirmation: true
preconditions:
  - administrator_required
  - network_available
procedure:
  - order: 1
    action: open_window
    target: 设置
    source_instruction: 打开系统设置
  - order: 2
    action: click
    target: 网络配置
    source_instruction: 点击网络配置
  - order: 3
    action: input_text
    target: 代理地址
    value_source: user
success_criteria:
  - window_contains: 配置成功
recovery:
  - restore_default_proxy
source:
  manual_id: product-manual
  procedure_id: procedure.network.configure_proxy
  file: network/proxy.html
validation:
  has_steps: true
  has_risk: true
  has_recovery: true
  source_verified: true
```

### Risk Policy

```text
low:
  read-only operations, navigation, viewing status, viewing logs

medium:
  configuration changes, export, import, restart component

high:
  delete, reset, disable protection, wipe data, irreversible state change
```

Confirmation rules:

```text
requires_confirmation = true if risk is medium or high
requires_confirmation = true if action changes security state
requires_confirmation = true if rollback is unavailable
requires_confirmation = true if source confidence is low
```

### Planner Retrieval Flow

For an operational user request:

```text
1. classify intent
2. search action catalog first
3. if no action found, search procedures
4. build plan from action/procedure
5. validate preconditions
6. apply confirmation gate
7. send plan to executor
```

The planner should call the manual knowledge layer in this order:

```text
1. action_search
2. get_action
3. procedure_search if action_search has no result
4. manual_search only for explanation or troubleshooting context
```

Planner input:

```json
{
  "user_request": "帮我配置代理",
  "product": "ProductName",
  "retrieved_actions": ["network.configure_proxy"],
  "retrieved_procedures": ["procedure.network.configure_proxy"]
}
```

Planner output:

```json
{
  "plan_id": "plan.network.configure_proxy.001",
  "action_id": "network.configure_proxy",
  "risk": "medium",
  "requires_confirmation": true,
  "preconditions": ["administrator_required"],
  "steps": [
    {"order": 1, "tool": "gui.open", "target": "设置"},
    {"order": 2, "tool": "gui.click", "target": "网络配置"},
    {"order": 3, "tool": "gui.input_text", "target": "代理地址"}
  ],
  "success_criteria": ["配置成功提示出现"]
}
```

### P2 Acceptance Criteria

P2 is complete when:

- `actions.yaml` and `actions.jsonl` are generated
- each action links back to a source procedure
- each action has risk and confirmation policy
- high-risk actions are not auto-published
- planner can retrieve action candidates before free-text RAG
- planner output references action IDs and source procedures
- a Python API exists for `manual_search`, `procedure_search`, `action_search`, and `get_action`

## 7. P3: Execution Feedback Loop

### Goal

Use execution results to improve the action catalog, procedure extraction, selectors, and retrieval metadata.

P3 turns the system from a static ingestion pipeline into a learning loop.

### Scope

Implement:

- execution log capture
- failure classification
- feedback item generation
- selector synonym updates
- action patch proposals
- re-index trigger
- edr-rag MCP server

### Feedback Sources

Capture feedback from:

- failed clicks
- missing UI labels
- low-confidence retrieval
- user corrections
- failed success criteria
- screenshots or dump_tree output
- manual version changes

### P3 Schema: Feedback Item

Write:

```text
logs/feedback.log
normalized/feedback.jsonl
```

Schema:

```json
{
  "feedback_id": "fb_001",
  "manual_id": "product-manual",
  "action_id": "network.configure_proxy",
  "procedure_id": "procedure.network.configure_proxy",
  "failure_type": "button_text_not_found",
  "expected": "保存",
  "observed": "保存配置",
  "evidence": {
    "screenshot": "runs/run_001/screenshot.png",
    "dump_tree": "runs/run_001/dump_tree.json"
  },
  "suggested_patch": {
    "selector_synonyms": ["保存", "保存配置"]
  },
  "status": "needs_review",
  "created_at": "2026-08-05T12:00:00+08:00"
}
```

### P3 Loop

```text
execute plan
  -> observe UI state
  -> verify success criteria
  -> classify failure if any
  -> write feedback item
  -> propose action/procedure patch
  -> human or reviewer approves
  -> update catalog/chunks
  -> re-index
```

### P3 Acceptance Criteria

P3 is complete when:

- every execution run can produce structured logs
- failures generate feedback items
- feedback can propose catalog patches
- patches are reviewable before publishing
- re-index can be triggered after approved patches
- edr-rag exposes MCP tools for search/action retrieval/feedback
- edr-wd remains the separate execution MCP

## 8. Image Handling Roadmap

### P0

Register images as assets only.

### P1

Link images to sections and procedure steps.

### P2

Use image references as planner context.

### P3

Optional visual intelligence:

- OCR for screenshots
- VLM-generated image descriptions
- template matching for GUI executors
- visual selector feedback

Initial asset schema:

```json
{
  "id": "asset.reset_button.png",
  "type": "screenshot",
  "path": "images/reset_button.png",
  "source_file": "device/reset.html",
  "related_procedure": "procedure.device.reset",
  "related_step": 3,
  "description": "Reset button location"
}
```

## 9. Suggested Components by Phase

| Phase | Need | Suggested Component |
|------|------|---------------------|
| P0 | CHM extraction | `hh.exe` on Windows, `extract_chmLib` on Linux |
| P0 | HTML parsing | BeautifulSoup |
| P0 | storage | JSONL + manifest JSON |
| P1 | procedure extraction | deterministic parser, optional LLM cleanup |
| P1 | retrieval chunks | JSONL first |
| P2 | vector DB | Qdrant |
| P2 | embedding | BGE-M3 |
| P2 | reranker | bge-reranker-v2 |
| P2 | orchestration | LlamaIndex or existing planner |
| P2 | edr-rag API | Python library over JSONL/vector index |
| P3 | MCP interface | separate edr-rag MCP server |
| P3 | execution feedback | existing executor logs + JSONL feedback |

## 10. Hermes Implementation Instructions

Hermes should implement in this order:

```text
1. P0 only:
   - create workdir
   - extract CHM
   - parse TOC and HTML
   - write manifest/sections/assets
   - add smoke test

2. P1:
   - add procedure extraction
   - write procedures/chunks
   - add local JSONL search CLI
   - add validation script

3. P2:
   - generate action candidates
   - add risk and confirmation fields
   - add planner retrieval API
   - add edr-rag Python API
   - optionally add vector index

4. P3:
   - expose edr-rag MCP tools
   - add execution feedback schema
   - add patch proposal workflow
   - add re-index workflow
```

Hermes should not:

- convert CHM to PDF
- chunk raw HTML before building IR
- put extracted CHM files into git
- publish high-risk action catalog entries without review
- use vector DB as the source of truth
- mix edr-rag tools into edr-wd MCP
- let RAG directly execute GUI actions

## 11. Recommended CLI Shape

P0:

```bash
python -m tools.chm_ingest.cli ingest \
  --chm ~/Desktop/edr-chm-rag-work/input/product-manual.chm \
  --manual-id product-manual \
  --product ProductName \
  --workdir ~/Desktop/edr-chm-rag-work
```

P1:

```bash
python -m tools.chm_ingest.cli extract-procedures \
  --manual-id product-manual \
  --workdir ~/Desktop/edr-chm-rag-work
```

```bash
python -m tools.chm_ingest.cli search \
  --query "配置代理" \
  --type procedure \
  --workdir ~/Desktop/edr-chm-rag-work
```

P2:

```bash
python -m tools.chm_ingest.cli build-actions \
  --manual-id product-manual \
  --workdir ~/Desktop/edr-chm-rag-work
```

```bash
python -m tools.chm_ingest.cli action-search \
  --query "配置代理" \
  --workdir ~/Desktop/edr-chm-rag-work
```

P3:

```bash
python -m tools.chm_ingest.cli apply-feedback \
  --manual-id product-manual \
  --workdir ~/Desktop/edr-chm-rag-work
```

Future MCP configuration:

```yaml
edr-rag:
  url: http://127.0.0.1:<port>/mcp
  transport: streamable-http

edr-wd:
  url: http://127.0.0.1:18765/mcp
  transport: streamable-http
```

## 12. Overall Acceptance Criteria

The project is successful when:

- P0 can extract and normalize a CHM manual
- P1 can produce procedures with cited steps
- P2 can generate reviewable action catalog candidates
- P3 can turn execution failures into structured feedback
- the planner uses action catalog entries before free-text RAG
- the executor never runs high-risk actions without confirmation
- all generated artifacts can be inspected as JSONL/YAML before indexing
- edr-rag and edr-wd are separate MCP servers
- skill instructions describe how to combine edr-rag retrieval with edr-wd execution

## 13. Final Summary

This is not a generic RAG chatbot design.

It is a phased CHM-to-agent operational knowledge system:

```text
P0: preserve structure
P1: extract procedures
P2: generate actions
P3: learn from execution
```

The correct source of truth is structured artifacts on disk, not embeddings. Embeddings and vector DB are retrieval accelerators only.

Final usage model:

```text
edr-rag tells the agent what to do.
edr-wd performs the GUI/PowerShell actions.
skill instructions tell the agent how to combine them safely.
```
