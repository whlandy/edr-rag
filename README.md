# edr-rag

`edr-rag` is the manual knowledge side of the EDR agent stack.

It is separate from `edr-wd`:

```text
edr-rag tells the agent what to do.
edr-wd performs GUI and PowerShell actions.
skills/instructions tell the agent how to combine them safely.
```

Implementation phases:

```text
P0: CHM extraction and structured Manual IR
P1: Procedure extraction and local JSONL search
P2: Action catalog and Python retrieval API
P3: edr-rag MCP tools and execution feedback loop
```

Design document:

```text
docs/requirements/EDR-RAG-DESIGN.md
```

Default temporary workspace:

```text
~/Desktop/edr-chm-rag-work/
```

Basic commands:

```bash
edr-rag ingest --chm ~/Desktop/edr-chm-rag-work/input/product-manual.chm \
  --manual-id product-manual \
  --product ProductName \
  --workdir ~/Desktop/edr-chm-rag-work

edr-rag validate --workdir ~/Desktop/edr-chm-rag-work
```

HTTP service (P4):

```bash
export EDR_RAG_API_KEY="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
edr-rag-serve --workdir ~/Desktop/edr-chm-rag-work --port 8080
```

Pure stdlib — no fastapi/uvicorn to ship into an air-gapped site. The service
refuses to start without an API key. See `docs/DEPLOYMENT.md` for the offline
deployment guide and the full environment-variable reference.

MCP boundary:

```text
edr-rag MCP:
  manual_search
  procedure_search
  action_search
  get_action
  submit_feedback

edr-wd MCP:
  connect
  dump_tree
  click_target
  run_powershell
```

Design documents:

```text
docs/requirements/EDR-RAG-DESIGN.md      P0–P3 原始设计
docs/requirements/P4-HARDENING-DESIGN.md P4 商用化硬化（阻塞级 + 严重级修复）
docs/DEPLOYMENT.md                       内网离线部署指南
```
