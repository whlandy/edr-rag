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
