# edr-rag

`edr-rag` is the manual knowledge side of the EDR agent stack.

It is intentionally separate from `edr-wd`:

```text
edr-rag tells the agent what to do.
edr-wd performs GUI and PowerShell actions.
skills/instructions tell the agent how to combine them safely.
```

Implementation plan:

```text
P0: CHM extraction and structured Manual IR
P1: Procedure extraction and local JSONL search
P2: Action catalog and Python retrieval API
P3: edr-rag MCP tools and execution feedback loop
```

See:

```text
docs/requirements/EDR-RAG-DESIGN.md
```

Current CLI:

```bash
edr-rag init-workdir
```

Default temporary workspace:

```text
~/Desktop/edr-chm-rag-work/
```
