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
docs/requirements/EDR-RAG-DESIGN-V2.md
```

The original phased proposal remains available at
`docs/requirements/EDR-RAG-DESIGN.md` as a historical baseline.

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

## LlamaIndex semantic and hybrid retrieval

LlamaIndex is an optional backend. The existing keyword/BM25 search continues
to work without it.

```bash
python -m pip install -e '.[rag]'

# Build local indexes for every available artifact type. The embedding model is
# downloaded from Hugging Face on first use.
edr-rag index \
  --workdir ~/Desktop/edr-chm-rag-work \
  --type all \
  --show-progress

# Vector-only or BM25 + vector reciprocal-rank fusion.
edr-rag search --workdir ~/Desktop/edr-chm-rag-work \
  --type section --strategy vector --query '如何设置网络代理'

edr-rag search --workdir ~/Desktop/edr-chm-rag-work \
  --type procedure --strategy hybrid --query '如何设置网络代理'
```

Indexes are stored under `WORKDIR/.llamaindex/`. The default local embedding
model is `BAAI/bge-small-zh-v1.5`; override it consistently during indexing and
search with `--embed-model`.

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
