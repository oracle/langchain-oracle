# Deepagents Agent (langchain-oci)

This module provides `create_deepagents_agent(...)` for multi-step research
workflows on OCI. It wires an OCI GenAI chat model (or any LangChain chat
model you bring) to auto-generated retrieval tools over Oracle Autonomous
Database and OpenSearch vector stores, and — on the full path — to the
`deepagents` harness: planning, a virtual filesystem, subagents, skills,
memory, and human-in-the-loop controls.

## Documentation

| Page | Covers |
|---|---|
| [Core guide](../../../docs/deepagents/guide.md) | Architecture, the two build paths, construction lifecycle, complete parameter reference, environment variables |
| [Datastores & retrieval](../../../docs/deepagents/datastores.md) | `VectorDataStore` contract, ADB and OpenSearch adapters, custom stores, generated tools, query routing, hybrid search (RRF), embeddings |
| [Persistence](../../../docs/deepagents/persistence.md) | Thread checkpoints (`OracleSaver`), long-term memory (`OracleStore`), durable virtual filesystem (`StoreBackend`) |
| [Capabilities](../../../docs/deepagents/capabilities.md) | Built-in tools, subagents, skills, memory, human-in-the-loop, filesystem permissions, structured output, middleware |
| [Operations](../../../docs/deepagents/operations.md) | Bring-your-own model, async & cleanup, logging, sharp edges, troubleshooting, testing |

## Installation & compatibility

```bash
pip install -U "langchain-oci[deepagents]"
```

The `deepagents` extra installs:

| Dependency | Pin | Why |
|---|---|---|
| `deepagents` | `>=0.6.1` (only on Python ≥3.11, <3.14) | The deep-agent harness (planning, filesystem, subagents) |
| `langchain-oracledb` | `>=1.3.0` | `OracleVS` vector store, `OracleTextSearchRetriever`, `OracleTextSplitter` |
| `langgraph-oracledb` | `>=1.0.1` | Oracle-backed checkpointers and stores for `checkpointer=` / `store=` |
| `opensearch-py` | `>=2.4.0` | OpenSearch datastore backend |

**Python matrix:**

| Python | Lightweight path (`middleware=[]`) | Deep path (default) |
|---|---|---|
| 3.9 – 3.10 | ✅ works (no `deepagents` needed) | ❌ `RuntimeError` (deepagents requires ≥3.11) |
| 3.11 – 3.13 | ✅ | ✅ |
| 3.14 | ✅ | ⚠️ extra does not install `deepagents` (not yet tested upstream) |

Only the pieces you use are imported: `oracledb`/`langchain-oracledb` only when
an `ADB` store connects, `opensearch-py` only for `OpenSearch`, `deepagents`
only on the deep path.

## Quickstart

```python
from langchain_core.messages import HumanMessage
from langchain_oci import create_deepagents_agent
from langchain_oci.datastores import ADB

agent = create_deepagents_agent(
    datastores={
        "research": ADB(
            dsn="mydb_low",
            user="ADMIN",
            password="***",
            wallet_location="~/wallets/mydb",
            datastore_description="medical research papers, clinical trials",
        ),
    },
    model_id="google.gemini-2.5-pro",
    compartment_id="ocid1.compartment...",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    auth_type="API_KEY",
    top_k=8,
)

result = agent.invoke(
    {"messages": [HumanMessage(content="Summarize the evidence on leukocytosis treatment")]}
)
print(result["messages"][-1].content)
```

When `datastores=...` is provided, the agent gets three datastore tools
automatically:

- `stats` — sizes and per-store metadata
- `search` — hybrid semantic + keyword retrieval with automatic store routing
- `get_document` — fetch a full document by id

Because this takes the deep path by default, the agent also gets the
deepagents built-ins (filesystem tools and the `task` subagent tool). See the
[Core guide](../../../docs/deepagents/guide.md) for the lightweight opt-out
(`middleware=[]`) and the full parameter reference.

## Why This Is In Scope For `langchain-oci`

This functionality is an OCI-to-LangChain integration concern, not an
application-layer feature: agent helpers are first-class public API in
`langchain_oci.agents`, the heavy dependencies are isolated in an optional
extra, the datastore and deepagents surfaces are covered by unit and
integration tests, and the implementation composes existing SDK primitives
(OCI chat model, OCI embeddings, Oracle/OpenSearch datastores, LangChain
tools) rather than introducing a separate product surface.

Scope boundary: the SDK provides integration primitives and reference
examples. Dataset hosting/curation and domain-specific prompts remain
user-owned.

## Data Provenance In This Repository

The deepagents examples use repository scripts to make provenance explicit:

1. `libs/oci/scripts/upload_research_datasets.py`
   - pulls MedMCQA, PubMedQA, and CUAD from Hugging Face
   - uploads JSON artifacts to OCI Object Storage buckets
2. `libs/oci/scripts/upload_large_datasets.py` (optional)
   - uploads larger corpora (Wikipedia, C4, ArXiv) to OCI Object Storage
3. `libs/oci/scripts/vectorize_datasets.py`
   - reads objects from buckets, generates embeddings, writes vectors into the
     ADB table `VECTOR_DOCUMENTS`

Runtime examples then perform retrieval/synthesis over these indexed
documents.

## Samples

| Sample | Shows |
|---|---|
| [`samples/11-deepagents/adb_multi_store_huggingface_example.py`](../../../../../samples/11-deepagents/adb_multi_store_huggingface_example.py) | Two ADB-backed stores: ingestion → multi-store research → markdown memo |
| [`samples/11-deepagents/opensearch_multi_index_huggingface_example.py`](../../../../../samples/11-deepagents/opensearch_multi_index_huggingface_example.py) | Same flow on two OpenSearch indices |
| [`samples/11-deepagents/local_docker_oracle_deepagents.ipynb`](../../../../../samples/11-deepagents/local_docker_oracle_deepagents.ipynb) | Local Docker Oracle + BYO Claude model, end-to-end |
| [`samples/11-deepagents/persistence_oracle_example.py`](../../../../../samples/11-deepagents/persistence_oracle_example.py) | `OracleSaver` thread resume + checkpoint inspection |

## Common Questions

**Where does the data come from?** Your ingestion pipeline — the stores search
what's already indexed. See Data Provenance above for the reference pipeline.

**Do I need to implement a search class for ADB?** No. `ADB(...)` +
`create_deepagents_agent(datastores=...)` is the canonical path; tools are
generated automatically.

**Can I use the tools without the deep agent?** Yes —
`create_datastore_tools(...)` returns plain LangChain tools for any agent
(e.g. `create_oci_agent`).

**Which chat models work best?** Any tool-calling model. On OCI GenAI,
Gemini 2.5 Pro is the tuned default; raise `max_tokens` for long reports. Any
provider works via `model=` — see
[Operations](../../../docs/deepagents/operations.md#bring-your-own-model).

**How do I make the agent remember across sessions?**
`checkpointer=OracleSaver(...)` + a stable `thread_id` for conversation state;
`store=`/`backend=StoreBackend()` for durable files and long-term memory — see
[Persistence](../../../docs/deepagents/persistence.md).

**Is Python 3.10 supported?** Only on the lightweight path (`middleware=[]`).
The deep path needs 3.11+.

## Async Cleanup Note

If your workflow keeps agents/models alive across many async calls, explicitly
close the underlying model client when done to avoid unclosed HTTP session
warnings:

```python
llm = getattr(agent, "_oci_llm", None)
if llm is not None and hasattr(llm, "aclose"):
    await llm.aclose()
```
