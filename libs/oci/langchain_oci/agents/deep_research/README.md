# Deep Research Agent (langchain-oci)

This module provides `create_deep_research_agent(...)` for multi-step research workflows on OCI.

It supports:
- OCI GenAI chat models for reasoning and synthesis
- optional datastore-backed retrieval (`ADB`, `OpenSearch`)
- optional custom tools

## Python Compatibility

Deep research support is provided via the optional extra:
- `pip install langchain-oci[deep-research]`

Current compatibility for the `deep-research` extra is:
- Python `>=3.11,<3.14` (driven by `deepagents` support bounds)

## Why This Is In Scope For `langchain-oci`

This functionality is in scope for this SDK because it is an OCI-to-LangChain
integration concern, not an unrelated application layer feature.

Evidence from this repository:
- Agent helpers are first-class public API in `langchain_oci.agents`
  (`create_oci_agent`, `create_deep_research_agent`, datastore adapters).
- The package explicitly declares deep research as an optional integration extra
  (`[project.optional-dependencies].deep-research` in `libs/oci/pyproject.toml`),
  keeping core installs lean while supporting advanced workflows.
- The datastore and deep-research surfaces are covered by unit and integration
  tests under `libs/oci/tests/unit_tests/agents/` and
  `libs/oci/tests/integration_tests/agents/`.
- The implementation composes existing SDK primitives (OCI chat model,
  OCI embeddings, OCI/OpenSearch datastores, LangChain tools) instead of
  introducing a separate product surface.

Scope boundary:
- The SDK provides integration primitives and reference examples.
- Dataset hosting/curation and domain-specific prompts remain user-owned.

PR rationale (review-ready):
This feature belongs here because `langchain-oci` is the OCI integration layer
for LangChain. Deep research support in this package wires OCI model/runtime
capabilities to agent abstractions via reusable APIs (agent factory, datastore
adapters, and tool generation), while keeping optional dependencies isolated in
an extra. That is repository-scope integration work, not app-level logic.

## Data Provenance In This Repository

The deep-research examples use repository scripts to make provenance explicit:

1. `libs/oci/scripts/upload_research_datasets.py`
   - pulls MedMCQA, PubMedQA, and CUAD from Hugging Face
   - uploads JSON artifacts to OCI Object Storage buckets
2. `libs/oci/scripts/upload_large_datasets.py` (optional)
   - uploads larger corpora (Wikipedia, C4, ArXiv) to OCI Object Storage
3. `libs/oci/scripts/vectorize_datasets.py`
   - reads objects from buckets
   - generates embeddings
   - writes vectors into ADB table `VECTOR_DOCUMENTS`

Runtime examples then perform retrieval/synthesis over these indexed documents.
For the Object Storage example, retrieval is done through agent tools
(`list/read/search` on objects), not through SQL.

## Common Questions

1. Where does ADB data come from?
- From your ingestion pipeline. In this repo: Hugging Face -> OCI Object Storage
  -> ADB vectors using the scripts listed above.

2. Which embeddings model is used?
- Default datastore embedding model is Cohere on OCI (`cohere.embed-v4.0`).
- Examples in this repo now pass that model explicitly.

3. Is there an additional ADB search class I must implement?
- No, not for the canonical path. Use `ADB` + `create_deep_research_agent(...)`
  with `datastores=...`; datastore tools are created automatically.

4. At runtime, what do I need to provide?
- Indexed documents in a datastore, OCI/auth config, and the user prompt/query.

## Datastore Descriptions And Routing (`datastore_description=...`)

`datastore_description` is datastore metadata used for auto-routing when you provide multiple
stores.

How it works:
1. The SDK embeds each store description once at initialization (for example, "SRE runbooks, incidents").
2. For each query, it embeds the query and compares similarity with description
   embeddings using cosine similarity.
3. The best-matching store is selected for `search`/`keyword_search`.

Guidance:
- Keep descriptions short and content-focused (domain, document types, topics).
- Use distinct descriptions across stores to reduce routing ambiguity.
- With one datastore, description has no routing effect but still appears in tool/stats
  descriptions.

Examples of effective descriptions:
- "incident reports, runbooks, system diagnostics, error logs"
- "legal contracts, compliance documents, policy manuals"
- "medical research papers, clinical trials, drug information"

## What You Need

1. OCI GenAI access:
- `compartment_id`
- `service_endpoint` (or `OCI_REGION`)
- auth config (`auth_type`, `auth_profile`, credentials)

2. For datastore-backed retrieval:
- one or more `VectorDataStore` instances (`ADB`, `OpenSearch`, or custom)
- indexed documents already loaded in the datastore

3. A research prompt/query at runtime

## Embeddings

If you use datastores and do not pass `embedding_model=...`, the default is:
- `cohere.embed-v4.0` via `OCIGenAIEmbeddings`

You can override embeddings by passing a custom model:

```python
from langchain_oci import OCIGenAIEmbeddings

embedding_model = OCIGenAIEmbeddings(
    model_id="cohere.embed-v4.0",
    compartment_id="ocid1.compartment...",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    auth_type="API_KEY",
)
```

Important:
- use the same embedding model for indexing and query-time search
- ensure embedding dimension matches your vector column definition
- `ADB` uses `OracleVS` (`langchain-oracledb`) and expects OracleVS table
  schema (`id/embedding/text/metadata`).

## Example: ADB Datastore (Auto Tools)

**Prerequisites:** `pip install langchain-oci langchain-oracledb`

```python
from langchain_core.messages import HumanMessage
from langchain_oci import OCIGenAIEmbeddings
from langchain_oci import create_deep_research_agent
from langchain_oci.datastores import ADB

# Requires langchain-oracledb for Oracle Database connectivity
store = ADB(
    dsn="mydb_low",
    user="ADMIN",
    password="***",
    table_name="VECTOR_DOCUMENTS",
    datastore_description="medical QA, legal clauses, web docs",
)

embedding_model = OCIGenAIEmbeddings(
    model_id="cohere.embed-v4.0",
    compartment_id="ocid1.compartment...",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    auth_type="API_KEY",
)

agent = create_deep_research_agent(
    datastores={"research": store},
    embedding_model=embedding_model,  # explicit, same as default
    model_id="google.gemini-2.5-pro",
    compartment_id="ocid1.compartment...",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    auth_type="API_KEY",
    top_k=8,
)

result = agent.invoke(
    {"messages": [HumanMessage(content="Research leukocytosis treatment evidence")]}
)
print(result["messages"][-1].content)
```

When `datastores=...` is provided, the agent gets:
- `stats`
- `search` (semantic)
- `keyword_search`
- `get_document`

## Example: ADB Datastore + Custom Embeddings

```python
from langchain_oci import OCIGenAIEmbeddings
from langchain_oci import create_deep_research_agent
from langchain_oci.datastores import ADB

embedding_model = OCIGenAIEmbeddings(
    model_id="cohere.embed-v4.0",
    compartment_id="ocid1.compartment...",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    auth_type="API_KEY",
)

agent = create_deep_research_agent(
    datastores={"research": ADB(dsn="mydb_low", user="ADMIN", password="***")},
    embedding_model=embedding_model,
    compartment_id="ocid1.compartment...",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
)
```

## Example: Tool-Only Deep Research (No Datastores)

```python
from langchain_oci.agents import create_deep_research_agent

agent = create_deep_research_agent(
    tools=[...],  # your own LangChain tools
    model_id="google.gemini-2.5-pro",
    compartment_id="ocid1.compartment...",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
)
```

## Async Cleanup Note

If your workflow keeps agents/models alive across many async calls, explicitly
close the underlying model client when done to avoid unclosed HTTP session
warnings.
