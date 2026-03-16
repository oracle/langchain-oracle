#!/usr/bin/env python
# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0
# ruff: noqa: T201

"""Deep Research Agent Demo - Agent Autonomously Uses Datastore Tools.

This is a didactic example of the standard `langchain-oci` deep-research flow:
1. Create VectorDataStore instances (OpenSearch, ADB)
2. Create deep research agent WITH datastores
3. Agent AUTONOMOUSLY calls tools to search, retrieve, and synthesize

The agent automatically has access to these tools:
- search: semantic search across datastores (auto-routes to best store)
- keyword_search: exact term matching
- get_document: retrieve full document by ID
- stats: get datastore statistics

## 1) Data source (where data comes from)

This script does not ingest documents. It reads from an existing OpenSearch index
that you provide via environment variables.

- `OPENSEARCH_INDEX` should point to a pre-populated index.
- Documents in that index are expected to represent SRE material
  (runbooks, incidents, diagnostics), as described in `datastore_description=...`.

If you need an end-to-end ingestion pipeline example, see:
- `examples/agents/deep_research_oci_storage.py`
  (reads datasets from OCI Object Storage)
- `examples/agents/deep_research_adb_datastore.py` (queries pre-ingested ADB vectors)

## 2) Embedding model used

This example explicitly passes:
- `OCIGenAIEmbeddings(model_id="cohere.embed-v4.0", ...)`

## 3) Search implementation used here

This example uses the built-in datastore tooling:
- `OpenSearch` datastore adapter
- auto-generated tools (`stats`, `search`, `keyword_search`, `get_document`)

No custom search class is implemented in this file.

## 4) What you provide at runtime

- pre-populated OpenSearch index and credentials
- OCI model/auth config
- research prompt/query

## Quick answers (review checklist)

- Data source: existing OpenSearch index (`OPENSEARCH_INDEX`), pre-populated by you.
- Embeddings: explicitly passed as `cohere.embed-v4.0` via
  `OCIGenAIEmbeddings`.
- Additional ADB search class: not used here; this example uses built-in OpenSearch
  datastore integration.
- Runtime inputs: datastore connection, OCI config, and user prompt.

## Prerequisites

1. Install dependencies:
   ```bash
   pip install langchain-oci deepagents opensearch-py
   ```

2. Set up OCI authentication (API key or session token)

## Running

```bash
cd libs/oci
python examples/agents/deep_research_agent_demo.py
```
"""

import os
import warnings

from langchain_core.messages import HumanMessage

from langchain_oci import OCIGenAIEmbeddings
from langchain_oci.agents import OpenSearch, create_deep_research_agent

# Configuration - replace with your values
COMPARTMENT_ID = os.environ["OCI_COMPARTMENT_ID"]  # Required
SERVICE_ENDPOINT = os.environ.get(
    "OCI_SERVICE_ENDPOINT",
    "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
)


def _suppress_tls_warnings() -> None:
    warnings.filterwarnings("ignore")
    try:
        import urllib3

        urllib3.disable_warnings()
    except ImportError:
        return


def main():
    """Run the deep research agent demo."""
    _suppress_tls_warnings()
    print("=" * 80)
    print("DEEP RESEARCH AGENT - AUTONOMOUS TOOL USAGE DEMO")
    print("=" * 80)

    # =========================================================================
    # STEP 1: Create Vector Datastores
    # =========================================================================
    print("\n[1/3] Initializing vector datastores...")

    # Example: OpenSearch datastore
    # Replace with your OpenSearch instance details
    sre_store = OpenSearch(
        endpoint=os.environ.get(
            "OPENSEARCH_ENDPOINT",
            "https://ai-dev.observ.us-ashburn-1.ocs.oraclecloud.com:9200",
        ),
        username=os.environ.get(
            "OPENSEARCH_USERNAME",
            os.environ.get("OPENSEARCH_USER", "ai_user"),
        ),
        password=os.environ.get("OPENSEARCH_PASSWORD", "your-password-here"),
        index_name=os.environ.get("OPENSEARCH_INDEX", "observai_diagnostic-patterns"),
        verify_certs=False,
        vector_field="vector_field",
        search_fields=["text", "metadata.title", "metadata.content"],
        # datastore_description guides query routing when
        # multiple datastores are configured
        datastore_description=(
            "SRE investigations, diagnostic patterns, error messages, runbooks"
        ),
    )

    print("   - SRE datastore (OpenSearch): ready")

    # =========================================================================
    # STEP 2: Create the Deep Research Agent
    # =========================================================================
    print("\n[2/3] Creating deep research agent...")

    # Pass the embedding model explicitly. Keep this aligned with index-time model.
    embedding_model = OCIGenAIEmbeddings(
        model_id="cohere.embed-v4.0",
        compartment_id=COMPARTMENT_ID,
        service_endpoint=SERVICE_ENDPOINT,
        auth_type="API_KEY",
        auth_profile=os.environ.get("OCI_AUTH_PROFILE", "DEFAULT"),
    )

    # The agent automatically creates tools from datastores and uses them
    agent = create_deep_research_agent(
        datastores={"sre": sre_store},
        embedding_model=embedding_model,
        model_id="google.gemini-2.5-pro",
        compartment_id=COMPARTMENT_ID,
        service_endpoint=SERVICE_ENDPOINT,
        auth_type="API_KEY",
        auth_profile=os.environ.get("OCI_AUTH_PROFILE", "DEFAULT"),
        temperature=0.4,
        max_tokens=8000,
        top_k=10,
        debug=False,  # Set True to see all tool calls
    )

    print("   Agent created with autonomous tool access")

    # =========================================================================
    # STEP 3: Let the Agent Do Research Autonomously
    # =========================================================================
    print("\n[3/3] Agent performing autonomous research...")
    print("-" * 80)

    # The agent AUTONOMOUSLY:
    # 1. Uses stats tool to understand available data
    # 2. Uses search/keyword_search to find relevant documents
    # 3. Uses get_document to retrieve full content
    # 4. Synthesizes findings into a response

    research_prompt = """You are an SRE analyst. Perform the following tasks:

1. First, use the 'stats' tool to see what data is available
2. Use the 'keyword_search' tool to find diagnostic patterns related to:
   - "connection_exhaustion"
   - "memory"
   - "timeout"
3. Use 'get_document' to retrieve full details for any interesting findings
4. Synthesize your findings into a brief technical summary (2-3 paragraphs)

Focus on what types of issues are documented and common diagnostic approaches."""

    # Invoke the agent
    result = agent.invoke({"messages": [HumanMessage(content=research_prompt)]})

    # Extract the final AI response
    messages = result.get("messages", [])
    final_content = ""
    for msg in reversed(messages):
        if hasattr(msg, "content") and msg.content:
            # Skip tool call messages
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                continue
            final_content = msg.content
            break

    print("\n" + "=" * 80)
    print("AGENT RESEARCH COMPLETE")
    print("=" * 80)
    print(f"\nFinal Response:\n{final_content}")

    # Show what tools the agent called
    print("\n" + "-" * 80)
    print("TOOLS CALLED BY AGENT:")
    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                args_str = str(tc["args"])
                if len(args_str) > 60:
                    args_str = args_str[:60] + "..."
                print(f"  - {tc['name']}({args_str})")

    return final_content


if __name__ == "__main__":
    main()
