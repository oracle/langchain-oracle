#!/usr/bin/env python
# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0
# ruff: noqa: T201

"""Deep Research Agent with ADB datastore integration.

This example shows the canonical integration path on top of langchain-oci:
1. Create an `ADB` datastore adapter
2. Pass it to `create_deep_research_agent(datastores=...)`
3. Let the agent autonomously use `stats/search/keyword_search/get_document`

## 1) Data source (where ADB data comes from)

This script expects ADB to already contain vectorized documents in
`VECTOR_DOCUMENTS` (or your table set in `ADB_TABLE_NAME`).

In this repository, a typical ingestion path is:
1. `scripts/upload_research_datasets.py` (Hugging Face -> OCI Object Storage)
2. `scripts/upload_large_datasets.py` (optional large corpora -> Object Storage)
3. `scripts/vectorize_datasets.py` (Object Storage -> ADB vectors)

## 2) Embedding model used

This example explicitly passes:
- `OCIGenAIEmbeddings(model_id="cohere.embed-v4.0", ...)`

## 3) Search implementation used here

This example uses the built-in SDK path:
- `ADB` datastore adapter
- auto-generated datastore tools (`stats`, `search`, `keyword_search`, `get_document`)

No extra ADB-specific search class is required in this approach.

## 4) What you provide at runtime

- pre-indexed ADB table data
- connection/env settings
- the research prompt/query

At runtime, this script performs retrieval and synthesis only.

## Quick answers (review checklist)

- Data source: ADB table populated by your ingestion pipeline (repo scripts provided).
- Embeddings: explicitly passed as `cohere.embed-v4.0` via
  `OCIGenAIEmbeddings`.
- Additional ADB search class: no; this uses built-in `ADB` + auto tools.
- Runtime inputs: ADB credentials/config, OCI config, and user prompt.

Required environment variables:
- OCI_COMPARTMENT_ID
- OCI_SERVICE_ENDPOINT (or OCI_REGION)
- ADB_DSN
- ADB_USER
- ADB_PASSWORD

Optional:
- ADB_WALLET_LOCATION
- ADB_WALLET_PASSWORD
- ADB_TABLE_NAME (default: VECTOR_DOCUMENTS)
- OCI_AUTH_PROFILE
- OCI_AGENT_LOG_LEVEL (default: INFO)
"""

from __future__ import annotations

import logging
import os

from langchain_core.messages import HumanMessage

from langchain_oci import OCIGenAIEmbeddings
from langchain_oci.agents import ADB, create_deep_research_agent


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def _configure_logging() -> None:
    level_name = os.environ.get("OCI_AGENT_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def main() -> None:
    _configure_logging()
    compartment_id = _required_env("OCI_COMPARTMENT_ID")
    service_endpoint = os.environ.get("OCI_SERVICE_ENDPOINT")
    if not service_endpoint:
        region = os.environ.get("OCI_REGION", "us-chicago-1")
        service_endpoint = (
            f"https://inference.generativeai.{region}.oci.oraclecloud.com"
        )
    auth_profile = os.environ.get("OCI_AUTH_PROFILE", "DEFAULT")

    adb_store = ADB(
        dsn=_required_env("ADB_DSN"),
        user=_required_env("ADB_USER"),
        password=_required_env("ADB_PASSWORD"),
        wallet_location=os.environ.get("ADB_WALLET_LOCATION"),
        wallet_password=os.environ.get("ADB_WALLET_PASSWORD"),
        table_name=os.environ.get("ADB_TABLE_NAME", "VECTOR_DOCUMENTS"),
        chunk_on_write=True,
        chunking_params={
            "split": "sentence",
            "max": 20,
            "normalize": "all",
        },
        # datastore_description guides query routing when
        # multiple datastores are configured
        datastore_description=(
            "vectorized research documents. Contains title/content/source/embedding "
            "for semantic and keyword retrieval."
        ),
    )

    # Pass the embedding model explicitly. Keep this aligned with index-time model.
    embedding_model = OCIGenAIEmbeddings(
        model_id="cohere.embed-v4.0",
        compartment_id=compartment_id,
        service_endpoint=service_endpoint,
        auth_type="API_KEY",
        auth_profile=auth_profile,
    )

    agent = create_deep_research_agent(
        datastores={"research": adb_store},
        embedding_model=embedding_model,
        model_id="google.gemini-2.5-pro",
        compartment_id=compartment_id,
        service_endpoint=service_endpoint,
        auth_type="API_KEY",
        auth_profile=auth_profile,
        top_k=8,
        system_prompt=(
            "You are a research analyst. Use datastore tools to gather evidence "
            "from documents, cite Doc IDs, and provide a concise synthesis."
        ),
    )

    query = (
        "Use stats first, then find the most relevant documents about "
        "leukocytosis diagnosis and treatment, and summarize key findings."
    )
    result = agent.invoke({"messages": [HumanMessage(content=query)]})
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
