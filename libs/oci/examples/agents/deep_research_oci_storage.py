#!/usr/bin/env python
# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0
# ruff: noqa: T201, I001
# mypy: disable-error-code=attr-defined

"""Deep Research Agent with OCI Object Storage.

This example demonstrates using the deep research agent with OCI Object Storage
tools to analyze medical and legal datasets stored in OCI buckets.

## 1) Data source (where data comes from)

This example reads JSON datasets from OCI Object Storage buckets.

In this repository, those datasets are created by:
1. `scripts/upload_research_datasets.py`:
   - downloads MedMCQA, PubMedQA, and CUAD from Hugging Face
   - uploads them to `deep-research-medical` and `deep-research-legal`
2. `scripts/upload_large_datasets.py` (optional):
   - uploads large corpora (Wikipedia, C4, ArXiv) to `deep-research-large`

## 2) Embedding model used

This example does not do vector retrieval, so it does not need query embeddings.
The LLM model is used for reasoning/synthesis over retrieved object data.

## 3) Search implementation used here

This file uses Object Storage tools (`list/read/search` over bucket objects),
not ADB/OpenSearch vector search.

## 4) What you provide at runtime

- bucket/namespace configuration and OCI credentials
- the research prompt/query

This script does not write embeddings or populate ADB.

## Quick answers (review checklist)

- Data source: OCI Object Storage buckets filled by repository upload scripts.
- Embeddings: not applicable in this example (no vector datastore retrieval path).
- Additional ADB search class: no; this example uses Object Storage tools only.
- Runtime inputs: OCI storage config and user prompt.

## Prerequisites

1. Set up OCI credentials:
   ```bash
   export OCI_COMPARTMENT_ID="ocid1.compartment..."
   export OCI_REGION="us-chicago-1"  # For GenAI
   ```

2. Install dependencies:
   ```bash
   pip install langchain-oci deepagents
   ```

3. Ensure datasets are uploaded to OCI Object Storage:
   ```bash
   python scripts/upload_research_datasets.py
   ```

## Running

```bash
python examples/agents/deep_research_oci_storage.py
```
"""

import asyncio
import os
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from oci.config import from_file
from oci.exceptions import ServiceError
from oci.object_storage import ObjectStorageClient

from langchain_oci.agents import create_deep_research_agent

# Configuration
COMPARTMENT_ID = os.environ["OCI_COMPARTMENT_ID"]  # Required
GENAI_REGION = os.environ.get("OCI_REGION", "us-chicago-1")
STORAGE_REGION = "us-ashburn-1"
NAMESPACE = os.environ["OCI_NAMESPACE"]  # Required - your OCI Object Storage namespace
AUTH_PROFILE = os.environ.get("OCI_AUTH_PROFILE", "DEFAULT")

# Bucket names
MEDICAL_BUCKET = "deep-research-medical"
LEGAL_BUCKET = "deep-research-legal"
LARGE_BUCKET = "deep-research-large"  # Wikipedia, C4, ArXiv


def _format_storage_error(error: ServiceError) -> str:
    """Create a concise, operator-friendly OCI Object Storage error message."""
    code = getattr(error, "code", "UnknownError")
    status = getattr(error, "status", "unknown")
    message = str(getattr(error, "message", "")).strip()
    if code == "ObjectNotFound":
        return (
            "Object not found in bucket. Verify object name and prefix with "
            "list_bucket_objects before reading."
        )
    if code == "BucketNotFound":
        return "Bucket not found. Verify bucket name and OCI namespace/region."
    return f"OCI Object Storage error ({status}/{code}): {message}"


def create_oci_object_storage_tools(
    *,
    namespace: str,
    buckets: list[str],
    region: str,
    auth_profile: str,
    auth_file_location: str = "~/.oci/config",
) -> list[Any]:
    """Create simple OCI Object Storage tools for Deep Research."""
    config = from_file(
        file_location=str(Path(auth_file_location).expanduser()),
        profile_name=auth_profile,
    )
    config["region"] = region
    client = ObjectStorageClient(config)
    allowed_buckets = set(buckets)

    @tool
    def list_bucket_objects(bucket: str, prefix: str = "", limit: int = 50) -> str:
        """List object names in a bucket, optionally filtered by prefix."""
        if bucket not in allowed_buckets:
            return f"Bucket '{bucket}' not allowed. Allowed: {sorted(allowed_buckets)}"
        try:
            response = client.list_objects(
                namespace_name=namespace,
                bucket_name=bucket,
                prefix=prefix or None,
                limit=min(max(limit, 1), 100),
            )
        except ServiceError as error:
            return _format_storage_error(error)
        objects = response.data.objects or []
        if not objects:
            return f"No objects found in bucket '{bucket}' (prefix='{prefix}')."
        lines = [f"{obj.name} ({obj.size} bytes)" for obj in objects]
        return "\n".join(lines)

    @tool
    def read_bucket_object(bucket: str, object_name: str, max_chars: int = 8000) -> str:
        """Read and return text content from a bucket object."""
        if bucket not in allowed_buckets:
            return f"Bucket '{bucket}' not allowed. Allowed: {sorted(allowed_buckets)}"
        try:
            response = client.get_object(
                namespace_name=namespace,
                bucket_name=bucket,
                object_name=object_name,
            )
        except ServiceError as error:
            return _format_storage_error(error)
        content = response.data.content.decode("utf-8", errors="ignore")
        if len(content) > max_chars:
            return content[:max_chars] + "\n\n[truncated]"
        return content

    @tool
    def search_bucket_data(
        query: str,
        bucket: str = "",
        prefix: str = "",
        max_objects: int = 20,
    ) -> str:
        """Search text objects for a query and return snippets."""
        selected = [bucket] if bucket else sorted(allowed_buckets)
        results: list[str] = []
        q = query.lower()
        for bkt in selected:
            if bkt not in allowed_buckets:
                continue
            try:
                listing = client.list_objects(
                    namespace_name=namespace,
                    bucket_name=bkt,
                    prefix=prefix or None,
                    limit=min(max(max_objects, 1), 50),
                )
            except ServiceError:
                continue
            for obj in listing.data.objects or []:
                name = obj.name or ""
                if not name.lower().endswith((".json", ".txt", ".md")):
                    continue
                try:
                    payload = client.get_object(
                        namespace_name=namespace,
                        bucket_name=bkt,
                        object_name=name,
                    )
                    text = payload.data.content.decode("utf-8", errors="ignore")
                except ServiceError:
                    continue
                idx = text.lower().find(q)
                if idx < 0:
                    continue
                start = max(0, idx - 200)
                end = min(len(text), idx + len(query) + 200)
                snippet = text[start:end].replace("\n", " ")
                results.append(f"[{bkt}/{name}] ...{snippet}...")
                if len(results) >= 10:
                    break
            if len(results) >= 10:
                break
        if not results:
            return f"No matches for '{query}' in selected bucket scope."
        return "\n\n".join(results)

    return [list_bucket_objects, read_bucket_object, search_bucket_data]


def main():
    """Run deep research agent with OCI Object Storage."""
    print("=" * 60)
    print("Deep Research Agent with OCI Object Storage")
    print("=" * 60)

    # Create OCI Object Storage tools
    print("\nCreating OCI Object Storage tools...")
    storage_tools = create_oci_object_storage_tools(
        namespace=NAMESPACE,
        buckets=[MEDICAL_BUCKET, LEGAL_BUCKET, LARGE_BUCKET],
        region=STORAGE_REGION,
        auth_profile=AUTH_PROFILE,
    )
    print(f"  Created {len(storage_tools)} tools:")
    for storage_tool in storage_tools:
        print(f"    - {storage_tool.name}: {storage_tool.description[:60]}...")

    # Create deep research agent
    print("\nCreating deep research agent...")
    service_endpoint = (
        f"https://inference.generativeai.{GENAI_REGION}.oci.oraclecloud.com"
    )

    agent = create_deep_research_agent(
        tools=storage_tools,
        model_id="meta.llama-4-scout-17b-16e-instruct",
        compartment_id=COMPARTMENT_ID,
        service_endpoint=service_endpoint,
        auth_type="API_KEY",
        auth_profile=AUTH_PROFILE,
        system_prompt=(
            "You are a deep research analyst with access to massive datasets "
            "stored in OCI Object Storage. "
            "Available buckets:\n"
            f"- {MEDICAL_BUCKET}: MedMCQA (medical questions) and "
            "PubMedQA (biomedical research)\n"
            f"- {LEGAL_BUCKET}: CUAD (contract clauses)\n"
            f"- {LARGE_BUCKET}: Wikipedia (1M+ articles), C4 web corpus (1M+), "
            "ArXiv papers (100K+)\n\n"
            "Use the tools to:\n"
            "1. list_bucket_objects: Discover available data files\n"
            "2. read_bucket_object: Read specific JSON files\n"
            "3. search_bucket_data: Search for specific terms\n\n"
            "Always cite your sources and provide thorough analysis."
        ),
        temperature=0.3,
        max_tokens=2048,
    )
    print("  Agent created successfully!")

    # Example research queries
    queries = [
        "What pathology questions are available in the medical dataset? "
        "List a few examples and explain the topics they cover.",
        "Search the medical bucket for questions about 'leukocytosis'. "
        "What information is available?",
        "What types of contract clauses are in the legal dataset? "
        "Give examples of termination or indemnification clauses.",
    ]

    try:
        for i, query in enumerate(queries, 1):
            print(f"\n{'=' * 60}")
            print(f"Research Query {i}:")
            print(f"{'=' * 60}")
            print(f"\n{query}\n")

            try:
                result = agent.invoke({"messages": [HumanMessage(content=query)]})

                # Extract response
                final_message = result["messages"][-1]
                print("\n--- Agent Response ---")
                print(final_message.content)

                # Show tool usage
                tool_messages = [
                    m for m in result["messages"] if type(m).__name__ == "ToolMessage"
                ]
                if tool_messages:
                    print(f"\n(Used {len(tool_messages)} tool calls)")

            except Exception as e:
                print(f"Error: {type(e).__name__}: {e}")

            print()
    finally:
        llm = getattr(agent, "_oci_llm", None)
        if llm is not None and hasattr(llm, "aclose"):
            asyncio.run(llm.aclose())


if __name__ == "__main__":
    main()
