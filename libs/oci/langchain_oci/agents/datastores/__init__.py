# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Datastores for OCI agents.

This module provides pluggable datastore backends and tools for search.
Use with any agent: create_oci_agent, OCIGenAIAgent, or create_deep_research_agent.

Submodules:
- vectorstores: Vector similarity search backends (OpenSearch, ADB)
- tools: LangChain tools for datastore operations

Example:
    >>> from langchain_oci.agents import (
    ...     OpenSearch,
    ...     ADB,
    ...     create_datastore_tools,
    ...     create_oci_agent,
    ... )
    >>>
    >>> tools = create_datastore_tools(
    ...     stores={
    ...         "docs": OpenSearch(endpoint="...", index_name="docs"),
    ...         "sales": ADB(dsn="...", user="...", password="..."),
    ...     },
    ...     compartment_id="ocid1.compartment...",
    ... )
    >>>
    >>> agent = create_oci_agent(
    ...     model_id="meta.llama-3.3-70b-instruct",
    ...     tools=tools,
    ... )
"""

# Vector stores
# Tools
from langchain_oci.agents.datastores.tools import (
    GetDocumentTool,
    KeywordSearchTool,
    ResultFormatter,
    SearchResult,
    SearchTool,
    StatsTool,
    StoreSelector,
    StoreStats,
    create_datastore_tools,
)
from langchain_oci.agents.datastores.vectorstores import (
    ADB,
    OpenSearch,
    VectorDataStore,
)

__all__ = [
    # Vector stores
    "VectorDataStore",
    "ADB",
    "OpenSearch",
    # Tools - factory
    "create_datastore_tools",
    # Tools - classes
    "SearchTool",
    "KeywordSearchTool",
    "GetDocumentTool",
    "StatsTool",
    # Tool internals
    "StoreSelector",
    "ResultFormatter",
    "SearchResult",
    "StoreStats",
]
