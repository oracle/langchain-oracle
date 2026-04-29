# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Datastore tools for agents.

This module provides LangChain tools for searching and retrieving documents
from vector datastores (OpenSearch, Oracle ADB). Tools automatically route
queries to the best datastore based on semantic similarity.

Example:
    >>> from langchain_oci.datastores import OpenSearch, create_datastore_tools
    >>> tools = create_datastore_tools(
    ...     stores={
    ...         "docs": OpenSearch(
    ...             endpoint="...", datastore_description="documentation"
    ...         )
    ...     },
    ...     compartment_id="ocid1.compartment...",
    ... )
"""

# Base classes and types
from langchain_oci.datastores.tools.base import (
    DatastoreTool,
    ResultFormatter,
    SearchResult,
    StoreStats,
)

# Factory function
from langchain_oci.datastores.tools.factory import create_datastore_tools

# Tool implementations
from langchain_oci.datastores.tools.get_document import GetDocumentTool
from langchain_oci.datastores.tools.keyword_search import KeywordSearchTool

# Input schemas
from langchain_oci.datastores.tools.schemas import (
    GetDocumentInput,
    SearchInput,
    StatsInput,
)
from langchain_oci.datastores.tools.search import SearchTool

# Selector
from langchain_oci.datastores.tools.selector import StoreSelector
from langchain_oci.datastores.tools.stats import StatsTool

__all__ = [
    # Base classes and types
    "DatastoreTool",
    "ResultFormatter",
    "SearchResult",
    "StoreStats",
    # Selector
    "StoreSelector",
    # Input schemas
    "SearchInput",
    "GetDocumentInput",
    "StatsInput",
    # Tool implementations
    "SearchTool",
    "KeywordSearchTool",
    "GetDocumentTool",
    "StatsTool",
    # Factory
    "create_datastore_tools",
]
