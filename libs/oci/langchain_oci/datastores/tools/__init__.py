# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Datastore tools for agents.

This module provides LangChain tools for searching and retrieving documents
from vector datastores (OpenSearch, Oracle ADB). Tools automatically route
queries to the best datastore based on semantic similarity.
"""

# Base classes and types
from langchain_oci.datastores.tools.base import (
    DatastoreTool,
    ResultFormatter,
    SearchResult,
    StoreStats,
)
from langchain_oci.datastores.tools.keyword_search import KeywordSearchTool

# Input schemas
from langchain_oci.datastores.tools.schemas import (
    GetDocumentInput,
    SearchInput,
    StatsInput,
)
from langchain_oci.datastores.tools.search import SearchTool

__all__ = [
    # Base classes and types
    "DatastoreTool",
    "ResultFormatter",
    "SearchResult",
    "StoreStats",
    # Input schemas
    "SearchInput",
    "GetDocumentInput",
    "StatsInput",
    # Tool implementations
    "SearchTool",
    "KeywordSearchTool",
]
