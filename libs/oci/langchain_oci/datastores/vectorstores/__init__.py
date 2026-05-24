# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Vector store implementations for OCI agents.

Available backends:
- OpenSearch: OpenSearch/Elasticsearch vector search
- ADB: Oracle Autonomous Database with vector support

Example:
    >>> from langchain_oci.datastores import OpenSearch, ADB, create_datastore_tools
    >>>
    >>> tools = create_datastore_tools(
    ...     stores={
    ...         "docs": OpenSearch(endpoint="...", index_name="docs"),
    ...         "sales": ADB(dsn="...", user="...", password="..."),
    ...     },
    ...     compartment_id="ocid1.compartment...",
    ... )
"""

from langchain_oci.datastores.vectorstores.adb import ADB
from langchain_oci.datastores.vectorstores.base import VectorDataStore
from langchain_oci.datastores.vectorstores.opensearch import OpenSearch

__all__ = [
    "VectorDataStore",
    "ADB",
    "OpenSearch",
]
