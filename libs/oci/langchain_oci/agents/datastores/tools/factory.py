# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Factory function for creating datastore tools."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Optional

from langchain_core.tools import BaseTool

from langchain_oci.agents.datastores.tools.base import ResultFormatter
from langchain_oci.agents.datastores.tools.get_document import GetDocumentTool
from langchain_oci.agents.datastores.tools.keyword_search import KeywordSearchTool
from langchain_oci.agents.datastores.tools.search import SearchTool
from langchain_oci.agents.datastores.tools.selector import StoreSelector
from langchain_oci.agents.datastores.tools.stats import StatsTool

if TYPE_CHECKING:
    from langchain_oci.agents.datastores.vectorstores import VectorDataStore


def create_datastore_tools(
    stores: dict[str, VectorDataStore],
    *,
    default_store: Optional[str] = None,
    embedding_model: Optional[Any] = None,
    compartment_id: Optional[str] = None,
    service_endpoint: Optional[str] = None,
    auth_type: str = "API_KEY",
    auth_profile: str = "DEFAULT",
    top_k: int = 5,
) -> list[BaseTool]:
    """Create datastore tools with automatic store routing.

    Creates a set of tools for searching and retrieving documents from
    vector datastores. Tools automatically route queries to the most
    relevant datastore based on semantic similarity to store descriptions.

    Args:
        stores: Dict mapping names to VectorDataStore instances.
            Each store should have a `datastore_description` describing its contents.
        default_store: Fallback store name. Defaults to first store.
        embedding_model: LangChain embedding model for routing and search.
            Defaults to OCI GenAI Cohere embeddings.
        compartment_id: OCI compartment OCID (required if using default model).
        service_endpoint: OCI GenAI service endpoint.
        auth_type: OCI auth type (API_KEY, SECURITY_TOKEN, etc.).
        auth_profile: OCI config profile name.
        top_k: Number of results to return from searches.

    Returns:
        List of tools: [stats, search, keyword_search, get_document]

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
        ...         "docs": OpenSearch(
        ...             endpoint="https://opensearch:9200",
        ...             index_name="company-docs",
        ...             datastore_description=(
        ...                 "internal documentation, engineering policies"
        ...             ),
        ...         ),
        ...         "medical": ADB(
        ...             dsn="research_low",
        ...             user="ADMIN",
        ...             password="...",
        ...             datastore_description=(
        ...                 "medical literature, clinical research papers"
        ...             ),
        ...         ),
        ...     },
        ...     compartment_id="ocid1.compartment...",
        ... )
        >>>
        >>> agent = create_oci_agent(tools=tools, ...)
    """
    if not stores:
        raise ValueError("At least one datastore is required")

    # Resolve default store
    if default_store is None:
        default_store = next(iter(stores.keys()))
    if default_store not in stores:
        raise ValueError(
            f"Default store '{default_store}' not found. "
            f"Available: {list(stores.keys())}"
        )

    # Set up embedding model
    if embedding_model is None:
        embedding_model = _create_default_embedding_model(
            compartment_id=compartment_id,
            service_endpoint=service_endpoint,
            auth_type=auth_type,
            auth_profile=auth_profile,
        )

    # Connect all stores
    for store in stores.values():
        store.connect(embedding_model)

    # Create selector and formatter
    selector = StoreSelector(
        stores=stores,
        embedding_model=embedding_model,
        default_store=default_store,
    )
    formatter = ResultFormatter()

    # Build store list for descriptions
    store_list = ", ".join(
        f"{name} ({s.datastore_description})" if s.datastore_description else name
        for name, s in stores.items()
    )

    # Create tools with injected dependencies
    common_kwargs: dict[str, Any] = {
        "selector": selector,
        "formatter": formatter,
        "store_list": store_list,
    }

    def build_description(base: str, usage: str = "") -> str:
        """Build full description with store list."""
        parts = [base]
        if usage:
            parts.append(usage)
        parts.append(f"Available stores: {store_list}")
        return " ".join(parts)

    return [
        StatsTool(
            description=build_description(StatsTool.base_description),
            **common_kwargs,
        ),
        SearchTool(
            description=build_description(
                SearchTool.base_description,
                SearchTool.usage_hint,
            ),
            top_k=top_k,
            **common_kwargs,
        ),
        KeywordSearchTool(
            description=build_description(
                KeywordSearchTool.base_description,
                KeywordSearchTool.usage_hint,
            ),
            top_k=top_k,
            **common_kwargs,
        ),
        GetDocumentTool(
            description=build_description(
                GetDocumentTool.base_description,
                GetDocumentTool.usage_hint,
            ),
            **common_kwargs,
        ),
    ]


def _create_default_embedding_model(
    compartment_id: Optional[str],
    service_endpoint: Optional[str],
    auth_type: str,
    auth_profile: str,
) -> Any:
    """Create the default OCI GenAI embedding model."""
    from langchain_oci import OCIGenAIEmbeddings

    if compartment_id is None:
        compartment_id = os.environ.get("OCI_COMPARTMENT_ID")
        if compartment_id is None:
            raise ValueError(
                "compartment_id is required for the default embedding model. "
                "Provide it directly or set OCI_COMPARTMENT_ID environment variable."
            )

    if service_endpoint is None:
        region = os.environ.get("OCI_REGION", "us-chicago-1")
        service_endpoint = (
            f"https://inference.generativeai.{region}.oci.oraclecloud.com"
        )

    # Tenancies differ in which models are enabled. Prefer env configuration
    # and keep a conservative fallback for local/dev.
    model_id = os.environ.get("OCI_EMBEDDING_MODEL_ID") or os.environ.get(
        "OCI_EMBEDDING_MODEL"
    )

    return OCIGenAIEmbeddings(
        model_id=model_id or "cohere.embed-v4.0",
        compartment_id=compartment_id,
        service_endpoint=service_endpoint,
        auth_type=auth_type,
        auth_profile=auth_profile,
    )
