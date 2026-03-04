# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""OCI Generative AI Agent helpers.

Agents:
    - create_oci_agent: Simple ReAct agent wrapper around LangGraph
    - create_deep_research_agent: Deep research agent with deepagents (optional)

Datastores (shared by all agents):
    - VectorDataStore: Abstract base class
    - OpenSearch: OpenSearch backend
    - ADB: Oracle Autonomous Database backend
    - create_datastore_tools: Create search tools for datastores

Example - Simple agent with datastore tools:
    >>> from langchain_oci.agents import (
    ...     create_oci_agent,
    ...     OpenSearch,
    ...     create_datastore_tools,
    ... )
    >>>
    >>> tools = create_datastore_tools(
    ...     stores={"docs": OpenSearch(endpoint="...", index_name="docs")},
    ...     compartment_id="ocid1.compartment...",
    ... )
    >>> agent = create_oci_agent(
    ...     model_id="meta.llama-3.3-70b-instruct",
    ...     tools=tools,
    ... )
"""

from typing import TYPE_CHECKING, Any

# Always available - datastores and tools
from langchain_oci.agents.datastores import (
    ADB,
    OpenSearch,
    VectorDataStore,
    create_datastore_tools,
)
from langchain_oci.agents.react.agent import create_oci_agent

if TYPE_CHECKING:
    from langchain_oci.agents.deep_research import create_deep_research_agent


def __getattr__(name: str) -> Any:
    """Lazy import for optional dependencies."""
    if name == "create_deep_research_agent":
        from langchain_oci.agents.deep_research import create_deep_research_agent

        return create_deep_research_agent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Simple ReAct agent
    "create_oci_agent",
    # Deep research agent (optional - requires deepagents)
    "create_deep_research_agent",
    # Datastores (shared)
    "VectorDataStore",
    "OpenSearch",
    "ADB",
    "create_datastore_tools",
]
