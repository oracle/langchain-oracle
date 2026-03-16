# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Deep Research Agent - deepagents-based research agent with OCI GenAI.

Detailed usage and additional examples:
`langchain_oci/agents/deep_research/README.md`

Example:
    >>> from langchain_oci.agents.deep_research import create_deep_research_agent
    >>> from langchain_oci.agents import OpenSearch, ADB
    >>>
    >>> agent = create_deep_research_agent(
    ...     datastores={
    ...         "hr": OpenSearch(
    ...             endpoint="https://opensearch:9200",
    ...             index_name="hr-docs",
    ...             datastore_description="HR policies, PTO, vacation, benefits",
    ...         ),
    ...         "sales": ADB(
    ...             dsn="mydb_low",
    ...             user="ADMIN",
    ...             password="...",
    ...             datastore_description="sales data, revenue, customers",
    ...         ),
    ...     },
    ...     compartment_id="ocid1.compartment...",
    ... )
    >>>
    >>> result = agent.invoke(
    ...     {"messages": [{"role": "user", "content": "Research Q4 sales trends"}]}
    ... )
"""

# Re-export datastores from shared location for convenience
from langchain_oci.agents.datastores import (
    ADB,
    OpenSearch,
    VectorDataStore,
    create_datastore_tools,
)
from langchain_oci.agents.deep_research.agent import create_deep_research_agent

__all__ = [
    # Main entry point
    "create_deep_research_agent",
    # Re-exports from shared datastores
    "VectorDataStore",
    "OpenSearch",
    "ADB",
    "create_datastore_tools",
]
