# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Deepagents Agent - deepagents-based research agent with OCI GenAI.

Detailed usage and additional examples:
`langchain_oci/agents/deepagents/README.md`

Example:
    >>> from langchain_oci.agents.deepagents import create_deepagents_agent
    >>> from langchain_oci.datastores import OpenSearch, ADB
    >>>
    >>> agent = create_deepagents_agent(
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

from langchain_oci.agents.deepagents.agent import create_deepagents_agent

__all__ = [
    "create_deepagents_agent",
]
