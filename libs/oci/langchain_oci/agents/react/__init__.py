# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""OCI ReAct Agent - simple wrapper around LangGraph's ReAct pattern.

Example:
    >>> from langchain_oci.agents import create_oci_agent
    >>> from langchain_oci.datastores import OpenSearch, create_datastore_tools
    >>>
    >>> # With custom tools
    >>> agent = create_oci_agent(
    ...     model_id="meta.llama-3.3-70b-instruct",
    ...     tools=[my_tool],
    ...     compartment_id="ocid1.compartment...",
    ... )
    >>>
    >>> # With datastore tools
    >>> tools = create_datastore_tools(
    ...     stores={"docs": OpenSearch(endpoint="...", index_name="docs")},
    ...     compartment_id="ocid1.compartment...",
    ... )
    >>> agent = create_oci_agent(
    ...     model_id="meta.llama-3.3-70b-instruct",
    ...     tools=tools,
    ... )
    >>>
    >>> result = agent.invoke(
    ...     {"messages": [{"role": "user", "content": "Search for policies"}]}
    ... )
"""

from langchain_oci.agents.react.agent import create_oci_agent

__all__ = ["create_oci_agent"]
