# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""OCI Deep Research Agent - deepagents-based research agent with OCI GenAI."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Sequence

from langchain_core.tools import BaseTool

from langchain_oci.agents.common import OCIConfig, filter_none, merge_model_kwargs
from langchain_oci.agents.datastores import VectorDataStore, create_datastore_tools
from langchain_oci.chat_models.oci_generative_ai import ChatOCIGenAI
from langchain_oci.common.auth import OCIAuthType

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


def create_deep_research_agent(
    tools: Sequence[BaseTool | Callable[..., Any]] | None = None,
    *,
    # Datastores - if provided, auto-routing search is enabled
    datastores: dict[str, VectorDataStore] | None = None,
    default_datastore: str | None = None,
    default_store: str | None = None,  # Alias for default_datastore
    embedding_model: Any = None,
    top_k: int = 5,
    # OCI-specific options
    model_id: str = "google.gemini-2.5-pro",
    compartment_id: str | None = None,
    service_endpoint: str | None = None,
    auth_type: str | OCIAuthType = OCIAuthType.API_KEY,
    auth_profile: str = "DEFAULT",
    auth_file_location: str = "~/.oci/config",
    # Deep agent options
    system_prompt: str | None = None,
    subagents: list[Any] | None = None,
    skills: list[str] | None = None,
    memory: list[str] | None = None,
    middleware: Sequence[Any] | None = None,
    # LangGraph options
    checkpointer: Any = None,
    store: Any = None,
    interrupt_before: list[str] | None = None,
    interrupt_after: list[str] | None = None,
    debug: bool = False,
    name: str | None = None,
    # Model kwargs
    temperature: float | None = None,
    max_tokens: int | None = None,
    max_input_tokens: int | None = None,  # noqa: ARG001 - Intentionally ignored
    **model_kwargs: Any,
) -> "CompiledStateGraph":
    """Create a Deep Research Agent using OCI GenAI and deepagents.

    This agent is designed for multi-step research tasks that require:
    - Searching multiple data sources (OpenSearch, ADB)
    - Planning and reflection
    - Synthesizing information into reports

    Args:
        tools: Custom tools for the agent.
        datastores: Dict of vector datastores for auto-routing search.
        default_datastore: Fallback datastore if routing is inconclusive.
        default_store: Alias for default_datastore.
        embedding_model: Custom embedding model for datastores.
        top_k: Number of search results to return.
        model_id: OCI model identifier (Gemini models recommended).
        compartment_id: OCI compartment OCID.
        service_endpoint: OCI GenAI service endpoint.
        auth_type: OCI authentication type.
        auth_profile: OCI config profile name.
        auth_file_location: Path to OCI config file.
        system_prompt: Custom system prompt for the agent.
        subagents: List of subagents for delegation.
        skills: List of skill names to enable.
        memory: List of memory namespaces.
        middleware: Custom middleware. Pass empty list to disable defaults.
        checkpointer: LangGraph checkpointer for persistence/memory.
        store: LangGraph store for long-term memory.
        interrupt_before: Tools to pause before for human-in-loop.
        interrupt_after: Tools to pause after for human-in-loop.
        debug: Enable debug mode.
        name: Name for the agent.
        temperature: Model temperature.
        max_tokens: Maximum output tokens (e.g., 65536 for Gemini 2.5 Pro).
        max_input_tokens: Ignored. Input limits are model-determined.
        **model_kwargs: Additional model kwargs.

    Returns:
        CompiledStateGraph: A compiled deep research agent.

    Example:
        >>> from langchain_oci.agents.deep_research import OpenSearch, ADB
        >>>
        >>> agent = create_deep_research_agent(
        ...     datastores={
        ...         "docs": OpenSearch(
        ...             endpoint="https://opensearch:9200",
        ...             index_name="company-docs",
        ...             datastore_description="internal documentation, policies",
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
    """
    try:
        from deepagents import create_deep_agent
    except ImportError as ex:
        raise ImportError(
            "deepagents required. Install with: pip install deepagents"
        ) from ex

    # Resolve OCI configuration
    oci_config = OCIConfig.resolve(
        compartment_id=compartment_id,
        service_endpoint=service_endpoint,
        auth_type=auth_type,
        auth_profile=auth_profile,
        auth_file_location=auth_file_location,
    )

    # Build tools list
    all_tools: list[BaseTool | Callable[..., Any]] = []

    if datastores:
        datastore_tools = create_datastore_tools(
            stores=datastores,
            default_store=default_store or default_datastore,
            embedding_model=embedding_model,
            compartment_id=oci_config.compartment_id,
            service_endpoint=oci_config.service_endpoint,
            auth_type=oci_config.auth_type,
            auth_profile=oci_config.auth_profile,
            top_k=top_k,
        )
        all_tools.extend(datastore_tools)

    if tools:
        all_tools.extend(tools)

    # Create OCI chat model
    llm = ChatOCIGenAI(
        model_id=model_id,
        compartment_id=oci_config.compartment_id,
        service_endpoint=oci_config.service_endpoint,
        auth_type=oci_config.auth_type,
        auth_profile=oci_config.auth_profile,
        auth_file_location=oci_config.auth_file_location,
        model_kwargs=merge_model_kwargs(
            model_kwargs,
            temperature,
            max_tokens,
            model_id=model_id,
        ),
    )

    # Build agent kwargs - only include non-None values
    agent_kwargs = {
        "model": llm,
        "tools": all_tools,
        **filter_none(
            system_prompt=system_prompt,
            subagents=subagents,
            skills=skills,
            memory=memory,
            middleware=middleware,
            checkpointer=checkpointer,
            store=store,
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after,
            name=name,
        ),
    }

    # debug=False is meaningful, so handle separately
    if debug:
        agent_kwargs["debug"] = True

    compiled = create_deep_agent(**agent_kwargs)
    # Expose the underlying OCI chat model for explicit cleanup in long-lived
    # processes (and in our integration tests). This avoids aiohttp
    # "Unclosed client session" warnings when async pooling is used.
    setattr(compiled, "_oci_llm", llm)
    return compiled
