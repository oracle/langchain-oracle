# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""OCI Deep Research Agent - deepagents-based research agent with OCI GenAI."""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Union,
)

from langchain_core.tools import BaseTool
from pydantic import Field

from langchain_oci.agents.common import (
    AgentConfig,
    _build_llm,
    _filter_none,
    _get_agent_factory,
    _langgraph_schema_fallback,
)
from langchain_oci.common.auth import OCIAuthType
from langchain_oci.datastores import VectorDataStore, create_datastore_tools

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


class DeepResearchConfig(AgentConfig):
    """Configuration for the Deep Research agent.

    Extends AgentConfig with datastore routing and deepagents options.
    Use Pydantic ``Field(alias=...)`` for alternative field names.
    """

    model_id: str = "google.gemini-2.5-pro"

    # Datastores (typed as Any to avoid Pydantic strict validation on ABC)
    datastores: Optional[Dict[str, Any]] = None
    default_datastore: Optional[str] = Field(default=None, alias="default_store")
    embedding_model: Optional[Any] = None
    top_k: int = 5

    # Deep agent options
    subagents: Optional[List[Any]] = None
    skills: Optional[List[str]] = None
    memory: Optional[List[str]] = None
    middleware: Optional[Sequence[Any]] = None
    response_format: Optional[Any] = None
    context_schema: Optional[type] = None

    # LangGraph options
    backend: Optional[Any] = None
    cache: Optional[Any] = None
    interrupt_on: Optional[Dict[str, Any]] = None

    # Tools
    tools: Optional[Sequence[Union[BaseTool, Callable[..., Any]]]] = None

    @property
    def _needs_deep_agent(self) -> bool:
        """Whether this config requires the full deep agent path."""
        if self.subagents or self.skills or self.memory:
            return True
        if self.backend or self.cache or self.interrupt_on:
            return True
        if self.response_format or self.context_schema:
            return True
        return False

    @property
    def _can_use_lightweight(self) -> bool:
        """Whether a lightweight ReAct agent suffices."""
        if self._needs_deep_agent:
            return False
        if self.datastores:
            return True
        return self.middleware is not None and len(self.middleware) == 0


def _check_deep_research_prerequisites() -> None:
    """Validate runtime prerequisites for deep research."""
    import sys

    min_version = (3, 11)
    if sys.version_info < min_version:
        msg = (
            "Deep Research requires Python 3.11 or later. "
            f"Current version: {sys.version.split()[0]}. "
            "The deepagents package and its middleware are not compatible "
            "with earlier Python versions."
        )
        raise RuntimeError(msg)

    try:
        import deepagents  # noqa: F401
    except ImportError:
        raise ImportError(
            "Deep Research requires the 'deepagents' package. "
            "Install it with: pip install 'langchain-oci[deep-research]' "
            "or: pip install deepagents"
        ) from None


def create_deep_research_agent(
    tools: Optional[Sequence[Union[BaseTool, Callable[..., Any]]]] = None,
    *,
    # Datastores
    datastores: Optional[Dict[str, VectorDataStore]] = None,
    default_datastore: Optional[str] = None,
    default_store: Optional[str] = None,
    embedding_model: Any = None,
    top_k: int = 5,
    # OCI options
    model_id: str = "google.gemini-2.5-pro",
    compartment_id: Optional[str] = None,
    service_endpoint: Optional[str] = None,
    auth_type: str | OCIAuthType = OCIAuthType.API_KEY,
    auth_profile: str = "DEFAULT",
    auth_file_location: str = "~/.oci/config",
    # Deep agent options
    system_prompt: Optional[str] = None,
    subagents: Optional[List[Any]] = None,
    skills: Optional[List[str]] = None,
    memory: Optional[List[str]] = None,
    middleware: Optional[Sequence[Any]] = None,
    response_format: Any = None,
    context_schema: Optional[type] = None,
    # LangGraph options
    checkpointer: Any = None,
    store: Any = None,
    backend: Any = None,
    cache: Any = None,
    interrupt_before: Optional[List[str]] = None,
    interrupt_after: Optional[List[str]] = None,
    interrupt_on: Optional[Dict[str, Any]] = None,
    debug: bool = False,
    name: Optional[str] = None,
    # Model kwargs
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    max_input_tokens: Optional[int] = None,  # noqa: ARG001
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
        response_format: Structured output response format for the agent.
        context_schema: Schema for typed context passed into the agent graph.
        checkpointer: LangGraph checkpointer for persistence/memory.
        store: LangGraph store for long-term memory.
        backend: State backend for the deep agent (e.g., StoreBackend).
        cache: LangGraph cache for caching LLM calls.
        interrupt_before: Node names to interrupt before (lightweight path).
        interrupt_after: Node names to interrupt after (lightweight path).
        interrupt_on: Mapping of tool names to interrupt configs.
        debug: Enable debug mode.
        name: Name for the agent.
        temperature: Model temperature.
        max_tokens: Maximum output tokens (e.g., 65536 for Gemini 2.5 Pro).
        max_input_tokens: Ignored. Input limits are model-determined.
        **model_kwargs: Additional model kwargs.

    Returns:
        CompiledStateGraph: A compiled deep research agent.

    Example:
        >>> from langchain_oci.datastores import OpenSearch, ADB
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
    _check_deep_research_prerequisites()

    config = DeepResearchConfig(
        model_id=model_id,
        compartment_id=compartment_id,
        service_endpoint=service_endpoint,
        auth_type=auth_type,
        auth_profile=auth_profile,
        auth_file_location=auth_file_location,
        system_prompt=system_prompt,
        checkpointer=checkpointer,
        store=store,
        interrupt_before=interrupt_before,
        interrupt_after=interrupt_after,
        debug=debug,
        name=name,
        temperature=temperature,
        max_tokens=max_tokens,
        model_kwargs=model_kwargs,
        # Deep research specific
        datastores=datastores,
        default_datastore=default_store or default_datastore,
        embedding_model=embedding_model,
        top_k=top_k,
        tools=tools,
        subagents=subagents,
        skills=skills,
        memory=memory,
        middleware=middleware,
        response_format=response_format,
        context_schema=context_schema,
        backend=backend,
        cache=cache,
        interrupt_on=interrupt_on,
    )

    return _create_from_config(config)


def _create_from_config(config: DeepResearchConfig) -> "CompiledStateGraph":
    """Build a deep research agent from a validated config."""
    # Build tools list
    all_tools: list[BaseTool | Callable[..., Any]] = []

    auth_type = (
        config.auth_type.name
        if isinstance(config.auth_type, OCIAuthType)
        else config.auth_type
    )

    if config.datastores:
        datastore_tools = create_datastore_tools(
            stores=config.datastores,
            default_store=config.default_datastore,
            embedding_model=config.embedding_model,
            compartment_id=config.compartment_id or "",
            service_endpoint=config.service_endpoint or "",
            auth_type=auth_type,
            auth_profile=config.auth_profile,
            top_k=config.top_k,
        )
        all_tools.extend(datastore_tools)

    if config.tools:
        all_tools.extend(config.tools)

    llm = _build_llm(config)

    if config._can_use_lightweight:
        compiled = _build_lightweight(config, llm, all_tools)
    else:
        compiled = _build_deep(config, llm, all_tools)

    # Expose the underlying OCI chat model for cleanup
    setattr(compiled, "_oci_llm", llm)
    return compiled


def _build_lightweight(
    config: DeepResearchConfig,
    llm: Any,
    tools: list[Any],
) -> Any:
    """Build a lightweight ReAct agent."""
    create_agent_func, use_legacy_api = _get_agent_factory()
    prompt_key = "prompt" if use_legacy_api else "system_prompt"
    agent_kwargs = {
        "model": llm,
        "tools": tools,
        **_filter_none(
            middleware=None if use_legacy_api else tuple(config.middleware or ()),
            checkpointer=config.checkpointer,
            store=config.store,
            interrupt_before=config.interrupt_before,
            interrupt_after=config.interrupt_after,
            name=config.name,
            **{prompt_key: config.system_prompt},
        ),
        "debug": config.debug,
    }
    with _langgraph_schema_fallback():
        return create_agent_func(
            **{key: value for key, value in agent_kwargs.items() if value is not None}
        )


def _build_deep(
    config: DeepResearchConfig,
    llm: Any,
    tools: list[Any],
) -> Any:
    """Build a full deep agent."""
    try:
        from deepagents import create_deep_agent
    except ImportError as ex:
        raise ImportError(
            "deepagents required. Install with: pip install deepagents"
        ) from ex

    agent_kwargs = {
        "model": llm,
        "tools": tools,
        **_filter_none(
            system_prompt=config.system_prompt,
            subagents=config.subagents,
            skills=config.skills,
            memory=config.memory,
            middleware=config.middleware,
            response_format=config.response_format,
            context_schema=config.context_schema,
            checkpointer=config.checkpointer,
            store=config.store,
            backend=config.backend,
            cache=config.cache,
            interrupt_on=config.interrupt_on,
            name=config.name,
        ),
    }

    if config.debug:
        agent_kwargs["debug"] = True

    with _langgraph_schema_fallback():
        return create_deep_agent(**agent_kwargs)
