# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""OCI ReAct Agent helper functions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Sequence

from langchain_core.tools import BaseTool

from langchain_oci.agents.common import OCIConfig, filter_none, merge_model_kwargs
from langchain_oci.chat_models.oci_generative_ai import ChatOCIGenAI
from langchain_oci.common.auth import OCIAuthType

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


def create_oci_agent(
    model_id: str,
    tools: Sequence[BaseTool | Callable[..., Any]],
    *,
    # OCI-specific options
    compartment_id: str | None = None,
    service_endpoint: str | None = None,
    auth_type: str | OCIAuthType = OCIAuthType.API_KEY,
    auth_profile: str = "DEFAULT",
    auth_file_location: str = "~/.oci/config",
    max_sequential_tool_calls: int = 8,
    tool_result_guidance: bool = False,
    # Agent options
    system_prompt: str | None = None,
    checkpointer: Any = None,
    store: Any = None,
    # Control flow
    interrupt_before: list[str] | None = None,
    interrupt_after: list[str] | None = None,
    # Model kwargs
    temperature: float | None = None,
    max_tokens: int | None = None,
    # Debug
    debug: bool = False,
    name: str | None = None,
    # Extra model kwargs
    **model_kwargs: Any,
) -> "CompiledStateGraph":
    """Create a ReAct agent using OCI Generative AI models.

    This is a convenience wrapper that creates a ChatOCIGenAI model,
    binds the provided tools, and creates an agent using langchain.agents.

    For advanced capabilities (planning, file operations, subagent spawning),
    see :func:`create_deep_research_agent`.

    Args:
        model_id: OCI model identifier (e.g., "meta.llama-4-scout-17b-16e-instruct")
        tools: List of tools the agent can use.
        compartment_id: OCI compartment OCID.
        service_endpoint: OCI GenAI service endpoint.
        auth_type: OCI authentication type.
        auth_profile: OCI config profile name.
        auth_file_location: Path to OCI config file.
        max_sequential_tool_calls: Max tool calls before forcing stop.
        tool_result_guidance: Inject guidance after tool results for Llama models.
        system_prompt: System message for the agent.
        checkpointer: LangGraph checkpointer for persistence.
        store: LangGraph store for long-term memory.
        interrupt_before: Node names to interrupt before.
        interrupt_after: Node names to interrupt after.
        temperature: Model temperature.
        max_tokens: Maximum tokens to generate.
        debug: Enable debug mode.
        name: Name for the agent graph.
        **model_kwargs: Additional model kwargs.

    Returns:
        CompiledGraph: A compiled LangGraph agent ready to invoke.

    Raises:
        ValueError: If compartment_id is not available.
        ImportError: If langchain or langgraph is not installed.

    Example:
        >>> from langchain_oci import create_oci_agent
        >>> from langchain_core.tools import tool
        >>>
        >>> @tool
        >>> def get_weather(city: str) -> str:
        ...     \"\"\"Get the current weather for a city.\"\"\"
        ...     return f"Weather in {city}: 72F, sunny"
        >>>
        >>> agent = create_oci_agent(
        ...     model_id="meta.llama-4-scout-17b-16e-instruct",
        ...     tools=[get_weather],
        ...     system_prompt="You are a helpful weather assistant.",
        ... )
    """
    # Get agent creation function - prefer langchain >= 1.0.0
    create_agent_func, use_legacy_api = _get_agent_factory()

    # Resolve OCI configuration
    oci_config = OCIConfig.resolve(
        compartment_id=compartment_id,
        service_endpoint=service_endpoint,
        auth_type=auth_type,
        auth_profile=auth_profile,
        auth_file_location=auth_file_location,
    )

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
        max_sequential_tool_calls=max_sequential_tool_calls,
        tool_result_guidance=tool_result_guidance,
    )

    # Build agent kwargs
    prompt_key = "prompt" if use_legacy_api else "system_prompt"
    agent_kwargs = {
        "model": llm,
        "tools": list(tools),
        **filter_none(
            checkpointer=checkpointer,
            store=store,
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after,
            name=name,
            **{prompt_key: system_prompt},
        ),
    }

    if debug:
        agent_kwargs["debug"] = True

    return create_agent_func(**agent_kwargs)


def _get_agent_factory() -> tuple[Callable[..., Any], bool]:
    """Get the appropriate agent factory function.

    Returns:
        Tuple of (factory_function, is_legacy_api).
        is_legacy_api is True when using langgraph.prebuilt.create_react_agent.
    """
    # Try langchain >= 1.0.0 first
    try:
        from langchain.agents import create_agent

        return create_agent, False
    except (ImportError, AttributeError):
        pass

    # Fall back to langgraph.prebuilt for langchain < 1.0.0
    try:
        from langgraph.prebuilt import create_react_agent

        return create_react_agent, True
    except ImportError as ex:
        raise ImportError(
            "Could not import agent creation function. "
            "Please install langchain>=1.0.0 or langgraph."
        ) from ex
