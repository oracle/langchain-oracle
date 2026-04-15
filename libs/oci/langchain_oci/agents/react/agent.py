# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""OCI ReAct Agent helper functions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Sequence

from langchain_core.tools import BaseTool

from langchain_oci.agents.common import (
    AgentConfig,
    _build_llm,
    _filter_none,
    _get_agent_factory,
)
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
    create_agent_func, use_legacy_api = _get_agent_factory()

    config = AgentConfig(
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
    )

    llm = _build_llm(
        config,
        max_sequential_tool_calls=max_sequential_tool_calls,
        tool_result_guidance=tool_result_guidance,
    )

    prompt_key = "prompt" if use_legacy_api else "system_prompt"
    agent_kwargs = {
        "model": llm,
        "tools": list(tools),
        **_filter_none(
            checkpointer=config.checkpointer,
            store=config.store,
            interrupt_before=config.interrupt_before,
            interrupt_after=config.interrupt_after,
            name=config.name,
            **{prompt_key: config.system_prompt},
        ),
    }

    if config.debug:
        agent_kwargs["debug"] = True

    return create_agent_func(**agent_kwargs)
