# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""OCI ReAct Agent helper functions."""

from __future__ import annotations

import os
from typing import Any, Callable, List, Optional, Sequence, Union

from langchain_core.tools import BaseTool

from langchain_oci.chat_models.oci_generative_ai import ChatOCIGenAI
from langchain_oci.common.auth import OCIAuthType


def create_oci_agent(
    model_id: str,
    tools: Sequence[Union[BaseTool, Callable[..., Any]]],
    *,
    # OCI-specific options
    compartment_id: Optional[str] = None,
    service_endpoint: Optional[str] = None,
    auth_type: Union[str, OCIAuthType] = OCIAuthType.API_KEY,
    auth_profile: str = "DEFAULT",
    auth_file_location: str = "~/.oci/config",
    max_sequential_tool_calls: int = 8,
    # Agent options
    system_prompt: Optional[str] = None,
    checkpointer: Optional[Any] = None,
    store: Optional[Any] = None,
    # Control flow
    interrupt_before: Optional[List[str]] = None,
    interrupt_after: Optional[List[str]] = None,
    # Model kwargs
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    # Debug
    debug: bool = False,
    name: Optional[str] = None,
    # Extra model kwargs
    **model_kwargs: Any,
) -> Any:
    """Create a ReAct agent using OCI Generative AI models.

    This is a convenience wrapper that creates a ChatOCIGenAI model,
    binds the provided tools, and creates an agent using langchain.agents.

    Args:
        model_id: OCI model identifier (e.g., "meta.llama-4-scout-17b-16e-instruct")
        tools: List of tools the agent can use. Can be BaseTool instances or
            callable functions with docstrings.
        compartment_id: OCI compartment OCID. Defaults to OCI_COMPARTMENT_ID env var.
        service_endpoint: OCI GenAI service endpoint. Defaults to
            OCI_SERVICE_ENDPOINT env var or constructs from OCI_REGION.
        auth_type: OCI authentication type (API_KEY, SECURITY_TOKEN,
            INSTANCE_PRINCIPAL, RESOURCE_PRINCIPAL). Default: API_KEY.
        auth_profile: OCI config profile name. Default: "DEFAULT".
        auth_file_location: Path to OCI config file. Default: "~/.oci/config".
        max_sequential_tool_calls: Max tool calls before forcing stop. Default: 8.
            Prevents infinite loops while allowing multi-step orchestration.
        system_prompt: System message for the agent.
        checkpointer: LangGraph checkpointer for persistence. Enables resuming
            conversations and human-in-the-loop workflows.
        store: LangGraph store for long-term memory across conversations.
        interrupt_before: Node names to interrupt before (for human-in-the-loop).
        interrupt_after: Node names to interrupt after.
        temperature: Model temperature. If not specified, uses model default.
        max_tokens: Maximum tokens to generate.
        debug: Enable debug mode for the agent graph.
        name: Name for the agent graph. Default: "oci_react_agent".
        **model_kwargs: Additional keyword arguments passed to ChatOCIGenAI
            model_kwargs parameter.

    Returns:
        CompiledGraph: A compiled LangGraph agent ready to invoke.

    Raises:
        ValueError: If compartment_id is not provided and OCI_COMPARTMENT_ID
            environment variable is not set.
        ImportError: If langchain is not installed.

    Example:
        Basic usage with a simple tool:

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
        >>>
        >>> from langchain_core.messages import HumanMessage
        >>> result = agent.invoke(
        ...     {"messages": [HumanMessage(content="What's the weather in Chicago?")]}
        ... )

    Example:
        With checkpointing for conversation persistence:

        >>> from langgraph.checkpoint.memory import MemorySaver
        >>>
        >>> checkpointer = MemorySaver()
        >>> agent = create_oci_agent(
        ...     model_id="meta.llama-4-scout-17b-16e-instruct",
        ...     tools=[get_weather],
        ...     checkpointer=checkpointer,
        ... )
        >>>
        >>> # Conversations are persisted by thread_id
        >>> result = agent.invoke(
        ...     {"messages": [HumanMessage(content="What's the weather?")]},
        ...     config={"configurable": {"thread_id": "user_123"}},
        ... )
    """
    # Try langchain >= 1.0.0 first, fall back to langgraph for older versions
    create_agent_func = None
    use_legacy_api = False

    try:
        from langchain.agents import create_agent

        create_agent_func = create_agent
    except (ImportError, AttributeError):
        # Fall back to langgraph.prebuilt for langchain < 1.0.0
        try:
            from langgraph.prebuilt import create_react_agent

            create_agent_func = create_react_agent
            use_legacy_api = True
        except ImportError as ex:
            raise ImportError(
                "Could not import agent creation function. "
                "Please install langchain>=1.0.0 or langgraph."
            ) from ex

    # Resolve compartment_id from environment if not provided
    if compartment_id is None:
        compartment_id = os.environ.get("OCI_COMPARTMENT_ID")
        if compartment_id is None:
            raise ValueError(
                "compartment_id must be provided or set via "
                "OCI_COMPARTMENT_ID environment variable"
            )

    # Resolve service_endpoint from environment if not provided
    if service_endpoint is None:
        service_endpoint = os.environ.get("OCI_SERVICE_ENDPOINT")
        if service_endpoint is None:
            # Try to construct from region
            region = os.environ.get("OCI_REGION")
            if region:
                service_endpoint = (
                    f"https://inference.generativeai.{region}.oci.oraclecloud.com"
                )

    # Handle auth_type as enum or string
    if isinstance(auth_type, OCIAuthType):
        auth_type_str = auth_type.name
    else:
        auth_type_str = auth_type

    # Build model kwargs
    llm_model_kwargs: dict[str, Any] = {**model_kwargs}
    if temperature is not None:
        llm_model_kwargs["temperature"] = temperature
    if max_tokens is not None:
        llm_model_kwargs["max_tokens"] = max_tokens

    # Create OCI chat model
    llm = ChatOCIGenAI(
        model_id=model_id,
        compartment_id=compartment_id,
        service_endpoint=service_endpoint,
        auth_type=auth_type_str,
        auth_profile=auth_profile,
        auth_file_location=auth_file_location,
        model_kwargs=llm_model_kwargs if llm_model_kwargs else None,
        max_sequential_tool_calls=max_sequential_tool_calls,
    )

    # Build kwargs for create_agent
    agent_kwargs: dict[str, Any] = {
        "model": llm,
        "tools": list(tools),
    }

    # Add optional parameters only if provided
    # Handle different parameter names between APIs
    if system_prompt is not None:
        if use_legacy_api:
            agent_kwargs["prompt"] = system_prompt
        else:
            agent_kwargs["system_prompt"] = system_prompt

    if checkpointer is not None:
        agent_kwargs["checkpointer"] = checkpointer

    if store is not None:
        agent_kwargs["store"] = store

    if interrupt_before is not None:
        agent_kwargs["interrupt_before"] = interrupt_before

    if interrupt_after is not None:
        agent_kwargs["interrupt_after"] = interrupt_after

    if debug:
        agent_kwargs["debug"] = debug

    if name is not None:
        agent_kwargs["name"] = name

    # Create the agent
    agent = create_agent_func(**agent_kwargs)

    return agent
