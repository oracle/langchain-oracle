# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""OCI Generative AI Agent helpers.

Two agent patterns are available:

1. create_oci_agent() - Thin wrapper around LangGraph's ReAct pattern.
   Simple, delegates control to LangGraph.

2. OCIGenAIAgent - Full agentic loop with state management, reflexion, audit trail.
   Provides typed event streaming, 5 termination conditions, and full control.

Example - create_oci_agent (simple):
    from langchain_oci import create_oci_agent
    agent = create_oci_agent(llm, tools)
    result = agent.invoke({"messages": [HumanMessage(content="Hello")]})

Example - OCIGenAIAgent (full control):
    from langchain_oci import OCIGenAIAgent, ThinkEvent, TerminateEvent
    agent = OCIGenAIAgent(model_id="meta.llama-4-scout-17b-16e-instruct", tools=tools)
    for event in agent.stream("What's the weather?"):
        if isinstance(event, ThinkEvent):
            print(f"Thinking: {event.thought}")
        elif isinstance(event, TerminateEvent):
            print(f"Done: {event.reason}")
"""

from langchain_oci.agents.oci_agent import (
    AgentEvent,
    AgentResult,
    AgentState,
    AssessmentCategory,
    OCIGenAIAgent,
    ReasoningStep,
    ReflectionResult,
    ReflectEvent,
    Reflector,
    TerminateEvent,
    TerminationReason,
    ThinkEvent,
    ToolCompleteEvent,
    ToolExecution,
    ToolStartEvent,
)
from langchain_oci.agents.react import create_oci_agent

__all__ = [
    # Simple ReAct wrapper
    "create_oci_agent",
    # OCIGenAIAgent - full agentic loop
    "OCIGenAIAgent",
    "AgentState",
    "AgentResult",
    "ReasoningStep",
    "ToolExecution",
    # Events
    "AgentEvent",
    "ThinkEvent",
    "ToolStartEvent",
    "ToolCompleteEvent",
    "ReflectEvent",
    "TerminateEvent",
    # Reflexion
    "Reflector",
    "ReflectionResult",
    "AssessmentCategory",
    # Termination
    "TerminationReason",
]
