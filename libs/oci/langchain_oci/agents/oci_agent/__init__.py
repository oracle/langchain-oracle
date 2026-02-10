# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""OCIGenAIAgent - Sophisticated agentic loop for OCI Generative AI.

This module provides a full agentic loop with:
- Immutable state management
- Typed event streaming
- Reflexion (confidence tracking + loop detection)
- 5 termination conditions
- LangChain Runnable interface for LCEL composability
- LangGraph node compatibility

Example:
    from langchain_oci import OCIGenAIAgent
    from langchain_core.tools import tool

    @tool
    def search(query: str) -> str:
        '''Search for information.'''
        return f"Results for: {query}"

    agent = OCIGenAIAgent(
        model_id="meta.llama-4-scout-17b-16e-instruct",
        tools=[search],
        compartment_id="ocid1.compartment...",
    )

    result = agent.invoke("What is the capital of France?")
    print(result.final_answer)

    # Stream typed events
    for event in agent.stream("What's the weather?"):
        if isinstance(event, ThinkEvent):
            print(f"Thinking: {event.thought}")
        elif isinstance(event, ToolCompleteEvent):
            print(f"Tool result: {event.result}")
        elif isinstance(event, TerminateEvent):
            print(f"Done: {event.reason}")
"""

from langchain_oci.agents.oci_agent.agent import OCIGenAIAgent
from langchain_oci.agents.oci_agent.checkpoint import (
    BaseCheckpointer,
    Checkpoint,
    FileCheckpointer,
    LangGraphCheckpointerAdapter,
    MemoryCheckpointer,
    wrap_checkpointer,
)
from langchain_oci.agents.oci_agent.compression import (
    CompressionConfig,
    CompressionResult,
    CompressionStrategy,
    compress_messages,
)
from langchain_oci.agents.oci_agent.confidence import (
    ConfidenceSignal,
    SignalType,
    compute_accumulated_confidence,
    detect_confidence_signals,
    should_early_exit,
)
from langchain_oci.agents.oci_agent.events import (
    AgentEvent,
    ReflectEvent,
    TerminateEvent,
    ThinkEvent,
    ToolCompleteEvent,
    ToolStartEvent,
)
from langchain_oci.agents.oci_agent.hooks import (
    AgentHooks,
    IterationContext,
    ToolHookContext,
    ToolResultContext,
    create_logging_hooks,
    create_metrics_hooks,
)
from langchain_oci.agents.oci_agent.reflexion import (
    AssessmentCategory,
    ReflectionResult,
    Reflector,
    assess_confidence,
    assess_progress,
    detect_loop,
)
from langchain_oci.agents.oci_agent.state import (
    AgentResult,
    AgentState,
    ReasoningStep,
    ToolExecution,
)
from langchain_oci.agents.oci_agent.termination import (
    TerminationReason,
    check_termination,
    get_termination_description,
)

__all__ = [
    # Main class
    "OCIGenAIAgent",
    # State
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
    "assess_confidence",
    "assess_progress",
    "detect_loop",
    # Compression
    "CompressionConfig",
    "CompressionResult",
    "CompressionStrategy",
    "compress_messages",
    # Confidence Signals
    "ConfidenceSignal",
    "SignalType",
    "compute_accumulated_confidence",
    "detect_confidence_signals",
    "should_early_exit",
    # Hooks
    "AgentHooks",
    "IterationContext",
    "ToolHookContext",
    "ToolResultContext",
    "create_logging_hooks",
    "create_metrics_hooks",
    # Termination
    "TerminationReason",
    "check_termination",
    "get_termination_description",
    # Checkpointing
    "BaseCheckpointer",
    "Checkpoint",
    "MemoryCheckpointer",
    "FileCheckpointer",
    "LangGraphCheckpointerAdapter",
    "wrap_checkpointer",
]
