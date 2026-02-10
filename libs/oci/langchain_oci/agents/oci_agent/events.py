# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Typed event system for OCIGenAIAgent streaming.

Provides strongly-typed events emitted during agent execution, enabling
consumers to handle different stages of the agentic loop with pattern matching.

Example:
    for event in agent.stream("What's the weather?"):
        if isinstance(event, ThinkEvent):
            print(f"Thinking: {event.thought}")
        elif isinstance(event, ToolCompleteEvent):
            print(f"Tool result: {event.result}")
        elif isinstance(event, TerminateEvent):
            print(f"Done: {event.reason}")
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class ThinkEvent(BaseModel):
    """Emitted when the agent produces a thought/reasoning.

    Attributes:
        event_type: Always "think".
        iteration: Current iteration number.
        thought: Agent's reasoning text.
        tool_calls_planned: Number of tool calls the agent plans to make.
    """

    model_config = ConfigDict(frozen=True)

    event_type: Literal["think"] = "think"
    iteration: int
    thought: str
    tool_calls_planned: int = 0


class ToolStartEvent(BaseModel):
    """Emitted when a tool execution begins.

    Attributes:
        event_type: Always "tool_start".
        iteration: Current iteration number.
        tool_name: Name of the tool being invoked.
        tool_call_id: Unique identifier for this tool call.
        arguments: Arguments passed to the tool.
    """

    model_config = ConfigDict(frozen=True)

    event_type: Literal["tool_start"] = "tool_start"
    iteration: int
    tool_name: str
    tool_call_id: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class ToolCompleteEvent(BaseModel):
    """Emitted when a tool execution completes.

    Attributes:
        event_type: Always "tool_complete".
        iteration: Current iteration number.
        tool_name: Name of the tool that was invoked.
        tool_call_id: Unique identifier for this tool call.
        result: Result returned by the tool.
        success: Whether the execution succeeded.
        error: Error message if execution failed.
        duration_ms: Execution time in milliseconds.
    """

    model_config = ConfigDict(frozen=True)

    event_type: Literal["tool_complete"] = "tool_complete"
    iteration: int
    tool_name: str
    tool_call_id: str
    result: str = ""
    success: bool = True
    error: Optional[str] = None
    duration_ms: float = 0.0


class ReflectEvent(BaseModel):
    """Emitted after reflexion analysis of the current iteration.

    Attributes:
        event_type: Always "reflect".
        iteration: Current iteration number.
        confidence: Current confidence score (0.0 to 1.0).
        confidence_delta: Change in confidence from this iteration.
        assessment: Progress assessment category.
        loop_detected: Whether a tool loop was detected.
        guidance: Optional guidance for escaping stuck states.
    """

    model_config = ConfigDict(frozen=True)

    event_type: Literal["reflect"] = "reflect"
    iteration: int
    confidence: float
    confidence_delta: float = 0.0
    assessment: str = "on_track"
    loop_detected: bool = False
    guidance: Optional[str] = None


class TerminateEvent(BaseModel):
    """Emitted when the agent terminates.

    Attributes:
        event_type: Always "terminate".
        reason: Termination reason code.
        final_answer: Agent's final response.
        total_iterations: Total iterations completed.
        total_tool_calls: Total tools invoked.
        confidence: Final confidence score.
    """

    model_config = ConfigDict(frozen=True)

    event_type: Literal["terminate"] = "terminate"
    reason: str
    final_answer: str
    total_iterations: int
    total_tool_calls: int
    confidence: float = 0.0


# Union type for pattern matching
AgentEvent = Union[
    ThinkEvent,
    ToolStartEvent,
    ToolCompleteEvent,
    ReflectEvent,
    TerminateEvent,
]
