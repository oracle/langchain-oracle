# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Immutable state management for OCIGenAIAgent.

Provides frozen Pydantic models with functional update patterns for maintaining
agent state throughout the agentic loop.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, ConfigDict, Field


class ToolExecution(BaseModel):
    """Record of a single tool execution.

    Attributes:
        tool_name: Name of the tool that was executed.
        tool_call_id: Unique identifier for this tool call.
        arguments: Arguments passed to the tool.
        result: Result returned by the tool.
        success: Whether the execution succeeded.
        error: Error message if execution failed.
        duration_ms: Execution time in milliseconds.
    """

    model_config = ConfigDict(frozen=True)

    tool_name: str
    tool_call_id: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    result: str = ""
    success: bool = True
    error: Optional[str] = None
    duration_ms: float = 0.0


class ReasoningStep(BaseModel):
    """Record of a single reasoning iteration.

    Captures the agent's thought process, tool executions, and confidence
    assessment for one iteration of the agentic loop.

    Attributes:
        iteration: Iteration number (0-indexed).
        thought: Agent's reasoning/thought for this step.
        tool_executions: Tools executed during this iteration.
        confidence: Confidence score after this iteration (0.0 to 1.0).
        assessment: Progress assessment
            ('on_track', 'stuck', 'new_findings', 'loop_detected').
    """

    model_config = ConfigDict(frozen=True)

    iteration: int
    thought: str = ""
    tool_executions: Tuple[ToolExecution, ...] = ()
    confidence: float = 0.0
    assessment: str = "on_track"


class AgentState(BaseModel):
    """Immutable agent state with functional update methods.

    All state updates return new instances, preserving immutability for
    reliable debugging, testing, and state tracking.

    Compatible with LangGraph state management patterns.

    Attributes:
        messages: Conversation history as LangChain messages.
        reasoning_steps: Audit trail of reasoning iterations.
        iteration: Current iteration number.
        confidence: Current confidence level (0.0 to 1.0).
        tool_history: Sequence of tool names called (for loop detection).
        metadata: Additional state metadata.
    """

    model_config = ConfigDict(frozen=True)

    messages: Tuple[BaseMessage, ...] = ()
    reasoning_steps: Tuple[ReasoningStep, ...] = ()
    iteration: int = 0
    confidence: float = 0.0
    tool_history: Tuple[str, ...] = ()
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def with_message(self, message: BaseMessage) -> AgentState:
        """Return new state with message appended.

        Args:
            message: Message to append to conversation history.

        Returns:
            New AgentState with the message added.
        """
        return self.model_copy(
            update={"messages": (*self.messages, message)},
        )

    def with_messages(self, messages: List[BaseMessage]) -> AgentState:
        """Return new state with multiple messages appended.

        Args:
            messages: Messages to append to conversation history.

        Returns:
            New AgentState with the messages added.
        """
        return self.model_copy(
            update={"messages": (*self.messages, *messages)},
        )

    def with_reasoning_step(self, step: ReasoningStep) -> AgentState:
        """Return new state with reasoning step appended.

        Args:
            step: Reasoning step to record.

        Returns:
            New AgentState with the step added.
        """
        return self.model_copy(
            update={"reasoning_steps": (*self.reasoning_steps, step)},
        )

    def with_tool_call(self, tool_name: str) -> AgentState:
        """Return new state with tool call recorded in history.

        Args:
            tool_name: Name of the tool that was called.

        Returns:
            New AgentState with the tool recorded.
        """
        return self.model_copy(
            update={"tool_history": (*self.tool_history, tool_name)},
        )

    def with_confidence(self, confidence: float) -> AgentState:
        """Return new state with updated confidence.

        Args:
            confidence: New confidence value (clamped to 0.0-1.0).

        Returns:
            New AgentState with updated confidence.
        """
        clamped = max(0.0, min(1.0, confidence))
        return self.model_copy(update={"confidence": clamped})

    def adjust_confidence(
        self,
        delta: float,
        diminishing: bool = True,
    ) -> AgentState:
        """Return new state with confidence adjusted by delta.

        Args:
            delta: Amount to adjust confidence (-1.0 to 1.0).
            diminishing: Apply diminishing returns for positive deltas.

        Returns:
            New AgentState with adjusted confidence.
        """
        if diminishing and delta > 0:
            # Diminishing returns as confidence approaches 1.0
            effective_delta = delta * (1.0 - self.confidence)
        else:
            effective_delta = delta

        new_confidence = max(0.0, min(1.0, self.confidence + effective_delta))
        return self.model_copy(update={"confidence": new_confidence})

    def increment_iteration(self) -> AgentState:
        """Return new state with iteration incremented.

        Returns:
            New AgentState with iteration + 1.
        """
        return self.model_copy(update={"iteration": self.iteration + 1})

    def with_metadata(self, key: str, value: Any) -> AgentState:
        """Return new state with metadata updated.

        Args:
            key: Metadata key.
            value: Metadata value.

        Returns:
            New AgentState with updated metadata.
        """
        new_metadata = {**self.metadata, key: value}
        return self.model_copy(update={"metadata": new_metadata})


class AgentResult(BaseModel):
    """Final result from OCIGenAIAgent execution.

    Compatible with LangGraph state reducers (messages as list for add_messages).

    Attributes:
        messages: Full conversation history.
        final_answer: The agent's final response.
        termination_reason: Why the agent stopped.
        reasoning_steps: Complete audit trail of reasoning.
        total_iterations: Number of iterations completed.
        total_tool_calls: Number of tools invoked.
        confidence: Final confidence score.
    """

    messages: List[BaseMessage]
    final_answer: str
    termination_reason: str
    reasoning_steps: List[ReasoningStep]
    total_iterations: int
    total_tool_calls: int
    confidence: float = 0.0

    @classmethod
    def from_state(
        cls,
        state: AgentState,
        final_answer: str,
        termination_reason: str,
    ) -> AgentResult:
        """Create result from final agent state.

        Args:
            state: Final agent state.
            final_answer: The agent's final response.
            termination_reason: Why the agent stopped.

        Returns:
            AgentResult summarizing the execution.
        """
        total_tool_calls = sum(
            len(step.tool_executions) for step in state.reasoning_steps
        )
        return cls(
            messages=list(state.messages),
            final_answer=final_answer,
            termination_reason=termination_reason,
            reasoning_steps=list(state.reasoning_steps),
            total_iterations=state.iteration,
            total_tool_calls=total_tool_calls,
            confidence=state.confidence,
        )
