# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Termination conditions for OCIGenAIAgent.

Implements 5 termination conditions that determine when the agent should stop:

1. max_iterations - Iteration count >= max_iterations
2. confidence_met - Confidence score >= threshold
3. terminal_tool - Agent called a terminal tool (e.g., "done", "submit")
4. tool_loop - Same tool called repeatedly with same arguments
5. no_tools - Model didn't call any tools (natural completion)
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from langchain_oci.agents.oci_agent.state import AgentState


class TerminationReason:
    """Standard termination reason codes."""

    MAX_ITERATIONS = "max_iterations"
    CONFIDENCE_MET = "confidence_met"
    TERMINAL_TOOL = "terminal_tool"
    TOOL_LOOP = "tool_loop"
    NO_TOOLS = "no_tools"


def check_max_iterations(
    state: AgentState,
    max_iterations: int,
) -> Optional[str]:
    """Check if max iterations reached.

    Args:
        state: Current agent state.
        max_iterations: Maximum allowed iterations.

    Returns:
        TerminationReason.MAX_ITERATIONS if limit reached, None otherwise.
    """
    if state.iteration >= max_iterations:
        return TerminationReason.MAX_ITERATIONS
    return None


def check_confidence_threshold(
    state: AgentState,
    confidence_threshold: float,
) -> Optional[str]:
    """Check if confidence threshold met.

    Args:
        state: Current agent state.
        confidence_threshold: Minimum confidence to consider complete.

    Returns:
        TerminationReason.CONFIDENCE_MET if threshold reached, None otherwise.
    """
    if state.confidence >= confidence_threshold:
        return TerminationReason.CONFIDENCE_MET
    return None


def check_terminal_tool(
    last_tool_calls: List[str],
    terminal_tools: set[str],
) -> Optional[str]:
    """Check if a terminal tool was called.

    Terminal tools signal explicit completion (e.g., "done", "submit", "finish").

    Args:
        last_tool_calls: Tool names called in the last iteration.
        terminal_tools: Set of tool names that signal termination.

    Returns:
        TerminationReason.TERMINAL_TOOL if terminal tool called, None otherwise.
    """
    for tool_name in last_tool_calls:
        if tool_name in terminal_tools:
            return TerminationReason.TERMINAL_TOOL
    return None


def check_tool_loop(
    state: AgentState,
    loop_threshold: int = 3,
) -> Optional[str]:
    """Check if agent is stuck in a tool loop.

    Detects patterns like:
    - Same tool called N times consecutively
    - Alternating pattern A->B->A->B

    Args:
        state: Current agent state.
        loop_threshold: Number of repeated calls to consider a loop.

    Returns:
        TerminationReason.TOOL_LOOP if loop detected, None otherwise.
    """
    if len(state.tool_history) < loop_threshold:
        return None

    recent_tools = state.tool_history[-loop_threshold:]
    tool_counts = Counter(recent_tools)

    # Check for single-tool loop (same tool called repeatedly)
    most_common = tool_counts.most_common(1)
    if most_common and most_common[0][1] == loop_threshold:
        return TerminationReason.TOOL_LOOP

    # Check for alternating pattern (A->B->A->B)
    if loop_threshold >= 4 and len(state.tool_history) >= 4:
        recent = state.tool_history[-4:]
        if recent[0] == recent[2] and recent[1] == recent[3] and recent[0] != recent[1]:
            return TerminationReason.TOOL_LOOP

    return None


def check_no_tools(
    tool_calls_in_response: int,
) -> Optional[str]:
    """Check if model produced no tool calls.

    When the model responds without calling any tools, it typically
    means it has gathered enough information to answer.

    Args:
        tool_calls_in_response: Number of tool calls in the model's response.

    Returns:
        TerminationReason.NO_TOOLS if no tools called, None otherwise.
    """
    if tool_calls_in_response == 0:
        return TerminationReason.NO_TOOLS
    return None


def check_termination(
    state: AgentState,
    *,
    max_iterations: int,
    confidence_threshold: float,
    terminal_tools: set[str],
    last_tool_calls: Optional[List[str]] = None,
    tool_calls_in_response: Optional[int] = None,
    loop_threshold: int = 3,
    check_confidence: bool = True,
) -> Optional[str]:
    """Check all termination conditions.

    Checks are performed in priority order:
    1. max_iterations - Hard limit, always checked
    2. terminal_tool - Explicit completion signal
    3. tool_loop - Stuck detection
    4. confidence_met - Soft threshold (if enabled)
    5. no_tools - Natural completion (checked last)

    Args:
        state: Current agent state.
        max_iterations: Maximum allowed iterations.
        confidence_threshold: Minimum confidence to consider complete.
        terminal_tools: Set of tool names that signal termination.
        last_tool_calls: Tool names called in the last iteration.
        tool_calls_in_response: Number of tool calls in latest response.
        loop_threshold: Number of repeated calls to consider a loop.
        check_confidence: Whether to check confidence threshold.

    Returns:
        Termination reason string if should stop, None to continue.
    """
    # 1. Max iterations (hard limit)
    reason = check_max_iterations(state, max_iterations)
    if reason:
        return reason

    # 2. Terminal tool (explicit completion)
    if last_tool_calls:
        reason = check_terminal_tool(last_tool_calls, terminal_tools)
        if reason:
            return reason

    # 3. Tool loop (stuck detection)
    reason = check_tool_loop(state, loop_threshold)
    if reason:
        return reason

    # 4. Confidence threshold (soft limit)
    if check_confidence:
        reason = check_confidence_threshold(state, confidence_threshold)
        if reason:
            return reason

    # 5. No tools (natural completion) - checked last
    if tool_calls_in_response is not None:
        reason = check_no_tools(tool_calls_in_response)
        if reason:
            return reason

    return None


def get_termination_description(reason: str) -> str:
    """Get human-readable description of termination reason.

    Args:
        reason: Termination reason code.

    Returns:
        Human-readable description.
    """
    descriptions = {
        TerminationReason.MAX_ITERATIONS: "Maximum iterations reached",
        TerminationReason.CONFIDENCE_MET: "Confidence threshold met",
        TerminationReason.TERMINAL_TOOL: "Terminal tool invoked",
        TerminationReason.TOOL_LOOP: "Tool loop detected",
        TerminationReason.NO_TOOLS: "Natural completion (no tools called)",
    }
    return descriptions.get(reason, f"Unknown reason: {reason}")
