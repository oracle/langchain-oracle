# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Hooks system for agent lifecycle callbacks.

Provides pre/post callbacks for tool execution, iteration, and termination.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Protocol

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from langchain_oci.agents.oci_agent.state import AgentState, ToolExecution


class ToolHookContext(BaseModel):
    """Context passed to tool hooks.

    Attributes:
        tool_name: Name of the tool being executed.
        tool_call_id: Unique identifier for this call.
        arguments: Arguments passed to the tool.
        iteration: Current iteration number.
    """

    model_config = ConfigDict(frozen=True)

    tool_name: str
    tool_call_id: str
    arguments: dict[str, Any]
    iteration: int


class ToolResultContext(BaseModel):
    """Context passed to post-tool hooks.

    Attributes:
        tool_name: Name of the tool that was executed.
        tool_call_id: Unique identifier for this call.
        arguments: Arguments that were passed.
        result: Result from the tool.
        success: Whether execution succeeded.
        error: Error message if failed.
        duration_ms: Execution time in milliseconds.
        iteration: Current iteration number.
    """

    model_config = ConfigDict(frozen=True)

    tool_name: str
    tool_call_id: str
    arguments: dict[str, Any]
    result: str
    success: bool
    error: str | None
    duration_ms: float
    iteration: int


class IterationContext(BaseModel):
    """Context passed to iteration hooks.

    Attributes:
        iteration: Current iteration number.
        confidence: Current confidence score.
        tool_count: Number of tools called so far.
    """

    model_config = ConfigDict(frozen=True)

    iteration: int
    confidence: float
    tool_count: int


# Hook type definitions
PreToolHook = Callable[[ToolHookContext], None]
PostToolHook = Callable[[ToolResultContext], None]
PreIterationHook = Callable[[IterationContext], None]
PostIterationHook = Callable[[IterationContext], None]
OnTerminateHook = Callable[[str, str], None]  # (reason, final_answer)


class AgentHooks(BaseModel):
    """Collection of agent lifecycle hooks.

    All hooks are optional. They receive context but cannot modify
    agent behavior (observation only).

    Attributes:
        on_tool_start: Called before each tool execution.
        on_tool_end: Called after each tool execution.
        on_iteration_start: Called at the start of each iteration.
        on_iteration_end: Called at the end of each iteration.
        on_terminate: Called when the agent terminates.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    on_tool_start: list[PreToolHook] | None = None
    on_tool_end: list[PostToolHook] | None = None
    on_iteration_start: list[PreIterationHook] | None = None
    on_iteration_end: list[PostIterationHook] | None = None
    on_terminate: list[OnTerminateHook] | None = None

    def trigger_tool_start(self, context: ToolHookContext) -> None:
        """Trigger all on_tool_start hooks."""
        if self.on_tool_start:
            for hook in self.on_tool_start:
                try:
                    hook(context)
                except Exception:
                    pass  # Hooks should not break agent execution

    def trigger_tool_end(self, context: ToolResultContext) -> None:
        """Trigger all on_tool_end hooks."""
        if self.on_tool_end:
            for hook in self.on_tool_end:
                try:
                    hook(context)
                except Exception:
                    pass

    def trigger_iteration_start(self, context: IterationContext) -> None:
        """Trigger all on_iteration_start hooks."""
        if self.on_iteration_start:
            for hook in self.on_iteration_start:
                try:
                    hook(context)
                except Exception:
                    pass

    def trigger_iteration_end(self, context: IterationContext) -> None:
        """Trigger all on_iteration_end hooks."""
        if self.on_iteration_end:
            for hook in self.on_iteration_end:
                try:
                    hook(context)
                except Exception:
                    pass

    def trigger_terminate(self, reason: str, final_answer: str) -> None:
        """Trigger all on_terminate hooks."""
        if self.on_terminate:
            for hook in self.on_terminate:
                try:
                    hook(reason, final_answer)
                except Exception:
                    pass


def create_logging_hooks() -> AgentHooks:
    """Create hooks that log agent activity.

    Returns:
        AgentHooks configured for logging.
    """
    import logging

    logger = logging.getLogger("langchain_oci.agents")

    def log_tool_start(ctx: ToolHookContext) -> None:
        logger.info(f"Tool start: {ctx.tool_name}({ctx.arguments})")

    def log_tool_end(ctx: ToolResultContext) -> None:
        status = "success" if ctx.success else "failed"
        logger.info(f"Tool end: {ctx.tool_name} [{status}] {ctx.duration_ms:.1f}ms")

    def log_iteration_start(ctx: IterationContext) -> None:
        logger.debug(f"Iteration {ctx.iteration} start")

    def log_iteration_end(ctx: IterationContext) -> None:
        logger.debug(
            f"Iteration {ctx.iteration} end, "
            f"confidence={ctx.confidence:.2f}, tools={ctx.tool_count}"
        )

    def log_terminate(reason: str, final_answer: str) -> None:
        logger.info(f"Agent terminated: {reason}")

    return AgentHooks(
        on_tool_start=[log_tool_start],
        on_tool_end=[log_tool_end],
        on_iteration_start=[log_iteration_start],
        on_iteration_end=[log_iteration_end],
        on_terminate=[log_terminate],
    )


def create_metrics_hooks() -> tuple[AgentHooks, dict[str, Any]]:
    """Create hooks that collect metrics.

    Returns:
        Tuple of (AgentHooks, metrics_dict).
        The metrics_dict is updated in-place by the hooks.
    """
    metrics: dict[str, Any] = {
        "total_tool_calls": 0,
        "tool_durations_ms": [],
        "tool_errors": 0,
        "iterations": 0,
        "termination_reason": None,
    }

    def count_tool_end(ctx: ToolResultContext) -> None:
        metrics["total_tool_calls"] += 1
        metrics["tool_durations_ms"].append(ctx.duration_ms)
        if not ctx.success:
            metrics["tool_errors"] += 1

    def count_iteration(ctx: IterationContext) -> None:
        metrics["iterations"] = ctx.iteration

    def record_terminate(reason: str, final_answer: str) -> None:
        metrics["termination_reason"] = reason

    hooks = AgentHooks(
        on_tool_end=[count_tool_end],
        on_iteration_end=[count_iteration],
        on_terminate=[record_terminate],
    )

    return hooks, metrics
