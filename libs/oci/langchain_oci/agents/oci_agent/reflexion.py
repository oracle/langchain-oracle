# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Reflexion pattern for OCIGenAIAgent self-evaluation.

Enables the agent to evaluate its own progress and adjust strategy based on
tool results, detecting loops, and building confidence over iterations.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from langchain_oci.agents.oci_agent.state import AgentState, ToolExecution


class AssessmentCategory:
    """Categories for agent progress assessment."""

    ON_TRACK = "on_track"
    STUCK = "stuck"
    NEW_FINDINGS = "new_findings"
    LOOP_DETECTED = "loop_detected"


class ReflectionResult(BaseModel):
    """Result of reflecting on agent progress.

    Attributes:
        confidence_delta: Adjustment to confidence score (-1.0 to 1.0).
        assessment: Category of agent's current progress.
        guidance: Suggestions for the next iteration.
        loop_pattern: Detected loop pattern if assessment is loop_detected.
        findings_summary: Summary of new information discovered.
    """

    model_config = ConfigDict(frozen=True)

    confidence_delta: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Adjustment to confidence score",
    )
    assessment: str = Field(
        default=AssessmentCategory.ON_TRACK,
        description="Category of agent's current progress",
    )
    guidance: str | None = Field(
        default=None,
        description="Suggestions for the next iteration",
    )
    loop_pattern: str | None = Field(
        default=None,
        description="Detected loop pattern if assessment is loop_detected",
    )
    findings_summary: str | None = Field(
        default=None,
        description="Summary of new information discovered",
    )


class Reflector:
    """Evaluates agent progress after each iteration.

    Analyzes tool execution patterns, results, and state to determine
    if the agent is making progress toward its goal.

    Attributes:
        loop_threshold: Number of repeated tool calls to consider a loop.
        success_weight: Weight for successful tool executions in confidence.
        error_penalty: Penalty for failed tool executions.
        diminishing_returns: Whether to apply diminishing returns to confidence.
        min_progress_delta: Minimum confidence delta for "on_track" assessment.
    """

    def __init__(
        self,
        loop_threshold: int = 3,
        success_weight: float = 0.15,
        error_penalty: float = 0.2,
        diminishing_returns: bool = True,
        min_progress_delta: float = 0.05,
    ) -> None:
        """Initialize the Reflector.

        Args:
            loop_threshold: Number of repeated tool calls to consider a loop.
            success_weight: Base confidence increase per successful tool call.
            error_penalty: Confidence decrease per failed tool call.
            diminishing_returns: Apply diminishing returns to positive deltas.
            min_progress_delta: Minimum delta to consider making progress.
        """
        self.loop_threshold = loop_threshold
        self.success_weight = success_weight
        self.error_penalty = error_penalty
        self.diminishing_returns = diminishing_returns
        self.min_progress_delta = min_progress_delta

    def reflect(
        self,
        state: AgentState,
        iteration_executions: list[ToolExecution] | None = None,
    ) -> ReflectionResult:
        """Evaluate agent progress and produce reflection result.

        Args:
            state: Current agent state with history.
            iteration_executions: Tool executions from the current iteration.

        Returns:
            ReflectionResult with assessment and guidance.
        """
        if iteration_executions is None:
            iteration_executions = []

        # Check for loops first (highest priority)
        loop_result = self._detect_loop(state)
        if loop_result is not None:
            return loop_result

        # Analyze tool execution results
        success_count, error_count, results_content = self._analyze_executions(
            iteration_executions
        )

        # Calculate base confidence delta
        confidence_delta = self._calculate_confidence_delta(
            success_count,
            error_count,
            state.confidence,
        )

        # Determine assessment category and guidance
        assessment, guidance, findings = self._assess_progress(
            confidence_delta,
            success_count,
            error_count,
            results_content,
        )

        return ReflectionResult(
            confidence_delta=confidence_delta,
            assessment=assessment,
            guidance=guidance,
            findings_summary=findings,
        )

    def _detect_loop(self, state: AgentState) -> ReflectionResult | None:
        """Detect if the agent is stuck in a tool loop.

        Returns:
            ReflectionResult if loop detected, None otherwise.
        """
        if len(state.tool_history) < self.loop_threshold:
            return None

        recent_tools = state.tool_history[-self.loop_threshold:]
        tool_counts = Counter(recent_tools)

        # Check for single-tool loop (same tool called repeatedly)
        most_common = tool_counts.most_common(1)
        if most_common and most_common[0][1] == self.loop_threshold:
            tool_name = most_common[0][0]
            pattern = (
                f"Tool '{tool_name}' called "
                f"{self.loop_threshold} times consecutively"
            )
            return ReflectionResult(
                confidence_delta=-0.3,
                assessment=AssessmentCategory.LOOP_DETECTED,
                guidance=self._generate_loop_guidance(tool_name),
                loop_pattern=pattern,
            )

        # Check for alternating pattern (A->B->A->B)
        if self.loop_threshold >= 4 and len(state.tool_history) >= 4:
            recent = state.tool_history[-4:]
            if (
                recent[0] == recent[2]
                and recent[1] == recent[3]
                and recent[0] != recent[1]
            ):
                pattern = f"Alternating pattern: {recent[0]} <-> {recent[1]}"
                return ReflectionResult(
                    confidence_delta=-0.25,
                    assessment=AssessmentCategory.LOOP_DETECTED,
                    guidance=(
                        f"Detected alternating loop between "
                        f"'{recent[0]}' and '{recent[1]}'. "
                        "Consider a different approach or "
                        "gathering additional context."
                    ),
                    loop_pattern=pattern,
                )

        return None

    def _analyze_executions(
        self,
        executions: list[ToolExecution],
    ) -> tuple[int, int, list[str]]:
        """Analyze tool executions to count successes, errors, and gather content.

        Returns:
            Tuple of (success_count, error_count, result_contents).
        """
        success_count = 0
        error_count = 0
        results_content: list[str] = []

        for execution in executions:
            if execution.success:
                success_count += 1
                if execution.result:
                    results_content.append(execution.result)
            else:
                error_count += 1

        return success_count, error_count, results_content

    def _calculate_confidence_delta(
        self,
        success_count: int,
        error_count: int,
        current_confidence: float,
    ) -> float:
        """Calculate the confidence adjustment based on execution results.

        Args:
            success_count: Number of successful tool executions.
            error_count: Number of failed tool executions.
            current_confidence: Current confidence level (0.0 to 1.0).

        Returns:
            Confidence delta (-1.0 to 1.0).
        """
        # Base delta from successes and errors
        raw_delta = (success_count * self.success_weight) - (
            error_count * self.error_penalty
        )

        # Apply diminishing returns for positive deltas
        if self.diminishing_returns and raw_delta > 0:
            # As confidence increases, gains decrease
            effective_delta = raw_delta * (1.0 - current_confidence)
        else:
            effective_delta = raw_delta

        # Clamp to valid range
        return max(-1.0, min(1.0, effective_delta))

    def _assess_progress(
        self,
        confidence_delta: float,
        success_count: int,
        error_count: int,
        results_content: list[str],
    ) -> tuple[str, str | None, str | None]:
        """Determine assessment category and generate guidance.

        Returns:
            Tuple of (assessment, guidance, findings_summary).
        """
        # Check if we got new findings (substantial results)
        has_findings = self._has_new_findings(results_content)

        # Assess based on delta and results
        if has_findings and success_count > 0:
            findings_summary = self._summarize_findings(results_content)
            return (
                AssessmentCategory.NEW_FINDINGS,
                "New information discovered. Continue analyzing the findings.",
                findings_summary,
            )

        if confidence_delta >= self.min_progress_delta:
            return (
                AssessmentCategory.ON_TRACK,
                None,  # No guidance needed when on track
                None,
            )

        if error_count > success_count or confidence_delta < -self.min_progress_delta:
            return (
                AssessmentCategory.STUCK,
                self._generate_stuck_guidance(error_count),
                None,
            )

        # Default to on_track with minimal progress
        return (
            AssessmentCategory.ON_TRACK,
            "Progress is slow. Consider alternative approaches if no improvement.",
            None,
        )

    def _has_new_findings(self, results_content: list[str]) -> bool:
        """Determine if results contain substantial new findings."""
        if not results_content:
            return False

        # Check for non-trivial content
        total_content = "".join(results_content)
        # Heuristic: significant findings have meaningful content
        return len(total_content) > 100

    def _summarize_findings(self, results_content: list[str]) -> str:
        """Create a brief summary of findings."""
        if not results_content:
            return ""

        # Simple summary: first 200 chars of combined content
        combined = " ".join(results_content)
        if len(combined) <= 200:
            return combined
        return combined[:197] + "..."

    def _generate_loop_guidance(self, tool_name: str) -> str:
        """Generate guidance for escaping a tool loop."""
        return (
            f"The tool '{tool_name}' has been called repeatedly without progress. "
            "Consider: 1) Using a different tool to gather new information, "
            "2) Reviewing the tool arguments for potential issues, "
            "3) If the task cannot be completed, report findings and limitations."
        )

    def _generate_stuck_guidance(self, error_count: int) -> str:
        """Generate guidance when the agent is stuck."""
        if error_count > 0:
            return (
                f"Encountering errors ({error_count} in this iteration). "
                "Consider adjusting approach or trying alternative tools."
            )

        return (
            "Progress has stalled. Consider: "
            "1) Using different tools, "
            "2) Reformulating the approach, "
            "3) Breaking the problem into smaller steps."
        )


def assess_confidence(
    state: AgentState,
    tool_executions: list[ToolExecution],
    success_weight: float = 0.15,
    error_penalty: float = 0.2,
    diminishing_returns: bool = True,
) -> float:
    """Calculate confidence based on tool success rate.

    Convenience function for quick confidence assessment.

    Args:
        state: Current agent state.
        tool_executions: Tool executions from the current iteration.
        success_weight: Weight for successful tool executions.
        error_penalty: Penalty for failed tool executions.
        diminishing_returns: Apply diminishing returns for positive deltas.

    Returns:
        New confidence value (0.0 to 1.0).
    """
    reflector = Reflector(
        success_weight=success_weight,
        error_penalty=error_penalty,
        diminishing_returns=diminishing_returns,
    )
    result = reflector.reflect(state, tool_executions)
    return max(0.0, min(1.0, state.confidence + result.confidence_delta))


def detect_loop(
    state: AgentState,
    threshold: int = 3,
) -> bool:
    """Detect repeated tool calls with same arguments.

    Convenience function for quick loop detection.

    Args:
        state: Current agent state.
        threshold: Number of repeated calls to consider a loop.

    Returns:
        True if loop detected, False otherwise.
    """
    reflector = Reflector(loop_threshold=threshold)
    result = reflector.reflect(state, [])
    return result.assessment == AssessmentCategory.LOOP_DETECTED


def assess_progress(
    state: AgentState,
    tool_executions: list[ToolExecution] | None = None,
) -> str:
    """Return progress assessment category.

    Convenience function for quick progress assessment.

    Args:
        state: Current agent state.
        tool_executions: Tool executions from the current iteration.

    Returns:
        Assessment category: 'on_track', 'stuck', 'new_findings', or 'loop_detected'.
    """
    reflector = Reflector()
    result = reflector.reflect(state, tool_executions or [])
    return result.assessment
