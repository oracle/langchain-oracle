# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Confidence signal detection for early exit.

Detects heuristic markers in LLM responses that indicate high confidence,
allowing the agent to exit early and save tokens/time.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    pass


class SignalType(str, Enum):
    """Types of confidence signals detected in LLM responses."""

    EXPLICIT_CONFIDENCE = "explicit_confidence"
    VERIFICATION = "verification"
    ROOT_CAUSE = "root_cause"
    CONCLUSION = "conclusion"
    TASK_COMPLETE = "task_complete"
    SOLUTION_FOUND = "solution_found"


class ConfidenceSignal(BaseModel):
    """A detected confidence signal.

    Attributes:
        signal_type: Type of confidence signal.
        text_match: The matched text pattern.
        weight: Signal weight (contribution to confidence).
        iteration: Iteration when signal was detected.
    """

    model_config = ConfigDict(frozen=True)

    signal_type: SignalType
    text_match: str
    weight: float
    iteration: int


# Heuristic patterns with their types and weights
CONFIDENCE_PATTERNS: dict[str, tuple[SignalType, float]] = {
    # Explicit confidence markers (weight: 0.15-0.20)
    "confident that": (SignalType.EXPLICIT_CONFIDENCE, 0.20),
    "i am confident": (SignalType.EXPLICIT_CONFIDENCE, 0.20),
    "with confidence": (SignalType.EXPLICIT_CONFIDENCE, 0.15),
    "high confidence": (SignalType.EXPLICIT_CONFIDENCE, 0.20),
    # Verification markers (weight: 0.15)
    "verified": (SignalType.VERIFICATION, 0.15),
    "confirmed": (SignalType.VERIFICATION, 0.15),
    "validated": (SignalType.VERIFICATION, 0.15),
    "checked and": (SignalType.VERIFICATION, 0.10),
    # Root cause markers (weight: 0.20-0.25)
    "root cause": (SignalType.ROOT_CAUSE, 0.25),
    "the cause is": (SignalType.ROOT_CAUSE, 0.20),
    "the issue is": (SignalType.ROOT_CAUSE, 0.20),
    "the problem is": (SignalType.ROOT_CAUSE, 0.20),
    "identified the": (SignalType.ROOT_CAUSE, 0.15),
    # Conclusion markers (weight: 0.15-0.20)
    "conclusive": (SignalType.CONCLUSION, 0.20),
    "definitive": (SignalType.CONCLUSION, 0.15),
    "certain that": (SignalType.CONCLUSION, 0.15),
    "clear that": (SignalType.CONCLUSION, 0.10),
    # Task completion markers (weight: 0.20-0.25)
    "successfully completed": (SignalType.TASK_COMPLETE, 0.25),
    "task is complete": (SignalType.TASK_COMPLETE, 0.25),
    "all steps completed": (SignalType.TASK_COMPLETE, 0.25),
    "finished implementing": (SignalType.TASK_COMPLETE, 0.20),
    "implementation complete": (SignalType.TASK_COMPLETE, 0.20),
    # Solution markers (weight: 0.15-0.20)
    "the solution is": (SignalType.SOLUTION_FOUND, 0.20),
    "this fixes": (SignalType.SOLUTION_FOUND, 0.15),
    "resolved the": (SignalType.SOLUTION_FOUND, 0.15),
    "the fix is": (SignalType.SOLUTION_FOUND, 0.15),
}

# Default thresholds
DEFAULT_EARLY_EXIT_THRESHOLD = 0.9
DEFAULT_MIN_ITERATIONS = 2
DEFAULT_BASE_CONFIDENCE = 0.5


def detect_confidence_signals(
    content: str,
    iteration: int,
) -> list[ConfidenceSignal]:
    """Detect confidence markers in LLM response.

    Args:
        content: LLM response text to analyze.
        iteration: Current iteration number.

    Returns:
        List of detected confidence signals.
    """
    signals: list[ConfidenceSignal] = []
    content_lower = content.lower()

    for pattern, (signal_type, weight) in CONFIDENCE_PATTERNS.items():
        if pattern in content_lower:
            signals.append(
                ConfidenceSignal(
                    signal_type=signal_type,
                    text_match=pattern,
                    weight=weight,
                    iteration=iteration,
                )
            )

    return signals


def compute_accumulated_confidence(
    signals: list[ConfidenceSignal],
    base_confidence: float = DEFAULT_BASE_CONFIDENCE,
) -> float:
    """Compute accumulated confidence from signals.

    Uses diminishing returns for multiple signals of the same type.

    Args:
        signals: List of all confidence signals detected so far.
        base_confidence: Starting confidence level.

    Returns:
        Accumulated confidence score (0.0 to 1.0).
    """
    if not signals:
        return base_confidence

    by_type: dict[SignalType, list[ConfidenceSignal]] = {}
    for sig in signals:
        by_type.setdefault(sig.signal_type, []).append(sig)

    total_boost = 0.0
    for signal_type, type_signals in by_type.items():
        type_boost = 0.0
        for i, sig in enumerate(type_signals):
            type_boost += sig.weight * (0.5**i)
        total_boost += type_boost

    return min(1.0, base_confidence + total_boost)


def should_early_exit(
    accumulated_confidence: float,
    iteration: int,
    min_iterations: int = DEFAULT_MIN_ITERATIONS,
    threshold: float = DEFAULT_EARLY_EXIT_THRESHOLD,
) -> bool:
    """Check if early exit conditions are met.

    Args:
        accumulated_confidence: Current confidence score.
        iteration: Current iteration.
        min_iterations: Minimum iterations before exit allowed.
        threshold: Confidence threshold for exit.

    Returns:
        True if should exit early.
    """
    return accumulated_confidence >= threshold and iteration >= min_iterations
