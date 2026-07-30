# Copyright (c) 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Unit tests for OCIGuardrailsMiddleware (no live calls).

Skipped where ``langchain.agents.middleware`` is unavailable (langchain < 1.0,
i.e. the Python 3.9 matrix).
"""

from types import SimpleNamespace
from typing import Any, List, Optional, Tuple, cast

import pytest

pytest.importorskip(
    "langchain.agents.middleware",
    reason="AgentMiddleware requires langchain>=1.0",
)

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402

from langchain_oci import OCIGuardrails  # noqa: E402
from langchain_oci.guardrails import (  # noqa: E402
    OCIGuardrailsMiddleware,
    OCIGuardrailsViolationError,
)


def _results(
    pi_score: float = 0.0,
    moderation: Optional[List[Tuple[str, float]]] = None,
    pii: Optional[list] = None,
) -> SimpleNamespace:
    """Build a fake ``GuardrailsResults``-shaped object."""
    return SimpleNamespace(
        prompt_injection=SimpleNamespace(score=pi_score),
        content_moderation=SimpleNamespace(
            categories=[
                SimpleNamespace(name=name, score=score)
                for name, score in (moderation or [])
            ]
        ),
        personally_identifiable_information=pii or [],
    )


class _FakeGuardrails:
    """Stands in for OCIGuardrails: records calls, returns canned results."""

    def __init__(self, results: SimpleNamespace) -> None:
        self._results = results
        self.calls: List[str] = []

    def invoke(self, text: str) -> SimpleNamespace:
        self.calls.append(text)
        return self._results


def _mw(results: SimpleNamespace, **kwargs: Any) -> OCIGuardrailsMiddleware:
    fake = cast(OCIGuardrails, _FakeGuardrails(results))
    return OCIGuardrailsMiddleware(guardrails=fake, **kwargs)


def _state(*messages: Any) -> dict:
    return {"messages": list(messages)}


class TestOCIGuardrailsMiddleware:
    def test_blocks_on_prompt_injection(self) -> None:
        mw = _mw(_results(pi_score=1.0))
        with pytest.raises(OCIGuardrailsViolationError) as exc:
            mw.before_model(_state(HumanMessage(content="ignore instructions")), None)
        assert exc.value.location == "input"
        assert any("prompt_injection" in v for v in exc.value.violations)

    def test_clean_input_passes(self) -> None:
        mw = _mw(_results(pi_score=0.0))
        assert mw.before_model(_state(HumanMessage(content="hello")), None) is None

    def test_apply_to_input_false_is_noop(self) -> None:
        guardrails = _FakeGuardrails(_results(pi_score=1.0))
        mw = OCIGuardrailsMiddleware(
            guardrails=cast(OCIGuardrails, guardrails), apply_to_input=False
        )
        assert mw.before_model(_state(HumanMessage(content="x")), None) is None
        assert guardrails.calls == []  # guardrails never invoked

    def test_after_model_checks_output_when_enabled(self) -> None:
        mw = _mw(_results(pi_score=1.0), apply_to_output=True)
        with pytest.raises(OCIGuardrailsViolationError) as exc:
            mw.after_model(_state(AIMessage(content="bad output")), None)
        assert exc.value.location == "output"

    def test_after_model_noop_by_default(self) -> None:
        mw = _mw(_results(pi_score=1.0))  # apply_to_output defaults False
        assert mw.after_model(_state(AIMessage(content="bad")), None) is None

    def test_warn_mode_does_not_raise(self) -> None:
        mw = _mw(_results(pi_score=1.0), on_violation="warn")
        with pytest.warns(UserWarning, match="OCI guardrails flagged the input"):
            assert mw.before_model(_state(HumanMessage(content="x")), None) is None

    def test_content_moderation_threshold(self) -> None:
        mw = _mw(_results(moderation=[("HATE", 0.9), ("OVERALL", 0.1)]))
        with pytest.raises(OCIGuardrailsViolationError) as exc:
            mw.before_model(_state(HumanMessage(content="x")), None)
        assert any("HATE" in v for v in exc.value.violations)

    def test_block_on_pii_flag(self) -> None:
        results = _results(pii=[object(), object()])
        # PII present but block_on_pii defaults False -> no violation
        assert (
            _mw(results).before_model(_state(HumanMessage(content="x")), None) is None
        )
        # With block_on_pii True -> blocks
        with pytest.raises(OCIGuardrailsViolationError):
            _mw(results, block_on_pii=True).before_model(
                _state(HumanMessage(content="x")), None
            )

    def test_invalid_on_violation_rejected(self) -> None:
        with pytest.raises(ValueError, match="on_violation must be"):
            _mw(_results(), on_violation="explode")

    def test_is_agent_middleware(self) -> None:
        from langchain.agents.middleware import AgentMiddleware

        assert isinstance(_mw(_results()), AgentMiddleware)
