# Copyright (c) 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""LangChain agent middleware that applies OCI guardrails in the agent loop.

This is the industry-standard way to wire guardrails into a LangChain agent
(``create_agent``): a middleware that runs before and/or after the model, in the
same spirit as LangChain's built-in :class:`~langchain.agents.middleware.PIIMiddleware`.
On a violation it either raises (``on_violation="block"``) or warns
(``on_violation="warn"``).

Requires ``langchain >= 1.0`` (``AgentMiddleware`` / ``create_agent``). For
standalone or LCEL-chain use — where you want the raw guardrail findings rather
than agent-loop enforcement — use :class:`~langchain_oci.guardrails.OCIGuardrails`
directly.

Example:
    .. code-block:: python

        from langchain.agents import create_agent
        from langchain_oci import ChatOCIGenAI, OCIGuardrailsMiddleware

        agent = create_agent(
            model=ChatOCIGenAI(model_id="meta.llama-3.3-70b-instruct", ...),
            tools=[...],
            middleware=[
                OCIGuardrailsMiddleware(
                    compartment_id="ocid1.compartment.oc1..example",
                    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
                    apply_to_output=True,
                ),
            ],
        )
"""

from __future__ import annotations

import warnings
from typing import Any, List, Optional

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from langchain_oci.guardrails.oci_guardrails import OCIGuardrails


class OCIGuardrailsViolationError(Exception):
    """Raised when an OCI guardrail blocks a message in the agent loop."""

    def __init__(self, violations: List[str], location: str) -> None:
        self.violations = violations
        self.location = location
        super().__init__(
            f"OCI guardrails blocked the {location}: " + "; ".join(violations)
        )


def _message_text(message: BaseMessage) -> str:
    """Return the plain-text content of a message (handles block content)."""
    content = message.content
    if isinstance(content, str):
        return content
    parts: List[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)


class OCIGuardrailsMiddleware(AgentMiddleware):
    """Apply OCI Generative AI guardrails before/after the model in ``create_agent``.

    Mirrors LangChain's ``PIIMiddleware`` pattern: it inspects messages in the
    agent loop and, when a guardrail is triggered, either raises
    :class:`OCIGuardrailsViolationError` (``on_violation="block"``) or emits a
    warning (``on_violation="warn"``).

    A ``violation`` is any of:

    - prompt-injection score >= ``prompt_injection_threshold``
    - a content-moderation category score >= ``content_moderation_threshold``
    - (when ``block_on_pii`` is True) any detected PII entity
    """

    def __init__(
        self,
        guardrails: Optional[OCIGuardrails] = None,
        *,
        apply_to_input: bool = True,
        apply_to_output: bool = False,
        prompt_injection_threshold: float = 0.5,
        content_moderation_threshold: float = 0.5,
        block_on_pii: bool = False,
        on_violation: str = "block",
        **guardrails_kwargs: Any,
    ) -> None:
        super().__init__()
        if on_violation not in ("block", "warn"):
            raise ValueError("on_violation must be 'block' or 'warn'.")
        self.guardrails = guardrails or OCIGuardrails(**guardrails_kwargs)
        self.apply_to_input = apply_to_input
        self.apply_to_output = apply_to_output
        self.prompt_injection_threshold = prompt_injection_threshold
        self.content_moderation_threshold = content_moderation_threshold
        self.block_on_pii = block_on_pii
        self.on_violation = on_violation

    def _violations(self, results: Any) -> List[str]:
        """Summarize threshold-exceeding findings in a ``GuardrailsResults``."""
        violations: List[str] = []

        prompt_injection = getattr(results, "prompt_injection", None)
        score = getattr(prompt_injection, "score", None)
        if score is not None and score >= self.prompt_injection_threshold:
            violations.append(f"prompt_injection score={score:.2f}")

        moderation = getattr(results, "content_moderation", None)
        for category in getattr(moderation, "categories", None) or []:
            cat_score = getattr(category, "score", None) or 0.0
            if cat_score >= self.content_moderation_threshold:
                name = getattr(category, "name", "?")
                violations.append(f"content_moderation:{name} score={cat_score:.2f}")

        if self.block_on_pii:
            pii = getattr(results, "personally_identifiable_information", None) or []
            if len(pii) > 0:
                violations.append(f"pii entities={len(pii)}")

        return violations

    def _check(self, text: str, location: str) -> None:
        if not text:
            return
        results = self.guardrails.invoke(text)
        violations = self._violations(results)
        if not violations:
            return
        if self.on_violation == "block":
            raise OCIGuardrailsViolationError(violations, location)
        warnings.warn(
            f"OCI guardrails flagged the {location}: " + "; ".join(violations),
            stacklevel=2,
        )

    def before_model(self, state: Any, runtime: Any = None) -> Optional[dict]:
        """Check the most recent user message before the model is called."""
        if not self.apply_to_input:
            return None
        for message in reversed(state.get("messages") or []):
            if isinstance(message, HumanMessage):
                self._check(_message_text(message), "input")
                break
        return None

    def after_model(self, state: Any, runtime: Any = None) -> Optional[dict]:
        """Check the most recent model message after the model is called."""
        if not self.apply_to_output:
            return None
        for message in reversed(state.get("messages") or []):
            if isinstance(message, AIMessage):
                self._check(_message_text(message), "output")
                break
        return None
