# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Integration tests for reasoning_content extraction.

Verifies that reasoning models (xAI Grok-3-mini, OpenAI GPT-oss) return
chain-of-thought reasoning in ``additional_kwargs["reasoning_content"]``,
and that standard models (Meta Llama, Cohere) do not.

Also tests null-guard robustness on real API responses.

Setup:
    export OCI_COMPARTMENT_ID=<your-compartment-id>
    export OCI_CONFIG_PROFILE=API_KEY_AUTH  # or DEFAULT
    export OCI_AUTH_TYPE=API_KEY            # or SECURITY_TOKEN

Run:
    pytest tests/integration_tests/chat_models/ -k reasoning -v -s
"""

import os

import pytest
from langchain_core.messages import HumanMessage

from langchain_oci.chat_models import ChatOCIGenAI

# Models known to return reasoning_content
REASONING_MODELS = [
    "xai.grok-3-mini-fast",
    "openai.gpt-oss-120b",
]

# Models that do NOT return reasoning_content
STANDARD_MODELS = [
    "meta.llama-3.3-70b-instruct",
    "cohere.command-r-08-2024",
]

SERVICE_ENDPOINT = "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"


def _get_compartment_id() -> str:
    cid = os.environ.get("OCI_COMPARTMENT_ID", "")
    if not cid:
        pytest.skip("OCI_COMPARTMENT_ID not set")
    return cid


def _make_llm(model_id: str) -> ChatOCIGenAI:
    # OpenAI models require 'max_completion_tokens' instead of 'max_tokens'
    if model_id.startswith("openai."):
        model_kwargs = {"max_completion_tokens": 100, "temperature": 1.0}
    else:
        model_kwargs = {"max_tokens": 100, "temperature": 0.0}

    return ChatOCIGenAI(
        model_id=model_id,
        compartment_id=_get_compartment_id(),
        service_endpoint=SERVICE_ENDPOINT,
        auth_profile=os.environ.get("OCI_CONFIG_PROFILE", "DEFAULT"),
        auth_type=os.environ.get("OCI_AUTH_TYPE", "API_KEY"),
        model_kwargs=model_kwargs,
    )


@pytest.mark.requires("oci")
class TestReasoningContentIntegration:
    """Verify reasoning_content extraction against live OCI endpoints."""

    @pytest.mark.parametrize("model_id", REASONING_MODELS)
    def test_reasoning_model_returns_reasoning_content(self, model_id: str) -> None:
        """Reasoning models should populate reasoning_content."""
        llm = _make_llm(model_id)
        result = llm.invoke([HumanMessage(content="What is 17 * 23?")])

        reasoning = result.additional_kwargs.get("reasoning_content")
        assert reasoning is not None, (
            f"{model_id}: expected reasoning_content in "
            f"additional_kwargs, got keys: "
            f"{list(result.additional_kwargs.keys())}"
        )
        assert len(reasoning) > 10, (
            f"{model_id}: reasoning_content too short: {reasoning!r}"
        )

        # The actual answer should also be present in content
        assert result.content, f"{model_id}: content should not be empty"

        print(f"\n  {model_id}:")  # noqa: T201
        print(f"    content: {str(result.content)[:80]}...")  # noqa: T201
        print(  # noqa: T201
            f"    reasoning ({len(reasoning)} chars): {reasoning[:100]}..."
        )

    @pytest.mark.parametrize("model_id", STANDARD_MODELS)
    def test_standard_model_has_no_reasoning_content(self, model_id: str) -> None:
        """Standard models should NOT have reasoning_content."""
        llm = _make_llm(model_id)
        result = llm.invoke([HumanMessage(content="What is 17 * 23?")])

        reasoning = result.additional_kwargs.get("reasoning_content")
        assert reasoning is None, (
            f"{model_id}: unexpected reasoning_content: {reasoning!r}"
        )
        assert result.content, f"{model_id}: content should not be empty"

        print(f"\n  {model_id}:")  # noqa: T201
        print(f"    content: {str(result.content)[:80]}...")  # noqa: T201
        print("    reasoning_content: (absent, as expected)")  # noqa: T201

    def test_usage_metadata_with_null_tokens(self) -> None:
        """Usage metadata should handle None token fields gracefully."""
        llm = _make_llm("meta.llama-3.3-70b-instruct")
        result = llm.invoke([HumanMessage(content="Say hello")])

        # usage_metadata should exist and have non-negative ints
        if hasattr(result, "usage_metadata") and result.usage_metadata is not None:
            assert result.usage_metadata["input_tokens"] >= 0
            assert result.usage_metadata["output_tokens"] >= 0
            assert result.usage_metadata["total_tokens"] >= 0

            print(  # noqa: T201
                f"\n  usage: {dict(result.usage_metadata)}"
            )
