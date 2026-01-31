# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for reasoning_content extraction from OCI reasoning models.

Reasoning models (e.g., xAI Grok-3-mini, OpenAI o1/GPT-oss) return a
``reasoning_content`` field containing chain-of-thought reasoning.
These tests verify that the field is correctly surfaced in
``generation_info`` and ``additional_kwargs`` on the AIMessage.

Related upstream fix for langchain-openai:
https://github.com/langchain-ai/langchain/pull/34705
"""

from typing import Optional
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import HumanMessage

from langchain_oci.chat_models.oci_generative_ai import ChatOCIGenAI


class MockResponseDict(dict):
    def __getattr__(self, val):
        return self.get(val)


def _make_generic_response(
    *,
    text: str = "The answer is 42.",
    reasoning_content: Optional[str] = None,
    finish_reason: str = "completed",
    model_id: str = "xai.grok-3-mini",
) -> MockResponseDict:
    """Build a mock OCI Generic API response with optional reasoning."""
    message = MockResponseDict(
        {
            "role": "ASSISTANT",
            "content": [MockResponseDict({"text": text, "type": "TEXT"})],
            "tool_calls": None,
        }
    )
    if reasoning_content is not None:
        message["reasoning_content"] = reasoning_content

    return MockResponseDict(
        {
            "status": 200,
            "data": MockResponseDict(
                {
                    "chat_response": MockResponseDict(
                        {
                            "api_format": "GENERIC",
                            "choices": [
                                MockResponseDict(
                                    {
                                        "message": message,
                                        "finish_reason": finish_reason,
                                    }
                                )
                            ],
                            "time_created": "2026-01-29T10:00:00Z",
                            "usage": MockResponseDict(
                                {
                                    "prompt_tokens": 20,
                                    "completion_tokens": 15,
                                    "total_tokens": 35,
                                }
                            ),
                        }
                    ),
                    "model_id": model_id,
                    "model_version": "1.0.0",
                }
            ),
            "request_id": "test-req-001",
            "headers": MockResponseDict({"content-length": "200"}),
        }
    )


def _make_empty_choices_response() -> MockResponseDict:
    """Build a mock response with empty choices list."""
    return MockResponseDict(
        {
            "status": 200,
            "data": MockResponseDict(
                {
                    "chat_response": MockResponseDict(
                        {
                            "api_format": "GENERIC",
                            "choices": [],
                            "time_created": "2026-01-29T10:00:00Z",
                            "usage": None,
                        }
                    ),
                    "model_id": "meta.llama-3.3-70b-instruct",
                    "model_version": "1.0.0",
                }
            ),
            "request_id": "test-req-002",
            "headers": MockResponseDict({"content-length": "100"}),
        }
    )


def _make_null_usage_response() -> MockResponseDict:
    """Build a mock response where usage token fields are None."""
    return MockResponseDict(
        {
            "status": 200,
            "data": MockResponseDict(
                {
                    "chat_response": MockResponseDict(
                        {
                            "api_format": "GENERIC",
                            "choices": [
                                MockResponseDict(
                                    {
                                        "message": MockResponseDict(
                                            {
                                                "role": "ASSISTANT",
                                                "content": [
                                                    MockResponseDict(
                                                        {
                                                            "text": "Hi",
                                                            "type": "TEXT",
                                                        }
                                                    )
                                                ],
                                                "tool_calls": None,
                                            }
                                        ),
                                        "finish_reason": "completed",
                                    }
                                )
                            ],
                            "time_created": "2026-01-29T10:00:00Z",
                            "usage": MockResponseDict(
                                {
                                    "prompt_tokens": None,
                                    "completion_tokens": None,
                                    "total_tokens": None,
                                }
                            ),
                        }
                    ),
                    "model_id": "meta.llama-3.3-70b-instruct",
                    "model_version": "1.0.0",
                }
            ),
            "request_id": "test-req-003",
            "headers": MockResponseDict({"content-length": "100"}),
        }
    )


@pytest.mark.requires("oci")
class TestReasoningContentExtraction:
    """Verify reasoning_content is surfaced from OCI reasoning models."""

    def test_reasoning_content_in_generation_info(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """reasoning_content appears in generation_info when present."""
        oci_client = MagicMock()
        llm = ChatOCIGenAI(model_id="xai.grok-3-mini", client=oci_client)

        reasoning_text = (
            "The user asked 7 * 8. Let me compute: 7 * 8 = 56. The answer is 56."
        )
        oci_client.chat.return_value = _make_generic_response(
            text="56",
            reasoning_content=reasoning_text,
        )

        result = llm.invoke([HumanMessage(content="What is 7 * 8?")])
        assert result.additional_kwargs["reasoning_content"] == reasoning_text

    def test_reasoning_content_absent_for_standard_models(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Standard models without reasoning_content don't add the key."""
        oci_client = MagicMock()
        llm = ChatOCIGenAI(model_id="meta.llama-3.3-70b-instruct", client=oci_client)

        oci_client.chat.return_value = _make_generic_response(
            text="56",
            reasoning_content=None,
            model_id="meta.llama-3.3-70b-instruct",
        )

        result = llm.invoke([HumanMessage(content="What is 7 * 8?")])
        assert "reasoning_content" not in result.additional_kwargs


@pytest.mark.requires("oci")
class TestNullGuards:
    """Verify GenericProvider handles empty/null choices gracefully."""

    def test_empty_choices_returns_empty_text(self) -> None:
        """chat_response_to_text returns '' when choices is empty."""
        from langchain_oci.chat_models.providers.generic import (
            GenericProvider,
        )

        provider = GenericProvider()
        response = _make_empty_choices_response()
        assert provider.chat_response_to_text(response) == ""

    def test_empty_choices_returns_no_tool_calls(self) -> None:
        """chat_tool_calls returns [] when choices is empty."""
        from langchain_oci.chat_models.providers.generic import (
            GenericProvider,
        )

        provider = GenericProvider()
        response = _make_empty_choices_response()
        assert provider.chat_tool_calls(response) == []

    def test_empty_choices_generation_info_has_null_finish(self) -> None:
        """chat_generation_info returns finish_reason=None for empty."""
        from langchain_oci.chat_models.providers.generic import (
            GenericProvider,
        )

        provider = GenericProvider()
        response = _make_empty_choices_response()
        info = provider.chat_generation_info(response)
        assert info["finish_reason"] is None

    def test_null_usage_tokens_default_to_zero(self) -> None:
        """Usage tokens that are None should resolve to 0."""
        oci_client = MagicMock()
        llm = ChatOCIGenAI(model_id="meta.llama-3.3-70b-instruct", client=oci_client)

        oci_client.chat.return_value = _make_null_usage_response()
        result = llm.invoke([HumanMessage(content="Hi")])

        if hasattr(result, "usage_metadata") and result.usage_metadata is not None:
            assert result.usage_metadata["input_tokens"] == 0
            assert result.usage_metadata["output_tokens"] == 0
            assert result.usage_metadata["total_tokens"] == 0
