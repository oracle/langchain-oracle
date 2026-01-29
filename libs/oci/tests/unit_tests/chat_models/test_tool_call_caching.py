# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Test that chat_tool_calls() is invoked exactly once per _generate() call."""

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from pytest import MonkeyPatch

from langchain_oci.chat_models.oci_generative_ai import ChatOCIGenAI


class MockResponseDict(dict):
    def __getattr__(self, val):
        return self.get(val)


def _make_generic_tool_response(model_id: str) -> MockResponseDict:
    """Build a mock Generic (Meta/Gemini/Grok) response with a tool call."""
    return MockResponseDict(
        {
            "status": 200,
            "data": MockResponseDict(
                {
                    "chat_response": MockResponseDict(
                        {
                            "choices": [
                                MockResponseDict(
                                    {
                                        "message": MockResponseDict(
                                            {
                                                "content": [
                                                    MockResponseDict(
                                                        {"text": "Let me check."}
                                                    )
                                                ],
                                                "tool_calls": [
                                                    MockResponseDict(
                                                        {
                                                            "type": "FUNCTION",
                                                            "id": "call_1",
                                                            "name": "get_weather",
                                                            "arguments": '{"location": "NYC"}',  # noqa: E501
                                                            "attribute_map": {
                                                                "id": "id",
                                                                "type": "type",
                                                                "name": "name",
                                                                "arguments": "arguments",  # noqa: E501
                                                            },
                                                        }
                                                    )
                                                ],
                                            }
                                        ),
                                        "finish_reason": "completed",
                                    }
                                )
                            ],
                            "time_created": "2025-01-01T00:00:00+00:00",
                            "usage": MockResponseDict(
                                {
                                    "total_tokens": 100,
                                    "prompt_tokens": 50,
                                    "completion_tokens": 50,
                                }
                            ),
                        }
                    ),
                    "model_id": model_id,
                    "model_version": "1.0",
                }
            ),
            "request_id": "req-1",
            "headers": MockResponseDict({"content-length": "100"}),
        }
    )


def _make_cohere_tool_response(model_id: str) -> MockResponseDict:
    """Build a mock Cohere response with a tool call."""
    return MockResponseDict(
        {
            "status": 200,
            "data": MockResponseDict(
                {
                    "chat_response": MockResponseDict(
                        {
                            "text": "Let me check.",
                            "tool_calls": [
                                MockResponseDict(
                                    {
                                        "type": "FUNCTION",
                                        "id": "call_c1",
                                        "name": "get_weather",
                                        "parameters": {"location": "NYC"},
                                        "attribute_map": {
                                            "id": "id",
                                            "type": "type",
                                            "name": "name",
                                            "parameters": "parameters",
                                        },
                                    }
                                )
                            ],
                            "documents": None,
                            "citations": None,
                            "search_queries": None,
                            "is_search_required": None,
                            "finish_reason": "COMPLETE",
                            "usage": MockResponseDict(
                                {
                                    "total_tokens": 100,
                                    "prompt_tokens": 50,
                                    "completion_tokens": 50,
                                }
                            ),
                        }
                    ),
                    "model_id": model_id,
                    "model_version": "1.0",
                }
            ),
            "request_id": "req-1",
            "headers": MockResponseDict({"content-length": "100"}),
        }
    )


@pytest.mark.requires("oci")
@pytest.mark.parametrize(
    "model_id",
    ["meta.llama-3.3-70b-instruct", "cohere.command-r-plus"],
)
def test_chat_tool_calls_invoked_once(monkeypatch: MonkeyPatch, model_id: str) -> None:
    """Verify chat_tool_calls() is called exactly once per _generate()."""
    oci_gen_ai_client = MagicMock()
    llm = ChatOCIGenAI(model_id=model_id, client=oci_gen_ai_client)

    if model_id.startswith("cohere"):
        mock_response = _make_cohere_tool_response(model_id)
    else:
        mock_response = _make_generic_tool_response(model_id)

    monkeypatch.setattr(llm.client, "chat", lambda *a, **kw: mock_response)

    original = llm._provider.chat_tool_calls
    call_count = 0

    def counting_wrapper(response):
        nonlocal call_count
        call_count += 1
        return original(response)

    monkeypatch.setattr(llm._provider, "chat_tool_calls", counting_wrapper)

    result = llm._generate([HumanMessage(content="Weather?")])
    msg = result.generations[0].message
    gen_info = result.generations[0].generation_info

    assert call_count == 1, f"chat_tool_calls() called {call_count} times, expected 1"
    assert gen_info is not None
    assert "tool_calls" in gen_info
    assert isinstance(msg, AIMessage)
    assert len(msg.tool_calls) == 1
    assert msg.tool_calls[0]["name"] == "get_weather"
