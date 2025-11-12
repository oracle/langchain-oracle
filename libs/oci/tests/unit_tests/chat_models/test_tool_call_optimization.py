# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Unit tests for tool call optimization."""

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import HumanMessage

from langchain_oci.chat_models.oci_generative_ai import ChatOCIGenAI


class MockResponseDict(dict):
    def __getattr__(self, val):  # type: ignore[no-untyped-def]
        return self.get(val)


@pytest.mark.requires("oci")
def test_meta_tool_call_optimization() -> None:
    """Test that tool calls are formatted once and cached for Meta models."""
    oci_gen_ai_client = MagicMock()

    # Mock response with tool call
    def mocked_response(*args):  # type: ignore[no-untyped-def]
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
                                                    "name": None,
                                                    "content": [
                                                        MockResponseDict(
                                                            {
                                                                "text": "",
                                                                "type": "TEXT",
                                                            }
                                                        )
                                                    ],
                                                    "tool_calls": [
                                                        MockResponseDict(
                                                            {
                                                                "id": "test_id_123",
                                                                "type": "FUNCTION",
                                                                "function": MockResponseDict(
                                                                    {
                                                                        "name": "get_weather",
                                                                        "arguments": '{"location": "San Francisco"}',
                                                                    }
                                                                ),
                                                            }
                                                        )
                                                    ],
                                                }
                                            ),
                                            "finish_reason": "TOOL_CALLS",
                                            "logprobs": None,
                                            "index": 0,
                                        }
                                    )
                                ],
                                "time_created": "2024-01-01T00:00:00Z",
                                "usage": MockResponseDict(
                                    {
                                        "input_tokens": 100,
                                        "output_tokens": 50,
                                        "total_tokens": 150,
                                    }
                                ),
                            }
                        ),
                        "model_id": "meta.llama-3.3-70b-instruct",
                        "model_version": "1.0.0",
                    }
                ),
                "request_id": "test_request_123",
                "headers": MockResponseDict(
                    {
                        "content-length": "500",
                    }
                ),
            }
        )

    oci_gen_ai_client.chat.side_effect = mocked_response

    # Create LLM with mocked client
    llm = ChatOCIGenAI(model_id="meta.llama-3.3-70b-instruct", client=oci_gen_ai_client)

    # Define a simple tool
    def get_weather(location: str) -> str:
        """Get weather for a location."""
        return f"Weather in {location}"

    # Bind tools
    llm_with_tools = llm.bind_tools([get_weather])

    # Invoke
    response = llm_with_tools.invoke([HumanMessage(content="What's the weather in SF?")])

    # Verify tool_calls field is populated
    assert len(response.tool_calls) == 1, "Should have one tool call"
    tool_call = response.tool_calls[0]
    assert tool_call["name"] == "get_weather"
    assert tool_call["args"] == {"location": "San Francisco"}
    assert "id" in tool_call

    # Verify additional_kwargs contains formatted tool calls
    assert "tool_calls" in response.additional_kwargs, "Should have tool_calls in additional_kwargs"
    additional_tool_calls = response.additional_kwargs["tool_calls"]
    assert len(additional_tool_calls) == 1
    assert additional_tool_calls[0]["type"] == "function"
    assert additional_tool_calls[0]["function"]["name"] == "get_weather"
    assert "location" in str(additional_tool_calls[0]["function"]["arguments"])


@pytest.mark.requires("oci")
def test_cohere_tool_call_optimization() -> None:
    """Test that tool calls are formatted once and cached for Cohere models."""
    oci_gen_ai_client = MagicMock()

    # Mock response with tool call
    def mocked_response(*args):  # type: ignore[no-untyped-def]
        return MockResponseDict(
            {
                "status": 200,
                "data": MockResponseDict(
                    {
                        "chat_response": MockResponseDict(
                            {
                                "text": "",
                                "finish_reason": "TOOL_CALL",
                                "tool_calls": [
                                    MockResponseDict(
                                        {
                                            "name": "get_weather",
                                            "parameters": {"location": "London"},
                                        }
                                    )
                                ],
                                "usage": MockResponseDict(
                                    {
                                        "total_tokens": 100,
                                    }
                                ),
                            }
                        ),
                        "model_id": "cohere.command-r-plus",
                        "model_version": "1.0.0",
                    }
                ),
                "request_id": "test_request_456",
                "headers": MockResponseDict(
                    {
                        "content-length": "300",
                    }
                ),
            }
        )

    oci_gen_ai_client.chat.side_effect = mocked_response

    # Create LLM with mocked client
    llm = ChatOCIGenAI(model_id="cohere.command-r-plus", client=oci_gen_ai_client)

    # Define a simple tool
    def get_weather(location: str) -> str:
        """Get weather for a location."""
        return f"Weather in {location}"

    # Bind tools
    llm_with_tools = llm.bind_tools([get_weather])

    # Invoke
    response = llm_with_tools.invoke([HumanMessage(content="What's the weather in London?")])

    # Verify tool_calls field is populated
    assert len(response.tool_calls) == 1, "Should have one tool call"
    tool_call = response.tool_calls[0]
    assert tool_call["name"] == "get_weather"
    assert tool_call["args"] == {"location": "London"}
    assert "id" in tool_call
    assert isinstance(tool_call["id"], str)
    assert len(tool_call["id"]) > 0, "Tool call ID should not be empty"

    # Verify additional_kwargs contains formatted tool calls
    assert "tool_calls" in response.additional_kwargs, "Should have tool_calls in additional_kwargs"
    additional_tool_calls = response.additional_kwargs["tool_calls"]
    assert len(additional_tool_calls) == 1
    assert additional_tool_calls[0]["type"] == "function"
    assert additional_tool_calls[0]["function"]["name"] == "get_weather"


@pytest.mark.requires("oci")
def test_multiple_tool_calls_optimization() -> None:
    """Test optimization with multiple tool calls."""
    oci_gen_ai_client = MagicMock()

    # Mock response with multiple tool calls
    def mocked_response(*args):  # type: ignore[no-untyped-def]
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
                                                                "text": "",
                                                                "type": "TEXT",
                                                            }
                                                        )
                                                    ],
                                                    "tool_calls": [
                                                        MockResponseDict(
                                                            {
                                                                "id": "call_1",
                                                                "type": "FUNCTION",
                                                                "function": MockResponseDict(
                                                                    {
                                                                        "name": "get_weather",
                                                                        "arguments": '{"location": "Tokyo"}',
                                                                    }
                                                                ),
                                                            }
                                                        ),
                                                        MockResponseDict(
                                                            {
                                                                "id": "call_2",
                                                                "type": "FUNCTION",
                                                                "function": MockResponseDict(
                                                                    {
                                                                        "name": "get_population",
                                                                        "arguments": '{"city": "Tokyo"}',
                                                                    }
                                                                ),
                                                            }
                                                        ),
                                                    ],
                                                }
                                            ),
                                            "finish_reason": "TOOL_CALLS",
                                            "index": 0,
                                        }
                                    )
                                ],
                                "usage": MockResponseDict(
                                    {
                                        "total_tokens": 200,
                                    }
                                ),
                            }
                        ),
                        "model_id": "meta.llama-3.3-70b-instruct",
                        "model_version": "1.0.0",
                    }
                ),
                "request_id": "test_request_789",
            }
        )

    oci_gen_ai_client.chat.side_effect = mocked_response

    # Create LLM with mocked client
    llm = ChatOCIGenAI(model_id="meta.llama-3.3-70b-instruct", client=oci_gen_ai_client)

    # Define tools
    def get_weather(location: str) -> str:
        """Get weather."""
        return f"Weather in {location}"

    def get_population(city: str) -> int:
        """Get population."""
        return 1000000

    # Bind tools
    llm_with_tools = llm.bind_tools([get_weather, get_population])

    # Invoke
    response = llm_with_tools.invoke([HumanMessage(content="Weather and population of Tokyo?")])

    # Verify tool_calls field has both calls
    assert len(response.tool_calls) == 2, "Should have two tool calls"

    # Check first tool call
    assert response.tool_calls[0]["name"] == "get_weather"
    assert response.tool_calls[0]["args"] == {"location": "Tokyo"}
    assert "id" in response.tool_calls[0]

    # Check second tool call
    assert response.tool_calls[1]["name"] == "get_population"
    assert response.tool_calls[1]["args"] == {"city": "Tokyo"}
    assert "id" in response.tool_calls[1]

    # Verify IDs are unique
    assert response.tool_calls[0]["id"] != response.tool_calls[1]["id"]

    # Verify additional_kwargs has both formatted calls
    assert "tool_calls" in response.additional_kwargs
    assert len(response.additional_kwargs["tool_calls"]) == 2
