# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Test OCI Generative AI LLM service"""

from typing import Optional, Union
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from pytest import MonkeyPatch

from langchain_oci.chat_models.oci_generative_ai import ChatOCIGenAI


class MockResponseDict(dict):
    def __getattr__(self, val):
        return self.get(val)


class MockToolCall(dict):
    def __getattr__(self, val):
        return self[val]


@pytest.mark.requires("oci")
@pytest.mark.parametrize(
    "test_model_id", ["cohere.command-r-16k", "meta.llama-3.3-70b-instruct"]
)
def test_llm_chat(monkeypatch: MonkeyPatch, test_model_id: str) -> None:
    """Test valid chat call to OCI Generative AI LLM service."""
    oci_gen_ai_client = MagicMock()
    llm = ChatOCIGenAI(model_id=test_model_id, client=oci_gen_ai_client)

    model_id = llm.model_id
    if model_id is None:
        raise ValueError("Model ID is required for OCI Generative AI LLM service.")

    provider = model_id.split(".")[0].lower()

    def mocked_response(*args):
        response_text = "Assistant chat reply."
        response = None
        if provider == "cohere":
            response = MockResponseDict(
                {
                    "status": 200,
                    "data": MockResponseDict(
                        {
                            "chat_response": MockResponseDict(
                                {
                                    "text": response_text,
                                    "finish_reason": "completed",
                                    "is_search_required": None,
                                    "search_queries": None,
                                    "citations": None,
                                    "documents": None,
                                    "tool_calls": None,
                                    "usage": MockResponseDict(
                                        {
                                            "prompt_tokens": 30,
                                            "completion_tokens": 20,
                                            "total_tokens": 50,
                                        }
                                    ),
                                }
                            ),
                            "model_id": "cohere.command-r-16k",
                            "model_version": "1.0.0",
                        }
                    ),
                    "request_id": "1234567890",
                    "headers": MockResponseDict(
                        {
                            "content-length": "123",
                        }
                    ),
                }
            )
        elif provider == "meta":
            response = MockResponseDict(
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
                                                                    "text": response_text,  # noqa: E501
                                                                    "type": "TEXT",
                                                                }
                                                            )
                                                        ],
                                                        "tool_calls": [
                                                            MockResponseDict(
                                                                {
                                                                    "type": "FUNCTION",
                                                                    "id": "call_123",
                                                                    "name": "get_weather",  # noqa: E501
                                                                    "arguments": '{"location": "current location"}',  # noqa: E501
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
                                    "time_created": "2025-08-14T10:00:01.100000+00:00",
                                    "usage": MockResponseDict(
                                        {
                                            "prompt_tokens": 45,
                                            "completion_tokens": 30,
                                            "total_tokens": 75,
                                        }
                                    ),
                                }
                            ),
                            "model_id": "meta.llama-3.3-70b-instruct",
                            "model_version": "1.0.0",
                        }
                    ),
                    "request_id": "1234567890",
                    "headers": MockResponseDict(
                        {
                            "content-length": "123",
                        }
                    ),
                }
            )
        return response

    monkeypatch.setattr(llm.client, "chat", mocked_response)

    messages = [
        HumanMessage(content="User message"),
    ]

    expected = "Assistant chat reply."
    actual = llm.invoke(messages, temperature=0.2)
    assert actual.content == expected

    # Test total_tokens in additional_kwargs
    assert "total_tokens" in actual.additional_kwargs
    if provider == "cohere":
        assert actual.additional_kwargs["total_tokens"] == 50
    elif provider == "meta":
        assert actual.additional_kwargs["total_tokens"] == 75

    # Test usage_metadata (new field, only available in langchain-core 1.0+)
    if hasattr(actual, "usage_metadata") and actual.usage_metadata is not None:
        if provider == "cohere":
            assert actual.usage_metadata["input_tokens"] == 30
            assert actual.usage_metadata["output_tokens"] == 20
            assert actual.usage_metadata["total_tokens"] == 50
        elif provider == "meta":
            assert actual.usage_metadata["input_tokens"] == 45
            assert actual.usage_metadata["output_tokens"] == 30
            assert actual.usage_metadata["total_tokens"] == 75


@pytest.mark.requires("oci")
def test_meta_tool_calling(monkeypatch: MonkeyPatch) -> None:
    """Test tool calling with Meta models."""
    import json

    oci_gen_ai_client = MagicMock()
    llm = ChatOCIGenAI(model_id="meta.llama-3-70b-instruct", client=oci_gen_ai_client)

    def mocked_response(*args, **kwargs):
        # Mock response with tool calls
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
                                                            {
                                                                "text": "Let me help you with that.",  # noqa: E501
                                                            }
                                                        )
                                                    ],
                                                    "tool_calls": [
                                                        MockResponseDict(
                                                            {
                                                                "type": "FUNCTION",
                                                                "id": "call_456",
                                                                "name": "get_weather",
                                                                "arguments": '{"location": "San Francisco"}',  # noqa: E501
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
                                "time_created": "2025-08-14T10:00:01.100000+00:00",
                            }
                        ),
                        "model_id": "meta.llama-3-70b-instruct",
                        "model_version": "1.0.0",
                    }
                ),
                "request_id": "1234567890",
                "headers": MockResponseDict({"content-length": "123"}),
            }
        )

    monkeypatch.setattr(llm.client, "chat", mocked_response)

    # Define a simple weather tool
    def get_weather(location: str) -> str:
        """Get the weather for a location."""
        return f"Weather for {location}"

    messages = [HumanMessage(content="What's the weather like?")]

    # Test different tool choice options
    tool_choices: list[Union[str, bool, dict[str, Union[str, dict[str, str]]]]] = [
        "get_weather",  # Specific tool
        "auto",  # Auto mode
        "none",  # No tools
        True,  # Required
        False,  # None
        {"type": "function", "function": {"name": "get_weather"}},  # Dict format
    ]

    for tool_choice in tool_choices:
        response = llm.bind_tools(
            tools=[get_weather],
            tool_choice=tool_choice,
        ).invoke(messages)

        assert response.content == "Let me help you with that."
        if tool_choice not in ["none", False]:
            assert response.additional_kwargs.get("tool_calls") is not None
            tool_call = response.additional_kwargs["tool_calls"][0]
            assert tool_call["type"] == "function"
            assert tool_call["function"]["name"] == "get_weather"

    # Test escaped JSON arguments (issue #52)
    def mocked_response_escaped(*args, **kwargs):
        """Mock response with escaped JSON arguments."""
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
                                                        MockResponseDict({"text": ""})
                                                    ],
                                                    "tool_calls": [
                                                        MockResponseDict(
                                                            {
                                                                "type": "FUNCTION",
                                                                "id": "call_escaped",
                                                                "name": "get_weather",
                                                                # Escaped JSON (the bug scenario) # noqa: E501
                                                                "arguments": '"{\\"location\\": \\"San Francisco\\"}"',  # noqa: E501
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
                                            "finish_reason": "tool_calls",
                                        }
                                    )
                                ],
                                "time_created": "2025-10-22T19:48:12.726000+00:00",
                            }
                        ),
                        "model_id": "meta.llama-3-70b-instruct",
                        "model_version": "1.0.0",
                    }
                ),
                "request_id": "test_escaped",
                "headers": MockResponseDict({"content-length": "366"}),
            }
        )

    monkeypatch.setattr(llm.client, "chat", mocked_response_escaped)
    response_escaped = llm.bind_tools(tools=[get_weather]).invoke(messages)

    # Verify escaped JSON was correctly parsed to a dict
    assert isinstance(response_escaped, AIMessage)
    assert len(response_escaped.tool_calls) == 1
    assert response_escaped.tool_calls[0]["name"] == "get_weather"
    assert response_escaped.tool_calls[0]["args"] == {"location": "San Francisco"}

    # Test streaming with missing text key (Gemini scenario - issue #86)
    mock_stream_events = [
        MagicMock(
            data=json.dumps(
                {
                    "index": 0,
                    "message": {
                        "role": "ASSISTANT",
                        "content": [{"type": "TEXT"}],  # No "text" key
                        "toolCalls": [
                            {  # No "id" key
                                "type": "FUNCTION",
                                "name": "get_weather",
                                "arguments": '{"location": "Boston"}',
                            }
                        ],
                    },
                }
            )
        ),
        MagicMock(data=json.dumps({"finishReason": "stop"})),
    ]
    mock_stream_response = MagicMock()
    mock_stream_response.data.events.return_value = mock_stream_events
    monkeypatch.setattr(  # noqa: E501
        llm.client, "chat", lambda *args, **kwargs: mock_stream_response
    )

    # Should not raise KeyError on missing text key
    chunks = list(llm.stream(messages))
    tool_chunk = next((c for c in chunks if c.tool_call_chunks), None)  # type: ignore[attr-defined, unused-ignore]
    assert tool_chunk is not None
    assert tool_chunk.tool_call_chunks[0]["name"] == "get_weather"  # type: ignore[attr-defined, unused-ignore]
    # Verify UUID was generated and index is correct (not -1)
    assert tool_chunk.tool_call_chunks[0]["id"] != ""  # type: ignore[attr-defined, unused-ignore]
    assert tool_chunk.tool_call_chunks[0]["index"] == 0  # type: ignore[attr-defined, unused-ignore]

    # Test GPT-OSS fragmented streaming (ID only in first chunk - issue #XX)
    mock_stream_events_gpt = [
        MagicMock(
            data=json.dumps(
                {
                    "index": 0,
                    "message": {
                        "role": "ASSISTANT",
                        "content": [{"type": "TEXT", "text": ""}],
                        "toolCalls": [
                            {
                                "id": "call_abc123",
                                "name": "get_weather",
                                "arguments": '{"loc',
                            }
                        ],
                    },
                }
            )
        ),
        MagicMock(
            data=json.dumps(
                {
                    "index": 0,
                    "message": {
                        "role": "ASSISTANT",
                        "content": [{"type": "TEXT", "text": ""}],
                        "toolCalls": [
                            {
                                "arguments": 'ation": "NYC"}',
                            }
                        ],
                    },
                }
            )
        ),
        MagicMock(data=json.dumps({"finishReason": "tool_calls"})),
    ]
    mock_stream_response_gpt = MagicMock()
    mock_stream_response_gpt.data.events.return_value = mock_stream_events_gpt
    monkeypatch.setattr(
        llm.client, "chat", lambda *args, **kwargs: mock_stream_response_gpt
    )

    chunks_gpt = list(llm.stream(messages))
    final_msg = None
    for c in chunks_gpt:
        final_msg = c if final_msg is None else final_msg + c
    assert final_msg is not None
    assert len(final_msg.tool_calls) == 1  # type: ignore[attr-defined, unused-ignore]
    assert final_msg.tool_calls[0]["name"] == "get_weather"  # type: ignore[attr-defined, unused-ignore]
    assert final_msg.tool_calls[0]["args"] == {"location": "NYC"}  # type: ignore[attr-defined, unused-ignore]

    # Test Grok parallel tool calls (same idx, different IDs - issue #XX)
    mock_stream_events_grok = [
        MagicMock(
            data=json.dumps(
                {
                    "index": 0,
                    "message": {
                        "role": "ASSISTANT",
                        "content": [{"type": "TEXT", "text": ""}],
                        "toolCalls": [
                            {
                                "id": "call_weather",
                                "name": "get_weather",
                                "arguments": '{"location": "Tokyo"}',
                            }
                        ],
                    },
                }
            )
        ),
        MagicMock(
            data=json.dumps(
                {
                    "index": 0,
                    "message": {
                        "role": "ASSISTANT",
                        "content": [{"type": "TEXT", "text": ""}],
                        "toolCalls": [
                            {
                                "id": "call_time",
                                "name": "get_time",
                                "arguments": '{"timezone": "PST"}',
                            }
                        ],
                    },
                }
            )
        ),
        MagicMock(data=json.dumps({"finishReason": "tool_calls"})),
    ]
    mock_stream_response_grok = MagicMock()
    mock_stream_response_grok.data.events.return_value = mock_stream_events_grok
    monkeypatch.setattr(
        llm.client, "chat", lambda *args, **kwargs: mock_stream_response_grok
    )

    chunks_grok = list(llm.stream(messages))
    final_msg_grok = None
    for c in chunks_grok:
        final_msg_grok = c if final_msg_grok is None else final_msg_grok + c
    assert final_msg_grok is not None
    assert len(final_msg_grok.tool_calls) == 2  # type: ignore[attr-defined, unused-ignore]
    assert final_msg_grok.tool_calls[0]["name"] == "get_weather"  # type: ignore[attr-defined, unused-ignore]
    assert final_msg_grok.tool_calls[1]["name"] == "get_time"  # type: ignore[attr-defined, unused-ignore]


@pytest.mark.requires("oci")
def test_cohere_tool_choice_validation(monkeypatch: MonkeyPatch) -> None:
    """Test that tool choice is not supported for Cohere models."""
    oci_gen_ai_client = MagicMock()
    llm = ChatOCIGenAI(model_id="cohere.command-r-16k", client=oci_gen_ai_client)

    def get_weather(location: str) -> str:
        """Get the weather for a location."""
        return f"Weather for {location}"

    messages = [HumanMessage(content="What's the weather like?")]

    # Test that tool choice raises ValueError
    with pytest.raises(
        ValueError, match="Tool choice is not supported for Cohere models"
    ):
        llm.bind_tools(
            tools=[get_weather],
            tool_choice="auto",
        ).invoke(messages)

    # Mock response for the case without tool choice
    def mocked_response(*args, **kwargs):
        return MockResponseDict(
            {
                "status": 200,
                "data": MockResponseDict(
                    {
                        "chat_response": MockResponseDict(
                            {
                                "text": "Response without tool choice",
                                "finish_reason": "completed",
                                "is_search_required": None,
                                "search_queries": None,
                                "citations": None,
                                "documents": None,
                                "tool_calls": None,
                            }
                        ),
                        "model_id": "cohere.command-r-16k",
                        "model_version": "1.0.0",
                    }
                ),
                "request_id": "1234567890",
                "headers": MockResponseDict({"content-length": "123"}),
            }
        )

    monkeypatch.setattr(llm.client, "chat", mocked_response)

    # Test that tools without tool choice works
    response = llm.bind_tools(tools=[get_weather]).invoke(messages)
    assert response.content == "Response without tool choice"


@pytest.mark.requires("oci")
def test_meta_tool_conversion(monkeypatch: MonkeyPatch) -> None:
    """Test tool conversion for Meta models."""
    from pydantic import BaseModel, Field

    oci_gen_ai_client = MagicMock()
    llm = ChatOCIGenAI(model_id="meta.llama-3.3-70b-instruct", client=oci_gen_ai_client)

    def mocked_response(*args, **kwargs):
        request = args[0]
        # Check the conversion of tools to oci generic API spec
        # Function tool
        assert request.chat_request.tools[0].parameters["properties"] == {
            "x": {"description": "Input number", "type": "integer"}
        }
        # Pydantic tool
        assert request.chat_request.tools[1].parameters["properties"] == {
            "x": {"description": "Input number", "type": "integer"},
            "y": {"description": "Input string", "type": "string"},
        }

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
                                                    "content": None,
                                                    "tool_calls": [
                                                        MockResponseDict(
                                                            {
                                                                "arguments": '{"x": "10"}',  # noqa: E501
                                                                "id": "chatcmpl-tool-d123",  # noqa: E501
                                                                "name": "function_tool",
                                                                "type": "FUNCTION",
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
                                            "finish_reason": "tool_calls",
                                        }
                                    )
                                ],
                                "time_created": "2025-08-14T10:00:01.100000+00:00",
                            }
                        ),
                        "model_id": "meta.llama-3.3-70b-instruct",
                        "model_version": "1.0.0",
                    }
                ),
                "request_id": "1234567890",
                "headers": MockResponseDict({"content-length": "123"}),
            }
        )

    monkeypatch.setattr(llm.client, "chat", mocked_response)

    # Test function tool
    def function_tool(x: int) -> int:
        """A simple function tool.

        Args:
            x: Input number
        """
        return x + 1

    # Test pydantic tool
    class PydanticTool(BaseModel):
        """A simple pydantic tool."""

        x: int = Field(description="Input number")
        y: str = Field(description="Input string")

    messages = [HumanMessage(content="Test message")]

    # Test that all tool types can be bound and used
    response = llm.bind_tools(
        tools=[function_tool, PydanticTool],
    ).invoke(messages)

    # For tool calls, the response content should be empty.
    assert response.content == ""
    assert isinstance(response, AIMessage)
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0]["name"] == "function_tool"


@pytest.mark.requires("oci")
def test_json_mode_output(monkeypatch: MonkeyPatch) -> None:
    """Test JSON mode output parsing."""
    from pydantic import BaseModel, Field

    class WeatherResponse(BaseModel):
        temperature: float = Field(description="Temperature in Celsius")
        conditions: str = Field(description="Weather conditions")

    oci_gen_ai_client = MagicMock()
    llm = ChatOCIGenAI(model_id="cohere.command-r-16k", client=oci_gen_ai_client)

    def mocked_response(*args, **kwargs):
        return MockResponseDict(
            {
                "status": 200,
                "data": MockResponseDict(
                    {
                        "chat_response": MockResponseDict(
                            {
                                "text": '{"temperature": 25.5, "conditions": "Sunny"}',
                                "finish_reason": "completed",
                                "is_search_required": None,
                                "search_queries": None,
                                "citations": None,
                                "documents": None,
                                "tool_calls": None,
                            }
                        ),
                        "model_id": "cohere.command-r-16k",
                        "model_version": "1.0.0",
                    }
                ),
                "request_id": "1234567890",
                "headers": MockResponseDict({"content-length": "123"}),
            }
        )

    monkeypatch.setattr(llm.client, "chat", mocked_response)

    messages = [HumanMessage(content="What's the weather like?")]

    # Test with pydantic model
    structured_llm = llm.with_structured_output(WeatherResponse, method="json_mode")
    response = structured_llm.invoke(messages)
    assert isinstance(response, WeatherResponse)
    assert response.temperature == 25.5
    assert response.conditions == "Sunny"


@pytest.mark.requires("oci")
def test_json_schema_output(monkeypatch: MonkeyPatch) -> None:
    """Test JSON schema output parsing."""
    from pydantic import BaseModel, Field

    class WeatherResponse(BaseModel):
        temperature: float = Field(description="Temperature in Celsius")
        conditions: str = Field(description="Weather conditions")

    oci_gen_ai_client = MagicMock()
    llm = ChatOCIGenAI(model_id="cohere.command-latest", client=oci_gen_ai_client)

    def mocked_response(*args, **kwargs):
        # Verify that response_format is a JsonSchemaResponseFormat object
        request = args[0]
        assert hasattr(request.chat_request, "response_format")
        assert request.chat_request.response_format is not None

        return MockResponseDict(
            {
                "status": 200,
                "data": MockResponseDict(
                    {
                        "chat_response": MockResponseDict(
                            {
                                "api_format": "COHERE",
                                "text": '{"temperature": 25.5, "conditions": "Sunny"}',
                                "finish_reason": "COMPLETE",
                                "is_search_required": None,
                                "search_queries": None,
                                "citations": None,
                                "documents": None,
                                "tool_calls": None,
                            }
                        ),
                        "model_id": "cohere.command-latest",
                        "model_version": "1.0.0",
                    }
                ),
                "request_id": "1234567890",
                "headers": MockResponseDict({"content-length": "123"}),
            }
        )

    monkeypatch.setattr(llm.client, "chat", mocked_response)

    messages = [HumanMessage(content="What's the weather like?")]

    # Test with pydantic model using json_schema method
    structured_llm = llm.with_structured_output(WeatherResponse, method="json_schema")
    response = structured_llm.invoke(messages)
    assert isinstance(response, WeatherResponse)
    assert response.temperature == 25.5
    assert response.conditions == "Sunny"


@pytest.mark.requires("oci")
def test_auth_file_location(monkeypatch: MonkeyPatch) -> None:
    """Test custom auth file location."""
    from unittest.mock import patch

    with patch("oci.config.from_file") as mock_from_file:
        with patch(
            "oci.generative_ai_inference.generative_ai_inference_client.validate_config"
        ):
            with patch("oci.base_client.validate_config"):
                with patch("oci.signer.load_private_key"):
                    custom_config_path = "/custom/path/config"
                    ChatOCIGenAI(
                        model_id="cohere.command-r-16k",
                        auth_file_location=custom_config_path,
                    )
                    mock_from_file.assert_called_once_with(
                        file_location=custom_config_path, profile_name="DEFAULT"
                    )


@pytest.mark.requires("oci")
def test_include_raw_output(monkeypatch: MonkeyPatch) -> None:
    """Test include_raw parameter in structured output."""
    from pydantic import BaseModel, Field

    class WeatherResponse(BaseModel):
        temperature: float = Field(description="Temperature in Celsius")
        conditions: str = Field(description="Weather conditions")

    oci_gen_ai_client = MagicMock()
    llm = ChatOCIGenAI(model_id="cohere.command-r-16k", client=oci_gen_ai_client)

    def mocked_response(*args, **kwargs):
        return MockResponseDict(
            {
                "status": 200,
                "data": MockResponseDict(
                    {
                        "chat_response": MockResponseDict(
                            {
                                "text": '{"temperature": 25.5, "conditions": "Sunny"}',
                                "finish_reason": "completed",
                                "is_search_required": None,
                                "search_queries": None,
                                "citations": None,
                                "documents": None,
                                "tool_calls": None,
                            }
                        ),
                        "model_id": "cohere.command-r-16k",
                        "model_version": "1.0.0",
                    }
                ),
                "request_id": "1234567890",
                "headers": MockResponseDict({"content-length": "123"}),
            }
        )

    monkeypatch.setattr(llm.client, "chat", mocked_response)

    messages = [HumanMessage(content="What's the weather like?")]

    # Test with include_raw=True
    structured_llm = llm.with_structured_output(
        WeatherResponse, method="json_schema", include_raw=True
    )
    response = structured_llm.invoke(messages)
    assert isinstance(response, dict)
    assert "parsed" in response
    assert "raw" in response
    assert isinstance(response["parsed"], WeatherResponse)
    assert response["parsed"].temperature == 25.5
    assert response["parsed"].conditions == "Sunny"


@pytest.mark.requires("oci")
def test_ai_message_tool_calls_direct_field(monkeypatch: MonkeyPatch) -> None:
    """Test AIMessage with tool_calls in the direct tool_calls field."""

    oci_gen_ai_client = MagicMock()
    llm = ChatOCIGenAI(model_id="meta.llama-3.3-70b-instruct", client=oci_gen_ai_client)

    # Track if the tool_calls processing branch is executed
    tool_calls_processed = False

    def mocked_response(*args, **kwargs):
        nonlocal tool_calls_processed
        # Check if the request contains tool_calls in the message
        request = args[0]
        has_chat_request = hasattr(request, "chat_request")
        has_messages = has_chat_request and hasattr(request.chat_request, "messages")
        if has_messages:
            for msg in request.chat_request.messages:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    tool_calls_processed = True
                    break
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
                                                                "text": (
                                                                    "I'll help you."
                                                                ),
                                                                "type": "TEXT",
                                                            }
                                                        )
                                                    ],
                                                    "tool_calls": [],
                                                }
                                            ),
                                            "finish_reason": "completed",
                                        }
                                    )
                                ],
                                "time_created": "2025-08-14T10:00:01.100000+00:00",
                            }
                        ),
                        "model_id": "meta.llama-3.3-70b-instruct",
                        "model_version": "1.0.0",
                    }
                ),
                "request_id": "1234567890",
                "headers": MockResponseDict({"content-length": "123"}),
            }
        )

    monkeypatch.setattr(llm.client, "chat", mocked_response)

    # Create AIMessage with tool_calls in the direct tool_calls field
    ai_message = AIMessage(
        content="I need to call a function",
        tool_calls=[
            {
                "id": "call_123",
                "name": "get_weather",
                "args": {"location": "San Francisco"},
            }
        ],
    )

    messages = [ai_message]

    # This should not raise an error and should process the tool_calls correctly
    response = llm.invoke(messages)
    assert response.content == "I'll help you."


@pytest.mark.requires("oci")
def test_ai_message_tool_calls_additional_kwargs(monkeypatch: MonkeyPatch) -> None:
    """Test AIMessage with tool_calls in additional_kwargs field."""

    oci_gen_ai_client = MagicMock()
    llm = ChatOCIGenAI(model_id="meta.llama-3.3-70b-instruct", client=oci_gen_ai_client)

    def mocked_response(*args, **kwargs):
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
                                                                "text": (
                                                                    "I'll help you."
                                                                ),
                                                                "type": "TEXT",
                                                            }
                                                        )
                                                    ],
                                                    "tool_calls": [],
                                                }
                                            ),
                                            "finish_reason": "completed",
                                        }
                                    )
                                ],
                                "time_created": "2025-08-14T10:00:01.100000+00:00",
                            }
                        ),
                        "model_id": "meta.llama-3.3-70b-instruct",
                        "model_version": "1.0.0",
                    }
                ),
                "request_id": "1234567890",
                "headers": MockResponseDict({"content-length": "123"}),
            }
        )

    monkeypatch.setattr(llm.client, "chat", mocked_response)

    # Create AIMessage with tool_calls in additional_kwargs
    ai_message = AIMessage(
        content="I need to call a function",
        additional_kwargs={
            "tool_calls": [
                {
                    "id": "call_456",
                    "name": "get_weather",
                    "args": {"location": "New York"},
                }
            ]
        },
    )

    messages = [ai_message]

    # This should not raise an error and should process the tool_calls correctly
    response = llm.invoke(messages)
    assert response.content == "I'll help you."


@pytest.mark.requires("oci")
def test_get_provider():
    """Test determining the provider based on the model_id."""
    oci_gen_ai_client = MagicMock()
    model_provider_map = {
        "cohere.command-latest": "CohereProvider",
        "meta.llama-3.3-70b-instruct": "MetaProvider",
        "xai.grok-3": "GenericProvider",
    }
    for model_id, provider_name in model_provider_map.items():
        llm = ChatOCIGenAI(model_id=model_id, client=oci_gen_ai_client)
        assert llm._provider.__class__.__name__ == provider_name


@pytest.mark.requires("oci")
def test_cohere_vision_detects_system_message_images(monkeypatch: MonkeyPatch) -> None:
    """Test that Cohere V2 API detects images in SystemMessage content."""
    from langchain_core.messages import SystemMessage

    from langchain_oci.chat_models.providers.cohere import CohereProvider

    provider = CohereProvider()

    # Mock _load_v2_classes to avoid RuntimeError in CI where V2 classes may not exist
    monkeypatch.setattr(provider, "_load_v2_classes", lambda: None)

    # Test with image in HumanMessage - should detect
    human_msg_with_image = HumanMessage(
        content=[
            {"type": "text", "text": "What is this?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,ABC"}},
        ]
    )
    assert provider._has_vision_content([human_msg_with_image]) is True

    # Test with image in SystemMessage - should also detect
    system_msg_with_image = SystemMessage(
        content=[
            {"type": "text", "text": "You are an assistant analyzing this image:"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,XYZ"}},
        ]
    )
    assert provider._has_vision_content([system_msg_with_image]) is True

    # Test with text-only messages - should not detect
    human_msg_text_only = HumanMessage(content="Hello")
    system_msg_text_only = SystemMessage(content="You are a helpful assistant.")
    text_only_msgs = [human_msg_text_only, system_msg_text_only]
    assert provider._has_vision_content(text_only_msgs) is False


@pytest.mark.requires("oci")
def test_v2_api_guard_for_non_cohere_providers(monkeypatch: MonkeyPatch) -> None:
    """Test that V2 API raises error for non-Cohere providers.

    The V2 API guard ensures that only providers with oci_chat_request_v2
    can use the V2 API path. This prevents runtime errors if someone
    accidentally sets _use_v2_api=True for a non-supporting provider.
    """
    oci_gen_ai_client = MagicMock()

    # Test with Meta model (uses GenericProvider via MetaProvider)
    llm = ChatOCIGenAI(model_id="meta.llama-3.3-70b-instruct", client=oci_gen_ai_client)

    # Mock the provider's messages_to_oci_params to return _use_v2_api=True
    # This simulates what would happen if V2 API was incorrectly triggered
    original_method = llm._provider.messages_to_oci_params

    def mock_messages_to_oci_params(*args, **kwargs):
        result = original_method(*args, **kwargs)
        result["_use_v2_api"] = True  # Force V2 API flag
        return result

    monkeypatch.setattr(
        llm._provider, "messages_to_oci_params", mock_messages_to_oci_params
    )

    message = HumanMessage(content="Test message")

    # Now when _use_v2_api=True but provider doesn't support V2, should raise
    with pytest.raises(ValueError, match="V2 API is not supported"):
        llm._prepare_request([message], stop=None, stream=False)


@pytest.mark.requires("oci")
def test_tool_choice_none_after_tool_results() -> None:
    """Test tool_choice='none' when max_sequential_tool_calls is exceeded.

    This prevents infinite loops with Meta Llama models by limiting the number
    of sequential tool calls.
    """
    from langchain_core.messages import ToolMessage
    from oci.generative_ai_inference import models

    oci_gen_ai_client = MagicMock()
    llm = ChatOCIGenAI(
        model_id="meta.llama-3.3-70b-instruct",
        client=oci_gen_ai_client,
        max_sequential_tool_calls=3,  # Set limit to 3 for testing
    )

    # Define a simple tool function (following the pattern from other tests)
    def get_weather(city: str) -> str:
        """Get weather for a city.

        Args:
            city: The city to get weather for
        """
        return f"Weather in {city}"

    # Bind tools to model
    llm_with_tools = llm.bind_tools([get_weather])

    # Create conversation with 3 ToolMessages (at the limit)
    messages = [
        HumanMessage(content="What's the weather?"),
        AIMessage(
            content="",
            tool_calls=[
                {"id": "call_1", "name": "get_weather", "args": {"city": "Chicago"}}
            ],
        ),
        ToolMessage(content="Sunny, 65°F", tool_call_id="call_1"),
        AIMessage(
            content="",
            tool_calls=[
                {"id": "call_2", "name": "get_weather", "args": {"city": "New York"}}
            ],
        ),
        ToolMessage(content="Rainy, 55°F", tool_call_id="call_2"),
        AIMessage(
            content="",
            tool_calls=[
                {"id": "call_3", "name": "get_weather", "args": {"city": "Seattle"}}
            ],
        ),
        ToolMessage(content="Cloudy, 60°F", tool_call_id="call_3"),
    ]

    # Prepare the request - need to pass tools from the bound model kwargs
    request = llm._prepare_request(
        messages,
        stop=None,
        stream=False,
        **llm_with_tools.kwargs,  # type: ignore[attr-defined]
    )

    # Verify that tool_choice is set to 'none' because limit was reached
    assert hasattr(request.chat_request, "tool_choice")
    assert isinstance(request.chat_request.tool_choice, models.ToolChoiceNone)
    # Verify tools are still present (not removed, just choice is 'none')
    assert hasattr(request.chat_request, "tools")
    assert len(request.chat_request.tools) > 0


# =============================================================================
# Reasoning Content Extraction Tests
# =============================================================================


def _make_reasoning_response(
    text: str = "The answer is 42.",
    reasoning_content: Optional[str] = None,
    finish_reason: str = "completed",
) -> MockResponseDict:
    """Build a mock Generic API response with optional reasoning_content."""
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
                                    {"message": message, "finish_reason": finish_reason}
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
                    "model_id": "xai.grok-3-mini",
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
                                                        {"text": "Hi", "type": "TEXT"}
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

    def test_reasoning_content_in_generation_info(self) -> None:
        """reasoning_content appears in additional_kwargs when present."""
        oci_client = MagicMock()
        llm = ChatOCIGenAI(model_id="xai.grok-3-mini", client=oci_client)

        reasoning_text = "Let me compute: 7 * 8 = 56."
        oci_client.chat.return_value = _make_reasoning_response(
            text="56", reasoning_content=reasoning_text
        )

        result = llm.invoke([HumanMessage(content="What is 7 * 8?")])
        assert result.additional_kwargs["reasoning_content"] == reasoning_text

    def test_reasoning_content_absent_for_standard_models(self) -> None:
        """Standard models without reasoning_content don't add the key."""
        oci_client = MagicMock()
        llm = ChatOCIGenAI(model_id="meta.llama-3.3-70b-instruct", client=oci_client)

        oci_client.chat.return_value = _make_reasoning_response(
            text="56", reasoning_content=None
        )

        result = llm.invoke([HumanMessage(content="What is 7 * 8?")])
        assert "reasoning_content" not in result.additional_kwargs


@pytest.mark.requires("oci")
class TestNullGuards:
    """Verify GenericProvider handles empty/null responses gracefully."""

    def test_empty_choices_returns_empty_text(self) -> None:
        """chat_response_to_text returns '' when choices is empty."""
        from langchain_oci.chat_models.providers.generic import GenericProvider

        provider = GenericProvider()
        response = _make_empty_choices_response()
        assert provider.chat_response_to_text(response) == ""

    def test_empty_choices_returns_no_tool_calls(self) -> None:
        """chat_tool_calls returns [] when choices is empty."""
        from langchain_oci.chat_models.providers.generic import GenericProvider

        provider = GenericProvider()
        response = _make_empty_choices_response()
        assert provider.chat_tool_calls(response) == []

    def test_empty_choices_generation_info_has_null_finish(self) -> None:
        """chat_generation_info returns finish_reason=None for empty."""
        from langchain_oci.chat_models.providers.generic import GenericProvider

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
