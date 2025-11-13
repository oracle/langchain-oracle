# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Test OCI Generative AI LLM service"""

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from pytest import MonkeyPatch

from langchain_oci.chat_models.oci_generative_ai import ChatOCIGenAI


class MockResponseDict(dict):
    def __getattr__(self, val):  # type: ignore[no-untyped-def]
        return self.get(val)


class MockToolCall(dict):
    def __getattr__(self, val):  # type: ignore[no-untyped-def]
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

    def mocked_response(*args):  # type: ignore[no-untyped-def]
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


@pytest.mark.requires("oci")
def test_meta_tool_calling(monkeypatch: MonkeyPatch) -> None:
    """Test tool calling with Meta models."""
    oci_gen_ai_client = MagicMock()
    llm = ChatOCIGenAI(model_id="meta.llama-3-70b-instruct", client=oci_gen_ai_client)

    def mocked_response(*args, **kwargs):  # type: ignore[no-untyped-def]
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
                                                                "name": "get_weather",  # noqa: E501
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
    tool_choices = [
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
    def mocked_response_escaped(*args, **kwargs):  # type: ignore[no-untyped-def]
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
                                                                # Escaped JSON (the bug scenario)  # noqa: E501
                                                                "arguments": '"{{\\"location\\": \\"San Francisco\\"}}"',  # noqa: E501
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
    assert len(response_escaped.tool_calls) == 1
    assert response_escaped.tool_calls[0]["name"] == "get_weather"
    assert response_escaped.tool_calls[0]["args"] == {"location": "San Francisco"}


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
    ):  # noqa: E501
        llm.bind_tools(
            tools=[get_weather],
            tool_choice="auto",
        ).invoke(messages)

    # Mock response for the case without tool choice
    def mocked_response(*args, **kwargs):  # type: ignore[no-untyped-def]
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

    def mocked_response(*args, **kwargs):  # type: ignore[no-untyped-def]
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

    def mocked_response(*args, **kwargs):  # type: ignore[no-untyped-def]
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

    def mocked_response(*args, **kwargs):  # type: ignore[no-untyped-def]
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
        ):  # noqa: E501
            with patch("oci.base_client.validate_config"):
                with patch("oci.signer.load_private_key"):
                    custom_config_path = "/custom/path/config"
                    ChatOCIGenAI(
                        model_id="cohere.command-r-16k",
                        auth_file_location=custom_config_path,
                    )
                    mock_from_file.assert_called_once_with(
                        file_location=custom_config_path, profile_name="DEFAULT"
                    )  # noqa: E501


@pytest.mark.requires("oci")
def test_include_raw_output(monkeypatch: MonkeyPatch) -> None:
    """Test include_raw parameter in structured output."""
    from pydantic import BaseModel, Field

    class WeatherResponse(BaseModel):
        temperature: float = Field(description="Temperature in Celsius")
        conditions: str = Field(description="Weather conditions")

    oci_gen_ai_client = MagicMock()
    llm = ChatOCIGenAI(model_id="cohere.command-r-16k", client=oci_gen_ai_client)

    def mocked_response(*args, **kwargs):  # type: ignore[no-untyped-def]
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
    )  # noqa: E501
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

    def mocked_response(*args, **kwargs):  # type: ignore[no-untyped-def]
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
                                                                ),  # noqa: E501
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

    def mocked_response(*args, **kwargs):  # type: ignore[no-untyped-def]
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
                                                                ),  # noqa: E501
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
def test_tool_choice_none_after_tool_results() -> None:  # noqa: E501
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
        ),  # noqa: E501
        ToolMessage(content="Sunny, 65°F", tool_call_id="call_1"),
        AIMessage(
            content="",
            tool_calls=[
                {"id": "call_2", "name": "get_weather", "args": {"city": "New York"}}
            ],
        ),  # noqa: E501
        ToolMessage(content="Rainy, 55°F", tool_call_id="call_2"),
        AIMessage(
            content="",
            tool_calls=[
                {"id": "call_3", "name": "get_weather", "args": {"city": "Seattle"}}
            ],
        ),  # noqa: E501
        ToolMessage(content="Cloudy, 60°F", tool_call_id="call_3"),
    ]

    # Prepare the request - need to pass tools from the bound model kwargs
    request = llm_with_tools._prepare_request(
        messages, stop=None, stream=False, **llm_with_tools.kwargs
    )  # noqa: E501

    # Verify that tool_choice is set to 'none' because limit was reached
    assert hasattr(request.chat_request, "tool_choice")
    assert isinstance(request.chat_request.tool_choice, models.ToolChoiceNone)
    # Verify tools are still present (not removed, just choice is 'none')
    assert hasattr(request.chat_request, "tools")
    assert len(request.chat_request.tools) > 0
