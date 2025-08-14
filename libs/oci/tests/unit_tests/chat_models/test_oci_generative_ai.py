# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Test OCI Generative AI LLM service"""

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import HumanMessage
from pytest import MonkeyPatch

from langchain_oci.chat_models.oci_generative_ai import ChatOCIGenAI


class MockResponseDict(dict):
    def __getattr__(self, val):  # type: ignore[no-untyped-def]
        return self[val]


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
        # Verify that response_format contains the schema
        request = args[0]
        assert request.chat_request.response_format["type"] == "JSON_OBJECT"
        assert "schema" in request.chat_request.response_format

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
    )
    response = structured_llm.invoke(messages)
    assert isinstance(response, dict)
    assert "parsed" in response
    assert "raw" in response
    assert isinstance(response["parsed"], WeatherResponse)
    assert response["parsed"].temperature == 25.5
    assert response["parsed"].conditions == "Sunny"


def test_get_provider():
    """Test determining the provider based on the model_id."""
    model_provider_map = {
        "cohere.command-latest": "CohereProvider",
        "meta.llama-3.3-70b-instruct": "MetaProvider",
        "xai.grok-3": "GenericProvider",
    }
    for model_id, provider_name in model_provider_map.items():
        assert (
            ChatOCIGenAI(model_id=model_id)._provider.__class__.__name__
            == provider_name
        )
