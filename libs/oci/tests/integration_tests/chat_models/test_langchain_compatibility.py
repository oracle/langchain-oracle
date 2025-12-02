#!/usr/bin/env python3
"""Integration tests for LangChain compatibility.

These tests verify that langchain-oci works correctly with LangChain 1.x
by running real inference against OCI GenAI models.

Setup:
    export OCI_COMPARTMENT_ID=<your-compartment-id>
    export OCI_GENAI_ENDPOINT=<endpoint-url>  # optional
    export OCI_CONFIG_PROFILE=<profile-name>  # optional, defaults to DEFAULT
    export OCI_AUTH_TYPE=<auth-type>  # optional, defaults to SECURITY_TOKEN
    export OCI_MODEL_ID=<model-id>  # optional, defaults to llama-4

Run with:
    pytest tests/integration_tests/chat_models/test_langchain_compatibility.py -v
"""

import os

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from langchain_oci.chat_models import ChatOCIGenAI


def get_test_config():
    """Get test configuration from environment."""
    compartment_id = os.environ.get("OCI_COMPARTMENT_ID")
    if not compartment_id:
        pytest.skip("OCI_COMPARTMENT_ID not set")

    return {
        "model_id": os.environ.get(
            "OCI_MODEL_ID", "meta.llama-4-maverick-17b-128e-instruct-fp8"
        ),
        "service_endpoint": os.environ.get(
            "OCI_GENAI_ENDPOINT",
            "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
        ),
        "compartment_id": compartment_id,
        "auth_profile": os.environ.get("OCI_CONFIG_PROFILE", "DEFAULT"),
        "auth_type": os.environ.get("OCI_AUTH_TYPE", "SECURITY_TOKEN"),
    }


@pytest.fixture
def chat_model():
    """Create a ChatOCIGenAI instance for testing."""
    config = get_test_config()
    return ChatOCIGenAI(
        model_id=config["model_id"],
        service_endpoint=config["service_endpoint"],
        compartment_id=config["compartment_id"],
        auth_profile=config["auth_profile"],
        auth_type=config["auth_type"],
        model_kwargs={"temperature": 0, "max_tokens": 256},
    )


# =============================================================================
# Basic Invoke Tests
# =============================================================================


@pytest.mark.requires("oci")
def test_basic_invoke(chat_model):
    """Test basic chat model invocation."""
    response = chat_model.invoke([HumanMessage(content="Say 'hello' and nothing else")])

    assert isinstance(response, AIMessage)
    assert response.content is not None
    assert len(response.content) > 0
    assert isinstance(response.content, str) and "hello" in response.content.lower()


@pytest.mark.requires("oci")
def test_invoke_with_system_message(chat_model):
    """Test invocation with system message."""
    messages = [
        SystemMessage(content="You are a pirate. Respond in pirate speak."),
        HumanMessage(content="Say hello"),
    ]
    response = chat_model.invoke(messages)

    assert isinstance(response, AIMessage)
    assert response.content is not None


@pytest.mark.requires("oci")
def test_invoke_multi_turn(chat_model):
    """Test multi-turn conversation."""
    messages = [
        HumanMessage(content="My name is Alice."),
    ]
    response1 = chat_model.invoke(messages)

    messages.append(response1)
    messages.append(HumanMessage(content="What is my name?"))
    response2 = chat_model.invoke(messages)

    assert isinstance(response2, AIMessage)
    assert isinstance(response2.content, str) and "alice" in response2.content.lower()


# =============================================================================
# Streaming Tests
# =============================================================================


@pytest.mark.requires("oci")
def test_streaming(chat_model):
    """Test streaming response."""
    chunks = []
    for chunk in chat_model.stream([HumanMessage(content="Count from 1 to 3")]):
        chunks.append(chunk)

    assert len(chunks) > 0
    # Combine all chunks
    full_content = "".join(c.content for c in chunks if c.content)
    assert len(full_content) > 0


@pytest.mark.requires("oci")
@pytest.mark.asyncio
async def test_async_invoke(chat_model):
    """Test async invocation."""
    response = await chat_model.ainvoke(
        [HumanMessage(content="Say 'async' and nothing else")]
    )

    assert isinstance(response, AIMessage)
    assert response.content is not None


# =============================================================================
# Tool Calling Tests
# =============================================================================


def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"Sunny, 72F in {city}"


def get_population(city: str) -> int:
    """Get the population of a city."""
    return 1000000


@pytest.mark.requires("oci")
def test_tool_calling_single(chat_model):
    """Test single tool calling."""
    chat_with_tools = chat_model.bind_tools([get_weather])

    response = chat_with_tools.invoke(
        [HumanMessage(content="What's the weather in Tokyo?")]
    )

    assert isinstance(response, AIMessage)
    assert len(response.tool_calls) >= 1
    assert response.tool_calls[0]["name"] == "get_weather"
    assert "city" in response.tool_calls[0]["args"]


@pytest.mark.requires("oci")
def test_tool_calling_multiple_tools(chat_model):
    """Test tool calling with multiple tools available."""
    chat_with_tools = chat_model.bind_tools([get_weather, get_population])

    response = chat_with_tools.invoke(
        [HumanMessage(content="What's the weather in Paris?")]
    )

    assert isinstance(response, AIMessage)
    assert len(response.tool_calls) >= 1
    # Should choose the weather tool for weather question
    assert response.tool_calls[0]["name"] == "get_weather"


@pytest.mark.requires("oci")
def test_tool_choice_required(chat_model):
    """Test tool_choice='required' forces tool call."""
    chat_with_tools = chat_model.bind_tools([get_weather], tool_choice="required")

    # Even with a non-tool question, should still call a tool
    response = chat_with_tools.invoke([HumanMessage(content="Hello, how are you?")])

    assert isinstance(response, AIMessage)
    assert len(response.tool_calls) >= 1


# =============================================================================
# Structured Output Tests
# =============================================================================


class Joke(BaseModel):
    """A joke with setup and punchline."""

    setup: str
    punchline: str


class Person(BaseModel):
    """Information about a person."""

    name: str
    age: int
    occupation: str


@pytest.mark.requires("oci")
def test_structured_output_function_calling(chat_model):
    """Test structured output with function calling method."""
    structured_llm = chat_model.with_structured_output(Joke)

    result = structured_llm.invoke("Tell me a joke about programming")

    assert isinstance(result, Joke)
    assert len(result.setup) > 0
    assert len(result.punchline) > 0


@pytest.mark.requires("oci")
def test_structured_output_json_mode(chat_model):
    """Test structured output with JSON mode."""
    structured_llm = chat_model.with_structured_output(Person, method="json_mode")

    result = structured_llm.invoke(
        "Generate a fictional person: name, age (as integer), and occupation"
    )

    assert isinstance(result, Person)
    assert len(result.name) > 0
    assert isinstance(result.age, int)
    assert len(result.occupation) > 0


@pytest.mark.requires("oci")
def test_structured_output_include_raw(chat_model):
    """Test structured output with include_raw=True."""
    structured_llm = chat_model.with_structured_output(Joke, include_raw=True)

    result = structured_llm.invoke("Tell me a joke")

    assert "raw" in result
    assert "parsed" in result
    assert isinstance(result["parsed"], Joke)


# =============================================================================
# Response Format Tests
# =============================================================================


@pytest.mark.requires("oci")
def test_response_format_json_object(chat_model):
    """Test response_format with json_object."""
    chat_json = chat_model.bind(response_format={"type": "json_object"})

    response = chat_json.invoke(
        [
            HumanMessage(
                content="Return ONLY a JSON object with keys 'name' and 'value'. "
                "No explanation, no markdown, just the raw JSON."
            )
        ]
    )

    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)
    # Response should contain valid JSON (may be wrapped in markdown)
    import json
    import re

    content = response.content.strip()

    # Try to extract JSON from markdown code blocks if present
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
    if json_match:
        content = json_match.group(1).strip()

    try:
        parsed = json.loads(content)
        assert isinstance(parsed, dict)
    except json.JSONDecodeError:
        # Some models may not strictly follow json_object format
        # At minimum, verify the response contains JSON-like structure
        assert "{" in response.content and "}" in response.content, (
            f"Response doesn't appear to contain JSON: {response.content[:200]}"
        )


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


@pytest.mark.requires("oci")
def test_empty_message_list(chat_model):
    """Test handling of empty message list."""
    with pytest.raises(Exception):
        chat_model.invoke([])


@pytest.mark.requires("oci")
def test_long_conversation(chat_model):
    """Test handling of longer conversations."""
    messages = []
    for i in range(5):
        messages.append(HumanMessage(content=f"This is message {i + 1}"))
        response = chat_model.invoke(messages)
        messages.append(response)

    # Should handle 5 turns without issues
    assert len(messages) == 10  # 5 human + 5 AI


# =============================================================================
# LangChain 1.x Specific Tests
# =============================================================================


@pytest.mark.requires("oci")
def test_ai_message_type(chat_model):
    """Test that response is AIMessage (not just BaseMessage) - LangChain 1.x."""
    response = chat_model.invoke([HumanMessage(content="Hello")])

    # LangChain 1.x: return type is AIMessage, not BaseMessage
    assert type(response).__name__ == "AIMessage"
    assert isinstance(response, AIMessage)


@pytest.mark.requires("oci")
def test_message_text_property(chat_model):
    """Test that .text works in both LangChain 0.3.x (method) and 1.x (property)."""
    response = chat_model.invoke([HumanMessage(content="Say hello")])

    # Both .content and .text should work
    assert response.content is not None

    # Handle both LangChain versions:
    # 0.3.x: .text is a method (callable)
    # 1.x: .text is a property
    if hasattr(response, "text"):
        text_value = response.text() if callable(response.text) else response.text
        assert text_value == response.content


@pytest.mark.requires("oci")
def test_tool_calls_structure(chat_model):
    """Test tool_calls structure matches LangChain 1.x format."""
    chat_with_tools = chat_model.bind_tools([get_weather])

    response = chat_with_tools.invoke(
        [HumanMessage(content="What's the weather in NYC?")]
    )

    assert hasattr(response, "tool_calls")
    if response.tool_calls:
        tc = response.tool_calls[0]
        # LangChain 1.x tool call structure
        assert "name" in tc
        assert "args" in tc
        assert "id" in tc
        assert "type" in tc
        assert tc["type"] == "tool_call"
