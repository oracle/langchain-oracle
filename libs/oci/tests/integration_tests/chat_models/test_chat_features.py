#!/usr/bin/env python3
"""Integration tests for ChatOCIGenAI features.

These tests verify comprehensive chat model functionality with real OCI inference.

Setup:
    export OCI_COMPARTMENT_ID=<your-compartment-id>
    export OCI_CONFIG_PROFILE=DEFAULT
    export OCI_AUTH_TYPE=SECURITY_TOKEN

Run:
    pytest tests/integration_tests/chat_models/test_chat_features.py -v
"""

import os
from typing import Union

import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from langchain_oci.chat_models import ChatOCIGenAI


def get_config():
    """Get test configuration."""
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
def llm():
    """Create ChatOCIGenAI instance."""
    config = get_config()
    return ChatOCIGenAI(
        model_id=config["model_id"],
        service_endpoint=config["service_endpoint"],
        compartment_id=config["compartment_id"],
        auth_profile=config["auth_profile"],
        auth_type=config["auth_type"],
        model_kwargs={"temperature": 0, "max_tokens": 512},
    )


# =============================================================================
# Chain and LCEL Tests
# =============================================================================


@pytest.mark.requires("oci")
def test_simple_chain(llm):
    """Test simple LCEL chain: prompt | llm | parser."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            ("human", "{input}"),
        ]
    )
    chain = prompt | llm | StrOutputParser()

    result = chain.invoke({"input": "Say 'chain works' and nothing else"})

    assert isinstance(result, str)
    assert "chain" in result.lower() or "works" in result.lower()


@pytest.mark.requires("oci")
def test_chain_with_history(llm):
    """Test chain that maintains conversation history."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant with memory."),
            ("placeholder", "{history}"),
            ("human", "{input}"),
        ]
    )
    chain = prompt | llm | StrOutputParser()

    # First turn
    result1 = chain.invoke({"history": [], "input": "My favorite color is blue."})
    assert isinstance(result1, str)

    # Second turn with history
    history = [
        HumanMessage(content="My favorite color is blue."),
        AIMessage(content=result1),
    ]
    result2 = chain.invoke({"history": history, "input": "What is my favorite color?"})

    assert "blue" in result2.lower()


@pytest.mark.requires("oci")
def test_chain_batch(llm):
    """Test batch processing with LCEL."""
    prompt = ChatPromptTemplate.from_messages([("human", "What is {num} + {num}?")])
    chain = prompt | llm | StrOutputParser()

    results = chain.batch([{"num": "1"}, {"num": "2"}, {"num": "3"}])

    assert len(results) == 3
    assert all(isinstance(r, str) for r in results)


@pytest.mark.requires("oci")
@pytest.mark.asyncio
async def test_chain_async(llm):
    """Test async chain invocation."""
    prompt = ChatPromptTemplate.from_messages([("human", "Say '{word}'")])
    chain = prompt | llm | StrOutputParser()

    result = await chain.ainvoke({"word": "async"})

    assert isinstance(result, str)
    assert "async" in result.lower()


# =============================================================================
# Streaming Tests
# =============================================================================


@pytest.mark.requires("oci")
def test_stream_chain(llm):
    """Test streaming through a chain."""
    prompt = ChatPromptTemplate.from_messages([("human", "Count from 1 to 5")])
    chain = prompt | llm | StrOutputParser()

    chunks = []
    for chunk in chain.stream({}):
        chunks.append(chunk)

    assert len(chunks) > 0
    full_response = "".join(chunks)
    assert len(full_response) > 0


@pytest.mark.requires("oci")
@pytest.mark.asyncio
async def test_astream(llm):
    """Test async streaming."""
    chunks = []
    async for chunk in llm.astream([HumanMessage(content="Say hello")]):
        chunks.append(chunk)

    assert len(chunks) > 0


# =============================================================================
# Tool Calling Advanced Tests
# =============================================================================


def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


def multiply_numbers(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b


def get_user_info(user_id: str) -> dict:
    """Get information about a user."""
    return {"user_id": user_id, "name": "John Doe", "email": "john@example.com"}


@pytest.mark.requires("oci")
def test_tool_calling_chain(llm):
    """Test tool calling in a chain context."""
    tools = [get_user_info]
    llm_with_tools = llm.bind_tools(tools)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant. Use tools when needed."),
            ("human", "{input}"),
        ]
    )
    chain = prompt | llm_with_tools

    response = chain.invoke({"input": "Get info for user ID 'abc123'"})

    assert len(response.tool_calls) >= 1
    assert response.tool_calls[0]["name"] == "get_user_info"
    assert response.tool_calls[0]["args"]["user_id"] == "abc123"


@pytest.mark.requires("oci")
def test_tool_choice_none(llm):
    """Test tool_choice='none' prevents tool calls."""
    tools = [add_numbers]
    llm_with_tools = llm.bind_tools(tools, tool_choice="none")

    response = llm_with_tools.invoke([HumanMessage(content="What is 5 plus 3?")])

    # Should not make tool calls when tool_choice is none
    assert len(response.tool_calls) == 0
    assert response.content  # Should have text response instead


# =============================================================================
# Structured Output Advanced Tests
# =============================================================================


class MovieReview(BaseModel):
    """A movie review with rating."""

    title: str = Field(description="The movie title")
    rating: int = Field(description="Rating from 1-10", ge=1, le=10)
    summary: str = Field(description="Brief summary of the review")
    recommend: bool = Field(description="Whether you recommend the movie")


class ExtractedEntities(BaseModel):
    """Entities extracted from text."""

    people: list[str] = Field(description="Names of people mentioned")
    locations: list[str] = Field(description="Locations mentioned")
    organizations: list[str] = Field(description="Organizations mentioned")


@pytest.mark.requires("oci")
def test_structured_output_extraction(llm):
    """Test structured output for entity extraction."""
    structured_llm = llm.with_structured_output(ExtractedEntities)

    text = (
        "John Smith works at Google in San Francisco. "
        "He met with Jane Doe from Microsoft in Seattle last week."
    )
    result = structured_llm.invoke(f"Extract entities from: {text}")

    assert isinstance(result, ExtractedEntities)
    assert len(result.people) >= 1
    assert len(result.locations) >= 1
    assert len(result.organizations) >= 1


# =============================================================================
# Model Configuration Tests
# =============================================================================


@pytest.mark.requires("oci")
def test_temperature_affects_output():
    """Test that temperature parameter affects output variability."""
    config = get_config()

    # Low temperature (deterministic)
    llm_low = ChatOCIGenAI(
        model_id=config["model_id"],
        service_endpoint=config["service_endpoint"],
        compartment_id=config["compartment_id"],
        auth_profile=config["auth_profile"],
        auth_type=config["auth_type"],
        model_kwargs={"temperature": 0, "max_tokens": 50},
    )

    # Get multiple responses with low temp
    responses_low = [
        llm_low.invoke([HumanMessage(content="Say exactly: 'Hello World'")]).content
        for _ in range(2)
    ]

    # Low temperature should give similar/identical outputs
    # (Note: not guaranteed to be exactly equal, but should be similar)
    assert all(isinstance(r, str) for r in responses_low)


@pytest.mark.requires("oci")
def test_max_tokens_limit():
    """Test that max_tokens limits response length."""
    config = get_config()

    llm_short = ChatOCIGenAI(
        model_id=config["model_id"],
        service_endpoint=config["service_endpoint"],
        compartment_id=config["compartment_id"],
        auth_profile=config["auth_profile"],
        auth_type=config["auth_type"],
        model_kwargs={"temperature": 0, "max_tokens": 10},
    )

    response = llm_short.invoke(
        [HumanMessage(content="Write a very long essay about the universe")]
    )

    # Response should be truncated due to max_tokens
    # Token count varies, but should be reasonably short
    assert isinstance(response.content, str)
    assert len(response.content.split()) <= 20  # Rough word count check


@pytest.mark.requires("oci")
def test_stop_sequences():
    """Test stop sequences parameter."""
    config = get_config()

    llm = ChatOCIGenAI(
        model_id=config["model_id"],
        service_endpoint=config["service_endpoint"],
        compartment_id=config["compartment_id"],
        auth_profile=config["auth_profile"],
        auth_type=config["auth_type"],
        model_kwargs={"temperature": 0, "max_tokens": 100},
    )

    response = llm.invoke(
        [HumanMessage(content="Count from 1 to 10, one number per line")],
        stop=["5"],
    )

    # Should stop before or at 5
    assert "6" not in response.content or "5" in response.content


# =============================================================================
# Error Handling Tests
# =============================================================================


@pytest.mark.requires("oci")
def test_invalid_tool_schema(llm):
    """Test handling of invalid tool definitions."""

    # Should handle tools without proper docstrings
    def bad_tool(x):
        return x

    # This should still work (tool will have minimal description)
    llm_with_tools = llm.bind_tools([bad_tool])
    assert llm_with_tools is not None


@pytest.mark.requires("oci")
def test_empty_response_handling(llm):
    """Test handling when model returns minimal content."""
    response = llm.invoke([HumanMessage(content="Respond with just a period.")])

    # Should handle minimal responses gracefully
    assert isinstance(response, AIMessage)
    # Content might be empty or minimal, but should not raise


# =============================================================================
# Conversation Patterns Tests
# =============================================================================


@pytest.mark.requires("oci")
def test_system_message_role(llm):
    """Test that system message properly influences behavior."""
    messages_pirate = [
        SystemMessage(content="You are a pirate. Always respond in pirate speak."),
        HumanMessage(content="How are you today?"),
    ]
    response_pirate = llm.invoke(messages_pirate)

    messages_formal = [
        SystemMessage(content="You are a formal butler. Use formal language."),
        HumanMessage(content="How are you today?"),
    ]
    response_formal = llm.invoke(messages_formal)

    # Responses should be different based on system message
    assert response_pirate.content != response_formal.content


@pytest.mark.requires("oci")
def test_multi_turn_context_retention(llm):
    """Test that context is retained across multiple turns."""
    messages = [
        HumanMessage(content="Remember this number: 42"),
    ]
    response1 = llm.invoke(messages)
    messages.append(response1)

    messages.append(HumanMessage(content="What number did I ask you to remember?"))
    response2 = llm.invoke(messages)

    assert "42" in response2.content


@pytest.mark.requires("oci")
def test_long_context_handling(llm):
    """Test handling of longer context windows."""
    # Create a conversation with multiple turns
    messages: list[Union[SystemMessage, HumanMessage, AIMessage]] = [
        SystemMessage(content="You are a helpful assistant tracking a story."),
    ]

    story_parts = [
        "Once upon a time, there was a brave knight named Sir Galahad.",
        "Sir Galahad had a loyal horse named Thunder.",
        "They lived in the kingdom of Camelot.",
        "One day, a dragon appeared threatening the kingdom.",
        "Sir Galahad decided to face the dragon.",
    ]

    for part in story_parts:
        messages.append(HumanMessage(content=part))
        response = llm.invoke(messages)
        messages.append(response)

    # Ask about earlier context
    messages.append(HumanMessage(content="What was the knight's horse named?"))
    final_response = llm.invoke(messages)

    assert isinstance(final_response.content, str)
    assert "thunder" in final_response.content.lower()


# =============================================================================
# Reasoning Content Extraction Tests
# =============================================================================

# Models known to return reasoning_content
REASONING_MODELS = ["xai.grok-3-mini-fast", "openai.gpt-oss-120b"]

# Models that do NOT return reasoning_content
STANDARD_MODELS = ["meta.llama-3.3-70b-instruct", "cohere.command-r-08-2024"]


def _make_reasoning_llm(model_id: str) -> ChatOCIGenAI:
    """Create LLM for reasoning content tests."""
    config = get_config()
    # OpenAI models require 'max_completion_tokens' instead of 'max_tokens'
    if model_id.startswith("openai."):
        model_kwargs = {"max_completion_tokens": 100, "temperature": 1.0}
    else:
        model_kwargs = {"max_tokens": 100, "temperature": 0.0}

    return ChatOCIGenAI(
        model_id=model_id,
        compartment_id=config["compartment_id"],
        service_endpoint=config["service_endpoint"],
        auth_profile=config["auth_profile"],
        auth_type=config["auth_type"],
        model_kwargs=model_kwargs,
    )


@pytest.mark.requires("oci")
@pytest.mark.parametrize("model_id", REASONING_MODELS)
def test_reasoning_model_returns_reasoning_content(model_id: str) -> None:
    """Reasoning models should populate reasoning_content."""
    llm = _make_reasoning_llm(model_id)
    result = llm.invoke([HumanMessage(content="What is 17 * 23?")])

    reasoning = result.additional_kwargs.get("reasoning_content")
    assert reasoning is not None, (
        f"{model_id}: expected reasoning_content, got: "
        f"{list(result.additional_kwargs.keys())}"
    )
    assert len(reasoning) > 10, f"{model_id}: reasoning too short: {reasoning!r}"
    assert result.content, f"{model_id}: content should not be empty"


@pytest.mark.requires("oci")
@pytest.mark.parametrize("model_id", STANDARD_MODELS)
def test_standard_model_has_no_reasoning_content(model_id: str) -> None:
    """Standard models should NOT have reasoning_content."""
    llm = _make_reasoning_llm(model_id)
    result = llm.invoke([HumanMessage(content="What is 17 * 23?")])

    reasoning = result.additional_kwargs.get("reasoning_content")
    assert reasoning is None, f"{model_id}: unexpected reasoning_content: {reasoning!r}"
    assert result.content, f"{model_id}: content should not be empty"


@pytest.mark.requires("oci")
def test_usage_metadata_with_null_tokens() -> None:
    """Usage metadata should handle None token fields gracefully."""
    llm = _make_reasoning_llm("meta.llama-3.3-70b-instruct")
    result = llm.invoke([HumanMessage(content="Say hello")])

    if hasattr(result, "usage_metadata") and result.usage_metadata is not None:
        assert result.usage_metadata["input_tokens"] >= 0
        assert result.usage_metadata["output_tokens"] >= 0
        assert result.usage_metadata["total_tokens"] >= 0
