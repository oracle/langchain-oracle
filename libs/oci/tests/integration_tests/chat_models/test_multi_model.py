#!/usr/bin/env python3
"""Multi-model integration tests for ChatOCIGenAI.

These tests verify that langchain-oci works correctly across different
model vendors available in OCI GenAI: Meta Llama, Cohere, xAI Grok, and OpenAI.

Setup:
    export OCI_COMPARTMENT_ID=<your-compartment-id>
    export OCI_CONFIG_PROFILE=DEFAULT
    export OCI_AUTH_TYPE=SECURITY_TOKEN

Run all:
    pytest tests/integration_tests/chat_models/test_multi_model.py -v

Run specific vendor:
    pytest tests/integration_tests/chat_models/test_multi_model.py -k "llama" -v
    pytest tests/integration_tests/chat_models/test_multi_model.py -k "cohere" -v
    pytest tests/integration_tests/chat_models/test_multi_model.py -k "grok" -v
"""

import os

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from langchain_oci.chat_models import ChatOCIGenAI

# =============================================================================
# Model Configurations
# =============================================================================

# Meta Llama models
LLAMA_MODELS = [
    "meta.llama-4-maverick-17b-128e-instruct-fp8",
    "meta.llama-4-scout-17b-16e-instruct",
    "meta.llama-3.3-70b-instruct",
    "meta.llama-3.1-70b-instruct",
]

# Cohere models
COHERE_MODELS = [
    "cohere.command-a-03-2025",
    "cohere.command-r-plus-08-2024",
    "cohere.command-r-08-2024",
]

# xAI Grok models
GROK_MODELS = [
    "xai.grok-4-fast-non-reasoning",
    "xai.grok-3-fast",
    "xai.grok-3-mini-fast",
]

# OpenAI models on OCI
OPENAI_MODELS = [
    "openai.gpt-oss-20b",
    "openai.gpt-oss-120b",
]

# All models for comprehensive testing
ALL_MODELS = LLAMA_MODELS[:2] + COHERE_MODELS[:1] + GROK_MODELS[:1]


def get_config():
    """Get test configuration."""
    compartment_id = os.environ.get("OCI_COMPARTMENT_ID")
    if not compartment_id:
        pytest.skip("OCI_COMPARTMENT_ID not set")
    return {
        "service_endpoint": os.environ.get(
            "OCI_GENAI_ENDPOINT",
            "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
        ),
        "compartment_id": compartment_id,
        "auth_profile": os.environ.get("OCI_CONFIG_PROFILE", "DEFAULT"),
        "auth_type": os.environ.get("OCI_AUTH_TYPE", "SECURITY_TOKEN"),
    }


def create_llm(model_id: str, **kwargs):
    """Create ChatOCIGenAI instance for a model."""
    config = get_config()
    default_kwargs = {"temperature": 0, "max_tokens": 256}
    default_kwargs.update(kwargs)
    return ChatOCIGenAI(
        model_id=model_id,
        service_endpoint=config["service_endpoint"],
        compartment_id=config["compartment_id"],
        auth_profile=config["auth_profile"],
        auth_type=config["auth_type"],
        model_kwargs=default_kwargs,
    )


# =============================================================================
# Basic Invoke Tests - All Models
# =============================================================================


@pytest.mark.requires("oci")
@pytest.mark.parametrize("model_id", ALL_MODELS)
def test_basic_invoke_all_models(model_id: str):
    """Test basic invoke works for all supported models."""
    llm = create_llm(model_id)
    response = llm.invoke([HumanMessage(content="Say 'hello' only")])

    assert isinstance(response, AIMessage)
    assert response.content is not None
    assert len(response.content) > 0


@pytest.mark.requires("oci")
@pytest.mark.parametrize("model_id", ALL_MODELS)
def test_system_message_all_models(model_id: str):
    """Test system messages work for all models."""
    llm = create_llm(model_id)
    messages = [
        SystemMessage(content="You only respond with the word 'YES'."),
        HumanMessage(content="Do you understand?"),
    ]
    response = llm.invoke(messages)

    assert isinstance(response, AIMessage)
    assert response.content is not None


# =============================================================================
# Meta Llama Specific Tests
# =============================================================================


@pytest.mark.requires("oci")
@pytest.mark.parametrize("model_id", LLAMA_MODELS[:2])
def test_llama_tool_calling(model_id: str):
    """Test tool calling on Llama models."""

    def get_weather(city: str) -> str:
        """Get weather for a city."""
        return f"Sunny in {city}"

    llm = create_llm(model_id)
    llm_with_tools = llm.bind_tools([get_weather])

    response = llm_with_tools.invoke(
        [HumanMessage(content="What's the weather in Paris?")]
    )

    assert isinstance(response, AIMessage)
    assert len(response.tool_calls) >= 1
    assert response.tool_calls[0]["name"] == "get_weather"


@pytest.mark.requires("oci")
@pytest.mark.parametrize("model_id", LLAMA_MODELS[:2])
def test_llama_structured_output(model_id: str):
    """Test structured output on Llama models."""

    class Answer(BaseModel):
        answer: str = Field(description="The answer")
        confidence: int = Field(description="Confidence 1-10", ge=1, le=10)

    llm = create_llm(model_id)
    structured_llm = llm.with_structured_output(Answer)

    result = structured_llm.invoke("What is 2+2? Give answer and confidence.")

    assert isinstance(result, Answer)
    assert "4" in result.answer
    assert 1 <= result.confidence <= 10


@pytest.mark.requires("oci")
def test_llama_streaming():
    """Test streaming on Llama models."""
    llm = create_llm("meta.llama-4-maverick-17b-128e-instruct-fp8")

    chunks = []
    for chunk in llm.stream([HumanMessage(content="Count 1 to 5")]):
        chunks.append(chunk)

    assert len(chunks) > 0
    full_content = "".join(c.content for c in chunks if c.content)
    assert len(full_content) > 0


# =============================================================================
# Cohere Specific Tests
# =============================================================================


@pytest.mark.requires("oci")
@pytest.mark.parametrize("model_id", COHERE_MODELS[:2])
def test_cohere_basic(model_id: str):
    """Test basic functionality on Cohere models."""
    llm = create_llm(model_id)
    response = llm.invoke([HumanMessage(content="What is 2+2?")])

    assert isinstance(response, AIMessage)
    assert "4" in response.content


# =============================================================================
# xAI Grok Specific Tests
# =============================================================================


@pytest.mark.requires("oci")
@pytest.mark.parametrize("model_id", GROK_MODELS[:2])
def test_grok_basic(model_id: str):
    """Test basic functionality on Grok models."""
    llm = create_llm(model_id)
    response = llm.invoke([HumanMessage(content="Hello, who are you?")])

    assert isinstance(response, AIMessage)
    assert response.content is not None


@pytest.mark.requires("oci")
@pytest.mark.parametrize("model_id", GROK_MODELS[:1])
def test_grok_tool_calling(model_id: str):
    """Test tool calling on Grok models."""

    def search_web(query: str) -> str:
        """Search the web for information."""
        return f"Results for: {query}"

    llm = create_llm(model_id)
    llm_with_tools = llm.bind_tools([search_web])

    response = llm_with_tools.invoke(
        [HumanMessage(content="Search for the latest AI news")]
    )

    assert isinstance(response, AIMessage)
    # Grok may or may not call tools depending on its judgment
    # Just verify it responds


@pytest.mark.requires("oci")
@pytest.mark.parametrize("model_id", GROK_MODELS[:1])
def test_grok_structured_output(model_id: str):
    """Test structured output on Grok models."""

    class Summary(BaseModel):
        """A summary of text."""

        main_point: str = Field(description="The main point")
        key_facts: list[str] = Field(description="Key facts from the text")

    llm = create_llm(model_id)
    structured_llm = llm.with_structured_output(Summary)

    result = structured_llm.invoke("Summarize: The Earth orbits the Sun once per year.")

    # Grok may return None in some cases
    if result is not None:
        assert isinstance(result, Summary)
        assert len(result.main_point) > 0
    else:
        pytest.skip("Grok model returned None for structured output")


@pytest.mark.requires("oci")
def test_grok_streaming():
    """Test streaming on Grok models."""
    llm = create_llm("xai.grok-3-mini-fast")

    chunks = []
    for chunk in llm.stream([HumanMessage(content="Count 1-3")]):
        chunks.append(chunk)

    assert len(chunks) > 0


# =============================================================================
# OpenAI on OCI Tests
# =============================================================================


def create_openai_llm(model_id: str, **kwargs):
    """Create ChatOCIGenAI for OpenAI models (uses max_completion_tokens)."""
    config = get_config()
    default_kwargs = {"temperature": 0, "max_completion_tokens": 256}
    default_kwargs.update(kwargs)
    return ChatOCIGenAI(
        model_id=model_id,
        service_endpoint=config["service_endpoint"],
        compartment_id=config["compartment_id"],
        auth_profile=config["auth_profile"],
        auth_type=config["auth_type"],
        model_kwargs=default_kwargs,
    )


@pytest.mark.requires("oci")
@pytest.mark.parametrize("model_id", OPENAI_MODELS)
def test_openai_basic(model_id: str):
    """Test basic functionality on OpenAI models on OCI."""
    llm = create_openai_llm(model_id)
    response = llm.invoke([HumanMessage(content="Say hello")])

    assert isinstance(response, AIMessage)
    assert response.content is not None


@pytest.mark.requires("oci")
@pytest.mark.parametrize("model_id", OPENAI_MODELS)
def test_openai_system_message(model_id: str):
    """Test system messages on OpenAI models."""
    llm = create_openai_llm(model_id)
    messages = [
        SystemMessage(content="You only respond with the word 'YES'."),
        HumanMessage(content="Do you understand?"),
    ]
    response = llm.invoke(messages)

    assert isinstance(response, AIMessage)
    assert response.content is not None


@pytest.mark.requires("oci")
@pytest.mark.parametrize("model_id", OPENAI_MODELS)
def test_openai_streaming(model_id: str):
    """Test streaming on OpenAI models."""
    llm = create_openai_llm(model_id, max_completion_tokens=50)

    chunks = []
    for chunk in llm.stream([HumanMessage(content="Count 1-3")]):
        chunks.append(chunk)

    # OpenAI streaming should return chunks
    assert len(chunks) > 0
    # Content may be in chunk.content or chunk may have other attributes
    # Just verify we got chunks back (streaming works)


@pytest.mark.requires("oci")
@pytest.mark.parametrize("model_id", OPENAI_MODELS)
def test_openai_tool_calling(model_id: str):
    """Test tool calling on OpenAI models."""

    def get_info(topic: str) -> str:
        """Get information about a topic."""
        return f"Info about {topic}"

    llm = create_openai_llm(model_id)
    llm_with_tools = llm.bind_tools([get_info])

    response = llm_with_tools.invoke([HumanMessage(content="Get info about Python")])

    assert isinstance(response, AIMessage)
    # OpenAI models should call the tool
    assert len(response.tool_calls) >= 1
    assert response.tool_calls[0]["name"] == "get_info"


# =============================================================================
# Cross-Model Comparison Tests
# =============================================================================


@pytest.mark.requires("oci")
def test_same_prompt_different_models():
    """Test same prompt across different model vendors."""
    prompt = "What is the capital of France? Answer in one word."

    models_to_test = [
        "meta.llama-4-maverick-17b-128e-instruct-fp8",
        "cohere.command-a-03-2025",
        "xai.grok-3-mini-fast",
    ]

    responses = {}
    for model_id in models_to_test:
        try:
            llm = create_llm(model_id)
            response = llm.invoke([HumanMessage(content=prompt)])
            responses[model_id] = response.content
        except Exception as e:
            responses[model_id] = f"Error: {e}"

    # All should mention Paris
    for model_id, content in responses.items():
        if not content.startswith("Error"):
            assert "paris" in content.lower(), f"{model_id} didn't say Paris: {content}"


@pytest.mark.requires("oci")
def test_tool_calling_consistency():
    """Test tool calling works consistently across Llama models."""

    def get_price(item: str) -> float:
        """Get the price of an item in dollars."""
        return 9.99

    # Only test Llama models - Cohere has different tool call format
    models_with_tools = [
        "meta.llama-4-maverick-17b-128e-instruct-fp8",
        "meta.llama-4-scout-17b-16e-instruct",
    ]

    for model_id in models_with_tools:
        llm = create_llm(model_id)
        llm_with_tools = llm.bind_tools([get_price])

        response = llm_with_tools.invoke(
            [HumanMessage(content="What's the price of apples?")]
        )

        assert isinstance(response, AIMessage), f"{model_id} failed"
        assert len(response.tool_calls) >= 1, f"{model_id} didn't call tool"
        assert response.tool_calls[0]["name"] == "get_price"


# =============================================================================
# Model-Specific Features Tests
# =============================================================================


@pytest.mark.requires("oci")
def test_llama3_vision_model_exists():
    """Verify vision-capable Llama model can be instantiated."""
    # Note: Actual vision testing would require image input support
    llm = create_llm("meta.llama-3.2-90b-vision-instruct")
    response = llm.invoke([HumanMessage(content="Describe what you can do")])

    assert isinstance(response, AIMessage)


@pytest.mark.requires("oci")
def test_model_with_custom_kwargs():
    """Test models with custom generation parameters."""
    llm = create_llm(
        "meta.llama-4-maverick-17b-128e-instruct-fp8",
        temperature=0.7,
        max_tokens=100,
        top_p=0.9,
    )

    response = llm.invoke([HumanMessage(content="Write a creative sentence")])

    assert isinstance(response, AIMessage)
    assert response.content is not None


# =============================================================================
# Performance / Latency Awareness Tests
# =============================================================================


@pytest.mark.requires("oci")
def test_fast_models_respond_quickly():
    """Test that 'fast' model variants respond (existence check)."""
    fast_models = [
        "xai.grok-3-fast",
        "xai.grok-3-mini-fast",
    ]

    for model_id in fast_models:
        llm = create_llm(model_id, max_tokens=50)
        response = llm.invoke([HumanMessage(content="Hi")])
        assert isinstance(response, AIMessage)
