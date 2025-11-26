"""Unit tests for parallel tool calling feature."""

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import HumanMessage

from langchain_oci.chat_models import ChatOCIGenAI


@pytest.mark.requires("oci")
def test_parallel_tool_calls_bind_tools_explicit_true():
    """Test parallel_tool_calls=True in bind_tools."""
    oci_gen_ai_client = MagicMock()
    llm = ChatOCIGenAI(
        model_id="meta.llama-4-maverick-17b-128e-instruct-fp8", client=oci_gen_ai_client
    )

    def tool1(x: int) -> int:
        """Tool 1."""
        return x + 1

    def tool2(x: int) -> int:
        """Tool 2."""
        return x * 2

    llm_with_tools = llm.bind_tools([tool1, tool2], parallel_tool_calls=True)

    # RunnableBinding has kwargs attribute at runtime
    assert llm_with_tools.kwargs.get("is_parallel_tool_calls") is True  # type: ignore[attr-defined]


@pytest.mark.requires("oci")
def test_parallel_tool_calls_bind_tools_explicit_false():
    """Test parallel_tool_calls=False in bind_tools."""
    oci_gen_ai_client = MagicMock()
    llm = ChatOCIGenAI(
        model_id="meta.llama-4-maverick-17b-128e-instruct-fp8", client=oci_gen_ai_client
    )

    def tool1(x: int) -> int:
        """Tool 1."""
        return x + 1

    llm_with_tools = llm.bind_tools([tool1], parallel_tool_calls=False)

    # When explicitly False, should not set the parameter
    # RunnableBinding has kwargs attribute at runtime
    assert "is_parallel_tool_calls" not in llm_with_tools.kwargs  # type: ignore[attr-defined]


@pytest.mark.requires("oci")
def test_parallel_tool_calls_bind_tools_default_none():
    """Test that bind_tools without parallel_tool_calls doesn't enable it."""
    oci_gen_ai_client = MagicMock()
    llm = ChatOCIGenAI(
        model_id="meta.llama-4-maverick-17b-128e-instruct-fp8", client=oci_gen_ai_client
    )

    def tool1(x: int) -> int:
        """Tool 1."""
        return x + 1

    # Don't specify parallel_tool_calls in bind_tools
    llm_with_tools = llm.bind_tools([tool1])

    # Should not have is_parallel_tool_calls set
    # RunnableBinding has kwargs attribute at runtime
    assert "is_parallel_tool_calls" not in llm_with_tools.kwargs  # type: ignore[attr-defined]


@pytest.mark.requires("oci")
def test_parallel_tool_calls_passed_to_oci_api_meta():
    """Test that is_parallel_tool_calls is passed to OCI API for Meta models."""
    oci_gen_ai_client = MagicMock()
    llm = ChatOCIGenAI(
        model_id="meta.llama-4-maverick-17b-128e-instruct-fp8", client=oci_gen_ai_client
    )

    def get_weather(city: str) -> str:
        """Get weather for a city."""
        return f"Weather in {city}"

    llm_with_tools = llm.bind_tools([get_weather], parallel_tool_calls=True)

    # Prepare a request
    # RunnableBinding has _prepare_request and kwargs attributes at runtime
    request = llm_with_tools._prepare_request(  # type: ignore[attr-defined]
        [HumanMessage(content="What's the weather?")],
        stop=None,
        stream=False,
        **llm_with_tools.kwargs,  # type: ignore[attr-defined]
    )

    # Verify is_parallel_tool_calls is in the request
    assert hasattr(request.chat_request, "is_parallel_tool_calls")
    assert request.chat_request.is_parallel_tool_calls is True


@pytest.mark.requires("oci")
def test_parallel_tool_calls_cohere_raises_error():
    """Test that Cohere models raise error for parallel tool calls at bind_tools."""
    oci_gen_ai_client = MagicMock()
    llm = ChatOCIGenAI(model_id="cohere.command-r-plus", client=oci_gen_ai_client)

    def tool1(x: int) -> int:
        """Tool 1."""
        return x + 1

    # Should raise ValueError at bind_tools time (not at request time)
    with pytest.raises(ValueError, match="not supported"):
        llm.bind_tools([tool1], parallel_tool_calls=True)


@pytest.mark.requires("oci")
def test_parallel_tool_calls_meta_allowed():
    """Test that Meta models are allowed parallel tool calling."""
    oci_gen_ai_client = MagicMock()
    llm = ChatOCIGenAI(
        model_id="meta.llama-4-maverick-17b-128e-instruct-fp8", client=oci_gen_ai_client
    )

    def tool1(x: int) -> int:
        """Tool 1."""
        return x + 1

    # Should NOT raise ValueError
    llm_with_tools = llm.bind_tools([tool1], parallel_tool_calls=True)
    # RunnableBinding has kwargs attribute at runtime
    assert llm_with_tools.kwargs.get("is_parallel_tool_calls") is True  # type: ignore[attr-defined]


@pytest.mark.requires("oci")
def test_parallel_tool_calls_other_generic_models_allowed():
    """Test that other GenericChatRequest models are allowed parallel tool calling."""
    oci_gen_ai_client = MagicMock()

    # Test with xAI Grok (uses GenericProvider)
    llm_grok = ChatOCIGenAI(model_id="xai.grok-4-fast", client=oci_gen_ai_client)

    def tool1(x: int) -> int:
        """Tool 1."""
        return x + 1

    # Should NOT raise ValueError for Grok
    llm_with_tools = llm_grok.bind_tools([tool1], parallel_tool_calls=True)
    # RunnableBinding has kwargs attribute at runtime
    assert llm_with_tools.kwargs.get("is_parallel_tool_calls") is True  # type: ignore[attr-defined]


@pytest.mark.requires("oci")
def test_provider_supports_parallel_tool_calls_property():
    """Test the provider supports_parallel_tool_calls property."""
    oci_gen_ai_client = MagicMock()

    # Meta model uses GenericProvider which supports parallel tool calls
    llm_meta = ChatOCIGenAI(
        model_id="meta.llama-4-maverick-17b-128e-instruct-fp8", client=oci_gen_ai_client
    )
    assert llm_meta._provider.supports_parallel_tool_calls is True

    # Cohere model uses CohereProvider which does NOT support parallel tool calls
    llm_cohere = ChatOCIGenAI(
        model_id="cohere.command-r-plus", client=oci_gen_ai_client
    )
    assert llm_cohere._provider.supports_parallel_tool_calls is False
