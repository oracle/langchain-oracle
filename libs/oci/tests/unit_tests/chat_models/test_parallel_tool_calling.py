"""Unit tests for parallel tool calling feature."""
import pytest
from unittest.mock import MagicMock

from langchain_core.messages import HumanMessage
from langchain_oci.chat_models import ChatOCIGenAI


@pytest.mark.requires("oci")
def test_parallel_tool_calls_class_level():
    """Test class-level parallel_tool_calls parameter."""
    oci_gen_ai_client = MagicMock()
    llm = ChatOCIGenAI(
        model_id="meta.llama-3.3-70b-instruct",
        parallel_tool_calls=True,
        client=oci_gen_ai_client
    )
    assert llm.parallel_tool_calls is True


@pytest.mark.requires("oci")
def test_parallel_tool_calls_default_false():
    """Test that parallel_tool_calls defaults to False."""
    oci_gen_ai_client = MagicMock()
    llm = ChatOCIGenAI(
        model_id="meta.llama-3.3-70b-instruct",
        client=oci_gen_ai_client
    )
    assert llm.parallel_tool_calls is False


@pytest.mark.requires("oci")
def test_parallel_tool_calls_bind_tools_explicit_true():
    """Test parallel_tool_calls=True in bind_tools."""
    oci_gen_ai_client = MagicMock()
    llm = ChatOCIGenAI(
        model_id="meta.llama-3.3-70b-instruct",
        client=oci_gen_ai_client
    )

    def tool1(x: int) -> int:
        """Tool 1."""
        return x + 1

    def tool2(x: int) -> int:
        """Tool 2."""
        return x * 2

    llm_with_tools = llm.bind_tools(
        [tool1, tool2],
        parallel_tool_calls=True
    )

    assert llm_with_tools.kwargs.get("is_parallel_tool_calls") is True


@pytest.mark.requires("oci")
def test_parallel_tool_calls_bind_tools_explicit_false():
    """Test parallel_tool_calls=False in bind_tools."""
    oci_gen_ai_client = MagicMock()
    llm = ChatOCIGenAI(
        model_id="meta.llama-3.3-70b-instruct",
        client=oci_gen_ai_client
    )

    def tool1(x: int) -> int:
        """Tool 1."""
        return x + 1

    llm_with_tools = llm.bind_tools(
        [tool1],
        parallel_tool_calls=False
    )

    # When explicitly False, should not set the parameter
    assert "is_parallel_tool_calls" not in llm_with_tools.kwargs


@pytest.mark.requires("oci")
def test_parallel_tool_calls_bind_tools_uses_class_default():
    """Test that bind_tools uses class default when not specified."""
    oci_gen_ai_client = MagicMock()
    llm = ChatOCIGenAI(
        model_id="meta.llama-3.3-70b-instruct",
        parallel_tool_calls=True,  # Set class default
        client=oci_gen_ai_client
    )

    def tool1(x: int) -> int:
        """Tool 1."""
        return x + 1

    # Don't specify parallel_tool_calls in bind_tools
    llm_with_tools = llm.bind_tools([tool1])

    # Should use class default (True)
    assert llm_with_tools.kwargs.get("is_parallel_tool_calls") is True


@pytest.mark.requires("oci")
def test_parallel_tool_calls_bind_tools_overrides_class_default():
    """Test that bind_tools parameter overrides class default."""
    oci_gen_ai_client = MagicMock()
    llm = ChatOCIGenAI(
        model_id="meta.llama-3.3-70b-instruct",
        parallel_tool_calls=True,  # Set class default to True
        client=oci_gen_ai_client
    )

    def tool1(x: int) -> int:
        """Tool 1."""
        return x + 1

    # Override with False in bind_tools
    llm_with_tools = llm.bind_tools([tool1], parallel_tool_calls=False)

    # Should not set the parameter when explicitly False
    assert "is_parallel_tool_calls" not in llm_with_tools.kwargs


@pytest.mark.requires("oci")
def test_parallel_tool_calls_passed_to_oci_api_meta():
    """Test that is_parallel_tool_calls is passed to OCI API for Meta models."""
    from oci.generative_ai_inference import models

    oci_gen_ai_client = MagicMock()
    llm = ChatOCIGenAI(
        model_id="meta.llama-3.3-70b-instruct",
        client=oci_gen_ai_client
    )

    def get_weather(city: str) -> str:
        """Get weather for a city."""
        return f"Weather in {city}"

    llm_with_tools = llm.bind_tools([get_weather], parallel_tool_calls=True)

    # Prepare a request
    request = llm_with_tools._prepare_request(
        [HumanMessage(content="What's the weather?")],
        stop=None,
        stream=False,
        **llm_with_tools.kwargs
    )

    # Verify is_parallel_tool_calls is in the request
    assert hasattr(request.chat_request, 'is_parallel_tool_calls')
    assert request.chat_request.is_parallel_tool_calls is True


@pytest.mark.requires("oci")
def test_parallel_tool_calls_cohere_raises_error():
    """Test that Cohere models raise error for parallel tool calls."""
    oci_gen_ai_client = MagicMock()
    llm = ChatOCIGenAI(
        model_id="cohere.command-r-plus",
        client=oci_gen_ai_client
    )

    def tool1(x: int) -> int:
        """Tool 1."""
        return x + 1

    llm_with_tools = llm.bind_tools([tool1], parallel_tool_calls=True)

    # Should raise ValueError when trying to prepare request
    with pytest.raises(ValueError, match="not supported for Cohere"):
        llm_with_tools._prepare_request(
            [HumanMessage(content="test")],
            stop=None,
            stream=False,
            **llm_with_tools.kwargs
        )


@pytest.mark.requires("oci")
def test_parallel_tool_calls_cohere_class_level_raises_error():
    """Test that Cohere models with class-level parallel_tool_calls raise error."""
    oci_gen_ai_client = MagicMock()
    llm = ChatOCIGenAI(
        model_id="cohere.command-r-plus",
        parallel_tool_calls=True,  # Set at class level
        client=oci_gen_ai_client
    )

    def tool1(x: int) -> int:
        """Tool 1."""
        return x + 1

    llm_with_tools = llm.bind_tools([tool1])  # Uses class default

    # Should raise ValueError when trying to prepare request
    with pytest.raises(ValueError, match="not supported for Cohere"):
        llm_with_tools._prepare_request(
            [HumanMessage(content="test")],
            stop=None,
            stream=False,
            **llm_with_tools.kwargs
        )
