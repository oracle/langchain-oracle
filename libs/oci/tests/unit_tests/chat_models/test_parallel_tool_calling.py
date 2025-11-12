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
        model_id="meta.llama-4-maverick-17b-128e-instruct-fp8",
        parallel_tool_calls=True,
        client=oci_gen_ai_client
    )
    assert llm.parallel_tool_calls is True


@pytest.mark.requires("oci")
def test_parallel_tool_calls_default_false():
    """Test that parallel_tool_calls defaults to False."""
    oci_gen_ai_client = MagicMock()
    llm = ChatOCIGenAI(
        model_id="meta.llama-4-maverick-17b-128e-instruct-fp8",
        client=oci_gen_ai_client
    )
    assert llm.parallel_tool_calls is False


@pytest.mark.requires("oci")
def test_parallel_tool_calls_bind_tools_explicit_true():
    """Test parallel_tool_calls=True in bind_tools."""
    oci_gen_ai_client = MagicMock()
    llm = ChatOCIGenAI(
        model_id="meta.llama-4-maverick-17b-128e-instruct-fp8",
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
        model_id="meta.llama-4-maverick-17b-128e-instruct-fp8",
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
        model_id="meta.llama-4-maverick-17b-128e-instruct-fp8",
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
        model_id="meta.llama-4-maverick-17b-128e-instruct-fp8",
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
        model_id="meta.llama-4-maverick-17b-128e-instruct-fp8",
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


@pytest.mark.requires("oci")
def test_version_filter_llama_3_0_blocked():
    """Test that Llama 3.0 models are blocked from parallel tool calling."""
    oci_gen_ai_client = MagicMock()
    llm = ChatOCIGenAI(
        model_id="meta.llama-3-70b-instruct",
        client=oci_gen_ai_client
    )

    def tool1(x: int) -> int:
        """Tool 1."""
        return x + 1

    # Should raise ValueError when trying to enable parallel tool calling
    with pytest.raises(ValueError, match="Llama 4\\+"):
        llm.bind_tools([tool1], parallel_tool_calls=True)


@pytest.mark.requires("oci")
def test_version_filter_llama_3_1_blocked():
    """Test that Llama 3.1 models are blocked from parallel tool calling."""
    oci_gen_ai_client = MagicMock()
    llm = ChatOCIGenAI(
        model_id="meta.llama-3.1-70b-instruct",
        client=oci_gen_ai_client
    )

    def tool1(x: int) -> int:
        """Tool 1."""
        return x + 1

    # Should raise ValueError
    with pytest.raises(ValueError, match="Llama 4\\+"):
        llm.bind_tools([tool1], parallel_tool_calls=True)


@pytest.mark.requires("oci")
def test_version_filter_llama_3_2_blocked():
    """Test that Llama 3.2 models are blocked from parallel tool calling."""
    oci_gen_ai_client = MagicMock()
    llm = ChatOCIGenAI(
        model_id="meta.llama-3.2-11b-vision-instruct",
        client=oci_gen_ai_client
    )

    def tool1(x: int) -> int:
        """Tool 1."""
        return x + 1

    # Should raise ValueError
    with pytest.raises(ValueError, match="Llama 4\\+"):
        llm.bind_tools([tool1], parallel_tool_calls=True)


@pytest.mark.requires("oci")
def test_version_filter_llama_3_3_blocked():
    """Test that Llama 3.3 models are blocked from parallel tool calling."""
    oci_gen_ai_client = MagicMock()
    llm = ChatOCIGenAI(
        model_id="meta.llama-3.3-70b-instruct",
        client=oci_gen_ai_client
    )

    def tool1(x: int) -> int:
        """Tool 1."""
        return x + 1

    # Should raise ValueError - Llama 3.3 doesn't actually support parallel calls
    with pytest.raises(ValueError, match="Llama 4\\+"):
        llm.bind_tools([tool1], parallel_tool_calls=True)


@pytest.mark.requires("oci")
def test_version_filter_llama_4_allowed():
    """Test that Llama 4 models are allowed parallel tool calling."""
    oci_gen_ai_client = MagicMock()
    llm = ChatOCIGenAI(
        model_id="meta.llama-4-maverick-17b-128e-instruct-fp8",
        client=oci_gen_ai_client
    )

    def tool1(x: int) -> int:
        """Tool 1."""
        return x + 1

    # Should NOT raise ValueError
    llm_with_tools = llm.bind_tools([tool1], parallel_tool_calls=True)
    assert llm_with_tools.kwargs.get("is_parallel_tool_calls") is True


@pytest.mark.requires("oci")
def test_version_filter_other_models_allowed():
    """Test that other GenericChatRequest models are allowed parallel tool calling."""
    oci_gen_ai_client = MagicMock()

    # Test with xAI Grok
    llm_grok = ChatOCIGenAI(
        model_id="xai.grok-4-fast",
        client=oci_gen_ai_client
    )

    def tool1(x: int) -> int:
        """Tool 1."""
        return x + 1

    # Should NOT raise ValueError for Grok
    llm_with_tools = llm_grok.bind_tools([tool1], parallel_tool_calls=True)
    assert llm_with_tools.kwargs.get("is_parallel_tool_calls") is True


@pytest.mark.requires("oci")
def test_version_filter_supports_parallel_tool_calls_method():
    """Test the _supports_parallel_tool_calls method directly."""
    oci_gen_ai_client = MagicMock()
    llm = ChatOCIGenAI(
        model_id="meta.llama-4-maverick-17b-128e-instruct-fp8",
        client=oci_gen_ai_client
    )

    # Test various model IDs
    assert llm._supports_parallel_tool_calls("meta.llama-4-maverick-17b-128e-instruct-fp8") is True
    assert llm._supports_parallel_tool_calls("meta.llama-3.3-70b-instruct") is False  # Llama 3.3 NOT supported
    assert llm._supports_parallel_tool_calls("meta.llama-3.2-11b-vision-instruct") is False
    assert llm._supports_parallel_tool_calls("meta.llama-3.1-70b-instruct") is False
    assert llm._supports_parallel_tool_calls("meta.llama-3-70b-instruct") is False
    assert llm._supports_parallel_tool_calls("cohere.command-r-plus") is False
    assert llm._supports_parallel_tool_calls("xai.grok-4-fast") is True
    assert llm._supports_parallel_tool_calls("openai.gpt-4") is True
    assert llm._supports_parallel_tool_calls("mistral.mistral-large") is True
