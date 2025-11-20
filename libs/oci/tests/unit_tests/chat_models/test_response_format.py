"""Unit tests for response_format feature."""

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import HumanMessage

from langchain_oci.chat_models import ChatOCIGenAI


@pytest.mark.requires("oci")
def test_response_format_via_model_kwargs():
    """Test response_format via model_kwargs."""
    oci_gen_ai_client = MagicMock()
    llm = ChatOCIGenAI(
        model_id="meta.llama-3.3-70b-instruct",
        model_kwargs={"response_format": {"type": "JSON_OBJECT"}},
        client=oci_gen_ai_client,
    )
    assert llm.model_kwargs["response_format"] == {"type": "JSON_OBJECT"}


@pytest.mark.requires("oci")
def test_response_format_default_not_in_model_kwargs():
    """Test that response_format is not set by default."""
    oci_gen_ai_client = MagicMock()
    llm = ChatOCIGenAI(model_id="meta.llama-3.3-70b-instruct", client=oci_gen_ai_client)
    assert llm.model_kwargs is None or "response_format" not in llm.model_kwargs


@pytest.mark.requires("oci")
def test_response_format_via_bind():
    """Test response_format set via bind()."""
    oci_gen_ai_client = MagicMock()
    llm = ChatOCIGenAI(model_id="meta.llama-3.3-70b-instruct", client=oci_gen_ai_client)

    # Should not raise TypeError anymore
    llm_with_format = llm.bind(response_format={"type": "JSON_OBJECT"})

    assert "response_format" in llm_with_format.kwargs
    assert llm_with_format.kwargs["response_format"] == {"type": "JSON_OBJECT"}


@pytest.mark.requires("oci")
def test_response_format_passed_to_api_generic():
    """Test that response_format is passed to OCI API for Generic models."""

    oci_gen_ai_client = MagicMock()
    llm = ChatOCIGenAI(model_id="meta.llama-3.3-70b-instruct", client=oci_gen_ai_client)

    llm_with_format = llm.bind(response_format={"type": "JSON_OBJECT"})

    # Prepare a request
    request = llm_with_format._prepare_request(
        [HumanMessage(content="Hello")],
        stop=None,
        stream=False,
        **llm_with_format.kwargs,
    )

    # Verify response_format is in the request
    assert hasattr(request.chat_request, "response_format")
    assert request.chat_request.response_format == {"type": "JSON_OBJECT"}


@pytest.mark.requires("oci")
def test_response_format_passed_to_api_cohere():
    """Test that response_format is passed to OCI API for Cohere models."""

    oci_gen_ai_client = MagicMock()
    llm = ChatOCIGenAI(model_id="cohere.command-r-plus", client=oci_gen_ai_client)

    llm_with_format = llm.bind(response_format={"type": "JSON_OBJECT"})

    # Prepare a request
    request = llm_with_format._prepare_request(
        [HumanMessage(content="Hello")],
        stop=None,
        stream=False,
        **llm_with_format.kwargs,
    )

    # Verify response_format is in the request
    assert hasattr(request.chat_request, "response_format")
    assert request.chat_request.response_format == {"type": "JSON_OBJECT"}


@pytest.mark.requires("oci")
def test_with_structured_output_json_mode():
    """Test with_structured_output with json_mode method."""
    oci_gen_ai_client = MagicMock()
    llm = ChatOCIGenAI(model_id="cohere.command-r-plus", client=oci_gen_ai_client)

    # This should not raise TypeError anymore
    from pydantic import BaseModel

    class TestSchema(BaseModel):
        name: str
        age: int

    structured_llm = llm.with_structured_output(schema=TestSchema, method="json_mode")

    # The structured LLM should have response_format in kwargs
    # It's wrapped in a Runnable, so we need to check the first step
    assert structured_llm is not None


@pytest.mark.requires("oci")
def test_with_structured_output_json_schema():
    """Test with_structured_output with json_schema method."""
    oci_gen_ai_client = MagicMock()
    llm = ChatOCIGenAI(model_id="meta.llama-3.3-70b-instruct", client=oci_gen_ai_client)

    # This should not raise TypeError anymore
    from pydantic import BaseModel

    class TestSchema(BaseModel):
        """Test schema"""

        name: str
        age: int

    structured_llm = llm.with_structured_output(schema=TestSchema, method="json_schema")

    # The structured LLM should be created without errors
    assert structured_llm is not None


@pytest.mark.requires("oci")
def test_response_format_json_schema_object():
    """Test response_format with JsonSchemaResponseFormat object."""
    from oci.generative_ai_inference import models

    oci_gen_ai_client = MagicMock()
    llm = ChatOCIGenAI(model_id="meta.llama-3.3-70b-instruct", client=oci_gen_ai_client)

    # Create a proper JsonSchemaResponseFormat object
    response_json_schema = models.ResponseJsonSchema(
        name="test_response",
        description="Test schema",
        schema={"type": "object", "properties": {"key": {"type": "string"}}},
        is_strict=True,
    )

    response_format_obj = models.JsonSchemaResponseFormat(
        json_schema=response_json_schema
    )

    llm_with_format = llm.bind(response_format=response_format_obj)

    # Verify it's stored in kwargs
    assert "response_format" in llm_with_format.kwargs
    assert llm_with_format.kwargs["response_format"] == response_format_obj


@pytest.mark.requires("oci")
def test_response_format_model_kwargs():
    """Test response_format via model_kwargs."""
    oci_gen_ai_client = MagicMock()
    llm = ChatOCIGenAI(
        model_id="meta.llama-3.3-70b-instruct",
        model_kwargs={"response_format": {"type": "JSON_OBJECT"}},
        client=oci_gen_ai_client,
    )

    request = llm._prepare_request(
        [HumanMessage(content="Hello")], stop=None, stream=False
    )

    # Verify response_format is in the request
    assert hasattr(request.chat_request, "response_format")
    assert request.chat_request.response_format == {"type": "JSON_OBJECT"}
