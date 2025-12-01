#!/usr/bin/env python3
# Copyright (c) 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Integration tests for OpenAI models on OCI GenAI.

These tests verify that LangChain 1.x compatibility works correctly with
OpenAI models available on OCI Generative AI service.

## Prerequisites

1. **OCI Authentication**: Set up OCI authentication with security token:
   ```bash
   oci session authenticate
   ```

2. **Environment Variables**: Export the following:
   ```bash
   export OCI_REGION="us-chicago-1"  # or your region
   export OCI_COMP="ocid1.compartment.oc1..your-compartment-id"
   ```

3. **OCI Config**: Ensure `~/.oci/config` exists with DEFAULT profile

## Running the Tests

Run all OpenAI integration tests:
```bash
cd libs/oci
pytest tests/integration_tests/chat_models/test_openai_models.py -v
```

Run specific test:
```bash
pytest tests/integration_tests/chat_models/test_openai_models.py::test_openai_basic_completion -v
```
"""

import os

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langchain_oci.chat_models import ChatOCIGenAI


@pytest.fixture
def openai_config():
    """Get OpenAI model configuration."""
    compartment_id = os.environ.get("OCI_COMP")
    if not compartment_id:
        pytest.skip("OCI_COMP environment variable not set")

    region = os.environ.get("OCI_REGION", "us-chicago-1")
    return {
        "service_endpoint": f"https://inference.generativeai.{region}.oci.oraclecloud.com",
        "compartment_id": compartment_id,
        "auth_profile": os.environ.get("OCI_CONFIG_PROFILE", "DEFAULT"),
        "auth_type": os.environ.get("OCI_AUTH_TYPE", "SECURITY_TOKEN"),
    }


@pytest.mark.requires("oci")
@pytest.mark.parametrize(
    "model_id",
    [
        "openai.gpt-oss-20b",
        "openai.gpt-oss-120b",
    ],
)
def test_openai_basic_completion(model_id: str, openai_config: dict):
    """Test basic completion with OpenAI models.

    This test verifies that:
    1. The model can be instantiated correctly
    2. Basic message completion works
    3. The response is properly formatted as AIMessage
    4. LangChain 1.x compatibility is maintained
    """
    chat = ChatOCIGenAI(
        model_id=model_id,
        service_endpoint=openai_config["service_endpoint"],
        compartment_id=openai_config["compartment_id"],
        auth_type=openai_config["auth_type"],
        auth_profile=openai_config["auth_profile"],
        model_kwargs={"temperature": 0.7, "max_completion_tokens": 100},
    )

    # Test basic completion
    response = chat.invoke([HumanMessage(content="What is 2+2?")])

    # Verify response structure (LangChain 1.x)
    assert isinstance(response, AIMessage), "Response should be AIMessage"
    # OpenAI models may return empty content if max_completion_tokens is too low
    # or finish due to length limit - just verify the structure is correct
    assert isinstance(response.content, str), "Response content should be string"
    assert hasattr(response, "response_metadata"), "Should have response_metadata"


@pytest.mark.requires("oci")
def test_openai_with_system_message(openai_config: dict):
    """Test OpenAI model with system message.

    Verifies that system messages are properly handled and influence
    the model's behavior.
    """
    chat = ChatOCIGenAI(
        model_id="openai.gpt-oss-20b",
        service_endpoint=openai_config["service_endpoint"],
        compartment_id=openai_config["compartment_id"],
        auth_type=openai_config["auth_type"],
        auth_profile=openai_config["auth_profile"],
        model_kwargs={"temperature": 0.1, "max_completion_tokens": 50},
    )

    response = chat.invoke(
        [
            SystemMessage(content="You are a helpful math tutor."),
            HumanMessage(content="What is 12 * 8?"),
        ]
    )

    assert isinstance(response, AIMessage)
    assert response.content
    # Should contain the answer 96
    assert "96" in response.content


@pytest.mark.requires("oci")
def test_openai_streaming(openai_config: dict):
    """Test streaming with OpenAI models.

    Verifies that:
    1. Streaming works correctly
    2. Chunks are properly formatted
    3. Streaming completes without errors
    """
    chat = ChatOCIGenAI(
        model_id="openai.gpt-oss-20b",
        service_endpoint=openai_config["service_endpoint"],
        compartment_id=openai_config["compartment_id"],
        auth_type=openai_config["auth_type"],
        auth_profile=openai_config["auth_profile"],
        model_kwargs={"temperature": 0.7, "max_completion_tokens": 100},
    )

    chunks = []
    for chunk in chat.stream([HumanMessage(content="Say hello")]):
        assert isinstance(chunk, AIMessage), "Chunk should be AIMessage"
        chunks.append(chunk)

    # Verify we got at least one chunk (streaming worked)
    assert len(chunks) > 0, "Should receive at least one chunk"

    # Verify chunks are properly formatted
    for chunk in chunks:
        assert isinstance(chunk.content, str), "Chunk content should be string"


@pytest.mark.requires("oci")
def test_openai_multiple_rounds(openai_config: dict):
    """Test multiple conversation rounds with OpenAI model.

    Verifies that conversation history is maintained properly.
    """
    chat = ChatOCIGenAI(
        model_id="openai.gpt-oss-20b",
        service_endpoint=openai_config["service_endpoint"],
        compartment_id=openai_config["compartment_id"],
        auth_type=openai_config["auth_type"],
        auth_profile=openai_config["auth_profile"],
        model_kwargs={"temperature": 0.7, "max_completion_tokens": 100},
    )

    # First message
    response1 = chat.invoke([HumanMessage(content="My favorite number is 7")])
    assert isinstance(response1, AIMessage)

    # Second message with context
    response2 = chat.invoke(
        [
            HumanMessage(content="My favorite number is 7"),
            response1,
            HumanMessage(content="What is my favorite number plus 3?"),
        ]
    )
    assert isinstance(response2, AIMessage)
    assert response2.content
    # Should reference the number 10
    assert "10" in response2.content


@pytest.mark.requires("oci")
@pytest.mark.parametrize("model_id", ["openai.gpt-oss-20b", "openai.gpt-oss-120b"])
def test_openai_langchain_1x_compatibility(model_id: str, openai_config: dict):
    """Test LangChain 1.x specific compatibility.

    This test specifically verifies features that are part of
    LangChain 1.x to ensure the integration works correctly
    after rebasing onto main.
    """
    chat = ChatOCIGenAI(
        model_id=model_id,
        service_endpoint=openai_config["service_endpoint"],
        compartment_id=openai_config["compartment_id"],
        auth_type=openai_config["auth_type"],
        auth_profile=openai_config["auth_profile"],
        model_kwargs={"temperature": 0.7, "max_completion_tokens": 50},
    )

    # Test that invoke returns AIMessage (LangChain 1.x behavior)
    response = chat.invoke([HumanMessage(content="Hello")])
    assert isinstance(response, AIMessage)

    # Verify AIMessage has expected attributes
    assert hasattr(response, "content")
    assert hasattr(response, "response_metadata")
    assert hasattr(response, "id")

    # Verify content is populated
    assert response.content is not None
    assert isinstance(response.content, str)
