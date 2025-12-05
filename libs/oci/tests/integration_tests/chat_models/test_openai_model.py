#!/usr/bin/env python3
# Copyright (c) 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Quick smoke tests for OpenAI models on OCI GenAI.

These are simple smoke tests to verify LangChain 1.x support with OpenAI models.
For comprehensive OpenAI model tests, see test_openai_models.py.

Setup:
    export OCI_COMP=<your-compartment-id>

Run:
    pytest tests/integration_tests/chat_models/test_openai_model.py -v
"""

import os

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langchain_oci.chat_models import ChatOCIGenAI


@pytest.fixture
def openai_chat():
    """Create ChatOCIGenAI instance for OpenAI model testing."""
    compartment_id = os.environ.get("OCI_COMP")
    if not compartment_id:
        pytest.skip("OCI_COMP environment variable not set")

    return ChatOCIGenAI(
        model_id="openai.gpt-oss-20b",
        service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
        compartment_id=compartment_id,
        auth_type="SECURITY_TOKEN",
        auth_profile="DEFAULT",
        model_kwargs={
            "temperature": 0.7,
            "max_completion_tokens": 100,
        },
    )


@pytest.mark.requires("oci")
def test_basic_completion(openai_chat):
    """Test basic completion with OpenAI model."""
    response = openai_chat.invoke([HumanMessage(content="Say hello in 5 words")])

    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)
    assert len(response.content) > 0


@pytest.mark.requires("oci")
def test_system_message(openai_chat):
    """Test completion with system message."""
    response = openai_chat.invoke(
        [
            SystemMessage(content="You are a helpful math tutor."),
            HumanMessage(content="What is 15 * 23?"),
        ]
    )

    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)
    assert len(response.content) > 0


@pytest.mark.requires("oci")
def test_streaming(openai_chat):
    """Test streaming with OpenAI model."""
    chunks = list(openai_chat.stream([HumanMessage(content="Count from 1 to 5")]))

    assert len(chunks) > 0
    for chunk in chunks:
        assert isinstance(chunk, AIMessage)
        assert isinstance(chunk.content, str)
