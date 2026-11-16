# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Standard LangChain conformance tests (langchain-tests) for chat models.

Unit tests run fully offline: the OCI client / signer is injected as a mock,
which skips the auth-time client construction in ``validate_environment``.
"""

from typing import Any, Dict, Type
from unittest.mock import MagicMock

import pytest
from langchain_tests.unit_tests import ChatModelUnitTests

from langchain_oci.chat_models.oci_data_science import ChatOCIModelDeployment
from langchain_oci.chat_models.oci_generative_ai import ChatOCIGenAI


class ChatOCIGenAIStandardBase(ChatModelUnitTests):
    """Shared configuration for ChatOCIGenAI standard tests."""

    @property
    def chat_model_class(self) -> Type[ChatOCIGenAI]:
        return ChatOCIGenAI

    @property
    def standard_chat_model_params(self) -> Dict[str, Any]:
        # timeout, max_retries and api_key are excluded from the defaults:
        # they are not accepted as top-level constructor kwargs yet (auth is
        # configured via auth_type/auth_profile, and the retry strategy lives
        # on the injected OCI client).
        # See https://github.com/oracle/langchain-oracle/issues/261.
        return {
            "temperature": 0,
            "max_tokens": 100,
            "stop": [],
        }


@pytest.mark.requires("oci")
class TestChatOCIGenAIMetaStandard(ChatOCIGenAIStandardBase):
    """Standard unit tests for ChatOCIGenAI on the Meta/Llama provider path."""

    @property
    def chat_model_params(self) -> Dict[str, Any]:
        return {
            "model_id": "meta.llama-3.3-70b-instruct",
            "client": MagicMock(),
            "compartment_id": "test-compartment-ocid",
        }


@pytest.mark.requires("oci")
class TestChatOCIGenAICohereStandard(ChatOCIGenAIStandardBase):
    """Standard unit tests for ChatOCIGenAI on the Cohere provider path."""

    @property
    def chat_model_params(self) -> Dict[str, Any]:
        return {
            "model_id": "cohere.command-r-plus-08-2024",
            "client": MagicMock(),
            "compartment_id": "test-compartment-ocid",
        }

    @property
    def has_tool_choice(self) -> bool:
        # tool_choice is rejected on the Cohere provider path.
        # See https://github.com/oracle/langchain-oracle/issues/266.
        return False


@pytest.mark.requires("oci", "ads")
class TestChatOCIModelDeploymentStandard(ChatModelUnitTests):
    """Standard unit tests for ChatOCIModelDeployment (Data Science endpoints)."""

    @property
    def chat_model_class(self) -> Type[ChatOCIModelDeployment]:
        return ChatOCIModelDeployment

    @property
    def chat_model_params(self) -> Dict[str, Any]:
        return {
            "endpoint": "https://modeldeployment.example.oci.customer-oci.com/ocid/predict",
            "model": "odsc-llm",
            "auth": {"signer": MagicMock()},
        }
