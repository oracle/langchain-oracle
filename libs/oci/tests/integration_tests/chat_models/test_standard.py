# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Standard LangChain conformance integration tests (langchain-tests).

Makes real OCI Generative AI calls. Follows the same env-var conventions as
the other chat-model integration tests:

Required:
    OCI_COMPARTMENT_ID  -- compartment OCID to bill the calls against.

Optional:
    OCI_REGION          -- GenAI region (default: us-chicago-1)
    OCI_AUTH_TYPE       -- auth type (default: API_KEY)
    OCI_CONFIG_PROFILE  -- ~/.oci/config profile (default: DEFAULT)
    OCI_STANDARD_TEST_MODEL_ID -- model to run the suite against
                                  (default: meta.llama-3.3-70b-instruct)

Capability flags are set for the default Meta model; some flags differ per
provider (e.g. Cohere has no tool_choice, Gemini adds pdf/video/audio), so
runs against another OCI_STANDARD_TEST_MODEL_ID may need adjusted flags.
"""

import os
from typing import Any, Dict, Type

import pytest
from langchain_tests.integration_tests import ChatModelIntegrationTests

from langchain_oci.chat_models.oci_generative_ai import ChatOCIGenAI

pytestmark = pytest.mark.skipif(
    not os.environ.get("OCI_COMPARTMENT_ID"),
    reason="OCI_COMPARTMENT_ID environment variable not set",
)


class TestChatOCIGenAIStandardIntegration(ChatModelIntegrationTests):
    """Standard integration tests for ChatOCIGenAI."""

    @property
    def chat_model_class(self) -> Type[ChatOCIGenAI]:
        return ChatOCIGenAI

    @property
    def chat_model_params(self) -> Dict[str, Any]:
        region = os.environ.get("OCI_REGION", "us-chicago-1")
        return {
            "model_id": os.environ.get(
                "OCI_STANDARD_TEST_MODEL_ID", "meta.llama-3.3-70b-instruct"
            ),
            "compartment_id": os.environ["OCI_COMPARTMENT_ID"],
            "service_endpoint": (
                f"https://inference.generativeai.{region}.oci.oraclecloud.com"
            ),
            "auth_type": os.environ.get("OCI_AUTH_TYPE", "API_KEY"),
            "auth_profile": os.environ.get("OCI_CONFIG_PROFILE", "DEFAULT"),
        }

    @property
    def standard_chat_model_params(self) -> Dict[str, Any]:
        # timeout/max_retries/api_key are not top-level kwargs yet; see
        # https://github.com/oracle/langchain-oracle/issues/261.
        return {
            "temperature": 0,
            "max_tokens": 512,
        }

    @property
    def supports_model_override(self) -> bool:
        # The serving mode is always built from model_id; there is no
        # per-call model override.
        return False

    @property
    def has_tool_choice(self) -> bool:
        # Supported on the Generic/Meta provider path used by the default
        # test model. Set to False when running against Cohere models.
        return True

    @property
    def supports_json_mode(self) -> bool:
        return True

    @property
    def supports_image_inputs(self) -> bool:
        # Default Meta test model is text-only; vision-capable model ids
        # (Llama vision, Gemini, GPT-5.x) accept image blocks.
        return False

    @property
    def returns_usage_metadata(self) -> bool:
        return True

    @pytest.mark.xfail(
        reason=(
            "OCI GenAI streaming responses carry no usage payload: the final "
            "SSE event contains only message/finishReason (verified against "
            "meta.llama-3.3-70b-instruct, 2026-07-22). Tracked in "
            "https://github.com/oracle/langchain-oracle/issues/261."
        )
    )
    def test_usage_metadata_streaming(self, model: Any) -> None:
        super().test_usage_metadata_streaming(model)

    @pytest.mark.xfail(
        reason=(
            "OCI GenAI service bug: streaming with response_format "
            "JSON_OBJECT returns empty content and finishReason=tool_calls "
            "on the Generic/Meta path; json_mode works on invoke but not "
            "stream. Tracked in "
            "https://github.com/oracle/langchain-oracle/issues/261."
        )
    )
    def test_json_mode(self, model: Any) -> None:
        super().test_json_mode(model)
