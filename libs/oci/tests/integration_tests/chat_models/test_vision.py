# Copyright (c) 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Integration tests for vision/multi-modal support.

These tests require:
1. Valid OCI credentials configured
2. Access to OCI Generative AI service
3. A compartment with vision-capable models enabled

Setup:
    export OCI_COMPARTMENT_ID=<your-compartment-id>
    export OCI_CONFIG_PROFILE=DEFAULT
    export OCI_AUTH_TYPE=API_KEY

To run these tests:
    pytest tests/integration_tests/chat_models/test_vision.py -v
"""

import os
import tempfile
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage

from langchain_oci import (
    ChatOCIGenAI,
    encode_image,
    is_vision_model,
    load_image,
)

# Test image URL (public domain - Wikimedia Commons)
TEST_IMAGE_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/thumb/"
    "a/a7/Camponotus_flavomarginatus_ant.jpg/"
    "320px-Camponotus_flavomarginatus_ant.jpg"
)

# A different test image for comparison tests
TEST_IMAGE_URL_2 = (
    "https://upload.wikimedia.org/wikipedia/commons/thumb/"
    "4/4d/Apis_mellifera_Western_honey_bee.jpg/"
    "320px-Apis_mellifera_Western_honey_bee.jpg"
)


def get_config():
    """Get test configuration from environment variables."""
    compartment_id = os.environ.get("OCI_COMPARTMENT_ID")
    if not compartment_id:
        pytest.skip("OCI_COMPARTMENT_ID not set")
    return {
        "service_endpoint": os.environ.get(
            "OCI_GENAI_ENDPOINT",
            "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
        ),
        "compartment_id": compartment_id,
        "auth_profile": os.environ.get("OCI_CONFIG_PROFILE", "API_KEY_AUTH"),
        "auth_type": os.environ.get("OCI_AUTH_TYPE", "API_KEY"),
    }


@pytest.fixture
def vision_llm():
    """Create a vision-capable ChatOCIGenAI instance with Llama 3.2 90B."""
    config = get_config()
    return ChatOCIGenAI(
        model_id="meta.llama-3.2-90b-vision-instruct",
        compartment_id=config["compartment_id"],
        service_endpoint=config["service_endpoint"],
        auth_profile=config["auth_profile"],
        auth_type=config["auth_type"],
    )


@pytest.fixture
def llama4_scout_llm():
    """Create a ChatOCIGenAI instance with Llama 4 Scout (vision-capable)."""
    config = get_config()
    return ChatOCIGenAI(
        model_id="meta.llama-4-scout-17b-16e-instruct",
        compartment_id=config["compartment_id"],
        service_endpoint=config["service_endpoint"],
        auth_profile=config["auth_profile"],
        auth_type=config["auth_type"],
    )


@pytest.fixture
def openai_llm():
    """Create a ChatOCIGenAI instance with OpenAI GPT-4o."""
    config = get_config()
    return ChatOCIGenAI(
        model_id="openai.gpt-4o",
        compartment_id=config["compartment_id"],
        service_endpoint=config["service_endpoint"],
        auth_profile=config["auth_profile"],
        auth_type=config["auth_type"],
    )


@pytest.mark.requires("oci")
class TestVisionUrlImages:
    """Tests for vision model capabilities with URL-based images."""

    def test_url_image_analysis(self, vision_llm):
        """Test analyzing an image from URL."""
        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "What animal is in this image? Answer briefly.",
                },
                {"type": "image_url", "image_url": {"url": TEST_IMAGE_URL}},
            ]
        )

        response = vision_llm.invoke([message])

        assert response.content
        assert len(response.content) > 0
        # The image is of an ant, so we expect the response to mention it
        assert "ant" in response.content.lower()

    def test_url_image_with_llama4_scout(self, llama4_scout_llm):
        """Test Llama 4 Scout model with URL image."""
        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "Describe what you see in this image briefly.",
                },
                {"type": "image_url", "image_url": {"url": TEST_IMAGE_URL}},
            ]
        )

        response = llama4_scout_llm.invoke([message])

        assert response.content
        assert len(response.content) > 0

    def test_url_image_with_openai_model(self, openai_llm):
        """Test OpenAI model via OCI with URL image."""
        message = HumanMessage(
            content=[
                {"type": "text", "text": "What animal is shown? One word answer."},
                {"type": "image_url", "image_url": {"url": TEST_IMAGE_URL}},
            ]
        )

        response = openai_llm.invoke([message])

        assert response.content
        assert len(response.content) > 0


@pytest.mark.requires("oci")
class TestVisionBase64Images:
    """Tests for vision model capabilities with base64-encoded images."""

    def test_base64_image_analysis(self, vision_llm):
        """Test analyzing a base64-encoded image."""
        # Create a minimal test PNG (1x1 red pixel)
        # This is a valid PNG that should be processable
        png_bytes = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
            b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
            b"\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0\x00\x00\x00\x03\x00\x01"
            b"\x00\x05\xfe\xd4\xef\x00\x00\x00\x00IEND\xaeB`\x82"
        )

        image_block = encode_image(png_bytes, "image/png")

        message = HumanMessage(
            content=[
                {"type": "text", "text": "What do you see? Describe the image."},
                image_block,
            ]
        )

        response = vision_llm.invoke([message])

        assert response.content
        assert len(response.content) > 0

    def test_load_and_analyze_local_image(self, vision_llm):
        """Test loading a local image file and analyzing it."""
        # Create a minimal test PNG
        png_bytes = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
            b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
            b"\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0\x00\x00\x00\x03\x00\x01"
            b"\x00\x05\xfe\xd4\xef\x00\x00\x00\x00IEND\xaeB`\x82"
        )

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(png_bytes)
            temp_path = f.name

        try:
            message = HumanMessage(
                content=[
                    {"type": "text", "text": "Describe what you see in this image."},
                    load_image(temp_path),
                ]
            )

            response = vision_llm.invoke([message])

            assert response.content
            assert len(response.content) > 0
        finally:
            Path(temp_path).unlink()


@pytest.mark.requires("oci")
class TestVisionMultipleImages:
    """Tests for processing multiple images in a single request."""

    def test_multiple_images_comparison(self, vision_llm):
        """Test analyzing and comparing multiple images."""
        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "Compare these two images. What insects are shown?",
                },
                {"type": "image_url", "image_url": {"url": TEST_IMAGE_URL}},
                {"type": "image_url", "image_url": {"url": TEST_IMAGE_URL_2}},
            ]
        )

        response = vision_llm.invoke([message])

        assert response.content
        assert len(response.content) > 0

    def test_mixed_text_and_images(self, vision_llm):
        """Test conversation with text and images interleaved."""
        message = HumanMessage(
            content=[
                {"type": "text", "text": "I'll show you an image."},
                {"type": "image_url", "image_url": {"url": TEST_IMAGE_URL}},
                {"type": "text", "text": "What kind of insect is this?"},
            ]
        )

        response = vision_llm.invoke([message])

        assert response.content
        assert len(response.content) > 0


@pytest.mark.requires("oci")
class TestVisionStreaming:
    """Tests for streaming with vision models."""

    def test_streaming_with_image(self, vision_llm):
        """Test streaming response for image analysis."""
        message = HumanMessage(
            content=[
                {"type": "text", "text": "Describe this image in 2-3 sentences."},
                {"type": "image_url", "image_url": {"url": TEST_IMAGE_URL}},
            ]
        )

        chunks = list(vision_llm.stream([message]))

        assert len(chunks) > 0
        # Combine all chunks to get full response
        full_response = "".join(chunk.content for chunk in chunks if chunk.content)
        assert len(full_response) > 0


@pytest.mark.requires("oci")
class TestVisionModelDetection:
    """Tests for vision model detection utility."""

    @pytest.mark.parametrize(
        "model_id,expected",
        [
            ("meta.llama-3.2-90b-vision-instruct", True),
            ("meta.llama-3.2-11b-vision-instruct", True),
            ("meta.llama-4-scout-17b-16e-instruct", True),
            ("meta.llama-4-maverick-17b-128e-instruct-fp8", True),
            ("meta.llama-3.3-70b-instruct", False),
            ("cohere.command-r-16k", False),
            ("cohere.command-a-03-2025", False),
        ],
    )
    def test_is_vision_model(self, model_id, expected):
        """Test vision model detection for various model IDs."""
        assert is_vision_model(model_id) == expected


@pytest.mark.requires("oci")
@pytest.mark.parametrize(
    "model_id",
    [
        "meta.llama-3.2-90b-vision-instruct",
        "meta.llama-4-scout-17b-16e-instruct",
        "openai.gpt-4o",
    ],
)
def test_vision_models_can_process_images(model_id):
    """Test that various vision-capable models can process images."""
    config = get_config()
    llm = ChatOCIGenAI(
        model_id=model_id,
        compartment_id=config["compartment_id"],
        service_endpoint=config["service_endpoint"],
        auth_profile=config["auth_profile"],
        auth_type=config["auth_type"],
    )

    message = HumanMessage(
        content=[
            {"type": "text", "text": "What is in this image? One sentence answer."},
            {"type": "image_url", "image_url": {"url": TEST_IMAGE_URL}},
        ]
    )

    response = llm.invoke([message])

    assert response.content
    assert len(response.content) > 0


@pytest.mark.requires("oci")
class TestVisionErrorHandling:
    """Tests for error handling in vision operations."""

    def test_invalid_image_url(self, vision_llm):
        """Test behavior with an invalid image URL."""
        message = HumanMessage(
            content=[
                {"type": "text", "text": "What is in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://invalid-url-that-does-not-exist.com/img.jpg"
                    },
                },
            ]
        )

        # This should either raise an error or return an error message
        # depending on how the model/service handles invalid URLs
        with pytest.raises(Exception):
            vision_llm.invoke([message])

    def test_text_only_model_with_image(self):
        """Test that text-only models handle images appropriately."""
        config = get_config()
        # Cohere models don't support vision
        llm = ChatOCIGenAI(
            model_id="cohere.command-r-16k",
            compartment_id=config["compartment_id"],
            service_endpoint=config["service_endpoint"],
            auth_profile=config["auth_profile"],
            auth_type=config["auth_type"],
        )

        message = HumanMessage(
            content=[
                {"type": "text", "text": "What is in this image?"},
                {"type": "image_url", "image_url": {"url": TEST_IMAGE_URL}},
            ]
        )

        # This should raise an error since Cohere doesn't support images
        with pytest.raises(Exception):
            llm.invoke([message])
