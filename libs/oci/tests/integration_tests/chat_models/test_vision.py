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

Note: OCI Generative AI service requires images to be base64-encoded.
URL-based images are not supported.
"""

import io
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

try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


def create_test_image(color: str = "red", size: tuple = (100, 100)) -> bytes:
    """Create a simple test image with PIL.

    Args:
        color: Color name (e.g., 'red', 'blue', 'green')
        size: Image dimensions as (width, height)

    Returns:
        PNG image as bytes
    """
    if not PIL_AVAILABLE:
        pytest.skip("PIL not available")

    img = Image.new("RGB", size, color=color)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


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
def gemini_llm():
    """Create a ChatOCIGenAI instance with Google Gemini 2.5 Flash."""
    config = get_config()
    return ChatOCIGenAI(
        model_id="google.gemini-2.5-flash",
        compartment_id=config["compartment_id"],
        service_endpoint=config["service_endpoint"],
        auth_profile=config["auth_profile"],
        auth_type=config["auth_type"],
    )


@pytest.mark.requires("oci")
class TestVisionBase64Images:
    """Tests for vision model capabilities with base64-encoded images."""

    def test_base64_image_analysis(self, vision_llm):
        """Test analyzing a base64-encoded image."""
        # Create a test image using PIL
        red_image = create_test_image("red")

        image_block = encode_image(red_image, "image/png")

        message = HumanMessage(
            content=[
                {"type": "text", "text": "What color is this image?"},
                image_block,
            ]
        )

        response = vision_llm.invoke([message])

        assert response.content
        assert len(response.content) > 0

    def test_load_and_analyze_local_image(self, vision_llm):
        """Test loading a local image file and analyzing it."""
        # Create a test image using PIL
        blue_image = create_test_image("blue")

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(blue_image)
            temp_path = f.name

        try:
            message = HumanMessage(
                content=[
                    {"type": "text", "text": "What color is this image?"},
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

    @pytest.mark.skip(reason="OCI GenAI currently only supports one image per message")
    def test_multiple_images_comparison(self, vision_llm):
        """Test analyzing and comparing multiple images."""
        # Create two different colored images
        red_image = create_test_image("red")
        blue_image = create_test_image("blue")

        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "Compare these two images. What colors are they?",
                },
                encode_image(red_image, "image/png"),
                encode_image(blue_image, "image/png"),
            ]
        )

        response = vision_llm.invoke([message])

        assert response.content
        assert len(response.content) > 0

    def test_mixed_text_and_images(self, vision_llm):
        """Test conversation with text and images interleaved."""
        green_image = create_test_image("green")

        message = HumanMessage(
            content=[
                {"type": "text", "text": "I'll show you an image."},
                encode_image(green_image, "image/png"),
                {"type": "text", "text": "What color is this image?"},
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
        yellow_image = create_test_image("yellow")

        message = HumanMessage(
            content=[
                {"type": "text", "text": "What color is this image?"},
                encode_image(yellow_image, "image/png"),
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
            ("google.gemini-2.5-flash", True),
            ("google.gemini-2.5-pro", True),
            ("google.gemini-2.5-flash-lite", True),
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
        "google.gemini-2.5-flash",
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

    purple_image = create_test_image("purple")

    message = HumanMessage(
        content=[
            {"type": "text", "text": "What color is this image?"},
            encode_image(purple_image, "image/png"),
        ]
    )

    response = llm.invoke([message])

    assert response.content
    assert len(response.content) > 0


@pytest.mark.requires("oci")
class TestVisionErrorHandling:
    """Tests for error handling in vision operations."""

    def test_invalid_base64_image(self, vision_llm):
        """Test behavior with invalid base64 image data."""
        # Create an invalid image block
        invalid_image = {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,invalid_base64_data!!!"},
        }

        message = HumanMessage(
            content=[
                {"type": "text", "text": "What is in this image?"},
                invalid_image,
            ]
        )

        # This should raise an error for invalid image data
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

        orange_image = create_test_image("orange")

        message = HumanMessage(
            content=[
                {"type": "text", "text": "What color is this image?"},
                encode_image(orange_image, "image/png"),
            ]
        )

        # This should raise an error since Cohere doesn't support images
        with pytest.raises(Exception):
            llm.invoke([message])
