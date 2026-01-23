# Copyright (c) 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Integration tests for vision with real-world images.

These tests download real images from the internet and verify that
vision models can correctly identify them.

Requirements:
1. Valid OCI credentials configured
2. Access to OCI Generative AI service in Phoenix region

Setup:
    export OCI_COMPARTMENT_ID=<your-compartment-id>
    export OCI_CONFIG_PROFILE=API_KEY_AUTH
    export OCI_AUTH_TYPE=API_KEY

To run these tests:
    pytest tests/integration_tests/test_vision_real_images.py -v
"""

import os
import ssl
import urllib.request

import pytest
from langchain_core.messages import HumanMessage

from langchain_oci import ChatOCIGenAI, encode_image


def get_config():
    """Get test configuration from environment variables."""
    compartment_id = os.environ.get("OCI_COMPARTMENT_ID")
    if not compartment_id:
        pytest.skip("OCI_COMPARTMENT_ID not set")
    return {
        "compartment_id": compartment_id,
        "service_endpoint": os.environ.get(
            "OCI_GENAI_ENDPOINT",
            "https://inference.generativeai.us-phoenix-1.oci.oraclecloud.com",
        ),
        "auth_profile": os.environ.get("OCI_CONFIG_PROFILE", "API_KEY_AUTH"),
        "auth_type": os.environ.get("OCI_AUTH_TYPE", "API_KEY"),
    }


def download_url(url: str) -> bytes:
    """Download content from URL."""
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36"
        },
    )
    with urllib.request.urlopen(req, context=ssl_context, timeout=30) as response:
        return response.read()


@pytest.fixture
def gemini_llm():
    """Create a Gemini vision-capable ChatOCIGenAI instance."""
    config = get_config()
    return ChatOCIGenAI(
        model_id="google.gemini-2.5-flash",
        compartment_id=config["compartment_id"],
        service_endpoint=config["service_endpoint"],
        auth_profile=config["auth_profile"],
        auth_type=config["auth_type"],
    )


@pytest.mark.requires("oci")
class TestVisionRealImages:
    """Tests for vision model capabilities with real-world images."""

    def test_oracle_logo_identification(self, gemini_llm):
        """Test that Gemini can identify the Oracle logo."""
        oracle_logo_url = (
            "https://logos-world.net/wp-content/uploads/2020/09/Oracle-Logo.png"
        )

        img_bytes = download_url(oracle_logo_url)
        assert len(img_bytes) > 0, "Failed to download Oracle logo"

        img_content = encode_image(img_bytes, "image/png")

        message = HumanMessage(
            content=[
                {"type": "text", "text": "What company logo is this? Just the name."},
                img_content,
            ]
        )

        result = gemini_llm.invoke([message])

        assert result.content
        assert "oracle" in result.content.lower(), (
            f"Expected 'Oracle' in response, got: {result.content}"
        )

    def test_google_logo_identification(self, gemini_llm):
        """Test that Gemini can identify the Google logo."""
        google_logo_url = (
            "https://www.google.com/images/branding/googlelogo/"
            "2x/googlelogo_color_272x92dp.png"
        )

        img_bytes = download_url(google_logo_url)
        assert len(img_bytes) > 0, "Failed to download Google logo"

        img_content = encode_image(img_bytes, "image/png")

        message = HumanMessage(
            content=[
                {"type": "text", "text": "What company logo is this? Just the name."},
                img_content,
            ]
        )

        result = gemini_llm.invoke([message])

        assert result.content
        assert "google" in result.content.lower(), (
            f"Expected 'Google' in response, got: {result.content}"
        )

    def test_landmark_identification(self, gemini_llm):
        """Test that Gemini can identify famous landmarks."""
        eiffel_url = (
            "https://upload.wikimedia.org/wikipedia/commons/thumb/"
            "8/85/Tour_Eiffel_Wikimedia_Commons_%28cropped%29.jpg/"
            "800px-Tour_Eiffel_Wikimedia_Commons_%28cropped%29.jpg"
        )

        img_bytes = download_url(eiffel_url)
        assert len(img_bytes) > 0, "Failed to download Eiffel Tower image"

        img_content = encode_image(img_bytes, "image/jpeg")

        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "What famous landmark is this? Just the name.",
                },
                img_content,
            ]
        )

        result = gemini_llm.invoke([message])

        assert result.content
        assert "eiffel" in result.content.lower(), (
            f"Expected 'Eiffel' in response, got: {result.content}"
        )

    def test_animal_identification(self, gemini_llm):
        """Test that Gemini can identify animals in photos."""
        cat_url = (
            "https://upload.wikimedia.org/wikipedia/commons/thumb/"
            "3/3a/Cat03.jpg/1200px-Cat03.jpg"
        )

        img_bytes = download_url(cat_url)
        assert len(img_bytes) > 0, "Failed to download cat image"

        img_content = encode_image(img_bytes, "image/jpeg")

        message = HumanMessage(
            content=[
                {"type": "text", "text": "What animal is in this photo? One word."},
                img_content,
            ]
        )

        result = gemini_llm.invoke([message])

        assert result.content
        assert "cat" in result.content.lower(), (
            f"Expected 'cat' in response, got: {result.content}"
        )
