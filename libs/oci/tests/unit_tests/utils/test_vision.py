# Copyright (c) 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Unit tests for vision utilities."""

import base64
import tempfile
from pathlib import Path

import pytest

from langchain_oci.utils.vision import (
    VISION_MODELS,
    encode_image,
    is_vision_model,
    load_image,
)


class TestEncodeImage:
    """Tests for encode_image function."""

    def test_encode_image_returns_dict(self):
        """Test that encode_image returns a properly formatted dict."""
        png_bytes = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
            b"\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
            b"\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01"
            b"\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        result = encode_image(png_bytes, "image/png")

        assert isinstance(result, dict)
        assert result["type"] == "image_url"
        assert "image_url" in result
        assert result["image_url"]["url"].startswith("data:image/png;base64,")

    def test_encode_image_content_decodable(self):
        """Test that the base64 content can be decoded back."""
        png_bytes = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
            b"\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
            b"\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01"
            b"\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        result = encode_image(png_bytes, "image/png")

        # Extract and decode base64 data
        data_url = result["image_url"]["url"]
        encoded_data = data_url.split(",")[1]
        decoded = base64.standard_b64decode(encoded_data)
        assert decoded == png_bytes

    def test_encode_image_jpeg(self):
        """Test encoding JPEG image bytes."""
        jpeg_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00"
        result = encode_image(jpeg_bytes, "image/jpeg")

        assert result["type"] == "image_url"
        assert result["image_url"]["url"].startswith("data:image/jpeg;base64,")

    def test_encode_image_default_mime_type(self):
        """Test that default mime type is image/png."""
        image_bytes = b"test image data"
        result = encode_image(image_bytes)

        assert result["image_url"]["url"].startswith("data:image/png;base64,")


class TestLoadImage:
    """Tests for load_image function."""

    def test_load_image_returns_dict(self):
        """Test that load_image returns a properly formatted dict."""
        png_bytes = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
            b"\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
            b"\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01"
            b"\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
        )

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(png_bytes)
            temp_path = f.name

        try:
            result = load_image(temp_path)

            # Should return a dict with correct structure
            assert isinstance(result, dict)
            assert result["type"] == "image_url"
            assert "image_url" in result
            assert "url" in result["image_url"]
            assert result["image_url"]["url"].startswith("data:image/png;base64,")
        finally:
            Path(temp_path).unlink()

    def test_load_image_jpeg(self):
        """Test loading a JPEG file returns correct mime type."""
        jpeg_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00"

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(jpeg_bytes)
            temp_path = f.name

        try:
            result = load_image(temp_path)

            assert result["type"] == "image_url"
            assert result["image_url"]["url"].startswith("data:image/jpeg;base64,")
        finally:
            Path(temp_path).unlink()

    def test_load_image_with_path_object(self):
        """Test loading using Path object instead of string."""
        png_bytes = b"\x89PNG\r\n"

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(png_bytes)
            temp_path = Path(f.name)

        try:
            result = load_image(temp_path)

            assert isinstance(result, dict)
            assert result["type"] == "image_url"
        finally:
            temp_path.unlink()

    def test_load_image_nonexistent_file(self):
        """Test loading a file that doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_image("/nonexistent/path/image.png")

    def test_load_image_content_decodable(self):
        """Test that the base64 content in the dict can be decoded."""
        png_bytes = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
            b"\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
            b"\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01"
            b"\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
        )

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(png_bytes)
            temp_path = f.name

        try:
            result = load_image(temp_path)

            # Extract and decode base64 data
            data_url = result["image_url"]["url"]
            encoded_data = data_url.split(",")[1]
            decoded = base64.standard_b64decode(encoded_data)
            assert decoded == png_bytes
        finally:
            Path(temp_path).unlink()


class TestIsVisionModel:
    """Tests for is_vision_model function."""

    def test_vision_model_llama_32_90b(self):
        """Test detection of Llama 3.2 90B vision model."""
        assert is_vision_model("meta.llama-3.2-90b-vision-instruct") is True

    def test_vision_model_llama_32_11b(self):
        """Test detection of Llama 3.2 11B vision model."""
        assert is_vision_model("meta.llama-3.2-11b-vision-instruct") is True

    def test_vision_model_llama_4_scout(self):
        """Test detection of Llama 4 Scout model (has vision)."""
        assert is_vision_model("meta.llama-4-scout-17b-16e-instruct") is True

    def test_vision_model_llama_4_maverick(self):
        """Test detection of Llama 4 Maverick model (has vision)."""
        assert is_vision_model("meta.llama-4-maverick-17b-128e-instruct-fp8") is True

    def test_non_vision_model_llama_33(self):
        """Test that Llama 3.3 70B is not detected as vision model."""
        assert is_vision_model("meta.llama-3.3-70b-instruct") is False

    def test_non_vision_model_cohere(self):
        """Test that Cohere models are not detected as vision models."""
        assert is_vision_model("cohere.command-r-16k") is False
        assert is_vision_model("cohere.command-a-03-2025") is False

    def test_unknown_model_returns_false(self):
        """Test that unknown models return False."""
        assert is_vision_model("some-unknown-model") is False
        assert is_vision_model("custom-vision-model") is False  # Not in VISION_MODELS

    def test_dedicated_endpoint_ocid_returns_false(self):
        """Test that dedicated endpoint OCIDs return False.

        Note: For dedicated endpoints, users should check the underlying model
        or explicitly handle vision features. The is_vision_model function
        only checks against known model IDs.
        """
        endpoint_ocid = "ocid1.generativeaiendpoint.oc1.us-chicago-1.xxx"
        assert is_vision_model(endpoint_ocid) is False


class TestVisionModelsConstant:
    """Tests for VISION_MODELS constant."""

    def test_vision_models_not_empty(self):
        """Test that VISION_MODELS list is not empty."""
        assert len(VISION_MODELS) > 0

    def test_vision_models_contains_expected(self):
        """Test that VISION_MODELS contains expected models."""
        assert "meta.llama-3.2-90b-vision-instruct" in VISION_MODELS
        assert "meta.llama-3.2-11b-vision-instruct" in VISION_MODELS
        assert "meta.llama-4-scout-17b-16e-instruct" in VISION_MODELS

    def test_all_vision_models_detected(self):
        """Test that all models in VISION_MODELS are detected by is_vision_model."""
        for model_id in VISION_MODELS:
            assert is_vision_model(model_id), f"{model_id} should be detected as vision"
