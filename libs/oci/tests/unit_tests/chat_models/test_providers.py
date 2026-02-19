# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Unit tests for OCI Generative AI providers."""

import warnings
from typing import Any, Dict, List

from langchain_oci.chat_models.providers import (
    CohereProvider,
    GeminiProvider,
    GenericProvider,
    MetaProvider,
    Provider,
)


class TestProviderBaseClass:
    """Tests for the Provider base class."""

    def test_normalize_params_no_transforms(self) -> None:
        """Test normalize_params returns params unchanged when no transforms defined."""
        provider = GenericProvider()
        params = {"temperature": 0.5, "max_tokens": 100}
        result = provider.normalize_params(params)
        assert result == params

    def test_normalize_params_does_not_mutate_input(self) -> None:
        """Test normalize_params does not mutate the input dictionary."""
        provider = GeminiProvider()
        params = {"max_output_tokens": 100, "temperature": 0.5}
        original_params = params.copy()
        provider.normalize_params(params)
        assert params == original_params


class TestGeminiProvider:
    """Tests for the GeminiProvider class."""

    def test_inherits_from_generic_provider(self) -> None:
        """Test GeminiProvider inherits from GenericProvider."""
        provider = GeminiProvider()
        assert isinstance(provider, GenericProvider)
        assert isinstance(provider, Provider)

    def test_normalize_params_maps_max_output_tokens(self) -> None:
        """Test max_output_tokens is mapped to max_tokens."""
        provider = GeminiProvider()
        params = {"max_output_tokens": 100, "temperature": 0.5}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = provider.normalize_params(params)

            assert "max_output_tokens" not in result
            assert result["max_tokens"] == 100
            assert result["temperature"] == 0.5

            # Should emit warning
            mapping_warnings = [x for x in w if "max_output_tokens" in str(x.message)]
            assert len(mapping_warnings) == 1
            assert "Mapped" in str(mapping_warnings[0].message)

    def test_normalize_params_prefers_max_tokens_when_both_provided(self) -> None:
        """Test max_tokens is preferred when both are provided."""
        provider = GeminiProvider()
        params = {"max_tokens": 50, "max_output_tokens": 100, "temperature": 0.5}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = provider.normalize_params(params)

            assert result["max_tokens"] == 50  # Prefer max_tokens
            assert "max_output_tokens" not in result
            assert result["temperature"] == 0.5

            # Should emit warning about both being provided
            both_warnings = [x for x in w if "Both" in str(x.message)]
            assert len(both_warnings) == 1
            assert "ignoring" in str(both_warnings[0].message).lower()

    def test_normalize_params_no_changes_when_only_max_tokens(self) -> None:
        """Test no changes when only max_tokens is provided."""
        provider = GeminiProvider()
        params = {"max_tokens": 100, "temperature": 0.5}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = provider.normalize_params(params)

            assert result == params
            # Should not emit any warnings
            mapping_warnings = [x for x in w if "max_output_tokens" in str(x.message)]
            assert len(mapping_warnings) == 0

    def test_normalize_params_empty_params(self) -> None:
        """Test normalize_params handles empty params."""
        provider = GeminiProvider()
        params: dict = {}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = provider.normalize_params(params)

            assert result == {}
            assert len(w) == 0

    def test_stop_sequence_key(self) -> None:
        """Test stop_sequence_key returns correct value."""
        provider = GeminiProvider()
        assert provider.stop_sequence_key == "stop"


class TestMetaProvider:
    """Tests for the MetaProvider class."""

    def test_inherits_from_generic_provider(self) -> None:
        """Test MetaProvider inherits from GenericProvider."""
        provider = MetaProvider()
        assert isinstance(provider, GenericProvider)

    def test_normalize_params_unchanged(self) -> None:
        """Test normalize_params returns params unchanged."""
        provider = MetaProvider()
        params = {"max_tokens": 100, "temperature": 0.5}
        result = provider.normalize_params(params)
        assert result == params


class TestCohereProvider:
    """Tests for the CohereProvider class."""

    def test_inherits_from_provider(self) -> None:
        """Test CohereProvider inherits from Provider."""
        provider = CohereProvider()
        assert isinstance(provider, Provider)

    def test_stop_sequence_key(self) -> None:
        """Test stop_sequence_key returns correct value for Cohere."""
        provider = CohereProvider()
        assert provider.stop_sequence_key == "stop_sequences"


class TestGenericProvider:
    """Tests for the GenericProvider class."""

    def test_inherits_from_provider(self) -> None:
        """Test GenericProvider inherits from Provider."""
        provider = GenericProvider()
        assert isinstance(provider, Provider)

    def test_stop_sequence_key(self) -> None:
        """Test stop_sequence_key returns correct value."""
        provider = GenericProvider()
        assert provider.stop_sequence_key == "stop"

    def test_normalize_params_unchanged(self) -> None:
        """Test normalize_params returns params unchanged by default."""
        provider = GenericProvider()
        params = {"max_tokens": 100, "temperature": 0.5}
        result = provider.normalize_params(params)
        assert result == params


class TestGenericProviderMultimodalContent:
    """Tests for GenericProvider multimodal content processing."""

    def test_process_text_content_string(self) -> None:
        """Test processing plain string content."""
        provider = GenericProvider()
        result = provider._process_message_content("Hello, world!")

        assert len(result) == 1
        assert result[0].text == "Hello, world!"

    def test_process_text_content_dict(self) -> None:
        """Test processing text content as dict."""
        provider = GenericProvider()
        content: List[Dict[str, Any]] = [{"type": "text", "text": "Hello, world!"}]
        result = provider._process_message_content(content)  # type: ignore[arg-type]  # type: ignore[arg-type]

        assert len(result) == 1
        assert result[0].text == "Hello, world!"

    def test_process_image_content(self) -> None:
        """Test processing image content."""
        provider = GenericProvider()
        content: List[Dict[str, Any]] = [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc123"}},
        ]
        result = provider._process_message_content(content)  # type: ignore[arg-type]  # type: ignore[arg-type]

        assert len(result) == 2
        assert result[0].text == "What's in this image?"
        assert result[1].image_url.url == "data:image/png;base64,abc123"

    def test_process_document_content(self) -> None:
        """Test processing document (PDF) content."""
        provider = GenericProvider()
        content: List[Dict[str, Any]] = [
            {"type": "text", "text": "Summarize this PDF"},
            {
                "type": "document_url",
                "document_url": {"url": "data:application/pdf;base64,JVBERi0xLjQ="},
            },
        ]
        result = provider._process_message_content(content)  # type: ignore[arg-type]  # type: ignore[arg-type]

        assert len(result) == 2
        assert result[0].text == "Summarize this PDF"
        assert result[1].document_url.url == "data:application/pdf;base64,JVBERi0xLjQ="

    def test_process_document_content_alternative_type(self) -> None:
        """Test processing document content with 'document' type."""
        provider = GenericProvider()
        content: List[Dict[str, Any]] = [
            {
                "type": "document",
                "document": {"url": "data:application/pdf;base64,JVBERi0xLjQ="},
            },
        ]
        result = provider._process_message_content(content)  # type: ignore[arg-type]  # type: ignore[arg-type]

        assert len(result) == 1
        assert result[0].document_url.url == "data:application/pdf;base64,JVBERi0xLjQ="

    def test_process_document_content_file_type(self) -> None:
        """Test processing document content with 'file' type."""
        provider = GenericProvider()
        content: List[Dict[str, Any]] = [
            {
                "type": "file",
                "file": {"url": "data:application/pdf;base64,JVBERi0xLjQ="},
            },
        ]
        result = provider._process_message_content(content)  # type: ignore[arg-type]  # type: ignore[arg-type]

        assert len(result) == 1
        assert result[0].document_url.url == "data:application/pdf;base64,JVBERi0xLjQ="

    def test_process_video_content(self) -> None:
        """Test processing video content."""
        provider = GenericProvider()
        content: List[Dict[str, Any]] = [
            {"type": "text", "text": "Describe this video"},
            {
                "type": "video_url",
                "video_url": {"url": "data:video/mp4;base64,AAAAIGZ0eXA="},
            },
        ]
        result = provider._process_message_content(content)  # type: ignore[arg-type]  # type: ignore[arg-type]

        assert len(result) == 2
        assert result[0].text == "Describe this video"
        assert result[1].video_url.url == "data:video/mp4;base64,AAAAIGZ0eXA="

    def test_process_video_content_alternative_type(self) -> None:
        """Test processing video content with 'video' type."""
        provider = GenericProvider()
        content: List[Dict[str, Any]] = [
            {
                "type": "video",
                "video": {"url": "data:video/mp4;base64,AAAAIGZ0eXA="},
            },
        ]
        result = provider._process_message_content(content)  # type: ignore[arg-type]  # type: ignore[arg-type]

        assert len(result) == 1
        assert result[1 - 1].video_url.url == "data:video/mp4;base64,AAAAIGZ0eXA="

    def test_process_audio_content(self) -> None:
        """Test processing audio content."""
        provider = GenericProvider()
        content: List[Dict[str, Any]] = [
            {"type": "text", "text": "Transcribe this audio"},
            {
                "type": "audio_url",
                "audio_url": {"url": "data:audio/wav;base64,UklGRg=="},
            },
        ]
        result = provider._process_message_content(content)  # type: ignore[arg-type]  # type: ignore[arg-type]

        assert len(result) == 2
        assert result[0].text == "Transcribe this audio"
        assert result[1].audio_url.url == "data:audio/wav;base64,UklGRg=="

    def test_process_audio_content_alternative_type(self) -> None:
        """Test processing audio content with 'audio' type."""
        provider = GenericProvider()
        content: List[Dict[str, Any]] = [
            {
                "type": "audio",
                "audio": {"url": "data:audio/mp3;base64,//uQxAAA="},
            },
        ]
        result = provider._process_message_content(content)  # type: ignore[arg-type]  # type: ignore[arg-type]

        assert len(result) == 1
        assert result[0].audio_url.url == "data:audio/mp3;base64,//uQxAAA="

    def test_process_mixed_multimodal_content(self) -> None:
        """Test processing mixed multimodal content (text, image, document)."""
        provider = GenericProvider()
        content: List[Dict[str, Any]] = [
            {"type": "text", "text": "Analyze these files:"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            {
                "type": "document_url",
                "document_url": {"url": "data:application/pdf;base64,def"},
            },
            {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,ghi"}},
        ]
        result = provider._process_message_content(content)  # type: ignore[arg-type]  # type: ignore[arg-type]

        assert len(result) == 4
        assert result[0].text == "Analyze these files:"
        assert result[1].image_url.url == "data:image/png;base64,abc"
        assert result[2].document_url.url == "data:application/pdf;base64,def"
        assert result[3].video_url.url == "data:video/mp4;base64,ghi"

    def test_process_document_content_missing_url_raises(self) -> None:
        """Test that missing URL in document content raises ValueError."""
        import pytest

        provider = GenericProvider()
        content: List[Dict[str, Any]] = [{"type": "document_url", "document_url": {}}]

        with pytest.raises(ValueError, match="must have a 'url' field"):
            provider._process_message_content(content)  # type: ignore[arg-type]

    def test_process_video_content_missing_url_raises(self) -> None:
        """Test that missing URL in video content raises ValueError."""
        import pytest

        provider = GenericProvider()
        content: List[Dict[str, Any]] = [{"type": "video_url", "video_url": {}}]

        with pytest.raises(ValueError, match="must have a 'url' field"):
            provider._process_message_content(content)  # type: ignore[arg-type]

    def test_process_audio_content_missing_url_raises(self) -> None:
        """Test that missing URL in audio content raises ValueError."""
        import pytest

        provider = GenericProvider()
        content: List[Dict[str, Any]] = [{"type": "audio_url", "audio_url": {}}]

        with pytest.raises(ValueError, match="must have a 'url' field"):
            provider._process_message_content(content)  # type: ignore[arg-type]

    def test_process_unsupported_content_type_raises(self) -> None:
        """Test that unsupported content type raises ValueError."""
        import pytest

        provider = GenericProvider()
        content: List[Dict[str, Any]] = [{"type": "unknown_type", "data": "test"}]

        with pytest.raises(ValueError, match="Unsupported content type"):
            provider._process_message_content(content)  # type: ignore[arg-type]

    def test_multimodal_models_initialized(self) -> None:
        """Test that multimodal content models are properly initialized."""
        provider = GenericProvider()

        # Verify all multimodal models are available
        assert hasattr(provider, "oci_chat_message_document_content")
        assert hasattr(provider, "oci_chat_message_document_url")
        assert hasattr(provider, "oci_chat_message_video_content")
        assert hasattr(provider, "oci_chat_message_video_url")
        assert hasattr(provider, "oci_chat_message_audio_content")
        assert hasattr(provider, "oci_chat_message_audio_url")
