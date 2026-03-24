# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Unit tests for the generic 'media' content type.

The 'media' content type accepts {"type": "media", "data": "<base64>",
"mime_type": "video/mp4"} and routes to the correct multimodal handler
based on the mime_type prefix.

This is a common format used in Google's docs and LangChain examples.
"""

from typing import Any, Dict, List

import pytest

from langchain_oci.chat_models.providers.generic import GenericProvider


class TestMediaContentType:
    """Tests for the media content type routing."""

    def test_video_routes_to_video_url(self) -> None:
        """media + video/mp4 produces a video_url content block."""
        provider = GenericProvider()
        content: List[Dict[str, Any]] = [
            {"type": "media", "data": "AAAA", "mime_type": "video/mp4"}
        ]
        result = provider._process_message_content(content)  # type: ignore[arg-type]
        assert len(result) == 1
        assert hasattr(result[0], "video_url")
        assert "data:video/mp4;base64,AAAA" in str(result[0].video_url.url)

    def test_video_webm(self) -> None:
        """media + video/webm produces a video_url content block."""
        provider = GenericProvider()
        content: List[Dict[str, Any]] = [
            {"type": "media", "data": "BBBB", "mime_type": "video/webm"}
        ]
        result = provider._process_message_content(content)  # type: ignore[arg-type]
        assert hasattr(result[0], "video_url")
        assert "data:video/webm;base64,BBBB" in str(result[0].video_url.url)

    def test_image_routes_to_image_url(self) -> None:
        """media + image/png produces an image_url content block."""
        provider = GenericProvider()
        content: List[Dict[str, Any]] = [
            {"type": "media", "data": "AAAA", "mime_type": "image/png"}
        ]
        result = provider._process_message_content(content)  # type: ignore[arg-type]
        assert len(result) == 1
        assert hasattr(result[0], "image_url")
        assert "data:image/png;base64,AAAA" in str(result[0].image_url.url)

    def test_image_jpeg(self) -> None:
        """media + image/jpeg produces an image_url content block."""
        provider = GenericProvider()
        content: List[Dict[str, Any]] = [
            {"type": "media", "data": "CCCC", "mime_type": "image/jpeg"}
        ]
        result = provider._process_message_content(content)  # type: ignore[arg-type]
        assert hasattr(result[0], "image_url")

    def test_audio_routes_to_audio_url(self) -> None:
        """media + audio/wav produces an audio_url content block."""
        provider = GenericProvider()
        content: List[Dict[str, Any]] = [
            {"type": "media", "data": "AAAA", "mime_type": "audio/wav"}
        ]
        result = provider._process_message_content(content)  # type: ignore[arg-type]
        assert len(result) == 1
        assert hasattr(result[0], "audio_url")

    def test_audio_mp3(self) -> None:
        """media + audio/mpeg produces an audio_url content block."""
        provider = GenericProvider()
        content: List[Dict[str, Any]] = [
            {"type": "media", "data": "DDDD", "mime_type": "audio/mpeg"}
        ]
        result = provider._process_message_content(content)  # type: ignore[arg-type]
        assert hasattr(result[0], "audio_url")

    def test_pdf_routes_to_document_url(self) -> None:
        """media + application/pdf produces a document_url content block."""
        provider = GenericProvider()
        content: List[Dict[str, Any]] = [
            {"type": "media", "data": "AAAA", "mime_type": "application/pdf"}
        ]
        result = provider._process_message_content(content)  # type: ignore[arg-type]
        assert len(result) == 1
        assert hasattr(result[0], "document_url")

    def test_missing_mime_type_raises(self) -> None:
        """media without mime_type raises ValueError."""
        provider = GenericProvider()
        content: List[Dict[str, Any]] = [{"type": "media", "data": "AAAA"}]
        with pytest.raises(ValueError, match="must have 'data' and 'mime_type'"):
            provider._process_message_content(content)  # type: ignore[arg-type]

    def test_missing_data_raises(self) -> None:
        """media without data raises ValueError."""
        provider = GenericProvider()
        content: List[Dict[str, Any]] = [
            {"type": "media", "mime_type": "video/mp4"}
        ]
        with pytest.raises(ValueError, match="must have 'data' and 'mime_type'"):
            provider._process_message_content(content)  # type: ignore[arg-type]

    def test_unsupported_mime_type_raises(self) -> None:
        """media with unsupported mime_type raises ValueError."""
        provider = GenericProvider()
        content: List[Dict[str, Any]] = [
            {"type": "media", "data": "AAAA", "mime_type": "text/plain"}
        ]
        with pytest.raises(ValueError, match="Unsupported mime_type"):
            provider._process_message_content(content)  # type: ignore[arg-type]

    def test_mixed_with_text(self) -> None:
        """media content works alongside text content."""
        provider = GenericProvider()
        content: List[Dict[str, Any]] = [
            {"type": "text", "text": "Describe this video."},
            {"type": "media", "data": "AAAA", "mime_type": "video/mp4"},
        ]
        result = provider._process_message_content(content)  # type: ignore[arg-type]
        assert len(result) == 2
        assert hasattr(result[0], "text")
        assert hasattr(result[1], "video_url")
