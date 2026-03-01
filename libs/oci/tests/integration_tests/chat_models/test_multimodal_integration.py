# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Integration tests for multimodal content support (PDF, video, audio).

These tests require real OCI credentials and access to OCI Generative AI.
Multimodal content (documents, video, audio) requires multimodal-capable models.

Currently supported models:
- Google Gemini (full multimodal: images, PDF, video, audio)
- Meta Llama Vision (images only)
- Cohere Vision (images only)
- xAI Grok (images only)

Usage:
    export OCI_COMPARTMENT_ID="ocid1.compartment..."
    export OCI_CONFIG_PROFILE="DEFAULT"
    export OCI_AUTH_TYPE="API_KEY"
    export OCI_REGION="us-chicago-1"
    export OCI_MODEL_ID="google.gemini-2.5-flash"  # or any multimodal model
    export OCI_RUN_MULTIMODAL_INTEGRATION=1

    pytest tests/integration_tests/chat_models/test_multimodal_integration.py -v
"""

import base64
import os
from typing import Any, Dict

import pytest
from langchain_core.messages import HumanMessage

from langchain_oci.chat_models import ChatOCIGenAI


def _get_config() -> Dict[str, Any]:
    """Get OCI configuration from environment."""
    compartment_id = os.environ.get("OCI_COMPARTMENT_ID")
    if not compartment_id:
        pytest.skip("OCI_COMPARTMENT_ID not set")

    region = os.getenv("OCI_REGION", "us-chicago-1")
    endpoint = os.environ.get(
        "OCI_GENAI_ENDPOINT",
        f"https://inference.generativeai.{region}.oci.oraclecloud.com",
    )

    return {
        "compartment_id": compartment_id,
        "service_endpoint": endpoint,
        "auth_type": os.environ.get("OCI_AUTH_TYPE", "API_KEY"),
        "auth_profile": os.environ.get("OCI_CONFIG_PROFILE", "DEFAULT"),
        "auth_file_location": os.path.expanduser("~/.oci/config"),
    }


def _get_llm() -> ChatOCIGenAI:
    """Create a ChatOCIGenAI instance with environment configuration.

    Uses OCI_MODEL_ID env var, defaulting to google.gemini-2.5-flash.
    Override with any multimodal-capable model for testing.
    """
    cfg = _get_config()
    model_id = os.getenv("OCI_MODEL_ID", "google.gemini-2.5-flash")

    return ChatOCIGenAI(
        model_id=model_id,
        service_endpoint=cfg["service_endpoint"],
        compartment_id=cfg["compartment_id"],
        auth_type=cfg["auth_type"],
        auth_profile=cfg["auth_profile"],
        auth_file_location=cfg["auth_file_location"],
        model_kwargs={"max_tokens": 500},
    )


def _create_simple_pdf() -> str:
    """Create a minimal valid PDF and return base64 encoded content.

    This creates the simplest possible valid PDF with just "Hello World" text.
    """
    pdf_content = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]
   /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>
endobj
4 0 obj
<< /Length 44 >>
stream
BT /F1 24 Tf 100 700 Td (Hello World) Tj ET
endstream
endobj
5 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
xref
0 6
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000266 00000 n
0000000359 00000 n
trailer
<< /Size 6 /Root 1 0 R >>
startxref
434
%%EOF"""
    return base64.b64encode(pdf_content).decode("utf-8")


def _create_simple_video() -> str:
    """Create a minimal valid MP4 and return base64 encoded content.

    This creates a tiny valid MP4 file structure (no actual video frames).
    Real video testing would require actual video content.
    """
    mp4_content = bytes(
        [
            # ftyp box (file type)
            0x00,
            0x00,
            0x00,
            0x14,  # box size (20 bytes)
            0x66,
            0x74,
            0x79,
            0x70,  # 'ftyp'
            0x69,
            0x73,
            0x6F,
            0x6D,  # 'isom' brand
            0x00,
            0x00,
            0x00,
            0x01,  # version
            0x69,
            0x73,
            0x6F,
            0x6D,  # compatible brand 'isom'
            # moov box (movie header) - minimal
            0x00,
            0x00,
            0x00,
            0x08,  # box size (8 bytes)
            0x6D,
            0x6F,
            0x6F,
            0x76,  # 'moov'
        ]
    )
    return base64.b64encode(mp4_content).decode("utf-8")


def _create_simple_audio() -> str:
    """Create a minimal valid WAV and return base64 encoded content.

    This creates a tiny valid WAV file with a short sine wave.
    """
    import math
    import struct

    sample_rate = 8000
    duration = 0.5  # seconds
    frequency = 440  # Hz (A4 note)

    # Generate samples
    samples = []
    for i in range(int(sample_rate * duration)):
        t = i / sample_rate
        sample = int(16000 * math.sin(2 * math.pi * frequency * t))
        samples.append(struct.pack("<h", sample))

    audio_data = b"".join(samples)

    # WAV header
    wav_header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + len(audio_data),
        b"WAVE",
        b"fmt ",
        16,  # fmt chunk size
        1,  # PCM format
        1,  # mono
        sample_rate,
        sample_rate * 2,  # byte rate
        2,  # block align
        16,  # bits per sample
        b"data",
        len(audio_data),
    )

    return base64.b64encode(wav_header + audio_data).decode("utf-8")


class TestPDFDocumentSupport:
    """Integration tests for PDF document support with multimodal models."""

    @pytest.mark.requires("oci")
    def test_pdf_document_analysis(self) -> None:
        """Test analyzing a PDF document with a multimodal model."""
        if os.environ.get("OCI_RUN_MULTIMODAL_INTEGRATION") != "1":
            pytest.skip("Set OCI_RUN_MULTIMODAL_INTEGRATION=1 to run this test")

        llm = _get_llm()
        pdf_b64 = _create_simple_pdf()

        message = HumanMessage(
            content=[
                {"type": "text", "text": "What text is shown in this PDF document?"},
                {
                    "type": "document_url",
                    "document_url": {"url": f"data:application/pdf;base64,{pdf_b64}"},
                },
            ]
        )

        result = llm.invoke([message])

        assert result is not None
        assert result.content is not None
        # The PDF contains "Hello World" - model should identify this
        assert isinstance(result.content, str)
        content_lower = result.content.lower()
        assert "hello" in content_lower or "world" in content_lower

    @pytest.mark.requires("oci")
    @pytest.mark.asyncio
    async def test_pdf_document_analysis_async(self) -> None:
        """Test analyzing a PDF document asynchronously."""
        if os.environ.get("OCI_RUN_MULTIMODAL_INTEGRATION") != "1":
            pytest.skip("Set OCI_RUN_MULTIMODAL_INTEGRATION=1 to run this test")

        llm = _get_llm()
        pdf_b64 = _create_simple_pdf()

        message = HumanMessage(
            content=[
                {"type": "text", "text": "What text is shown in this PDF document?"},
                {
                    "type": "document_url",
                    "document_url": {"url": f"data:application/pdf;base64,{pdf_b64}"},
                },
            ]
        )

        result = await llm.ainvoke([message])

        assert result is not None
        assert result.content is not None
        assert isinstance(result.content, str)
        content_lower = result.content.lower()
        assert "hello" in content_lower or "world" in content_lower

    @pytest.mark.requires("oci")
    def test_pdf_with_alternative_type_names(self) -> None:
        """Test PDF with 'document' and 'file' type names."""
        if os.environ.get("OCI_RUN_MULTIMODAL_INTEGRATION") != "1":
            pytest.skip("Set OCI_RUN_MULTIMODAL_INTEGRATION=1 to run this test")

        llm = _get_llm()
        pdf_b64 = _create_simple_pdf()

        # Test with 'document' type
        message = HumanMessage(
            content=[
                {"type": "text", "text": "What text is in this document?"},
                {
                    "type": "document",
                    "document": {"url": f"data:application/pdf;base64,{pdf_b64}"},
                },
            ]
        )

        result = llm.invoke([message])
        assert result is not None
        assert result.content is not None


class TestVideoSupport:
    """Integration tests for video support with multimodal models."""

    @pytest.mark.requires("oci")
    def test_video_content_processing(self) -> None:
        """Test that video content is properly processed.

        Note: This test verifies the content processing works, but may not
        produce meaningful results since we're using a minimal MP4 without
        actual video frames.
        """
        if os.environ.get("OCI_RUN_MULTIMODAL_INTEGRATION") != "1":
            pytest.skip("Set OCI_RUN_MULTIMODAL_INTEGRATION=1 to run this test")

        llm = _get_llm()
        video_b64 = _create_simple_video()

        message = HumanMessage(
            content=[
                {"type": "text", "text": "Describe what you see in this video."},
                {
                    "type": "video_url",
                    "video_url": {"url": f"data:video/mp4;base64,{video_b64}"},
                },
            ]
        )

        # This may fail or return an error message since the video is minimal
        # The test verifies the content type is processed correctly
        try:
            result = llm.invoke([message])
            assert result is not None
        except Exception as e:
            # Some errors are expected with minimal video content
            # We just want to verify the content type processing works
            assert "video" in str(e).lower() or "content" in str(e).lower()


class TestAudioSupport:
    """Integration tests for audio support with multimodal models."""

    @pytest.mark.requires("oci")
    def test_audio_content_processing(self) -> None:
        """Test that audio content is properly processed."""
        if os.environ.get("OCI_RUN_MULTIMODAL_INTEGRATION") != "1":
            pytest.skip("Set OCI_RUN_MULTIMODAL_INTEGRATION=1 to run this test")

        llm = _get_llm()
        audio_b64 = _create_simple_audio()

        message = HumanMessage(
            content=[
                {"type": "text", "text": "What kind of sound do you hear?"},
                {
                    "type": "audio_url",
                    "audio_url": {"url": f"data:audio/wav;base64,{audio_b64}"},
                },
            ]
        )

        result = llm.invoke([message])
        assert result is not None
        assert result.content is not None


class TestMultimodalMixedContent:
    """Integration tests for mixed multimodal content."""

    @pytest.mark.requires("oci")
    def test_text_and_pdf_combined(self) -> None:
        """Test combining text instructions with PDF content."""
        if os.environ.get("OCI_RUN_MULTIMODAL_INTEGRATION") != "1":
            pytest.skip("Set OCI_RUN_MULTIMODAL_INTEGRATION=1 to run this test")

        llm = _get_llm()
        pdf_b64 = _create_simple_pdf()

        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "I'm going to show you a PDF. "
                    "Please extract any text you see and tell me what it says.",
                },
                {
                    "type": "document_url",
                    "document_url": {"url": f"data:application/pdf;base64,{pdf_b64}"},
                },
                {
                    "type": "text",
                    "text": "Respond with just the extracted text, nothing else.",
                },
            ]
        )

        result = llm.invoke([message])

        assert result is not None
        assert result.content is not None
        # Should contain the text from the PDF
        assert isinstance(result.content, str)
        content_lower = result.content.lower()
        assert "hello" in content_lower or "world" in content_lower


class TestMultimodalStreaming:
    """Integration tests for multimodal streaming."""

    @pytest.mark.requires("oci")
    @pytest.mark.asyncio
    async def test_pdf_streaming_async(self) -> None:
        """Test streaming response with PDF content."""
        if os.environ.get("OCI_RUN_MULTIMODAL_INTEGRATION") != "1":
            pytest.skip("Set OCI_RUN_MULTIMODAL_INTEGRATION=1 to run this test")

        llm = _get_llm()
        pdf_b64 = _create_simple_pdf()

        message = HumanMessage(
            content=[
                {"type": "text", "text": "What text is in this PDF?"},
                {
                    "type": "document_url",
                    "document_url": {"url": f"data:application/pdf;base64,{pdf_b64}"},
                },
            ]
        )

        chunks = []
        async for chunk in llm.astream([message]):
            chunks.append(chunk)

        assert len(chunks) > 0
        # Combine all chunk content
        full_content = "".join(
            str(chunk.content) for chunk in chunks if chunk.content
        ).lower()
        assert "hello" in full_content or "world" in full_content
