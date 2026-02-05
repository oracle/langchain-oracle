# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Unit tests for async support in ChatOCIGenAI."""

from typing import Any, AsyncIterator, Dict
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

from langchain_oci.chat_models.oci_generative_ai import ChatOCIGenAI
from langchain_oci.common.async_support import OCIAsyncClient


@pytest.fixture
def mock_oci_client():
    """Create a mock OCI client."""
    return MagicMock()


@pytest.fixture
def mock_signer():
    """Create a mock OCI signer."""
    signer = MagicMock()

    def sign_request(prepared_request):
        prepared_request.headers["Authorization"] = "signed"
        return prepared_request

    signer.side_effect = sign_request
    return signer


@pytest.fixture
def llm(mock_oci_client, mock_signer):
    """Create a ChatOCIGenAI instance with mocked dependencies."""
    llm = ChatOCIGenAI(
        model_id="meta.llama-3-70b-instruct",
        compartment_id="test-compartment",
        service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
        client=mock_oci_client,
    )
    # Manually set the signer for testing
    llm.oci_signer = mock_signer
    llm.oci_config = {}
    return llm


class TestOCIAsyncClient:
    """Tests for OCIAsyncClient."""

    def test_init_with_signer(self, mock_signer):
        """Test client initialization with a signer."""
        client = OCIAsyncClient(
            service_endpoint="https://test.endpoint.com",
            signer=mock_signer,
        )
        assert client.service_endpoint == "https://test.endpoint.com"
        assert client.signer == mock_signer

    def test_init_creates_signer_from_config(self):
        """Test that signer is created from config when not provided."""
        with patch("oci.signer.Signer") as mock_signer_class:
            mock_signer_class.return_value = MagicMock()
            config = {
                "tenancy": "test-tenancy",
                "user": "test-user",
                "fingerprint": "test-fingerprint",
                "key_file": "/path/to/key",
            }
            OCIAsyncClient(
                service_endpoint="https://test.endpoint.com",
                signer=None,
                config=config,
            )
            mock_signer_class.assert_called_once()

    def test_sign_headers(self, mock_signer):
        """Test request signing."""
        client = OCIAsyncClient(
            service_endpoint="https://test.endpoint.com",
            signer=mock_signer,
        )
        headers = client._sign_headers(
            method="POST",
            url="https://test.endpoint.com/chat",
            body={"test": "data"},
            stream=True,
        )
        assert "Authorization" in headers
        assert headers["Accept"] == "text/event-stream"


class TestChatOCIGenAIAsyncMixin:
    """Tests for ChatOCIGenAI async functionality."""

    def test_has_async_methods(self, llm):
        """Test that async methods are available."""
        assert hasattr(llm, "_agenerate")
        assert hasattr(llm, "_astream")
        assert callable(llm._agenerate)
        assert callable(llm._astream)

    def test_get_async_client(self, llm, mock_signer):
        """Test async client creation."""
        client = llm._get_async_client()
        assert isinstance(client, OCIAsyncClient)
        # Should return same instance on second call
        assert llm._get_async_client() is client

    @pytest.mark.asyncio
    async def test_agenerate_non_streaming(self, llm):
        """Test async generation without streaming."""
        # Mock response data
        mock_response = {
            "chatResponse": {
                "choices": [
                    {
                        "message": {
                            "content": [{"type": "TEXT", "text": "Hello, world!"}]
                        }
                    }
                ],
                "finishReason": "stop",
                "usage": {
                    "promptTokens": 10,
                    "completionTokens": 5,
                    "totalTokens": 15,
                },
            },
            "modelId": "meta.llama-3-70b-instruct",
            "modelVersion": "1.0",
        }

        async def mock_chat_async(*args, **kwargs) -> AsyncIterator[Dict[str, Any]]:
            yield mock_response

        with patch.object(OCIAsyncClient, "chat_async", side_effect=mock_chat_async):
            messages = [HumanMessage(content="Hello")]
            result = await llm._agenerate(messages)

            assert len(result.generations) == 1
            assert result.generations[0].message.content == "Hello, world!"

    @pytest.mark.asyncio
    async def test_astream(self, llm):
        """Test async streaming."""
        # Mock streaming events
        stream_events = [
            {"message": {"content": [{"type": "TEXT", "text": "Hello"}]}},
            {"message": {"content": [{"type": "TEXT", "text": ", "}]}},
            {"message": {"content": [{"type": "TEXT", "text": "world!"}]}},
            {"finishReason": "stop"},
        ]

        async def mock_chat_async(*args, **kwargs) -> AsyncIterator[Dict[str, Any]]:
            for event in stream_events:
                yield event

        with patch.object(OCIAsyncClient, "chat_async", side_effect=mock_chat_async):
            messages = [HumanMessage(content="Hello")]
            chunks = []
            async for chunk in llm._astream(messages):
                chunks.append(chunk)

            assert len(chunks) == 4
            # Check content from first 3 chunks
            content = "".join(c.message.content for c in chunks[:3])
            assert content == "Hello, world!"

    @pytest.mark.asyncio
    async def test_agenerate_with_streaming_flag(self, llm):
        """Test that agenerate uses streaming when is_stream is True."""
        llm.is_stream = True

        stream_events = [
            {"message": {"content": [{"type": "TEXT", "text": "Streamed!"}]}},
            {"finishReason": "stop"},
        ]

        async def mock_chat_async(*args, **kwargs) -> AsyncIterator[Dict[str, Any]]:
            for event in stream_events:
                yield event

        with patch.object(OCIAsyncClient, "chat_async", side_effect=mock_chat_async):
            messages = [HumanMessage(content="Hello")]
            result = await llm._agenerate(messages)

            assert len(result.generations) == 1
            assert "Streamed!" in result.generations[0].message.content


class TestAsyncResponseParsing:
    """Tests for async response parsing helpers."""

    def test_extract_content_generic_format(self, llm):
        """Test content extraction from Generic/Meta format response."""
        response_data = {
            "chatResponse": {
                "choices": [
                    {
                        "message": {
                            "content": [
                                {"type": "TEXT", "text": "Part 1"},
                                {"type": "TEXT", "text": "Part 2"},
                            ]
                        }
                    }
                ]
            }
        }
        content = llm._extract_content_from_response(response_data)
        assert content == "Part 1Part 2"

    def test_extract_content_cohere_v1_format(self, llm):
        """Test content extraction from Cohere V1 format response."""
        response_data = {"chatResponse": {"text": "Cohere response"}}
        content = llm._extract_content_from_response(response_data)
        assert content == "Cohere response"

    def test_extract_content_cohere_v2_format(self, llm):
        """Test content extraction from Cohere V2 format response."""
        response_data = {
            "chatResponse": {
                "message": {"content": [{"type": "TEXT", "text": "V2 response"}]}
            }
        }
        content = llm._extract_content_from_response(response_data)
        assert content == "V2 response"

    def test_extract_tool_calls_generic_format(self, llm):
        """Test tool call extraction from Generic format."""
        response_data = {
            "chatResponse": {
                "choices": [
                    {
                        "message": {
                            "toolCalls": [{"name": "get_weather", "arguments": "{}"}]
                        }
                    }
                ]
            }
        }
        tool_calls = llm._extract_tool_calls(response_data)
        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "get_weather"

    def test_extract_usage_metadata(self, llm):
        """Test usage metadata extraction."""
        # Import UsageMetadata availability check
        try:
            from langchain_core.messages import UsageMetadata

            has_usage_metadata = UsageMetadata is not None
        except ImportError:
            has_usage_metadata = False

        response_data = {
            "chatResponse": {
                "usage": {
                    "promptTokens": 100,
                    "completionTokens": 50,
                    "totalTokens": 150,
                }
            }
        }
        usage = llm._extract_usage_metadata(response_data)

        if not has_usage_metadata:
            # UsageMetadata not available in older langchain-core versions
            assert usage is None
        else:
            assert usage is not None
            # UsageMetadata is a TypedDict, so access via dict keys
            assert usage["input_tokens"] == 100
            assert usage["output_tokens"] == 50
            assert usage["total_tokens"] == 150
