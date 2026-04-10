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
    # Set up base_client with signer (as accessed by async mixin)
    mock_oci_client.base_client = MagicMock()
    mock_oci_client.base_client.signer = mock_signer
    mock_oci_client.base_client.config = {}

    llm = ChatOCIGenAI(
        model_id="meta.llama-3-70b-instruct",
        compartment_id="test-compartment",
        service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
        client=mock_oci_client,
    )
    return llm


@pytest.fixture
def llm_cohere(mock_oci_client, mock_signer):
    """Create a Cohere ChatOCIGenAI instance with mocked dependencies."""
    mock_oci_client.base_client = MagicMock()
    mock_oci_client.base_client.signer = mock_signer
    mock_oci_client.base_client.config = {}

    llm = ChatOCIGenAI(
        model_id="cohere.command-r-plus",
        compartment_id="test-compartment",
        service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
        client=mock_oci_client,
    )
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
            mock_signer_class.from_config.return_value = MagicMock()
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
            mock_signer_class.from_config.assert_called_once_with(config)

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
        """Test async client creation via cached_property."""
        client = llm._async_client
        assert isinstance(client, OCIAsyncClient)
        # Should return same instance on second call (cached_property)
        assert llm._async_client is client

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

        async def mock_chat_async(
            *args: Any, **kwargs: Any
        ) -> AsyncIterator[Dict[str, Any]]:
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
        stream_events: list[Dict[str, Any]] = [
            {"message": {"content": [{"type": "TEXT", "text": "Hello"}]}},
            {"message": {"content": [{"type": "TEXT", "text": ", "}]}},
            {"message": {"content": [{"type": "TEXT", "text": "world!"}]}},
            {"finishReason": "stop"},
        ]

        async def mock_chat_async(
            *args: Any, **kwargs: Any
        ) -> AsyncIterator[Dict[str, Any]]:
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

        stream_events: list[Dict[str, Any]] = [
            {"message": {"content": [{"type": "TEXT", "text": "Streamed!"}]}},
            {"finishReason": "stop"},
        ]

        async def mock_chat_async(
            *args: Any, **kwargs: Any
        ) -> AsyncIterator[Dict[str, Any]]:
            for event in stream_events:
                yield event

        with patch.object(OCIAsyncClient, "chat_async", side_effect=mock_chat_async):
            messages = [HumanMessage(content="Hello")]
            result = await llm._agenerate(messages)

            assert len(result.generations) == 1
            assert "Streamed!" in result.generations[0].message.content


class TestAsyncClientSessionManagement:
    """Tests for OCIAsyncClient session management."""

    @pytest.mark.asyncio
    async def test_session_reuse(self, mock_signer):
        """Test that ClientSession is reused across requests."""
        client = OCIAsyncClient(
            service_endpoint="https://test.endpoint.com",
            signer=mock_signer,
        )
        session1 = await client._get_session()
        session2 = await client._get_session()
        assert session1 is session2
        await client.close()

    @pytest.mark.asyncio
    async def test_close_releases_session(self, mock_signer):
        """Test that close() releases the session."""
        client = OCIAsyncClient(
            service_endpoint="https://test.endpoint.com",
            signer=mock_signer,
        )
        session = await client._get_session()
        assert not session.closed
        await client.close()
        assert client._session is None

    @pytest.mark.asyncio
    async def test_get_session_after_close_creates_new(self, mock_signer):
        """Test that _get_session creates new session after close."""
        client = OCIAsyncClient(
            service_endpoint="https://test.endpoint.com",
            signer=mock_signer,
        )
        session1 = await client._get_session()
        await client.close()
        session2 = await client._get_session()
        assert session1 is not session2
        await client.close()


class TestChatOCIGenAIAsyncClose:
    """Tests for ChatOCIGenAI async close functionality."""

    @pytest.mark.asyncio
    async def test_aclose_cleans_up_client(self, mock_oci_client, mock_signer):
        """Test that aclose() cleans up the async client."""
        # Set up base_client with signer
        mock_oci_client.base_client = MagicMock()
        mock_oci_client.base_client.signer = mock_signer
        mock_oci_client.base_client.config = {}

        llm = ChatOCIGenAI(
            model_id="meta.llama-3-70b-instruct",
            compartment_id="test-compartment",
            service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
            client=mock_oci_client,
        )

        # Create async client via cached_property
        client = llm._async_client
        assert client is not None
        assert "_async_client" in llm.__dict__

        # Close should clean up (removes from __dict__ to allow re-creation)
        await llm.aclose()
        assert "_async_client" not in llm.__dict__


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

    def test_extract_content_cohere_v1_format(self, llm_cohere):
        """Test content extraction from Cohere V1 format response."""
        response_data = {"chatResponse": {"text": "Cohere response"}}
        content = llm_cohere._extract_content_from_response(response_data)
        assert content == "Cohere response"

    def test_extract_content_cohere_v2_format(self, llm_cohere):
        """Test content extraction from Cohere V2 format response."""
        response_data = {
            "chatResponse": {
                "message": {"content": [{"type": "TEXT", "text": "V2 response"}]}
            }
        }
        content = llm_cohere._extract_content_from_response(response_data)
        assert content == "V2 response"

    def test_extract_content_cohere_provider_with_generic_payload(self, llm_cohere):
        """Test Cohere provider does not parse Generic-format payloads."""
        response_data = {
            "chatResponse": {
                "choices": [
                    {
                        "message": {
                            "content": [{"type": "TEXT", "text": "Generic response"}]
                        }
                    }
                ]
            }
        }
        with pytest.warns(UserWarning, match="selected provider matches"):
            content = llm_cohere._extract_content_from_response(response_data)
        assert content == ""

    def test_extract_content_generic_provider_with_cohere_payload(self, llm):
        """Test Generic provider does not parse Cohere-format payloads."""
        response_data: Dict[str, Any] = {
            "chatResponse": {
                "message": {"content": [{"type": "TEXT", "text": "V2 response"}]}
            }
        }
        with pytest.warns(UserWarning, match="selected provider matches"):
            content = llm._extract_content_from_response(response_data)
        assert content == ""
        response_data = {"chatResponse": {"text": "Cohere response"}}
        with pytest.warns(UserWarning, match="selected provider matches"):
            content = llm._extract_content_from_response(response_data)
        assert content == ""

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

        assert usage is not None
        # UsageMetadata is a TypedDict, access via dict-style keys
        assert usage["input_tokens"] == 100
        assert usage["output_tokens"] == 50
        assert usage["total_tokens"] == 150


class TestSerializeOciModelHelpers:
    """Tests for serialize_oci_model helper behavior."""

    def test_list_of_models(self):
        """Test serialization of a list of OCI models."""
        from oci.generative_ai_inference.models import TextContent

        from langchain_oci.common.async_support import serialize_oci_model

        items = [TextContent(text="a"), TextContent(text="b")]
        result = serialize_oci_model(items)
        assert result == [{"type": "TEXT", "text": "a"}, {"type": "TEXT", "text": "b"}]

    def test_none_attributes_excluded(self):
        """Test that None attributes are excluded from output."""
        from oci.generative_ai_inference.models import FunctionDefinition

        from langchain_oci.common.async_support import serialize_oci_model

        tool = FunctionDefinition(name="test", description="desc")
        result = serialize_oci_model(tool)
        assert "parameters" not in result
        assert result["name"] == "test"


class TestSerializeOciModel:
    """Tests for serialize_oci_model preserving user-defined tool parameter names.

    Regression tests for issue #188: the previous approach (to_dict +
    _convert_keys_to_camel) converted snake_case tool argument names to
    camelCase, breaking tool execution. serialize_oci_model uses the SDK's
    own attribute_map so only OCI fields get renamed.
    """

    def test_generic_tool_parameters_preserved(self):
        """JSON Schema property names inside tool 'parameters' must not be converted."""
        from oci.generative_ai_inference.models import (
            FunctionDefinition,
            GenericChatRequest,
            TextContent,
            UserMessage,
        )

        from langchain_oci.common.async_support import serialize_oci_model

        req = GenericChatRequest(
            messages=[UserMessage(content=[TextContent(text="hello")])],
            max_tokens=100,
            tools=[
                FunctionDefinition(
                    name="websearch",
                    description="do websearch",
                    parameters={
                        "type": "object",
                        "properties": {
                            "query_web": {"type": "string"},
                            "max_results": {"type": "integer"},
                        },
                        "required": ["query_web"],
                    },
                )
            ],
        )
        result = serialize_oci_model(req)
        props = result["tools"][0]["parameters"]["properties"]
        assert "query_web" in props, f"got {list(props.keys())}"
        assert "max_results" in props
        # OCI structural keys must be camelCase
        assert "apiFormat" in result
        assert "maxTokens" in result

    def test_cohere_v1_parameter_definitions_preserved(self):
        """Cohere V1: user-defined parameter names preserved, SDK fields converted."""
        from oci.generative_ai_inference.models import CohereTool

        from langchain_oci.common.async_support import serialize_oci_model

        tool = CohereTool(
            name="websearch",
            description="do websearch",
            parameter_definitions={
                "query_web": {
                    "type": "string",
                    "description": "search query",
                    "is_required": True,
                },
            },
        )
        result = serialize_oci_model(tool)
        pd = result["parameterDefinitions"]
        # User-defined parameter name preserved
        assert "query_web" in pd
        assert "queryWeb" not in pd

    def test_cohere_v2_nested_function_parameters_preserved(self):
        """Cohere V2: function.parameters JSON Schema preserved."""
        from oci.generative_ai_inference.models import CohereToolV2, Function

        from langchain_oci.common.async_support import serialize_oci_model

        tool = CohereToolV2(
            type="FUNCTION",
            function=Function(
                name="websearch",
                description="do websearch",
                parameters={
                    "type": "object",
                    "properties": {"query_web": {"type": "string"}},
                    "required": ["query_web"],
                },
            ),
        )
        result = serialize_oci_model(tool)
        props = result["function"]["parameters"]["properties"]
        assert "query_web" in props
        assert "queryWeb" not in props

    def test_plain_dict_keys_preserved(self):
        """Plain dict keys are never renamed; values are recursed."""
        from langchain_oci.common.async_support import serialize_oci_model

        d = {"snake_key": "value", "nested": {"inner_key": 1}}
        result = serialize_oci_model(d)
        assert result == d
        assert "snake_key" in result
        assert "inner_key" in result["nested"]

    def test_none_and_primitives(self):
        """Primitives pass through unchanged."""
        from langchain_oci.common.async_support import serialize_oci_model

        assert serialize_oci_model("hello") == "hello"
        assert serialize_oci_model(42) == 42
        assert serialize_oci_model(None) is None
        assert serialize_oci_model(True) is True


class TestAsyncClientErrorHandling:
    """Tests for OCIAsyncClient error handling."""

    def test_ensure_signer_raises_on_invalid_config(self):
        """Test that invalid config raises an error."""
        invalid_config = {
            "tenancy": "invalid",
            "user": "invalid",
            "fingerprint": "invalid",
            "key_file": "/nonexistent/path",
        }

        with pytest.raises(ValueError, match="Failed to create OCI signer"):
            OCIAsyncClient(
                service_endpoint="https://test.endpoint.com",
                signer=None,
                config=invalid_config,
            )

    @pytest.mark.asyncio
    async def test_chat_async_http_error(self, mock_signer):
        """Test that HTTP errors are properly raised."""
        from contextlib import asynccontextmanager

        client = OCIAsyncClient(
            service_endpoint="https://test.endpoint.com",
            signer=mock_signer,
        )

        class MockResponse:
            status = 400

            async def text(self):
                return "Bad Request"

        @asynccontextmanager
        async def mock_arequest(*args, **kwargs):
            yield MockResponse()

        with patch.object(client, "_arequest", mock_arequest):
            with pytest.raises(RuntimeError, match="OCI GenAI request failed"):
                async for _ in client.chat_async(
                    compartment_id="test",
                    chat_request_dict={},
                    serving_mode_dict={},
                ):
                    pass


class TestAsyncStreamParsing:
    """Tests for SSE stream parsing."""

    @pytest.mark.asyncio
    async def test_parse_sse_async_valid_data(self, mock_signer):
        """Test parsing valid SSE data."""
        client = OCIAsyncClient(
            service_endpoint="https://test.endpoint.com",
            signer=mock_signer,
        )

        class MockStreamReader:
            def __init__(self, lines):
                self.lines = iter(lines)

            def __aiter__(self):
                return self

            async def __anext__(self):
                try:
                    return next(self.lines)
                except StopIteration:
                    raise StopAsyncIteration

        sse_lines = [
            b'data: {"message": "hello"}',
            b"",  # Empty line should be skipped
            b'data: {"message": "world"}',
        ]

        stream = MockStreamReader(sse_lines)
        events = []
        async for event in client._parse_sse_async(stream):  # type: ignore[arg-type]
            events.append(event)

        assert len(events) == 2
        assert events[0] == {"message": "hello"}
        assert events[1] == {"message": "world"}

    @pytest.mark.asyncio
    async def test_parse_sse_async_done_marker(self, mock_signer):
        """Test that [DONE] marker is skipped."""
        client = OCIAsyncClient(
            service_endpoint="https://test.endpoint.com",
            signer=mock_signer,
        )

        class MockStreamReader:
            def __init__(self, lines):
                self.lines = iter(lines)

            def __aiter__(self):
                return self

            async def __anext__(self):
                try:
                    return next(self.lines)
                except StopIteration:
                    raise StopAsyncIteration

        sse_lines = [
            b'data: {"message": "hello"}',
            b"data: [DONE]",
        ]

        stream = MockStreamReader(sse_lines)
        events = []
        async for event in client._parse_sse_async(stream):  # type: ignore[arg-type]
            events.append(event)

        assert len(events) == 1
        assert events[0] == {"message": "hello"}

    @pytest.mark.asyncio
    async def test_parse_sse_async_invalid_json(self, mock_signer):
        """Test that invalid JSON is skipped."""
        client = OCIAsyncClient(
            service_endpoint="https://test.endpoint.com",
            signer=mock_signer,
        )

        class MockStreamReader:
            def __init__(self, lines):
                self.lines = iter(lines)

            def __aiter__(self):
                return self

            async def __anext__(self):
                try:
                    return next(self.lines)
                except StopIteration:
                    raise StopAsyncIteration

        sse_lines = [
            b'data: {"valid": "json"}',
            b"data: {invalid json}",
            b'data: {"another": "valid"}',
        ]

        stream = MockStreamReader(sse_lines)
        events = []
        async for event in client._parse_sse_async(stream):  # type: ignore[arg-type]
            events.append(event)

        assert len(events) == 2
        assert events[0] == {"valid": "json"}
        assert events[1] == {"another": "valid"}


class TestAsyncGenerationEdgeCases:
    """Tests for async generation edge cases."""

    @pytest.fixture
    def llm_with_mock(self, mock_oci_client, mock_signer):
        """Create a ChatOCIGenAI instance with mocked dependencies."""
        # Set up base_client with signer
        mock_oci_client.base_client = MagicMock()
        mock_oci_client.base_client.signer = mock_signer
        mock_oci_client.base_client.config = {}

        llm = ChatOCIGenAI(
            model_id="meta.llama-3-70b-instruct",
            compartment_id="test-compartment",
            service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
            client=mock_oci_client,
        )
        return llm

    @pytest.mark.asyncio
    async def test_agenerate_no_response_raises_error(self, llm_with_mock):
        """Test that missing response raises RuntimeError."""

        async def mock_chat_async_empty(*args, **kwargs):
            return
            yield  # Make this a generator that yields nothing

        with patch.object(
            OCIAsyncClient, "chat_async", side_effect=mock_chat_async_empty
        ):
            messages = [HumanMessage(content="Hello")]
            with pytest.raises(RuntimeError, match="No response received"):
                await llm_with_mock._agenerate(messages)

    @pytest.mark.asyncio
    async def test_astream_with_run_manager(self, llm_with_mock):
        """Test streaming with callback manager."""
        from unittest.mock import AsyncMock

        stream_events: list[Dict[str, Any]] = [
            {"message": {"content": [{"type": "TEXT", "text": "Hello"}]}},
            {"finishReason": "stop"},
        ]

        async def mock_chat_async(*args, **kwargs):
            for event in stream_events:
                yield event

        mock_run_manager = AsyncMock()
        mock_run_manager.on_llm_new_token = AsyncMock()

        with patch.object(OCIAsyncClient, "chat_async", side_effect=mock_chat_async):
            messages = [HumanMessage(content="Hello")]
            chunks = []
            async for chunk in llm_with_mock._astream(
                messages, run_manager=mock_run_manager
            ):
                chunks.append(chunk)

            # Should have called on_llm_new_token for the content chunk
            assert mock_run_manager.on_llm_new_token.called

    def test_extract_content_empty_response(self, llm_with_mock):
        """Test content extraction with empty response."""
        response_data: Dict[str, Any] = {"chatResponse": {}}
        content = llm_with_mock._extract_content_from_response(response_data)
        assert content == ""

    def test_extract_content_string_content(self, llm_with_mock):
        """Test content extraction when content is a string."""
        response_data = {
            "chatResponse": {"choices": [{"message": {"content": "direct string"}}]}
        }
        content = llm_with_mock._extract_content_from_response(response_data)
        assert content == "direct string"

    def test_extract_tool_calls_cohere_v1(self, llm_with_mock):
        """Test tool call extraction from Cohere V1 format."""
        response_data = {
            "chatResponse": {
                "toolCalls": [{"name": "search", "parameters": {"query": "test"}}]
            }
        }
        tool_calls = llm_with_mock._extract_tool_calls(response_data)
        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "search"

    def test_extract_tool_calls_cohere_v2(self, llm_with_mock):
        """Test tool call extraction from Cohere V2 format."""
        response_data = {
            "chatResponse": {
                "message": {"toolCalls": [{"name": "calculator", "arguments": "{}"}]}
            }
        }
        tool_calls = llm_with_mock._extract_tool_calls(response_data)
        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "calculator"

    def test_extract_tool_calls_empty(self, llm_with_mock):
        """Test tool call extraction with no tool calls."""
        response_data: Dict[str, Any] = {"chatResponse": {}}
        tool_calls = llm_with_mock._extract_tool_calls(response_data)
        assert tool_calls == []

    def test_extract_generation_info_with_citations(self, llm_with_mock):
        """Test generation info extraction with Cohere documents/citations."""
        response_data = {
            "chatResponse": {
                "finishReason": "stop",
                "documents": [{"id": "doc1", "snippet": "test"}],
                "citations": [{"start": 0, "end": 5, "document_ids": ["doc1"]}],
            }
        }
        info = llm_with_mock._extract_generation_info(response_data)
        assert info["finish_reason"] == "stop"
        assert "documents" in info
        assert "citations" in info

    def test_extract_usage_metadata_none(self, llm_with_mock):
        """Test usage metadata extraction with no usage data."""
        response_data: Dict[str, Any] = {"chatResponse": {}}
        usage = llm_with_mock._extract_usage_metadata(response_data)
        assert usage is None
