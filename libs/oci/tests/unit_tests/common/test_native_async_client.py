# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Delegation to the SDK's native AsyncGenerativeAiInferenceClient.

``OCIAsyncClient`` uses the native async client when the installed ``oci``
SDK provides one, and the aiohttp fallback otherwise. These tests inject a
fake native client so both routes are exercised regardless of which SDK
version is installed.
"""

import json
from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import HumanMessage
from oci.exceptions import ServiceError

from langchain_oci.chat_models.oci_generative_ai import ChatOCIGenAI
from langchain_oci.common import async_support
from langchain_oci.common.async_support import OCIAsyncClient, OCIAsyncRequestError
from langchain_oci.common.param_compat import adjust_request_for_param_error

ENDPOINT = "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"


class FakeResponse:
    def __init__(self, data: Any):
        self.data = data


class FakeAsyncBaseClient:
    def __init__(self, owner: "FakeNativeClient"):
        self._owner = owner

    def call_api_stream(self, **kwargs: Any) -> Any:
        self._owner.stream_calls.append(kwargs)
        owner = self._owner

        async def _events() -> Any:
            if owner.stream_error is not None:
                raise owner.stream_error
            for event in owner.stream_events:
                yield event

        return _events()


class FakeNativeClient:
    """Stands in for oci.generative_ai_inference.AsyncGenerativeAiInferenceClient."""

    last_instance: "FakeNativeClient"

    def __init__(self, config: Dict[str, Any], **kwargs: Any):
        self.config = config
        self.kwargs = kwargs
        self.calls: List[Tuple[str, Dict[str, Any]]] = []
        self.stream_calls: List[Dict[str, Any]] = []
        self.stream_events: List[Dict[str, Any]] = []
        self.stream_error: Any = None
        self.response_data: Dict[str, Any] = {"ok": True}
        self.error: Any = None
        self.closed = False
        self.async_base_client = FakeAsyncBaseClient(self)
        FakeNativeClient.last_instance = self

    async def _op(self, name: str, details: Dict[str, Any]) -> FakeResponse:
        self.calls.append((name, details))
        if self.error is not None:
            raise self.error
        return FakeResponse(self.response_data)

    async def chat(self, details: Dict[str, Any], **kwargs: Any) -> FakeResponse:
        return await self._op("chat", details)

    async def embed_text(self, details: Dict[str, Any], **kwargs: Any) -> FakeResponse:
        return await self._op("embed_text", details)

    async def rerank_text(self, details: Dict[str, Any], **kwargs: Any) -> FakeResponse:
        return await self._op("rerank_text", details)

    async def generate_text(
        self, details: Dict[str, Any], **kwargs: Any
    ) -> FakeResponse:
        return await self._op("generate_text", details)

    async def close(self) -> None:
        self.closed = True


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> OCIAsyncClient:
    monkeypatch.setattr(
        async_support, "_load_native_async_client_cls", lambda: FakeNativeClient
    )
    return OCIAsyncClient(
        service_endpoint=ENDPOINT,
        signer=MagicMock(),
        config={"region": "us-chicago-1"},
    )


def _native(client: OCIAsyncClient) -> FakeNativeClient:
    native = client._native
    assert isinstance(native, FakeNativeClient)
    return native


class TestNativeConstruction:
    def test_constructed_with_sync_credentials(self, client: OCIAsyncClient) -> None:
        native = _native(client)
        assert native.config == {"region": "us-chicago-1"}
        assert native.kwargs["signer"] is client.signer
        assert native.kwargs["service_endpoint"] == ENDPOINT
        # Responses must stay camelCase wire dicts for the shared parsers.
        assert native.kwargs["skip_deserialization"] is True

    def test_no_native_class_falls_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            async_support, "_load_native_async_client_cls", lambda: None
        )
        client = OCIAsyncClient(service_endpoint=ENDPOINT, signer=MagicMock())
        assert client._native is None

    def test_construction_failure_falls_back(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class Boom:
            def __init__(self, config: Dict[str, Any], **kwargs: Any):
                raise ValueError("invalid config")

        monkeypatch.setattr(
            async_support, "_load_native_async_client_cls", lambda: Boom
        )
        client = OCIAsyncClient(service_endpoint=ENDPOINT, signer=MagicMock())
        assert client._native is None

    async def test_close_closes_native(self, client: OCIAsyncClient) -> None:
        native = _native(client)
        await client.close()
        assert native.closed is True


class TestNativeChat:
    async def test_chat_non_streaming(self, client: OCIAsyncClient) -> None:
        native = _native(client)
        native.response_data = {"chatResponse": {"choices": []}, "modelId": "m"}

        results = [
            data
            async for data in client.chat_async(
                compartment_id="ocid1.compartment.test",
                chat_request_dict={"apiFormat": "GENERIC"},
                serving_mode_dict={"servingType": "ON_DEMAND"},
                stream=False,
            )
        ]

        assert results == [native.response_data]
        name, body = native.calls[0]
        assert name == "chat"
        assert body == {
            "compartmentId": "ocid1.compartment.test",
            "servingMode": {"servingType": "ON_DEMAND"},
            "chatRequest": {"apiFormat": "GENERIC"},
        }

    async def test_chat_streaming_uses_call_api_stream(
        self, client: OCIAsyncClient
    ) -> None:
        native = _native(client)
        native.stream_events = [
            {"message": {"content": [{"type": "TEXT", "text": "Hi"}]}},
            {"finishReason": "stop"},
        ]

        events = [
            event
            async for event in client.chat_async(
                compartment_id="ocid1.compartment.test",
                chat_request_dict={"apiFormat": "GENERIC"},
                serving_mode_dict={"servingType": "ON_DEMAND"},
                stream=True,
            )
        ]

        assert events == native.stream_events
        assert native.calls == []  # the generated chat() op is never used
        stream_call = native.stream_calls[0]
        assert stream_call["resource_path"] == "/actions/chat"
        assert stream_call["method"] == "POST"
        assert stream_call["body"]["compartmentId"] == "ocid1.compartment.test"

    async def test_service_error_translated(self, client: OCIAsyncClient) -> None:
        native = _native(client)
        native.error = ServiceError(400, "400", {}, "temperature not supported")

        with pytest.raises(OCIAsyncRequestError) as exc_info:
            async for _ in client.chat_async(
                compartment_id="c",
                chat_request_dict={},
                serving_mode_dict={},
                stream=False,
            ):
                pass

        assert exc_info.value.status == 400
        body = json.loads(exc_info.value.body)
        assert body == {"code": "400", "message": "temperature not supported"}

    async def test_stream_service_error_translated(
        self, client: OCIAsyncClient
    ) -> None:
        native = _native(client)
        native.stream_error = ServiceError(429, "TooManyRequests", {}, "slow down")

        with pytest.raises(OCIAsyncRequestError) as exc_info:
            async for _ in client.chat_async(
                compartment_id="c",
                chat_request_dict={},
                serving_mode_dict={},
                stream=True,
            ):
                pass

        assert exc_info.value.status == 429

    async def test_translated_error_supports_param_compat_retry(
        self, client: OCIAsyncClient
    ) -> None:
        """A gpt-5-style 400 keeps working with the param-compat retry.

        OCI double-encodes the OpenAI-style error inside the envelope's
        ``message`` string; the translated body must round-trip through
        ``adjust_request_for_param_error`` exactly like the fallback path.
        """
        native = _native(client)
        inner = json.dumps(
            {
                "error": {
                    "param": "temperature",
                    "code": "unsupported_value",
                    "message": "temperature does not support 0.5",
                }
            }
        )
        native.error = ServiceError(400, "400", {}, inner)

        with pytest.raises(OCIAsyncRequestError) as exc_info:
            async for _ in client.chat_async(
                compartment_id="c",
                chat_request_dict={"temperature": 0.5},
                serving_mode_dict={},
                stream=False,
            ):
                pass

        chat_request = {"temperature": 0.5}
        assert adjust_request_for_param_error(exc_info.value.body, chat_request)
        assert "temperature" not in chat_request


class TestNativeOtherOps:
    async def test_embed_text(self, client: OCIAsyncClient) -> None:
        native = _native(client)
        native.response_data = {"embeddings": [[0.1, 0.2]]}
        details = {"compartmentId": "c", "inputs": ["hello"]}

        data = await client.embed_text_async(details)

        assert data == native.response_data
        assert native.calls == [("embed_text", details)]

    async def test_rerank_text(self, client: OCIAsyncClient) -> None:
        native = _native(client)
        native.response_data = {"documentRanks": [{"index": 0, "relevanceScore": 0.9}]}
        details = {"compartmentId": "c", "input": "q", "documents": ["d"]}

        data = await client.rerank_text_async(details)

        assert data == native.response_data
        assert native.calls == [("rerank_text", details)]

    async def test_generate_text_non_streaming(self, client: OCIAsyncClient) -> None:
        native = _native(client)
        native.response_data = {"inferenceResponse": {"generatedTexts": []}}
        details = {"compartmentId": "c", "inferenceRequest": {"prompt": "p"}}

        results = [data async for data in client.generate_text_async(details)]

        assert results == [native.response_data]
        assert native.calls == [("generate_text", details)]

    async def test_generate_text_streaming(self, client: OCIAsyncClient) -> None:
        native = _native(client)
        native.stream_events = [{"text": "a"}, {"text": "b"}]
        details = {"compartmentId": "c", "inferenceRequest": {"prompt": "p"}}

        events = [
            event async for event in client.generate_text_async(details, stream=True)
        ]

        assert events == native.stream_events
        assert native.stream_calls[0]["resource_path"] == "/actions/generateText"


class TestChatModelThroughNativeClient:
    """End-to-end: ChatOCIGenAI async paths ride the native client."""

    @pytest.fixture
    def llm(self, monkeypatch: pytest.MonkeyPatch) -> ChatOCIGenAI:
        from oci.base_client import BaseClient

        monkeypatch.setattr(
            async_support, "_load_native_async_client_cls", lambda: FakeNativeClient
        )
        mock_oci_client = MagicMock()
        mock_oci_client.base_client = MagicMock()
        mock_oci_client.base_client.signer = MagicMock()
        mock_oci_client.base_client.config = {}
        mock_oci_client.base_client.sanitize_for_serialization = (
            BaseClient.sanitize_for_serialization.__get__(
                mock_oci_client.base_client, type(mock_oci_client.base_client)
            )
        )
        mock_oci_client.base_client.complex_type_mappings = {}
        return ChatOCIGenAI(
            model_id="meta.llama-3-70b-instruct",
            compartment_id="test-compartment",
            service_endpoint=ENDPOINT,
            client=mock_oci_client,
        )

    async def test_agenerate(self, llm: ChatOCIGenAI) -> None:
        llm._async_client  # instantiate the adapter (and the fake native client)
        native = FakeNativeClient.last_instance
        native.response_data = {
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

        result = await llm._agenerate([HumanMessage(content="Hello")])

        assert result.generations[0].message.content == "Hello, world!"
        name, body = native.calls[0]
        assert name == "chat"
        assert body["compartmentId"] == "test-compartment"

    async def test_astream(self, llm: ChatOCIGenAI) -> None:
        llm._async_client
        native = FakeNativeClient.last_instance
        native.stream_events = [
            {"message": {"content": [{"type": "TEXT", "text": "Hello"}]}},
            {"message": {"content": [{"type": "TEXT", "text": ", world!"}]}},
            {"finishReason": "stop"},
        ]

        chunks = [chunk async for chunk in llm._astream([HumanMessage(content="Hi")])]

        content = "".join(str(c.message.content) for c in chunks)
        assert content == "Hello, world!"
        assert native.stream_calls[0]["resource_path"] == "/actions/chat"
