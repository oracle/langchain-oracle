# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Tests for token usage reporting while streaming.

OCI only reports token usage on a chat stream when the request sets
``stream_options.is_include_usage``. It then sends the counts in one extra,
usage-only event *after* the finish event::

    {"message": {...}, "finishReason": "stop"}
    {"usage": {"completionTokens": 4, "promptTokens": 18, "totalTokens": 22}}

``ChatOCIGenAI`` must ask for it and surface it as ``usage_metadata`` on the
last streamed chunk, so that streamed and non-streamed calls report the same.
"""

from __future__ import annotations

import json
from functools import reduce
from typing import Any, AsyncIterator, Dict, List
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessageChunk, HumanMessage
from oci.generative_ai_inference import models

from langchain_oci.chat_models import ChatOCIGenAI
from langchain_oci.chat_models.providers.cohere import CohereProvider
from langchain_oci.chat_models.providers.generic import (
    GeminiProvider,
    GenericProvider,
    MetaProvider,
    OpenAIProvider,
)
from langchain_oci.common.async_support import OCIAsyncClient
from langchain_oci.common.param_compat import drop_unsupported_param
from langchain_oci.common.utils import OCIUtils

ENDPOINT = "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
USAGE = {"completionTokens": 4, "promptTokens": 18, "totalTokens": 22}
EXPECTED_USAGE = {"input_tokens": 18, "output_tokens": 4, "total_tokens": 22}
MESSAGES = [HumanMessage(content="Say hello.")]


def _content(text: str) -> Dict[str, Any]:
    return {
        "message": {"role": "ASSISTANT", "content": [{"type": "TEXT", "text": text}]}
    }


FINISH: Dict[str, Any] = {**_content(""), "finishReason": "stop"}
USAGE_EVENT: Dict[str, Any] = {"usage": USAGE}
# What OCI sends with stream_options.is_include_usage=True
EVENTS_WITH_USAGE: List[Dict[str, Any]] = [
    _content("Hel"),
    _content("lo"),
    FINISH,
    USAGE_EVENT,
]
# What OCI sends without it (or from a model that never reports usage)
EVENTS_WITHOUT_USAGE: List[Dict[str, Any]] = [_content("Hel"), _content("lo"), FINISH]


def _chat_model(model_id: str = "meta.llama-3-70b-instruct", **kwargs: Any) -> Any:
    return ChatOCIGenAI(
        model_id=model_id,
        compartment_id="test-compartment",
        service_endpoint=ENDPOINT,
        client=MagicMock(),
        **kwargs,
    )


def _mock_sync_stream(llm: Any, events: List[Dict[str, Any]]) -> List[Any]:
    """Make ``llm.client.chat`` return ``events``; returns the requests it receives."""
    requests: List[Any] = []
    response = MagicMock()
    response.data.events.return_value = [MagicMock(data=json.dumps(e)) for e in events]

    def chat(request: Any) -> Any:
        requests.append(request)
        return response

    llm.client.chat = chat
    return requests


def _merge(chunks: List[Any]) -> AIMessageChunk:
    return reduce(
        lambda a, b: a + b, [c.message if hasattr(c, "message") else c for c in chunks]
    )


# ---------------------------------------------------------------------------
# Provider hook
# ---------------------------------------------------------------------------

ALL_PROVIDERS = [
    GenericProvider,
    MetaProvider,
    OpenAIProvider,
    GeminiProvider,
    CohereProvider,
]


@pytest.mark.parametrize("provider_cls", ALL_PROVIDERS)
def test_provider_returns_usage_of_usage_only_event(provider_cls: Any) -> None:
    assert provider_cls().chat_stream_usage(USAGE_EVENT) == USAGE


@pytest.mark.parametrize("provider_cls", ALL_PROVIDERS)
@pytest.mark.parametrize("event", [_content("hi"), FINISH, {"text": "hi"}])
def test_provider_ignores_events_without_usage(provider_cls: Any, event: Any) -> None:
    assert provider_cls().chat_stream_usage(event) is None


def test_provider_does_not_swallow_events_that_also_carry_content() -> None:
    """Only a usage-only event is short-circuited; anything else keeps its old path."""
    provider = GenericProvider()
    assert provider.chat_stream_usage({**_content("hi"), "usage": USAGE}) is None
    assert provider.chat_stream_usage({"finishReason": "stop", "usage": USAGE}) is None


# ---------------------------------------------------------------------------
# UsageMetadata conversion
# ---------------------------------------------------------------------------


def test_usage_metadata_from_dict() -> None:
    assert OCIUtils.usage_metadata_from_dict(USAGE) == EXPECTED_USAGE


def test_usage_metadata_from_dict_derives_missing_total() -> None:
    usage = OCIUtils.usage_metadata_from_dict(
        {"promptTokens": 5, "completionTokens": 2}
    )
    assert usage == {"input_tokens": 5, "output_tokens": 2, "total_tokens": 7}


@pytest.mark.parametrize("empty", [None, {}])
def test_usage_metadata_from_dict_without_usage(empty: Any) -> None:
    assert OCIUtils.usage_metadata_from_dict(empty) is None


# ---------------------------------------------------------------------------
# Request: stream_options
# ---------------------------------------------------------------------------


@pytest.mark.requires("oci")
@pytest.mark.parametrize(
    "model_id", ["meta.llama-3-70b-instruct", "cohere.command-r-plus"]
)
def test_streaming_request_asks_for_usage(model_id: str) -> None:
    request = _chat_model(model_id)._prepare_request(MESSAGES, None, stream=True)
    assert request.chat_request.stream_options.is_include_usage is True


@pytest.mark.requires("oci")
def test_non_streaming_request_does_not_set_stream_options() -> None:
    request = _chat_model()._prepare_request(MESSAGES, None, stream=False)
    assert request.chat_request.stream_options is None


@pytest.mark.requires("oci")
def test_stream_usage_false_does_not_set_stream_options() -> None:
    request = _chat_model(stream_usage=False)._prepare_request(
        MESSAGES, None, stream=True
    )
    assert request.chat_request.stream_options is None


@pytest.mark.requires("oci")
def test_stream_options_from_model_kwargs_take_precedence() -> None:
    llm = _chat_model(
        model_kwargs={"stream_options": models.StreamOptions(is_include_usage=False)}
    )
    request = llm._prepare_request(MESSAGES, None, stream=True)
    assert request.chat_request.stream_options.is_include_usage is False


# ---------------------------------------------------------------------------
# Streaming: sync
# ---------------------------------------------------------------------------


@pytest.mark.requires("oci")
def test_stream_reports_usage_on_last_chunk() -> None:
    llm = _chat_model()
    requests = _mock_sync_stream(llm, EVENTS_WITH_USAGE)

    chunks = list(llm.stream(MESSAGES))

    assert requests[0].chat_request.stream_options.is_include_usage is True
    usage_positions = [i for i, c in enumerate(chunks) if c.usage_metadata]
    finish_positions = [
        i for i, c in enumerate(chunks) if c.additional_kwargs.get("finish_reason")
    ]
    assert len(usage_positions) == 1 and len(finish_positions) == 1
    assert usage_positions[0] > finish_positions[0]  # usage arrives after the finish
    assert chunks[usage_positions[0]].usage_metadata == EXPECTED_USAGE
    assert chunks[usage_positions[0]].content == ""
    merged = _merge(chunks)
    assert merged.content == "Hello"
    assert merged.usage_metadata == EXPECTED_USAGE


@pytest.mark.requires("oci")
def test_stream_without_usage_event_is_unchanged() -> None:
    llm = _chat_model()
    _mock_sync_stream(llm, EVENTS_WITHOUT_USAGE)

    chunks = list(llm.stream(MESSAGES))

    assert all(c.usage_metadata is None for c in chunks)
    assert _merge(chunks).content == "Hello"


@pytest.mark.requires("oci")
def test_invoke_with_is_stream_reports_usage() -> None:
    """``is_stream=True`` makes ``invoke`` aggregate a stream; usage must survive."""
    llm = _chat_model(is_stream=True)
    _mock_sync_stream(llm, EVENTS_WITH_USAGE)

    response = llm.invoke(MESSAGES)

    assert response.content == "Hello"
    assert response.usage_metadata == EXPECTED_USAGE


# ---------------------------------------------------------------------------
# Streaming: async
# ---------------------------------------------------------------------------


def _mock_async_stream(
    events: List[Dict[str, Any]], requests: List[Dict[str, Any]]
) -> Any:
    async def chat_async(*args: Any, **kwargs: Any) -> AsyncIterator[Dict[str, Any]]:
        requests.append(kwargs["chat_request_dict"])
        for event in events:
            yield event

    return patch.object(OCIAsyncClient, "chat_async", side_effect=chat_async)


def _async_chat_model() -> Any:
    from oci.base_client import BaseClient

    client = MagicMock()
    client.base_client = MagicMock()
    client.base_client.config = {}
    # Real serializer, so the test sees the actual request JSON
    client.base_client.sanitize_for_serialization = (
        BaseClient.sanitize_for_serialization.__get__(
            client.base_client, type(client.base_client)
        )
    )
    client.base_client.complex_type_mappings = {}
    return ChatOCIGenAI(
        model_id="meta.llama-3-70b-instruct",
        compartment_id="test-compartment",
        service_endpoint=ENDPOINT,
        client=client,
    )


@pytest.mark.requires("oci")
async def test_astream_reports_usage_on_last_chunk() -> None:
    llm = _async_chat_model()
    requests: List[Dict[str, Any]] = []

    with _mock_async_stream(EVENTS_WITH_USAGE, requests):
        chunks = [chunk async for chunk in llm._astream(MESSAGES)]

    assert requests[0]["streamOptions"] == {"isIncludeUsage": True}
    with_usage = [c for c in chunks if c.message.usage_metadata]
    assert len(with_usage) == 1
    assert with_usage[0].message.usage_metadata == EXPECTED_USAGE
    assert with_usage[0].message.content == ""
    assert any(
        c.message.additional_kwargs.get("finish_reason") == "stop" for c in chunks
    )
    merged = _merge(chunks)
    assert merged.content == "Hello"
    assert merged.usage_metadata == EXPECTED_USAGE


@pytest.mark.requires("oci")
async def test_astream_without_usage_event_is_unchanged() -> None:
    llm = _async_chat_model()

    with _mock_async_stream(EVENTS_WITHOUT_USAGE, []):
        chunks = [chunk async for chunk in llm._astream(MESSAGES)]

    assert all(c.message.usage_metadata is None for c in chunks)
    assert _merge(chunks).content == "Hello"


# ---------------------------------------------------------------------------
# 400 retry: a model that rejects the option can have it dropped
# ---------------------------------------------------------------------------


def test_drop_unsupported_stream_options_on_sdk_request() -> None:
    request = models.GenericChatRequest(
        stream_options=models.StreamOptions(is_include_usage=True)
    )
    assert drop_unsupported_param(request, "streamOptions") is True
    assert request.stream_options is None


def test_drop_unsupported_stream_options_on_wire_dict() -> None:
    request = {"streamOptions": {"isIncludeUsage": True}, "maxTokens": 10}
    assert drop_unsupported_param(request, "streamOptions") is True
    assert request == {"maxTokens": 10}
