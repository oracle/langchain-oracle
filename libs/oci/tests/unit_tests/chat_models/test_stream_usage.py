# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Token usage on streamed ChatOCIGenAI responses (#316).

OCI reports usage on a chat stream only when the request sets
``stream_options.is_include_usage``, and where the counts land depends on the
API format. The event shapes below were captured live (``pad`` fields
omitted).

GENERIC format (Meta, OpenAI, Google, xAI): one usage-only event *after* the
finish event, followed on some models by the ``[DONE]`` sentinel::

    {
        "index": 0,
        "message": {"role": "ASSISTANT", "content": [{"type": "TEXT", "text": "Hi"}]},
    }
    {
        "message": {"role": "ASSISTANT", "content": [{"type": "TEXT", "text": ""}]},
        "finishReason": "stop",
    }
    {"usage": {"completionTokens": 10, "promptTokens": 42, "totalTokens": 52}}
    [DONE]

COHERE format: usage rides on the finish event itself::

    {"apiFormat": "COHERE", "text": "Hi"}
    {
        "apiFormat": "COHERE",
        "text": "",
        "finishReason": "COMPLETE",
        "chatHistory": [],
        "usage": {"completionTokens": 2, "promptTokens": 7, "totalTokens": 9},
    }
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages.ai import add_usage
from oci.exceptions import ServiceError
from oci.generative_ai_inference.models import (
    CohereChatRequest,
    CompletionTokensDetails,
    GenericChatRequest,
    PromptTokensDetails,
    StreamOptions,
    Usage,
)

from langchain_oci.chat_models import ChatOCIGenAI
from langchain_oci.chat_models.providers.cohere import CohereProvider
from langchain_oci.chat_models.providers.generic import GenericProvider
from langchain_oci.common.async_support import OCIAsyncClient
from langchain_oci.common.param_compat import drop_unsupported_param
from langchain_oci.common.utils import OCIUtils

META = "meta.llama-3.3-70b-instruct"
COHERE = "cohere.command-a-03-2025"
OPENAI = "openai.gpt-4.1-mini"
GEMINI = "google.gemini-2.5-flash"


def _meta_text(text: str) -> Dict[str, Any]:
    return {
        "index": 0,
        "message": {"role": "ASSISTANT", "content": [{"type": "TEXT", "text": text}]},
    }


META_FINISH: Dict[str, Any] = {
    "message": {"role": "ASSISTANT", "content": [{"type": "TEXT", "text": ""}]},
    "finishReason": "stop",
}
META_USAGE: Dict[str, Any] = {
    "usage": {"completionTokens": 10, "promptTokens": 42, "totalTokens": 52}
}
META_USAGE_METADATA = {"input_tokens": 42, "output_tokens": 10, "total_tokens": 52}

# OpenAI-served models add serviceTier to every event and token details to usage.
OPENAI_FINISH: Dict[str, Any] = {"finishReason": "stop", "serviceTier": "default"}
OPENAI_USAGE: Dict[str, Any] = {
    "serviceTier": "default",
    "usage": {
        "completionTokens": 2,
        "promptTokens": 14,
        "totalTokens": 16,
        "completionTokensDetails": {
            "acceptedPredictionTokens": 0,
            "reasoningTokens": 0,
        },
        "promptTokensDetails": {"cachedTokens": 0},
    },
}
OPENAI_USAGE_METADATA = {
    "input_tokens": 14,
    "output_tokens": 2,
    "total_tokens": 16,
    "input_token_details": {"cached_tokens": 0},
    "output_token_details": {"accepted_prediction_tokens": 0, "reasoning_tokens": 0},
}

# Gemini sends usage even when not asked and, when the whole output budget
# went to reasoning, omits completionTokens.
GEMINI_FINISH: Dict[str, Any] = {"finishReason": "max_tokens"}
GEMINI_USAGE: Dict[str, Any] = {
    "usage": {
        "promptTokens": 7,
        "totalTokens": 24,
        "completionTokensDetails": {"reasoningTokens": 17},
    }
}
GEMINI_USAGE_METADATA = {
    "input_tokens": 7,
    "output_tokens": 0,
    "total_tokens": 24,
    "output_token_details": {"reasoning_tokens": 17},
}

COHERE_TEXT: Dict[str, Any] = {"apiFormat": "COHERE", "text": "Hi"}
COHERE_FINISH: Dict[str, Any] = {
    "apiFormat": "COHERE",
    "text": "",
    "finishReason": "COMPLETE",
    "chatHistory": [],
    "usage": {"completionTokens": 2, "promptTokens": 7, "totalTokens": 9},
}
COHERE_USAGE_METADATA = {"input_tokens": 7, "output_tokens": 2, "total_tokens": 9}


def _sse(*payloads: Any) -> MagicMock:
    """A mocked SDK stream response whose events carry the given payloads."""
    response = MagicMock()
    response.data.events.return_value = [
        MagicMock(data=p if isinstance(p, str) else json.dumps(p)) for p in payloads
    ]
    return response


def _llm(model_id: str, **kwargs: Any) -> ChatOCIGenAI:
    return ChatOCIGenAI(model_id=model_id, client=MagicMock(), **kwargs)


def _merge(chunks: List[Any]) -> Any:
    full = None
    for chunk in chunks:
        full = chunk if full is None else full + chunk
    return full


def _sent_request(llm: ChatOCIGenAI) -> Any:
    return llm.client.chat.call_args[0][0].chat_request


def _model_chunks(chunks: List[Any]) -> List[Any]:
    """Drop the empty ``chunk_position="last"`` marker that langchain-core>=1.0
    appends in ``stream()``/``astream()``; ChatOCIGenAI itself never yields it."""
    return [c for c in chunks if getattr(c, "chunk_position", None) != "last"]


# ---------------------------------------------------------------------------
# Request building: stream_options.is_include_usage
# ---------------------------------------------------------------------------


@pytest.mark.requires("oci")
@pytest.mark.parametrize("model_id", [META, COHERE, OPENAI, GEMINI])
def test_streaming_request_asks_for_usage(model_id: str) -> None:
    request = _llm(model_id)._prepare_request(
        [HumanMessage(content="hi")], stop=None, stream=True
    )
    assert request.chat_request.is_stream is True
    assert request.chat_request.stream_options.is_include_usage is True


@pytest.mark.requires("oci")
@pytest.mark.parametrize("model_id", [META, COHERE])
def test_non_streaming_request_has_no_stream_options(model_id: str) -> None:
    request = _llm(model_id)._prepare_request(
        [HumanMessage(content="hi")], stop=None, stream=False
    )
    assert request.chat_request.stream_options is None


@pytest.mark.requires("oci")
def test_stream_usage_false_opts_out() -> None:
    request = _llm(META, stream_usage=False)._prepare_request(
        [HumanMessage(content="hi")], stop=None, stream=True
    )
    assert request.chat_request.stream_options is None


@pytest.mark.requires("oci")
def test_explicit_stream_options_in_model_kwargs_win() -> None:
    llm = _llm(
        META, model_kwargs={"stream_options": StreamOptions(is_include_usage=False)}
    )
    request = llm._prepare_request([HumanMessage(content="hi")], stop=None, stream=True)
    assert request.chat_request.stream_options.is_include_usage is False


# ---------------------------------------------------------------------------
# Provider hooks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "event", [META_USAGE, OPENAI_USAGE, GEMINI_USAGE], ids=["meta", "openai", "gemini"]
)
def test_generic_usage_only_event(event: Dict[str, Any]) -> None:
    provider = GenericProvider()
    assert provider.chat_stream_usage(event) == event["usage"]
    assert provider.is_chat_stream_usage_only(event) is True
    assert provider.is_chat_stream_end(event) is False


@pytest.mark.parametrize(
    "event",
    [_meta_text("Hi"), META_FINISH, GEMINI_FINISH],
    ids=["content", "finish", "bare-finish"],
)
def test_generic_content_and_finish_events_carry_no_usage(
    event: Dict[str, Any],
) -> None:
    provider = GenericProvider()
    assert provider.chat_stream_usage(event) is None
    assert provider.is_chat_stream_usage_only(event) is False


def test_cohere_usage_rides_on_the_finish_event() -> None:
    provider = CohereProvider()
    assert provider.chat_stream_usage(COHERE_FINISH) == COHERE_FINISH["usage"]
    assert provider.is_chat_stream_end(COHERE_FINISH) is True
    assert provider.is_chat_stream_usage_only(COHERE_FINISH) is False
    assert provider.chat_stream_usage(COHERE_TEXT) is None


def test_non_dict_usage_is_ignored() -> None:
    provider = GenericProvider()
    assert provider.chat_stream_usage({"usage": None}) is None
    assert provider.is_chat_stream_usage_only({"usage": "n/a"}) is False


# ---------------------------------------------------------------------------
# Wire-dict usage mapper (shared by stream, astream and ainvoke)
# ---------------------------------------------------------------------------


def test_usage_metadata_from_dict_counts() -> None:
    assert OCIUtils.usage_metadata_from_dict(META_USAGE["usage"]) == META_USAGE_METADATA


def test_usage_metadata_from_dict_maps_token_details_like_invoke() -> None:
    assert (
        OCIUtils.usage_metadata_from_dict(OPENAI_USAGE["usage"])
        == OPENAI_USAGE_METADATA
    )


def test_usage_metadata_from_dict_drops_none_details() -> None:
    usage = OCIUtils.usage_metadata_from_dict(
        {
            "promptTokens": 1,
            "completionTokens": 1,
            "totalTokens": 2,
            "completionTokensDetails": {
                "reasoningTokens": 1,
                "rejectedPredictionTokens": None,
            },
        }
    )
    assert usage is not None
    assert usage["output_token_details"] == {"reasoning_tokens": 1}


def test_usage_metadata_from_dict_missing_completion_tokens() -> None:
    assert (
        OCIUtils.usage_metadata_from_dict(GEMINI_USAGE["usage"])
        == GEMINI_USAGE_METADATA
    )


def test_usage_metadata_from_dict_total_falls_back_to_sum() -> None:
    assert OCIUtils.usage_metadata_from_dict(
        {"promptTokens": 3, "completionTokens": 4}
    ) == {"input_tokens": 3, "output_tokens": 4, "total_tokens": 7}


@pytest.mark.parametrize("usage", [None, {}])
def test_usage_metadata_from_dict_empty(usage: Optional[Dict[str, Any]]) -> None:
    assert OCIUtils.usage_metadata_from_dict(usage) is None


@pytest.mark.requires("oci")
def test_ainvoke_usage_mapper_reports_token_details() -> None:
    llm = _llm(OPENAI)
    usage = llm._extract_usage_metadata(
        {"chatResponse": {"usage": OPENAI_USAGE["usage"]}}
    )
    assert usage == OPENAI_USAGE_METADATA
    assert llm._extract_usage_metadata({"chatResponse": {}}) is None


# ---------------------------------------------------------------------------
# Sync streaming
# ---------------------------------------------------------------------------


@pytest.mark.requires("oci")
def test_stream_generic_reports_usage_from_trailing_event() -> None:
    llm = _llm(META)
    llm.client.chat.return_value = _sse(
        _meta_text("Hel"), _meta_text("lo"), META_FINISH, META_USAGE, "[DONE]"
    )

    chunks = _model_chunks(list(llm.stream("hi")))
    full = _merge(chunks)

    assert _sent_request(llm).stream_options.is_include_usage is True
    assert full.content == "Hello"
    assert full.usage_metadata == META_USAGE_METADATA
    assert full.additional_kwargs["finish_reason"] == "stop"
    # two deltas, the finish chunk, then the usage chunk; no extra empty delta
    assert [c.content for c in chunks] == ["Hel", "lo", "", ""]
    assert chunks[-2].usage_metadata is None
    assert chunks[-1].usage_metadata == META_USAGE_METADATA


@pytest.mark.requires("oci")
def test_stream_openai_reports_token_details() -> None:
    llm = _llm(OPENAI)
    llm.client.chat.return_value = _sse(_meta_text("Hi"), OPENAI_FINISH, OPENAI_USAGE)

    full = _merge(list(llm.stream("hi")))

    assert full.content == "Hi"
    assert full.usage_metadata == OPENAI_USAGE_METADATA


@pytest.mark.requires("oci")
def test_stream_cohere_reports_usage_from_finish_event() -> None:
    llm = _llm(COHERE)
    llm.client.chat.return_value = _sse(COHERE_TEXT, COHERE_FINISH)

    chunks = _model_chunks(list(llm.stream("hi")))
    full = _merge(chunks)

    assert _sent_request(llm).stream_options.is_include_usage is True
    assert full.content == "Hi"
    assert full.usage_metadata == COHERE_USAGE_METADATA
    assert len(chunks) == 2
    assert chunks[-1].usage_metadata == COHERE_USAGE_METADATA
    assert chunks[-1].additional_kwargs["finish_reason"] == "COMPLETE"


@pytest.mark.requires("oci")
def test_stream_without_usage_event_is_unchanged() -> None:
    llm = _llm(META)
    llm.client.chat.return_value = _sse(_meta_text("Hi"), META_FINISH, "[DONE]")

    chunks = _model_chunks(list(llm.stream("hi")))

    assert _merge(chunks).usage_metadata is None
    assert len(chunks) == 2


@pytest.mark.requires("oci")
def test_stream_usage_false_still_reads_unsolicited_usage() -> None:
    """Gemini reports usage even when not asked; the parser never drops it."""
    llm = _llm(GEMINI, stream_usage=False)
    llm.client.chat.return_value = _sse(_meta_text("Hi"), GEMINI_FINISH, GEMINI_USAGE)

    full = _merge(list(llm.stream("hi")))

    assert _sent_request(llm).stream_options is None
    assert full.usage_metadata == GEMINI_USAGE_METADATA


@pytest.mark.requires("oci")
def test_invoke_with_is_stream_merges_usage() -> None:
    llm = _llm(META, is_stream=True)
    llm.client.chat.return_value = _sse(
        _meta_text("Hi"), META_FINISH, META_USAGE, "[DONE]"
    )

    message = llm.invoke("hi")

    assert isinstance(message, AIMessage)
    assert message.content == "Hi"
    assert message.usage_metadata == META_USAGE_METADATA


@pytest.mark.requires("oci")
def test_stream_retries_without_stream_options_when_model_rejects_it() -> None:
    """A model that 400s on streamOptions gets the plain streaming request."""
    llm = _llm(META)
    seen: List[Optional[StreamOptions]] = []

    def chat(request: Any) -> Any:
        seen.append(request.chat_request.stream_options)
        if len(seen) == 1:
            raise ServiceError(
                status=400,
                code="400",
                headers={},
                message=json.dumps(
                    {
                        "error": {
                            "param": "streamOptions",
                            "code": "unsupported_parameter",
                        }
                    }
                ),
            )
        return _sse(_meta_text("Hi"), META_FINISH)

    llm.client.chat.side_effect = chat

    with pytest.warns(UserWarning, match="streamOptions"):
        full = _merge(list(llm.stream("hi")))

    assert full.content == "Hi"
    assert full.usage_metadata is None
    assert seen[0] is not None and seen[0].is_include_usage is True
    assert seen[1] is None


# ---------------------------------------------------------------------------
# Async streaming
# ---------------------------------------------------------------------------


def _fake_serialize(obj: Any) -> Any:
    """Stand-in for BaseClient.sanitize_for_serialization: SDK objects ->
    camelCase wire dicts, skipping None fields."""
    if hasattr(obj, "attribute_map"):
        return {
            wire: _fake_serialize(getattr(obj, attr))
            for attr, wire in obj.attribute_map.items()
            if getattr(obj, attr) is not None
        }
    if isinstance(obj, list):
        return [_fake_serialize(o) for o in obj]
    if isinstance(obj, dict):
        return {k: _fake_serialize(v) for k, v in obj.items()}
    return obj


def _async_llm(model_id: str, **kwargs: Any) -> ChatOCIGenAI:
    client = MagicMock()
    client.base_client.signer = MagicMock()
    client.base_client.config = {}
    client.base_client.sanitize_for_serialization = _fake_serialize
    return ChatOCIGenAI(
        model_id=model_id,
        client=client,
        service_endpoint="https://example.invalid",
        **kwargs,
    )


def _install_fake_chat_async(
    monkeypatch: pytest.MonkeyPatch,
    events: List[Dict[str, Any]],
    sent: List[Dict[str, Any]],
) -> None:
    def fake_chat_async(
        self: Any,
        compartment_id: str,
        chat_request_dict: Dict[str, Any],
        serving_mode_dict: Dict[str, Any],
        stream: bool,
    ) -> Any:
        async def gen() -> Any:
            sent.append(dict(chat_request_dict))
            for event in events:
                yield event

        return gen()

    monkeypatch.setattr(OCIAsyncClient, "chat_async", fake_chat_async)


@pytest.mark.requires("oci")
async def test_astream_generic_reports_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: List[Dict[str, Any]] = []
    _install_fake_chat_async(
        monkeypatch,
        [_meta_text("Hel"), _meta_text("lo"), META_FINISH, META_USAGE],
        sent,
    )
    llm = _async_llm(META)

    chunks = _model_chunks([c async for c in llm.astream("hi")])
    full = _merge(chunks)

    assert sent[0]["streamOptions"] == {"isIncludeUsage": True}
    assert full.content == "Hello"
    assert full.usage_metadata == META_USAGE_METADATA
    assert [c.content for c in chunks] == ["Hel", "lo", "", ""]


@pytest.mark.requires("oci")
async def test_astream_cohere_reports_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: List[Dict[str, Any]] = []
    _install_fake_chat_async(monkeypatch, [COHERE_TEXT, COHERE_FINISH], sent)
    llm = _async_llm(COHERE)

    chunks = _model_chunks([c async for c in llm.astream("hi")])
    full = _merge(chunks)

    assert sent[0]["streamOptions"] == {"isIncludeUsage": True}
    assert full.content == "Hi"
    assert full.usage_metadata == COHERE_USAGE_METADATA
    assert len(chunks) == 2


@pytest.mark.requires("oci")
async def test_astream_openai_reports_token_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent: List[Dict[str, Any]] = []
    _install_fake_chat_async(
        monkeypatch, [_meta_text("Hi"), OPENAI_FINISH, OPENAI_USAGE], sent
    )
    full = _merge([c async for c in _async_llm(OPENAI).astream("hi")])
    assert full.usage_metadata == OPENAI_USAGE_METADATA


@pytest.mark.requires("oci")
async def test_astream_stream_usage_false_sends_no_stream_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent: List[Dict[str, Any]] = []
    _install_fake_chat_async(monkeypatch, [_meta_text("Hi"), META_FINISH], sent)
    llm = _async_llm(META, stream_usage=False)

    full = _merge([c async for c in llm.astream("hi")])

    assert "streamOptions" not in sent[0]
    assert full.usage_metadata is None


# ---------------------------------------------------------------------------
# Parameter-compatibility retry can drop streamOptions
# ---------------------------------------------------------------------------


def test_drop_stream_options_on_sdk_request() -> None:
    for request in (
        GenericChatRequest(stream_options=StreamOptions(is_include_usage=True)),
        CohereChatRequest(stream_options=StreamOptions(is_include_usage=True)),
    ):
        assert drop_unsupported_param(request, "streamOptions") is True
        assert request.stream_options is None
    assert drop_unsupported_param(GenericChatRequest(), "streamOptions") is False


def test_drop_stream_options_on_wire_dict() -> None:
    wire = {"isStream": True, "streamOptions": {"isIncludeUsage": True}}
    assert drop_unsupported_param(wire, "streamOptions") is True
    assert wire == {"isStream": True}


# ---------------------------------------------------------------------------
# Non-streaming parity: SDK usage objects map to the same summable shape
# ---------------------------------------------------------------------------


class _Attr(dict):
    """dict whose keys are also attributes, mimicking deserialized SDK objects."""

    def __getattr__(self, name: str) -> Any:
        return self.get(name)


def _sdk_usage_with_details() -> Usage:
    """The OpenAI-on-OCI usage object; unset detail fields deserialize as None."""
    return Usage(
        prompt_tokens=14,
        completion_tokens=2,
        total_tokens=16,
        prompt_tokens_details=PromptTokensDetails(cached_tokens=0),
        completion_tokens_details=CompletionTokensDetails(
            accepted_prediction_tokens=0, reasoning_tokens=0
        ),
    )


def _invoke_response(model_id: str, usage: Usage, text: str = "ok") -> _Attr:
    """A non-streaming SDK chat response for a GENERIC-format model."""
    return _Attr(
        status=200,
        request_id="req-1",
        headers=_Attr({"content-length": "10"}),
        data=_Attr(
            model_id=model_id,
            model_version="1.0",
            chat_response=_Attr(
                api_format="GENERIC",
                time_created="2026-01-01T00:00:00+00:00",
                usage=usage,
                choices=[
                    _Attr(
                        finish_reason="stop",
                        message=_Attr(
                            role="ASSISTANT",
                            content=[_Attr(text=text, type="TEXT")],
                            tool_calls=[],
                        ),
                    )
                ],
            ),
        ),
    )


def test_create_usage_metadata_drops_none_details_and_matches_wire_mapper() -> None:
    from_sdk = OCIUtils.create_usage_metadata(_sdk_usage_with_details())
    assert from_sdk == OPENAI_USAGE_METADATA
    assert from_sdk == OCIUtils.usage_metadata_from_dict(OPENAI_USAGE["usage"])


def test_create_usage_metadata_omits_all_none_details() -> None:
    usage = Usage(
        prompt_tokens=1,
        completion_tokens=1,
        total_tokens=2,
        completion_tokens_details=CompletionTokensDetails(),
    )
    assert OCIUtils.create_usage_metadata(usage) == {
        "input_tokens": 1,
        "output_tokens": 1,
        "total_tokens": 2,
    }


def test_usage_metadata_with_details_can_be_summed() -> None:
    usage = OCIUtils.create_usage_metadata(_sdk_usage_with_details())
    total = add_usage(usage, usage)
    assert total["input_tokens"] == 28
    assert total["output_tokens"] == 4
    assert total["total_tokens"] == 32
    assert total["input_token_details"] == {"cached_tokens": 0}
    assert total["output_token_details"] == {
        "accepted_prediction_tokens": 0,
        "reasoning_tokens": 0,
    }


@pytest.mark.requires("oci")
def test_usage_callback_totals_invokes_with_token_details() -> None:
    """UsageMetadataCallbackHandler sums per model via add_usage; it used to
    raise (and log) on the None details, silently keeping one call's numbers."""
    usage_cb = pytest.importorskip("langchain_core.callbacks.usage")
    llm = _llm(OPENAI)
    llm.client.chat.return_value = _invoke_response(OPENAI, _sdk_usage_with_details())

    with usage_cb.get_usage_metadata_callback() as cb:
        first = llm.invoke("hi")
        llm.invoke("hi")

    assert first.usage_metadata == OPENAI_USAGE_METADATA
    assert cb.usage_metadata[OPENAI]["input_tokens"] == 28
    assert cb.usage_metadata[OPENAI]["output_tokens"] == 4
    assert cb.usage_metadata[OPENAI]["total_tokens"] == 32
