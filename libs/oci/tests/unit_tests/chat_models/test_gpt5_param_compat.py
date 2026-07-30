# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Unit tests for reactive parameter-compatibility retries (GPT-5, issue-style).

Some models reject request parameters only at call time — e.g. legacy
``openai.gpt-5`` returns ``400 unsupported_value`` for non-default
``temperature``/``top_p``. These tests cover the structured-error
extraction, the request adjustment on both sync SDK objects and async
wire dicts, and the end-to-end retry on both paths.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import HumanMessage
from oci.exceptions import ServiceError

from langchain_oci.chat_models import ChatOCIGenAI
from langchain_oci.common.async_support import OCIAsyncClient, OCIAsyncRequestError
from langchain_oci.common.param_compat import (
    adjust_request_for_param_error,
    drop_unsupported_param,
    extract_unsupported_param,
)


class MockResponseDict(dict):
    def __getattr__(self, val):
        return self.get(val)


OPENAI_STYLE_ERROR = {"error": {"param": "temperature", "code": "unsupported_value"}}


def _service_error(deserialized_data=None, message=None):
    return ServiceError(
        400,
        "unsupported_value",
        {},
        message,
        deserialized_data=deserialized_data,
    )


# --------------- extraction ---------------


def test_extract_from_service_error_structured_body():
    """OpenAI-style bodies live in args[0], not .message (which is None)."""
    e = _service_error(deserialized_data=OPENAI_STYLE_ERROR)
    assert e.message is None
    assert extract_unsupported_param(e) == ("temperature", "unsupported_value")


def test_extract_from_service_error_none_message_no_typeerror():
    """A bare 400 with no body must yield (None, None), not raise."""
    e = _service_error()
    assert e.message is None
    assert extract_unsupported_param(e) == (None, None)


def test_extract_from_service_error_json_string_message():
    """OCI-standard bodies may carry the JSON in the message string."""
    body = '{"error": {"param": "topP", "code": "unsupported_value"}}'
    e = _service_error(message=body)
    assert extract_unsupported_param(e) == ("topP", "unsupported_value")


def test_extract_from_async_body_string():
    """The async path hands over the raw response body text."""
    body = '{"error": {"param": "temperature", "code": "unsupported_value"}}'
    assert extract_unsupported_param(body) == ("temperature", "unsupported_value")


def test_extract_from_garbage_is_none():
    assert extract_unsupported_param("Internal error") == (None, None)
    assert extract_unsupported_param(None) == (None, None)
    assert extract_unsupported_param({"error": "nope"}) == (None, None)


# --------------- request adjustment ---------------


def test_drop_param_on_sdk_object_with_wire_name():
    req = SimpleNamespace(top_p=0.9, temperature=0.3)
    assert drop_unsupported_param(req, "topP") is True
    assert req.top_p is None
    assert req.temperature == 0.3


def test_rename_max_tokens_on_sdk_object():
    req = SimpleNamespace(max_tokens=512, max_completion_tokens=None)
    assert drop_unsupported_param(req, "maxTokens") is True
    assert req.max_tokens is None
    assert req.max_completion_tokens == 512


def test_drop_param_on_wire_dict():
    req = {"temperature": 0.3, "topP": 0.9}
    assert drop_unsupported_param(req, "temperature") is True
    assert "temperature" not in req
    assert req["topP"] == 0.9


def test_rename_max_tokens_on_wire_dict():
    req = {"maxTokens": 512}
    assert drop_unsupported_param(req, "maxTokens") is True
    assert req == {"maxCompletionTokens": 512}


def test_adjust_returns_false_when_param_absent():
    """Unfixable errors must not claim success (caller re-raises the 400)."""
    assert (
        adjust_request_for_param_error(
            '{"error": {"param": "temperature", "code": "unsupported_value"}}',
            {"topP": 0.9},
        )
        is False
    )
    assert adjust_request_for_param_error("not json", {"temperature": 0.3}) is False


# --------------- sync end-to-end retry ---------------


def _mock_chat_response(text="ok"):
    return MockResponseDict(
        {
            "status": 200,
            "data": MockResponseDict(
                {
                    "chat_response": MockResponseDict(
                        {
                            "api_format": "GENERIC",
                            "choices": [
                                MockResponseDict(
                                    {
                                        "message": MockResponseDict(
                                            {
                                                "role": "ASSISTANT",
                                                "content": [
                                                    MockResponseDict(
                                                        {"text": text, "type": "TEXT"}
                                                    )
                                                ],
                                                "tool_calls": [],
                                            }
                                        ),
                                        "finish_reason": "stop",
                                    }
                                )
                            ],
                            "time_created": "2026-01-01T00:00:00+00:00",
                        }
                    ),
                    "model_id": "openai.gpt-5",
                    "model_version": "1.0",
                }
            ),
            "request_id": "req-1",
            "headers": MockResponseDict({"content-length": "10"}),
        }
    )


@pytest.mark.requires("oci")
def test_sync_invoke_retries_after_unsupported_value():
    """First call 400s on temperature; retry succeeds with it removed."""
    oci_client = MagicMock()
    llm = ChatOCIGenAI(
        model_id="openai.gpt-5",
        client=oci_client,
        model_kwargs={"temperature": 0.3, "max_tokens": 100},
    )

    seen_temperatures = []

    def chat(request):
        seen_temperatures.append(request.chat_request.temperature)
        if len(seen_temperatures) == 1:
            raise _service_error(deserialized_data=OPENAI_STYLE_ERROR)
        return _mock_chat_response()

    oci_client.chat.side_effect = chat

    with pytest.warns(UserWarning, match="temperature"):
        response = llm.invoke([HumanMessage(content="hi")])

    assert response.content == "ok"
    assert seen_temperatures == [0.3, None]


@pytest.mark.requires("oci")
def test_sync_invoke_reraises_unfixable_400():
    """400s without a fixable parameter propagate unchanged."""
    oci_client = MagicMock()
    llm = ChatOCIGenAI(
        model_id="openai.gpt-5",
        client=oci_client,
        model_kwargs={"temperature": 0.3},
    )
    oci_client.chat.side_effect = _service_error()

    with pytest.raises(ServiceError):
        llm.invoke([HumanMessage(content="hi")])
    assert oci_client.chat.call_count == 1


# --------------- async end-to-end retry ---------------


ASYNC_ERROR_BODY = '{"error": {"param": "temperature", "code": "unsupported_value"}}'


def _install_fake_chat_async(monkeypatch, calls):
    """chat_async that 400s the first call, then succeeds."""

    def fake_chat_async(
        self, compartment_id, chat_request_dict, serving_mode_dict, stream
    ):
        async def gen():
            calls.append(dict(chat_request_dict))
            if len(calls) == 1:
                raise OCIAsyncRequestError(400, ASYNC_ERROR_BODY)
            if stream:
                yield {
                    "message": {
                        "role": "ASSISTANT",
                        "content": [{"type": "TEXT", "text": "ok"}],
                    }
                }
                yield {"finishReason": "stop"}
            else:
                yield {
                    "chatResponse": {
                        "choices": [
                            {
                                "message": {
                                    "role": "ASSISTANT",
                                    "content": [{"type": "TEXT", "text": "ok"}],
                                },
                                "finishReason": "stop",
                            }
                        ]
                    },
                    "modelId": "openai.gpt-5",
                    "modelVersion": "1.0",
                }

        return gen()

    monkeypatch.setattr(OCIAsyncClient, "chat_async", fake_chat_async)


def _fake_serialize(obj):
    """Minimal stand-in for BaseClient.sanitize_for_serialization: SDK
    model objects -> camelCase wire dicts, skipping None fields."""
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


def _make_async_llm():
    oci_client = MagicMock()
    oci_client.base_client.signer = MagicMock()
    oci_client.base_client.config = {}
    oci_client.base_client.sanitize_for_serialization = _fake_serialize
    return ChatOCIGenAI(
        model_id="openai.gpt-5",
        client=oci_client,
        service_endpoint="https://example.invalid",
        model_kwargs={"temperature": 0.3},
    )


@pytest.mark.requires("oci")
async def test_async_invoke_retries_after_unsupported_value(monkeypatch):
    calls: list = []
    _install_fake_chat_async(monkeypatch, calls)
    llm = _make_async_llm()

    with pytest.warns(UserWarning, match="temperature"):
        response = await llm.ainvoke([HumanMessage(content="hi")])

    assert response.content == "ok"
    assert len(calls) == 2
    assert calls[0].get("temperature") == 0.3
    assert "temperature" not in calls[1]


@pytest.mark.requires("oci")
async def test_async_stream_retries_after_unsupported_value(monkeypatch):
    calls: list = []
    _install_fake_chat_async(monkeypatch, calls)
    llm = _make_async_llm()

    with pytest.warns(UserWarning, match="temperature"):
        chunks = [c async for c in llm.astream([HumanMessage(content="hi")])]

    assert any(c.content == "ok" for c in chunks)
    assert len(calls) == 2
    assert "temperature" not in calls[1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def test_extract_from_double_encoded_oci_envelope():
    """Live async wire shape: OCI envelope with the error JSON re-encoded
    inside the message string (observed against openai.gpt-5)."""
    body = (
        '{ "code": "400", "message": "{\\n  \\"error\\": {\\n    '
        '\\"message\\": \\"Unsupported value...\\",\\n    '
        '\\"type\\": \\"invalid_request_error\\",\\n    '
        '\\"param\\": \\"temperature\\",\\n    '
        '\\"code\\": \\"unsupported_value\\"\\n  }\\n}" }'
    )
    assert extract_unsupported_param(body) == ("temperature", "unsupported_value")
