# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Unit tests for the OCI Responses API path in ChatOCIGenAI.

Covers:
- Payload construction (``_call_responses_api`` sends correct schema)
- Response parsing (``_process_responses_api_response`` extracts text correctly)
- SSE streaming (``_stream_responses_api`` yields proper chunks)
- No ``MagicMock`` in production response objects
"""

import json
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers — lightweight fakes
# ---------------------------------------------------------------------------


@pytest.fixture()
def chat_model() -> Any:
    """Build a ChatOCIGenAI with use_responses_api=True and a mock client."""
    from langchain_oci.chat_models.oci_generative_ai import ChatOCIGenAI

    return ChatOCIGenAI(
        model_id="xai.grok-4.20-multi-agent-0309",
        client=MagicMock(),
        use_responses_api=True,
    )


def _fake_chat_request(
    messages: List[Dict[str, str]],
    max_tokens: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
) -> SimpleNamespace:
    """Simulate the OCI SDK ChatDetails.chat_request attribute."""
    msg_objects = [
        SimpleNamespace(role=m["role"], content=m["content"]) for m in messages
    ]
    return SimpleNamespace(
        messages=msg_objects,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
    )


def _fake_request_details(chat_request: Any) -> SimpleNamespace:
    return SimpleNamespace(chat_request=chat_request)


# ===================================================================
# 1. Payload construction
# ===================================================================


@pytest.mark.requires("oci")
class TestPayloadConstruction:
    """Verify ``_call_responses_api`` builds the correct request body."""

    def test_uses_input_not_messages(self, chat_model: Any) -> None:
        """The payload must use ``input`` (Responses schema), not ``messages``."""
        chat_req = _fake_chat_request(
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
            ],
            max_tokens=200,
            top_k=40,
        )
        request = _fake_request_details(chat_req)

        captured_payload: Dict[str, Any] = {}

        def _fake_post(url: str, **kw: Any) -> Any:
            captured_payload.update(kw.get("json", {}))
            resp = MagicMock()
            resp.raise_for_status = lambda: None
            resp.json.return_value = {
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Hi"}],
                    }
                ],
                "usage": {"input_tokens": 5, "output_tokens": 2},
            }
            resp.headers = {"opc-request-id": "req-1", "content-length": "42"}
            return resp

        with patch.object(chat_model, "_get_oci_signer", return_value=None):
            with patch("requests.post", side_effect=_fake_post):
                chat_model._call_responses_api(request, stream=False)

        # Core assertions
        assert "input" in captured_payload, "Payload must use 'input'"
        assert "messages" not in captured_payload, "Must NOT contain 'messages'"
        assert "max_output_tokens" in captured_payload, (
            "max_tokens -> max_output_tokens"
        )
        assert "max_tokens" not in captured_payload, "Must NOT send max_tokens"
        assert "top_k" not in captured_payload, "Must NOT send top_k in body"
        assert "compartment_id" not in captured_payload, "Must NOT send compartment_id"
        assert captured_payload["max_output_tokens"] == 200
        assert len(captured_payload["input"]) == 2
        assert captured_payload["input"][0]["role"] == "system"
        assert captured_payload["input"][1]["content"] == "Hello"

    def test_compartment_in_header_not_body(self, chat_model: Any) -> None:
        """compartment_id must be in opc-compartment-id header, not body."""
        chat_req = _fake_chat_request(
            messages=[{"role": "user", "content": "hi"}],
        )
        request = _fake_request_details(chat_req)

        captured_headers: Dict[str, str] = {}
        captured_payload: Dict[str, Any] = {}

        def _fake_post(url: str, **kw: Any) -> Any:
            captured_headers.update(kw.get("headers", {}))
            captured_payload.update(kw.get("json", {}))
            resp = MagicMock()
            resp.raise_for_status = lambda: None
            resp.json.return_value = {
                "output": [],
                "usage": {},
            }
            resp.headers = {}
            return resp

        with patch.object(chat_model, "_get_oci_signer", return_value=None):
            with patch("requests.post", side_effect=_fake_post):
                chat_model._call_responses_api(request, stream=False)

        assert "compartment_id" not in captured_payload
        # The header should contain the compartment if set on the model
        if chat_model.compartment_id:
            assert "opc-compartment-id" in captured_headers


# ===================================================================
# 2. Response parsing
# ===================================================================


@pytest.mark.requires("oci")
class TestResponseParsing:
    """Verify response extraction and wrapper construction."""

    def test_extracts_text_from_output_content(self, chat_model: Any) -> None:
        """Text at output[].content[].text where type == 'output_text'."""
        data = {
            "model": "xai.grok-4.20-multi-agent-0309",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": "Hello, "},
                        {"type": "output_text", "text": "world!"},
                    ],
                }
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "status": "completed",
        }
        headers = {"opc-request-id": "test-req-123", "content-length": "200"}

        result = chat_model._process_responses_api_response(data, headers)

        assert (
            result.data.chat_response.choices[0].message.content[0].text
            == "Hello, world!"
        )
        assert result.data.model_id == "xai.grok-4.20-multi-agent-0309"
        assert result.request_id == "test-req-123"

    def test_extracts_usage_correctly(self, chat_model: Any) -> None:
        """Usage tokens from the Responses API format."""
        data = {
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "OK"}],
                }
            ],
            "usage": {"input_tokens": 15, "output_tokens": 3},
        }
        headers = {"content-length": "100"}

        result = chat_model._process_responses_api_response(data, headers)
        usage = result.data.chat_response.usage

        assert usage.input_tokens == 15
        assert usage.output_tokens == 3
        assert usage.total_tokens == 18

    def test_empty_output(self, chat_model: Any) -> None:
        """Empty output should produce empty text, not crash."""
        data = {"output": [], "usage": {}}
        headers = {}

        result = chat_model._process_responses_api_response(data, headers)
        assert result.data.chat_response.choices[0].message.content[0].text == ""

    def test_no_magicmock_in_response(self, chat_model: Any) -> None:
        """Response objects must NOT use unittest.mock.MagicMock."""
        data = {
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "test"}],
                }
            ],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
        headers = {"opc-request-id": "r1"}

        result = chat_model._process_responses_api_response(data, headers)

        from unittest.mock import MagicMock as _MM

        assert not isinstance(result, _MM)
        assert not isinstance(result.data, _MM)
        assert not isinstance(result.data.chat_response, _MM)
        assert not isinstance(result.data.chat_response.choices[0], _MM)
        assert not isinstance(result.data.chat_response.choices[0].message, _MM)

    def test_extract_text_skips_non_output_text(self) -> None:
        """_extract_responses_api_text skips non-output_text parts."""
        from langchain_oci.chat_models.oci_generative_ai import ChatOCIGenAI

        data = {
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": "part1"},
                        {"type": "reasoning", "text": "thinking..."},
                        {"type": "output_text", "text": "part2"},
                    ],
                }
            ],
        }
        assert ChatOCIGenAI._extract_responses_api_text(data) == "part1part2"

    def test_multiple_output_items(self, chat_model: Any) -> None:
        """Multiple items in output[] should all contribute text."""
        data = {
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "First. "}],
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Second."}],
                },
            ],
            "usage": {"input_tokens": 5, "output_tokens": 4},
        }
        headers = {}

        result = chat_model._process_responses_api_response(data, headers)
        assert (
            result.data.chat_response.choices[0].message.content[0].text
            == "First. Second."
        )


# ===================================================================
# 3. SSE Streaming
# ===================================================================


@pytest.mark.requires("oci")
class TestSSEStreaming:
    """Verify ``_stream_responses_api`` parses SSE events correctly."""

    @staticmethod
    def _build_sse_lines(events: List[Dict[str, Any]]) -> List[str]:
        """Build SSE-formatted lines from event dicts."""
        lines: List[str] = []
        for ev in events:
            if "event" in ev:
                lines.append(f"event: {ev['event']}")
            lines.append(f"data: {json.dumps(ev['data'])}")
            lines.append("")  # blank line separator
        return lines

    def test_stream_yields_text_deltas(self, chat_model: Any) -> None:
        """response.output_text.delta events should yield text chunks."""
        events = [
            {"event": "response.created", "data": {"type": "response.created"}},
            {"event": "response.output_text.delta", "data": {"delta": "Hello"}},
            {"event": "response.output_text.delta", "data": {"delta": " world"}},
            {
                "event": "response.completed",
                "data": {
                    "response": {
                        "id": "resp-123",
                        "status": "completed",
                        "usage": {"input_tokens": 5, "output_tokens": 2},
                    }
                },
            },
        ]
        sse_lines = self._build_sse_lines(events)

        fake_response = MagicMock()
        fake_response.iter_lines.return_value = iter(sse_lines)

        with patch.object(chat_model, "_prepare_request", return_value=MagicMock()):
            with patch.object(
                chat_model, "_call_responses_api", return_value=fake_response
            ):
                from langchain_core.messages import HumanMessage

                chunks = list(
                    chat_model._stream_responses_api([HumanMessage(content="hi")])
                )

        assert len(chunks) == 3
        assert chunks[0].message.content == "Hello"
        assert chunks[1].message.content == " world"
        assert chunks[2].message.content == ""
        assert chunks[2].generation_info["status"] == "completed"
        assert chunks[2].generation_info["response_id"] == "resp-123"

    def test_stream_handles_done_sentinel(self, chat_model: Any) -> None:
        """[DONE] data line should terminate the stream."""
        sse_lines = [
            "event: response.output_text.delta",
            'data: {"delta": "hi"}',
            "",
            "data: [DONE]",
            "",
            "event: response.output_text.delta",
            'data: {"delta": "should not appear"}',
            "",
        ]

        fake_response = MagicMock()
        fake_response.iter_lines.return_value = iter(sse_lines)

        with patch.object(chat_model, "_prepare_request", return_value=MagicMock()):
            with patch.object(
                chat_model, "_call_responses_api", return_value=fake_response
            ):
                from langchain_core.messages import HumanMessage

                chunks = list(
                    chat_model._stream_responses_api([HumanMessage(content="test")])
                )

        assert len(chunks) == 1
        assert chunks[0].message.content == "hi"

    def test_stream_ignores_unknown_events(self, chat_model: Any) -> None:
        """Non-delta/non-completed events should be silently skipped."""
        sse_lines = [
            "event: response.created",
            'data: {"type": "response.created"}',
            "",
            "event: response.in_progress",
            'data: {"type": "response.in_progress"}',
            "",
            "event: response.output_text.delta",
            'data: {"delta": "OK"}',
            "",
        ]

        fake_response = MagicMock()
        fake_response.iter_lines.return_value = iter(sse_lines)

        with patch.object(chat_model, "_prepare_request", return_value=MagicMock()):
            with patch.object(
                chat_model, "_call_responses_api", return_value=fake_response
            ):
                from langchain_core.messages import HumanMessage

                chunks = list(
                    chat_model._stream_responses_api([HumanMessage(content="test")])
                )

        assert len(chunks) == 1
        assert chunks[0].message.content == "OK"

    def test_stream_raw_wire_without_event_header_line(self, chat_model: Any) -> None:
        """The live endpoint emits SSE lines without preceding event: headers.

        Verify parser extracts type from data JSON directly.
        """
        sse_lines = [
            'data: {"sequence_number":0,"type":"response.created"}',
            "",
            'data: {"sequence_number":4,"type":"response.output_text.delta",'
            '"content_index":0,"delta":"OK"}',
            "",
        ]

        fake_response = MagicMock()
        fake_response.iter_lines.return_value = iter(sse_lines)

        with patch.object(chat_model, "_prepare_request", return_value=MagicMock()):
            with patch.object(
                chat_model, "_call_responses_api", return_value=fake_response
            ):
                from langchain_core.messages import HumanMessage

                chunks = list(
                    chat_model._stream_responses_api([HumanMessage(content="test")])
                )

        assert len(chunks) == 1
        assert chunks[0].message.content == "OK"


# ===================================================================
# 4. End-to-End invoke() and stream()
# ===================================================================


@pytest.mark.requires("oci")
class TestEndToEndPublicApi:
    """Verify end-to-end ``invoke()`` and ``stream()`` calls."""

    def test_invoke_end_to_end(self, chat_model: Any) -> None:
        """llm.invoke() should return AIMessage with correct content."""
        fake_res = MagicMock()
        fake_res.raise_for_status = lambda: None
        fake_res.json.return_value = {
            "model": "xai.grok-3",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "OK"}],
                }
            ],
            "usage": {"input_tokens": 5, "output_tokens": 1},
        }
        fake_res.headers = {"opc-request-id": "req-999", "content-length": "123"}

        with patch.object(chat_model, "_get_oci_signer", return_value=None):
            with patch("requests.post", return_value=fake_res):
                res = chat_model.invoke("Reply with exactly: OK")

        assert res.content == "OK"

    def test_stream_end_to_end(self, chat_model: Any) -> None:
        """llm.stream() should yield AIMessageChunks without errors."""
        sse_lines = [
            'data: {"sequence_number":0,"type":"response.created"}',
            "",
            'data: {"type":"response.output_text.delta","delta":"1 "}',
            "",
            'data: {"type":"response.output_text.delta","delta":"2 "}',
            "",
            'data: {"type":"response.output_text.delta","delta":"3"}',
            "",
        ]
        fake_res = MagicMock()
        fake_res.raise_for_status = lambda: None
        fake_res.iter_lines.return_value = iter(sse_lines)

        with patch.object(chat_model, "_get_oci_signer", return_value=None):
            with patch("requests.post", return_value=fake_res):
                chunks = list(chat_model.stream("Count: 1 2 3"))

        text_chunks = [c for c in chunks if c.content]
        assert len(text_chunks) == 3
        text = "".join(c.content for c in chunks)
        assert text == "1 2 3"
