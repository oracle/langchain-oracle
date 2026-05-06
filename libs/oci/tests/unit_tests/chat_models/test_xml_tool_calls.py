# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Unit tests for the inline ``<tool_call>...</tool_call>`` parser used by
GenericProvider when talking to Hermes/Llama-style fine-tunes (e.g. served
through OCI Dedicated AI Cluster endpoints) that return tool calls inside the
assistant message text instead of in the structured ``tool_calls`` field.
"""

from __future__ import annotations

import json

import pytest

from langchain_oci.chat_models.providers.generic import GenericProvider
from langchain_oci.common.xml_tool_call_parser import (
    XmlStreamBuffer,
    extract_xml_tool_calls,
    parse_xml_tool_call_payload,
    safe_emit_split,
)

# ---------------------------------------------------------------------------
# parse_xml_tool_call_payload
# ---------------------------------------------------------------------------


def test_parse_payload_happy_path() -> None:
    payload = '{"name": "add", "arguments": {"a": 1, "b": 2}}'
    parsed = parse_xml_tool_call_payload(payload)
    assert parsed is not None
    assert parsed["name"] == "add"
    assert json.loads(parsed["arguments"]) == {"a": 1, "b": 2}


def test_parse_payload_string_arguments_preserved() -> None:
    """Some models double-encode args; we leave the string for downstream parsers."""
    payload = '{"name": "add", "arguments": "{\\"a\\": 1}"}'
    parsed = parse_xml_tool_call_payload(payload)
    assert parsed is not None
    assert parsed["arguments"] == '{"a": 1}'


def test_parse_payload_missing_arguments_defaults_to_empty_object() -> None:
    parsed = parse_xml_tool_call_payload('{"name": "ping"}')
    assert parsed == {"name": "ping", "arguments": "{}"}


@pytest.mark.parametrize(
    "payload",
    [
        "not json at all",
        '{"arguments": {}}',  # missing name
        '{"name": ""}',  # empty name
        '{"name": 123}',  # name not a string
        "[1, 2, 3]",  # not an object
    ],
)
def test_parse_payload_returns_none_on_invalid(payload: str) -> None:
    assert parse_xml_tool_call_payload(payload) is None


# ---------------------------------------------------------------------------
# extract_xml_tool_calls
# ---------------------------------------------------------------------------


def test_extract_no_markers_returns_text_unchanged() -> None:
    text = "Just a normal response with < and > characters."
    cleaned, calls = extract_xml_tool_calls(text)
    assert cleaned == text
    assert calls == []


def test_extract_single_block() -> None:
    text = (
        "Sure, calling the tool now.\n"
        '<tool_call>{"name": "add", "arguments": {"a": 1, "b": 2}}</tool_call>'
    )
    cleaned, calls = extract_xml_tool_calls(text)
    assert cleaned == "Sure, calling the tool now."
    assert len(calls) == 1
    assert calls[0]["name"] == "add"
    assert json.loads(calls[0]["arguments"]) == {"a": 1, "b": 2}
    assert calls[0]["id"]  # uuid populated


def test_extract_multiple_blocks_preserves_order_and_strips_text() -> None:
    text = (
        "Step 1.\n"
        '<tool_call>{"name": "first", "arguments": {"x": 1}}</tool_call>\n'
        "Step 2.\n"
        '<tool_call>{"name": "second", "arguments": {"y": 2}}</tool_call>\n'
        "All done."
    )
    cleaned, calls = extract_xml_tool_calls(text)
    assert "tool_call" not in cleaned
    assert "Step 1." in cleaned and "Step 2." in cleaned and "All done." in cleaned
    assert [c["name"] for c in calls] == ["first", "second"]


def test_extract_malformed_block_left_in_text() -> None:
    text = (
        "<tool_call>not valid json</tool_call>"
        ' <tool_call>{"name": "ok", "arguments": {}}</tool_call>'
    )
    cleaned, calls = extract_xml_tool_calls(text)
    assert "<tool_call>not valid json</tool_call>" in cleaned
    assert len(calls) == 1
    assert calls[0]["name"] == "ok"


def test_extract_handles_whitespace_around_payload() -> None:
    text = '<tool_call>\n  {"name": "ws", "arguments": {"a": 1}}\n  </tool_call>'
    cleaned, calls = extract_xml_tool_calls(text)
    assert cleaned == ""
    assert len(calls) == 1
    assert calls[0]["name"] == "ws"


def test_extract_supports_tool_calling_variant() -> None:
    """Qwen3 also emits ``<tool_calling>...</tool_calling>``; both must work.

    See https://github.com/oracle/langchain-oracle/issues/207.
    """
    text = (
        "Calling now.\n"
        '<tool_calling>{"name": "ping", "arguments": {"x": 1}}</tool_calling>'
    )
    cleaned, calls = extract_xml_tool_calls(text)
    assert cleaned == "Calling now."
    assert len(calls) == 1
    assert calls[0]["name"] == "ping"
    assert json.loads(calls[0]["arguments"]) == {"x": 1}


def test_extract_mixed_tag_variants_in_one_response() -> None:
    """One model turn occasionally mixes both tags; both must extract."""
    text = (
        '<tool_call>{"name": "first", "arguments": {"a": 1}}</tool_call>'
        " then "
        '<tool_calling>{"name": "second", "arguments": {"b": 2}}</tool_calling>'
    )
    cleaned, calls = extract_xml_tool_calls(text)
    assert "tool_call" not in cleaned
    assert "tool_calling" not in cleaned
    assert [c["name"] for c in calls] == ["first", "second"]


def test_extract_does_not_match_mismatched_open_close() -> None:
    """``<tool_call>...</tool_calling>`` is malformed — must NOT be parsed."""
    text = '<tool_call>{"name": "x", "arguments": {}}</tool_calling>'
    cleaned, calls = extract_xml_tool_calls(text)
    assert calls == []
    # The text comes back unchanged so the caller can see what happened.
    assert "<tool_call>" in cleaned
    assert "</tool_calling>" in cleaned


# ---------------------------------------------------------------------------
# safe_emit_split
# ---------------------------------------------------------------------------


def test_safe_emit_no_lt_emits_everything() -> None:
    safe, hold = safe_emit_split("hello world")
    assert safe == "hello world"
    assert hold == ""


def test_safe_emit_unrelated_lt_emits_everything() -> None:
    safe, hold = safe_emit_split("use < to compare")
    assert safe == "use < to compare"
    assert hold == ""


@pytest.mark.parametrize(
    "tail",
    [
        # <tool_call> partials and full opener
        "<",
        "<t",
        "<too",
        "<tool",
        "<tool_call",
        "<tool_call>",
        "<tool_call>{partial",
        # <tool_calling> partials and full opener (extra disambiguation chars)
        "<tool_",
        "<tool_c",
        "<tool_cal",
        "<tool_calli",
        "<tool_calling",
        "<tool_calling>",
        "<tool_calling>{partial",
    ],
)
def test_safe_emit_holds_back_partial_or_open_tag(tail: str) -> None:
    safe, hold = safe_emit_split("text " + tail)
    assert safe == "text "
    assert hold == tail


# ---------------------------------------------------------------------------
# XmlStreamBuffer (direct unit tests of the lifted class)
# ---------------------------------------------------------------------------


def test_xml_stream_buffer_plain_text_passes_through() -> None:
    buf = XmlStreamBuffer()
    assert buf.feed("hello ") == "hello "
    assert buf.feed("world") == "world"
    assert buf.drain_completed() == []
    assert buf.flush() == ""


def test_xml_stream_buffer_holds_back_partial_opener() -> None:
    buf = XmlStreamBuffer()
    safe = buf.feed("see <")
    assert safe == "see "
    safe = buf.feed("nope")
    # '<n' is no longer a possible opener — held-back '<' is released.
    assert safe == "<nope"


def test_xml_stream_buffer_drains_completed_block() -> None:
    buf = XmlStreamBuffer()
    safe = buf.feed("Calling: ")
    safe += buf.feed('<tool_call>{"name": "add", "arguments": {"a": 1}}</tool_call>')
    safe += buf.feed(" done")
    completed = buf.drain_completed()
    assert "<tool_call>" not in safe
    assert safe.startswith("Calling: ")
    assert safe.endswith(" done")
    assert len(completed) == 1
    assert completed[0]["name"] == "add"
    assert json.loads(completed[0]["arguments"]) == {"a": 1}


def test_xml_stream_buffer_flush_releases_unclosed_block() -> None:
    buf = XmlStreamBuffer()
    buf.feed('<tool_call>{"name": "incompl')
    flushed = buf.flush()
    assert flushed.startswith("<tool_call>")
    # After flush the buffer is reset.
    assert buf.feed("fresh") == "fresh"


def test_xml_stream_buffer_reset_clears_state() -> None:
    buf = XmlStreamBuffer()
    buf.feed("leftover <tool_call>{")
    buf.reset()
    assert buf.feed("fresh") == "fresh"
    assert buf.drain_completed() == []
    assert buf.flush() == ""


def test_xml_stream_buffer_drain_completed_is_idempotent() -> None:
    buf = XmlStreamBuffer()
    buf.feed('<tool_call>{"name": "ping", "arguments": {}}</tool_call>')
    first = buf.drain_completed()
    second = buf.drain_completed()
    assert len(first) == 1
    assert second == []


# ---------------------------------------------------------------------------
# GenericProvider.{chat_stream_to_text, process_stream_tool_calls}
# ---------------------------------------------------------------------------


def _stream_event(text: str) -> dict:
    """Wrap a text delta in the OCI streaming event shape GenericProvider expects."""
    return {"message": {"content": [{"type": "TEXT", "text": text}]}}


def test_stream_plain_text_passes_through() -> None:
    p = GenericProvider()
    p.reset_stream_state()
    out = [
        p.chat_stream_to_text(_stream_event("Hello, ")),
        p.chat_stream_to_text(_stream_event("world!")),
    ]
    assert "".join(out) == "Hello, world!"
    assert p.flush_stream_state() == ""


def test_stream_tool_call_split_across_chunks_emits_chunk_after_close() -> None:
    """The realistic scenario: <tool_call>...</tool_call> arrives across many events."""
    p = GenericProvider()
    p.reset_stream_state()
    deltas = [
        "Calling tool: ",
        "<tool_",
        'call>{"name": ',
        '"add", "argum',
        'ents": {"a": 1, ',
        '"b": 2}}</to',
        "ol_call>",
        " all done",
    ]
    text_buf = ""
    tool_calls = []
    tool_call_ids: dict = {}
    for d in deltas:
        text_buf += p.chat_stream_to_text(_stream_event(d))
        # process_stream_tool_calls drains XML tool calls; pass an empty
        # event so the OCI-native code path is a no-op.
        tool_calls.extend(p.process_stream_tool_calls({}, tool_call_ids))
    text_buf += p.flush_stream_state()

    # Text never contains the marker characters — they were buffered.
    assert "<tool_call>" not in text_buf
    assert "</tool_call>" not in text_buf
    assert text_buf.startswith("Calling tool: ")
    assert text_buf.endswith(" all done")

    assert len(tool_calls) == 1
    chunk = tool_calls[0]
    assert chunk["name"] == "add"
    args = chunk["args"]
    assert args is not None
    assert json.loads(args) == {"a": 1, "b": 2}
    assert chunk["id"]


def test_stream_partial_open_tag_then_unrelated_text_releases_held_back() -> None:
    """If '<' turns out not to be a <tool_call> opener, we must release the buffer."""
    p = GenericProvider()
    p.reset_stream_state()

    held = p.chat_stream_to_text(_stream_event("see <"))
    # Held back: "<" alone could still be a partial opener.
    assert held == "see "

    released = p.chat_stream_to_text(_stream_event("nope, just less-than"))
    # Once we see '<n', it's no longer a possible <tool_call> prefix.
    assert released == "<nope, just less-than"
    assert p.flush_stream_state() == ""


def test_stream_unclosed_tool_call_flushed_at_end_of_stream() -> None:
    """If a stream ends mid-block, the buffered text must surface, not vanish."""
    p = GenericProvider()
    p.reset_stream_state()
    p.chat_stream_to_text(_stream_event('<tool_call>{"name": "incompl'))
    flushed = p.flush_stream_state()
    assert flushed.startswith("<tool_call>")


def test_reset_stream_state_clears_between_streams() -> None:
    p = GenericProvider()
    p.chat_stream_to_text(_stream_event("leftover <tool_call>{"))
    p.reset_stream_state()
    out = p.chat_stream_to_text(_stream_event("fresh"))
    assert out == "fresh"
    assert p.flush_stream_state() == ""


def test_stream_tool_calling_variant_split_across_chunks() -> None:
    """The longer ``<tool_calling>`` opener must also stream correctly."""
    p = GenericProvider()
    p.reset_stream_state()
    deltas = [
        "Doing it: ",
        "<tool_call",
        'ing>{"name":',
        ' "ping", "arguments": {"x": 1}}',
        "</tool_calling>",
        " ok",
    ]
    text_buf = ""
    tool_calls: list = []
    tool_call_ids: dict = {}
    for d in deltas:
        text_buf += p.chat_stream_to_text(_stream_event(d))
        tool_calls.extend(p.process_stream_tool_calls({}, tool_call_ids))
    text_buf += p.flush_stream_state()

    assert "<tool_calling>" not in text_buf
    assert "</tool_calling>" not in text_buf
    assert text_buf.startswith("Doing it: ")
    assert text_buf.endswith(" ok")

    assert len(tool_calls) == 1
    assert tool_calls[0]["name"] == "ping"
    args = tool_calls[0]["args"]
    assert args is not None
    assert json.loads(args) == {"x": 1}


# ---------------------------------------------------------------------------
# GenericProvider.chat_tool_calls fallback to text-parsing
# ---------------------------------------------------------------------------


class _FakePart:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeMessage:
    def __init__(self, text: str, structured_tool_calls: list | None = None) -> None:
        self.content = [_FakePart(text)] if text else []
        self.tool_calls = structured_tool_calls or []


class _FakeChoice:
    def __init__(self, message: _FakeMessage) -> None:
        self.message = message


class _FakeChatResponse:
    def __init__(self, choices: list) -> None:
        self.choices = choices


class _FakeData:
    def __init__(self, chat_response: _FakeChatResponse) -> None:
        self.chat_response = chat_response


class _FakeResponse:
    def __init__(self, data: _FakeData) -> None:
        self.data = data


def _response_with_text(text: str, structured: list | None = None) -> _FakeResponse:
    msg = _FakeMessage(text, structured)
    return _FakeResponse(_FakeData(_FakeChatResponse([_FakeChoice(msg)])))


def test_chat_tool_calls_prefers_native_field() -> None:
    p = GenericProvider()
    sentinel = object()
    resp = _response_with_text(
        '<tool_call>{"name": "x", "arguments": {}}</tool_call>',
        structured=[sentinel],
    )
    assert p.chat_tool_calls(resp) == [sentinel]


def test_chat_tool_calls_falls_back_to_xml_when_native_empty() -> None:
    p = GenericProvider()
    resp = _response_with_text(
        '<tool_call>{"name": "add", "arguments": {"a": 1, "b": 2}}</tool_call>'
    )
    calls = p.chat_tool_calls(resp)
    assert len(calls) == 1
    assert calls[0].name == "add"
    assert json.loads(calls[0].arguments) == {"a": 1, "b": 2}


def test_chat_response_to_text_strips_xml_blocks() -> None:
    p = GenericProvider()
    resp = _response_with_text(
        'Here it is: <tool_call>{"name": "x", "arguments": {}}</tool_call>'
    )
    assert p.chat_response_to_text(resp) == "Here it is:"


def test_chat_response_to_text_from_dict_strips_xml_blocks() -> None:
    p = GenericProvider()
    text = '<tool_call>{"name": "x", "arguments": {}}</tool_call>'
    response_data = {
        "chatResponse": {
            "choices": [{"message": {"content": [{"type": "TEXT", "text": text}]}}]
        }
    }
    assert p.chat_response_to_text_from_dict(response_data) == ""
