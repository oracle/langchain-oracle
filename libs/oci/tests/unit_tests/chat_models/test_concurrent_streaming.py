# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Regression tests for concurrent streaming on a shared chat-model instance.

A single ``ChatOCIGenAI`` (and therefore a single agent built on it) is
commonly shared across requests. The GenericProvider's incremental
``<tool_call>`` XML parser used to live on the provider instance, which is
cached per chat-model instance — so two concurrent streams fed one buffer:
one stream's ``reset`` wiped the other's partial block, and one stream could
drain (and execute) tool calls the model emitted in the *other* stream.

The fix gives each ``_stream``/``_astream`` call its own parser state via
``Provider.new_stream_state()``. These tests interleave two streams on one
shared instance and assert full isolation.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, Iterator, List, Optional
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import HumanMessage

from langchain_oci.chat_models import ChatOCIGenAI
from langchain_oci.chat_models.providers.generic import GenericProvider

# ---------------------------------------------------------------------------
# Provider level: per-stream state isolates interleaved streams
# ---------------------------------------------------------------------------


def _event(text: str) -> Dict[str, Any]:
    return {"message": {"content": [{"type": "TEXT", "text": text}]}}


def test_new_stream_state_returns_fresh_buffer_each_call() -> None:
    p = GenericProvider()
    assert p.new_stream_state() is not p.new_stream_state()


def test_interleaved_streams_do_not_wipe_each_others_partial_block() -> None:
    """Stream B starting mid-way through stream A must not destroy A's
    half-received <tool_call> block (previously reset_stream_state did)."""
    p = GenericProvider()

    state_a = p.new_stream_state()
    p.chat_stream_to_text(
        _event('<tool_call>{"name": "tool_a", "arguments": {"u": "A"'),
        stream_state=state_a,
    )
    # Stream B starts concurrently on the same shared provider.
    state_b = p.new_stream_state()
    p.chat_stream_to_text(_event("hello from B"), stream_state=state_b)
    # A's closing fragment arrives; A must recover its complete tool call.
    tail = p.chat_stream_to_text(_event("}}</tool_call>"), stream_state=state_a)
    a_calls = p.process_stream_tool_calls({}, {}, stream_state=state_a)

    assert tail == ""  # no raw XML leaks into A's text
    assert len(a_calls) == 1
    assert a_calls[0]["name"] == "tool_a"
    assert json.loads(a_calls[0]["args"] or "") == {"u": "A"}


def test_interleaved_streams_do_not_receive_each_others_tool_calls() -> None:
    """A tool call completed in stream A must never surface in stream B."""
    p = GenericProvider()

    state_a = p.new_stream_state()
    state_b = p.new_stream_state()
    p.chat_stream_to_text(
        _event('<tool_call>{"name": "tool_a", "arguments": {}}</tool_call>'),
        stream_state=state_a,
    )
    # B processes its next event before A drains.
    p.chat_stream_to_text(_event("hello from B"), stream_state=state_b)
    b_calls = p.process_stream_tool_calls({}, {}, stream_state=state_b)
    a_calls = p.process_stream_tool_calls({}, {}, stream_state=state_a)

    assert b_calls == []
    assert len(a_calls) == 1
    assert a_calls[0]["name"] == "tool_a"


# ---------------------------------------------------------------------------
# Model level (sync): two interleaved llm.stream() calls on one instance
# ---------------------------------------------------------------------------


def _stream_response(deltas: List[str]) -> MagicMock:
    events = [
        MagicMock(
            data=json.dumps(
                {
                    "index": 0,
                    "message": {
                        "role": "ASSISTANT",
                        "content": [{"type": "TEXT", "text": d}],
                    },
                }
            )
        )
        for d in deltas
    ]
    events.append(MagicMock(data=json.dumps({"finishReason": "stop"})))
    response = MagicMock()
    response.data.events.return_value = events
    return response


def _tool_call_deltas(user: str) -> List[str]:
    """Text deltas where the tool call is split across events."""
    return [
        f"{user} says: ",
        f'<tool_call>{{"name": "tool_{user}", ',
        f'"arguments": {{"user": "{user}"}}}}',
        "</tool_call>",
        f" done {user}",
    ]


def _drain_interleaved(gen_a: Iterator, gen_b: Iterator) -> tuple:
    """Alternate between two stream generators until both are exhausted."""
    merged: List[Optional[Any]] = [None, None]
    gens = [gen_a, gen_b]
    live = [True, True]
    while any(live):
        for i, gen in enumerate(gens):
            if not live[i]:
                continue
            try:
                chunk = next(gen)
            except StopIteration:
                live[i] = False
                continue
            merged[i] = chunk if merged[i] is None else merged[i] + chunk
    return merged[0], merged[1]


@pytest.mark.requires("oci")
def test_sync_interleaved_streams_on_shared_instance_are_isolated() -> None:
    oci_client = MagicMock()
    oci_client.chat.side_effect = [
        _stream_response(_tool_call_deltas("A")),
        _stream_response(_tool_call_deltas("B")),
    ]
    llm = ChatOCIGenAI(model_id="meta.llama-3.3-70b-instruct", client=oci_client)

    gen_a = llm.stream([HumanMessage(content="query A")])
    gen_b = llm.stream([HumanMessage(content="query B")])
    merged_a, merged_b = _drain_interleaved(gen_a, gen_b)

    for merged, user in ((merged_a, "A"), (merged_b, "B")):
        assert "tool_call" not in merged.content, merged.content
        assert merged.content == f"{user} says:  done {user}"
        assert len(merged.tool_call_chunks) == 1
        chunk = merged.tool_call_chunks[0]
        assert chunk["name"] == f"tool_{user}"
        assert json.loads(chunk["args"]) == {"user": user}


# ---------------------------------------------------------------------------
# Model level (async): concurrent astream() calls on one instance
# ---------------------------------------------------------------------------


def _fake_serialize(obj: Any) -> Any:
    """Minimal stand-in for BaseClient.sanitize_for_serialization."""
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


def _make_async_llm() -> ChatOCIGenAI:
    oci_client = MagicMock()
    oci_client.base_client.signer = MagicMock()
    oci_client.base_client.config = {}
    oci_client.base_client.sanitize_for_serialization = _fake_serialize
    return ChatOCIGenAI(
        model_id="meta.llama-3.3-70b-instruct",
        client=oci_client,
        service_endpoint="https://example.invalid",
    )


@pytest.mark.requires("oci")
async def test_async_concurrent_streams_on_shared_instance_are_isolated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reported production scenario: one shared agent, two users calling
    astream concurrently, both turns involving tool calls."""
    from langchain_oci.common.async_support import OCIAsyncClient

    def fake_chat_async(
        self: Any,
        compartment_id: str,
        chat_request_dict: Dict[str, Any],
        serving_mode_dict: Dict[str, Any],
        stream: bool,
    ) -> Any:
        # Route each request to its own scripted stream by user message text.
        text = chat_request_dict["messages"][-1]["content"][0]["text"]
        user = "A" if "A" in text else "B"

        async def gen() -> Any:
            for delta in _tool_call_deltas(user):
                # Force a real suspension so the two gathered streams
                # interleave event-by-event, as they do over the network.
                await asyncio.sleep(0)
                yield {
                    "message": {
                        "role": "ASSISTANT",
                        "content": [{"type": "TEXT", "text": delta}],
                    }
                }
            await asyncio.sleep(0)
            yield {"finishReason": "stop"}

        return gen()

    monkeypatch.setattr(OCIAsyncClient, "chat_async", fake_chat_async)
    llm = _make_async_llm()

    async def run(user: str) -> Any:
        merged = None
        async for chunk in llm.astream([HumanMessage(content=f"query {user}")]):
            merged = chunk if merged is None else merged + chunk
        return merged

    merged_a, merged_b = await asyncio.gather(run("A"), run("B"))

    for merged, user in ((merged_a, "A"), (merged_b, "B")):
        assert "tool_call" not in merged.content, merged.content
        assert merged.content == f"{user} says:  done {user}"
        assert len(merged.tool_call_chunks) == 1
        chunk = merged.tool_call_chunks[0]
        assert chunk["name"] == f"tool_{user}"
        assert json.loads(chunk["args"]) == {"user": user}
