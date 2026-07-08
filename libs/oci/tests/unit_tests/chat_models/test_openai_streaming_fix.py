# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Unit tests for OpenAI streaming tool call fix.

Tests verify that streaming tool calls with missing/empty fields are handled
correctly and don't cause API errors when messages are sent back.
"""

import pytest
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessageChunk,
    HumanMessage,
)

from langchain_oci.chat_models.providers import GenericProvider


def test_format_stream_tool_calls_with_missing_fields():
    """Test that missing fields in streaming tool calls are set to None,
    not empty strings.
    """
    provider = GenericProvider()

    # Simulate streaming chunks where some fields are missing
    tool_calls = [
        {"id": "call_123", "name": "ask_sbc"},  # Missing arguments
        {"id": "call_456"},  # Missing name and arguments
        {},  # All fields missing
    ]

    result = provider.format_stream_tool_calls(tool_calls)

    assert len(result) == 3

    # First tool call: has id and name, but arguments should be None
    assert result[0]["id"] == "call_123"
    assert result[0]["function"]["name"] == "ask_sbc"
    assert result[0]["function"]["arguments"] is None

    # Second tool call: has id, but name and arguments should be None
    assert result[1]["id"] == "call_456"
    assert result[1]["function"]["name"] is None
    assert result[1]["function"]["arguments"] is None

    # Third tool call: all fields should be None
    assert result[2]["id"] is None
    assert result[2]["function"]["name"] is None
    assert result[2]["function"]["arguments"] is None


def test_messages_to_oci_params_filters_invalid_tool_calls():
    """Test that AIMessages with empty tool call names/ids are filtered out."""
    provider = GenericProvider()

    messages = [
        HumanMessage(content="Test query"),
        AIMessage(
            content="",
            tool_calls=[
                # Valid tool call
                {
                    "id": "call_valid",
                    "name": "ask_sbc",
                    "args": {"query": "test", "version": "9.3.0"},
                    "type": "tool_call",
                },
                # Invalid tool call with empty name (from bad streaming merge)
                {
                    "id": "call_invalid",
                    "name": "",
                    "args": {},
                    "type": "tool_call",
                },
                # Invalid tool call with missing id
                {
                    "id": "",
                    "name": "ask_sbc",
                    "args": {"query": "test2"},
                    "type": "tool_call",
                },
            ],
        ),
    ]

    result = provider.messages_to_oci_params(messages)

    # Should have 2 messages (HumanMessage + AIMessage)
    assert len(result["messages"]) == 2

    # AIMessage should only have 1 valid tool call (the invalid ones filtered out)
    ai_message = result["messages"][1]
    assert len(ai_message.tool_calls) == 1
    assert ai_message.tool_calls[0].name == "ask_sbc"
    assert ai_message.tool_calls[0].id == "call_valid"


def test_process_stream_tool_calls_handles_none_values():
    """Test that None values in stream tool calls are handled correctly."""
    provider = GenericProvider()
    tool_call_ids: dict[int, str] = {}

    # Simulate fragmented streaming (gpt-oss pattern)
    # First chunk: has id and name, no arguments yet
    event_data_1 = {
        "message": {
            "toolCalls": [{"id": "call_123", "name": "ask_sbc", "arguments": ""}]
        }
    }

    chunks_1 = provider.process_stream_tool_calls(event_data_1, tool_call_ids)
    assert len(chunks_1) == 1
    assert chunks_1[0]["id"] == "call_123"
    assert chunks_1[0]["name"] == "ask_sbc"
    # Arguments is None (from our fix), not empty string
    assert chunks_1[0]["args"] is None

    # Second chunk: no id or name, just arguments
    event_data_2 = {
        "message": {
            "toolCalls": [{"arguments": '{"query": "test", "version": "9.3.0"}'}]
        }
    }

    chunks_2 = provider.process_stream_tool_calls(event_data_2, tool_call_ids)
    assert len(chunks_2) == 1
    # Should reuse the ID from the first chunk
    assert chunks_2[0]["id"] == "call_123"
    assert chunks_2[0]["name"] is None
    assert chunks_2[0]["args"] == '{"query": "test", "version": "9.3.0"}'


def test_process_stream_tool_calls_parallel_gpt_pattern():
    """Parallel tool calls streamed sequentially at position 0 (issue #253).

    GPT models open each tool call with a chunk carrying id/name and empty
    arguments, then stream id-less argument fragments — all at toolCalls[0].
    Fragments arriving after a second call opens must attach to that call,
    not to the first one.
    """
    provider = GenericProvider()
    tool_call_ids: dict[int, str] = {}

    def event(tool_call: dict) -> dict:
        return {"message": {"toolCalls": [{"type": "FUNCTION", **tool_call}]}}

    events = [
        event({"id": "call_A", "name": "market_research_tool", "arguments": ""}),
        event({"arguments": '{"idea": "app",'}),
        event({"arguments": ' "marketSegment": "smb"}'}),
        event({"id": "call_B", "name": "customer_signal_tool", "arguments": ""}),
        event({"arguments": '{"idea": "app",'}),
        event({"arguments": ' "audience": "devs"}'}),
        event({"id": "call_C", "name": "roadmap_risk_tool", "arguments": ""}),
        event({"arguments": '{"productArea": "auth",'}),
        event({"arguments": ' "riskFocus": "churn"}'}),
    ]

    chunks = []
    for e in events:
        chunks.extend(provider.process_stream_tool_calls(e, tool_call_ids))

    # Each call's argument fragments must carry that call's logical index.
    assert [(c["id"], c["index"]) for c in chunks] == [
        ("call_A", 0),
        ("call_A", 0),
        ("call_A", 0),
        ("call_B", 1),
        ("call_B", 1),
        ("call_B", 1),
        ("call_C", 2),
        ("call_C", 2),
        ("call_C", 2),
    ]

    # Merge the chunks the way LangChain does during streaming and verify
    # each tool call reconstructs with its own complete arguments.
    merged: BaseMessageChunk = AIMessageChunk(content="")
    for c in chunks:
        merged = merged + AIMessageChunk(content="", tool_call_chunks=[c])

    assert isinstance(merged, AIMessageChunk)
    assert len(merged.tool_calls) == 3
    assert merged.tool_calls[0]["name"] == "market_research_tool"
    assert merged.tool_calls[0]["args"] == {"idea": "app", "marketSegment": "smb"}
    assert merged.tool_calls[1]["name"] == "customer_signal_tool"
    assert merged.tool_calls[1]["args"] == {"idea": "app", "audience": "devs"}
    assert merged.tool_calls[2]["name"] == "roadmap_risk_tool"
    assert merged.tool_calls[2]["args"] == {"productArea": "auth", "riskFocus": "churn"}


def test_process_stream_tool_calls_resent_id_keeps_logical_index():
    """A re-announced id keeps its logical index even at a new position.

    If a provider re-sends an id-bearing chunk for an already-open call at
    a position occupied by a different call, the call must keep its
    original logical index — otherwise its arguments split across two
    tool calls. (Hardening ported from the approach in PR #252.)
    """
    provider = GenericProvider()
    tool_call_ids: dict[int, str] = {}

    def event(tool_call: dict) -> dict:
        return {"message": {"toolCalls": [tool_call]}}

    chunks = []
    for e in [
        event({"id": "call_A", "name": "a", "arguments": ""}),
        event({"id": "call_B", "name": "b", "arguments": ""}),  # Grok branch -> 1
        event({"id": "call_B", "arguments": '{"y": 2}'}),  # re-send at pos 0
        event({"id": "call_A", "arguments": '{"x": 1}'}),  # re-send at pos 0
    ]:
        chunks.extend(provider.process_stream_tool_calls(e, tool_call_ids))

    assert [(c["id"], c["index"]) for c in chunks] == [
        ("call_A", 0),
        ("call_B", 1),
        ("call_B", 1),
        ("call_A", 0),
    ]


def test_process_stream_tool_calls_position_map_resets_between_streams():
    """reset_stream_state clears the position map so a new stream starts clean."""
    provider = GenericProvider()
    tool_call_ids: dict[int, str] = {}

    provider.process_stream_tool_calls(
        {"message": {"toolCalls": [{"id": "call_A", "name": "t", "arguments": ""}]}},
        tool_call_ids,
    )
    provider.process_stream_tool_calls(
        {"message": {"toolCalls": [{"id": "call_B", "name": "u", "arguments": ""}]}},
        tool_call_ids,
    )
    assert provider._active_tool_call_indices == {0: 1}

    provider.reset_stream_state()
    assert provider._active_tool_call_indices == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
