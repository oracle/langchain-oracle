# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Unit tests for OpenAI streaming tool call fix.

Tests verify that streaming tool calls with missing/empty fields are handled
correctly and don't cause API errors when messages are sent back.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
