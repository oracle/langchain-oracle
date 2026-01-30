# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Unit tests for Gemini parallel tool call flattening."""

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from langchain_oci.chat_models.providers.generic import GenericProvider


class TestFlattenParallelToolCalls:
    """Tests for GenericProvider._flatten_parallel_tool_calls."""

    def test_no_tool_calls_unchanged(self):
        """Messages without tool calls pass through unchanged."""
        messages = [
            SystemMessage(content="You are helpful."),
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there!"),
        ]
        result = GenericProvider._flatten_parallel_tool_calls(messages)
        assert len(result) == 3
        assert result[0].content == "You are helpful."
        assert result[1].content == "Hello"
        assert result[2].content == "Hi there!"

    def test_single_tool_call_unchanged(self):
        """A single tool call (no parallel) passes through unchanged."""
        messages = [
            HumanMessage(content="What's the weather?"),
            AIMessage(
                content="Let me check.",
                tool_calls=[
                    {"id": "call_1", "name": "get_weather", "args": {"city": "NYC"}}
                ],
            ),
            ToolMessage(content="Sunny, 72F", tool_call_id="call_1"),
        ]
        result = GenericProvider._flatten_parallel_tool_calls(messages)
        assert len(result) == 3
        assert isinstance(result[0], HumanMessage)
        assert isinstance(result[1], AIMessage)
        assert len(result[1].tool_calls) == 1
        assert isinstance(result[2], ToolMessage)

    def test_parallel_tool_calls_flattened(self):
        """Two parallel tool calls get split into 2 sequential AI->Tool pairs."""
        messages = [
            HumanMessage(content="Weather in NYC and LA?"),
            AIMessage(
                content="Let me check both.",
                tool_calls=[
                    {"id": "call_1", "name": "get_weather", "args": {"city": "NYC"}},
                    {"id": "call_2", "name": "get_weather", "args": {"city": "LA"}},
                ],
            ),
            ToolMessage(content="Sunny, 72F", tool_call_id="call_1"),
            ToolMessage(content="Cloudy, 65F", tool_call_id="call_2"),
        ]
        result = GenericProvider._flatten_parallel_tool_calls(messages)

        # HumanMessage + 2x (AIMessage, ToolMessage) = 5
        assert len(result) == 5

        # First: original HumanMessage
        assert isinstance(result[0], HumanMessage)

        # Second: AI with first tool call, keeps original content
        assert isinstance(result[1], AIMessage)
        assert result[1].content == "Let me check both."
        assert len(result[1].tool_calls) == 1
        assert result[1].tool_calls[0]["id"] == "call_1"

        # Third: matching ToolMessage
        assert isinstance(result[2], ToolMessage)
        assert result[2].tool_call_id == "call_1"
        assert result[2].content == "Sunny, 72F"

        # Fourth: AI with second tool call, placeholder content
        assert isinstance(result[3], AIMessage)
        assert result[3].content == "."
        assert len(result[3].tool_calls) == 1
        assert result[3].tool_calls[0]["id"] == "call_2"

        # Fifth: matching ToolMessage
        assert isinstance(result[4], ToolMessage)
        assert result[4].tool_call_id == "call_2"
        assert result[4].content == "Cloudy, 65F"

    def test_three_parallel_tool_calls(self):
        """Three parallel tool calls get split into 3 sequential pairs."""
        messages = [
            HumanMessage(content="Check three cities"),
            AIMessage(
                content="Checking all three.",
                tool_calls=[
                    {"id": "c1", "name": "get_weather", "args": {"city": "NYC"}},
                    {"id": "c2", "name": "get_weather", "args": {"city": "LA"}},
                    {"id": "c3", "name": "get_weather", "args": {"city": "CHI"}},
                ],
            ),
            ToolMessage(content="Sunny", tool_call_id="c1"),
            ToolMessage(content="Cloudy", tool_call_id="c2"),
            ToolMessage(content="Rainy", tool_call_id="c3"),
        ]
        result = GenericProvider._flatten_parallel_tool_calls(messages)

        # HumanMessage + 3x (AIMessage, ToolMessage) = 7
        assert len(result) == 7

        # First AI keeps original content
        assert result[1].content == "Checking all three."
        assert result[1].tool_calls[0]["id"] == "c1"  # type: ignore[attr-defined]

        # Second and third AI get placeholder
        assert result[3].content == "."
        assert result[3].tool_calls[0]["id"] == "c2"  # type: ignore[attr-defined]

        assert result[5].content == "."
        assert result[5].tool_calls[0]["id"] == "c3"  # type: ignore[attr-defined]

        # Tool messages match correctly
        assert result[2].content == "Sunny"
        assert result[4].content == "Cloudy"
        assert result[6].content == "Rainy"

    def test_empty_content_gets_placeholder(self):
        """AI message with empty content gets placeholder '.' for all splits."""
        messages = [
            HumanMessage(content="Do two things"),
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "c1", "name": "tool_a", "args": {}},
                    {"id": "c2", "name": "tool_b", "args": {}},
                ],
            ),
            ToolMessage(content="result_a", tool_call_id="c1"),
            ToolMessage(content="result_b", tool_call_id="c2"),
        ]
        result = GenericProvider._flatten_parallel_tool_calls(messages)

        # Both AI messages should get "." since original content is empty
        assert result[1].content == "."
        assert result[3].content == "."

    def test_mixed_parallel_and_single(self):
        """A conversation with both parallel and single tool calls."""
        messages = [
            HumanMessage(content="Step 1"),
            # Single tool call — should pass through
            AIMessage(
                content="One tool.",
                tool_calls=[{"id": "s1", "name": "single_tool", "args": {}}],
            ),
            ToolMessage(content="single result", tool_call_id="s1"),
            # Then parallel tool calls — should be flattened
            AIMessage(
                content="Two tools now.",
                tool_calls=[
                    {"id": "p1", "name": "tool_a", "args": {}},
                    {"id": "p2", "name": "tool_b", "args": {}},
                ],
            ),
            ToolMessage(content="result_a", tool_call_id="p1"),
            ToolMessage(content="result_b", tool_call_id="p2"),
            AIMessage(content="Done."),
        ]
        result = GenericProvider._flatten_parallel_tool_calls(messages)

        # Human + (AI,Tool) + (AI,Tool) + (AI,Tool) + AI = 8
        assert len(result) == 8
        # Single tool call unchanged
        assert result[1].tool_calls[0]["id"] == "s1"  # type: ignore[attr-defined]
        assert result[2].tool_call_id == "s1"  # type: ignore[attr-defined]
        # Parallel flattened
        assert result[3].tool_calls[0]["id"] == "p1"  # type: ignore[attr-defined]
        assert result[4].tool_call_id == "p1"  # type: ignore[attr-defined]
        assert result[5].tool_calls[0]["id"] == "p2"  # type: ignore[attr-defined]
        assert result[6].tool_call_id == "p2"  # type: ignore[attr-defined]
        # Final AI message
        assert result[7].content == "Done."

    def test_missing_tool_message_skipped(self):
        """If a ToolMessage is missing for a tool_call, no ToolMessage is added."""
        messages = [
            HumanMessage(content="Do two things"),
            AIMessage(
                content="Calling two tools.",
                tool_calls=[
                    {"id": "c1", "name": "tool_a", "args": {}},
                    {"id": "c2", "name": "tool_b", "args": {}},
                ],
            ),
            # Only one ToolMessage provided
            ToolMessage(content="result_a", tool_call_id="c1"),
        ]
        result = GenericProvider._flatten_parallel_tool_calls(messages)

        # HumanMessage + (AI, Tool) + AI (no matching Tool) = 4
        assert len(result) == 4
        assert result[1].tool_calls[0]["id"] == "c1"  # type: ignore[attr-defined]
        assert result[2].tool_call_id == "c1"  # type: ignore[attr-defined]
        assert result[3].tool_calls[0]["id"] == "c2"  # type: ignore[attr-defined]

    def test_empty_messages(self):
        """Empty message list returns empty list."""
        result = GenericProvider._flatten_parallel_tool_calls([])
        assert result == []


@pytest.mark.requires("oci")
class TestGeminiModelIdRouting:
    """Tests that model_id routing triggers flattening for Google models."""

    def test_google_model_triggers_flattening(self):
        """Google model ID triggers _flatten_parallel_tool_calls."""
        provider = GenericProvider()
        messages = [
            HumanMessage(content="Check two cities"),
            AIMessage(
                content="Checking.",
                tool_calls=[
                    {"id": "c1", "name": "weather", "args": {"city": "NYC"}},
                    {"id": "c2", "name": "weather", "args": {"city": "LA"}},
                ],
            ),
            ToolMessage(content="Sunny", tool_call_id="c1"),
            ToolMessage(content="Cloudy", tool_call_id="c2"),
        ]
        result = provider.messages_to_oci_params(
            messages, model_id="google.gemini-2.5-flash"
        )

        # After flattening: 2 sequential pairs = 4 AI/Tool OCI messages + 1 Human
        # Each pair is (AssistantMessage, ToolMessage)
        oci_msgs = result["messages"]
        assert len(oci_msgs) == 5

    def test_non_google_model_no_flattening(self):
        """Non-Google model ID does not trigger flattening."""
        provider = GenericProvider()
        messages = [
            HumanMessage(content="Check two cities"),
            AIMessage(
                content="Checking.",
                tool_calls=[
                    {"id": "c1", "name": "weather", "args": {"city": "NYC"}},
                    {"id": "c2", "name": "weather", "args": {"city": "LA"}},
                ],
            ),
            ToolMessage(content="Sunny", tool_call_id="c1"),
            ToolMessage(content="Cloudy", tool_call_id="c2"),
        ]
        result = provider.messages_to_oci_params(
            messages, model_id="meta.llama-3.3-70b-instruct"
        )

        # No flattening: 1 Human + 1 AI (with 2 tool_calls) + 2 Tool = 4
        oci_msgs = result["messages"]
        assert len(oci_msgs) == 4

    def test_no_model_id_no_flattening(self):
        """No model_id kwarg does not trigger flattening."""
        provider = GenericProvider()
        messages = [
            HumanMessage(content="Check two cities"),
            AIMessage(
                content="Checking.",
                tool_calls=[
                    {"id": "c1", "name": "weather", "args": {"city": "NYC"}},
                    {"id": "c2", "name": "weather", "args": {"city": "LA"}},
                ],
            ),
            ToolMessage(content="Sunny", tool_call_id="c1"),
            ToolMessage(content="Cloudy", tool_call_id="c2"),
        ]
        result = provider.messages_to_oci_params(messages)

        # No flattening: 4 messages
        oci_msgs = result["messages"]
        assert len(oci_msgs) == 4


@pytest.mark.requires("oci")
class TestModelIdPassthrough:
    """Tests that ChatOCIGenAI passes model_id to messages_to_oci_params."""

    def test_model_id_passed_to_provider(self):
        """ChatOCIGenAI passes model_id to the provider's messages_to_oci_params."""
        from langchain_oci.chat_models import ChatOCIGenAI

        oci_gen_ai_client = MagicMock()
        llm = ChatOCIGenAI(
            model_id="google.gemini-2.5-flash",
            client=oci_gen_ai_client,
        )

        # Mock the provider to capture call args
        original_provider = llm._provider
        llm._cached_provider_instance = MagicMock(wraps=original_provider)
        llm._cached_provider_instance.oci_chat_request = (
            original_provider.oci_chat_request
        )
        llm._cached_provider_instance.stop_sequence_key = (
            original_provider.stop_sequence_key
        )

        llm._prepare_request(
            [HumanMessage(content="Hello")],
            stop=None,
            stream=False,
        )

        # Verify messages_to_oci_params was called with model_id
        call_kwargs = llm._cached_provider_instance.messages_to_oci_params.call_args
        assert call_kwargs.kwargs.get("model_id") == "google.gemini-2.5-flash"
