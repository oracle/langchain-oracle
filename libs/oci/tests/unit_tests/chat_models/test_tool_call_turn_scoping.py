# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Tests for per-turn tool call counting and unexpected_tool_call recovery.

Bug 1: _should_allow_more_tool_calls counted ToolMessages across the entire
conversation, so a second user prompt would inherit the count from the first
turn and immediately get tool_choice=NONE.

Bug 2: When Gemini ignores tool_choice=NONE and the OCI API returns
finish_reason='unexpected_tool_call' with message=null, _generate produced
an empty AIMessage that silently terminated the agent loop.
"""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langchain_oci.chat_models.providers.generic import _should_allow_more_tool_calls

# ---------------------------------------------------------------------------
# Bug 1: _should_allow_more_tool_calls per-turn scoping
# ---------------------------------------------------------------------------


class TestToolCallTurnScoping:
    """Verify tool call counting resets at each HumanMessage boundary."""

    @staticmethod
    def _make_tool_turn(n_tool_calls: int) -> list:
        """Build an AI→Tool sequence with n_tool_calls pairs."""
        msgs: list = []
        for i in range(n_tool_calls):
            msgs.append(
                AIMessage(
                    content="",
                    tool_calls=[{"name": f"tool_{i}", "args": {}, "id": f"call_{i}"}],
                )
            )
            msgs.append(ToolMessage(content=f"result_{i}", tool_call_id=f"call_{i}"))
        return msgs

    def test_first_turn_under_limit(self):
        """First turn with fewer tool calls than max should allow more."""
        messages = [
            HumanMessage(content="Do stuff"),
            *self._make_tool_turn(3),
        ]
        assert _should_allow_more_tool_calls(messages, max_tool_calls=8) is True

    def test_first_turn_at_limit(self):
        """First turn hitting the limit should block."""
        messages = [
            HumanMessage(content="Do stuff"),
            *self._make_tool_turn(8),
        ]
        assert _should_allow_more_tool_calls(messages, max_tool_calls=8) is False

    def test_second_turn_resets_count(self):
        """A new HumanMessage resets the count — the core bug fix."""
        first_turn = [
            HumanMessage(content="First prompt"),
            *self._make_tool_turn(9),  # exceeds max=8
        ]
        second_turn = [
            HumanMessage(content="Second prompt"),
            # No tool calls yet in second turn
        ]
        messages = first_turn + second_turn
        # Even though 9 tool calls happened in turn 1, turn 2 has 0
        assert _should_allow_more_tool_calls(messages, max_tool_calls=8) is True

    def test_second_turn_with_own_tools_under_limit(self):
        """Second turn with its own tool calls under limit should allow more."""
        first_turn = [
            HumanMessage(content="First prompt"),
            *self._make_tool_turn(10),
        ]
        second_turn = [
            HumanMessage(content="Second prompt"),
            *self._make_tool_turn(3),
        ]
        messages = first_turn + second_turn
        assert _should_allow_more_tool_calls(messages, max_tool_calls=8) is True

    def test_second_turn_with_own_tools_at_limit(self):
        """Second turn exceeding its own limit should block."""
        first_turn = [
            HumanMessage(content="First prompt"),
            *self._make_tool_turn(10),
        ]
        second_turn = [
            HumanMessage(content="Second prompt"),
            *self._make_tool_turn(8),
        ]
        messages = first_turn + second_turn
        assert _should_allow_more_tool_calls(messages, max_tool_calls=8) is False

    def test_no_human_message_counts_all(self):
        """Edge case: no HumanMessage means count starts from index 0."""
        messages = self._make_tool_turn(5)
        assert _should_allow_more_tool_calls(messages, max_tool_calls=4) is False
        assert _should_allow_more_tool_calls(messages, max_tool_calls=8) is True

    def test_infinite_loop_detection_scoped_to_turn(self):
        """Infinite loop detection should only look at the current turn."""
        first_turn = [
            HumanMessage(content="First prompt"),
            AIMessage(
                content="",
                tool_calls=[{"name": "fetch", "args": {"x": 1}, "id": "c1"}],
            ),
            ToolMessage(content="ok", tool_call_id="c1"),
            AIMessage(
                content="",
                tool_calls=[{"name": "fetch", "args": {"x": 1}, "id": "c2"}],
            ),
            ToolMessage(content="ok", tool_call_id="c2"),
        ]
        second_turn = [
            HumanMessage(content="Second prompt"),
            # Same tool+args as turn 1, but first time in turn 2
            AIMessage(
                content="",
                tool_calls=[{"name": "fetch", "args": {"x": 1}, "id": "c3"}],
            ),
            ToolMessage(content="ok", tool_call_id="c3"),
        ]
        messages = first_turn + second_turn
        # Should allow — only 1 call in current turn, no loop yet
        assert _should_allow_more_tool_calls(messages, max_tool_calls=8) is True

    def test_three_turns(self):
        """Tool count resets correctly across three turns."""
        messages = [
            HumanMessage(content="Turn 1"),
            *self._make_tool_turn(7),
            HumanMessage(content="Turn 2"),
            *self._make_tool_turn(7),
            HumanMessage(content="Turn 3"),
            *self._make_tool_turn(2),
        ]
        # Turn 3 has only 2 tool calls
        assert _should_allow_more_tool_calls(messages, max_tool_calls=8) is True
