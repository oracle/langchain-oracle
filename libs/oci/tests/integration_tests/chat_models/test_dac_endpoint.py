# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Integration tests for ChatOCIGenAI against an OCI Dedicated AI Cluster (DAC).

These tests verify that DAC fine-tunes which emit tool calls inline as
``<tool_call>{...}</tool_call>`` blocks (Hermes/Llama style) round-trip
through `GenericProvider`'s parsing — both for `invoke` (non-streaming) and
`stream` — so calling code never has to subclass `ChatOCIGenAI`.

Required env (skip otherwise):

    DAC_ENDPOINT_OCID    # ocid1.generativeaiendpoint.oc1.<region>.<...>
    DAC_COMPARTMENT_ID   # ocid1.compartment.oc1..<...>
    DAC_SERVICE_ENDPOINT # https://inference.generativeai.<region>.oci.oraclecloud.com
    OCI_CONFIG_PROFILE   # ~/.oci/config profile name (default: DEFAULT)
    OCI_AUTH_TYPE        # default: API_KEY
"""

from __future__ import annotations

import json
import os

import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from langchain_oci.chat_models import ChatOCIGenAI

REQUIRED_ENV = ("DAC_ENDPOINT_OCID", "DAC_COMPARTMENT_ID", "DAC_SERVICE_ENDPOINT")


def _missing_env() -> bool:
    return any(not os.environ.get(k) for k in REQUIRED_ENV)


pytestmark = [
    pytest.mark.requires("oci"),
    pytest.mark.skipif(
        _missing_env(),
        reason=f"DAC integration tests need {', '.join(REQUIRED_ENV)}",
    ),
]


def _make_llm(**kw):
    return ChatOCIGenAI(
        model_id=os.environ["DAC_ENDPOINT_OCID"],
        compartment_id=os.environ["DAC_COMPARTMENT_ID"],
        service_endpoint=os.environ["DAC_SERVICE_ENDPOINT"],
        auth_profile=os.environ.get("OCI_CONFIG_PROFILE", "DEFAULT"),
        auth_type=os.environ.get("OCI_AUTH_TYPE", "API_KEY"),
        **kw,
    )


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------


def add_numbers(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def multiply_numbers(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


def lookup_user(user_id: str, fields: list[str]) -> dict:
    """Look up a user by id, returning the requested fields."""
    return {"user_id": user_id, **{f: f"<{f}>" for f in fields}}


# ---------------------------------------------------------------------------
# Plain text — no tool calling involved
# ---------------------------------------------------------------------------


def test_invoke_plain_text() -> None:
    result = _make_llm().invoke("Reply with exactly: 'pong'")
    assert isinstance(result.content, str) and result.content
    assert result.tool_calls == []


def test_stream_plain_text_reassembles() -> None:
    chunks = list(_make_llm().stream("Count: one two three four five."))
    assert len(chunks) > 1, "expected more than one streamed chunk"
    text = "".join(c.content for c in chunks)
    assert text, "reassembled text should be non-empty"
    # Plain-text stream must never leak tool_call XML markers.
    assert "<tool_call>" not in text
    assert "</tool_call>" not in text
    assert not any(c.tool_call_chunks for c in chunks)


def test_stream_plain_text_no_xml_marker_leak_with_literal_lt() -> None:
    """Literal `<` characters in plain text must not be held back forever."""
    chunks = list(
        _make_llm().stream(
            "Print these comparison expressions one per line and nothing else: "
            "a < b, c < d, e < f"
        )
    )
    text = "".join(c.content for c in chunks)
    assert "<" in text, "expected the literal '<' to round-trip"
    assert "<tool_call>" not in text
    assert not any(c.tool_call_chunks for c in chunks)


def test_invoke_with_system_and_multi_turn() -> None:
    result = _make_llm().invoke(
        [
            SystemMessage(
                content="You are a terse cat. Answer with one short sentence."
            ),
            HumanMessage(content="What's 2+2?"),
            AIMessage(content="Four, mrow."),
            HumanMessage(content="And 3+3?"),
        ]
    )
    assert "6" in result.content or "six" in result.content.lower()


def test_invoke_respects_max_tokens() -> None:
    short = _make_llm(model_kwargs={"temperature": 0.0, "max_tokens": 20}).invoke(
        "Reply with the alphabet."
    )
    # 20-token cap → short response. Don't pin to exact length (tokenisation
    # varies); just assert it didn't run away with hundreds of characters.
    assert len(short.content) < 200


# ---------------------------------------------------------------------------
# Tool calling — invoke (non-streaming)
# ---------------------------------------------------------------------------


def test_invoke_single_tool_call_returns_structured() -> None:
    chat = _make_llm().bind_tools([add_numbers])
    result = chat.invoke("Use add_numbers to compute 17 plus 25.")

    # Inline <tool_call> XML must not bleed into the text.
    assert "<tool_call>" not in (result.content or "")
    assert len(result.tool_calls) == 1

    tc = result.tool_calls[0]
    assert tc["name"] == "add_numbers"
    assert tc["args"] == {"a": 17, "b": 25}
    assert tc["id"], "tool call must have a non-empty id for the agent loop"


def test_invoke_picks_correct_tool_from_two_options() -> None:
    chat = _make_llm().bind_tools([add_numbers, multiply_numbers])
    result = chat.invoke("Use the right tool to compute 7 times 8.")
    assert len(result.tool_calls) == 1
    tc = result.tool_calls[0]
    assert tc["name"] == "multiply_numbers"
    assert tc["args"] == {"a": 7, "b": 8}


def test_invoke_tool_with_nested_args() -> None:
    chat = _make_llm().bind_tools([lookup_user])
    result = chat.invoke(
        "Look up user 'u-42' and return their name and email "
        "using the lookup_user tool."
    )
    assert len(result.tool_calls) == 1
    tc = result.tool_calls[0]
    assert tc["name"] == "lookup_user"
    assert tc["args"]["user_id"] == "u-42"
    assert set(tc["args"]["fields"]) == {"name", "email"}


def test_invoke_multi_step_tool_orchestration() -> None:
    """Tool call → ToolMessage feedback → final natural-language answer."""
    chat = _make_llm().bind_tools([add_numbers])
    prompt = "Use add_numbers to compute 200 plus 50, then tell me the answer."

    turn1 = chat.invoke(prompt)
    assert len(turn1.tool_calls) == 1
    tc = turn1.tool_calls[0]

    tool_result = add_numbers(**tc["args"])
    turn2 = chat.invoke(
        [
            HumanMessage(content=prompt),
            turn1,
            ToolMessage(
                content=str(tool_result), name=tc["name"], tool_call_id=tc["id"]
            ),
        ]
    )
    assert turn2.tool_calls == [], "second turn should not call a tool again"
    assert "250" in turn2.content


# ---------------------------------------------------------------------------
# Tool calling — stream
# ---------------------------------------------------------------------------


def _collect_stream_tool_calls(chunks):
    return [tc for c in chunks if c.tool_call_chunks for tc in c.tool_call_chunks]


def test_stream_single_tool_call_returns_chunks() -> None:
    chat = _make_llm().bind_tools([add_numbers])
    chunks = list(chat.stream("Use add_numbers to compute 100 plus 1."))
    text = "".join(c.content for c in chunks)

    # The XML markers and JSON payload must never appear in the streamed text.
    assert "<tool_call>" not in text
    assert "</tool_call>" not in text

    tool_chunks = _collect_stream_tool_calls(chunks)
    assert len(tool_chunks) == 1
    tc = tool_chunks[0]
    assert tc["name"] == "add_numbers"
    assert tc["args"] is not None
    assert json.loads(tc["args"]) == {"a": 100, "b": 1}
    assert tc["id"]


def test_stream_picks_correct_tool_from_two_options() -> None:
    chat = _make_llm().bind_tools([add_numbers, multiply_numbers])
    chunks = list(chat.stream("Use the right tool to compute 7 times 8."))
    tool_chunks = _collect_stream_tool_calls(chunks)
    assert len(tool_chunks) == 1
    tc = tool_chunks[0]
    assert tc["name"] == "multiply_numbers"
    assert tc["args"] is not None
    assert json.loads(tc["args"]) == {"a": 7, "b": 8}


# ---------------------------------------------------------------------------
# Round-trip / robustness
# ---------------------------------------------------------------------------


def test_invoke_unicode_round_trip() -> None:
    target = "café — naïve — ✓ — 漢字"
    result = _make_llm().invoke(f"Reply with: '{target}' and nothing else.")
    for token in ("café", "naïve", "✓", "漢字"):
        assert token in result.content


def test_stream_unicode_round_trip() -> None:
    target = "café — naïve — ✓ — 漢字"
    chunks = list(_make_llm().stream(f"Reply with: '{target}' and nothing else."))
    text = "".join(c.content for c in chunks)
    for token in ("café", "naïve", "✓", "漢字"):
        assert token in text


def test_invoke_does_not_leak_tool_calling_variant_marker() -> None:
    """Belt-and-braces: even if the model emits ``<tool_calling>...`` instead of
    ``<tool_call>...``, the tags must not bleed into ``content``.

    See https://github.com/oracle/langchain-oracle/issues/207.
    """
    chat = _make_llm().bind_tools([add_numbers])
    result = chat.invoke("Use add_numbers to compute 41 plus 1.")
    content = result.content or ""
    assert "<tool_call>" not in content
    assert "<tool_calling>" not in content
    assert "</tool_call>" not in content
    assert "</tool_calling>" not in content


def test_chat_instance_reuse_across_stream_invoke_stream() -> None:
    """The provider's per-stream buffer must not leak across calls on the same chat."""
    shared = _make_llm().bind_tools([add_numbers])

    r1 = list(shared.stream("Use add_numbers for 1 plus 1."))
    tcs1 = _collect_stream_tool_calls(r1)
    assert len(tcs1) == 1
    assert json.loads(tcs1[0]["args"]) == {"a": 1, "b": 1}

    r2 = shared.invoke("Use add_numbers for 2 plus 2.")
    assert len(r2.tool_calls) == 1
    assert r2.tool_calls[0]["args"] == {"a": 2, "b": 2}

    r3 = list(shared.stream("Use add_numbers for 3 plus 3."))
    tcs3 = _collect_stream_tool_calls(r3)
    assert len(tcs3) == 1
    assert json.loads(tcs3[0]["args"]) == {"a": 3, "b": 3}
