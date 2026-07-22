# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Unit tests for standard content blocks / output_version="v1" (issue #260).

langchain-core >= 1.0 rewrites ``AIMessage.content`` into the standardized
list-of-blocks form when ``output_version="v1"`` is set. The provider request
builders must therefore accept v1-format message *input*: text blocks render,
while ``reasoning``/``tool_call``/``tool_use`` blocks are carried by
dedicated fields and are skipped, not fatal.
"""

from typing import Any, List
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langchain_oci.chat_models.oci_generative_ai import ChatOCIGenAI
from langchain_oci.common.utils import OCIUtils

V1_AI_MESSAGE = AIMessage(
    content=[
        {"type": "reasoning", "reasoning": "User wants weather; call the tool."},
        {"type": "text", "text": "Checking."},
        {
            "type": "tool_call",
            "id": "c1",
            "name": "get_weather",
            "args": {"city": "Rome"},
        },
    ],
    tool_calls=[
        {
            "name": "get_weather",
            "args": {"city": "Rome"},
            "id": "c1",
            "type": "tool_call",
        }
    ],
)

HISTORY: List[Any] = [
    HumanMessage("Weather in Rome?"),
    V1_AI_MESSAGE,
    ToolMessage("Sunny, 28C", tool_call_id="c1"),
]


def test_content_to_text_str() -> None:
    assert OCIUtils.content_to_text("plain") == "plain"


def test_content_to_text_v1_blocks() -> None:
    assert (
        OCIUtils.content_to_text(V1_AI_MESSAGE.content) == "Checking."
    )  # reasoning/tool_call skipped


def test_content_to_text_mixed_str_items() -> None:
    assert OCIUtils.content_to_text(["a", {"type": "text", "text": "b"}]) == "ab"


@pytest.mark.requires("oci")
def test_generic_provider_accepts_v1_history() -> None:
    """Meta/Generic path: v1 blocks in history must build a valid request."""
    llm = ChatOCIGenAI(
        model_id="meta.llama-3.3-70b-instruct",
        client=MagicMock(),
        compartment_id="c",
    )
    request = llm._prepare_request(HISTORY, stop=None, stream=False)
    # The reasoning/tool_call blocks are skipped; the text block survives.
    rendered = str(request.chat_request.messages)
    assert "Checking." in rendered
    assert "User wants weather; call the tool." not in rendered


@pytest.mark.requires("oci")
def test_cohere_provider_accepts_v1_history() -> None:
    """Cohere V1 path coerces list content to text for its string fields."""
    llm = ChatOCIGenAI(
        model_id="cohere.command-r-plus-08-2024",
        client=MagicMock(),
        compartment_id="c",
    )
    request = llm._prepare_request(
        [
            HumanMessage("hi"),
            AIMessage(content=[{"type": "text", "text": "hello"}]),
            HumanMessage("bye"),
        ],
        stop=None,
        stream=False,
    )
    history = request.chat_request.chat_history
    assert any(getattr(m, "message", None) == "hello" for m in history)


@pytest.mark.requires("oci")
def test_output_version_v1_rewrites_content(monkeypatch: pytest.MonkeyPatch) -> None:
    """End to end with a mocked client: invoke returns list-of-blocks content."""

    class MockResponseDict(dict):
        def __getattr__(self, val: str) -> Any:
            return self.get(val)

    client = MagicMock()
    llm = ChatOCIGenAI(
        model_id="meta.llama-3.3-70b-instruct",
        client=client,
        compartment_id="c",
        output_version="v1",
    )

    def mocked_chat(*args: Any, **kwargs: Any) -> Any:
        return MockResponseDict(
            {
                "status": 200,
                "data": MockResponseDict(
                    {
                        "chat_response": MockResponseDict(
                            {
                                "choices": [
                                    MockResponseDict(
                                        {
                                            "message": MockResponseDict(
                                                {
                                                    "content": [
                                                        MockResponseDict(
                                                            {"text": "Hello world"}
                                                        )
                                                    ],
                                                    "tool_calls": None,
                                                }
                                            ),
                                            "finish_reason": "stop",
                                        }
                                    )
                                ],
                                "time_created": "2026-01-01T00:00:00Z",
                            }
                        ),
                        "model_id": "meta.llama-3.3-70b-instruct",
                        "model_version": "1.0",
                    }
                ),
                "request_id": "r",
                "headers": MockResponseDict({"content-length": "1"}),
            }
        )

    monkeypatch.setattr(llm.client, "chat", mocked_chat)

    result = llm.invoke("hi")
    assert isinstance(result.content, list)
    assert result.content == [{"type": "text", "text": "Hello world"}]
    assert result.content_blocks == [{"type": "text", "text": "Hello world"}]
