# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Inline ``<tool_call>...</tool_call>`` parser for Hermes-style fine-tunes.

Some fine-tuned chat models hosted on OCI Dedicated AI Cluster (DAC) endpoints
follow the Hermes (NousResearch) function-calling convention and emit tool
calls inline as either ``<tool_call>{...}</tool_call>`` or
``<tool_calling>{...}</tool_calling>`` blocks inside the assistant message
text — instead of populating the structured ``tool_calls`` field. Qwen3 in
particular has been observed emitting either form (see issue #207).

This module owns:

* The pure helpers (:func:`parse_xml_tool_call_payload`,
  :func:`extract_xml_tool_calls`, :func:`safe_emit_split`,
  :class:`XmlToolCall`) used in the non-streaming response path.
* :class:`XmlStreamBuffer`, the stateful class that drives the incremental
  streaming parse — append a delta, drain any complete blocks, return only
  the prefix that cannot still be the start of an opener.

The parsing logic itself is provider-agnostic — providers wire it in (today
only ``GenericProvider``).
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

# Hermes/Llama/Qwen-style inline tool-call format. Two equivalent tag pairs
# in the wild — Qwen3 is known to emit either form.
XML_TOOL_TAG_NAMES: Tuple[str, ...] = ("tool_call", "tool_calling")
XML_TOOL_BLOCK_RE = re.compile(
    r"<(?P<tag>" + "|".join(XML_TOOL_TAG_NAMES) + r")>\s*(?P<body>.*?)\s*</(?P=tag)>",
    re.DOTALL,
)
# Used by the streaming buffer to detect that the trailing portion of the
# buffer could still be the start of an opening tag for either variant.
XML_TOOL_OPENERS: Tuple[str, ...] = tuple(f"<{name}>" for name in XML_TOOL_TAG_NAMES)


def parse_xml_tool_call_payload(payload: str) -> Optional[Dict[str, Any]]:
    """Parse the JSON payload of a single ``<tool_call>...</tool_call>`` block.

    Returns ``{"name": str, "arguments": str}`` (arguments serialised as JSON)
    on success, or ``None`` if the payload is not parseable / not in the
    expected ``{"name": ..., "arguments": ...}`` shape — which lets callers
    leave the original text alone instead of silently dropping it.
    """
    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError:
        return None
    if not isinstance(decoded, dict):
        return None
    name = decoded.get("name")
    if not isinstance(name, str) or not name:
        return None
    arguments = decoded.get("arguments", {})
    if isinstance(arguments, str):
        # Some models double-encode arguments as a JSON string. Keep it as-is —
        # downstream parsers already handle both shapes.
        arguments_str = arguments
    else:
        arguments_str = json.dumps(arguments)
    return {"name": name, "arguments": arguments_str}


def extract_xml_tool_calls(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Strip ``<tool_call>``/``<tool_calling>`` blocks and return parsed calls.

    Returns ``(cleaned_text, [{"id", "name", "arguments"}, ...])``. Blocks
    whose JSON payload doesn't parse are left intact in ``cleaned_text`` so
    no information is silently lost.
    """
    # Cheap pre-check: skip the regex scan entirely when the text doesn't
    # contain either opener — the common case for plain assistant turns.
    if not any(opener in text for opener in XML_TOOL_OPENERS):
        return text, []

    parsed: List[Dict[str, Any]] = []
    cleaned_parts: List[str] = []
    cursor = 0
    for match in XML_TOOL_BLOCK_RE.finditer(text):
        payload = parse_xml_tool_call_payload(match.group("body"))
        if payload is None:
            # Malformed block — leave it in the text and continue scanning.
            continue
        cleaned_parts.append(text[cursor : match.start()])
        cursor = match.end()
        parsed.append(
            {
                "id": str(uuid.uuid4()),
                "name": payload["name"],
                "arguments": payload["arguments"],
            }
        )
    cleaned_parts.append(text[cursor:])
    cleaned = "".join(cleaned_parts).strip()
    return cleaned, parsed


def safe_emit_split(buffer: str) -> Tuple[str, str]:
    """Split a streaming buffer into ``(safe_to_emit, hold_back)``.

    ``hold_back`` covers any trailing portion that could still be the start of
    a ``<tool_call>`` / ``<tool_calling>`` opening tag, or a
    complete-but-not-yet-closed tag whose closing marker hasn't arrived.
    Everything before the earliest such ``<`` is safe to forward as text.
    """
    earliest_hold = len(buffer)
    pos = 0
    while True:
        idx = buffer.find("<", pos)
        if idx == -1:
            break
        rem = buffer[idx:]
        # Either rem is a strict prefix of an opener (still arriving) or
        # rem already starts with one (full opener, body in flight).
        if any(
            opener.startswith(rem) or rem.startswith(opener)
            for opener in XML_TOOL_OPENERS
        ):
            earliest_hold = idx
            break
        pos = idx + 1
    return buffer[:earliest_hold], buffer[earliest_hold:]


class XmlToolCall:
    """Stand-in for ``oci.generative_ai_inference.models.FunctionCall``.

    Provides the attributes (``id``, ``name``, ``arguments``) and the
    ``attribute_map`` that ``OCIUtils.convert_oci_tool_call_to_langchain``
    inspects, so calls parsed out of inline ``<tool_call>`` text flow through
    the same downstream conversion code as the structured ones.
    """

    attribute_map = {"id": "id", "name": "name", "arguments": "arguments"}

    def __init__(self, *, id: str, name: str, arguments: str) -> None:
        self.id = id
        self.name = name
        self.arguments = arguments


class XmlStreamBuffer:
    """Per-stream incremental parser for inline tool-call blocks.

    Handles ``<tool_call>...</tool_call>`` and the equivalent
    ``<tool_calling>...</tool_calling>`` variant. Wire into a streaming
    provider as:

    1. ``reset()`` at start of a new stream so leftover state from a prior
       stream can't leak forward.
    2. For each delta: ``safe = buffer.feed(delta)`` returns safe-to-emit
       text. Blocks that completed during this delta are queued internally;
       retrieve them with :meth:`drain_completed`.
    3. ``flush()`` at end-of-stream returns any held-back tail (e.g. an
       unclosed block, or a partial-opener prefix that never resolved).
    """

    def __init__(self) -> None:
        self._buffer: str = ""
        self._completed: List[Dict[str, Any]] = []

    def reset(self) -> None:
        """Clear the buffer and any pending completed blocks."""
        self._buffer = ""
        self._completed = []

    def feed(self, delta: str) -> str:
        """Append ``delta`` and return the safe-to-emit prefix.

        Any ``<tool_call>...</tool_call>`` blocks that completed inside the
        buffer during this call are queued internally for
        :meth:`drain_completed`.
        """
        self._buffer += delta
        # Drain any complete <tool_call>...</tool_call> blocks present in
        # the buffer, leaving partial / incomplete blocks in place.
        cleaned, parsed = extract_xml_tool_calls(self._buffer)
        if parsed:
            self._completed.extend(parsed)
            self._buffer = cleaned
        safe, hold_back = safe_emit_split(self._buffer)
        self._buffer = hold_back
        return safe

    def drain_completed(self) -> List[Dict[str, Any]]:
        """Return and clear any completed XML tool-call dicts."""
        if not self._completed:
            return []
        out = self._completed
        self._completed = []
        return out

    def flush(self) -> str:
        """Return any held-back text and reset state for the next stream.

        Trailing characters that were held back as a possible-opener prefix
        but never resolved into a ``<tool_call>`` block surface as a final
        delta here so they don't get dropped.
        """
        tail = self._buffer
        self._buffer = ""
        self._completed = []
        return tail
