# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Message compression for context management.

Prevents unbounded context growth by trimming old messages while
preserving message integrity (AI + Tool message pairs stay together).
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage


class CompressionStrategy(str, Enum):
    """Message compression strategies."""

    NONE = "none"
    FIXED_WINDOW = "fixed_window"
    SMART_TRIM = "smart_trim"


class CompressionConfig(BaseModel):
    """Configuration for message compression.

    Attributes:
        strategy: Compression strategy to use.
        max_messages: Maximum messages to keep.
        preserve_system: Always keep system messages.
    """

    model_config = ConfigDict(frozen=True)

    strategy: CompressionStrategy = CompressionStrategy.SMART_TRIM
    max_messages: int = 20
    preserve_system: bool = True


class CompressionResult(BaseModel):
    """Result of message compression.

    Attributes:
        messages: Compressed message list.
        dropped_count: Number of messages removed.
    """

    model_config = ConfigDict(frozen=True)

    messages: tuple = Field(default_factory=tuple)
    dropped_count: int = 0


def compress_messages(
    messages: tuple[BaseMessage, ...],
    config: CompressionConfig,
) -> CompressionResult:
    """Compress message history based on strategy.

    Args:
        messages: Current message tuple.
        config: Compression configuration.

    Returns:
        Compression result with new message tuple.
    """
    if config.strategy == CompressionStrategy.NONE:
        return CompressionResult(messages=messages, dropped_count=0)

    if len(messages) <= config.max_messages:
        return CompressionResult(messages=messages, dropped_count=0)

    if config.strategy == CompressionStrategy.FIXED_WINDOW:
        return _fixed_window_compress(messages, config)

    if config.strategy == CompressionStrategy.SMART_TRIM:
        return _smart_trim_compress(messages, config)

    return CompressionResult(messages=messages, dropped_count=0)


def _fixed_window_compress(
    messages: tuple[BaseMessage, ...],
    config: CompressionConfig,
) -> CompressionResult:
    """Keep system messages + most recent N messages."""
    from langchain_core.messages import SystemMessage

    system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
    other_msgs = [m for m in messages if not isinstance(m, SystemMessage)]

    keep_count = config.max_messages - len(system_msgs)
    if keep_count <= 0:
        kept_others: list[BaseMessage] = []
    else:
        kept_others = other_msgs[-keep_count:]

    dropped = len(other_msgs) - len(kept_others)
    result = tuple(system_msgs) + tuple(kept_others)

    return CompressionResult(messages=result, dropped_count=dropped)


def _smart_trim_compress(
    messages: tuple[BaseMessage, ...],
    config: CompressionConfig,
) -> CompressionResult:
    """Trim while preserving AI+Tool message pairs.

    Groups that must stay together:
    - AI message with tool_calls + subsequent ToolMessages
    - Human message + following AI response
    """
    from langchain_core.messages import (
        AIMessage,
        HumanMessage,
        SystemMessage,
        ToolMessage,
    )

    groups = _identify_message_groups(list(messages))

    system_groups = [
        g for g in groups if any(isinstance(m, SystemMessage) for m in g)
    ]
    other_groups = [
        g for g in groups if not any(isinstance(m, SystemMessage) for m in g)
    ]

    system_msg_count = sum(len(g) for g in system_groups)
    available = config.max_messages - system_msg_count

    kept_other: list[list[BaseMessage]] = []
    running_count = 0
    dropped_count = 0

    for group in reversed(other_groups):
        if running_count + len(group) <= available:
            kept_other.insert(0, group)
            running_count += len(group)
        else:
            dropped_count += len(group)

    result_messages: list[BaseMessage] = []
    for g in system_groups:
        result_messages.extend(g)
    for g in kept_other:
        result_messages.extend(g)

    return CompressionResult(
        messages=tuple(result_messages),
        dropped_count=dropped_count,
    )


def _identify_message_groups(
    messages: list[BaseMessage],
) -> list[list[BaseMessage]]:
    """Group messages that must stay together."""
    from langchain_core.messages import (
        AIMessage,
        HumanMessage,
        SystemMessage,
        ToolMessage,
    )

    groups: list[list[BaseMessage]] = []
    i = 0

    while i < len(messages):
        msg = messages[i]

        if isinstance(msg, SystemMessage):
            groups.append([msg])
            i += 1
            continue

        if isinstance(msg, AIMessage) and msg.tool_calls:
            group: list[BaseMessage] = [msg]
            i += 1
            while i < len(messages) and isinstance(messages[i], ToolMessage):
                group.append(messages[i])
                i += 1
            groups.append(group)
            continue

        if isinstance(msg, HumanMessage):
            group = [msg]
            i += 1
            if i < len(messages) and isinstance(messages[i], AIMessage):
                group.append(messages[i])
                i += 1
            groups.append(group)
            continue

        groups.append([msg])
        i += 1

    return groups
