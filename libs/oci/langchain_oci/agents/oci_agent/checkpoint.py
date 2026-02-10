# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Checkpointing support for OCIGenAIAgent.

Enables saving and restoring agent state across invocations,
supporting persistent conversations and resumable agent runs.

Compatible with LangGraph's checkpoint interface.

Example:
    from langchain_oci import OCIGenAIAgent
    from langchain_oci.agents.oci_agent.checkpoint import MemoryCheckpointer

    checkpointer = MemoryCheckpointer()

    agent = OCIGenAIAgent(
        model_id="openai.gpt-5.2",
        tools=[...],
        checkpointer=checkpointer,
    )

    # First conversation
    result1 = agent.invoke("What's 2+2?", thread_id="user-123")

    # Continue same conversation later
    result2 = agent.invoke("Now multiply by 10", thread_id="user-123")

    # List all checkpoints for a thread
    checkpoints = checkpointer.list("user-123")
"""

from __future__ import annotations

import json
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)


@dataclass(frozen=True)
class Checkpoint:
    """Immutable snapshot of agent state.

    Attributes:
        id: Unique checkpoint identifier.
        thread_id: Conversation thread identifier.
        iteration: Iteration number when checkpoint was created.
        messages: Conversation messages at this point.
        metadata: Additional metadata (timestamps, tool calls, etc.).
        parent_id: ID of the previous checkpoint in this thread.
    """

    id: str
    thread_id: str
    iteration: int
    messages: tuple[dict[str, Any], ...]
    metadata: dict[str, Any] = field(default_factory=dict)
    parent_id: str | None = None

    @classmethod
    def create(
        cls,
        thread_id: str,
        iteration: int,
        messages: Sequence[BaseMessage],
        metadata: dict[str, Any] | None = None,
        parent_id: str | None = None,
    ) -> Checkpoint:
        """Create a new checkpoint from current state.

        Args:
            thread_id: Conversation thread identifier.
            iteration: Current iteration number.
            messages: Current conversation messages.
            metadata: Optional additional metadata.
            parent_id: ID of the previous checkpoint.

        Returns:
            New Checkpoint instance.
        """
        checkpoint_id = f"ckpt_{uuid.uuid4().hex[:12]}"

        # Serialize messages to dicts
        serialized = tuple(_serialize_message(m) for m in messages)

        # Add timestamp to metadata
        meta = {
            "created_at": time.time(),
            "created_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **(metadata or {}),
        }

        return cls(
            id=checkpoint_id,
            thread_id=thread_id,
            iteration=iteration,
            messages=serialized,
            metadata=meta,
            parent_id=parent_id,
        )

    def get_messages(self) -> list[BaseMessage]:
        """Deserialize and return messages.

        Returns:
            List of BaseMessage instances.
        """
        return [_deserialize_message(m) for m in self.messages]

    def to_dict(self) -> dict[str, Any]:
        """Convert checkpoint to dictionary for serialization.

        Returns:
            Dictionary representation.
        """
        return {
            "id": self.id,
            "thread_id": self.thread_id,
            "iteration": self.iteration,
            "messages": list(self.messages),
            "metadata": self.metadata,
            "parent_id": self.parent_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Checkpoint:
        """Create checkpoint from dictionary.

        Args:
            data: Dictionary representation.

        Returns:
            Checkpoint instance.
        """
        return cls(
            id=data["id"],
            thread_id=data["thread_id"],
            iteration=data["iteration"],
            messages=tuple(data["messages"]),
            metadata=data.get("metadata", {}),
            parent_id=data.get("parent_id"),
        )


def _serialize_message(msg: BaseMessage) -> dict[str, Any]:
    """Serialize a message to a dictionary.

    Args:
        msg: Message to serialize.

    Returns:
        Dictionary representation.
    """
    base = {
        "type": msg.__class__.__name__,
        "content": msg.content,
    }

    # Add type-specific fields
    if isinstance(msg, AIMessage):
        if msg.tool_calls:
            base["tool_calls"] = msg.tool_calls
        if hasattr(msg, "additional_kwargs") and msg.additional_kwargs:
            base["additional_kwargs"] = msg.additional_kwargs
    elif isinstance(msg, ToolMessage):
        base["tool_call_id"] = msg.tool_call_id
        if hasattr(msg, "name"):
            base["name"] = msg.name

    return base


def _deserialize_message(data: dict[str, Any]) -> BaseMessage:
    """Deserialize a message from a dictionary.

    Args:
        data: Dictionary representation.

    Returns:
        BaseMessage instance.
    """
    msg_type = data["type"]
    content = data["content"]

    if msg_type == "HumanMessage":
        return HumanMessage(content=content)
    elif msg_type == "AIMessage":
        kwargs = {"content": content}
        if "tool_calls" in data:
            kwargs["tool_calls"] = data["tool_calls"]
        if "additional_kwargs" in data:
            kwargs["additional_kwargs"] = data["additional_kwargs"]
        return AIMessage(**kwargs)
    elif msg_type == "ToolMessage":
        return ToolMessage(
            content=content,
            tool_call_id=data.get("tool_call_id", ""),
            name=data.get("name"),
        )
    elif msg_type == "SystemMessage":
        return SystemMessage(content=content)
    else:
        # Fallback to HumanMessage
        return HumanMessage(content=content)


class BaseCheckpointer(ABC):
    """Abstract base class for checkpoint storage.

    Subclass this to implement custom storage backends
    (Redis, PostgreSQL, S3, etc.).
    """

    @abstractmethod
    def get(self, thread_id: str) -> Checkpoint | None:
        """Get the latest checkpoint for a thread.

        Args:
            thread_id: Thread identifier.

        Returns:
            Latest checkpoint or None if not found.
        """
        pass

    @abstractmethod
    def get_by_id(self, checkpoint_id: str) -> Checkpoint | None:
        """Get a specific checkpoint by ID.

        Args:
            checkpoint_id: Checkpoint identifier.

        Returns:
            Checkpoint or None if not found.
        """
        pass

    @abstractmethod
    def put(self, checkpoint: Checkpoint) -> None:
        """Save a checkpoint.

        Args:
            checkpoint: Checkpoint to save.
        """
        pass

    @abstractmethod
    def list(self, thread_id: str) -> Iterator[Checkpoint]:
        """List all checkpoints for a thread.

        Args:
            thread_id: Thread identifier.

        Yields:
            Checkpoints in chronological order.
        """
        pass

    @abstractmethod
    def delete(self, thread_id: str) -> int:
        """Delete all checkpoints for a thread.

        Args:
            thread_id: Thread identifier.

        Returns:
            Number of checkpoints deleted.
        """
        pass


class MemoryCheckpointer(BaseCheckpointer):
    """In-memory checkpoint storage.

    Useful for development and testing. Not persistent across restarts.

    Example:
        checkpointer = MemoryCheckpointer()
        agent = OCIGenAIAgent(..., checkpointer=checkpointer)

        result = agent.invoke("Hello", thread_id="thread-1")

        # Later, continue the conversation
        result = agent.invoke("Continue...", thread_id="thread-1")
    """

    def __init__(self) -> None:
        """Initialize empty storage."""
        self._storage: dict[str, list[Checkpoint]] = {}
        self._by_id: dict[str, Checkpoint] = {}

    def get(self, thread_id: str) -> Checkpoint | None:
        """Get the latest checkpoint for a thread."""
        checkpoints = self._storage.get(thread_id, [])
        return checkpoints[-1] if checkpoints else None

    def get_by_id(self, checkpoint_id: str) -> Checkpoint | None:
        """Get a specific checkpoint by ID."""
        return self._by_id.get(checkpoint_id)

    def put(self, checkpoint: Checkpoint) -> None:
        """Save a checkpoint."""
        if checkpoint.thread_id not in self._storage:
            self._storage[checkpoint.thread_id] = []
        self._storage[checkpoint.thread_id].append(checkpoint)
        self._by_id[checkpoint.id] = checkpoint

    def list(self, thread_id: str) -> Iterator[Checkpoint]:
        """List all checkpoints for a thread."""
        yield from self._storage.get(thread_id, [])

    def delete(self, thread_id: str) -> int:
        """Delete all checkpoints for a thread."""
        checkpoints = self._storage.pop(thread_id, [])
        for ckpt in checkpoints:
            self._by_id.pop(ckpt.id, None)
        return len(checkpoints)

    def clear(self) -> None:
        """Clear all checkpoints."""
        self._storage.clear()
        self._by_id.clear()

    @property
    def thread_count(self) -> int:
        """Number of threads with checkpoints."""
        return len(self._storage)

    @property
    def checkpoint_count(self) -> int:
        """Total number of checkpoints."""
        return len(self._by_id)


class FileCheckpointer(BaseCheckpointer):
    """File-based checkpoint storage using JSON.

    Persists checkpoints to a JSON file. Suitable for single-process
    applications that need persistence.

    Example:
        checkpointer = FileCheckpointer("/path/to/checkpoints.json")
        agent = OCIGenAIAgent(..., checkpointer=checkpointer)
    """

    def __init__(self, path: str) -> None:
        """Initialize file-based storage.

        Args:
            path: Path to JSON file for storage.
        """
        self._path = path
        self._storage: dict[str, list[dict[str, Any]]] = {}
        self._load()

    def _load(self) -> None:
        """Load checkpoints from file."""
        try:
            with open(self._path, "r") as f:
                self._storage = json.load(f)
        except FileNotFoundError:
            self._storage = {}
        except json.JSONDecodeError:
            self._storage = {}

    def _save(self) -> None:
        """Save checkpoints to file."""
        with open(self._path, "w") as f:
            json.dump(self._storage, f, indent=2)

    def get(self, thread_id: str) -> Checkpoint | None:
        """Get the latest checkpoint for a thread."""
        checkpoints = self._storage.get(thread_id, [])
        if checkpoints:
            return Checkpoint.from_dict(checkpoints[-1])
        return None

    def get_by_id(self, checkpoint_id: str) -> Checkpoint | None:
        """Get a specific checkpoint by ID."""
        for thread_checkpoints in self._storage.values():
            for ckpt_data in thread_checkpoints:
                if ckpt_data["id"] == checkpoint_id:
                    return Checkpoint.from_dict(ckpt_data)
        return None

    def put(self, checkpoint: Checkpoint) -> None:
        """Save a checkpoint."""
        if checkpoint.thread_id not in self._storage:
            self._storage[checkpoint.thread_id] = []
        self._storage[checkpoint.thread_id].append(checkpoint.to_dict())
        self._save()

    def list(self, thread_id: str) -> Iterator[Checkpoint]:
        """List all checkpoints for a thread."""
        for ckpt_data in self._storage.get(thread_id, []):
            yield Checkpoint.from_dict(ckpt_data)

    def delete(self, thread_id: str) -> int:
        """Delete all checkpoints for a thread."""
        checkpoints = self._storage.pop(thread_id, [])
        self._save()
        return len(checkpoints)
