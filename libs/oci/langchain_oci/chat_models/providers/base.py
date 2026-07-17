# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Abstract base class for OCI Generative AI providers."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Literal, Optional, Type, Union

from langchain_core.messages import BaseMessage
from langchain_core.messages.tool import ToolCallChunk
from langchain_core.tools import BaseTool
from pydantic import BaseModel


class Provider(ABC):
    """Abstract base class for OCI Generative AI providers."""

    @abstractmethod
    def normalize_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize parameters for this provider.

        Subclasses should implement provider-specific parameter transformations.

        Args:
            params: Dictionary of parameters to normalize

        Returns:
            Normalized parameters dictionary
        """
        ...

    @property
    @abstractmethod
    def stop_sequence_key(self) -> str:
        """Return the stop sequence key for the provider."""
        ...

    @abstractmethod
    def chat_response_to_text(self, response: Any) -> str:
        """Extract chat text from a provider's response (SDK object)."""
        ...

    @abstractmethod
    def chat_response_to_text_from_dict(self, response_data: Dict[str, Any]) -> str:
        """Extract chat text from a provider's response (JSON dict).

        Used by async path which works with raw JSON instead of SDK objects.
        """
        ...

    def new_stream_state(self) -> Optional[Any]:
        """Create per-stream parsing state for one ``_stream``/``_astream`` call.

        Providers that need to accumulate state across streaming events (e.g.
        GenericProvider's incremental ``<tool_call>`` XML buffer) return a
        fresh state object here. The caller owns it for the lifetime of one
        stream and passes it back via the ``stream_state`` keyword of
        :meth:`chat_stream_to_text` and :meth:`process_stream_tool_calls` —
        never store it on the provider, which is shared across concurrent
        streams.

        The default returns ``None`` (stateless provider); callers must not
        pass ``stream_state`` to providers that returned ``None``.
        """
        return None

    @abstractmethod
    def chat_stream_to_text(
        self, event_data: Dict, stream_state: Optional[Any] = None
    ) -> str:
        """Extract chat text from a streaming event.

        ``stream_state`` is the object returned by :meth:`new_stream_state`
        for the current stream (``None`` for stateless providers).
        """
        ...

    @abstractmethod
    def is_chat_stream_end(self, event_data: Dict) -> bool:
        """Determine if the chat stream event marks the end of a stream."""
        ...

    @abstractmethod
    def chat_generation_info(self, response: Any) -> Dict[str, Any]:
        """Extract generation metadata from a provider's response."""
        ...

    @abstractmethod
    def chat_stream_generation_info(self, event_data: Dict) -> Dict[str, Any]:
        """Extract generation metadata from a chat stream event."""
        ...

    def chat_stream_to_reasoning(self, event_data: Dict) -> str:
        """Extract incremental reasoning text from a streaming event.

        Reasoning models (e.g. xAI Grok, OpenAI o-series) emit a separate
        ``reasoning_content`` channel alongside the visible answer. Returning
        it per-delta here lets ``ChatOCIGenAI._stream`` / ``_astream`` accumulate
        it into ``AIMessageChunk.additional_kwargs["reasoning_content"]`` so the
        merged message exposes a standard ``reasoning`` content block — at parity
        with the non-streaming path.

        The default returns ``""`` (no reasoning), so providers that do not
        surface reasoning are unaffected.
        """
        return ""

    @abstractmethod
    def chat_tool_calls(self, response: Any) -> List[Any]:
        """Extract tool calls from a provider's response."""
        ...

    @abstractmethod
    def chat_stream_tool_calls(self, event_data: Dict) -> List[Any]:
        """Extract tool calls from a streaming event."""
        ...

    @abstractmethod
    def format_response_tool_calls(self, tool_calls: List[Any]) -> List[Any]:
        """Format response tool calls into LangChain's expected structure."""
        ...

    @abstractmethod
    def format_stream_tool_calls(self, tool_calls: List[Any]) -> List[Any]:
        """Format stream tool calls into LangChain's expected structure."""
        ...

    @abstractmethod
    def get_role(self, message: BaseMessage) -> str:
        """Map a LangChain message to the provider's role representation."""
        ...

    @abstractmethod
    def messages_to_oci_params(self, messages: Any, **kwargs: Any) -> Dict[str, Any]:
        """Convert LangChain messages to OCI API parameters."""
        ...

    @abstractmethod
    def convert_to_oci_tool(
        self, tool: Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]
    ) -> Dict[str, Any]:
        """Convert a tool definition into the provider-specific OCI tool format."""
        ...

    @abstractmethod
    def process_tool_choice(
        self,
        tool_choice: Optional[
            Union[dict, str, Literal["auto", "none", "required", "any"], bool]
        ],
    ) -> Optional[Any]:
        """Process tool choice parameter for the provider."""
        ...

    @abstractmethod
    def process_stream_tool_calls(
        self,
        event_data: Dict,
        tool_call_ids: Dict[int, str],
        stream_state: Optional[Any] = None,
        active_tool_call_indices: Optional[Dict[int, int]] = None,
    ) -> List[ToolCallChunk]:
        """Process streaming tool calls from event data into chunks.

        ``stream_state`` is the object returned by :meth:`new_stream_state`
        for the current stream (``None`` for stateless providers).

        ``tool_call_ids`` and ``active_tool_call_indices`` are per-stream
        aggregation state owned by the caller (``_stream`` / ``_astream``),
        never by the provider, so concurrent streams sharing one provider
        instance can't corrupt each other's tool-call routing.
        """
        ...

    @property
    def supports_tool_choice(self) -> bool:
        """Whether this provider supports the tool_choice parameter."""
        return False

    @property
    def supports_parallel_tool_calls(self) -> bool:
        """Whether this provider supports parallel tool calling.

        Parallel tool calling allows the model to call multiple tools
        simultaneously in a single response.

        Returns:
            bool: True if parallel tool calling is supported, False otherwise.
        """
        return False
