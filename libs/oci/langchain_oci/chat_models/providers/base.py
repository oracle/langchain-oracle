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
        """Extract chat text from a provider's response."""
        ...

    @abstractmethod
    def chat_stream_to_text(self, event_data: Dict) -> str:
        """Extract chat text from a streaming event."""
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
    ) -> List[ToolCallChunk]:
        """Process streaming tool calls from event data into chunks."""
        ...

    @property
    def supports_parallel_tool_calls(self) -> bool:
        """Whether this provider supports parallel tool calling.

        Parallel tool calling allows the model to call multiple tools
        simultaneously in a single response.

        Returns:
            bool: True if parallel tool calling is supported, False otherwise.
        """
        return False
