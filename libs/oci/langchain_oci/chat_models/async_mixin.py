# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Async mixin for OCI Generative AI chat models.

This module provides async support for ChatOCIGenAI through a mixin class,
keeping the main module clean and focused.
"""

import json
import uuid
from functools import cached_property
from typing import Any, AsyncIterator, Dict, List, Optional

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
from langchain_core.language_models.chat_models import agenerate_from_stream
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, ToolCall
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from langchain_oci.common.async_support import OCIAsyncClient
from langchain_oci.llms.utils import enforce_stop_tokens


class ChatOCIGenAIAsyncMixin:
    """Mixin providing async support for ChatOCIGenAI.

    This mixin adds _agenerate and _astream methods that use true async
    HTTP requests instead of thread pool wrappers.
    """

    @cached_property
    def _async_client(self) -> OCIAsyncClient:
        """Get the async client, creating it on first access.

        Uses @cached_property for thread-safe lazy initialization, following
        the pattern used by langchain-anthropic and langchain-aws.
        """
        base_client = self.client.base_client  # type: ignore[attr-defined]
        return OCIAsyncClient(
            service_endpoint=self.service_endpoint,  # type: ignore[attr-defined]
            signer=base_client.signer,
            config=getattr(base_client, "config", {}),
        )

    async def aclose(self) -> None:
        """Close the async HTTP client and release resources.

        Call this when done with async operations to clean up connections.
        If not called, connections will be cleaned up on garbage collection.
        """
        if "_async_client" in self.__dict__:
            await self._async_client.close()
            del self.__dict__["_async_client"]

    def _prepare_async_request(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]],
        stream: bool,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Prepare request data for async chat.

        Returns dict with compartment_id, chat_request_dict, serving_mode_dict.

        Reuses _prepare_request from the main class and converts OCI model
        objects to dicts for JSON serialization in async HTTP requests.
        """
        from oci.util import to_dict

        # Reuse the sync _prepare_request which returns a ChatDetails object
        chat_details = self._prepare_request(messages, stop, stream, **kwargs)  # type: ignore[attr-defined]

        return {
            "compartment_id": chat_details.compartment_id,
            "chat_request_dict": to_dict(chat_details.chat_request),
            "serving_mode_dict": to_dict(chat_details.serving_mode),
        }

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Asynchronously generate chat response.

        Args:
            messages: List of messages to send.
            stop: Optional stop sequences.
            run_manager: Callback manager for async operations.
            **kwargs: Additional arguments.

        Returns:
            ChatResult with generated response.
        """
        if self.is_stream:  # type: ignore[attr-defined]
            stream_iter = self._astream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return await agenerate_from_stream(stream_iter)

        client = self._async_client
        request_data = self._prepare_async_request(
            messages, stop, stream=False, **kwargs
        )

        # Get single response
        response_data = None
        async for data in client.chat_async(
            compartment_id=request_data["compartment_id"],
            chat_request_dict=request_data["chat_request_dict"],
            serving_mode_dict=request_data["serving_mode_dict"],
            stream=False,
        ):
            response_data = data
            break

        if response_data is None:
            raise RuntimeError("No response received from OCI GenAI")

        # Parse response
        content = self._extract_content_from_response(response_data)

        if stop is not None:
            content = enforce_stop_tokens(content, stop)

        generation_info = self._extract_generation_info(response_data)

        tool_calls = []
        if "tool_calls" in generation_info:
            tool_calls = self._convert_dict_tool_calls_to_langchain(
                self._extract_tool_calls(response_data)
            )

        usage_metadata = self._extract_usage_metadata(response_data)

        message = AIMessage(
            content=content or "",
            additional_kwargs=generation_info,
            tool_calls=tool_calls,
            usage_metadata=usage_metadata,
        )

        llm_output = {
            "model_id": response_data.get("modelId"),
            "model_version": response_data.get("modelVersion"),
        }

        return ChatResult(
            generations=[
                ChatGeneration(message=message, generation_info=generation_info)
            ],
            llm_output=llm_output,
        )

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Asynchronously stream chat responses.

        Args:
            messages: List of messages to send.
            stop: Optional stop sequences.
            run_manager: Callback manager for async operations.
            **kwargs: Additional arguments.

        Yields:
            ChatGenerationChunk objects as they arrive.
        """
        client = self._async_client
        request_data = self._prepare_async_request(
            messages, stop, stream=True, **kwargs
        )
        tool_call_ids: Dict[int, str] = {}

        async for event_data in client.chat_async(
            compartment_id=request_data["compartment_id"],
            chat_request_dict=request_data["chat_request_dict"],
            serving_mode_dict=request_data["serving_mode_dict"],
            stream=True,
        ):
            if not self._provider.is_chat_stream_end(event_data):  # type: ignore[attr-defined]
                # Process streaming content
                delta = self._provider.chat_stream_to_text(event_data)  # type: ignore[attr-defined]
                tool_call_chunks = self._provider.process_stream_tool_calls(  # type: ignore[attr-defined]
                    event_data, tool_call_ids
                )

                chunk = ChatGenerationChunk(
                    message=AIMessageChunk(
                        content=delta,
                        tool_call_chunks=tool_call_chunks,
                    )
                )
                if run_manager:
                    await run_manager.on_llm_new_token(delta, chunk=chunk)
                yield chunk
            else:
                generation_info = self._provider.chat_stream_generation_info(event_data)  # type: ignore[attr-defined]
                yield ChatGenerationChunk(
                    message=AIMessageChunk(
                        content="",
                        additional_kwargs=generation_info,
                    ),
                    generation_info=generation_info,
                )

    def _extract_content_from_response(self, response_data: Dict[str, Any]) -> str:
        """Extract text content from async response data.

        Delegate to provider-specific parser so async and sync paths share
        the same parsing logic.
        """
        return self._provider.chat_response_to_text_from_dict(response_data)  # type: ignore[attr-defined]

    def _extract_generation_info(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract generation info from async response data."""
        chat_response = response_data.get("chatResponse", {})
        info: Dict[str, Any] = {
            "finish_reason": chat_response.get("finishReason"),
        }

        # Cohere-specific fields
        if "documents" in chat_response:
            info["documents"] = chat_response["documents"]
        if "citations" in chat_response:
            info["citations"] = chat_response["citations"]

        # Tool calls - format directly from dict since async returns JSON dicts
        tool_calls = self._extract_tool_calls(response_data)
        if tool_calls:
            info["tool_calls"] = self._format_tool_calls_from_dict(tool_calls)

        return {k: v for k, v in info.items() if v is not None}

    def _safe_parse_json_args(self, raw_args: Any) -> Dict[str, Any]:
        """Parse JSON arguments with fallback for malformed JSON.

        Handles both string JSON and dict arguments, with graceful fallback
        for malformed JSON from LLMs.

        Args:
            raw_args: Either a JSON string or dict of arguments.

        Returns:
            Parsed arguments dict, or {"_raw_arguments": ...} on parse failure.
        """
        if isinstance(raw_args, dict):
            return raw_args
        if isinstance(raw_args, str):
            try:
                parsed = json.loads(raw_args)
                # Handle double-encoded JSON (some models do this)
                if isinstance(parsed, str):
                    parsed = json.loads(parsed)
                return parsed
            except json.JSONDecodeError:
                return {"_raw_arguments": raw_args}
        return {}

    def _format_tool_calls_from_dict(
        self, tool_calls: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Format tool calls from async response dicts to LangChain format.

        Used for additional_kwargs metadata - empty string IDs are acceptable here
        since this is just for response metadata, not for tool execution matching.
        """
        formatted = []
        for tc in tool_calls:
            # Handle both 'arguments' (Generic/Meta) and 'parameters' (Cohere)
            if "arguments" in tc:
                raw_args = tc.get("arguments")
            else:
                raw_args = tc.get("parameters", {})
            args = self._safe_parse_json_args(raw_args)

            formatted.append(
                {
                    # Empty string ID is acceptable for metadata (additional_kwargs)
                    "id": tc.get("id", ""),
                    "function": {
                        "name": tc.get("name", ""),
                        "arguments": args,
                    },
                    "type": "function",
                }
            )
        return formatted

    def _convert_dict_tool_calls_to_langchain(
        self, tool_calls: List[Dict[str, Any]]
    ) -> List[ToolCall]:
        """Convert dict tool calls from async response to LangChain ToolCall format.

        Used for creating ToolCall objects that LangChain uses to match tool results
        back to tool calls. A valid ID is required or the agent loop breaks.
        """
        return [
            ToolCall(
                name=tc.get("name", ""),
                args=self._parse_tool_args(tc),
                # UUID fallback required: LangChain uses ID to match tool results
                # back to calls. Missing/empty ID breaks the agent tool loop.
                id=tc.get("id") or uuid.uuid4().hex,
            )
            for tc in tool_calls
        ]

    def _parse_tool_args(self, tc: Dict[str, Any]) -> Dict[str, Any]:
        """Parse tool call arguments from dict, handling JSON strings."""
        # Handle both 'arguments' (Generic/Meta) and 'parameters' (Cohere)
        if "arguments" in tc:
            raw_args = tc.get("arguments")
        else:
            raw_args = tc.get("parameters", {})
        return self._safe_parse_json_args(raw_args)

    def _extract_tool_calls(self, response_data: Dict[str, Any]) -> List[Any]:
        """Extract tool calls from async response data."""
        chat_response = response_data.get("chatResponse", {})

        # Cohere V1
        if "toolCalls" in chat_response:
            return chat_response.get("toolCalls", [])

        # Generic/Meta format
        if "choices" in chat_response:
            choices = chat_response.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                return message.get("toolCalls", [])

        # Cohere V2
        if "message" in chat_response:
            message = chat_response.get("message", {})
            return message.get("toolCalls", [])

        return []

    def _extract_usage_metadata(
        self, response_data: Dict[str, Any]
    ) -> Optional[UsageMetadata]:
        """Extract usage metadata from async response data.

        Uses LangChain's UsageMetadata directly for consistency.
        """
        chat_response = response_data.get("chatResponse", {})
        usage = chat_response.get("usage")

        if usage:
            return UsageMetadata(
                input_tokens=usage.get("promptTokens", 0),
                output_tokens=usage.get("completionTokens", 0),
                total_tokens=usage.get("totalTokens", 0),
            )

        return None
