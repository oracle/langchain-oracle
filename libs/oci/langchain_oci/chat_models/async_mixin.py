# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Async mixin for OCI Generative AI chat models.

This module provides async support for ChatOCIGenAI through a mixin class,
keeping the main module clean and focused.
"""

from typing import Any, AsyncIterator, Dict, List, Optional

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
from langchain_core.language_models.chat_models import agenerate_from_stream
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from langchain_oci.common.async_support import OCIAsyncClient
from langchain_oci.common.utils import OCIUtils
from langchain_oci.llms.utils import enforce_stop_tokens


class ChatOCIGenAIAsyncMixin:
    """Mixin providing async support for ChatOCIGenAI.

    This mixin adds _agenerate and _astream methods that use true async
    HTTP requests instead of thread pool wrappers.
    """

    _async_client: Optional[OCIAsyncClient] = None

    def _get_async_client(self) -> OCIAsyncClient:
        """Get or create the async client."""
        if self._async_client is None:
            self._async_client = OCIAsyncClient(
                service_endpoint=self.service_endpoint,  # type: ignore[attr-defined]
                signer=self.oci_signer,  # type: ignore[attr-defined]
                config=self.oci_config,  # type: ignore[attr-defined]
            )
        return self._async_client

    def _prepare_async_request(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]],
        stream: bool,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Prepare request data for async chat.

        Returns dict with compartment_id, chat_request_dict, serving_mode_dict.
        """
        from oci.generative_ai_inference import models
        from oci.util import to_dict

        oci_params = self._provider.messages_to_oci_params(  # type: ignore[attr-defined]
            messages,
            max_sequential_tool_calls=self.max_sequential_tool_calls,  # type: ignore[attr-defined]
            model_id=self.model_id,  # type: ignore[attr-defined]
            **kwargs,
        )

        oci_params["is_stream"] = stream
        _model_kwargs = self.model_kwargs or {}  # type: ignore[attr-defined]

        if stop is not None:
            _model_kwargs[self._provider.stop_sequence_key] = stop  # type: ignore[attr-defined]

        chat_params = {**_model_kwargs, **kwargs, **oci_params}

        if not self.model_id:  # type: ignore[attr-defined]
            raise ValueError("Model ID is required for chat.")

        # Build serving mode
        from langchain_oci.common.utils import CUSTOM_ENDPOINT_PREFIX

        if self.model_id.startswith(CUSTOM_ENDPOINT_PREFIX):  # type: ignore[attr-defined]
            serving_mode = models.DedicatedServingMode(endpoint_id=self.model_id)  # type: ignore[attr-defined]
        else:
            serving_mode = models.OnDemandServingMode(model_id=self.model_id)  # type: ignore[attr-defined]

        # Check for V2 API
        use_v2 = chat_params.pop("_use_v2_api", False)

        if use_v2:
            v2_request_class = getattr(self._provider, "oci_chat_request_v2", None)  # type: ignore[attr-defined]
            if v2_request_class is None:
                raise ValueError(
                    "V2 API is not supported by the current provider. "
                    "V2 API with multimodal support is only available for "
                    "Cohere models."
                )
            chat_request = v2_request_class(**chat_params)
        else:
            chat_request = self._provider.oci_chat_request(**chat_params)  # type: ignore[attr-defined]

        return {
            "compartment_id": self.compartment_id,  # type: ignore[attr-defined]
            "chat_request_dict": to_dict(chat_request),
            "serving_mode_dict": to_dict(serving_mode),
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

        client = self._get_async_client()
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
            tool_calls = [
                OCIUtils.convert_oci_tool_call_to_langchain(tc)
                for tc in self._extract_tool_calls(response_data)
            ]

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
        client = self._get_async_client()
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
        """Extract text content from async response data."""
        chat_response = response_data.get("chatResponse", {})

        # Try different response formats
        # Cohere V1
        if "text" in chat_response:
            return chat_response.get("text", "")

        # Generic/Meta format
        if "choices" in chat_response:
            choices = chat_response.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                content = message.get("content", [])
                if isinstance(content, list):
                    texts = [
                        c.get("text", "")
                        for c in content
                        if isinstance(c, dict) and c.get("type") == "TEXT"
                    ]
                    return "".join(texts)
                return str(content) if content else ""

        # Cohere V2
        if "message" in chat_response:
            message = chat_response.get("message", {})
            content = message.get("content", [])
            if isinstance(content, list):
                texts = [
                    c.get("text", "")
                    for c in content
                    if isinstance(c, dict) and c.get("type") == "TEXT"
                ]
                return "".join(texts)

        return ""

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

        # Tool calls
        tool_calls = self._extract_tool_calls(response_data)
        if tool_calls:
            info["tool_calls"] = self._provider.format_response_tool_calls(tool_calls)  # type: ignore[attr-defined]

        return {k: v for k, v in info.items() if v is not None}

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

    def _extract_usage_metadata(self, response_data: Dict[str, Any]) -> Optional[Any]:
        """Extract usage metadata from async response data."""
        chat_response = response_data.get("chatResponse", {})
        usage = chat_response.get("usage")

        if usage:
            # Create a simple object that OCIUtils.create_usage_metadata can handle
            class UsageData:
                pass

            usage_obj = UsageData()
            usage_obj.prompt_tokens = usage.get("promptTokens", 0)  # type: ignore
            usage_obj.completion_tokens = usage.get("completionTokens", 0)  # type: ignore
            usage_obj.total_tokens = usage.get("totalTokens", 0)  # type: ignore

            return OCIUtils.create_usage_metadata(usage_obj)

        return None
