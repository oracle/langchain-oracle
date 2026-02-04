# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Generic provider implementation for OCI Generative AI.

Supports Meta Llama, xAI Grok, OpenAI, and Mistral models.
"""

import json
import uuid
from typing import Any, Callable, Dict, List, Literal, Optional, Type, Union

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.tool import ToolCallChunk, tool_call_chunk
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_function
from pydantic import BaseModel

from langchain_oci.chat_models.providers.base import Provider
from langchain_oci.common.utils import OCIUtils


def _should_allow_more_tool_calls(
    messages: List[BaseMessage], max_tool_calls: int
) -> bool:
    """
    Determine if the model should be allowed to call more tools.

    Returns False (force stop) if:
    - Tool call limit exceeded
    - Infinite loop detected (same tool called repeatedly with same args)

    Returns True otherwise to allow multi-step tool orchestration.

    Args:
        messages: Conversation history
        max_tool_calls: Maximum number of tool calls before forcing stop
    """
    # Count total tool calls made so far
    tool_call_count = sum(1 for msg in messages if isinstance(msg, ToolMessage))

    # Safety limit: prevent runaway tool calling
    if tool_call_count >= max_tool_calls:
        return False

    # Detect infinite loop: same tool called with same arguments in succession
    recent_calls: list = []
    for msg in reversed(messages):
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                # Create signature: (tool_name, sorted_args)
                try:
                    args_str = json.dumps(tc.get("args", {}), sort_keys=True)
                    signature = (tc.get("name", ""), args_str)

                    # Check if this exact call was made in last 2 calls
                    if signature in recent_calls[-2:]:
                        return False  # Infinite loop detected

                    recent_calls.append(signature)
                except Exception:
                    # If we can't serialize args, be conservative and continue
                    pass

        # Only check last 4 AI messages (last 4 tool call attempts)
        if len(recent_calls) >= 4:
            break

    return True


class GenericProvider(Provider):
    """Provider for models using generic API spec."""

    stop_sequence_key: str = "stop"

    @property
    def supports_parallel_tool_calls(self) -> bool:
        """GenericProvider models support parallel tool calling."""
        return True

    def __init__(self) -> None:
        from oci.generative_ai_inference import models

        # Chat request and message models
        self.oci_chat_request = models.GenericChatRequest
        self.oci_chat_message = {
            "USER": models.UserMessage,
            "SYSTEM": models.SystemMessage,
            "ASSISTANT": models.AssistantMessage,
            "TOOL": models.ToolMessage,
        }

        # Content models
        self.oci_chat_message_content = models.ChatContent
        self.oci_chat_message_text_content = models.TextContent
        self.oci_chat_message_image_content = models.ImageContent
        self.oci_chat_message_image_url = models.ImageUrl

        # Tool-related models
        self.oci_function_definition = models.FunctionDefinition
        self.oci_tool_choice_auto = models.ToolChoiceAuto
        self.oci_tool_choice_function = models.ToolChoiceFunction
        self.oci_tool_choice_none = models.ToolChoiceNone
        self.oci_tool_choice_required = models.ToolChoiceRequired
        self.oci_tool_call = models.FunctionCall
        self.oci_tool_message = models.ToolMessage

        # Response format models
        self.oci_response_json_schema = models.ResponseJsonSchema
        self.oci_json_schema_response_format = models.JsonSchemaResponseFormat

        self.chat_api_format = models.BaseChatRequest.API_FORMAT_GENERIC

    def chat_response_to_text(self, response: Any) -> str:
        """Extract text from chat response, or '' if unavailable."""
        chat_resp = getattr(response.data, "chat_response", None)
        choices = getattr(chat_resp, "choices", None)
        if not choices:
            return ""
        msg = getattr(choices[0], "message", None)
        content = msg.content[0] if msg and msg.content else None
        return getattr(content, "text", "") or ""

    def chat_stream_to_text(self, event_data: Dict) -> str:
        """Extract text from Meta chat stream event."""
        content = event_data.get("message", {}).get("content", None)
        if not content:
            return ""
        return content[0].get("text", "")

    def is_chat_stream_end(self, event_data: Dict) -> bool:
        """Determine if Meta chat stream event indicates the end."""
        return "finishReason" in event_data

    def chat_generation_info(self, response: Any) -> Dict[str, Any]:
        """Extract generation metadata from chat response."""
        choices = response.data.chat_response.choices
        finish_reason = choices[0].finish_reason if choices else None
        generation_info: Dict[str, Any] = {
            "finish_reason": finish_reason,
            "time_created": str(response.data.chat_response.time_created),
        }

        # Surface reasoning_content from reasoning models (xAI Grok, OpenAI o1).
        if choices and choices[0].message is not None:
            reasoning = getattr(choices[0].message, "reasoning_content", None)
            if reasoning:
                generation_info["reasoning_content"] = reasoning

        # Include token usage if available
        if (
            hasattr(response.data.chat_response, "usage")
            and response.data.chat_response.usage
        ):
            generation_info["total_tokens"] = (
                response.data.chat_response.usage.total_tokens
            )

        if self.chat_tool_calls(response):
            generation_info["tool_calls"] = self.format_response_tool_calls(
                self.chat_tool_calls(response)
            )
        return generation_info

    def chat_stream_generation_info(self, event_data: Dict) -> Dict[str, Any]:
        """Extract generation metadata from Meta chat stream event."""
        return {"finish_reason": event_data["finishReason"]}

    def chat_tool_calls(self, response: Any) -> List[Any]:
        """Retrieve tool calls from chat response."""
        choices = response.data.chat_response.choices
        if not choices or choices[0].message is None:
            return []
        return choices[0].message.tool_calls

    def chat_stream_tool_calls(self, event_data: Dict) -> List[Any]:
        """Retrieve tool calls from Meta stream event."""
        return event_data.get("message", {}).get("toolCalls", [])

    def format_response_tool_calls(self, tool_calls: List[Any]) -> List[Dict]:
        """
        Formats a OCI GenAI API Meta response
        into the tool call format used in Langchain.
        """

        if not tool_calls:
            return []

        formatted_tool_calls: List[Dict] = []
        for tool_call in tool_calls:
            formatted_tool_calls.append(
                {
                    "id": tool_call.id,
                    "function": {
                        "name": tool_call.name,
                        "arguments": json.loads(tool_call.arguments),
                    },
                    "type": "function",
                }
            )
        return formatted_tool_calls

    def format_stream_tool_calls(
        self,
        tool_calls: Optional[List[Any]] = None,
    ) -> List[Dict]:
        """
        Formats a OCI GenAI API Meta stream response
        into the tool call format used in Langchain.
        """
        if not tool_calls:
            return []

        formatted_tool_calls: List[Dict] = []
        for tool_call in tool_calls:
            # empty string for fields not present in the tool call
            formatted_tool_calls.append(
                {
                    "id": tool_call.get("id", ""),
                    "function": {
                        "name": tool_call.get("name", ""),
                        "arguments": tool_call.get("arguments", ""),
                    },
                    "type": "function",
                }
            )
        return formatted_tool_calls

    def get_role(self, message: BaseMessage) -> str:
        """Map a LangChain message to Meta's role representation."""
        if isinstance(message, HumanMessage):
            return "USER"
        elif isinstance(message, AIMessage):
            return "ASSISTANT"
        elif isinstance(message, SystemMessage):
            return "SYSTEM"
        elif isinstance(message, ToolMessage):
            return "TOOL"
        raise ValueError(f"Unknown message type: {type(message)}")

    def messages_to_oci_params(
        self, messages: List[BaseMessage], **kwargs: Any
    ) -> Dict[str, Any]:
        """Convert LangChain messages to OCI chat parameters.

        Args:
            messages: List of LangChain BaseMessage objects
            **kwargs: Additional keyword arguments

        Returns:
            Dict containing OCI chat parameters

        Raises:
            ValueError: If message content is invalid
        """
        oci_messages = []

        for message in messages:
            role = self.get_role(message)
            if isinstance(message, ToolMessage):
                # For tool messages, wrap the content in a text content object.
                tool_content = [
                    self.oci_chat_message_text_content(text=str(message.content))
                ]
                if message.tool_call_id:
                    oci_message = self.oci_chat_message[role](
                        content=tool_content,
                        tool_call_id=message.tool_call_id,
                    )
                else:
                    oci_message = self.oci_chat_message[role](content=tool_content)
            elif isinstance(message, AIMessage) and (
                message.tool_calls or message.additional_kwargs.get("tool_calls")
            ):
                # Process content and tool calls for assistant messages
                if message.content:
                    content = self._process_message_content(message.content)
                # Issue 78 fix: Check if original content is empty BEFORE processing
                # to prevent NullPointerException in OCI backend
                else:
                    content = [self.oci_chat_message_text_content(text=".")]
                tool_calls = []
                for tool_call in message.tool_calls:
                    tool_calls.append(
                        self.oci_tool_call(
                            id=tool_call["id"],
                            name=tool_call["name"],
                            arguments=json.dumps(tool_call["args"]),
                        )
                    )
                oci_message = self.oci_chat_message[role](
                    content=content,
                    tool_calls=tool_calls,
                )
            else:
                # For regular messages, process content normally.
                content = self._process_message_content(message.content)
                oci_message = self.oci_chat_message[role](content=content)
            oci_messages.append(oci_message)

        result = {
            "messages": oci_messages,
            "api_format": self.chat_api_format,
        }

        # BUGFIX: Intelligently manage tool_choice to prevent infinite loops
        # while allowing legitimate multi-step tool orchestration.
        # This addresses a known issue with Meta Llama models that
        # continue calling tools even after receiving results.
        has_tool_results = any(isinstance(msg, ToolMessage) for msg in messages)
        if has_tool_results and "tools" in kwargs and "tool_choice" not in kwargs:
            max_tool_calls = kwargs.get("max_sequential_tool_calls", 8)
            if not _should_allow_more_tool_calls(messages, max_tool_calls):
                # Force model to stop and provide final answer
                result["tool_choice"] = self.oci_tool_choice_none()
            # else: Allow model to decide (default behavior)

        # Add parallel tool calls support (GenericChatRequest models)
        if "is_parallel_tool_calls" in kwargs:
            result["is_parallel_tool_calls"] = kwargs["is_parallel_tool_calls"]

        return result

    def _process_message_content(
        self, content: Union[str, List[Union[str, Dict]]]
    ) -> List[Any]:
        """Process message content into OCI chat content format.

        Args:
            content: Message content as string or list

        Returns:
            List of OCI chat content objects

        Raises:
            ValueError: If content format is invalid
        """
        if isinstance(content, str):
            return [self.oci_chat_message_text_content(text=content)]

        if not isinstance(content, list):
            raise ValueError("Message content must be a string or a list of items.")
        processed_content = []
        for item in content:
            if isinstance(item, str):
                processed_content.append(self.oci_chat_message_text_content(text=item))
            elif isinstance(item, dict):
                if "type" not in item:
                    raise ValueError("Dict content item must have a 'type' key.")
                if item["type"] == "image_url":
                    processed_content.append(
                        self.oci_chat_message_image_content(
                            image_url=self.oci_chat_message_image_url(
                                url=item["image_url"]["url"]
                            )
                        )
                    )
                elif item["type"] == "text":
                    processed_content.append(
                        self.oci_chat_message_text_content(text=item["text"])
                    )
                else:
                    raise ValueError(f"Unsupported content type: {item['type']}")
            else:
                raise ValueError(
                    f"Content items must be str or dict, got: {type(item)}"
                )
        return processed_content

    def convert_to_oci_tool(
        self,
        tool: Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool],
    ) -> Dict[str, Any]:
        """Convert a BaseTool instance, TypedDict or BaseModel type
        to a OCI tool in Meta's format.

        Args:
            tool: The tool to convert, can be a BaseTool instance, TypedDict,
                or BaseModel type.

        Returns:
            Dict containing the tool definition in Meta's format.

        Raises:
            ValueError: If the tool type is not supported.
        """
        # Check BaseTool first since it's callable but needs special handling
        if isinstance(tool, BaseTool):
            # Use model_json_schema() if available to preserve json_schema_extra
            # constraints (enum, format, etc.) that convert_to_openai_function strips
            if tool.args_schema and hasattr(tool.args_schema, "model_json_schema"):
                schema = tool.args_schema.model_json_schema()
                parameters = schema
            else:
                as_json_schema_function = convert_to_openai_function(tool)
                parameters = as_json_schema_function.get("parameters", {})

            # Resolve $ref/$defs and anyOf — OCI doesn't support them
            resolved_params = OCIUtils.resolve_schema_refs(parameters)
            resolved_params = OCIUtils.resolve_anyof(resolved_params)
            properties = resolved_params.get("properties", {})

            return self.oci_function_definition(
                name=tool.name,
                description=OCIUtils.remove_signature_from_tool_description(
                    tool.name, tool.description
                ),
                parameters={
                    "type": "object",
                    "properties": properties,
                    "required": resolved_params.get("required", []),
                },
            )
        if (isinstance(tool, type) and issubclass(tool, BaseModel)) or callable(tool):
            as_json_schema_function = convert_to_openai_function(tool)
            parameters = as_json_schema_function.get("parameters", {})
            return self.oci_function_definition(
                name=as_json_schema_function.get("name"),
                description=as_json_schema_function.get(
                    "description",
                    as_json_schema_function.get("name"),
                ),
                parameters={
                    "type": "object",
                    "properties": parameters.get("properties", {}),
                    "required": parameters.get("required", []),
                },
            )
        raise ValueError(
            f"Unsupported tool type {type(tool)}. "
            "Tool must be passed in as a BaseTool "
            "instance, TypedDict class, or BaseModel type."
        )

    def process_tool_choice(
        self,
        tool_choice: Optional[
            Union[dict, str, Literal["auto", "none", "required", "any"], bool]
        ],
    ) -> Optional[Any]:
        """Process tool choice for Meta provider.

        Args:
            tool_choice: Which tool to require the model to call. Options are:
                - str of the form "<<tool_name>>": calls <<tool_name>> tool.
                - "auto": automatically selects a tool (including no tool).
                - "none": does not call a tool.
                - "any" or "required" or True: force at least one tool to be called.
                - dict of the form
                    {"type": "function", "function": {"name": <<tool_name>>}}:
                calls <<tool_name>> tool.
                - False or None: no effect, default Meta behavior.

        Returns:
            Meta-specific tool choice object.

        Raises:
            ValueError: If tool_choice type is not recognized.
        """
        if tool_choice is None:
            return None

        if isinstance(tool_choice, str):
            if tool_choice not in ("auto", "none", "any", "required"):
                return self.oci_tool_choice_function(name=tool_choice)
            elif tool_choice == "auto":
                return self.oci_tool_choice_auto()
            elif tool_choice == "none":
                return self.oci_tool_choice_none()
            elif tool_choice in ("any", "required"):
                return self.oci_tool_choice_required()
        elif isinstance(tool_choice, bool):
            if tool_choice:
                return self.oci_tool_choice_required()
            else:
                return self.oci_tool_choice_none()
        elif isinstance(tool_choice, dict):
            # For Meta, we use ToolChoiceAuto for tool selection
            return self.oci_tool_choice_auto()
        raise ValueError(
            f"Unrecognized tool_choice type. Expected str, bool or dict. "
            f"Received: {tool_choice}"
        )

    def process_stream_tool_calls(
        self, event_data: Dict, tool_call_ids: Dict[int, str]
    ) -> List[ToolCallChunk]:
        """
        Process Meta stream tool calls and convert them to ToolCallChunks.

        Args:
            event_data: The event data from the stream
            tool_call_ids: Dict mapping tool call index to ID for aggregation

        Returns:
            List of ToolCallChunk objects
        """
        tool_call_chunks: List[ToolCallChunk] = []
        tool_call_response = self.chat_stream_tool_calls(event_data)

        if not tool_call_response:
            return tool_call_chunks

        for idx, tool_call in enumerate(
            self.format_stream_tool_calls(tool_call_response)
        ):
            tool_id = tool_call.get("id")

            if tool_id:
                if idx not in tool_call_ids or tool_call_ids[idx] == tool_id:
                    # New idx - use idx as index
                    # Same ID at same idx - reuse idx as index
                    index = idx
                else:
                    # Different ID at same idx - parallel tool call (Grok pattern)
                    # New idx - use len(tool_call_ids) as index
                    index = len(tool_call_ids)
            elif idx in tool_call_ids:
                # Subsequent chunk - reuse stored ID (gpt-oss pattern)
                index = idx
                tool_id = tool_call_ids[index]
            else:
                # No ID and no stored ID - generate new one (e.g., Gemini models)
                tool_id = str(uuid.uuid4())
                index = idx
            tool_call_ids[index] = tool_id

            tool_call_chunks.append(
                tool_call_chunk(
                    name=tool_call["function"].get("name"),
                    args=tool_call["function"].get("arguments"),
                    id=tool_id,
                    index=index,
                )
            )
        return tool_call_chunks


class MetaProvider(GenericProvider):
    """Provider for Meta models. This provider is for backward compatibility."""

    pass
