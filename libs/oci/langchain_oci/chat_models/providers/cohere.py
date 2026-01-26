# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Cohere provider implementation for OCI Generative AI."""

import json
import uuid
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Set,
    Type,
    Union,
)

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
from langchain_oci.common.utils import JSON_TO_PYTHON_TYPES, OCIUtils


class CohereProvider(Provider):
    """Provider implementation for Cohere."""

    stop_sequence_key: str = "stop_sequences"

    def __init__(self) -> None:
        from oci.generative_ai_inference import models

        self.oci_chat_request = models.CohereChatRequest
        self.oci_chat_request_v2 = models.CohereChatRequestV2
        self.oci_tool = models.CohereTool
        self.oci_tool_param = models.CohereParameterDefinition
        self.oci_tool_result = models.CohereToolResult
        self.oci_tool_call = models.CohereToolCall
        self.oci_chat_message = {
            "USER": models.CohereUserMessage,
            "CHATBOT": models.CohereChatBotMessage,
            "SYSTEM": models.CohereSystemMessage,
            "TOOL": models.CohereToolMessage,
        }
        # V2 API message classes for vision support
        self.oci_chat_message_v2 = {
            "USER": models.CohereUserMessageV2,
            "ASSISTANT": models.CohereAssistantMessageV2,
            "SYSTEM": models.CohereSystemMessageV2,
            "TOOL": models.CohereToolMessageV2,
        }
        self.oci_text_content_v2 = models.CohereTextContentV2
        self.oci_image_content_v2 = models.CohereImageContentV2
        self.oci_image_url_v2 = models.CohereImageUrlV2

        self.oci_response_json_schema = models.ResponseJsonSchema
        self.oci_json_schema_response_format = models.JsonSchemaResponseFormat
        self.chat_api_format = models.BaseChatRequest.API_FORMAT_COHERE
        self.chat_api_format_v2 = models.CohereChatRequestV2.API_FORMAT_COHEREV2

    def chat_response_to_text(self, response: Any) -> str:
        """Extract text from a Cohere chat response."""
        return response.data.chat_response.text

    def chat_stream_to_text(self, event_data: Dict) -> str:
        """Extract text from a Cohere chat stream event."""
        if "text" in event_data:
            # Return empty string if finish reason or tool calls are present in stream
            if "finishReason" in event_data or "toolCalls" in event_data:
                return ""
            else:
                return event_data["text"]
        return ""

    def is_chat_stream_end(self, event_data: Dict) -> bool:
        """Determine if the Cohere stream event indicates the end."""
        return "finishReason" in event_data

    def chat_generation_info(self, response: Any) -> Dict[str, Any]:
        """Extract generation information from a Cohere chat response."""
        generation_info: Dict[str, Any] = {
            "documents": response.data.chat_response.documents,
            "citations": response.data.chat_response.citations,
            "search_queries": response.data.chat_response.search_queries,
            "is_search_required": response.data.chat_response.is_search_required,
            "finish_reason": response.data.chat_response.finish_reason,
        }

        # Include token usage if available
        if (
            hasattr(response.data.chat_response, "usage")
            and response.data.chat_response.usage
        ):
            generation_info["total_tokens"] = (
                response.data.chat_response.usage.total_tokens
            )

        # Include tool calls if available
        if self.chat_tool_calls(response):
            generation_info["tool_calls"] = self.format_response_tool_calls(
                self.chat_tool_calls(response)
            )
        return generation_info

    def chat_stream_generation_info(self, event_data: Dict) -> Dict[str, Any]:
        """Extract generation info from a Cohere chat stream event."""
        generation_info: Dict[str, Any] = {
            "documents": event_data.get("documents"),
            "citations": event_data.get("citations"),
            "finish_reason": event_data.get("finishReason"),
        }
        # Remove keys with None values
        return {k: v for k, v in generation_info.items() if v is not None}

    def chat_tool_calls(self, response: Any) -> List[Any]:
        """Retrieve tool calls from a Cohere chat response."""
        return response.data.chat_response.tool_calls

    def chat_stream_tool_calls(self, event_data: Dict) -> List[Any]:
        """Retrieve tool calls from Cohere stream event data."""
        return event_data.get("toolCalls", [])

    def format_response_tool_calls(
        self,
        tool_calls: Optional[List[Any]] = None,
    ) -> List[Dict]:
        """
        Formats a OCI GenAI API Cohere response
        into the tool call format used in Langchain.
        """
        if not tool_calls:
            return []

        formatted_tool_calls: List[Dict] = []
        for tool_call in tool_calls:
            formatted_tool_calls.append(
                {
                    "id": uuid.uuid4().hex[:],
                    "function": {
                        "name": tool_call.name,
                        "arguments": json.dumps(tool_call.parameters),
                    },
                    "type": "function",
                }
            )
        return formatted_tool_calls

    def format_stream_tool_calls(self, tool_calls: List[Any]) -> List[Dict]:
        """
        Formats a OCI GenAI API Cohere stream response
        into the tool call format used in Langchain.
        """
        if not tool_calls:
            return []

        formatted_tool_calls: List[Dict] = []
        for tool_call in tool_calls:
            formatted_tool_calls.append(
                {
                    "id": uuid.uuid4().hex[:],
                    "function": {
                        "name": tool_call["name"],
                        "arguments": json.dumps(tool_call["parameters"]),
                    },
                    "type": "function",
                }
            )
        return formatted_tool_calls

    def get_role(self, message: BaseMessage) -> str:
        """Map a LangChain message to Cohere's role representation."""
        if isinstance(message, HumanMessage):
            return "USER"
        elif isinstance(message, AIMessage):
            return "CHATBOT"
        elif isinstance(message, SystemMessage):
            return "SYSTEM"
        elif isinstance(message, ToolMessage):
            return "TOOL"
        raise ValueError(f"Unknown message type: {type(message)}")

    def get_role_v2(self, message: BaseMessage) -> str:
        """Map a LangChain message to Cohere V2's role representation."""
        if isinstance(message, HumanMessage):
            return "USER"
        elif isinstance(message, AIMessage):
            return "ASSISTANT"
        elif isinstance(message, SystemMessage):
            return "SYSTEM"
        elif isinstance(message, ToolMessage):
            return "TOOL"
        raise ValueError(f"Unknown message type: {type(message)}")

    def _has_vision_content(self, messages: Sequence[BaseMessage]) -> bool:
        """Check if any message contains image content."""
        for msg in messages:
            if isinstance(msg, HumanMessage) and isinstance(msg.content, list):
                for block in msg.content:
                    if isinstance(block, dict) and block.get("type") == "image_url":
                        return True
        return False

    def _content_to_v2(self, content: Union[str, List]) -> List[Any]:
        """Convert LangChain message content to Cohere V2 content format."""
        from oci.generative_ai_inference import models

        if isinstance(content, str):
            return [self.oci_text_content_v2(type=models.CohereContentV2.TYPE_TEXT, text=content)]

        v2_content = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    v2_content.append(
                        self.oci_text_content_v2(type=models.CohereContentV2.TYPE_TEXT, text=block["text"])
                    )
                elif block.get("type") == "image_url":
                    image_url = block.get("image_url", {})
                    url = image_url.get("url") if isinstance(image_url, dict) else image_url
                    v2_content.append(
                        self.oci_image_content_v2(
                            type=models.CohereContentV2.TYPE_IMAGE_URL,
                            image_url=self.oci_image_url_v2(url=url),
                        )
                    )
            elif isinstance(block, str):
                v2_content.append(self.oci_text_content_v2(type=models.CohereContentV2.TYPE_TEXT, text=block))
        return v2_content

    def _messages_to_oci_params_v2(
        self, messages: Sequence[BaseMessage], **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Convert LangChain messages to OCI parameters for Cohere V2 API (vision support).
        """
        v2_messages = []

        for msg in messages:
            role = self.get_role_v2(msg)
            if isinstance(msg, (HumanMessage, SystemMessage)):
                content = self._content_to_v2(msg.content)
                v2_messages.append(
                    self.oci_chat_message_v2[role](role=role, content=content)
                )
            elif isinstance(msg, AIMessage):
                # For AI messages, include text content
                content = self._content_to_v2(msg.content if msg.content else " ")
                v2_messages.append(
                    self.oci_chat_message_v2[role](role=role, content=content)
                )
            # Note: Tool messages not yet fully implemented for V2

        oci_params = {
            "messages": v2_messages,
            "api_format": self.chat_api_format_v2,
            "_use_v2_api": True,  # Flag to indicate V2 API should be used
        }
        return {k: v for k, v in oci_params.items() if v is not None}

    def messages_to_oci_params(
        self, messages: Sequence[BaseMessage], **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Convert LangChain messages to OCI parameters for Cohere.

        This includes conversion of chat history and tool call results.
        """
        # Check if vision content is present - if so, use V2 API
        if self._has_vision_content(messages):
            return self._messages_to_oci_params_v2(messages, **kwargs)

        # Cohere models don't support parallel tool calls
        if kwargs.get("is_parallel_tool_calls"):
            raise ValueError(
                "Parallel tool calls are not supported for Cohere models. "
                "This feature is only available for models using GenericChatRequest "
                "(Meta, Llama, xAI Grok, OpenAI, Mistral)."
            )

        is_force_single_step = kwargs.get("is_force_single_step", False)
        oci_chat_history = []

        # Process all messages except the last one for chat history
        for msg in messages[:-1]:
            role = self.get_role(msg)
            if role in ("USER", "SYSTEM"):
                oci_chat_history.append(
                    self.oci_chat_message[role](message=msg.content)
                )
            elif isinstance(msg, AIMessage):
                # Skip tool calls if forcing single step
                if msg.tool_calls and is_force_single_step:
                    continue
                tool_calls = (
                    [
                        self.oci_tool_call(name=tc["name"], parameters=tc["args"])
                        for tc in msg.tool_calls
                    ]
                    if msg.tool_calls
                    else None
                )
                msg_content = msg.content if msg.content else " "
                oci_chat_history.append(
                    self.oci_chat_message[role](
                        message=msg_content, tool_calls=tool_calls
                    )
                )
            elif isinstance(msg, ToolMessage):
                oci_chat_history.append(
                    self.oci_chat_message[self.get_role(msg)](
                        tool_results=[
                            self.oci_tool_result(
                                call=self.oci_tool_call(name=msg.name, parameters={}),
                                outputs=[{"output": msg.content}],
                            )
                        ],
                    )
                )

        # Process current turn messages in reverse order until a HumanMessage
        current_turn = []
        for i, message in enumerate(messages[::-1]):
            current_turn.append(message)
            if isinstance(message, HumanMessage):
                if len(messages) > i and isinstance(
                    messages[len(messages) - i - 2], ToolMessage
                ):
                    # add dummy message REPEATING the tool_result to avoid
                    # the error about ToolMessage needing to be followed
                    # by an AI message
                    oci_chat_history.append(
                        self.oci_chat_message["CHATBOT"](
                            message=messages[len(messages) - i - 2].content
                        )
                    )
                break
        current_turn = list(reversed(current_turn))

        # Process tool results from the current turn
        oci_tool_results: Optional[List[Any]] = []
        for message in current_turn:
            if isinstance(message, ToolMessage):
                tool_msg = message
                previous_ai_msgs = [
                    m for m in current_turn if isinstance(m, AIMessage) and m.tool_calls
                ]
                if previous_ai_msgs:
                    previous_ai_msg = previous_ai_msgs[-1]
                    for lc_tool_call in previous_ai_msg.tool_calls:
                        if lc_tool_call["id"] == tool_msg.tool_call_id:
                            tool_result = self.oci_tool_result()
                            tool_result.call = self.oci_tool_call(
                                name=lc_tool_call["name"],
                                parameters=lc_tool_call["args"],
                            )
                            tool_result.outputs = [{"output": tool_msg.content}]
                            oci_tool_results.append(tool_result)  # type: ignore[union-attr]
        if not oci_tool_results:
            oci_tool_results = None

        # Use last message's content if no tool results are present
        message_str = "" if oci_tool_results else messages[-1].content

        oci_params = {
            "message": message_str,
            "chat_history": oci_chat_history,
            "tool_results": oci_tool_results,
            "api_format": self.chat_api_format,
        }
        # Remove keys with None values
        return {k: v for k, v in oci_params.items() if v is not None}

    def convert_to_oci_tool(
        self,
        tool: Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool],
    ) -> Dict[str, Any]:
        """
        Convert a tool definition to an OCI tool for Cohere.

        Supports BaseTool instances, JSON schema dictionaries,
        or Pydantic models/callables.
        """
        if isinstance(tool, BaseTool):
            return self.oci_tool(
                name=tool.name,
                description=OCIUtils.remove_signature_from_tool_description(
                    tool.name, tool.description
                ),
                parameter_definitions={
                    p_name: self.oci_tool_param(
                        description=p_def.get("description", ""),
                        type=JSON_TO_PYTHON_TYPES.get(
                            p_def.get("type"),
                            p_def.get("type", "any"),
                        ),
                        is_required="default" not in p_def,
                    )
                    for p_name, p_def in tool.args.items()
                },
            )
        elif isinstance(tool, dict):
            if not all(k in tool for k in ("title", "description", "properties")):
                raise ValueError(
                    "Unsupported dict type. Tool must be a BaseTool instance, "
                    "JSON schema dict, or Pydantic model."
                )
            return self.oci_tool(
                name=tool.get("title"),
                description=tool.get("description"),
                parameter_definitions={
                    p_name: self.oci_tool_param(
                        description=p_def.get("description", ""),
                        type=JSON_TO_PYTHON_TYPES.get(
                            p_def.get("type"),
                            p_def.get("type", "any"),
                        ),
                        is_required="default" not in p_def,
                    )
                    for p_name, p_def in tool.get("properties", {}).items()
                },
            )
        elif (isinstance(tool, type) and issubclass(tool, BaseModel)) or callable(tool):
            as_json_schema_function = convert_to_openai_function(tool)
            parameters = as_json_schema_function.get("parameters", {})
            properties = parameters.get("properties", {})
            return self.oci_tool(
                name=as_json_schema_function.get("name"),
                description=as_json_schema_function.get(
                    "description",
                    as_json_schema_function.get("name"),
                ),
                parameter_definitions={
                    p_name: self.oci_tool_param(
                        description=p_def.get("description", ""),
                        type=JSON_TO_PYTHON_TYPES.get(
                            p_def.get("type"),
                            p_def.get("type", "any"),
                        ),
                        is_required=p_name in parameters.get("required", []),
                    )
                    for p_name, p_def in properties.items()
                },
            )
        raise ValueError(
            f"Unsupported tool type {type(tool)}. Must be BaseTool instance, "
            "JSON schema dict, or Pydantic model."
        )

    def process_tool_choice(
        self,
        tool_choice: Optional[
            Union[dict, str, Literal["auto", "none", "required", "any"], bool]
        ],
    ) -> Optional[Any]:
        """Cohere does not support tool choices."""
        if tool_choice is not None:
            raise ValueError(
                "Tool choice is not supported for Cohere models."
                "Please remove the tool_choice parameter."
            )
        return None

    def process_stream_tool_calls(
        self, event_data: Dict, tool_call_ids: Set[str]
    ) -> List[ToolCallChunk]:
        """
        Process Cohere stream tool calls and return them as ToolCallChunk objects.

        Args:
            event_data: The event data from the stream
            tool_call_ids: Set of existing tool call IDs for index tracking

        Returns:
            List of ToolCallChunk objects
        """
        tool_call_chunks: List[ToolCallChunk] = []
        tool_call_response = self.chat_stream_tool_calls(event_data)

        if not tool_call_response:
            return tool_call_chunks

        for tool_call in self.format_stream_tool_calls(tool_call_response):
            tool_id = tool_call.get("id")
            if tool_id:
                tool_call_ids.add(tool_id)

            tool_call_chunks.append(
                tool_call_chunk(
                    name=tool_call["function"].get("name"),
                    args=tool_call["function"].get("arguments"),
                    id=tool_id,
                    index=len(tool_call_ids) - 1,  # index tracking
                )
            )
        return tool_call_chunks
