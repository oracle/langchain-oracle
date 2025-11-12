# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import json
import re
import uuid
from abc import ABC, abstractmethod
from operator import itemgetter
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Set,
    Type,
    Union,
)

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import (
    BaseChatModel,
    generate_from_stream,
)
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.messages.tool import ToolCallChunk, tool_call_chunk
from langchain_core.output_parsers import (
    JsonOutputParser,
    PydanticOutputParser,
)
from langchain_core.output_parsers.base import OutputParserLike
from langchain_core.output_parsers.openai_tools import (
    JsonOutputKeyToolsParser,
    PydanticToolsParser,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable, RunnableMap, RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_function
from pydantic import BaseModel, ConfigDict

from langchain_oci.llms.oci_generative_ai import OCIGenAIBase
from langchain_oci.llms.utils import enforce_stop_tokens

CUSTOM_ENDPOINT_PREFIX = "ocid1.generativeaiendpoint"

# Mapping of JSON schema types to Python types
JSON_TO_PYTHON_TYPES = {
    "string": "str",
    "number": "float",
    "boolean": "bool",
    "integer": "int",
    "array": "List",
    "object": "Dict",
    "any": "any",
}


class OCIUtils:
    """Utility functions for OCI Generative AI integration."""

    @staticmethod
    def is_pydantic_class(obj: Any) -> bool:
        """Check if an object is a Pydantic BaseModel subclass."""
        return isinstance(obj, type) and issubclass(obj, BaseModel)

    @staticmethod
    def remove_signature_from_tool_description(name: str, description: str) -> str:
        """
        Remove the tool signature and Args section from a tool description.

        The signature is typically prefixed to the description and followed

        by an Args section.
        """
        description = re.sub(rf"^{name}\(.*?\) -(?:> \w+? -)? ", "", description)
        description = re.sub(r"(?s)(?:\n?\n\s*?)?Args:.*$", "", description)
        return description

    @staticmethod
    def convert_oci_tool_call_to_langchain(tool_call: Any) -> ToolCall:
        """Convert an OCI tool call to a LangChain ToolCall."""
        parsed = json.loads(tool_call.arguments)

        # If the parsed result is a string, it means the JSON was escaped, so parse again
        if isinstance(parsed, str):
            try:
                parsed = json.loads(parsed)
            except json.JSONDecodeError:
                # If it's not valid JSON, keep it as a string
                pass

        return ToolCall(
            name=tool_call.name,
            args=parsed
            if "arguments" in tool_call.attribute_map
            else tool_call.parameters,
            id=tool_call.id if "id" in tool_call.attribute_map else uuid.uuid4().hex[:],
        )


class Provider(ABC):
    """Abstract base class for OCI Generative AI providers."""

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
        tool_call_ids: Set[str],
    ) -> List[ToolCallChunk]:
        """Process streaming tool calls from event data into chunks."""
        ...


class CohereProvider(Provider):
    """Provider implementation for Cohere."""

    stop_sequence_key: str = "stop_sequences"

    def __init__(self) -> None:
        from oci.generative_ai_inference import models

        self.oci_chat_request = models.CohereChatRequest
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

        self.oci_response_json_schema = models.ResponseJsonSchema
        self.oci_json_schema_response_format = models.JsonSchemaResponseFormat
        self.chat_api_format = models.BaseChatRequest.API_FORMAT_COHERE

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
        if hasattr(response.data.chat_response, "usage") and response.data.chat_response.usage:
            generation_info["total_tokens"] = response.data.chat_response.usage.total_tokens

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

    def messages_to_oci_params(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Convert LangChain messages to OCI parameters for Cohere.

        This includes conversion of chat history and tool call results.
        """
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
        oci_tool_results: List[Any] = []
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
                            oci_tool_results.append(tool_result)
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
                    "Unsupported dict type. Tool must be a BaseTool instance, JSON schema dict, or Pydantic model."  # noqa: E501
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
            f"Unsupported tool type {type(tool)}. Must be BaseTool instance, JSON schema dict, or Pydantic model."  # noqa: E501
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
        tool_call_chunks = []
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


class GenericProvider(Provider):
    """Provider for models using generic API spec."""

    stop_sequence_key: str = "stop"

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
        """Extract text from Meta chat response."""
        message = response.data.chat_response.choices[0].message
        content = message.content[0] if message.content else None
        return content.text if content else ""

    def chat_stream_to_text(self, event_data: Dict) -> str:
        """Extract text from Meta chat stream event."""
        content = event_data.get("message", {}).get("content", None)
        if not content:
            return ""
        return content[0]["text"]

    def is_chat_stream_end(self, event_data: Dict) -> bool:
        """Determine if Meta chat stream event indicates the end."""
        return "finishReason" in event_data

    def chat_generation_info(self, response: Any) -> Dict[str, Any]:
        """Extract generation metadata from Meta chat response."""
        generation_info: Dict[str, Any] = {
            "finish_reason": response.data.chat_response.choices[0].finish_reason,
            "time_created": str(response.data.chat_response.time_created),
        }

        # Include token usage if available
        if hasattr(response.data.chat_response, "usage") and response.data.chat_response.usage:
            generation_info["total_tokens"] = response.data.chat_response.usage.total_tokens
            
        if self.chat_tool_calls(response):
            generation_info["tool_calls"] = self.format_response_tool_calls(
                self.chat_tool_calls(response)
            )
        return generation_info

    def chat_stream_generation_info(self, event_data: Dict) -> Dict[str, Any]:
        """Extract generation metadata from Meta chat stream event."""
        return {"finish_reason": event_data["finishReason"]}

    def chat_tool_calls(self, response: Any) -> List[Any]:
        """Retrieve tool calls from Meta chat response."""
        return response.data.chat_response.choices[0].message.tool_calls

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
                content = self._process_message_content(message.content)
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

        def _should_allow_more_tool_calls(
            messages: List[BaseMessage],
            max_tool_calls: int
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
            tool_call_count = sum(
                1 for msg in messages
                if isinstance(msg, ToolMessage)
            )

            # Safety limit: prevent runaway tool calling
            if tool_call_count >= max_tool_calls:
                return False

            # Detect infinite loop: same tool called with same arguments in succession
            recent_calls = []
            for msg in reversed(messages):
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tc in msg.tool_calls:
                        # Create signature: (tool_name, sorted_args)
                        try:
                            args_str = json.dumps(tc.get('args', {}), sort_keys=True)
                            signature = (tc.get('name', ''), args_str)

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

        has_tool_results = any(isinstance(msg, ToolMessage) for msg in messages)
        if has_tool_results and "tools" in kwargs and "tool_choice" not in kwargs:
            max_tool_calls = kwargs.get("max_sequential_tool_calls", 8)
            if not _should_allow_more_tool_calls(messages, max_tool_calls):
                # Force model to stop and provide final answer
                result["tool_choice"] = self.oci_tool_choice_none()
            # else: Allow model to decide (default behavior)

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
        tool: Union[Type[BaseModel], Callable, BaseTool],
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
        elif isinstance(tool, BaseTool):
            return self.oci_function_definition(
                name=tool.name,
                description=OCIUtils.remove_signature_from_tool_description(
                    tool.name, tool.description
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        p_name: {
                            "type": p_def.get("type", "any"),
                            "description": p_def.get("description", ""),
                        }
                        for p_name, p_def in tool.args.items()
                    },
                    "required": [
                        p_name
                        for p_name, p_def in tool.args.items()
                        if "default" not in p_def
                    ],
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
        self, event_data: Dict, tool_call_ids: Set[str]
    ) -> List[ToolCallChunk]:
        """
        Process Meta stream tool calls and convert them to ToolCallChunks.

        Args:
            event_data: The event data from the stream
            tool_call_ids: Set of existing tool call IDs for index tracking

        Returns:
            List of ToolCallChunk objects
        """
        tool_call_chunks = []
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


class MetaProvider(GenericProvider):
    """Provider for Meta models. This provider is for backward compatibility."""

    pass


class ChatOCIGenAI(BaseChatModel, OCIGenAIBase):
    """ChatOCIGenAI chat model integration.

    Setup:
      Install ``langchain-oci`` and the ``oci`` sdk.

      .. code-block:: bash

          pip install -U langchain-oci oci

    Key init args — completion params:
        model_id: str
            Id of the OCIGenAI chat model to use, e.g., cohere.command-r-16k.
        is_stream: bool
            Whether to stream back partial progress
        model_kwargs: Optional[Dict]
            Keyword arguments to pass to the specific model used, e.g., temperature, max_tokens.

    Key init args — client params:
        service_endpoint: str
            The endpoint URL for the OCIGenAI service, e.g., https://inference.generativeai.us-chicago-1.oci.oraclecloud.com.
        compartment_id: str
            The compartment OCID.
        auth_type: str
            The authentication type to use, e.g., API_KEY (default), SECURITY_TOKEN, INSTANCE_PRINCIPAL, RESOURCE_PRINCIPAL.
        auth_profile: Optional[str]
            The name of the profile in ~/.oci/config, if not specified , DEFAULT will be used.
        auth_file_location: Optional[str]
            Path to the config file, If not specified, ~/.oci/config will be used.
        provider: str
            Provider name of the model. Default to None, will try to be derived from the model_id otherwise, requires user input.
    See full list of supported init args and their descriptions in the params section.

    Instantiate:
        .. code-block:: python

            from langchain_oci.chat_models import ChatOCIGenAI

            chat = ChatOCIGenAI(
                model_id="cohere.command-r-16k",
                service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
                compartment_id="MY_OCID",
                model_kwargs={"temperature": 0.7, "max_tokens": 500},
            )

    Invoke:
        .. code-block:: python
            messages = [
                SystemMessage(content="your are an AI assistant."),
                AIMessage(content="Hi there human!"),
                HumanMessage(content="tell me a joke."),
            ]
            response = chat.invoke(messages)

    Stream:
        .. code-block:: python

        for r in chat.stream(messages):
            print(r.content, end="", flush=True)

    Response metadata
        .. code-block:: python

        response = chat.invoke(messages)
        print(response.response_metadata)

    """  # noqa: E501

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    @property
    def _llm_type(self) -> str:
        """Return the type of the language model."""
        return "oci_generative_ai_chat"

    @property
    def _provider_map(self) -> Mapping[str, Provider]:
        """Mapping from provider name to provider instance."""
        return {
            "cohere": CohereProvider(),
            "meta": MetaProvider(),
            "generic": GenericProvider(),
        }

    @property
    def _provider(self) -> Any:
        """Get the internal provider object"""
        return self._get_provider(provider_map=self._provider_map)

    def _prepare_request(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]],
        stream: bool,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Prepare the OCI chat request from LangChain messages.

        This method consolidates model kwargs, stop tokens, and message history.
        """
        try:
            from oci.generative_ai_inference import models

        except ImportError as ex:
            raise ModuleNotFoundError(
                "Could not import oci python package. "
                "Please make sure you have the oci package installed."
            ) from ex

        oci_params = self._provider.messages_to_oci_params(
            messages,
            max_sequential_tool_calls=self.max_sequential_tool_calls,
            **kwargs
        )

        oci_params["is_stream"] = stream
        _model_kwargs = self.model_kwargs or {}

        if stop is not None:
            _model_kwargs[self._provider.stop_sequence_key] = stop

        # Warn if using max_tokens with OpenAI models
        if self.model_id and self.model_id.startswith("openai.") and "max_tokens" in _model_kwargs:
            import warnings
            warnings.warn(
                f"OpenAI models require 'max_completion_tokens' instead of 'max_tokens'.",
                UserWarning,
                stacklevel=2
            )

        chat_params = {**_model_kwargs, **kwargs, **oci_params}

        if not self.model_id:
            raise ValueError("Model ID is required for chat.")
        if self.model_id.startswith(CUSTOM_ENDPOINT_PREFIX):
            serving_mode = models.DedicatedServingMode(endpoint_id=self.model_id)
        else:
            serving_mode = models.OnDemandServingMode(model_id=self.model_id)

        request = models.ChatDetails(
            compartment_id=self.compartment_id,
            serving_mode=serving_mode,
            chat_request=self._provider.oci_chat_request(**chat_params),
        )

        return request

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]],
        *,
        tool_choice: Optional[
            Union[dict, str, Literal["auto", "none", "required", "any"], bool]
        ] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tool-like objects to this chat model.

        Assumes model is compatible with Meta's tool-calling API.

        Args:
            tools: A list of tool definitions to bind to this chat model.
                Can be a dictionary, pydantic model, or callable. Pydantic
                models and callables will be automatically converted to
                their schema dictionary representation.
            tool_choice: Which tool to require the model to call. Options are:
                - str of the form "<<tool_name>>": calls <<tool_name>> tool.
                - "auto": automatically selects a tool (including no tool).
                - "none": does not call a tool.
                - "any" or "required" or True: force at least one tool to be called.
                - dict of the form
                    {"type": "function", "function": {"name": <<tool_name>>}}:
                calls <<tool_name>> tool.
                - False or None: no effect, default Meta behavior.
            kwargs: Any additional parameters are passed directly to
                :meth:`~langchain_oci.chat_models.oci_generative_ai.ChatOCIGenAI.bind`.
        """

        formatted_tools = [self._provider.convert_to_oci_tool(tool) for tool in tools]

        if tool_choice is not None:
            kwargs["tool_choice"] = self._provider.process_tool_choice(tool_choice)

        return super().bind(tools=formatted_tools, **kwargs)

    def with_structured_output(
        self,
        schema: Optional[Union[Dict, Type[BaseModel]]] = None,
        *,
        method: Literal[
            "function_calling", "json_schema", "json_mode"
        ] = "function_calling",
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, Union[Dict, BaseModel]]:
        """Model wrapper that returns outputs formatted to match the given schema.

        Args:
            schema: The output schema as a dict or a Pydantic class. If a Pydantic class
                then the model output will be an object of that class. If a dict then
                the model output will be a dict. With a Pydantic class the returned
                attributes will be validated, whereas with a dict they will not be. If
                `method` is "function_calling" and `schema` is a dict, then the dict
                must match the OCI Generative AI function-calling spec.
            method:
                The method for steering model generation, either "function_calling" (default method)
                or "json_mode" or "json_schema". If "function_calling" then the schema
                will be converted to an OCI function and the returned model will make
                use of the function-calling API. If "json_mode" then Cohere's JSON mode will be
                used. Note that if using "json_mode" then you must include instructions
                for formatting the output into the desired schema into the model call.
                If "json_schema" then it allows the user to pass a json schema (or pydantic)
                to the model for structured output. 
            include_raw:
                If False then only the parsed structured output is returned. If
                an error occurs during model output parsing it will be raised. If True
                then both the raw model response (a BaseMessage) and the parsed model
                response will be returned. If an error occurs during output parsing it
                will be caught and returned as well. The final output is always a dict
                with keys "raw", "parsed", and "parsing_error".

        Returns:
            A Runnable that takes any ChatModel input and returns as output:

                If include_raw is True then a dict with keys:
                    raw: BaseMessage
                    parsed: Optional[_DictOrPydantic]
                    parsing_error: Optional[BaseException]

                If include_raw is False then just _DictOrPydantic is returned,
                where _DictOrPydantic depends on the schema:

                If schema is a Pydantic class then _DictOrPydantic is the Pydantic
                    class.

                If schema is a dict then _DictOrPydantic is a dict.

        """  # noqa: E501
        if kwargs:
            raise ValueError(f"Unsupported arguments: {kwargs}")
        is_pydantic_schema = OCIUtils.is_pydantic_class(schema)
        if method == "function_calling":
            if schema is None:
                raise ValueError("Schema must be provided for function_calling method.")
            llm = self.bind_tools([schema], **kwargs)
            tool_name = getattr(self._provider.convert_to_oci_tool(schema), "name")
            if is_pydantic_schema:
                output_parser: OutputParserLike = PydanticToolsParser(
                    tools=[schema],  # type: ignore[list-item]
                    first_tool_only=True,  # type: ignore[list-item]
                )
            else:
                output_parser = JsonOutputKeyToolsParser(
                    key_name=tool_name, first_tool_only=True
                )
        elif method == "json_mode":
            llm = self.bind(response_format={"type": "JSON_OBJECT"})
            output_parser = (
                PydanticOutputParser(pydantic_object=schema)  # type: ignore[type-var, arg-type]
                if is_pydantic_schema
                else JsonOutputParser()
            )
        elif method == "json_schema":
            json_schema_dict = (
                schema.model_json_schema()  # type: ignore[union-attr]
                if is_pydantic_schema
                else schema
            )
            
            response_json_schema = self._provider.oci_response_json_schema(
                name=json_schema_dict.get("title", "response"),
                description=json_schema_dict.get("description", ""),
                schema=json_schema_dict,
                is_strict=True
            )
            
            response_format_obj = self._provider.oci_json_schema_response_format(
                json_schema=response_json_schema
            )
            
            llm = self.bind(response_format=response_format_obj)
            if is_pydantic_schema:
                output_parser = PydanticOutputParser(pydantic_object=schema)
            else:
                output_parser = JsonOutputParser()
        else:
            raise ValueError(
                f"Unrecognized method argument. "
                f"Expected `function_calling`, `json_schema` or `json_mode`."
                f"Received: `{method}`."
            )
        if include_raw:
            parser_assign = RunnablePassthrough.assign(
                parsed=itemgetter("raw") | output_parser, parsing_error=lambda _: None
            )
            parser_none = RunnablePassthrough.assign(parsed=lambda _: None)
            parser_with_fallback = parser_assign.with_fallbacks(
                [parser_none], exception_key="parsing_error"
            )
            return RunnableMap(raw=llm) | parser_with_fallback
        return llm | output_parser

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Call out to a OCIGenAI chat model.

        Args:
            messages: list of LangChain messages
            stop: Optional list of stop words to use.

        Returns:
            LangChain ChatResult

        Example:
            .. code-block:: python

               messages = [
                            HumanMessage(content="hello!"),
                            AIMessage(content="Hi there human!"),
                            HumanMessage(content="Meow!")
                          ]

               response = llm.invoke(messages)
        """
        if self.is_stream:
            stream_iter = self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return generate_from_stream(stream_iter)

        request = self._prepare_request(messages, stop=stop, stream=False, **kwargs)
        response = self.client.chat(request)

        content = self._provider.chat_response_to_text(response)

        if stop is not None:
            content = enforce_stop_tokens(content, stop)

        generation_info = self._provider.chat_generation_info(response)

        llm_output = {
            "model_id": response.data.model_id,
            "model_version": response.data.model_version,
            "request_id": response.request_id,
            "content-length": response.headers["content-length"],
        }
        tool_calls = []
        if "tool_calls" in generation_info:
            tool_calls = [
                OCIUtils.convert_oci_tool_call_to_langchain(tool_call)
                for tool_call in self._provider.chat_tool_calls(response)
            ]
        message = AIMessage(
            content=content or "",
            additional_kwargs=generation_info,
            tool_calls=tool_calls,
        )
        return ChatResult(
            generations=[
                ChatGeneration(message=message, generation_info=generation_info)
            ],
            llm_output=llm_output,
        )

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """
        Stream chat responses from OCI.

        Processes each event and yields chunks until the stream ends.
        """
        request = self._prepare_request(messages, stop=stop, stream=True, **kwargs)
        response = self.client.chat(request)
        tool_call_ids: Set[str] = set()

        for event in response.data.events():
            event_data = json.loads(event.data)

            if not self._provider.is_chat_stream_end(event_data):
                # Process streaming content
                delta = self._provider.chat_stream_to_text(event_data)
                tool_call_chunks = self._provider.process_stream_tool_calls(
                    event_data, tool_call_ids
                )

                chunk = ChatGenerationChunk(
                    message=AIMessageChunk(
                        content=delta,
                        tool_call_chunks=tool_call_chunks,
                    )
                )
                if run_manager:
                    run_manager.on_llm_new_token(delta, chunk=chunk)
                yield chunk
            else:
                generation_info = self._provider.chat_stream_generation_info(event_data)
                yield ChatGenerationChunk(
                    message=AIMessageChunk(
                        content="",
                        additional_kwargs=generation_info,
                    ),
                    generation_info=generation_info,
                )
