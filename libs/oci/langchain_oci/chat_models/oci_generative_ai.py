# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""OCI Generative AI Chat Models."""

import importlib
import json
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

import httpx
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
)
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
from langchain_openai import ChatOpenAI
from openai import DefaultHttpxClient
from pydantic import BaseModel, ConfigDict, SecretStr, model_validator

from langchain_oci.chat_models.providers import (
    CohereProvider,
    GenericProvider,
    MetaProvider,
    Provider,
)
from langchain_oci.common.utils import CUSTOM_ENDPOINT_PREFIX, OCIUtils
from langchain_oci.llms.oci_generative_ai import OCIGenAIBase
from langchain_oci.llms.utils import enforce_stop_tokens

API_KEY = "<NOTUSED>"
COMPARTMENT_ID_HEADER = "opc-compartment-id"
CONVERSATION_STORE_ID_HEADER = "opc-conversation-store-id"
OUTPUT_VERSION = "responses/v1"


def _build_headers(
    compartment_id: str,
    conversation_store_id: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, str]:
    """Build headers for OCI OpenAI API requests."""
    store = kwargs.get("store", True)

    headers = {COMPARTMENT_ID_HEADER: compartment_id}

    if store:
        if conversation_store_id is None:
            raise ValueError(
                "Conversation Store Id must be provided when store is set to True"
            )
        headers[CONVERSATION_STORE_ID_HEADER] = conversation_store_id

    return headers


class ChatOCIGenAI(BaseChatModel, OCIGenAIBase):
    """ChatOCIGenAI chat model integration.

    Setup:
      Install ``langchain-oci`` and the ``oci`` sdk.

      .. code-block:: bash

          pip install -U langchain-oci oci

    Key init args - completion params:
        model_id: str
            Id of the OCIGenAI chat model to use, e.g., cohere.command-r-16k.
        is_stream: bool
            Whether to stream back partial progress
        model_kwargs: Optional[Dict]
            Keyword arguments to pass to the specific model used, e.g., temperature, max_tokens.

    Key init args - client params:
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
    ) -> Any:
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
            messages, max_sequential_tool_calls=self.max_sequential_tool_calls, **kwargs
        )

        oci_params["is_stream"] = stream
        _model_kwargs = self.model_kwargs or {}

        if stop is not None:
            _model_kwargs[self._provider.stop_sequence_key] = stop

        # Warn if using max_tokens with OpenAI models
        if (
            self.model_id
            and self.model_id.startswith("openai.")
            and "max_tokens" in _model_kwargs
        ):
            import warnings

            warnings.warn(
                "OpenAI models require 'max_completion_tokens' "
                "instead of 'max_tokens'.",
                UserWarning,
                stacklevel=2,
            )

        chat_params = {**_model_kwargs, **kwargs, **oci_params}

        if not self.model_id:
            raise ValueError("Model ID is required for chat.")
        if self.model_id.startswith(CUSTOM_ENDPOINT_PREFIX):
            serving_mode = models.DedicatedServingMode(endpoint_id=self.model_id)
        else:
            serving_mode = models.OnDemandServingMode(model_id=self.model_id)

        # Check if V2 API should be used (currently for Cohere vision models)
        # This flag is set by the provider's messages_to_oci_params() method when it
        # detects multimodal content. The V2 API check is kept at this level (rather
        # than within the provider) to maintain consistency across all providers and
        # allow future providers to use V2 APIs without modifying core logic.
        use_v2 = chat_params.pop("_use_v2_api", False)

        if use_v2:
            # Use V2 API: Supports multimodal content (text + images)
            # Currently used by Cohere Command A Vision for image analysis
            chat_request = self._provider.oci_chat_request_v2(**chat_params)
        else:
            # Use V1 API: Standard text-only chat requests
            # Used by all models that don't require multimodal capabilities
            chat_request = self._provider.oci_chat_request(**chat_params)

        request = models.ChatDetails(
            compartment_id=self.compartment_id,
            serving_mode=serving_mode,
            chat_request=chat_request,
        )

        return request

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], type, Callable, BaseTool]],
        *,
        tool_choice: Optional[
            Union[dict, str, Literal["auto", "none", "required", "any"], bool]
        ] = None,
        parallel_tool_calls: Optional[bool] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
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
            parallel_tool_calls: Whether to enable parallel function calling.
                If True, the model can call multiple tools simultaneously.
                If False or None (default), tools are called sequentially.
                Supported for models using GenericChatRequest (Meta, xAI Grok,
                OpenAI, Mistral). Not supported for Cohere models.
            kwargs: Any additional parameters are passed directly to
                :meth:`~langchain_oci.chat_models.oci_generative_ai.ChatOCIGenAI.bind`.
        """

        formatted_tools = [self._provider.convert_to_oci_tool(tool) for tool in tools]

        if tool_choice is not None:
            kwargs["tool_choice"] = self._provider.process_tool_choice(tool_choice)

        # Add parallel tool calls support (only when explicitly enabled)
        if parallel_tool_calls:
            if not self._provider.supports_parallel_tool_calls:
                raise ValueError(
                    "Parallel tool calls not supported for this provider. "
                    "Only GenericChatRequest models support parallel tool calling."
                )
            kwargs["is_parallel_tool_calls"] = True

        return super().bind(tools=formatted_tools, **kwargs)  # type: ignore[return-value, unused-ignore]

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
                    tools=[schema],
                    first_tool_only=True,
                )
            else:
                output_parser = JsonOutputKeyToolsParser(
                    key_name=tool_name, first_tool_only=True
                )
        elif method == "json_mode":
            llm = self.bind(response_format={"type": "JSON_OBJECT"})  # type: ignore[assignment, unused-ignore]
            output_parser = (
                PydanticOutputParser(pydantic_object=schema)
                if is_pydantic_schema
                else JsonOutputParser()
            )
        elif method == "json_schema":
            json_schema_dict: Dict[str, Any] = (
                schema.model_json_schema()  # type: ignore[union-attr]
                if is_pydantic_schema
                else schema  # type: ignore[assignment]
            )

            # Resolve $ref references as OCI doesn't support $ref and $defs
            json_schema_dict = OCIUtils.resolve_schema_refs(json_schema_dict)

            response_json_schema = self._provider.oci_response_json_schema(
                name=json_schema_dict.get("title", "response"),
                description=json_schema_dict.get("description", ""),
                schema=json_schema_dict,
                is_strict=True,
            )

            response_format_obj = self._provider.oci_json_schema_response_format(
                json_schema=response_json_schema
            )

            llm = self.bind(response_format=response_format_obj)  # type: ignore[assignment, unused-ignore]
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
                   HumanMessage(content="Meow!"),
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

        # Create usage_metadata if usage information is available
        usage_metadata = None
        if hasattr(response.data.chat_response, "usage"):
            usage_metadata = OCIUtils.create_usage_metadata(
                response.data.chat_response.usage
            )

        message = AIMessage(
            content=content or "",
            additional_kwargs=generation_info,
            tool_calls=tool_calls,
            usage_metadata=usage_metadata,
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


class ChatOCIOpenAI(ChatOpenAI):
    """A custom OCI OpenAI client implementation conforming to OpenAI Responses API.

    Setup:
      Install ``openai`` and ``langchain-openai``.

      .. code-block:: bash

          pip install -U openai langchain-openai langchain-oci

    Attributes:
        auth (httpx.Auth): Authentication handler for OCI request signing.
        compartment_id (str): OCI compartment ID for resource isolation
        model (str): Name of OpenAI model to use.
        conversation_store_id (str | None): Conversation Store Id to use
                                            when generating responses.
                                            Must be provided if store is set to False
        region (str | None): The OCI service region, e.g., 'us-chicago-1'.
                             Must be provided if service_endpoint and base_url are None
        service_endpoint (str | None): The OCI service endpoint. when service_endpoint
                                       is provided, the region will be ignored.
        base_url (str | None): The OCI service full path URL.
                               when base_url is provided, the region
                               and service_endpoint will be ignored.

    Instantiate:
        .. code-block:: python

            from oci_openai import OciResourcePrincipalAuth
            from langchain_oci import ChatOCIOpenAI

            client = ChatOCIOpenAI(
                auth=OciResourcePrincipalAuth(),
                compartment_id=COMPARTMENT_ID,
                region="us-chicago-1",
                model=MODEL,
                conversation_store_id=CONVERSATION_STORE_ID,
            )

    Invoke:
        .. code-block:: python

            messages = [
                (
                    "system",
                    "You are a helpful translator. Translate the user
                     sentence to French.",
                ),
                ("human", "I love programming."),
            ]
            response = client.invoke(messages)

    Prompt Chaining:
        .. code-block:: python

            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are a helpful assistant that translates
                        {input_language} to {output_language}.",
                    ),
                    ("human", "{input}"),
                ]
            )
            chain = prompt | client
            response = chain.invoke(
                {
                    "input_language": "English",
                    "output_language": "German",
                    "input": "I love programming.",
                }
            )

    Function Calling:
        .. code-block:: python

            class GetWeather(BaseModel):
                location: str = Field(
                    ..., description="The city and state, e.g. San Francisco, CA"
                )


            llm_with_tools = client.bind_tools([GetWeather])
            ai_msg = llm_with_tools.invoke(
                "what is the weather like in San Francisco",
            )
            response = ai_msg.tool_calls

    Web Search:
        .. code-block:: python

            tool = {"type": "web_search_preview"}
            llm_with_tools = client.bind_tools([tool])
            response = llm_with_tools.invoke("What was a
            positive news story from today?")

    Hosted MCP Calling:
        .. code-block:: python

             llm_with_mcp_tools = client.bind_tools(
                [
                    {
                        "type": "mcp",
                        "server_label": "deepwiki",
                        "server_url": "https://mcp.deepwiki.com/mcp",
                        "require_approval": "never",
                    }
                ]
            )
            response = llm_with_mcp_tools.invoke(
                "What transport protocols does the 2025-03-26 version of the MCP "
                "spec (modelcontextprotocol/modelcontextprotocol) support?"
            )
    """

    @model_validator(mode="before")
    @classmethod
    def validate_openai(cls, values: Any) -> Any:
        """Checks if langchain_openai is installed."""
        if not importlib.util.find_spec("langchain_openai"):
            raise ImportError(
                "Could not import langchain_openai package. "
                "Please install it with `pip install langchain_openai`."
            )
        return values

    def __init__(
        self,
        auth: httpx.Auth,
        compartment_id: str,
        model: str,
        conversation_store_id: Optional[str] = None,
        region: Optional[str] = None,
        service_endpoint: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ):
        try:
            from oci_openai.oci_openai import _resolve_base_url
        except ImportError as e:
            raise ImportError(
                "Could not import _resolve_base_url. "
                "Please install: pip install oci-openai"
            ) from e

        super().__init__(
            model=model,
            api_key=SecretStr(API_KEY),
            http_client=DefaultHttpxClient(
                auth=auth,
                headers=_build_headers(
                    compartment_id=compartment_id,
                    conversation_store_id=conversation_store_id,
                    **kwargs,
                ),
            ),
            base_url=_resolve_base_url(
                region=region, service_endpoint=service_endpoint, base_url=base_url
            ),
            use_responses_api=True,
            output_version=OUTPUT_VERSION,
            **kwargs,
        )
