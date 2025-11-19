import importlib
from typing import Any

import httpx

from langchain_openai import ChatOpenAI
from oci_openai.oci_openai import _build_base_url, _build_service_endpoint, _resolve_base_url
from openai import DefaultHttpxClient

from pydantic import SecretStr, model_validator


API_KEY = "<NOTUSED>"
COMPARTMENT_ID_HEADER = "opc-compartment-id"
CONVERSATION_STORE_ID_HEADER = "opc-conversation-store-id"
OUTPUT_VERSION = "responses/v1"

def get_sync_httpx_client(auth: httpx.Auth, compartment_id: str, conversation_store_id: str) -> httpx.Client:
    return DefaultHttpxClient(
                auth=auth,
                headers={
                    COMPARTMENT_ID_HEADER: compartment_id,
                    CONVERSATION_STORE_ID_HEADER: conversation_store_id
                },
            )

class ChatOCIOpenAI(ChatOpenAI):
    """A custom OCI OpenAI client implementation confirming to OpenAI Responses API.

    Setup:
      Install ``openai`` and ``langchain-openai``.

      .. code-block:: bash

          pip install -U langchain-oci openai langchain-openai

    Attributes:
        auth (httpx.Auth): Authentication handler for OCI
        compartment_id (str): OCI compartment ID for resource isolation
        conversation_store_id (str): Conversation Store Id to use when generating responses
        region (str): OCI Region where the OCI GenAI Unified API resides
        model (str): Name of OpenAI model to use.
        service_endpoint (str): OCI Unified API URL to override , if it's different from base url

    Instantiate:
        .. code-block:: python

            from langchain_oci.chat_models.oci_generative_ai_responses_api import (
                ChatOCIOpenAI, OCIResourcePrincipleAuth,
            )
            client = ChatOCIOpenAI(
                auth=OCIResourcePrincipleAuth(),
                compartment_id=COMPARTMENT_ID,
                service_endpoint=OVERRIDE_URL,
                model=MODEL,
                conversation_store_id=CONVERSATION_STORE_ID,
            )

    Invoke:
        .. code-block:: python

            messages = [
                (
                    "system",
                    "You are a helpful translator. Translate the user sentence to French.",
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
                        "You are a helpful assistant that translates {input_language} to {output_language}.",
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
            response = llm_with_tools.invoke("What was a positive news story from today?")

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
        auth: Any,
        compartment_id: str,
        conversation_store_id: str,
        region: str,
        model: str,
        service_endpoint: str = "",
        **kwargs: Any,
    ):
        super().__init__(
            model=model,
            api_key=SecretStr(API_KEY),
            http_client=get_sync_httpx_client(
                auth=auth,
                compartment_id=compartment_id,
                conversation_store_id=conversation_store_id
            ),
            base_url=_resolve_base_url(region=region, base_url=service_endpoint),
            use_responses_api=True,
            output_version=OUTPUT_VERSION,
            **kwargs,
        )
