import importlib
from typing import Any, Generator, Mapping, Union

import httpx
import oci
import requests
from langchain_openai import ChatOpenAI
from oci.config import DEFAULT_LOCATION, DEFAULT_PROFILE
from openai import (
    DEFAULT_MAX_RETRIES,
    NOT_GIVEN,
    AsyncOpenAI,
    NotGiven,
    OpenAI,
    Timeout,
)
from pydantic import SecretStr, model_validator

from langchain_oci.llms.utils import get_base_url, get_sync_httpx_client, get_async_httpx_client

API_KEY = "<NOTUSED>"
COMPARTMENT_ID_HEADER = "opc-compartment-id"
CONVERSATION_STORE_ID_HEADER = "opc-conversation-store-id"
OUTPUT_VERSION = "responses/v1"


class OCIHttpxAuth(httpx.Auth):
    """
    Custom HTTPX authentication class that implements OCI request signing.

    This class handles the authentication flow for HTTPX requests by signing them
    using the OCI Signer, which adds the necessary authentication headers for OCI
     API calls.

    Attributes:
        signer (oci.signer.Signer): The OCI signer instance used for request signing
    """

    def __init__(self, signer: Any):
        self.signer = signer

    def auth_flow(
        self, request: httpx.Request
    ) -> Generator[httpx.Request, httpx.Response, None]:
        req = requests.Request(
            method=request.method,
            url=str(request.url),
            headers=dict(request.headers),
            data=request.content,
        )
        prepared_request = req.prepare()

        # Sign the request using the OCI Signer
        self.signer.do_request_sign(prepared_request)

        # Update the original HTTPX request with the signed headers
        request.headers.update(prepared_request.headers)

        yield request


class OCISessionAuth(OCIHttpxAuth):
    """
    OCI authentication implementation using session-based authentication.

    This class implements OCI authentication using a session token and private key
    loaded from the OCI configuration file. It's suitable for interactive user sessions.

    """

    def __init__(
        self, config_file: str = DEFAULT_LOCATION, profile_name: str = DEFAULT_PROFILE
    ):
        config = oci.config.from_file(config_file, profile_name)
        token = self._load_token(config)
        private_key = self._load_private_key(config)
        super().__init__(
            signer=oci.auth.signers.SecurityTokenSigner(token, private_key)
        )

    def _load_token(self, config: Any) -> str:
        token_file = config["security_token_file"]
        with open(token_file, "r") as f:
            return f.read().strip()

    def _load_private_key(self, config: Any) -> str:
        return oci.signer.load_private_key_from_file(config["key_file"])


class OCIResourcePrincipleAuth(OCIHttpxAuth):
    """
    OCI authentication implementation using Resource Principal authentication.

    This class implements OCI authentication using Resource Principal credentials,
    which is suitable for services running within OCI that need to access other
    OCI services.
    """

    def __init__(self, signer: Any) -> None:
        super().__init__(signer=oci.auth.signers.get_resource_principals_signer())


class OCIInstancePrincipleAuth(OCIHttpxAuth):
    """
    OCI authentication implementation using Instance Principal authentication.

    This class implements OCI authentication using Instance Principal credentials,
    which is suitable for compute instances that need to access OCI services.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            signer=oci.auth.signers.InstancePrincipalsSecurityTokenSigner(**kwargs)
        )


class OCIUserPrincipleAuth(OCIHttpxAuth):
    """
    OCI authentication implementation using user principle authentication.

        This class implements OCI authentication using API Key credentials loaded from
    the OCI configuration file. It's suitable for programmatic access to OCI services.

    Attributes:
        signer (oci.signer.Signer): OCI signer configured with API key credentials
    """

    def __init__(
        self, config_file: str = DEFAULT_LOCATION, profile_name: str = DEFAULT_PROFILE
    ) -> None:
        config = oci.config.from_file(config_file, profile_name)
        oci.config.validate_config(config)

        signer = oci.signer.Signer(
            tenancy=config["tenancy"],
            user=config["user"],
            fingerprint=config["fingerprint"],
            private_key_file_location=config.get("key_file"),
            # pass_phrase is optional and can be None
            pass_phrase=oci.config.get_config_value_or_default(config, "pass_phrase"),
            # private_key_content is optional and can be None
            private_key_content=config.get("key_content"),
        )
        super().__init__(signer=signer)


class OCIOpenAI(OpenAI):
    """
    A custom OpenAI client implementation for Oracle Cloud Infrastructure (OCI)
     GenAI Unified API confirming to OpenAI Responses API.


    Attributes:
        auth (httpx.Auth): Authentication handler for OCI
        compartment_id (str): Compartment ID for resource isolation
        conversation_store_id (str): Conversation Store Id to use when generating responses
        region (str): OCI Region where the OCI GenAI Unified API resides
        timeout (float | Timeout | None | NotGiven): Request timeout configuration
        max_retries (int): Maximum number of retry attempts for failed requests
        default_headers (Mapping[str, str] | None): Default HTTP headers
        default_query (Mapping[str, object] | None): Default query parameters
        override_url (str): OCI Unified API URL to override , if it's different from base url
    """

    def __init__(
        self,
        auth: httpx.Auth,
        compartment_id: str,
        conversation_store_id: str,
        region: str = "",
        override_url: str = "",
        timeout: Union[float, Timeout, None, NotGiven] = NOT_GIVEN,
        max_retries: int = DEFAULT_MAX_RETRIES,
        default_headers: Union[Mapping[str, str], None] = None,
        default_query: Union[Mapping[str, object], None] = None,
    ) -> None:
        super().__init__(
            api_key=API_KEY,
            base_url=get_base_url(region=region, override_url=override_url),
            timeout=timeout,
            max_retries=max_retries,
            default_headers=default_headers,
            default_query=default_query,
            http_client=get_sync_httpx_client(
                auth=auth,
                headers={
                    COMPARTMENT_ID_HEADER: compartment_id,
                    CONVERSATION_STORE_ID_HEADER: conversation_store_id,
                }
            ),
        )


class OCIAsyncOpenAI(AsyncOpenAI):
    """
    A async custom OpenAI client implementation for Oracle Cloud Infrastructure (OCI) GenAI
    Unified API confirming to OpenAI Responses API.

    Attributes:
        auth (httpx.Auth): Authentication handler for OCI
        compartment_id (str): OCI compartment ID for resource isolation
        conversation_store_id (str): Conversation Store Id to use when generating responses
        region (str): OCI Region where the OCI GenAI Unified API resides
        timeout (float | Timeout | None | NotGiven): Request timeout configuration
        max_retries (int): Maximum number of retry attempts for failed requests
        default_headers (Mapping[str, str] | None): Default HTTP headers
        default_query (Mapping[str, object] | None): Default query parameters
        override_url (str): OCI Unified API URL to override , if it's different from base url
    """

    def __init__(
        self,
        auth: httpx.Auth,
        compartment_id: str,
        conversation_store_id: str,
        region: str = "",
        override_url: str = "",
        timeout: Union[float, Timeout, None, NotGiven] = NOT_GIVEN,
        max_retries: int = DEFAULT_MAX_RETRIES,
        default_headers: Union[Mapping[str, str], None] = None,
        default_query: Union[Mapping[str, object], None] = None,
    ) -> None:
        super().__init__(
            api_key=API_KEY,
            base_url=get_base_url(region=region, override_url=override_url),
            timeout=timeout,
            max_retries=max_retries,
            default_headers=default_headers,
            default_query=default_query,
            http_client=get_async_httpx_client(
                auth=auth,
                headers={
                    COMPARTMENT_ID_HEADER: compartment_id,
                    CONVERSATION_STORE_ID_HEADER: conversation_store_id,
                }
            ),
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
        override_url (str): OCI Unified API URL to override , if it's different from base url

    Instantiate:
        .. code-block:: python

            from langchain_oci.chat_models.oci_generative_ai_responses_api import (
                ChatOCIOpenAI, OCIResourcePrincipleAuth,
            )
            client = ChatOCIOpenAI(
                auth=OCIResourcePrincipleAuth(),
                compartment_id=COMPARTMENT_ID,
                region=REGION,
                override_url=OVERRIDE_URL,
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
        auth: OCISessionAuth,
        compartment_id: str,
        conversation_store_id: str,
        region: str,
        model: str,
        override_url: str = "",
        **kwargs: Any,
    ):
        super().__init__(
            model=model,
            api_key=SecretStr(API_KEY),
            http_client=get_sync_httpx_client(
                auth=auth,
                headers={
                    COMPARTMENT_ID_HEADER: compartment_id,
                    CONVERSATION_STORE_ID_HEADER: conversation_store_id,
                }
            ),
            base_url=get_base_url(region, override_url),
            use_responses_api=True,
            output_version=OUTPUT_VERSION,
            **kwargs,
        )
