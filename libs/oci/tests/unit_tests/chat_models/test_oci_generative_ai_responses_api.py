import json
from typing import List, TypedDict
from unittest.mock import MagicMock, patch

import pytest
from httpx import Request
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from oci_openai import (
    OciInstancePrincipalAuth,
    OciResourcePrincipalAuth,
    OciSessionAuth,
)
from oci_openai.oci_openai import _resolve_base_url
from pydantic import BaseModel, Field

from langchain_oci import ChatOCIOpenAI
from langchain_oci.chat_models.oci_generative_ai import (
    COMPARTMENT_ID_HEADER,
    CONVERSATION_STORE_ID_HEADER,
    _build_headers,
)

COMPARTMENT_ID = "ocid1.compartment.oc1..dummy"
CONVERSATION_STORE_ID = "ocid1.generativeaiconversationstore.oc1..dummy"
REGION = "us-chicago-1"
MODEL = "openai.gpt-4o"
SESSION_PRINCIPAL = "session_principal"
RESOURCE_PRINCIPAL = "resource_principal"
INSTANCE_PRINCIPAL = "instance_principal"
BASE_URL = _resolve_base_url(region="us-chicago-1")
RESPONSES_URL = f"{BASE_URL}/responses"
RESPONSE_ID = "resp_123"


# Fixtures
@pytest.fixture(
    params=[
        (SESSION_PRINCIPAL, OciSessionAuth, {"profile_name": "DEFAULT"}),
        (RESOURCE_PRINCIPAL, OciResourcePrincipalAuth, {}),
        (INSTANCE_PRINCIPAL, OciInstancePrincipalAuth, {}),
    ],
    ids=[SESSION_PRINCIPAL, RESOURCE_PRINCIPAL, INSTANCE_PRINCIPAL],
)
def auth_instance(request):
    name, auth_class, kwargs = request.param

    def set_signer(signer_name: str):
        dummy_signer = MagicMock()
        patcher = patch(signer_name, return_value=dummy_signer)
        patcher.start()
        request.addfinalizer(patcher.stop)
        kwargs["signer"] = dummy_signer

    if name == RESOURCE_PRINCIPAL:
        set_signer(signer_name="oci.auth.signers.get_resource_principals_signer")
    elif name == INSTANCE_PRINCIPAL:
        set_signer(signer_name="oci.auth.signers.InstancePrincipalsSecurityTokenSigner")
    elif name == "session_principal":
        # --- Patch config + signer + token/private_key loading ---
        patch_config = patch(
            "oci.config.from_file",
            return_value={
                "user": "ocid1.user.oc1..dummy",
                "fingerprint": "dummyfp",
                "key_file": "/fake/key.pem",
                "tenancy": "ocid1.tenancy.oc1..dummy",
                "region": "us-chicago-1",
                "security_token_file": "/fake/token",
            },
        )
        patch_token = patch.object(
            OciSessionAuth, "_load_token", return_value="fake_token_string"
        )
        patch_private_key = patch.object(
            OciSessionAuth, "_load_private_key", return_value="fake_private_key_data"
        )
        patch_signer = patch(
            "oci.auth.signers.SecurityTokenSigner", return_value=MagicMock()
        )
        # Start all patches
        for p in [patch_config, patch_token, patch_private_key, patch_signer]:
            p.start()
            request.addfinalizer(p.stop)

    return auth_class(**kwargs)


def _assert_common(httpx_mock):
    last_request: Request = httpx_mock.get_requests()[0]
    assert "Authorization" in last_request.headers
    assert last_request.headers.get(COMPARTMENT_ID_HEADER) == COMPARTMENT_ID
    assert (
        last_request.headers.get(CONVERSATION_STORE_ID_HEADER) == CONVERSATION_STORE_ID
    )


function_tools = [
    {
        "type": "function",
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location", "unit"],
        },
    }
]


def _set_mock_client_invoke_response(httpx_mock):
    httpx_mock.add_response(
        url=RESPONSES_URL,
        method="POST",
        json={
            "id": RESPONSE_ID,
            "output": [
                {
                    "type": "message",
                    "status": "completed",
                    "content": [
                        {"type": "output_text", "text": "j'adore la programmation"}
                    ],
                }
            ],
        },
        status_code=200,
    )


class GetWeather(BaseModel):
    """Get the current weather in a given location"""

    location: str = Field(..., description="The city and state, e.g. San Francisco, CA")


def _set_mock_create_response_with_fc_tools(httpx_mock):
    httpx_mock.add_response(
        url=RESPONSES_URL,
        method="POST",
        json={
            "id": RESPONSE_ID,
            "output": [
                {
                    "type": "function_call",
                    "id": "fc_67ca09c6bedc8190a7abfec07b1a1332096610f474011cc0",
                    "call_id": "call_unLAR8MvFNptuiZK6K6HCy5k",
                    "name": "get_current_weather",
                    "arguments": '{"location":"San Francisco, MA","unit":"celsius"}',
                    "status": "completed",
                }
            ],
            "tools": function_tools,
        },
        status_code=200,
    )


def _set_mock_create_response_with_file_input(httpx_mock):
    text = "The file seems to contain excerpts from a letter to the shareholders"
    httpx_mock.add_response(
        url=RESPONSES_URL,
        method="POST",
        json={
            "id": RESPONSE_ID,
            "output": [
                {
                    "id": "msg_686ee",
                    "type": "message",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "annotations": [],
                            "logprobs": [],
                            "text": text,
                        }
                    ],
                    "role": "assistant",
                }
            ],
        },
        status_code=200,
    )


#
def _set_mock_create_response_with_web_search(httpx_mock):
    text = "As of today, Oct 7, 2025, one notable positive news story..."
    httpx_mock.add_response(
        url=RESPONSES_URL,
        method="POST",
        json={
            "id": RESPONSE_ID,
            "output": [
                {"type": "web_search_call", "id": "ws_67cc", "status": "completed"},
                {
                    "type": "message",
                    "id": "msg_67cc",
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": text,
                            "annotations": [
                                {
                                    "type": "url_citation",
                                    "start_index": 442,
                                    "end_index": 557,
                                    "url": "https://.../?utm_source=chatgpt.com",
                                    "title": "...",
                                },
                                {
                                    "type": "url_citation",
                                    "start_index": 962,
                                    "end_index": 1077,
                                    "url": "https://.../?utm_source=chatgpt.com",
                                    "title": "...",
                                },
                            ],
                        }
                    ],
                },
            ],
        },
        status_code=200,
    )


@pytest.fixture
def oci_openai_client(auth_instance):
    """Return a ready OCIOpenAI client for any auth type."""
    client = ChatOCIOpenAI(
        auth=auth_instance,
        compartment_id=COMPARTMENT_ID,
        conversation_store_id=CONVERSATION_STORE_ID,
        region=REGION,
        model=MODEL,
    )
    return client


@pytest.mark.requires("langchain_openai")
@pytest.mark.usefixtures("httpx_mock")
def test_client_invoke(httpx_mock, auth_instance, oci_openai_client):
    # ---- Arrange ----
    _set_mock_client_invoke_response(httpx_mock=httpx_mock)
    messages = [
        (
            "system",
            "You are a helpful translator. Translate the user sentence to French.",
        ),
        ("human", "I love programming."),
    ]

    # ---- Act ----
    result = oci_openai_client.invoke(messages)

    # ---- Assert ----
    assert result.content[0]["text"] == "j'adore la programmation"
    _assert_common(httpx_mock=httpx_mock)


@pytest.mark.requires("langchain_openai")
@pytest.mark.usefixtures("httpx_mock")
def test_prompt_chaining(httpx_mock, auth_instance, oci_openai_client):
    # ---- Arrange ----
    message = """
    You are a helpful assistant that translates {input_language} to {output_language}.
    """
    _set_mock_client_invoke_response(httpx_mock=httpx_mock)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                message,
            ),
            ("human", "{input}"),
        ]
    )
    chain = prompt | oci_openai_client

    # ---- Act ----
    result = chain.invoke(
        {
            "input_language": "English",
            "output_language": "German",
            "input": "I love programming.",
        }
    )
    # ---- Assert ----
    assert result.content[0]["text"] == "j'adore la programmation"
    _assert_common(httpx_mock=httpx_mock)


@pytest.mark.requires("langchain_openai")
@pytest.mark.usefixtures("httpx_mock")
def test_tools_invoke(httpx_mock, auth_instance, oci_openai_client):
    # ---- Arrange ----
    _set_mock_create_response_with_fc_tools(httpx_mock=httpx_mock)
    llm_with_tools = oci_openai_client.bind_tools([GetWeather])

    # ---- Act ----
    ai_msg = llm_with_tools.invoke(
        "what is the weather like in San Francisco",
    )

    # ---- Assert ----
    assert ai_msg.content[0]["type"] == "function_call"
    assert ai_msg.content[0]["name"] == "get_current_weather"
    json.loads((ai_msg.content[0]["arguments"]))["location"] == "San Francisco, MA"
    _assert_common(httpx_mock=httpx_mock)


@pytest.mark.requires("langchain_openai")
@pytest.mark.usefixtures("httpx_mock")
def test_web_search(httpx_mock, auth_instance, oci_openai_client):
    # ---- Arrange ----
    _set_mock_create_response_with_web_search(httpx_mock=httpx_mock)
    tool = {"type": "web_search_preview"}
    llm_with_tools = oci_openai_client.bind_tools([tool])

    # ---- Act ----
    response = llm_with_tools.invoke("What was a positive news story from today?")

    # ---- Assert ----
    assert len(response.content) == 2
    assert response.content[0]["type"] == "web_search_call"
    assert response.content[1]["type"] == "text"
    assert len(response.content[1]["annotations"]) == 2
    _assert_common(httpx_mock=httpx_mock)


def _set_mock_client_invoke_response_langgraph(httpx_mock):
    httpx_mock.add_response(
        url=RESPONSES_URL,
        method="POST",
        json={
            "id": RESPONSE_ID,
            "output": [
                {
                    "type": "message",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "Paris is the capital of France",
                        }
                    ],
                }
            ],
        },
        status_code=200,
    )


@pytest.mark.requires("langchain_openai")
@pytest.mark.usefixtures("httpx_mock")
def test_chat_graph(httpx_mock, auth_instance, oci_openai_client):
    # ---- Arrange ----
    _set_mock_client_invoke_response_langgraph(httpx_mock=httpx_mock)

    class AgentState(TypedDict):
        messages: List[BaseMessage]

    def call_model(state: AgentState):
        messages = state["messages"]
        response = oci_openai_client.invoke(messages)
        return {"messages": messages + [response]}

    workflow = StateGraph(AgentState)
    workflow.add_node("llm_node", call_model)
    workflow.add_edge(START, "llm_node")
    workflow.add_edge("llm_node", END)

    # ---- Act ----
    app = workflow.compile()
    input_message: BaseMessage = HumanMessage(content="What is the capital of France?")
    result = app.invoke({"messages": [input_message]})

    # ---- Assert ----
    content = result["messages"][1].content[0]
    assert content["type"] == "text"
    assert content["text"] == "Paris is the capital of France"
    _assert_common(httpx_mock=httpx_mock)


def test_store_true_with_valid_store_id():
    headers = _build_headers("comp123", conversation_store_id="store456", store=True)
    assert headers == {
        COMPARTMENT_ID_HEADER: "comp123",
        CONVERSATION_STORE_ID_HEADER: "store456",
    }


def test_store_true_missing_store_id_raises():
    with pytest.raises(ValueError) as excinfo:
        _build_headers("comp123", conversation_store_id=None, store=True)

    assert "Conversation Store Id must be provided" in str(excinfo.value)


def test_store_default_true_requires_store_id():
    # store defaults to True → should still raise
    with pytest.raises(ValueError):
        _build_headers("comp123")


def test_store_false_ignores_store_id_requirement():
    headers = _build_headers("comp123", conversation_store_id=None, store=False)
    assert headers == {
        COMPARTMENT_ID_HEADER: "comp123",
    }


def test_store_false_includes_store_id_if_provided_but_not_required():
    headers = _build_headers("comp123", conversation_store_id="store456", store=False)
    # Should NOT include store ID because store=False
    assert headers == {
        COMPARTMENT_ID_HEADER: "comp123",
    }
