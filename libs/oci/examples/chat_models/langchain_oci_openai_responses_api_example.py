from langchain_core.prompts import ChatPromptTemplate
from oci_openai import OciSessionAuth
from pydantic import BaseModel, Field
from rich import print

from langchain_oci import ChatOCIOpenAI

COMPARTMENT_ID = "ocid1.compartment.oc1..aaaaaaaaexample"
CONVERSATION_STORE_ID = (
    "ocid1.generativeaiconversationstore.oc1.us-chicago-1.aaaaaaaaexample"
)
SERVICE_ENDPOINT = "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
REGION = "us-chicago-1"
MODEL = "openai.gpt-4o"
PROFILE_NAME = "oc1"


def get_oci_openai_client():
    return ChatOCIOpenAI(
        auth=OciSessionAuth(profile_name=PROFILE_NAME),
        compartment_id=COMPARTMENT_ID,
        service_endpoint=SERVICE_ENDPOINT,
        model=MODEL,
        conversation_store_id=CONVERSATION_STORE_ID,
    )


def do_model_invoke():
    client = get_oci_openai_client()
    messages = [
        (
            "system",
            "You are a helpful translator. Translate the user sentence to French.",
        ),
        ("human", "I love programming."),
    ]
    response = client.invoke(messages)
    return response


def do_prompt_chaining():
    client = get_oci_openai_client()
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant that translates {input_language}"
                " to {output_language}.",
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
    return response


def do_function_calling():
    class GetWeather(BaseModel):
        """Get the current weather in a given location"""

        location: str = Field(
            ..., description="The city and state, e.g. San Francisco, CA"
        )

    client = get_oci_openai_client()
    llm_with_tools = client.bind_tools([GetWeather])
    response = llm_with_tools.invoke(
        "what is the weather like in San Francisco",
    )
    return response


def do_web_search():
    client = get_oci_openai_client()
    tool = {"type": "web_search"}
    llm_with_tools = client.bind_tools([tool])

    response = llm_with_tools.invoke("What was a positive news story from today?")
    return response


def do_hosted_mcp_calling():
    client = get_oci_openai_client()
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
    return response


def main():
    print(do_model_invoke())
    print(do_prompt_chaining())
    print(do_function_calling())
    print(do_web_search())
    print(do_hosted_mcp_calling())


if __name__ == "__main__":
    main()
