from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from langchain_oci import ChatOCIOpenAI, OCISessionAuth

COMPARTMENT_ID = ""
CONVERSATION_STORE_ID = ""
OVERRIDE_URL = ""
REGION = ""
MODEL = ""
PROFILE_NAME = ""


def get_oci_openai_client():
    return ChatOCIOpenAI(
        auth=OCISessionAuth(profile_name=PROFILE_NAME),
        compartment_id=COMPARTMENT_ID,
        region=REGION,
        override_url=OVERRIDE_URL,
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
    do_model_invoke()
    do_prompt_chaining()
    do_function_calling()
    do_web_search()
    do_hosted_mcp_calling()


if __name__ == "__main__":
    main()
