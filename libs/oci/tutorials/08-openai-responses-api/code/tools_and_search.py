# Tutorial 08: Tools and Web Search Example
# Demonstrates function calling, web search, and MCP with ChatOCIOpenAI

from pydantic import BaseModel, Field

# Note: Requires oci-openai package
# pip install oci-openai langchain-openai langchain-oci

# Configuration - replace with your values
COMPARTMENT_ID = "ocid1.compartment.oc1..your-compartment-id"
REGION = "us-chicago-1"
MODEL = "openai.gpt-4.1"


def setup_client():
    """Set up ChatOCIOpenAI client."""
    from oci_openai import OciSessionAuth

    from langchain_oci import ChatOCIOpenAI

    auth = OciSessionAuth(profile_name="DEFAULT")

    client = ChatOCIOpenAI(
        auth=auth,
        compartment_id=COMPARTMENT_ID,
        region=REGION,
        model=MODEL,
    )

    return client


# Define tools as Pydantic models
class GetWeather(BaseModel):
    """Get the current weather for a location."""

    location: str = Field(..., description="City and state, e.g. San Francisco, CA")
    unit: str = Field(default="fahrenheit", description="Unit: celsius or fahrenheit")


class SearchDatabase(BaseModel):
    """Search a database for information."""

    query: str = Field(..., description="The search query")
    table: str = Field(..., description="The database table to search")
    limit: int = Field(default=10, description="Maximum number of results")


def function_calling_demo():
    """Demonstrate function calling with custom tools."""
    print("Function Calling Demo")
    print("=" * 50)

    client = setup_client()

    # Bind tools to the model
    llm_with_tools = client.bind_tools([GetWeather, SearchDatabase])

    # Ask a question that requires tool use
    response = llm_with_tools.invoke("What is the weather like in San Francisco?")

    print(f"Response content: {response.content}")
    print("\nTool calls:")
    for tool_call in response.tool_calls:
        print(f"  - Function: {tool_call['name']}")
        print(f"    Arguments: {tool_call['args']}")


def web_search_demo():
    """Demonstrate web search capability."""
    print("\nWeb Search Demo")
    print("=" * 50)

    client = setup_client()

    # Enable web search
    web_search_tool = {"type": "web_search_preview"}
    llm_with_search = client.bind_tools([web_search_tool])

    # Ask about current events
    response = llm_with_search.invoke("What are the latest developments in AI today?")

    print(f"Response: {response.content}")


def hosted_mcp_demo():
    """Demonstrate hosted MCP integration."""
    print("\nHosted MCP Demo")
    print("=" * 50)

    client = setup_client()

    # Configure MCP tool
    mcp_tool = {
        "type": "mcp",
        "server_label": "deepwiki",
        "server_url": "https://mcp.deepwiki.com/mcp",
        "require_approval": "never",
    }

    llm_with_mcp = client.bind_tools([mcp_tool])

    # Query using MCP
    response = llm_with_mcp.invoke(
        "What is the Model Context Protocol and what are its main features?"
    )

    print(f"Response: {response.content}")


def combined_tools_demo():
    """Demonstrate combining multiple tools."""
    print("\nCombined Tools Demo")
    print("=" * 50)

    client = setup_client()

    # Combine custom tools with web search
    tools = [
        GetWeather,
        {"type": "web_search_preview"},
    ]

    llm_with_tools = client.bind_tools(tools)

    # The model can choose which tool to use
    print("Query 1: Weather question")
    response1 = llm_with_tools.invoke("What's the weather in Tokyo?")
    print(f"Tool calls: {[tc['name'] for tc in response1.tool_calls]}")

    print("\nQuery 2: Current events question")
    response2 = llm_with_tools.invoke("What happened in tech news today?")
    print(f"Response: {response2.content[:200]}...")


if __name__ == "__main__":
    print("Tools and Web Search Examples")
    print("Note: Requires oci-openai package and valid OCI session\n")

    # Uncomment to run (requires actual OCI setup):
    # function_calling_demo()
    # web_search_demo()
    # hosted_mcp_demo()
    # combined_tools_demo()

    print("Examples are commented out - configure credentials and uncomment to run.")
