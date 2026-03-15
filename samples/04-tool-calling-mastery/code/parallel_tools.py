# Sample 04: Parallel Tool Calling Example
# Demonstrates calling multiple tools in parallel

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool

from langchain_oci import ChatOCIGenAI
import os

# Configuration - uses environment variables or defaults
COMPARTMENT_ID = os.environ.get(
    "OCI_COMPARTMENT_ID", "ocid1.compartment.oc1..your-compartment-id"
)
SERVICE_ENDPOINT = os.environ.get(
    "OCI_SERVICE_ENDPOINT",
    "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
)
AUTH_PROFILE = os.environ.get("OCI_AUTH_PROFILE", "DEFAULT")


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    weather = {
        "Chicago": "72°F, sunny",
        "New York": "68°F, partly cloudy",
        "Los Angeles": "85°F, clear",
    }
    return weather.get(city, f"No data for {city}")


@tool
def get_time(city: str) -> str:
    """Get the current time in a city."""
    times = {
        "Chicago": "2:00 PM CST",
        "New York": "3:00 PM EST",
        "Los Angeles": "12:00 PM PST",
    }
    return times.get(city, f"No time data for {city}")


@tool
def get_population(city: str) -> str:
    """Get the population of a city."""
    populations = {
        "Chicago": "2.7 million",
        "New York": "8.3 million",
        "Los Angeles": "3.9 million",
    }
    return populations.get(city, f"No population data for {city}")


def execute_tools(tool_calls: list, tools_dict: dict) -> list:
    """Execute multiple tool calls and return ToolMessages."""
    results = []
    for tc in tool_calls:
        tool_func = tools_dict[tc["name"]]
        result = tool_func.invoke(tc["args"])
        results.append(ToolMessage(content=result, tool_call_id=tc["id"]))
    return results


def main():
    # Create chat model
    llm = ChatOCIGenAI(
        auth_profile=AUTH_PROFILE,
        model_id="meta.llama-4-scout-17b-16e-instruct",
        service_endpoint=SERVICE_ENDPOINT,
        compartment_id=COMPARTMENT_ID,
    )

    # Tools dictionary for lookup
    tools = [get_weather, get_time, get_population]
    tools_dict = {t.name: t for t in tools}

    # Bind tools with parallel calls enabled
    llm_with_tools = llm.bind_tools(
        tools,
        parallel_tool_calls=True,  # Enable parallel execution
    )

    # Query that requires multiple tools
    print("Query: Tell me about weather, time, and population of both cities")
    print("-" * 60)

    messages = [
        HumanMessage(
            content="Tell me the weather, time, and population of Chicago and New York."
        )
    ]

    response = llm_with_tools.invoke(messages)

    if response.tool_calls:
        print(f"\nModel requested {len(response.tool_calls)} tool calls:")
        for tc in response.tool_calls:
            print(f"  - {tc['name']}({tc['args']})")

        # Execute all tools
        print("\nExecuting tools...")
        tool_results = execute_tools(response.tool_calls, tools_dict)

        for tc, result in zip(response.tool_calls, tool_results):
            print(f"  - {tc['name']}: {result.content}")

        # Send results back to model
        messages.append(response)
        messages.extend(tool_results)

        final_response = llm_with_tools.invoke(messages)
        print(f"\nFinal Answer:\n{final_response.content}")


if __name__ == "__main__":
    main()
