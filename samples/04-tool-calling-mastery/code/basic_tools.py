# Sample 04: Basic Tool Calling Example
# Demonstrates defining tools and using bind_tools()

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

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


# Simple tool with @tool decorator
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: The city name (e.g., "Chicago", "Tokyo")
    """
    weather_data = {
        "Chicago": "72°F, sunny",
        "Tokyo": "68°F, cloudy",
        "London": "55°F, rainy",
    }
    return weather_data.get(city, f"Weather data not available for {city}")


# Tool with Pydantic schema for complex parameters
class CalculatorInput(BaseModel):
    """Input for the calculator tool."""

    expression: str = Field(description="Math expression to evaluate (e.g., '2 + 2')")


@tool(args_schema=CalculatorInput)
def calculate(expression: str) -> str:
    """Calculate a mathematical expression."""
    try:
        # WARNING: Use a safe evaluator in production!
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"


def main():
    # Create chat model
    llm = ChatOCIGenAI(
        auth_profile=AUTH_PROFILE,
        model_id="meta.llama-4-scout-17b-16e-instruct",
        service_endpoint=SERVICE_ENDPOINT,
        compartment_id=COMPARTMENT_ID,
    )

    # Bind tools to the model
    llm_with_tools = llm.bind_tools([get_weather, calculate])

    # Test 1: Weather query
    print("Test 1: Weather Query")
    print("-" * 40)
    messages = [HumanMessage(content="What's the weather in Tokyo?")]
    response = llm_with_tools.invoke(messages)

    if response.tool_calls:
        tool_call = response.tool_calls[0]
        print(f"Tool requested: {tool_call['name']}")
        print(f"Arguments: {tool_call['args']}")

        # Execute the tool
        result = get_weather.invoke(tool_call["args"])
        print(f"Tool result: {result}")

        # Send result back to model
        messages.append(response)
        messages.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))

        final_response = llm_with_tools.invoke(messages)
        print(f"Final answer: {final_response.content}")

    # Test 2: Calculator query
    print("\nTest 2: Calculator Query")
    print("-" * 40)
    messages = [HumanMessage(content="What is 123 * 456?")]
    response = llm_with_tools.invoke(messages)

    if response.tool_calls:
        tool_call = response.tool_calls[0]
        print(f"Tool requested: {tool_call['name']}")
        print(f"Arguments: {tool_call['args']}")

        result = calculate.invoke(tool_call["args"])
        print(f"Tool result: {result}")

        messages.append(response)
        messages.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))

        final_response = llm_with_tools.invoke(messages)
        print(f"Final answer: {final_response.content}")


if __name__ == "__main__":
    main()
