# Sample 03: Basic Agent Example
# Demonstrates creating a simple ReAct agent with tools

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from langchain_oci import create_oci_agent

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


# Define tools using the @tool decorator
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: The city name (e.g., "Chicago", "Paris", "Tokyo")
    """
    # In production, call a real weather API
    weather_data = {
        "Chicago": "72°F, sunny",
        "New York": "68°F, cloudy",
        "Los Angeles": "85°F, clear",
        "London": "55°F, rainy",
        "Tokyo": "70°F, partly cloudy",
    }
    return weather_data.get(city, f"Weather data not available for {city}")


@tool
def get_time(city: str) -> str:
    """Get the current time in a city.

    Args:
        city: The city name
    """
    # In production, use a timezone library
    times = {
        "Chicago": "2:00 PM CST",
        "New York": "3:00 PM EST",
        "Los Angeles": "12:00 PM PST",
        "London": "8:00 PM GMT",
        "Tokyo": "5:00 AM JST (next day)",
    }
    return times.get(city, f"Time data not available for {city}")


def main():
    # Create the agent
    agent = create_oci_agent(
        model_id="meta.llama-4-scout-17b-16e-instruct",
        tools=[get_weather, get_time],
        compartment_id=COMPARTMENT_ID,
        service_endpoint=SERVICE_ENDPOINT,
        auth_profile=AUTH_PROFILE,
        system_prompt="You are a helpful travel assistant. "
        "Use the available tools to answer questions about weather and time.",
    )

    # Run the agent
    result = agent.invoke(
        {"messages": [HumanMessage(content="What's the weather and time in Tokyo?")]}
    )

    # Print all messages to see the agent's reasoning
    print("Agent Execution Trace:")
    print("-" * 50)
    for msg in result["messages"]:
        msg_type = msg.type.upper()
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            print(f"{msg_type}: [Tool calls: {[tc['name'] for tc in msg.tool_calls]}]")
        elif hasattr(msg, "tool_call_id"):
            print(f"{msg_type} (tool result): {msg.content[:100]}")
        else:
            print(f"{msg_type}: {msg.content[:200] if msg.content else '(empty)'}")
    print("-" * 50)

    # Final answer
    print(f"\nFinal Answer: {result['messages'][-1].content}")


if __name__ == "__main__":
    main()
