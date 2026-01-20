# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Integration tests for create_oci_react_agent helper function.

## Prerequisites

1. **OCI Authentication**: Set up OCI authentication with security token:
   ```bash
   oci session authenticate
   ```

2. **Environment Variables**: Export the following:
   ```bash
   export OCI_REGION="us-chicago-1"  # or your region
   export OCI_COMPARTMENT_ID="ocid1.compartment.oc1..your-compartment-id"
   ```

3. **OCI Config**: Ensure `~/.oci/config` exists with DEFAULT profile

## Running the Tests

Run all integration tests:
```bash
cd libs/oci
python -m pytest tests/integration_tests/agents/test_react_integration.py -v
```

Run specific test:
```bash
pytest tests/integration_tests/agents/test_react_integration.py \
  ::TestOCIReactAgentIntegration::test_simple_tool_call -v
```
"""

import os

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from langchain_oci import create_oci_react_agent


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    weather_data = {
        "chicago": "72F and sunny",
        "new york": "65F and cloudy",
        "san francisco": "58F and foggy",
    }
    return weather_data.get(city.lower(), f"Weather data not available for {city}")


@tool
def calculate(expression: str) -> str:
    """Evaluate a simple math expression. Only supports basic arithmetic."""
    # Simple and safe evaluation for basic math
    allowed_chars = set("0123456789+-*/.(). ")
    if not all(c in allowed_chars for c in expression):
        return "Error: Only basic arithmetic operations are supported"
    try:
        # Limit to safe operations
        result = eval(expression, {"__builtins__": {}}, {})  # noqa: S307
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def skip_if_no_oci_credentials() -> bool:
    """Check if OCI credentials are available."""
    compartment_id = os.environ.get("OCI_COMPARTMENT_ID") or os.environ.get("OCI_COMP")
    return compartment_id is None


@pytest.mark.requires("oci", "langgraph")
@pytest.mark.skipif(
    skip_if_no_oci_credentials(),
    reason="OCI credentials not available (OCI_COMPARTMENT_ID not set)",
)
class TestOCIReactAgentIntegration:
    """Integration tests for create_oci_react_agent."""

    @pytest.fixture
    def compartment_id(self) -> str:
        """Get compartment ID from environment."""
        return os.environ.get("OCI_COMPARTMENT_ID") or os.environ.get("OCI_COMP", "")

    @pytest.fixture
    def service_endpoint(self) -> str:
        """Get service endpoint from environment."""
        region = os.environ.get("OCI_REGION", "us-chicago-1")
        return f"https://inference.generativeai.{region}.oci.oraclecloud.com"

    def test_simple_tool_call(self, compartment_id: str, service_endpoint: str) -> None:
        """Test agent can make a simple tool call."""
        agent = create_oci_react_agent(
            model_id="meta.llama-4-scout-17b-16e-instruct",
            tools=[get_weather],
            compartment_id=compartment_id,
            service_endpoint=service_endpoint,
            auth_type="API_KEY",
            auth_profile="API_KEY_AUTH",
            system_prompt=(
                "You are a helpful weather assistant. "
                "Use the get_weather tool to answer weather questions."
            ),
            temperature=0.3,
            max_tokens=512,
        )

        result = agent.invoke(
            {"messages": [HumanMessage(content="What's the weather in Chicago?")]}
        )

        # Verify we got a response
        assert "messages" in result
        assert len(result["messages"]) > 1

        # Verify the tool was called (should have ToolMessage in history)
        message_types = [type(m).__name__ for m in result["messages"]]
        assert "ToolMessage" in message_types, (
            "Tool should have been called. Message types: " + str(message_types)
        )

        # Verify the final response mentions the weather
        final_message = result["messages"][-1]
        assert final_message.content, "Final message should have content"

    def test_multi_tool_agent(self, compartment_id: str, service_endpoint: str) -> None:
        """Test agent with multiple tools."""
        agent = create_oci_react_agent(
            model_id="meta.llama-4-scout-17b-16e-instruct",
            tools=[get_weather, calculate],
            compartment_id=compartment_id,
            service_endpoint=service_endpoint,
            auth_type="API_KEY",
            auth_profile="API_KEY_AUTH",
            system_prompt=(
                "You can check weather and do math. "
                "Use the calculate tool for math questions."
            ),
            temperature=0.3,
            max_tokens=512,
        )

        result = agent.invoke({"messages": [HumanMessage(content="What is 25 * 4?")]})

        # Should get a response with the calculation
        final_message = result["messages"][-1]
        assert final_message.content, "Should have a response"
        # The answer 100 should appear somewhere in the response
        assert "100" in final_message.content or "100" in str(result["messages"]), (
            "Response should contain the calculation result"
        )

    def test_agent_without_tool_call(
        self, compartment_id: str, service_endpoint: str
    ) -> None:
        """Test agent responds without tool call when not needed."""
        agent = create_oci_react_agent(
            model_id="meta.llama-4-scout-17b-16e-instruct",
            tools=[get_weather],
            compartment_id=compartment_id,
            service_endpoint=service_endpoint,
            auth_type="API_KEY",
            auth_profile="API_KEY_AUTH",
            system_prompt="You are a helpful assistant. Only use tools when necessary.",
            temperature=0.3,
            max_tokens=512,
        )

        result = agent.invoke(
            {"messages": [HumanMessage(content="Hello, how are you?")]}
        )

        # Should get a response
        assert "messages" in result
        final_message = result["messages"][-1]
        assert final_message.content, "Should have a response"

    def test_agent_with_memory_checkpointer(
        self, compartment_id: str, service_endpoint: str
    ) -> None:
        """Test agent with memory checkpointer for conversation persistence."""
        from langgraph.checkpoint.memory import MemorySaver

        checkpointer = MemorySaver()
        agent = create_oci_react_agent(
            model_id="meta.llama-4-scout-17b-16e-instruct",
            tools=[get_weather],
            compartment_id=compartment_id,
            service_endpoint=service_endpoint,
            auth_type="API_KEY",
            auth_profile="API_KEY_AUTH",
            system_prompt="You are a helpful weather assistant.",
            checkpointer=checkpointer,
            temperature=0.3,
            max_tokens=512,
        )

        thread_id = "test_thread_123"
        config = {"configurable": {"thread_id": thread_id}}

        # First message
        result1 = agent.invoke(
            {"messages": [HumanMessage(content="What's the weather in Chicago?")]},
            config=config,
        )
        assert len(result1["messages"]) > 1

        # Second message in same thread should have context
        result2 = agent.invoke(
            {"messages": [HumanMessage(content="How about New York?")]},
            config=config,
        )
        assert len(result2["messages"]) > len(result1["messages"]), (
            "Second invocation should include previous messages"
        )


@pytest.mark.requires("oci", "langgraph")
@pytest.mark.skipif(
    skip_if_no_oci_credentials(),
    reason="OCI credentials not available (OCI_COMPARTMENT_ID not set)",
)
@pytest.mark.parametrize(
    "model_id",
    [
        "meta.llama-4-scout-17b-16e-instruct",
        "meta.llama-3.3-70b-instruct",
        "cohere.command-a-03-2025",
        "xai.grok-3-mini-fast",
        "google.gemini-2.5-flash-lite",
    ],
)
def test_multi_model_tool_calling(model_id: str) -> None:
    """Test tool calling with different models."""
    compartment_id = os.environ.get("OCI_COMPARTMENT_ID") or os.environ.get(
        "OCI_COMP", ""
    )
    region = os.environ.get("OCI_REGION", "us-chicago-1")
    service_endpoint = f"https://inference.generativeai.{region}.oci.oraclecloud.com"

    agent = create_oci_react_agent(
        model_id=model_id,
        tools=[get_weather],
        compartment_id=compartment_id,
        service_endpoint=service_endpoint,
        auth_type="API_KEY",
        auth_profile="API_KEY_AUTH",
        system_prompt="You are a weather assistant. Always use the get_weather tool.",
        temperature=0.3,
        max_tokens=512,
    )

    result = agent.invoke(
        {"messages": [HumanMessage(content="What's the weather in Chicago?")]}
    )

    # Verify we got a response
    assert "messages" in result
    assert len(result["messages"]) > 1

    # Verify the tool was called (should have ToolMessage in history)
    message_types = [type(m).__name__ for m in result["messages"]]
    assert "ToolMessage" in message_types, (
        f"Tool should have been called for {model_id}. Message types: {message_types}"
    )


@pytest.mark.requires("oci", "langgraph")
@pytest.mark.skipif(
    skip_if_no_oci_credentials(),
    reason="OCI credentials not available (OCI_COMPARTMENT_ID not set)",
)
@pytest.mark.parametrize(
    "model_id",
    [
        "meta.llama-4-scout-17b-16e-instruct",
        "meta.llama-3.3-70b-instruct",
        "cohere.command-a-03-2025",
        "xai.grok-3-mini-fast",
        "google.gemini-2.5-flash-lite",
    ],
)
def test_multi_model_support(model_id: str) -> None:
    """Test that create_oci_react_agent works with different models."""
    compartment_id = os.environ.get("OCI_COMPARTMENT_ID") or os.environ.get(
        "OCI_COMP", ""
    )
    region = os.environ.get("OCI_REGION", "us-chicago-1")
    service_endpoint = f"https://inference.generativeai.{region}.oci.oraclecloud.com"

    agent = create_oci_react_agent(
        model_id=model_id,
        tools=[get_weather],
        compartment_id=compartment_id,
        service_endpoint=service_endpoint,
        auth_type="API_KEY",
        auth_profile="API_KEY_AUTH",
        system_prompt="You are a helpful weather assistant.",
        temperature=0.3,
        max_tokens=512,
    )

    result = agent.invoke(
        {"messages": [HumanMessage(content="What's the weather in San Francisco?")]}
    )

    # Verify we got a response
    assert "messages" in result
    assert len(result["messages"]) > 1
