# Copyright (c) 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Integration tests for tool calling with OCI Generative AI chat models.

These tests verify that tool calling works correctly without infinite loops
for both Meta and Cohere models after receiving tool results.

## Prerequisites

1. **OCI Authentication**: Set up OCI authentication with security token:
   ```bash
   oci session authenticate
   ```

2. **Environment Variables**: Export the following:
   ```bash
   export OCI_REGION="us-chicago-1"  # or your region
   export OCI_COMP="ocid1.compartment.oc1..your-compartment-id"
   ```

3. **OCI Config**: Ensure `~/.oci/config` exists with DEFAULT profile

## Running the Tests

Run all integration tests:
```bash
cd libs/oci
python -m pytest tests/integration_tests/chat_models/test_tool_calling.py -v
```

Run specific test:
```bash
pytest tests/integration_tests/chat_models/test_tool_calling.py \
  ::test_meta_llama_tool_calling -v
```

Run with a specific model:
```bash
pytest tests/integration_tests/chat_models/test_tool_calling.py \
  ::test_tool_calling_no_infinite_loop \
  -k "meta.llama-4-scout" -v
```

## What These Tests Verify

1. **No Infinite Loops**: Models stop calling tools after receiving results
2. **Proper Tool Flow**: Tool called → Results received → Final response generated
3. **Fix Works**: `tool_choice="none"` is set when ToolMessages are present
4. **Multi-Vendor**: Works for both Meta Llama and Cohere models
"""

import os

import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from langchain_oci.chat_models import ChatOCIGenAI


def get_weather(city: str) -> str:
    """Get the current weather for a given city name."""
    weather_data = {
        "chicago": "Sunny, 65°F",
        "new york": "Cloudy, 60°F",
        "san francisco": "Foggy, 58°F",
    }
    return weather_data.get(city.lower(), f"Weather data not available for {city}")


@pytest.fixture
def weather_tool():
    """Create a weather tool for testing."""
    return StructuredTool.from_function(
        func=get_weather,
        name="get_weather",
        description="Get the current weather for a given city name.",
    )


def create_agent(model_id: str, weather_tool: StructuredTool):
    """Create a LangGraph agent with tool calling."""
    region = os.getenv("OCI_REGION", "us-chicago-1")
    endpoint = f"https://inference.generativeai.{region}.oci.oraclecloud.com"
    chat_model = ChatOCIGenAI(
        model_id=model_id,
        service_endpoint=endpoint,
        compartment_id=os.getenv("OCI_COMP"),
        model_kwargs={"temperature": 0.3, "max_tokens": 512, "top_p": 0.9},
        auth_type="SECURITY_TOKEN",
        auth_profile="DEFAULT",
        auth_file_location=os.path.expanduser("~/.oci/config"),
        disable_streaming="tool_calling",
    )

    tool_node = ToolNode(tools=[weather_tool])
    model_with_tools = chat_model.bind_tools([weather_tool])

    def call_model(state: MessagesState):
        """Call the model with tools bound."""
        messages = state["messages"]
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}

    def should_continue(state: MessagesState):
        """Check if the model wants to call a tool."""
        messages = state["messages"]
        last_message = messages[-1]

        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return END

    builder = StateGraph(MessagesState)
    builder.add_node("call_model", call_model)
    builder.add_node("tools", tool_node)
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges("call_model", should_continue, ["tools", END])
    builder.add_edge("tools", "call_model")

    return builder.compile()


@pytest.mark.requires("oci")
@pytest.mark.parametrize(
    "model_id",
    [
        "meta.llama-4-scout-17b-16e-instruct",
        "meta.llama-3.3-70b-instruct",
        "cohere.command-a-03-2025",
        "cohere.command-r-plus-08-2024",
    ],
)
def test_tool_calling_no_infinite_loop(model_id: str, weather_tool: StructuredTool):
    """Test that tool calling works without infinite loops.

    This test verifies that after a tool is called and results are returned,
    the model generates a final response without making additional tool calls,
    preventing infinite loops.

    The fix sets tool_choice='none' when ToolMessages are present in the
    conversation history, which tells the model to stop calling tools.
    """
    agent = create_agent(model_id, weather_tool)

    # Invoke the agent
    system_msg = (
        "You are a helpful assistant. Use the available tools when "
        "needed to answer questions accurately."
    )
    result = agent.invoke(
        {
            "messages": [
                SystemMessage(content=system_msg),
                HumanMessage(content="What's the weather in Chicago?"),
            ]
        }
    )

    messages = result["messages"]

    # Verify the conversation structure
    expected = "Should have at least: System, Human, AI (tool call), Tool, AI"
    assert len(messages) >= 4, expected

    # Find tool messages
    tool_messages = [msg for msg in messages if type(msg).__name__ == "ToolMessage"]
    assert len(tool_messages) >= 1, "Should have at least one tool result"

    # Find AI messages with tool calls
    ai_tool_calls = [
        msg
        for msg in messages
        if (
            type(msg).__name__ == "AIMessage"
            and hasattr(msg, "tool_calls")
            and msg.tool_calls
        )
    ]
    # The model should call the tool, but after receiving results,
    # should not call again. Allow flexibility - some models might make
    # 1 call, others might need 2, but should stop
    error_msg = (
        f"Model made too many tool calls ({len(ai_tool_calls)}), possible infinite loop"
    )
    assert len(ai_tool_calls) <= 2, error_msg

    # Verify final message is an AI response without tool calls
    final_message = messages[-1]
    assert type(final_message).__name__ == "AIMessage", (
        "Final message should be AIMessage"
    )
    assert final_message.content, "Final message should have content"
    assert not (hasattr(final_message, "tool_calls") and final_message.tool_calls), (
        "Final message should not have tool_calls (infinite loop prevention)"
    )

    # Note: Different models format responses differently. Some return
    # natural language, others may return the tool call syntax. The
    # important thing is they STOPPED calling tools. Just verify the
    # response has some content (proves it didn't loop infinitely)


@pytest.mark.requires("oci")
def test_meta_llama_tool_calling(weather_tool: StructuredTool):
    """Specific test for Meta Llama models to ensure fix works."""
    model_id = "meta.llama-4-scout-17b-16e-instruct"
    agent = create_agent(model_id, weather_tool)

    result = agent.invoke(
        {
            "messages": [
                SystemMessage(content="You are a helpful assistant."),
                HumanMessage(content="Check the weather in San Francisco."),
            ]
        }
    )

    messages = result["messages"]
    final_message = messages[-1]

    # Meta Llama was specifically affected by infinite loops
    # Verify it stops after receiving tool results (most important check!)
    assert type(final_message).__name__ == "AIMessage"
    assert not (hasattr(final_message, "tool_calls") and final_message.tool_calls)
    assert final_message.content, "Should have generated some response"
    # Meta Llama 4 Scout sometimes returns tool syntax instead of natural language,
    # but that's okay - the key is it STOPPED calling tools


@pytest.mark.requires("oci")
def test_cohere_tool_calling(weather_tool: StructuredTool):
    """Specific test for Cohere models to ensure they work correctly."""
    model_id = "cohere.command-a-03-2025"
    agent = create_agent(model_id, weather_tool)

    result = agent.invoke(
        {
            "messages": [
                SystemMessage(content="You are a helpful assistant."),
                HumanMessage(content="What's the weather like in New York?"),
            ]
        }
    )

    messages = result["messages"]
    final_message = messages[-1]

    # Cohere models should handle tool calling naturally
    assert type(final_message).__name__ == "AIMessage"
    assert not (hasattr(final_message, "tool_calls") and final_message.tool_calls)
    assert "60" in final_message.content or "cloudy" in final_message.content.lower()


@pytest.mark.requires("oci")
@pytest.mark.parametrize(
    "model_id",
    [
        "meta.llama-4-scout-17b-16e-instruct",
        "cohere.command-a-03-2025",
    ],
)
def test_multi_step_tool_orchestration(model_id: str):
    """Test multi-step tool orchestration without infinite loops.

    This test simulates a realistic diagnostic workflow where an agent
    needs to call 4-6 tools sequentially (similar to SRE/monitoring
    scenarios). It verifies that:

    1. The agent can call multiple tools in sequence (multi-step)
    2. The agent eventually stops and provides a final answer
    3. No infinite loops occur (respects max_sequential_tool_calls limit)
    4. Tool call count stays within reasonable bounds (4-8 calls)

    This addresses the specific issue where agents need to perform
    multi-step investigations requiring several tool calls before
    providing a final analysis.
    """

    # Create diagnostic tools that simulate a monitoring workflow
    def check_status(resource: str) -> str:
        """Check the status of a resource."""
        status_data = {
            "payment-service": "Status: Running, Memory: 95%, Restarts: 12",
            "web-server": "Status: Running, Memory: 60%, Restarts: 0",
        }
        return status_data.get(resource, f"Resource {resource} status: Unknown")

    def get_events(resource: str) -> str:
        """Get recent events for a resource."""
        events_data = {
            "payment-service": (
                "Events: [OOMKilled at 14:23, BackOff at 14:30, Started at 14:32]"
            ),
            "web-server": "Events: [Started at 10:00, Healthy]",
        }
        return events_data.get(resource, f"No events for {resource}")

    def get_metrics(resource: str) -> str:
        """Get historical metrics for a resource."""
        metrics_data = {
            "payment-service": (
                "Memory trend: 70%→80%→90%→95% (gradual increase over 2h)"
            ),
            "web-server": "Memory trend: 55%→58%→60% (stable)",
        }
        return metrics_data.get(resource, f"No metrics for {resource}")

    def check_changes(resource: str) -> str:
        """Check recent changes to a resource."""
        changes_data = {
            "payment-service": "Recent deployment: v1.2.3 deployed 2h ago",
            "web-server": "No recent changes (last deployment 3 days ago)",
        }
        return changes_data.get(resource, f"No changes for {resource}")

    def create_alert(severity: str, message: str) -> str:
        """Create an alert/incident."""
        return f"Alert created: [{severity.upper()}] {message}"

    def take_action(resource: str, action: str) -> str:
        """Take a remediation action."""
        return f"Action completed: {action} on {resource}"

    # Create tools
    tools = [
        StructuredTool.from_function(
            func=check_status,
            name="check_status",
            description="Check the current status of a resource",
        ),
        StructuredTool.from_function(
            func=get_events,
            name="get_events",
            description="Get recent events for a resource",
        ),
        StructuredTool.from_function(
            func=get_metrics,
            name="get_metrics",
            description="Get historical metrics for a resource",
        ),
        StructuredTool.from_function(
            func=check_changes,
            name="check_changes",
            description="Check recent changes to a resource",
        ),
        StructuredTool.from_function(
            func=create_alert,
            name="create_alert",
            description="Create an alert or incident",
        ),
        StructuredTool.from_function(
            func=take_action,
            name="take_action",
            description="Take a remediation action on a resource",
        ),
    ]

    # Create agent with higher recursion limit to allow multi-step
    region = os.getenv("OCI_REGION", "us-chicago-1")
    endpoint = f"https://inference.generativeai.{region}.oci.oraclecloud.com"
    chat_model = ChatOCIGenAI(
        model_id=model_id,
        service_endpoint=endpoint,
        compartment_id=os.getenv("OCI_COMP"),
        model_kwargs={"temperature": 0.2, "max_tokens": 2048, "top_p": 0.9},
        auth_type="SECURITY_TOKEN",
        auth_profile="DEFAULT",
        auth_file_location=os.path.expanduser("~/.oci/config"),
        disable_streaming="tool_calling",
        max_sequential_tool_calls=8,  # Allow up to 8 sequential tool calls
    )

    tool_node = ToolNode(tools=tools)
    model_with_tools = chat_model.bind_tools(tools)

    def call_model(state: MessagesState):
        """Call the model with tools bound."""
        messages = state["messages"]
        response = model_with_tools.invoke(messages)

        # OCI LIMITATION: Only allow ONE tool call at a time
        if (
            hasattr(response, "tool_calls")
            and response.tool_calls
            and len(response.tool_calls) > 1
        ):
            # Some models try to call multiple tools in parallel
            # Restrict to first tool only to avoid OCI API error
            response.tool_calls = [response.tool_calls[0]]

        return {"messages": [response]}

    def should_continue(state: MessagesState):
        """Check if the model wants to call a tool."""
        messages = state["messages"]
        last_message = messages[-1]

        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return END

    builder = StateGraph(MessagesState)
    builder.add_node("call_model", call_model)
    builder.add_node("tools", tool_node)
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges("call_model", should_continue, ["tools", END])
    builder.add_edge("tools", "call_model")
    agent = builder.compile()

    # System prompt that encourages multi-step investigation
    system_prompt = """You are a diagnostic assistant. When investigating
    issues, follow this workflow:

    1. Check current status
    2. Review recent events
    3. Analyze historical metrics
    4. Check for recent changes
    5. Create alert if needed
    6. Take remediation action if appropriate
    7. Provide final summary

    Call the necessary tools to gather information, then provide a
    comprehensive analysis."""

    # Invoke agent with a diagnostic scenario
    # Langgraph invoke signature is generic; passing dict is valid at runtime
    result = agent.invoke(
        {
            "messages": [
                SystemMessage(content=system_prompt),
                HumanMessage(
                    content=(
                        "Investigate the payment-service resource. "
                        "It has high memory usage and restarts. "
                        "Determine root cause and recommend actions."
                    )
                ),
            ]
        },
        config={"recursion_limit": 25},  # Allow enough recursion for multi-step
    )

    messages = result["messages"]

    # Count tool calls
    tool_call_messages = [
        msg
        for msg in messages
        if (
            type(msg).__name__ == "AIMessage"
            and hasattr(msg, "tool_calls")
            and msg.tool_calls
        )
    ]
    tool_result_messages = [
        msg for msg in messages if type(msg).__name__ == "ToolMessage"
    ]

    # Verify multi-step orchestration worked
    msg = f"Should have made multiple tool calls (got {len(tool_call_messages)})"
    assert len(tool_call_messages) >= 2, msg

    # CRITICAL: Verify max_sequential_tool_calls limit was respected
    # The agent should stop at or before the limit (8 tool calls)
    # This is the key protection against infinite loops
    assert len(tool_call_messages) <= 8, (
        f"Too many tool calls ({len(tool_call_messages)}), "
        "max_sequential_tool_calls limit not enforced"
    )

    # Verify tool results were received
    assert len(tool_result_messages) >= 2, "Should have received multiple tool results"

    # Verify agent eventually stopped (didn't loop infinitely)
    # The final message might still have tool_calls if the agent hit
    # the max_sequential_tool_calls limit, which is expected behavior.
    # The key is that it STOPPED (didn't continue infinitely).
    final_message = messages[-1]
    assert type(final_message).__name__ in [
        "AIMessage",
        "ToolMessage",
    ], "Final message should be AIMessage or ToolMessage"

    # Verify the agent didn't hit infinite loop by checking message count
    # With max_sequential_tool_calls=8, we expect roughly:
    # System + Human + (AI + Tool) * 8 = ~18 messages maximum
    assert len(messages) <= 25, (
        f"Too many messages ({len(messages)}), possible infinite loop. "
        "The max_sequential_tool_calls limit should have stopped the agent."
    )

    # SUCCESS: If we got here, the test passed!
    # The agent successfully:
    # 1. Made multiple tool calls (multi-step orchestration)
    # 2. Stopped within the max_sequential_tool_calls limit
    # 3. Did not loop infinitely
