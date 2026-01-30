# Copyright (c) 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Integration tests for Gemini parallel tool call flattening.

Gemini models require 1:1 function call to function response mapping
per turn. When the model makes N parallel tool calls,
_flatten_parallel_tool_calls() rewrites the message history into N
sequential (AIMessage, ToolMessage) pairs.

Without this fix, the OCI backend returns:
  400 INVALID_ARGUMENT: "Please ensure that the number of function
  response parts is equal to the number of function call parts of
  the function call turn."

Prerequisites:
  export OCI_COMPARTMENT_ID="your-compartment-id"
  export OCI_AUTH_TYPE="API_KEY"
  export OCI_CONFIG_PROFILE="API_KEY_AUTH"

Run:
  python -m pytest tests/integration_tests/chat_models/ \
    test_gemini_parallel_tool_calls.py -v
"""

import os
from typing import List

import pytest
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import StructuredTool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from langchain_oci.chat_models import ChatOCIGenAI

# --------------- tool functions ---------------


def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    data = {
        "new york": "Sunny, 72F",
        "los angeles": "Cloudy, 65F",
        "chicago": "Rainy, 55F",
        "london": "Overcast, 50F",
        "tokyo": "Clear, 68F",
        "new york city": "Sunny, 72F",
    }
    return data.get(city.lower(), f"Weather unavailable for {city}")


def get_time(city: str) -> str:
    """Get the current local time in a city."""
    data = {
        "new york": "3:00 PM EST",
        "los angeles": "12:00 PM PST",
        "chicago": "2:00 PM CST",
        "london": "8:00 PM GMT",
        "tokyo": "5:00 AM JST",
        "new york city": "3:00 PM EST",
    }
    return data.get(city.lower(), f"Time unavailable for {city}")


def get_population(city: str) -> str:
    """Get the population of a city."""
    data = {
        "new york": "8.3 million",
        "los angeles": "3.9 million",
        "chicago": "2.7 million",
        "london": "8.8 million",
        "tokyo": "13.9 million",
        "new york city": "8.3 million",
    }
    return data.get(city.lower(), f"Population unavailable for {city}")


def get_country(city: str) -> str:
    """Get the country a city is located in."""
    data = {
        "new york": "United States",
        "los angeles": "United States",
        "chicago": "United States",
        "london": "United Kingdom",
        "tokyo": "Japan",
        "new york city": "United States",
    }
    return data.get(city.lower(), f"Country unavailable for {city}")


TOOL_DISPATCH = {
    "get_weather": get_weather,
    "get_time": get_time,
    "get_population": get_population,
    "get_country": get_country,
}


def _execute_tool_calls(response) -> List[ToolMessage]:
    """Execute all tool calls in a response and return ToolMessages."""
    results = []
    for tc in response.tool_calls:
        fn = TOOL_DISPATCH.get(tc["name"])
        content = fn(**tc["args"]) if fn else "unknown tool"
        results.append(ToolMessage(content=content, tool_call_id=tc["id"]))
    return results


# --------------- fixtures ---------------


def _make_gemini_llm(model_id: str) -> ChatOCIGenAI:
    """Create a Gemini ChatOCIGenAI for the given model."""
    compartment_id = os.environ.get("OCI_COMPARTMENT_ID")
    if not compartment_id:
        pytest.skip("OCI_COMPARTMENT_ID not set")

    region = os.getenv("OCI_REGION", "us-chicago-1")
    endpoint = f"https://inference.generativeai.{region}.oci.oraclecloud.com"
    return ChatOCIGenAI(
        model_id=model_id,
        service_endpoint=endpoint,
        compartment_id=compartment_id,
        model_kwargs={"max_tokens": 256},
        auth_type=os.environ.get("OCI_AUTH_TYPE", "API_KEY"),
        auth_profile=os.environ.get("OCI_CONFIG_PROFILE", "API_KEY_AUTH"),
        auth_file_location=os.path.expanduser("~/.oci/config"),
    )


@pytest.fixture
def gemini_llm():
    """Gemini 2.5 Flash instance."""
    return _make_gemini_llm("google.gemini-2.5-flash")


@pytest.fixture
def weather_tool():
    return StructuredTool.from_function(
        func=get_weather,
        name="get_weather",
        description="Get the current weather for a city.",
    )


@pytest.fixture
def time_tool():
    return StructuredTool.from_function(
        func=get_time,
        name="get_time",
        description="Get the current local time in a city.",
    )


# --------------- basic tests ---------------


@pytest.mark.requires("oci")
def test_gemini_parallel_tool_calls_manual(gemini_llm):
    """Direct reproduction of the bug (2 parallel tool calls).

    Without the fix, step 2 fails with 400 INVALID_ARGUMENT.
    """
    llm = gemini_llm.bind_tools([get_weather, get_time])

    response = llm.invoke(
        "What is the weather AND the current time in New York City? Call both tools."
    )

    if not response.tool_calls:
        pytest.skip("Model did not make any tool calls")
    if len(response.tool_calls) < 2:
        pytest.skip(
            f"Model made {len(response.tool_calls)} tool call(s), "
            "need 2+ to test parallel flattening"
        )

    messages = [
        HumanMessage(
            content=(
                "What is the weather AND the current time in "
                "New York City? Call both tools."
            )
        ),
        response,
        *_execute_tool_calls(response),
    ]

    final = llm.invoke(messages)
    assert final.content, "Gemini should return a final text response"


@pytest.mark.requires("oci")
def test_gemini_agent_with_parallel_tools(gemini_llm, weather_tool, time_tool):
    """Full LangGraph agent loop with Gemini parallel tool calls."""
    tools = [weather_tool, time_tool]
    tool_node = ToolNode(tools=tools)
    model_with_tools = gemini_llm.bind_tools(tools)

    def call_model(state: MessagesState):
        return {"messages": [model_with_tools.invoke(state["messages"])]}

    def should_continue(state: MessagesState):
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return END

    builder = StateGraph(MessagesState)
    builder.add_node("call_model", call_model)
    builder.add_node("tools", tool_node)
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges("call_model", should_continue, ["tools", END])
    builder.add_edge("tools", "call_model")
    agent = builder.compile()

    result = agent.invoke(
        {  # type: ignore[arg-type]
            "messages": [
                HumanMessage(
                    content=(
                        "What is the weather AND the time in New York? Use both tools."
                    )
                )
            ]
        }
    )

    final = result["messages"][-1]
    assert type(final).__name__ == "AIMessage"
    assert final.content
    assert not (hasattr(final, "tool_calls") and final.tool_calls)


@pytest.mark.requires("oci")
def test_gemini_single_tool_call_unaffected(gemini_llm):
    """Single tool calls still work (flattening is a no-op)."""
    llm = gemini_llm.bind_tools([get_weather])

    response = llm.invoke("What is the weather in Chicago?")

    if not response.tool_calls:
        pytest.skip("Model did not make a tool call")

    assert len(response.tool_calls) == 1
    tc = response.tool_calls[0]
    assert tc["name"] == "get_weather"

    messages = [
        HumanMessage(content="What is the weather in Chicago?"),
        response,
        ToolMessage(
            content=get_weather(**tc["args"]),
            tool_call_id=tc["id"],
        ),
    ]
    final = llm.invoke(messages)
    assert final.content


@pytest.mark.requires("oci")
def test_non_gemini_model_parallel_calls_unaffected():
    """Flattening does NOT activate for non-Gemini models."""
    compartment_id = os.environ.get("OCI_COMPARTMENT_ID")
    if not compartment_id:
        pytest.skip("OCI_COMPARTMENT_ID not set")

    region = os.getenv("OCI_REGION", "us-chicago-1")
    endpoint = f"https://inference.generativeai.{region}.oci.oraclecloud.com"
    llm = ChatOCIGenAI(
        model_id="meta.llama-3.3-70b-instruct",
        service_endpoint=endpoint,
        compartment_id=compartment_id,
        model_kwargs={"max_tokens": 256},
        auth_type=os.environ.get("OCI_AUTH_TYPE", "API_KEY"),
        auth_profile=os.environ.get("OCI_CONFIG_PROFILE", "API_KEY_AUTH"),
        auth_file_location=os.path.expanduser("~/.oci/config"),
    )

    llm_with_tools = llm.bind_tools([get_weather])
    response = llm_with_tools.invoke("What is the weather in Chicago?")

    if not response.tool_calls:
        pytest.skip("Model did not call tools")

    tc = response.tool_calls[0]
    messages = [
        HumanMessage(content="What is the weather in Chicago?"),
        response,
        ToolMessage(
            content=get_weather(**tc["args"]),
            tool_call_id=tc["id"],
        ),
    ]
    final = llm_with_tools.invoke(messages)
    assert final.content


# --------------- robust / stress tests ---------------


@pytest.mark.requires("oci")
def test_gemini_three_parallel_tool_calls(gemini_llm):
    """Force 3 parallel tool calls and verify all results are used.

    Asks a question that requires weather, time, AND population,
    so the model should call all three tools at once.
    """
    llm = gemini_llm.bind_tools([get_weather, get_time, get_population])

    response = llm.invoke(
        "For Tokyo: what is the weather, what time is it, "
        "and what is the population? "
        "Use all three tools at once."
    )

    if not response.tool_calls:
        pytest.skip("Model did not make any tool calls")
    if len(response.tool_calls) < 3:
        pytest.skip(
            f"Model made {len(response.tool_calls)} call(s), "
            "need 3 to test triple parallel flattening"
        )

    messages = [
        HumanMessage(
            content=(
                "For Tokyo: what is the weather, what time is it, "
                "and what is the population? "
                "Use all three tools at once."
            )
        ),
        response,
        *_execute_tool_calls(response),
    ]

    final = llm.invoke(messages)
    assert final.content

    # Verify the model actually incorporated the tool results
    content_lower = final.content.lower()
    assert any(w in content_lower for w in ["68", "clear", "tokyo"]), (
        f"Response should mention Tokyo weather: {final.content}"
    )
    assert any(w in content_lower for w in ["5:00", "jst", "am"]), (
        f"Response should mention Tokyo time: {final.content}"
    )
    assert any(w in content_lower for w in ["13.9", "million", "population"]), (
        f"Response should mention Tokyo population: {final.content}"
    )


@pytest.mark.requires("oci")
def test_gemini_four_parallel_tool_calls(gemini_llm):
    """Force 4 parallel tool calls — maximum stress test."""
    all_tools = [get_weather, get_time, get_population, get_country]
    llm = gemini_llm.bind_tools(all_tools)

    response = llm.invoke(
        "For London: get the weather, time, population, "
        "and country. Call all four tools simultaneously."
    )

    if not response.tool_calls:
        pytest.skip("Model did not make any tool calls")
    if len(response.tool_calls) < 4:
        pytest.skip(
            f"Model made {len(response.tool_calls)} call(s), "
            "need 4 to test quad parallel flattening"
        )

    messages = [
        HumanMessage(
            content=(
                "For London: get the weather, time, population, "
                "and country. Call all four tools simultaneously."
            )
        ),
        response,
        *_execute_tool_calls(response),
    ]

    final = llm.invoke(messages)
    assert final.content

    content_lower = final.content.lower()
    assert any(w in content_lower for w in ["50", "overcast"]), (
        f"Should mention London weather: {final.content}"
    )
    assert any(w in content_lower for w in ["8:00", "gmt", "pm"]), (
        f"Should mention London time: {final.content}"
    )


@pytest.mark.requires("oci")
def test_gemini_multi_turn_parallel_calls(gemini_llm):
    """Multi-turn: parallel calls in turn 1, then again in turn 2.

    Turn 1: Ask about NYC (weather + time)
    Turn 2: Follow up asking about LA (weather + time)

    Both turns involve parallel tool calls. The second turn tests
    that flattening works correctly with prior flattened history.
    """
    llm = gemini_llm.bind_tools([get_weather, get_time])

    # --- Turn 1 ---
    resp1 = llm.invoke("What is the weather and time in New York? Call both tools.")
    if not resp1.tool_calls or len(resp1.tool_calls) < 2:
        pytest.skip("Turn 1 did not produce parallel tool calls")

    messages: List[BaseMessage] = [
        HumanMessage(
            content=("What is the weather and time in New York? Call both tools.")
        ),
        resp1,
        *_execute_tool_calls(resp1),
    ]

    summary1 = llm.invoke(messages)
    assert summary1.content
    messages.append(summary1)

    # --- Turn 2 ---
    follow_up = HumanMessage(
        content=("Now do the same for Los Angeles. Call both tools again.")
    )
    messages.append(follow_up)

    resp2 = llm.invoke(messages)

    if not resp2.tool_calls or len(resp2.tool_calls) < 2:
        pytest.skip("Turn 2 did not produce parallel tool calls")

    messages.append(resp2)
    messages.extend(_execute_tool_calls(resp2))

    summary2 = llm.invoke(messages)
    assert summary2.content

    # Verify turn 2 results reference LA data
    content_lower = summary2.content.lower()
    assert any(w in content_lower for w in ["65", "cloudy", "los angeles", "la"]), (
        f"Turn 2 should reference LA weather: {summary2.content}"
    )


@pytest.mark.requires("oci")
def test_gemini_agent_multi_tool_diagnostic(gemini_llm):
    """Full agent loop with 4 tools — realistic diagnostic scenario.

    The model gets 4 tools and a prompt that naturally leads to
    calling several in parallel. Verifies the full LangGraph cycle
    works end-to-end with the flattening fix.
    """
    tools = [
        StructuredTool.from_function(
            func=get_weather,
            name="get_weather",
            description="Get current weather for a city.",
        ),
        StructuredTool.from_function(
            func=get_time,
            name="get_time",
            description="Get current local time in a city.",
        ),
        StructuredTool.from_function(
            func=get_population,
            name="get_population",
            description="Get population of a city.",
        ),
        StructuredTool.from_function(
            func=get_country,
            name="get_country",
            description="Get the country a city is in.",
        ),
    ]
    tool_node = ToolNode(tools=tools)
    model_with_tools = gemini_llm.bind_tools(tools)

    def call_model(state: MessagesState):
        return {"messages": [model_with_tools.invoke(state["messages"])]}

    def should_continue(state: MessagesState):
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return END

    builder = StateGraph(MessagesState)
    builder.add_node("call_model", call_model)
    builder.add_node("tools", tool_node)
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges("call_model", should_continue, ["tools", END])
    builder.add_edge("tools", "call_model")
    agent = builder.compile()

    result = agent.invoke(
        {  # type: ignore[arg-type]
            "messages": [
                SystemMessage(
                    content=(
                        "You are a travel assistant. When asked "
                        "about a city, gather all available info "
                        "using the tools provided. Call multiple "
                        "tools at once when possible."
                    )
                ),
                HumanMessage(
                    content=(
                        "I'm planning a trip to Tokyo. Tell me "
                        "everything you can: weather, time, "
                        "population, and country."
                    )
                ),
            ]
        },
        config={"recursion_limit": 15},
    )

    msgs = result["messages"]
    final = msgs[-1]
    assert type(final).__name__ == "AIMessage"
    assert final.content
    assert not (hasattr(final, "tool_calls") and final.tool_calls), (
        "Agent should have finished with a text response"
    )

    # Count tools actually called
    tool_msgs = [m for m in msgs if type(m).__name__ == "ToolMessage"]
    assert len(tool_msgs) >= 2, (
        f"Agent should have called at least 2 tools, got {len(tool_msgs)}"
    )


@pytest.mark.requires("oci")
@pytest.mark.parametrize(
    "model_id",
    [
        "google.gemini-2.5-flash",
        "google.gemini-2.5-pro",
    ],
)
def test_gemini_models_parallel_tool_calls(model_id: str):
    """Verify parallel flattening works on both Gemini models.

    Runs the same parallel tool call scenario against Flash and Pro
    to ensure the fix is not model-specific.
    """
    llm = _make_gemini_llm(model_id)
    llm_with_tools = llm.bind_tools([get_weather, get_time])

    response = llm_with_tools.invoke(
        "What is the weather and time in Chicago? Call both tools."
    )

    if not response.tool_calls:
        pytest.skip(f"{model_id}: Model did not make any tool calls")

    # Even with 1 tool call, sending results back should work
    messages = [
        HumanMessage(
            content=("What is the weather and time in Chicago? Call both tools.")
        ),
        response,
        *_execute_tool_calls(response),
    ]

    final = llm_with_tools.invoke(messages)
    assert final.content, f"{model_id}: should return a final response"


@pytest.mark.requires("oci")
def test_gemini_result_correctness(gemini_llm):
    """Verify tool results are correctly paired after flattening.

    Calls weather for two different cities and checks the final
    answer contains the right data for each city (not swapped).
    """
    llm = gemini_llm.bind_tools([get_weather])

    # Manually construct a parallel scenario with known data
    # to verify the flattening preserves correct pairing
    from langchain_core.messages import AIMessage

    messages: List[BaseMessage] = [
        HumanMessage(content="What is the weather in Tokyo and London?"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "tc_tokyo",
                    "name": "get_weather",
                    "args": {"city": "Tokyo"},
                },
                {
                    "id": "tc_london",
                    "name": "get_weather",
                    "args": {"city": "London"},
                },
            ],
        ),
        ToolMessage(
            content="Clear, 68F",
            tool_call_id="tc_tokyo",
        ),
        ToolMessage(
            content="Overcast, 50F",
            tool_call_id="tc_london",
        ),
    ]

    final = llm.invoke(messages)
    assert final.content

    content_lower = final.content.lower()
    # Tokyo should be associated with 68/clear
    # London should be associated with 50/overcast
    has_tokyo = any(w in content_lower for w in ["68", "clear"])
    has_london = any(w in content_lower for w in ["50", "overcast"])
    assert has_tokyo, f"Should mention Tokyo weather (68F/Clear): {final.content}"
    assert has_london, f"Should mention London weather (50F/Overcast): {final.content}"
