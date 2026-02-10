#!/usr/bin/env python
"""
OCIGenAIAgent Complete Guide
============================

A comprehensive example showcasing all features of OCIGenAIAgent:
- Basic invocation and streaming
- Typed event streaming
- 5 termination conditions
- Reflexion (confidence tracking + loop detection)
- Confidence signals (heuristic early exit)
- Message compression
- Hooks system (callbacks)
- Conversation history
- LCEL chain composition
- LangGraph node integration

Usage:
    export OCI_COMPARTMENT_ID="ocid1.compartment.oc1..xxx"
    export OCI_AUTH_PROFILE="API_KEY_AUTH"  # or "DEFAULT" for security token
    export OCI_AUTH_TYPE="API_KEY"          # or "SECURITY_TOKEN"

    python examples/agents/oci_genai_agent_complete_guide.py
"""

import os
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from langchain_oci import (
    OCIGenAIAgent,
    AgentResult,
    ThinkEvent,
    ToolStartEvent,
    ToolCompleteEvent,
    ReflectEvent,
    TerminateEvent,
    AgentHooks,
    ToolHookContext,
    ToolResultContext,
    IterationContext,
    create_logging_hooks,
    create_metrics_hooks,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

COMPARTMENT_ID = os.environ.get("OCI_COMPARTMENT_ID", "")
AUTH_TYPE = os.environ.get("OCI_AUTH_TYPE", "API_KEY")
AUTH_PROFILE = os.environ.get("OCI_AUTH_PROFILE", "API_KEY_AUTH")
REGION = os.environ.get("OCI_REGION", "us-chicago-1")
MODEL_ID = os.environ.get("OCI_MODEL_ID", "meta.llama-4-scout-17b-16e-instruct")
SERVICE_ENDPOINT = f"https://inference.generativeai.{REGION}.oci.oraclecloud.com"


# =============================================================================
# TOOLS
# =============================================================================

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression.

    Args:
        expression: A math expression like "2 + 2" or "15 * 8"

    Returns:
        The result of the calculation.
    """
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        return "Error: Only basic arithmetic supported"
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"The result is {result}"
    except Exception as e:
        return f"Error: {e}"


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: Name of the city.

    Returns:
        Weather description.
    """
    weather_data = {
        "chicago": "72°F and sunny in Chicago",
        "new york": "65°F and cloudy in New York",
        "san francisco": "58°F and foggy in San Francisco",
        "paris": "68°F and partly cloudy in Paris",
        "london": "55°F and rainy in London",
    }
    return weather_data.get(city.lower(), f"Weather data not available for {city}")


@tool
def search_knowledge(query: str) -> str:
    """Search the knowledge base for information.

    Args:
        query: Search query.

    Returns:
        Search results.
    """
    knowledge = {
        "capital france": "Paris is the capital of France, known for the Eiffel Tower.",
        "capital japan": "Tokyo is the capital of Japan, the world's largest metropolitan area.",
        "python creator": "Python was created by Guido van Rossum in 1991.",
    }
    query_lower = query.lower()
    for key, value in knowledge.items():
        if key in query_lower or all(word in query_lower for word in key.split()):
            return value
    return f"No information found for: {query}"


@tool
def submit_answer(answer: str) -> str:
    """Submit the final answer (terminal tool).

    Call this when you have the complete answer ready.

    Args:
        answer: The final answer to submit.

    Returns:
        Confirmation message.
    """
    return f"ANSWER SUBMITTED: {answer}"


# =============================================================================
# EXAMPLES
# =============================================================================

def example_1_basic_invocation():
    """Basic agent invocation."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Basic Invocation")
    print("=" * 70)

    agent = OCIGenAIAgent(
        model_id=MODEL_ID,
        tools=[calculate, get_weather],
        compartment_id=COMPARTMENT_ID,
        service_endpoint=SERVICE_ENDPOINT,
        auth_type=AUTH_TYPE,
        auth_profile=AUTH_PROFILE,
        max_iterations=5,
    )

    # Simple invoke
    result = agent.invoke("What is 15 * 8?")

    print(f"Query: What is 15 * 8?")
    print(f"Answer: {result.final_answer}")
    print(f"Termination: {result.termination_reason}")
    print(f"Iterations: {result.total_iterations}")
    print(f"Tool calls: {result.total_tool_calls}")


def example_2_streaming_events():
    """Stream typed events during execution."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Streaming Typed Events")
    print("=" * 70)

    agent = OCIGenAIAgent(
        model_id=MODEL_ID,
        tools=[calculate, get_weather],
        compartment_id=COMPARTMENT_ID,
        service_endpoint=SERVICE_ENDPOINT,
        auth_type=AUTH_TYPE,
        auth_profile=AUTH_PROFILE,
        max_iterations=5,
    )

    print("Query: What's the weather in Chicago?")
    print("\nEvents:")

    for event in agent.stream("What's the weather in Chicago?"):
        if isinstance(event, ThinkEvent):
            print(f"  [THINK] Iteration {event.iteration}: Planning {event.tool_calls_planned} tool(s)")
        elif isinstance(event, ToolStartEvent):
            print(f"  [TOOL START] {event.tool_name}({event.arguments})")
        elif isinstance(event, ToolCompleteEvent):
            status = "✓" if event.success else "✗"
            print(f"  [TOOL END] {status} {event.tool_name} -> {event.result[:50]}... ({event.duration_ms:.1f}ms)")
        elif isinstance(event, ReflectEvent):
            print(f"  [REFLECT] Confidence: {event.confidence:.2f}, Assessment: {event.assessment}")
        elif isinstance(event, TerminateEvent):
            print(f"  [TERMINATE] Reason: {event.reason}")
            print(f"\nFinal Answer: {event.final_answer}")


def example_3_hooks_system():
    """Use hooks for monitoring and logging."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Hooks System (Callbacks)")
    print("=" * 70)

    # Custom hooks
    tool_log = []

    def on_tool_start(ctx: ToolHookContext):
        print(f"  -> Starting: {ctx.tool_name}")

    def on_tool_end(ctx: ToolResultContext):
        tool_log.append({
            "tool": ctx.tool_name,
            "duration_ms": ctx.duration_ms,
            "success": ctx.success,
        })
        status = "SUCCESS" if ctx.success else "FAILED"
        print(f"  <- Completed: {ctx.tool_name} [{status}] in {ctx.duration_ms:.1f}ms")

    def on_iteration_end(ctx: IterationContext):
        print(f"  -- Iteration {ctx.iteration} done, confidence: {ctx.confidence:.2f}")

    def on_terminate(reason: str, answer: str):
        print(f"  == Agent finished: {reason}")

    hooks = AgentHooks(
        on_tool_start=[on_tool_start],
        on_tool_end=[on_tool_end],
        on_iteration_end=[on_iteration_end],
        on_terminate=[on_terminate],
    )

    agent = OCIGenAIAgent(
        model_id=MODEL_ID,
        tools=[calculate],
        compartment_id=COMPARTMENT_ID,
        service_endpoint=SERVICE_ENDPOINT,
        auth_type=AUTH_TYPE,
        auth_profile=AUTH_PROFILE,
        hooks=hooks,
        max_iterations=5,
    )

    print("Query: Calculate 25 * 4")
    print("\nHook Events:")
    result = agent.invoke("Calculate 25 * 4")

    print(f"\nAnswer: {result.final_answer}")
    print(f"Tool Log: {tool_log}")


def example_4_metrics_collection():
    """Use built-in metrics hooks."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Metrics Collection")
    print("=" * 70)

    # Built-in metrics hooks
    hooks, metrics = create_metrics_hooks()

    agent = OCIGenAIAgent(
        model_id=MODEL_ID,
        tools=[calculate, get_weather],
        compartment_id=COMPARTMENT_ID,
        service_endpoint=SERVICE_ENDPOINT,
        auth_type=AUTH_TYPE,
        auth_profile=AUTH_PROFILE,
        hooks=hooks,
        max_iterations=5,
    )

    result = agent.invoke("What's the weather in Paris and calculate 100 / 4?")

    print(f"Query: What's the weather in Paris and calculate 100 / 4?")
    print(f"Answer: {result.final_answer}")
    print(f"\nCollected Metrics:")
    print(f"  - Total tool calls: {metrics['total_tool_calls']}")
    print(f"  - Tool durations: {[f'{d:.1f}ms' for d in metrics['tool_durations_ms']]}")
    print(f"  - Tool errors: {metrics['tool_errors']}")
    print(f"  - Iterations: {metrics['iterations']}")
    print(f"  - Termination reason: {metrics['termination_reason']}")


def example_5_conversation_history():
    """Multi-turn conversation with history."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Conversation History (Multi-turn)")
    print("=" * 70)

    agent = OCIGenAIAgent(
        model_id=MODEL_ID,
        tools=[calculate],
        compartment_id=COMPARTMENT_ID,
        service_endpoint=SERVICE_ENDPOINT,
        auth_type=AUTH_TYPE,
        auth_profile=AUTH_PROFILE,
        max_iterations=3,
    )

    # History is stored externally (you manage it)
    history = []

    # Turn 1
    print("\nTurn 1:")
    q1 = "What is 10 + 5?"
    print(f"  User: {q1}")
    r1 = agent.invoke(q1)
    print(f"  Agent: {r1.final_answer}")

    # Add to history
    history.append({"role": "user", "content": q1})
    history.append({"role": "assistant", "content": r1.final_answer})

    # Turn 2 - uses context from turn 1
    print("\nTurn 2:")
    q2 = "Now double that result"
    print(f"  User: {q2}")
    r2 = agent.invoke(q2, message_history=history)
    print(f"  Agent: {r2.final_answer}")

    # Add to history
    history.append({"role": "user", "content": q2})
    history.append({"role": "assistant", "content": r2.final_answer})

    # Turn 3
    print("\nTurn 3:")
    q3 = "What was my first question?"
    print(f"  User: {q3}")
    r3 = agent.invoke(q3, message_history=history)
    print(f"  Agent: {r3.final_answer}")

    print(f"\nHistory size: {len(history)} messages")


def example_6_message_compression():
    """Message compression for long conversations."""
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Message Compression")
    print("=" * 70)

    agent = OCIGenAIAgent(
        model_id=MODEL_ID,
        tools=[],  # No tools for simple demo
        compartment_id=COMPARTMENT_ID,
        service_endpoint=SERVICE_ENDPOINT,
        auth_type=AUTH_TYPE,
        auth_profile=AUTH_PROFILE,
        # Compression settings
        enable_compression=True,
        max_messages=10,  # Keep only 10 most recent messages
        max_iterations=3,
    )

    print(f"Compression enabled: {agent.enable_compression}")
    print(f"Max messages: {agent._compression_config.max_messages}")
    print(f"Strategy: {agent._compression_config.strategy.value}")

    # Simulate long history
    history = [
        {"role": "user", "content": f"Message {i}"}
        for i in range(15)
    ]

    print(f"\nInput history: {len(history)} messages")
    result = agent.invoke("What number am I on?", message_history=history)
    print(f"Answer: {result.final_answer}")
    print("(Compression prevents context overflow with long histories)")


def example_7_confidence_signals():
    """Heuristic confidence detection for early exit."""
    print("\n" + "=" * 70)
    print("EXAMPLE 7: Confidence Signals (Early Exit)")
    print("=" * 70)

    agent = OCIGenAIAgent(
        model_id=MODEL_ID,
        tools=[search_knowledge],
        compartment_id=COMPARTMENT_ID,
        service_endpoint=SERVICE_ENDPOINT,
        auth_type=AUTH_TYPE,
        auth_profile=AUTH_PROFILE,
        # Confidence signal settings
        enable_confidence_signals=True,
        min_iterations_for_early_exit=1,
        confidence_threshold=0.5,
        max_iterations=10,
    )

    print("Confidence signals detect patterns like:")
    print("  - 'verified', 'confirmed' -> VERIFICATION signal")
    print("  - 'the solution is', 'this fixes' -> SOLUTION_FOUND signal")
    print("  - 'successfully completed' -> TASK_COMPLETE signal")
    print("  - 'I am confident that' -> EXPLICIT_CONFIDENCE signal")

    result = agent.invoke("What is the capital of France?")

    print(f"\nQuery: What is the capital of France?")
    print(f"Answer: {result.final_answer}")
    print(f"Final confidence: {result.confidence:.2f}")
    print(f"Termination: {result.termination_reason}")


def example_8_terminal_tools():
    """Explicit completion with terminal tools."""
    print("\n" + "=" * 70)
    print("EXAMPLE 8: Terminal Tools")
    print("=" * 70)

    agent = OCIGenAIAgent(
        model_id=MODEL_ID,
        tools=[search_knowledge, submit_answer],
        compartment_id=COMPARTMENT_ID,
        service_endpoint=SERVICE_ENDPOINT,
        auth_type=AUTH_TYPE,
        auth_profile=AUTH_PROFILE,
        terminal_tools=["submit_answer"],  # This tool signals completion
        max_iterations=10,
    )

    print("Terminal tools: ['submit_answer']")
    print("When agent calls submit_answer, it terminates immediately.")

    result = agent.invoke(
        "Find the capital of Japan and submit the answer using submit_answer."
    )

    print(f"\nAnswer: {result.final_answer}")
    print(f"Termination: {result.termination_reason}")


def example_9_reflexion():
    """Self-evaluation with reflexion."""
    print("\n" + "=" * 70)
    print("EXAMPLE 9: Reflexion (Self-Evaluation)")
    print("=" * 70)

    agent = OCIGenAIAgent(
        model_id=MODEL_ID,
        tools=[calculate, get_weather],
        compartment_id=COMPARTMENT_ID,
        service_endpoint=SERVICE_ENDPOINT,
        auth_type=AUTH_TYPE,
        auth_profile=AUTH_PROFILE,
        enable_reflexion=True,  # Enable self-evaluation
        loop_threshold=3,  # Detect loops after 3 repetitions
        max_iterations=5,
    )

    print("Reflexion provides:")
    print("  - Confidence tracking (increases with successful tools)")
    print("  - Loop detection (detects repeated tool calls)")
    print("  - Progress assessment (on_track, stuck, new_findings)")

    result = agent.invoke("Get weather for Chicago and calculate 50 * 2")

    print(f"\nAnswer: {result.final_answer}")
    print(f"Final confidence: {result.confidence:.2f}")

    print("\nReasoning Steps:")
    for step in result.reasoning_steps:
        print(f"  Iteration {step.iteration}: confidence={step.confidence:.2f}, "
              f"tools={len(step.tool_executions)}, assessment={step.assessment}")


def example_10_lcel_chain():
    """Use agent in LangChain Expression Language chain."""
    print("\n" + "=" * 70)
    print("EXAMPLE 10: LCEL Chain Composition")
    print("=" * 70)

    from langchain_core.runnables import RunnablePassthrough, RunnableLambda

    agent = OCIGenAIAgent(
        model_id=MODEL_ID,
        tools=[calculate],
        compartment_id=COMPARTMENT_ID,
        service_endpoint=SERVICE_ENDPOINT,
        auth_type=AUTH_TYPE,
        auth_profile=AUTH_PROFILE,
        max_iterations=3,
    )

    # Build a chain: passthrough -> agent -> extract answer
    chain = (
        RunnablePassthrough()
        | agent
        | RunnableLambda(lambda r: f"RESULT: {r.final_answer}")
    )

    print("Chain: RunnablePassthrough() | agent | extract_answer")

    output = chain.invoke("What is 99 + 1?")
    print(f"\nOutput: {output}")


def example_11_langgraph_node():
    """Use agent as a LangGraph node."""
    print("\n" + "=" * 70)
    print("EXAMPLE 11: LangGraph Node")
    print("=" * 70)

    try:
        from langgraph.graph import StateGraph, START, END
    except ImportError:
        print("LangGraph not installed. Skipping this example.")
        print("Install with: pip install langgraph")
        return

    agent = OCIGenAIAgent(
        model_id=MODEL_ID,
        tools=[calculate],
        compartment_id=COMPARTMENT_ID,
        service_endpoint=SERVICE_ENDPOINT,
        auth_type=AUTH_TYPE,
        auth_profile=AUTH_PROFILE,
        max_iterations=3,
    )

    # Create graph with agent as node
    graph = StateGraph(dict)
    graph.add_node("agent", agent)
    graph.add_edge(START, "agent")
    graph.add_edge("agent", END)

    compiled = graph.compile()

    print("Graph: START -> agent -> END")

    result = compiled.invoke({"input": "What is 7 * 7?"})
    print(f"\nGraph output: {result}")


def example_12_all_features():
    """Combine all features together."""
    print("\n" + "=" * 70)
    print("EXAMPLE 12: All Features Combined")
    print("=" * 70)

    # Metrics hooks
    hooks, metrics = create_metrics_hooks()

    agent = OCIGenAIAgent(
        model_id=MODEL_ID,
        tools=[calculate, get_weather, search_knowledge],
        compartment_id=COMPARTMENT_ID,
        service_endpoint=SERVICE_ENDPOINT,
        auth_type=AUTH_TYPE,
        auth_profile=AUTH_PROFILE,
        # Hooks
        hooks=hooks,
        # Compression
        enable_compression=True,
        max_messages=20,
        # Confidence signals
        enable_confidence_signals=True,
        min_iterations_for_early_exit=2,
        # Reflexion
        enable_reflexion=True,
        loop_threshold=3,
        confidence_threshold=0.8,
        max_iterations=10,
    )

    # Multi-turn with all features
    history = []
    queries = [
        "What's the weather in London?",
        "Calculate 144 / 12",
        "What is the capital of France?",
    ]

    for q in queries:
        print(f"\nUser: {q}")
        result = agent.invoke(q, message_history=history)
        print(f"Agent: {result.final_answer}")

        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": result.final_answer})

    print(f"\n--- Session Metrics ---")
    print(f"Total tool calls: {metrics['total_tool_calls']}")
    print(f"Tool errors: {metrics['tool_errors']}")
    print(f"History size: {len(history)} messages")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run all examples."""
    print("=" * 70)
    print("OCIGenAIAgent Complete Guide")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Model: {MODEL_ID}")
    print(f"  Region: {REGION}")
    print(f"  Auth: {AUTH_TYPE} ({AUTH_PROFILE})")

    examples = [
        ("Basic Invocation", example_1_basic_invocation),
        ("Streaming Events", example_2_streaming_events),
        ("Hooks System", example_3_hooks_system),
        ("Metrics Collection", example_4_metrics_collection),
        ("Conversation History", example_5_conversation_history),
        ("Message Compression", example_6_message_compression),
        ("Confidence Signals", example_7_confidence_signals),
        ("Terminal Tools", example_8_terminal_tools),
        ("Reflexion", example_9_reflexion),
        ("LCEL Chain", example_10_lcel_chain),
        ("LangGraph Node", example_11_langgraph_node),
        ("All Features", example_12_all_features),
    ]

    for name, fn in examples:
        try:
            fn()
        except Exception as e:
            print(f"\nError in {name}: {e}")
            break

    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
