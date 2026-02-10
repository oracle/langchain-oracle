# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Example usage of OCIGenAIAgent.

OCIGenAIAgent provides a sophisticated agentic loop with:
- Immutable state management
- Typed event streaming
- Reflexion (confidence tracking + loop detection)
- 5 termination conditions
- LangChain Runnable interface for LCEL composability
- LangGraph node compatibility

This example demonstrates various usage patterns.
"""
# ruff: noqa: T201

import os

from langchain_core.tools import tool

from langchain_oci import (
    OCIGenAIAgent,
    ReflectEvent,
    TerminateEvent,
    ThinkEvent,
    ToolCompleteEvent,
    ToolStartEvent,
)


# Define tools
@tool
def search(query: str) -> str:
    """Search for information about a topic.

    Args:
        query: The search query.

    Returns:
        Search results.
    """
    # Simulated search results
    results = {
        "capital of france": "Paris is the capital of France.",
        "weather": "The weather is sunny with a high of 72F.",
        "python": "Python is a programming language created by Guido van Rossum.",
    }
    query_lower = query.lower()
    for key, value in results.items():
        if key in query_lower:
            return value
    return f"No results found for: {query}"


@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression.

    Args:
        expression: A mathematical expression to evaluate.

    Returns:
        The result of the calculation.
    """
    try:
        # Simple and safe evaluation
        allowed = set("0123456789+-*/.(). ")
        if not all(c in allowed for c in expression):
            return "Error: Invalid characters in expression"
        result = eval(expression)  # noqa: S307
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"


def basic_usage():
    """Basic OCIGenAIAgent usage."""
    print("=" * 60)
    print("Basic OCIGenAIAgent Usage")
    print("=" * 60)

    # Create agent
    agent = OCIGenAIAgent(
        model_id="meta.llama-4-scout-17b-16e-instruct",
        tools=[search, calculate],
        compartment_id=os.environ.get("OCI_COMPARTMENT_ID"),
        service_endpoint=os.environ.get("OCI_GENAI_ENDPOINT"),
        auth_type="API_KEY",
        system_prompt="You are a helpful assistant. Use tools when needed.",
        max_iterations=5,
    )

    # Invoke the agent
    result = agent.invoke("What is the capital of France?")

    print(f"Final Answer: {result.final_answer}")
    print(f"Termination Reason: {result.termination_reason}")
    print(f"Total Iterations: {result.total_iterations}")
    print(f"Total Tool Calls: {result.total_tool_calls}")
    print(f"Confidence: {result.confidence:.2f}")


def streaming_usage():
    """Stream events from OCIGenAIAgent."""
    print("\n" + "=" * 60)
    print("Streaming OCIGenAIAgent Events")
    print("=" * 60)

    agent = OCIGenAIAgent(
        model_id="meta.llama-4-scout-17b-16e-instruct",
        tools=[search, calculate],
        compartment_id=os.environ.get("OCI_COMPARTMENT_ID"),
        service_endpoint=os.environ.get("OCI_GENAI_ENDPOINT"),
        enable_reflexion=True,
    )

    # Stream events
    for event in agent.stream("What is 25 * 4 and also tell me about Python?"):
        if isinstance(event, ThinkEvent):
            thought = event.thought[:100] if event.thought else ""
            print(f"\n[Iteration {event.iteration}] Thinking: {thought}...")
            print(f"  Planning {event.tool_calls_planned} tool call(s)")

        elif isinstance(event, ToolStartEvent):
            print(f"  -> Starting tool: {event.tool_name}")
            print(f"     Arguments: {event.arguments}")

        elif isinstance(event, ToolCompleteEvent):
            result = event.result[:100] if event.result else ""
            print(f"  <- Tool {event.tool_name} completed in {event.duration_ms:.1f}ms")
            print(f"     Result: {result}...")

        elif isinstance(event, ReflectEvent):
            print(f"  [Reflect] Confidence: {event.confidence:.2f}")
            print(f"            Assessment: {event.assessment}")

        elif isinstance(event, TerminateEvent):
            print(f"\n[Terminated] Reason: {event.reason}")
            print(f"Final Answer: {event.final_answer}")

def with_custom_termination():
    """OCIGenAIAgent with custom termination tools."""
    print("\n" + "=" * 60)
    print("OCIGenAIAgent with Terminal Tool")
    print("=" * 60)

    @tool
    def submit_answer(answer: str) -> str:
        """Submit the final answer.

        Call this tool when you have found the answer.

        Args:
            answer: The final answer to submit.

        Returns:
            Confirmation message.
        """
        return f"Answer submitted: {answer}"

    agent = OCIGenAIAgent(
        model_id="meta.llama-4-scout-17b-16e-instruct",
        tools=[search, submit_answer],
        compartment_id=os.environ.get("OCI_COMPARTMENT_ID"),
        terminal_tools=["submit_answer"],  # Agent terminates when this is called
        enable_reflexion=False,  # Disable reflexion for simpler flow
    )

    result = agent.invoke(
        "Find out what the capital of France is and submit the answer."
    )
    print(f"Result: {result.final_answer}")
    print(f"Terminated because: {result.termination_reason}")


def lcel_chain_usage():
    """Use OCIGenAIAgent in an LCEL chain."""
    print("\n" + "=" * 60)
    print("OCIGenAIAgent in LCEL Chain")
    print("=" * 60)

    from langchain_core.runnables import RunnablePassthrough

    agent = OCIGenAIAgent(
        model_id="meta.llama-4-scout-17b-16e-instruct",
        tools=[search],
        compartment_id=os.environ.get("OCI_COMPARTMENT_ID"),
    )

    # Create a chain that extracts just the final answer
    chain = RunnablePassthrough() | agent | (lambda r: r.final_answer)

    answer = chain.invoke("Tell me about Python programming")
    print(f"Answer: {answer}")


def langgraph_node_usage():
    """Use OCIGenAIAgent as a LangGraph node."""
    print("\n" + "=" * 60)
    print("OCIGenAIAgent as LangGraph Node")
    print("=" * 60)

    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError:
        print("LangGraph not installed. Skipping this example.")
        return

    agent = OCIGenAIAgent(
        model_id="meta.llama-4-scout-17b-16e-instruct",
        tools=[search],
        compartment_id=os.environ.get("OCI_COMPARTMENT_ID"),
    )

    # Create a simple graph with OCIGenAIAgent as a node
    graph = StateGraph(dict)
    graph.add_node("agent", agent)
    graph.add_edge(START, "agent")
    graph.add_edge("agent", END)

    compiled = graph.compile()
    result = compiled.invoke({"messages": []})
    print(f"Graph result: {result}")


def inspect_reasoning():
    """Inspect the agent's reasoning steps."""
    print("\n" + "=" * 60)
    print("Inspecting Reasoning Steps")
    print("=" * 60)

    agent = OCIGenAIAgent(
        model_id="meta.llama-4-scout-17b-16e-instruct",
        tools=[search, calculate],
        compartment_id=os.environ.get("OCI_COMPARTMENT_ID"),
        enable_reflexion=True,
    )

    result = agent.invoke("Calculate 15 + 27 and search for weather")

    print(f"\nTotal Reasoning Steps: {len(result.reasoning_steps)}")
    for i, step in enumerate(result.reasoning_steps):
        print(f"\n--- Step {i} ---")
        print(f"Thought: {step.thought[:200]}...")
        print(f"Confidence: {step.confidence:.2f}")
        print(f"Assessment: {step.assessment}")
        print(f"Tools executed: {len(step.tool_executions)}")
        for exec in step.tool_executions:
            print(f"  - {exec.tool_name}: {'success' if exec.success else 'failed'}")


if __name__ == "__main__":
    # Run examples (requires OCI credentials)
    print("OCIGenAIAgent Examples")
    print("=" * 60)
    print("Note: These examples require OCI credentials and a valid")
    print("compartment ID. Set OCI_COMPARTMENT_ID and OCI_GENAI_ENDPOINT")
    print("environment variables before running.")
    print("=" * 60)

    # Uncomment to run examples:
    # basic_usage()
    # streaming_usage()
    # with_custom_termination()
    # lcel_chain_usage()
    # langgraph_node_usage()
    # inspect_reasoning()

    # For now, just show the agent can be created
    print("\nCreating OCIGenAIAgent instance (mock mode)...")
    try:
        agent = OCIGenAIAgent(
            model_id="meta.llama-4-scout-17b-16e-instruct",
            tools=[search, calculate],
            compartment_id="mock-compartment",
        )
        print(f"Agent created: {agent}")
    except Exception as e:
        print(f"Note: Agent creation requires valid OCI config: {e}")
