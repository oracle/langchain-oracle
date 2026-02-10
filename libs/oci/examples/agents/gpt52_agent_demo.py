#!/usr/bin/env python
"""
OCIGenAIAgent Demo with OpenAI GPT 5.2 on OCI

This demo showcases the full power of OCIGenAIAgent using OpenAI's GPT 5.2
model available through Oracle Cloud Infrastructure's Generative AI service.

Features demonstrated:
- Tool calling with GPT 5.2
- Real-time event streaming
- Hooks for observability
- Multi-turn conversations
- Confidence-based early exit

Requirements:
    pip install langchain-oci

Usage:
    export OCI_COMPARTMENT_ID="your-compartment-ocid"
    python gpt52_agent_demo.py
"""

import os
import sys
from datetime import datetime

# Add parent path for local development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from langchain_core.tools import tool

from langchain_oci import (
    AgentHooks,
    IterationContext,
    OCIGenAIAgent,
    ReflectEvent,
    TerminateEvent,
    ThinkEvent,
    ToolCompleteEvent,
    ToolHookContext,
    ToolResultContext,
    ToolStartEvent,
    create_metrics_hooks,
)

# =============================================================================
# Define Tools for the Agent
# =============================================================================


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression.

    Args:
        expression: A math expression like "2 + 2" or "sqrt(16) * 3"

    Returns:
        The calculated result
    """
    import math

    # Safe evaluation with math functions
    allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
    allowed_names.update({"abs": abs, "round": round, "min": min, "max": max})

    try:
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"Result: {result}"
    except Exception as e:
        return f"Error calculating: {e}"


@tool
def get_stock_quote(symbol: str) -> dict:
    """Get real-time stock quote information.

    Args:
        symbol: Stock ticker symbol (e.g., ORCL, GOOG, AAPL)

    Returns:
        Stock quote with price, change, and volume
    """
    # Simulated stock data
    quotes = {
        "ORCL": {
            "price": 178.50,
            "change": 2.34,
            "change_pct": 1.33,
            "volume": "12.5M",
        },
        "GOOG": {
            "price": 175.20,
            "change": -1.15,
            "change_pct": -0.65,
            "volume": "8.2M",
        },
        "AAPL": {
            "price": 225.80,
            "change": 0.50,
            "change_pct": 0.22,
            "volume": "45.1M",
        },
        "MSFT": {
            "price": 448.90,
            "change": 3.20,
            "change_pct": 0.72,
            "volume": "18.7M",
        },
        "NVDA": {
            "price": 142.30,
            "change": 5.40,
            "change_pct": 3.94,
            "volume": "52.3M",
        },
    }

    symbol = symbol.upper()
    if symbol in quotes:
        q = quotes[symbol]
        return {
            "symbol": symbol,
            "price": f"${q['price']:.2f}",
            "change": f"{'+' if q['change'] > 0 else ''}{q['change']:.2f}",
            "change_percent": f"{'+' if q['change_pct'] > 0 else ''}{q['change_pct']:.2f}%",
            "volume": q["volume"],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    return {"error": f"Symbol '{symbol}' not found"}


@tool
def search_knowledge_base(query: str, category: str = "all") -> list:
    """Search the internal knowledge base for information.

    Args:
        query: Search query
        category: Category filter (all, products, support, docs)

    Returns:
        List of matching knowledge base articles
    """
    # Simulated knowledge base
    kb = [
        {
            "id": 1,
            "category": "products",
            "title": "Oracle Cloud Infrastructure Overview",
            "summary": "OCI provides enterprise-grade cloud services including compute, storage, and AI.",
        },
        {
            "id": 2,
            "category": "products",
            "title": "OCI Generative AI Service",
            "summary": "Access foundation models from Meta, Cohere, OpenAI, and Google on OCI.",
        },
        {
            "id": 3,
            "category": "support",
            "title": "Getting Started with OCI",
            "summary": "Step-by-step guide to creating your first OCI resources.",
        },
        {
            "id": 4,
            "category": "docs",
            "title": "LangChain OCI Integration",
            "summary": "Use LangChain with OCI for building AI applications.",
        },
        {
            "id": 5,
            "category": "docs",
            "title": "OCIGenAIAgent Documentation",
            "summary": "Full reference for the OCIGenAIAgent class and features.",
        },
    ]

    query_lower = query.lower()
    results = []
    for article in kb:
        if category != "all" and article["category"] != category:
            continue
        if (
            query_lower in article["title"].lower()
            or query_lower in article["summary"].lower()
        ):
            results.append(article)

    return results if results else [{"message": f"No results found for '{query}'"}]


@tool
def create_reminder(title: str, due_date: str, priority: str = "medium") -> dict:
    """Create a reminder or task.

    Args:
        title: Reminder title
        due_date: Due date (e.g., "tomorrow", "2024-01-15")
        priority: Priority level (low, medium, high)

    Returns:
        Created reminder details
    """
    return {
        "status": "created",
        "reminder": {
            "id": "rem_" + datetime.now().strftime("%Y%m%d%H%M%S"),
            "title": title,
            "due_date": due_date,
            "priority": priority,
            "created_at": datetime.now().isoformat(),
        },
    }


# =============================================================================
# Demo Functions
# =============================================================================


def demo_basic_invocation(agent):
    """Demo 1: Basic tool calling with GPT 5.2."""
    print("\n" + "=" * 70)
    print("DEMO 1: Basic Tool Calling with GPT 5.2")
    print("=" * 70)

    query = "What's Oracle's stock price and calculate how much 100 shares would cost?"
    print(f"\nQuery: {query}\n")

    result = agent.invoke(query)

    print(f"Answer: {result.final_answer}")
    print(
        f"\nStats: {result.total_iterations} iterations, {result.total_tool_calls} tool calls"
    )


def demo_streaming_events(agent):
    """Demo 2: Real-time event streaming."""
    print("\n" + "=" * 70)
    print("DEMO 2: Real-Time Event Streaming")
    print("=" * 70)

    query = "Search for information about OCI AI services and create a reminder to review it tomorrow"
    print(f"\nQuery: {query}\n")
    print("-" * 50)

    for event in agent.stream(query):
        if isinstance(event, ThinkEvent):
            thought = event.thought[:100] if event.thought else ""
            print(f"🧠 [Think] {thought}...")
        elif isinstance(event, ToolStartEvent):
            print(f"🔧 [Tool] Calling {event.tool_name}...")
        elif isinstance(event, ToolCompleteEvent):
            status = "✅" if event.success else "❌"
            print(
                f"{status} [Result] {event.tool_name} completed in {event.duration_ms:.0f}ms"
            )
        elif isinstance(event, ReflectEvent):
            print(f"🔍 [Reflect] Confidence: {event.confidence:.2f}")
        elif isinstance(event, TerminateEvent):
            print(f"\n🏁 [Done] Reason: {event.reason}")
            print(f"\nFinal Answer:\n{event.final_answer}")


def demo_with_hooks(agent_config):
    """Demo 3: Custom hooks for observability."""
    print("\n" + "=" * 70)
    print("DEMO 3: Custom Hooks for Observability")
    print("=" * 70)

    # Track all events
    event_log = []

    def on_tool_start(ctx: ToolHookContext):
        event_log.append(f"START: {ctx.tool_name}")
        print(f"  📍 Hook: Tool '{ctx.tool_name}' starting with {ctx.arguments}")

    def on_tool_end(ctx: ToolResultContext):
        event_log.append(f"END: {ctx.tool_name} ({ctx.duration_ms:.0f}ms)")
        status = "succeeded" if ctx.success else "failed"
        print(f"  📍 Hook: Tool '{ctx.tool_name}' {status} in {ctx.duration_ms:.0f}ms")

    def on_iteration(ctx: IterationContext):
        print(
            f"  📍 Hook: Iteration {ctx.iteration} complete, confidence: {ctx.confidence:.2f}"
        )

    def on_terminate(reason: str, answer: str):
        print(f"  📍 Hook: Agent terminated - {reason}")

    hooks = AgentHooks(
        on_tool_start=[on_tool_start],
        on_tool_end=[on_tool_end],
        on_iteration_end=[on_iteration],
        on_terminate=[on_terminate],
    )

    agent = OCIGenAIAgent(**agent_config, hooks=hooks)

    query = "Calculate sqrt(144) + sqrt(256)"
    print(f"\nQuery: {query}\n")

    result = agent.invoke(query)

    print(f"\nAnswer: {result.final_answer}")
    print(f"Events logged: {event_log}")


def demo_metrics_collection(agent_config):
    """Demo 4: Automatic metrics collection."""
    print("\n" + "=" * 70)
    print("DEMO 4: Automatic Metrics Collection")
    print("=" * 70)

    hooks, metrics = create_metrics_hooks()
    agent = OCIGenAIAgent(**agent_config, hooks=hooks)

    queries = [
        "What's NVDA stock price?",
        "Calculate 15 * 8 + 42",
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        result = agent.invoke(query)
        print(f"Answer: {result.final_answer[:80]}...")

    print("\n" + "-" * 50)
    print("Collected Metrics:")
    print(f"  • Total iterations: {metrics['iterations']}")
    print(f"  • Total tool calls: {metrics['total_tool_calls']}")
    print(f"  • Tool errors: {metrics['tool_errors']}")
    print(f"  • Last termination: {metrics['termination_reason']}")

    if metrics["tool_durations_ms"]:
        avg_duration = sum(metrics["tool_durations_ms"]) / len(
            metrics["tool_durations_ms"]
        )
        print(f"  • Avg tool duration: {avg_duration:.1f}ms")


def demo_multi_turn_conversation(agent):
    """Demo 5: Multi-turn conversation with history."""
    print("\n" + "=" * 70)
    print("DEMO 5: Multi-Turn Conversation")
    print("=" * 70)

    # Turn 1
    print("\n--- Turn 1 ---")
    query1 = "What's Apple's stock price?"
    print(f"User: {query1}")
    result1 = agent.invoke(query1)
    print(f"Agent: {result1.final_answer}")

    # Turn 2 (with history)
    print("\n--- Turn 2 ---")
    history = [
        {"role": "user", "content": query1},
        {"role": "assistant", "content": result1.final_answer},
    ]
    query2 = "Now calculate how much 50 shares would cost"
    print(f"User: {query2}")
    result2 = agent.invoke(query2, message_history=history)
    print(f"Agent: {result2.final_answer}")


def demo_checkpointing(agent_config):
    """Demo 6: Persistent conversations with checkpointing."""
    print("\n" + "=" * 70)
    print("DEMO 6: Checkpointing (Persistent Conversations)")
    print("=" * 70)

    from langchain_oci import MemoryCheckpointer

    # Create checkpointer for persistent conversations
    checkpointer = MemoryCheckpointer()

    agent = OCIGenAIAgent(**agent_config, checkpointer=checkpointer)

    thread_id = "user-session-456"

    # First turn
    print("\n--- Turn 1 (saved to checkpoint) ---")
    print("User: What's Oracle's stock price?")
    result1 = agent.invoke("What's Oracle's stock price?", thread_id=thread_id)
    print(f"Agent: {result1.final_answer}")

    # Show checkpoint was saved
    checkpoint = checkpointer.get(thread_id)
    print(f"\nCheckpoint saved: {checkpoint.id}")
    print(f"Messages stored: {len(checkpoint.messages)}")

    # Second turn - automatically restores from checkpoint
    print("\n--- Turn 2 (restored from checkpoint) ---")
    print("User: Calculate 100 shares of that")
    result2 = agent.invoke("Calculate 100 shares of that", thread_id=thread_id)
    print(f"Agent: {result2.final_answer}")

    # Show checkpoint history
    print(f"\nCheckpoint count: {checkpointer.checkpoint_count}")
    print("Checkpoint history:")
    for ckpt in checkpointer.list(thread_id):
        print(f"  - {ckpt.id}: {len(ckpt.messages)} messages")


# =============================================================================
# Main
# =============================================================================


def main():
    """Run all demos."""
    print("=" * 70)
    print("  OCIGenAIAgent Demo with OpenAI GPT 5.2")
    print("  Oracle Cloud Infrastructure Generative AI Service")
    print("=" * 70)

    # Configuration
    compartment_id = os.environ.get(
        "OCI_COMPARTMENT_ID",
        "ocid1.tenancy.oc1..aaaaaaaa5hwtrus75rauufcfvtnjnz3mc4xm2bzibbigva2bw4ne7ezkvzha",
    )

    agent_config = {
        "model_id": "openai.gpt-5.2",
        "tools": [calculate, get_stock_quote, search_knowledge_base, create_reminder],
        "compartment_id": compartment_id,
        "service_endpoint": "https://inference.generativeai.eu-frankfurt-1.oci.oraclecloud.com",
        "auth_type": "API_KEY",
        "auth_profile": "LUIGI_FRA_API",
        "max_iterations": 5,
        "enable_confidence_signals": True,
    }

    print(f"\nModel: {agent_config['model_id']}")
    print(f"Tools: {[t.name for t in agent_config['tools']]}")
    print("Region: eu-frankfurt-1")

    # Create main agent
    agent = OCIGenAIAgent(**agent_config)

    # Run demos
    demo_basic_invocation(agent)
    demo_streaming_events(agent)
    demo_with_hooks(agent_config)
    demo_metrics_collection(agent_config)
    demo_multi_turn_conversation(agent)
    demo_checkpointing(agent_config)

    print("\n" + "=" * 70)
    print("  All demos completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
