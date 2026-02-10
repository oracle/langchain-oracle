#!/usr/bin/env python
"""Demo script to test OCIGenAIAgent new features.

Run after refreshing OCI session:
    oci session authenticate --profile-name DEFAULT --region us-ashburn-1

Usage:
    python examples/agents/test_new_features_demo.py
"""

import os
import sys

# Add parent to path for local development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from langchain_oci import OCIGenAIAgent
from langchain_oci.agents.oci_agent.hooks import (
    AgentHooks,
    IterationContext,
    ToolHookContext,
    ToolResultContext,
    create_metrics_hooks,
)


# =============================================================================
# Test Tools
# =============================================================================

@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression.

    Args:
        expression: A mathematical expression (e.g., "2 + 2").

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
    """Get weather for a city.

    Args:
        city: City name.

    Returns:
        Weather description.
    """
    data = {
        "chicago": "72°F and sunny",
        "new york": "65°F and cloudy",
        "paris": "68°F and partly cloudy",
    }
    return data.get(city.lower(), f"Weather unavailable for {city}")


@tool
def search_info(query: str) -> str:
    """Search for information.

    Args:
        query: Search query.

    Returns:
        Search results.
    """
    data = {
        "capital france": "Paris is the capital of France.",
        "capital japan": "Tokyo is the capital of Japan.",
    }
    q = query.lower()
    for key, val in data.items():
        if key in q:
            return val
    return f"No results for: {query}"


# =============================================================================
# Configuration
# =============================================================================

COMPARTMENT_ID = os.environ.get("OCI_COMPARTMENT_ID", "")
AUTH_TYPE = os.environ.get("OCI_AUTH_TYPE", "SECURITY_TOKEN")
AUTH_PROFILE = os.environ.get("OCI_AUTH_PROFILE", "DEFAULT")
REGION = os.environ.get("OCI_REGION", "us-chicago-1")
MODEL_ID = os.environ.get("OCI_MODEL_ID", "meta.llama-4-scout-17b-16e-instruct")


def test_hooks_system():
    """Test 1: Hooks System"""
    print("\n" + "=" * 60)
    print("TEST 1: HOOKS SYSTEM")
    print("=" * 60)

    # Track hook calls
    hook_log = []

    def on_tool_start(ctx: ToolHookContext):
        hook_log.append(f"[START] {ctx.tool_name}")
        print(f"  🔧 Tool starting: {ctx.tool_name}")

    def on_tool_end(ctx: ToolResultContext):
        hook_log.append(f"[END] {ctx.tool_name}")
        status = "✓" if ctx.success else "✗"
        print(f"  {status} Tool completed: {ctx.tool_name} ({ctx.duration_ms:.1f}ms)")

    def on_iteration_start(ctx: IterationContext):
        print(f"  📍 Iteration {ctx.iteration} starting...")

    def on_iteration_end(ctx: IterationContext):
        print(f"  📍 Iteration {ctx.iteration} done (confidence: {ctx.confidence:.2f})")

    def on_terminate(reason: str, answer: str):
        print(f"  🏁 Terminated: {reason}")

    hooks = AgentHooks(
        on_tool_start=[on_tool_start],
        on_tool_end=[on_tool_end],
        on_iteration_start=[on_iteration_start],
        on_iteration_end=[on_iteration_end],
        on_terminate=[on_terminate],
    )

    agent = OCIGenAIAgent(
        model_id=MODEL_ID,
        tools=[calculate],
        compartment_id=COMPARTMENT_ID,
        service_endpoint=f"https://inference.generativeai.{REGION}.oci.oraclecloud.com",
        auth_type=AUTH_TYPE,
        auth_profile=AUTH_PROFILE,
        hooks=hooks,
        max_iterations=3,
    )

    print("\nQuery: 'What is 7 * 8?'")
    result = agent.invoke("What is 7 * 8?")

    print(f"\nAnswer: {result.final_answer}")
    print(f"Hook events logged: {len(hook_log)}")
    return True


def test_metrics_hooks():
    """Test 2: Metrics Collection"""
    print("\n" + "=" * 60)
    print("TEST 2: METRICS COLLECTION")
    print("=" * 60)

    hooks, metrics = create_metrics_hooks()

    agent = OCIGenAIAgent(
        model_id=MODEL_ID,
        tools=[calculate, get_weather],
        compartment_id=COMPARTMENT_ID,
        service_endpoint=f"https://inference.generativeai.{REGION}.oci.oraclecloud.com",
        auth_type=AUTH_TYPE,
        auth_profile=AUTH_PROFILE,
        hooks=hooks,
        max_iterations=3,
    )

    print("\nQuery: 'Calculate 15 + 25'")
    result = agent.invoke("Calculate 15 + 25")

    print(f"\nAnswer: {result.final_answer}")
    print("\nCollected Metrics:")
    print(f"  - Iterations: {metrics['iterations']}")
    print(f"  - Tool calls: {metrics['total_tool_calls']}")
    print(f"  - Tool errors: {metrics['tool_errors']}")
    print(f"  - Termination: {metrics['termination_reason']}")
    if metrics['tool_durations_ms']:
        avg = sum(metrics['tool_durations_ms']) / len(metrics['tool_durations_ms'])
        print(f"  - Avg tool duration: {avg:.1f}ms")
    return True


def test_message_compression():
    """Test 3: Message Compression"""
    print("\n" + "=" * 60)
    print("TEST 3: MESSAGE COMPRESSION")
    print("=" * 60)

    agent = OCIGenAIAgent(
        model_id=MODEL_ID,
        tools=[get_weather, calculate],
        compartment_id=COMPARTMENT_ID,
        service_endpoint=f"https://inference.generativeai.{REGION}.oci.oraclecloud.com",
        auth_type=AUTH_TYPE,
        auth_profile=AUTH_PROFILE,
        enable_compression=True,
        max_messages=10,
        max_iterations=3,
    )

    print(f"\nCompression enabled: {agent.enable_compression}")
    print(f"Max messages: {agent.max_messages}")

    # Build history
    history = [
        HumanMessage(content="Hi, I need help."),
        HumanMessage(content="What's the weather in Chicago?"),
        HumanMessage(content="Thanks!"),
    ]

    print(f"History messages: {len(history)}")
    print("\nQuery: 'Now calculate 20 + 30'")

    result = agent.invoke("Now calculate 20 + 30", message_history=history)

    print(f"\nAnswer: {result.final_answer}")
    print(f"Result messages: {len(result.messages)}")
    return True


def test_confidence_signals():
    """Test 4: Confidence Signal Detection"""
    print("\n" + "=" * 60)
    print("TEST 4: CONFIDENCE SIGNAL DETECTION")
    print("=" * 60)

    agent = OCIGenAIAgent(
        model_id=MODEL_ID,
        tools=[search_info],
        compartment_id=COMPARTMENT_ID,
        service_endpoint=f"https://inference.generativeai.{REGION}.oci.oraclecloud.com",
        auth_type=AUTH_TYPE,
        auth_profile=AUTH_PROFILE,
        enable_confidence_signals=True,
        enable_reflexion=True,
        min_iterations_for_early_exit=1,
        confidence_threshold=0.3,
        max_iterations=5,
    )

    print("\nConfidence signals enabled")
    print("Detects patterns like: 'verified', 'confirmed', 'the solution is'")

    print("\nQuery: 'What is the capital of France?'")
    result = agent.invoke("What is the capital of France?")

    print(f"\nAnswer: {result.final_answer}")
    print(f"Final confidence: {result.confidence:.2f}")
    print(f"Termination reason: {result.termination_reason}")
    return True


def test_conversation_history():
    """Test 5: Multi-turn Conversation"""
    print("\n" + "=" * 60)
    print("TEST 5: MULTI-TURN CONVERSATION")
    print("=" * 60)

    agent = OCIGenAIAgent(
        model_id=MODEL_ID,
        tools=[calculate],
        compartment_id=COMPARTMENT_ID,
        service_endpoint=f"https://inference.generativeai.{REGION}.oci.oraclecloud.com",
        auth_type=AUTH_TYPE,
        auth_profile=AUTH_PROFILE,
        max_iterations=3,
    )

    # Turn 1
    print("\nTurn 1: 'What is 10 + 5?'")
    result1 = agent.invoke("What is 10 + 5?")
    print(f"  Answer: {result1.final_answer}")

    # Turn 2 with history
    history = list(result1.messages)
    print(f"\nTurn 2: 'Now double that result' (with {len(history)} history messages)")
    result2 = agent.invoke("Now double that result", message_history=history)
    print(f"  Answer: {result2.final_answer}")
    return True


def test_all_features():
    """Test 6: All Features Together"""
    print("\n" + "=" * 60)
    print("TEST 6: ALL FEATURES COMBINED")
    print("=" * 60)

    hooks, metrics = create_metrics_hooks()

    agent = OCIGenAIAgent(
        model_id=MODEL_ID,
        tools=[get_weather, calculate, search_info],
        compartment_id=COMPARTMENT_ID,
        service_endpoint=f"https://inference.generativeai.{REGION}.oci.oraclecloud.com",
        auth_type=AUTH_TYPE,
        auth_profile=AUTH_PROFILE,
        # Hooks
        hooks=hooks,
        # Compression
        enable_compression=True,
        max_messages=15,
        # Confidence
        enable_confidence_signals=True,
        min_iterations_for_early_exit=2,
        # Reflexion
        enable_reflexion=True,
        max_iterations=5,
    )

    history = [HumanMessage(content="I need help with several tasks.")]

    print("\nQuery: 'Weather in Paris and calculate 12 * 12'")
    result = agent.invoke(
        "What's the weather in Paris and calculate 12 * 12?",
        message_history=history,
    )

    print(f"\nAnswer: {result.final_answer}")
    print(f"\nFeature Results:")
    print(f"  - Hooks: {metrics['total_tool_calls']} tool calls tracked")
    print(f"  - Compression: {len(result.messages)} messages in result")
    print(f"  - Confidence: {result.confidence:.2f}")
    print(f"  - Iterations: {result.total_iterations}")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("OCIGenAIAgent New Features Integration Tests")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Model: {MODEL_ID}")
    print(f"  Region: {REGION}")
    print(f"  Auth: {AUTH_TYPE} ({AUTH_PROFILE})")

    tests = [
        ("Hooks System", test_hooks_system),
        ("Metrics Collection", test_metrics_hooks),
        ("Message Compression", test_message_compression),
        ("Confidence Signals", test_confidence_signals),
        ("Conversation History", test_conversation_history),
        ("All Features Combined", test_all_features),
    ]

    results = []
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, "PASS" if success else "FAIL"))
        except Exception as e:
            print(f"\n❌ Error: {e}")
            results.append((name, f"ERROR: {e}"))
            break  # Stop on first error

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, status in results:
        icon = "✓" if status == "PASS" else "✗"
        print(f"  {icon} {name}: {status}")

    all_passed = all(s == "PASS" for _, s in results)
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
