#!/usr/bin/env python3
"""
Integration test for parallel tool calling feature.

This script tests parallel tool calling with actual OCI GenAI API calls.

Setup:
    export OCI_COMPARTMENT_ID=<your-compartment-id>
    export OCI_GENAI_ENDPOINT=<endpoint-url>  # optional
    export OCI_CONFIG_PROFILE=<profile-name>  # optional
    export OCI_AUTH_TYPE=<auth-type>  # optional

Run with:
    python test_parallel_tool_calling_integration.py
"""

import os
import sys
import time
from typing import List

from langchain_core.messages import HumanMessage
from langchain_oci.chat_models import ChatOCIGenAI


def get_weather(city: str, unit: str = "fahrenheit") -> str:
    """Get the current weather in a given location."""
    # Simulate API delay
    time.sleep(0.5)
    return f"Weather in {city}: Sunny, 72°{unit[0].upper()}"


def calculate_tip(amount: float, percent: float = 15.0) -> float:
    """Calculate tip amount."""
    # Simulate API delay
    time.sleep(0.5)
    return round(amount * (percent / 100), 2)


def get_population(city: str) -> int:
    """Get the population of a city."""
    # Simulate API delay
    time.sleep(0.5)
    populations = {
        "tokyo": 14000000,
        "new york": 8000000,
        "london": 9000000,
        "paris": 2000000,
        "chicago": 2700000,
        "los angeles": 4000000,
    }
    return populations.get(city.lower(), 1000000)


def test_parallel_tool_calling_enabled():
    """Test parallel tool calling with parallel_tool_calls=True."""
    print("\n" + "=" * 80)
    print("TEST 1: Parallel Tool Calling ENABLED")
    print("=" * 80)

    chat = ChatOCIGenAI(
        model_id=os.environ.get("OCI_MODEL_ID", "meta.llama-3.3-70b-instruct"),
        service_endpoint=os.environ.get(
            "OCI_GENAI_ENDPOINT",
            "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
        ),
        compartment_id=os.environ.get("OCI_COMPARTMENT_ID"),
        auth_profile=os.environ.get("OCI_CONFIG_PROFILE", "DEFAULT"),
        auth_type=os.environ.get("OCI_AUTH_TYPE", "SECURITY_TOKEN"),
        model_kwargs={"temperature": 0, "max_tokens": 500},
        parallel_tool_calls=True,  # Enable parallel calling
    )

    # Bind tools
    chat_with_tools = chat.bind_tools([get_weather, calculate_tip, get_population])

    # Invoke with query that needs weather info
    print("\nQuery: 'What's the weather in New York City?'")

    start_time = time.time()
    response = chat_with_tools.invoke([
        HumanMessage(content="What's the weather in New York City?")
    ])
    elapsed_time = time.time() - start_time

    print(f"\nResponse time: {elapsed_time:.2f}s")
    print(f"Response content: {response.content[:200] if response.content else '(empty)'}...")
    print(f"Tool calls count: {len(response.tool_calls)}")

    if response.tool_calls:
        print("\nTool calls:")
        for i, tc in enumerate(response.tool_calls, 1):
            print(f"  {i}. {tc['name']}({tc['args']})")
    else:
        print("\n⚠️  No tool calls in response.tool_calls")
        print(f"Additional kwargs: {response.additional_kwargs.keys()}")

    # Verify we got tool calls
    assert len(response.tool_calls) >= 1, f"Should have at least one tool call, got {len(response.tool_calls)}"

    # Verify parallel_tool_calls was set
    print("\n✓ TEST 1 PASSED: Parallel tool calling enabled and working")
    return elapsed_time


def test_parallel_tool_calling_disabled():
    """Test tool calling with parallel_tool_calls=False (sequential)."""
    print("\n" + "=" * 80)
    print("TEST 2: Parallel Tool Calling DISABLED (Sequential)")
    print("=" * 80)

    chat = ChatOCIGenAI(
        model_id=os.environ.get("OCI_MODEL_ID", "meta.llama-3.3-70b-instruct"),
        service_endpoint=os.environ.get(
            "OCI_GENAI_ENDPOINT",
            "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
        ),
        compartment_id=os.environ.get("OCI_COMPARTMENT_ID"),
        auth_profile=os.environ.get("OCI_CONFIG_PROFILE", "DEFAULT"),
        auth_type=os.environ.get("OCI_AUTH_TYPE", "SECURITY_TOKEN"),
        model_kwargs={"temperature": 0, "max_tokens": 500},
        parallel_tool_calls=False,  # Disable parallel calling (default)
    )

    # Bind tools
    chat_with_tools = chat.bind_tools([get_weather, calculate_tip, get_population])

    # Same query as test 1
    print("\nQuery: 'What's the weather in New York City?'")

    start_time = time.time()
    response = chat_with_tools.invoke([
        HumanMessage(content="What's the weather in New York City?")
    ])
    elapsed_time = time.time() - start_time

    print(f"\nResponse time: {elapsed_time:.2f}s")
    print(f"Response content: {response.content[:200] if response.content else '(empty)'}...")
    print(f"Tool calls count: {len(response.tool_calls)}")

    if response.tool_calls:
        print("\nTool calls:")
        for i, tc in enumerate(response.tool_calls, 1):
            print(f"  {i}. {tc['name']}({tc['args']})")

    # Verify we got tool calls
    assert len(response.tool_calls) >= 1, f"Should have at least one tool call, got {len(response.tool_calls)}"

    print("\n✓ TEST 2 PASSED: Sequential tool calling works")
    return elapsed_time


def test_bind_tools_override():
    """Test that bind_tools can override class-level setting."""
    print("\n" + "=" * 80)
    print("TEST 3: bind_tools Override of Class Setting")
    print("=" * 80)

    # Create chat with parallel_tool_calls=False at class level
    chat = ChatOCIGenAI(
        model_id=os.environ.get("OCI_MODEL_ID", "meta.llama-3.3-70b-instruct"),
        service_endpoint=os.environ.get(
            "OCI_GENAI_ENDPOINT",
            "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
        ),
        compartment_id=os.environ.get("OCI_COMPARTMENT_ID"),
        auth_profile=os.environ.get("OCI_CONFIG_PROFILE", "DEFAULT"),
        auth_type=os.environ.get("OCI_AUTH_TYPE", "SECURITY_TOKEN"),
        model_kwargs={"temperature": 0, "max_tokens": 500},
        parallel_tool_calls=False,  # Class default: disabled
    )

    # Override with True in bind_tools
    chat_with_tools = chat.bind_tools(
        [get_weather, get_population],
        parallel_tool_calls=True  # Override to enable
    )

    print("\nQuery: 'What's the weather and population of Tokyo?'")

    response = chat_with_tools.invoke([
        HumanMessage(content="What's the weather and population of Tokyo?")
    ])

    print(f"\nResponse content: {response.content}")
    print(f"Tool calls count: {len(response.tool_calls)}")

    if response.tool_calls:
        print("\nTool calls:")
        for i, tc in enumerate(response.tool_calls, 1):
            print(f"  {i}. {tc['name']}({tc['args']})")

    print("\n✓ TEST 3 PASSED: bind_tools override works")


def test_cohere_model_error():
    """Test that Cohere models raise an error with parallel_tool_calls."""
    print("\n" + "=" * 80)
    print("TEST 4: Cohere Model Error Handling")
    print("=" * 80)

    chat = ChatOCIGenAI(
        model_id="cohere.command-r-plus",
        service_endpoint=os.environ.get(
            "OCI_GENAI_ENDPOINT",
            "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
        ),
        compartment_id=os.environ.get("OCI_COMPARTMENT_ID"),
        auth_profile=os.environ.get("OCI_CONFIG_PROFILE", "DEFAULT"),
        auth_type=os.environ.get("OCI_AUTH_TYPE", "SECURITY_TOKEN"),
    )

    # Try to enable parallel tool calls with Cohere (should fail)
    chat_with_tools = chat.bind_tools(
        [get_weather],
        parallel_tool_calls=True
    )

    print("\nAttempting to use parallel_tool_calls with Cohere model...")

    try:
        response = chat_with_tools.invoke([
            HumanMessage(content="What's the weather in Paris?")
        ])
        print("❌ TEST FAILED: Should have raised ValueError")
        return False
    except ValueError as e:
        if "not supported for Cohere" in str(e):
            print(f"\n✓ Correctly raised error: {e}")
            print("\n✓ TEST 4 PASSED: Cohere validation works")
            return True
        else:
            print(f"❌ Wrong error: {e}")
            return False


def main():
    print("=" * 80)
    print("PARALLEL TOOL CALLING INTEGRATION TESTS")
    print("=" * 80)

    # Check required env vars
    if not os.environ.get("OCI_COMPARTMENT_ID"):
        print("\n❌ ERROR: OCI_COMPARTMENT_ID environment variable not set")
        print("Please set: export OCI_COMPARTMENT_ID=<your-compartment-id>")
        sys.exit(1)

    print(f"\nUsing configuration:")
    print(f"  Model: {os.environ.get('OCI_MODEL_ID', 'meta.llama-3.3-70b-instruct')}")
    print(f"  Endpoint: {os.environ.get('OCI_GENAI_ENDPOINT', 'default')}")
    print(f"  Profile: {os.environ.get('OCI_CONFIG_PROFILE', 'DEFAULT')}")
    print(f"  Compartment: {os.environ.get('OCI_COMPARTMENT_ID')[:25]}...")

    results = []

    try:
        # Run tests
        parallel_time = test_parallel_tool_calling_enabled()
        results.append(("Parallel Enabled", True))

        sequential_time = test_parallel_tool_calling_disabled()
        results.append(("Sequential (Disabled)", True))

        test_bind_tools_override()
        results.append(("bind_tools Override", True))

        cohere_test = test_cohere_model_error()
        results.append(("Cohere Validation", cohere_test))

        # Print summary
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)

        for test_name, passed in results:
            status = "✓ PASSED" if passed else "✗ FAILED"
            print(f"{status}: {test_name}")

        passed_count = sum(1 for _, passed in results if passed)
        total_count = len(results)

        print(f"\nTotal: {passed_count}/{total_count} tests passed")

        # Performance comparison
        if parallel_time and sequential_time:
            print("\n" + "=" * 80)
            print("PERFORMANCE COMPARISON")
            print("=" * 80)
            print(f"Parallel:   {parallel_time:.2f}s")
            print(f"Sequential: {sequential_time:.2f}s")
            if sequential_time > 0:
                speedup = sequential_time / parallel_time
                print(f"Speedup:    {speedup:.2f}×")

        if passed_count == total_count:
            print("\n🎉 ALL TESTS PASSED!")
            return 0
        else:
            print(f"\n⚠️  {total_count - passed_count} test(s) failed")
            return 1

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
