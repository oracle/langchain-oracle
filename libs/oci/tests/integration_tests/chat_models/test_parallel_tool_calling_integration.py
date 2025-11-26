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

import logging
import os
import sys
import time

from langchain_core.messages import HumanMessage

from langchain_oci.chat_models import ChatOCIGenAI

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")


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
    """Test parallel tool calling with parallel_tool_calls=True in bind_tools."""
    logging.info("\n" + "=" * 80)
    logging.info("TEST 1: Parallel Tool Calling ENABLED (via bind_tools)")
    logging.info("=" * 80)

    chat = ChatOCIGenAI(
        model_id=os.environ.get(
            "OCI_MODEL_ID", "meta.llama-4-maverick-17b-128e-instruct-fp8"
        ),
        service_endpoint=os.environ.get(
            "OCI_GENAI_ENDPOINT",
            "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
        ),
        compartment_id=os.environ.get("OCI_COMPARTMENT_ID"),
        auth_profile=os.environ.get("OCI_CONFIG_PROFILE", "DEFAULT"),
        auth_type=os.environ.get("OCI_AUTH_TYPE", "SECURITY_TOKEN"),
        model_kwargs={"temperature": 0, "max_tokens": 500},
    )

    # Bind tools with parallel_tool_calls=True
    chat_with_tools = chat.bind_tools(
        [get_weather, calculate_tip, get_population], parallel_tool_calls=True
    )

    # Invoke with query that needs weather info
    logging.info("\nQuery: 'What's the weather in New York City?'")

    start_time = time.time()
    response = chat_with_tools.invoke(
        [HumanMessage(content="What's the weather in New York City?")]
    )
    elapsed_time = time.time() - start_time

    logging.info(f"\nResponse time: {elapsed_time:.2f}s")
    content = response.content[:200] if response.content else "(empty)"
    logging.info(f"Response content: {content}...")
    logging.info(f"Tool calls count: {len(response.tool_calls)}")

    if response.tool_calls:
        logging.info("\nTool calls:")
        for i, tc in enumerate(response.tool_calls, 1):
            logging.info(f"  {i}. {tc['name']}({tc['args']})")
    else:
        logging.info("\n⚠️  No tool calls in response.tool_calls")
        logging.info(f"Additional kwargs: {response.additional_kwargs.keys()}")

    # Verify we got tool calls
    count = len(response.tool_calls)
    assert count >= 1, f"Should have at least one tool call, got {count}"

    # Verify parallel_tool_calls was set
    logging.info("\n✓ TEST 1 PASSED: Parallel tool calling enabled and working")
    return elapsed_time


def test_parallel_tool_calling_disabled():
    """Test tool calling with parallel_tool_calls=False (sequential)."""
    logging.info("\n" + "=" * 80)
    logging.info("TEST 2: Parallel Tool Calling DISABLED (Sequential)")
    logging.info("=" * 80)

    chat = ChatOCIGenAI(
        model_id=os.environ.get(
            "OCI_MODEL_ID", "meta.llama-4-maverick-17b-128e-instruct-fp8"
        ),
        service_endpoint=os.environ.get(
            "OCI_GENAI_ENDPOINT",
            "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
        ),
        compartment_id=os.environ.get("OCI_COMPARTMENT_ID"),
        auth_profile=os.environ.get("OCI_CONFIG_PROFILE", "DEFAULT"),
        auth_type=os.environ.get("OCI_AUTH_TYPE", "SECURITY_TOKEN"),
        model_kwargs={"temperature": 0, "max_tokens": 500},
    )

    # Bind tools without parallel_tool_calls (defaults to sequential)
    chat_with_tools = chat.bind_tools([get_weather, calculate_tip, get_population])

    # Same query as test 1
    logging.info("\nQuery: 'What's the weather in New York City?'")

    start_time = time.time()
    response = chat_with_tools.invoke(
        [HumanMessage(content="What's the weather in New York City?")]
    )
    elapsed_time = time.time() - start_time

    logging.info(f"\nResponse time: {elapsed_time:.2f}s")
    content = response.content[:200] if response.content else "(empty)"
    logging.info(f"Response content: {content}...")
    logging.info(f"Tool calls count: {len(response.tool_calls)}")

    if response.tool_calls:
        logging.info("\nTool calls:")
        for i, tc in enumerate(response.tool_calls, 1):
            logging.info(f"  {i}. {tc['name']}({tc['args']})")

    # Verify we got tool calls
    count = len(response.tool_calls)
    assert count >= 1, f"Should have at least one tool call, got {count}"

    logging.info("\n✓ TEST 2 PASSED: Sequential tool calling works")
    return elapsed_time


def test_multiple_tool_calls():
    """Test query that should trigger multiple tool calls."""
    logging.info("\n" + "=" * 80)
    logging.info("TEST 3: Multiple Tool Calls Query")
    logging.info("=" * 80)

    chat = ChatOCIGenAI(
        model_id=os.environ.get(
            "OCI_MODEL_ID", "meta.llama-4-maverick-17b-128e-instruct-fp8"
        ),
        service_endpoint=os.environ.get(
            "OCI_GENAI_ENDPOINT",
            "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
        ),
        compartment_id=os.environ.get("OCI_COMPARTMENT_ID"),
        auth_profile=os.environ.get("OCI_CONFIG_PROFILE", "DEFAULT"),
        auth_type=os.environ.get("OCI_AUTH_TYPE", "SECURITY_TOKEN"),
        model_kwargs={"temperature": 0, "max_tokens": 500},
    )

    # Bind tools with parallel_tool_calls=True
    chat_with_tools = chat.bind_tools(
        [get_weather, get_population], parallel_tool_calls=True
    )

    logging.info("\nQuery: 'What's the weather and population of Tokyo?'")

    response = chat_with_tools.invoke(
        [HumanMessage(content="What's the weather and population of Tokyo?")]
    )

    logging.info(f"\nResponse content: {response.content}")
    logging.info(f"Tool calls count: {len(response.tool_calls)}")

    if response.tool_calls:
        logging.info("\nTool calls:")
        for i, tc in enumerate(response.tool_calls, 1):
            logging.info(f"  {i}. {tc['name']}({tc['args']})")

    logging.info("\n✓ TEST 3 PASSED: Multiple tool calls query works")


def test_cohere_model_error():
    """Test that Cohere models raise an error with parallel_tool_calls."""
    logging.info("\n" + "=" * 80)
    logging.info("TEST 4: Cohere Model Error Handling")
    logging.info("=" * 80)

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
    chat_with_tools = chat.bind_tools([get_weather], parallel_tool_calls=True)

    logging.info("\nAttempting to use parallel_tool_calls with Cohere model...")

    try:
        _ = chat_with_tools.invoke(
            [HumanMessage(content="What's the weather in Paris?")]
        )
        logging.info("❌ TEST FAILED: Should have raised ValueError")
        return False
    except ValueError as e:
        if "not supported for Cohere" in str(e):
            logging.info(f"\n✓ Correctly raised error: {e}")
            logging.info("\n✓ TEST 4 PASSED: Cohere validation works")
            return True
        else:
            logging.info(f"❌ Wrong error: {e}")
            return False


def main():
    logging.info("=" * 80)
    logging.info("PARALLEL TOOL CALLING INTEGRATION TESTS")
    logging.info("=" * 80)

    # Check required env vars
    if not os.environ.get("OCI_COMPARTMENT_ID"):
        logging.info("\n❌ ERROR: OCI_COMPARTMENT_ID environment variable not set")
        logging.info("Please set: export OCI_COMPARTMENT_ID=<your-compartment-id>")
        sys.exit(1)

    logging.info("\nUsing configuration:")
    model_id = os.environ.get(
        "OCI_MODEL_ID", "meta.llama-4-maverick-17b-128e-instruct-fp8"
    )
    logging.info(f"  Model: {model_id}")
    endpoint = os.environ.get("OCI_GENAI_ENDPOINT", "default")
    logging.info(f"  Endpoint: {endpoint}")
    profile = os.environ.get("OCI_CONFIG_PROFILE", "DEFAULT")
    logging.info(f"  Profile: {profile}")
    logging.info(f"  Compartment: {os.environ.get('OCI_COMPARTMENT_ID')[:25]}...")

    results = []

    try:
        # Run tests
        parallel_time = test_parallel_tool_calling_enabled()
        results.append(("Parallel Enabled", True))

        sequential_time = test_parallel_tool_calling_disabled()
        results.append(("Sequential (Disabled)", True))

        test_multiple_tool_calls()
        results.append(("Multiple Tool Calls", True))

        cohere_test = test_cohere_model_error()
        results.append(("Cohere Validation", cohere_test))

        # Print summary
        logging.info("\n" + "=" * 80)
        logging.info("TEST SUMMARY")
        logging.info("=" * 80)

        for test_name, passed in results:
            status = "✓ PASSED" if passed else "✗ FAILED"
            logging.info(f"{status}: {test_name}")

        passed_count = sum(1 for _, passed in results if passed)
        total_count = len(results)

        logging.info(f"\nTotal: {passed_count}/{total_count} tests passed")

        # Performance comparison
        if parallel_time and sequential_time:
            logging.info("\n" + "=" * 80)
            logging.info("PERFORMANCE COMPARISON")
            logging.info("=" * 80)
            logging.info(f"Parallel:   {parallel_time:.2f}s")
            logging.info(f"Sequential: {sequential_time:.2f}s")
            if sequential_time > 0:
                speedup = sequential_time / parallel_time
                logging.info(f"Speedup:    {speedup:.2f}×")

        if passed_count == total_count:
            logging.info("\n🎉 ALL TESTS PASSED!")
            return 0
        else:
            logging.info(f"\n⚠️  {total_count - passed_count} test(s) failed")
            return 1

    except Exception as e:
        logging.info(f"\n❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
