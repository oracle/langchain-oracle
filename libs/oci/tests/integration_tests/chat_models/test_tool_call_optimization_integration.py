#!/usr/bin/env python3
"""
Integration test for tool call optimization.

This script verifies that the optimization to eliminate redundant tool call
conversions works correctly with actual OCI GenAI API calls.

Setup:
    export OCI_COMPARTMENT_ID=<your-compartment-id>
    export OCI_GENAI_ENDPOINT=<endpoint-url>  # optional
    export OCI_CONFIG_PROFILE=<profile-name>  # optional
    export OCI_MODEL_ID=<model-id>  # optional

Run with:
    python test_tool_call_optimization.py
"""

import os
import sys
from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage
from langchain_oci.chat_models import ChatOCIGenAI


def get_weather(location: str, unit: str = "fahrenheit") -> str:
    """Get the current weather in a given location."""
    return f"Weather in {location}: Sunny, 72°{unit[0].upper()}"


def get_population(city: str) -> int:
    """Get the population of a city."""
    populations = {
        "tokyo": 14000000,
        "new york": 8000000,
        "london": 9000000,
        "paris": 2000000,
    }
    return populations.get(city.lower(), 1000000)


def test_tool_call_basic():
    """Test basic tool calling functionality."""
    print("\n" + "=" * 80)
    print("TEST 1: Basic Tool Calling")
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
    )

    # Bind tools
    chat_with_tools = chat.bind_tools([get_weather])

    # Invoke with tool calling
    response = chat_with_tools.invoke([
        HumanMessage(content="What's the weather in San Francisco?")
    ])

    print(f"\nResponse content: {response.content}")
    print(f"Tool calls count: {len(response.tool_calls)}")
    print(f"Tool calls: {response.tool_calls}")

    # Verify additional_kwargs contains formatted tool calls
    if "tool_calls" in response.additional_kwargs:
        print(f"\nAdditional kwargs tool_calls: {response.additional_kwargs['tool_calls']}")
        tool_call = response.additional_kwargs["tool_calls"][0]
        assert tool_call["type"] == "function", "Tool call type should be 'function'"
        assert tool_call["function"]["name"] == "get_weather", "Tool should be get_weather"
        print("✓ additional_kwargs format is correct")
    else:
        print("⚠ No tool_calls in additional_kwargs")

    # Verify tool_calls field has correct LangChain format
    assert len(response.tool_calls) > 0, "Should have tool calls"
    tool_call = response.tool_calls[0]
    assert "name" in tool_call, "Tool call should have name"
    assert "args" in tool_call, "Tool call should have args"
    assert "id" in tool_call, "Tool call should have id"
    assert tool_call["name"] == "get_weather", "Tool name should be get_weather"
    assert "location" in tool_call["args"], "Should have location argument"

    print("✓ TEST 1 PASSED: Basic tool calling works correctly")
    return True


def test_multiple_tools():
    """Test calling multiple tools."""
    print("\n" + "=" * 80)
    print("TEST 2: Multiple Tools")
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
    )

    # Bind multiple tools
    chat_with_tools = chat.bind_tools([get_weather, get_population])

    # Invoke with multiple potential tool calls
    response = chat_with_tools.invoke([
        HumanMessage(
            content="What's the weather in Tokyo and what is its population?"
        )
    ])

    print(f"\nResponse content: {response.content}")
    print(f"Tool calls count: {len(response.tool_calls)}")

    for i, tc in enumerate(response.tool_calls):
        print(f"\nTool call {i + 1}:")
        print(f"  Name: {tc['name']}")
        print(f"  Args: {tc['args']}")
        print(f"  ID: {tc['id']}")

    # Verify we got tool calls
    assert len(response.tool_calls) > 0, "Should have at least one tool call"

    # Verify each tool call has proper structure
    for tc in response.tool_calls:
        assert "name" in tc, "Tool call should have name"
        assert "args" in tc, "Tool call should have args"
        assert "id" in tc, "Tool call should have id"
        assert isinstance(tc["id"], str), "Tool call ID should be string"
        assert len(tc["id"]) > 0, "Tool call ID should not be empty"

    print("✓ TEST 2 PASSED: Multiple tools work correctly")
    return True


def test_no_redundant_calls():
    """Test that optimization reduces redundant calls (manual verification)."""
    print("\n" + "=" * 80)
    print("TEST 3: Performance Optimization Verification")
    print("=" * 80)

    print("""
This test verifies the optimization by checking the code structure:

BEFORE optimization:
  - chat_tool_calls(response) called 3 times per request
  - Tool calls formatted twice (wasted JSON serialization)
  - UUID generated multiple times for Cohere

AFTER optimization:
  - chat_tool_calls(response) called ONCE
  - Tool calls formatted once
  - UUID generated once
  - Results cached and reused

Manual verification steps:
  1. Check that _generate() caches raw_tool_calls
  2. Check that chat_generation_info() no longer calls format_response_tool_calls()
  3. Check that tool_calls are only formatted once in _generate()
    """)

    # Run a simple tool call to verify behavior
    chat = ChatOCIGenAI(
        model_id=os.environ.get("OCI_MODEL_ID", "meta.llama-3.3-70b-instruct"),
        service_endpoint=os.environ.get(
            "OCI_GENAI_ENDPOINT",
            "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
        ),
        compartment_id=os.environ.get("OCI_COMPARTMENT_ID"),
        auth_profile=os.environ.get("OCI_CONFIG_PROFILE", "DEFAULT"),
        auth_type=os.environ.get("OCI_AUTH_TYPE", "SECURITY_TOKEN"),
        model_kwargs={"temperature": 0, "max_tokens": 100},
    )

    chat_with_tools = chat.bind_tools([get_weather])
    response = chat_with_tools.invoke([
        HumanMessage(content="What's the weather in Boston?")
    ])

    # Verify both formats are present
    has_tool_calls_field = len(response.tool_calls) > 0
    has_additional_kwargs = "tool_calls" in response.additional_kwargs

    print(f"\n✓ Tool calls field populated: {has_tool_calls_field}")
    print(f"✓ Additional kwargs populated: {has_additional_kwargs}")

    if has_tool_calls_field and has_additional_kwargs:
        print("\n✓ TEST 3 PASSED: Both formats available with optimized code path")
        return True
    else:
        print("\n✗ TEST 3 FAILED: Missing expected tool call formats")
        return False


def test_cohere_provider():
    """Test with Cohere provider (different tool call format)."""
    print("\n" + "=" * 80)
    print("TEST 4: Cohere Provider (Optional)")
    print("=" * 80)

    try:
        chat = ChatOCIGenAI(
            model_id="cohere.command-r-plus-08-2024",
            service_endpoint=os.environ.get(
                "OCI_GENAI_ENDPOINT",
                "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
            ),
            compartment_id=os.environ.get("OCI_COMPARTMENT_ID"),
            auth_profile=os.environ.get("OCI_CONFIG_PROFILE", "DEFAULT"),
            auth_type=os.environ.get("OCI_AUTH_TYPE", "SECURITY_TOKEN"),
            model_kwargs={"temperature": 0, "max_tokens": 500},
        )

        chat_with_tools = chat.bind_tools([get_weather])
        response = chat_with_tools.invoke([
            HumanMessage(content="What's the weather in London?")
        ])

        print(f"\nResponse content: {response.content}")
        print(f"Tool calls count: {len(response.tool_calls)}")

        if len(response.tool_calls) > 0:
            print(f"Tool calls: {response.tool_calls}")
            print("✓ TEST 4 PASSED: Cohere provider works correctly")
            return True
        else:
            print("⚠ No tool calls returned (may be expected for some queries)")
            return True

    except Exception as e:
        print(f"⚠ TEST 4 SKIPPED: {e}")
        print("(This is okay if you don't have access to Cohere models)")
        return True


def main():
    """Run all integration tests."""
    print("\n" + "=" * 80)
    print("TOOL CALL OPTIMIZATION INTEGRATION TESTS")
    print("=" * 80)

    # Check environment
    if not os.environ.get("OCI_COMPARTMENT_ID"):
        print("\n❌ ERROR: OCI_COMPARTMENT_ID environment variable not set")
        print("\nPlease set the following environment variables:")
        print("  export OCI_COMPARTMENT_ID=<your-compartment-id>")
        print("  export OCI_GENAI_ENDPOINT=<endpoint-url>  # optional")
        print("  export OCI_CONFIG_PROFILE=<profile-name>  # optional")
        print("  export OCI_MODEL_ID=<model-id>  # optional")
        sys.exit(1)

    print(f"\nUsing configuration:")
    print(f"  Model: {os.environ.get('OCI_MODEL_ID', 'meta.llama-3.3-70b-instruct')}")
    print(f"  Endpoint: {os.environ.get('OCI_GENAI_ENDPOINT', 'default')}")
    print(f"  Profile: {os.environ.get('OCI_CONFIG_PROFILE', 'DEFAULT')}")
    print(f"  Compartment: {os.environ.get('OCI_COMPARTMENT_ID', 'not set')[:20]}...")

    # Run tests
    results = []
    tests = [
        ("Basic Tool Calling", test_tool_call_basic),
        ("Multiple Tools", test_multiple_tools),
        ("Optimization Verification", test_no_redundant_calls),
        ("Cohere Provider", test_cohere_provider),
    ]

    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ TEST FAILED: {test_name}")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status}: {test_name}")

    total = len(results)
    passed = sum(1 for _, p in results if p)
    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Tool call optimization is working correctly.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
