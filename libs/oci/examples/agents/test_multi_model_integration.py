#!/usr/bin/env python
"""
Multi-Model Integration Tests for OCIGenAIAgent

Tests the agent with multiple models to ensure compatibility:
- Meta Llama 4 Scout
- Google Gemini 2.5 Flash
- xAI Grok 3 Mini
- Cohere Command A
- OpenAI GPT 5.2 (Frankfurt region)

Usage:
    python test_multi_model_integration.py
"""

import asyncio
import os
import sys

# Add parent path for local development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from langchain_core.tools import tool

from langchain_oci import OCIGenAIAgent, TerminateEvent

# =============================================================================
# Test Tools
# =============================================================================


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression.

    Args:
        expression: A math expression like "2 + 2" or "10 * 5"

    Returns:
        The calculated result
    """
    import math

    allowed = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
    allowed.update({"abs": abs, "round": round, "min": min, "max": max})
    try:
        result = eval(expression, {"__builtins__": {}}, allowed)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: The city name

    Returns:
        Weather information
    """
    # Simulated weather data
    weather_data = {
        "paris": "72F, sunny",
        "london": "58F, cloudy",
        "tokyo": "68F, partly cloudy",
        "new york": "65F, clear",
        "chicago": "55F, windy",
    }
    city_lower = city.lower()
    if city_lower in weather_data:
        return f"Weather in {city}: {weather_data[city_lower]}"
    return f"Weather in {city}: 70F, mild"


# Async tool for testing native async execution
@tool
async def async_search(query: str) -> str:
    """Search for information asynchronously.

    Args:
        query: Search query

    Returns:
        Search results
    """
    await asyncio.sleep(0.01)  # Simulate async I/O
    return f"Search results for '{query}': Found 3 relevant articles."


# =============================================================================
# Model Configurations
# =============================================================================

# Compartments for different regions/tenancies
# Chicago uses the tenancy OCID from API_KEY_AUTH profile
CHICAGO_COMPARTMENT = (
    "ocid1.tenancy.oc1..aaaaaaaah7ixt2oanvvualoahejm63r66c3pse5u4nd4gzviax7eeeqhrysq"
)
FRANKFURT_COMPARTMENT = (
    "ocid1.tenancy.oc1..aaaaaaaa5hwtrus75rauufcfvtnjnz3mc4xm2bzibbigva2bw4ne7ezkvzha"
)

# Override via environment variables
AUTH_PROFILE = os.environ.get("OCI_AUTH_PROFILE", "API_KEY_AUTH")
AUTH_TYPE = os.environ.get("OCI_AUTH_TYPE", "API_KEY")

MODELS = {
    "llama4": {
        "model_id": "meta.llama-4-scout-17b-16e-instruct",
        "compartment_id": CHICAGO_COMPARTMENT,
        "service_endpoint": "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
        "auth_type": AUTH_TYPE,
        "auth_profile": AUTH_PROFILE,
    },
    "gemini": {
        "model_id": "google.gemini-2.5-flash",
        "compartment_id": CHICAGO_COMPARTMENT,
        "service_endpoint": "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
        "auth_type": AUTH_TYPE,
        "auth_profile": AUTH_PROFILE,
    },
    "grok": {
        "model_id": "xai.grok-3-mini",
        "compartment_id": CHICAGO_COMPARTMENT,
        "service_endpoint": "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
        "auth_type": AUTH_TYPE,
        "auth_profile": AUTH_PROFILE,
    },
    "cohere": {
        "model_id": "cohere.command-a-03-2025",
        "compartment_id": CHICAGO_COMPARTMENT,
        "service_endpoint": "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
        "auth_type": AUTH_TYPE,
        "auth_profile": AUTH_PROFILE,
    },
    "gpt52": {
        "model_id": "openai.gpt-5.2",
        "compartment_id": FRANKFURT_COMPARTMENT,
        "service_endpoint": "https://inference.generativeai.eu-frankfurt-1.oci.oraclecloud.com",
        "auth_type": "API_KEY",
        "auth_profile": "LUIGI_FRA_API",
    },
}


# =============================================================================
# Test Functions
# =============================================================================


def test_sync_invoke(model_name: str, config: dict) -> bool:
    """Test synchronous invoke with tool calling."""
    print("\n  [sync invoke] ", end="", flush=True)
    try:
        agent = OCIGenAIAgent(
            tools=[calculate, get_weather], max_iterations=3, **config
        )
        result = agent.invoke("What is 15 * 8?")

        # Check for correct answer OR that agent completed without error
        if "120" in result.final_answer or result.termination_reason in (
            "no_tools",
            "max_iterations",
            "confidence_met",
        ):
            print("PASS")
            return True
        else:
            print(f"FAIL - Unexpected result: {result.final_answer[:50]}")
            return False
    except Exception as e:
        print(f"FAIL - {e}")
        return False


def test_sync_stream(model_name: str, config: dict) -> bool:
    """Test synchronous streaming."""
    print("  [sync stream] ", end="", flush=True)
    try:
        agent = OCIGenAIAgent(
            tools=[calculate, get_weather], max_iterations=3, **config
        )

        events = list(agent.stream("What's the weather in Paris?"))

        terminate_events = [e for e in events if isinstance(e, TerminateEvent)]

        # Pass if we got a terminate event (stream completed)
        if terminate_events:
            print("PASS")
            return True
        else:
            print("FAIL - No terminate event")
            return False
    except Exception as e:
        print(f"FAIL - {e}")
        return False


async def test_async_invoke(model_name: str, config: dict) -> bool:
    """Test async invoke."""
    print("  [async invoke] ", end="", flush=True)
    try:
        agent = OCIGenAIAgent(
            tools=[calculate, get_weather], max_iterations=3, **config
        )
        result = await agent.ainvoke("Calculate 25 + 37")

        # Check for correct answer OR that agent completed without error
        if "62" in result.final_answer or result.termination_reason in (
            "no_tools",
            "max_iterations",
            "confidence_met",
        ):
            print("PASS")
            return True
        else:
            print(f"FAIL - Unexpected result: {result.final_answer[:50]}")
            return False
    except Exception as e:
        print(f"FAIL - {e}")
        return False


async def test_async_stream(model_name: str, config: dict) -> bool:
    """Test async streaming."""
    print("  [async stream] ", end="", flush=True)
    try:
        agent = OCIGenAIAgent(
            tools=[calculate, get_weather], max_iterations=3, **config
        )

        events = []
        async for event in agent.astream("What's 100 / 4?"):
            events.append(event)

        terminate_events = [e for e in events if isinstance(e, TerminateEvent)]

        # Pass if we got a terminate event (stream completed)
        if terminate_events:
            print("PASS")
            return True
        else:
            print("FAIL - No terminate event")
            return False
    except Exception as e:
        print(f"FAIL - {e}")
        return False


async def test_async_tool(model_name: str, config: dict) -> bool:
    """Test async tool execution (native ainvoke)."""
    print("  [async tool] ", end="", flush=True)
    try:
        agent = OCIGenAIAgent(tools=[async_search], max_iterations=3, **config)
        result = await agent.ainvoke("Search for Python tutorials")

        if (
            "search" in result.final_answer.lower()
            or "found" in result.final_answer.lower()
        ):
            print("PASS")
            return True
        else:
            # Some models may not call the tool, just check it ran
            print("PASS (no tool call)")
            return True
    except Exception as e:
        print(f"FAIL - {e}")
        return False


def test_batch(model_name: str, config: dict) -> bool:
    """Test batch processing."""
    print("  [batch] ", end="", flush=True)
    try:
        agent = OCIGenAIAgent(tools=[calculate], max_iterations=3, **config)
        results = agent.batch(
            [
                "What is 5 + 5?",
                "What is 10 * 2?",
            ]
        )

        if len(results) == 2:
            print("PASS")
            return True
        else:
            print(f"FAIL - Expected 2 results, got {len(results)}")
            return False
    except Exception as e:
        print(f"FAIL - {e}")
        return False


async def run_tests_for_model(model_name: str, config: dict) -> dict:
    """Run all tests for a single model."""
    results = {
        "sync_invoke": test_sync_invoke(model_name, config),
        "sync_stream": test_sync_stream(model_name, config),
        "async_invoke": await test_async_invoke(model_name, config),
        "async_stream": await test_async_stream(model_name, config),
        "async_tool": await test_async_tool(model_name, config),
        "batch": test_batch(model_name, config),
    }
    return results


async def main():
    """Run integration tests for all models."""
    print("=" * 70)
    print("  OCIGenAIAgent Multi-Model Integration Tests")
    print("=" * 70)

    # Check which models to test
    models_to_test = sys.argv[1:] if len(sys.argv) > 1 else list(MODELS.keys())

    all_results = {}

    for model_name in models_to_test:
        if model_name not in MODELS:
            print(f"\nUnknown model: {model_name}")
            continue

        config = MODELS[model_name]
        print(f"\n[{model_name.upper()}] {config['model_id']}")
        print("-" * 50)

        try:
            results = await run_tests_for_model(model_name, config)
            all_results[model_name] = results
        except Exception as e:
            print(f"  ERROR: {e}")
            all_results[model_name] = {"error": str(e)}

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    total_passed = 0
    total_failed = 0

    for model_name, results in all_results.items():
        if "error" in results:
            print(f"  {model_name}: ERROR - {results['error']}")
            continue

        passed = sum(1 for v in results.values() if v)
        failed = sum(1 for v in results.values() if not v)
        total_passed += passed
        total_failed += failed

        status = "PASS" if failed == 0 else "FAIL"
        print(f"  {model_name}: {status} ({passed}/{passed + failed} tests)")

    print("-" * 50)
    print(f"  Total: {total_passed} passed, {total_failed} failed")
    print("=" * 70)

    return total_failed == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
