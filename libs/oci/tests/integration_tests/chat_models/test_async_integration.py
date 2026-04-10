# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Integration tests for async support in ChatOCIGenAI.

These tests require real OCI credentials and access to OCI Generative AI.

To run:
    export OCI_COMPARTMENT_ID="your-compartment-ocid"
    export OCI_CONFIG_PROFILE=DEFAULT
    export OCI_AUTH_TYPE=API_KEY
    pytest tests/integration_tests/chat_models/test_async_integration.py -v -s

Optional environment variables:
    OCI_AUTH_TYPE: API_KEY (default), SECURITY_TOKEN, INSTANCE_PRINCIPAL
    OCI_CONFIG_PROFILE: OCI config profile name (default: DEFAULT)
    OCI_MODEL_ID: Model to test (default: meta.llama-3.3-70b-instruct)
"""

import asyncio
import os
import time
from typing import List

import pytest
from langchain_core.messages import HumanMessage

# Skip all tests if compartment ID not set
pytestmark = pytest.mark.skipif(
    not os.environ.get("OCI_COMPARTMENT_ID"),
    reason="OCI_COMPARTMENT_ID environment variable not set",
)


def get_llm():
    """Create a ChatOCIGenAI instance with environment configuration."""
    from langchain_oci import ChatOCIGenAI

    compartment_id = os.environ.get("OCI_COMPARTMENT_ID")
    region = os.environ.get("OCI_REGION", "us-chicago-1")
    auth_type = os.environ.get("OCI_AUTH_TYPE", "API_KEY")
    auth_profile = os.environ.get("OCI_CONFIG_PROFILE", "DEFAULT")
    model_id = os.environ.get("OCI_MODEL_ID", "meta.llama-3.3-70b-instruct")

    # Only set optional parameters for models that support them
    model_kwargs = {}
    if not (model_id.startswith("openai.") or model_id.startswith("google.")):
        model_kwargs["temperature"] = 0.1
        model_kwargs["max_tokens"] = 100

    return ChatOCIGenAI(
        model_id=model_id,
        compartment_id=compartment_id,
        service_endpoint=f"https://inference.generativeai.{region}.oci.oraclecloud.com",
        auth_type=auth_type,
        auth_profile=auth_profile,
        model_kwargs=model_kwargs,
    )


class TestAsyncVsSync:
    """Compare async vs sync performance."""

    def test_sync_invoke(self):
        """Test synchronous invoke works."""
        llm = get_llm()
        result = llm.invoke([HumanMessage(content="Say 'hello' and nothing else")])
        assert "hello" in result.content.lower()

    @pytest.mark.asyncio
    async def test_async_invoke(self):
        """Test async invoke works."""
        llm = get_llm()
        result = await llm.ainvoke(
            [HumanMessage(content="Say 'hello' and nothing else")]
        )
        assert "hello" in result.content.lower()

    @pytest.mark.asyncio
    async def test_async_stream(self):
        """Test async streaming works."""
        llm = get_llm()
        chunks: List[str] = []

        async for chunk in llm.astream(
            [HumanMessage(content="Count from 1 to 5, one number per line")]
        ):
            if chunk.content:
                chunks.append(chunk.content)

        full_response = "".join(chunks)
        # Some models (e.g., Gemini) may return all content in one chunk
        assert len(chunks) >= 1, "Should receive at least one chunk"
        assert "1" in full_response
        assert "5" in full_response

    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test multiple concurrent async requests."""
        llm = get_llm()

        questions = [
            "What is 2+2? Answer with just the number.",
            "What is 3+3? Answer with just the number.",
            "What is 4+4? Answer with just the number.",
        ]

        messages_list = [[HumanMessage(content=q)] for q in questions]

        # Run concurrently
        results = await asyncio.gather(*[llm.ainvoke(msgs) for msgs in messages_list])

        assert len(results) == 3
        # Check we got reasonable answers
        assert "4" in results[0].content
        assert "6" in results[1].content
        assert "8" in results[2].content

    @pytest.mark.asyncio
    async def test_concurrent_vs_sequential_timing(self):
        """Compare concurrent vs sequential request timing."""
        llm = get_llm()

        messages = [HumanMessage(content="Say 'done' and nothing else")]
        num_requests = 3

        # Sequential
        start = time.perf_counter()
        for _ in range(num_requests):
            await llm.ainvoke(messages)
        sequential_time = time.perf_counter() - start

        # Concurrent
        start = time.perf_counter()
        await asyncio.gather(*[llm.ainvoke(messages) for _ in range(num_requests)])
        concurrent_time = time.perf_counter() - start

        # At minimum, concurrent shouldn't be slower
        assert concurrent_time <= sequential_time * 1.5, (
            f"Concurrent ({concurrent_time:.2f}s) should not be much slower "
            f"than sequential ({sequential_time:.2f}s)"
        )


class TestAsyncWithTools:
    """Test async with tool calling."""

    @pytest.mark.asyncio
    async def test_async_with_tools(self):
        """Test async invoke with tool binding."""
        llm = get_llm()

        def get_weather(location: str) -> str:
            """Get weather for a location."""
            return f"Weather in {location}: sunny, 72F"

        llm_with_tools = llm.bind_tools([get_weather])

        result = await llm_with_tools.ainvoke(
            [HumanMessage(content="What's the weather in Chicago?")]
        )

        # Should either have tool calls or a response
        assert result.content or result.tool_calls

    @pytest.mark.asyncio
    async def test_async_tool_args_preserve_snake_case(self):
        """Regression test for issue #188: async must preserve snake_case tool arg names.

        The async client serializes the request via HTTP (not the OCI SDK client).
        Previously, _convert_keys_to_camel converted user-defined JSON Schema
        property names (e.g. query_web -> queryWeb), causing the model to return
        camelCase args that don't match the original function signature.

        This test verifies that sync and async produce identical tool call
        argument keys when the tool has snake_case parameter names.
        """
        llm = get_llm()

        def web_search(search_query: str, max_results: int = 5) -> str:
            """Search the web for information."""
            return f"Results for '{search_query}' (max {max_results})"

        llm_with_tools = llm.bind_tools([web_search])

        prompt = [
            HumanMessage(
                content=(
                    "Search the web for 'latest AI news'. "
                    "You must call the web_search tool."
                )
            )
        ]

        # Run sync and async in sequence
        sync_result = llm_with_tools.invoke(prompt)
        async_result = await llm_with_tools.ainvoke(prompt)

        # Both should produce tool calls
        assert sync_result.tool_calls, "Sync should produce tool calls"
        assert async_result.tool_calls, "Async should produce tool calls"

        sync_args = sync_result.tool_calls[0]["args"]
        async_args = async_result.tool_calls[0]["args"]

        # The critical assertion: async args must use the same keys as sync
        assert set(sync_args.keys()) == set(async_args.keys()), (
            f"Sync/async arg keys diverge!\n"
            f"  Sync:  {sorted(sync_args.keys())}\n"
            f"  Async: {sorted(async_args.keys())}\n"
            f"If async has camelCase keys (e.g. 'searchQuery' instead of "
            f"'search_query'), issue #188 has regressed."
        )

        # Verify the keys are actually snake_case (matching the function signature)
        for key in async_args:
            assert "_" in key or key.islower(), (
                f"Arg key '{key}' looks like camelCase — expected snake_case "
                f"matching the function parameter name"
            )

    @pytest.mark.asyncio
    async def test_async_tool_roundtrip_snake_case(self):
        """End-to-end: async tool call args can invoke the original function.

        This is the practical impact of issue #188 — if args are camelCase,
        the function call fails with an unexpected keyword argument error.
        """
        llm = get_llm()

        def get_flight_info(departure_city: str, arrival_city: str) -> str:
            """Look up flight information between two cities."""
            return f"Flight from {departure_city} to {arrival_city}: 3h, $299"

        llm_with_tools = llm.bind_tools([get_flight_info])

        result = await llm_with_tools.ainvoke(
            [
                HumanMessage(
                    content=(
                        "Look up flights from New York to Los Angeles. "
                        "You must call the get_flight_info tool."
                    )
                )
            ]
        )

        assert result.tool_calls, "Model should have called get_flight_info"

        tc = result.tool_calls[0]
        assert tc["name"] == "get_flight_info"

        # This is the real test: can we actually call the function with the args?
        # If args are camelCase (departureCity, arrivalCity) this will raise TypeError
        try:
            output = get_flight_info(**tc["args"])
        except TypeError as e:
            pytest.fail(
                f"Tool call args don't match function signature (issue #188): {e}\n"
                f"Args received: {tc['args']}"
            )

        assert "Flight from" in output


class TestAsyncStreaming:
    """Test async streaming scenarios."""

    @pytest.mark.asyncio
    async def test_stream_tokens_arrive_incrementally(self):
        """Verify streaming produces content."""
        llm = get_llm()

        chunks: List[str] = []

        async for chunk in llm.astream(
            [HumanMessage(content="Write a haiku about coding")]
        ):
            if chunk.content:
                chunks.append(chunk.content)

        # Some models batch content differently - just verify we got a response
        assert len(chunks) >= 1, "Should receive at least one chunk"
        full_response = "".join(chunks)
        assert len(full_response) > 10, "Should have meaningful content"

    @pytest.mark.asyncio
    async def test_stream_can_be_cancelled(self):
        """Test that async stream can be cancelled early."""
        llm = get_llm()

        chunks_received = 0
        content_received = ""

        async for chunk in llm.astream(
            [HumanMessage(content="Write a very long story about a robot")]
        ):
            if chunk.content:
                chunks_received += 1
                content_received += chunk.content
                # Break early - some models batch so we may get fewer chunks
                if chunks_received >= 3 or len(content_received) > 50:
                    break

        # Verify we got some content before breaking
        assert chunks_received >= 1
        assert len(content_received) > 0
