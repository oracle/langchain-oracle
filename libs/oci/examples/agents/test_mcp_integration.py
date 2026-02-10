#!/usr/bin/env python
"""Test OCIGenAIAgent with MCP tools.

This script demonstrates using MCP (Model Context Protocol) tools with OCIGenAIAgent.
It starts a local MCP server and connects to it using langchain-mcp-adapters.

Usage:
    python examples/agents/test_mcp_integration.py
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from langchain_mcp_adapters.client import MultiServerMCPClient

from langchain_oci import (
    OCIGenAIAgent,
    TerminateEvent,
    ThinkEvent,
    ToolCompleteEvent,
    ToolStartEvent,
)

# Configuration
COMPARTMENT_ID = os.environ.get(
    "OCI_COMPARTMENT_ID",
    "ocid1.tenancy.oc1..aaaaaaaah7ixt2oanvvualoahejm63r66c3pse5u4nd4gzviax7eeeqhrysq",
)
AUTH_TYPE = os.environ.get("OCI_AUTH_TYPE", "API_KEY")
AUTH_PROFILE = os.environ.get("OCI_AUTH_PROFILE", "API_KEY_AUTH")
REGION = os.environ.get("OCI_REGION", "us-chicago-1")
MODEL_ID = os.environ.get("OCI_MODEL_ID", "google.gemini-2.5-flash")


async def test_mcp_tools():
    """Test OCIGenAIAgent with MCP tools."""
    print("=" * 60)
    print("OCIGenAIAgent + MCP Tools Integration Test")
    print("=" * 60)

    # Path to our test MCP server
    server_path = os.path.join(os.path.dirname(__file__), "mcp_test_server.py")

    # Connect to MCP server
    print(f"\n1. Connecting to MCP server: {server_path}")

    client = MultiServerMCPClient(
        {
            "test_server": {
                "command": sys.executable,
                "args": [server_path],
                "transport": "stdio",
            }
        }
    )

    # Get tools from MCP server
    tools = await client.get_tools()
    print(f"   Loaded {len(tools)} tools from MCP server:")
    for tool in tools:
        print(f"   - {tool.name}: {tool.description[:50]}...")

    # Create agent with MCP tools
    print("\n2. Creating OCIGenAIAgent with MCP tools")
    print(f"   Model: {MODEL_ID}")

    agent = OCIGenAIAgent(
        model_id=MODEL_ID,
        tools=tools,
        compartment_id=COMPARTMENT_ID,
        service_endpoint=f"https://inference.generativeai.{REGION}.oci.oraclecloud.com",
        auth_type=AUTH_TYPE,
        auth_profile=AUTH_PROFILE,
        max_iterations=5,
    )

    # Test 1: Math operations
    print("\n" + "-" * 60)
    print("Test 1: Math with MCP tools")
    print("-" * 60)
    print("Query: 'What is 15 + 27, and then multiply that by 3?'")

    for event in agent.stream("What is 15 + 27, and then multiply that by 3?"):
        if isinstance(event, ThinkEvent):
            thought = event.thought[:80] if event.thought else ""
            print(f"  [Think] {thought}...")
        elif isinstance(event, ToolStartEvent):
            print(f"  [Tool] Calling {event.tool_name}({event.arguments})")
        elif isinstance(event, ToolCompleteEvent):
            print(f"  [Result] {event.tool_name} -> {event.result}")
        elif isinstance(event, TerminateEvent):
            print(f"  [Done] {event.final_answer}")

    # Test 2: Stock lookup
    print("\n" + "-" * 60)
    print("Test 2: Stock price lookup")
    print("-" * 60)
    print("Query: 'What is Oracle's stock price?'")

    result = agent.invoke("What is Oracle's current stock price?")
    print(f"  Answer: {result.final_answer}")
    print(f"  Tools used: {result.total_tool_calls}")

    # Test 3: Database search
    print("\n" + "-" * 60)
    print("Test 3: Database search")
    print("-" * 60)
    print("Query: 'Search for cloud-related items in the database'")

    result = agent.invoke("Search the database for anything related to cloud")
    print(f"  Answer: {result.final_answer}")
    print(f"  Iterations: {result.total_iterations}")

    print("\n" + "=" * 60)
    print("MCP Integration Test Complete!")
    print("=" * 60)


async def test_multiple_models_with_mcp():
    """Test MCP tools with multiple models."""
    print("\n" + "=" * 60)
    print("Multi-Model MCP Test")
    print("=" * 60)

    server_path = os.path.join(os.path.dirname(__file__), "mcp_test_server.py")

    models = [
        ("meta.llama-4-scout-17b-16e-instruct", "Llama 4"),
        ("google.gemini-2.5-flash", "Gemini"),
        ("xai.grok-3-mini-fast", "Grok"),
    ]

    client = MultiServerMCPClient(
        {
            "test_server": {
                "command": sys.executable,
                "args": [server_path],
                "transport": "stdio",
            }
        }
    )

    tools = await client.get_tools()
    print(f"Loaded {len(tools)} MCP tools\n")

    results = []
    for model_id, name in models:
        print(f">>> Testing {name} with MCP tools...")
        try:
            agent = OCIGenAIAgent(
                model_id=model_id,
                tools=tools,
                compartment_id=COMPARTMENT_ID,
                service_endpoint=f"https://inference.generativeai.{REGION}.oci.oraclecloud.com",
                auth_type=AUTH_TYPE,
                auth_profile=AUTH_PROFILE,
                max_iterations=3,
            )
            result = agent.invoke("Add 100 and 200")
            answer = result.final_answer[:60] if result.final_answer else "(no answer)"
            print(f"    Answer: {answer}...")
            print(f"    Tools: {result.total_tool_calls}")
            results.append((name, "PASS"))
        except Exception as e:
            print(f"    ERROR: {e}")
            results.append((name, f"FAIL: {str(e)[:30]}"))

    print("\n" + "-" * 40)
    print("Results:")
    for name, status in results:
        icon = "✓" if status == "PASS" else "✗"
        print(f"  {icon} {name}: {status}")


if __name__ == "__main__":
    asyncio.run(test_mcp_tools())
    asyncio.run(test_multiple_models_with_mcp())
