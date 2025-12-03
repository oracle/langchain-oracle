#!/usr/bin/env python3
"""Quick test script for OpenAI models on OCI GenAI.

This tests the rebased LangChain 1.x support with OpenAI models.

Setup:
    export OCI_COMP=<your-compartment-id>

Run:
    python test_openai_model.py
"""

import os
import sys

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_oci.chat_models import ChatOCIGenAI

# Configuration
COMPARTMENT_ID = os.environ.get("OCI_COMP")
if not COMPARTMENT_ID:
    print("ERROR: OCI_COMP environment variable not set")
    print("Set it with: export OCI_COMP=<your-compartment-id>")
    sys.exit(1)

MODEL_ID = "openai.gpt-oss-20b"  # or openai.gpt-oss-120b
ENDPOINT = "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"

print(f"Testing OpenAI model: {MODEL_ID}")
print(f"Compartment: {COMPARTMENT_ID[:50]}...")
print("-" * 60)

# Create chat model
chat = ChatOCIGenAI(
    model_id=MODEL_ID,
    service_endpoint=ENDPOINT,
    compartment_id=COMPARTMENT_ID,
    auth_type="SECURITY_TOKEN",
    auth_profile="DEFAULT",
    model_kwargs={
        "temperature": 0.7,
        "max_completion_tokens": 100,  # OpenAI uses max_completion_tokens
    },
)

# Test 1: Basic completion
print("\nTest 1: Basic completion")
print("-" * 60)
try:
    response = chat.invoke([HumanMessage(content="Say hello in 5 words")])
    print(f"✓ Response: {response.content}")
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

# Test 2: With system message
print("\nTest 2: With system message")
print("-" * 60)
try:
    response = chat.invoke([
        SystemMessage(content="You are a helpful math tutor."),
        HumanMessage(content="What is 15 * 23?")
    ])
    print(f"✓ Response: {response.content}")
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

# Test 3: Streaming
print("\nTest 3: Streaming")
print("-" * 60)
try:
    print("Response: ", end="", flush=True)
    for chunk in chat.stream([HumanMessage(content="Count from 1 to 5")]):
        print(chunk.content, end="", flush=True)
    print("\n✓ Streaming works!")
except Exception as e:
    print(f"\n✗ Error: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ All tests passed! LangChain 1.x + OpenAI model working correctly")
print("=" * 60)
