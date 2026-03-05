# Sample 01: Multi-turn Conversation Example
# Demonstrates how to maintain conversation context

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langchain_oci import ChatOCIGenAI

import os

# Configuration - uses environment variables or defaults
COMPARTMENT_ID = os.environ.get(
    "OCI_COMPARTMENT_ID", "ocid1.compartment.oc1..your-compartment-id"
)
SERVICE_ENDPOINT = os.environ.get(
    "OCI_SERVICE_ENDPOINT",
    "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
)
AUTH_PROFILE = os.environ.get("OCI_AUTH_PROFILE", "DEFAULT")

# Create chat model
llm = ChatOCIGenAI(
    auth_profile=AUTH_PROFILE,
    model_id="meta.llama-3.3-70b-instruct",
    service_endpoint=SERVICE_ENDPOINT,
    compartment_id=COMPARTMENT_ID,
)

# Multi-turn conversation with context
messages = [
    SystemMessage(content="You are a helpful cooking assistant. Be concise."),
    HumanMessage(content="I have chicken, rice, and vegetables."),
]

# First turn
print("User: I have chicken, rice, and vegetables.")
response = llm.invoke(messages)
print(f"Assistant: {response.content}\n")

# Add assistant response to history
messages.append(AIMessage(content=response.content))

# Second turn
messages.append(HumanMessage(content="How do I make a stir-fry?"))
print("User: How do I make a stir-fry?")
response = llm.invoke(messages)
print(f"Assistant: {response.content}\n")

# Add to history and continue
messages.append(AIMessage(content=response.content))
messages.append(HumanMessage(content="What sauce should I use?"))
print("User: What sauce should I use?")
response = llm.invoke(messages)
print(f"Assistant: {response.content}")
