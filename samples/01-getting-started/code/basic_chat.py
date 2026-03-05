# Sample 01: Basic Chat Example
# This is the simplest way to use OCI Generative AI with LangChain

import os

from langchain_oci import ChatOCIGenAI

# Configuration - uses environment variables or defaults
COMPARTMENT_ID = os.environ.get(
    "OCI_COMPARTMENT_ID", "ocid1.compartment.oc1..your-compartment-id"
)
SERVICE_ENDPOINT = os.environ.get(
    "OCI_SERVICE_ENDPOINT",
    "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
)
AUTH_PROFILE = os.environ.get("OCI_AUTH_PROFILE", "DEFAULT")

# Create the chat model
llm = ChatOCIGenAI(
    model_id="meta.llama-3.3-70b-instruct",
    service_endpoint=SERVICE_ENDPOINT,
    compartment_id=COMPARTMENT_ID,
    auth_profile=AUTH_PROFILE,
    model_kwargs={
        "temperature": 0.7,
        "max_tokens": 500,
    },
)

# Simple invocation
response = llm.invoke("What is the capital of France?")
print(f"Response: {response.content}")

# Streaming response
print("\nStreaming response:")
for chunk in llm.stream("Tell me 3 interesting facts about Paris."):
    print(chunk.content, end="", flush=True)
print()  # Newline at the end
