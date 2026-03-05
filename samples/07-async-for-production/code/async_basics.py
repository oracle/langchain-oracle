# Sample 07: Async Basics Example
# Demonstrates ainvoke, astream, and abatch

import asyncio

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


async def main():
    llm = ChatOCIGenAI(
        auth_profile=AUTH_PROFILE,
        model_id="meta.llama-3.3-70b-instruct",
        service_endpoint=SERVICE_ENDPOINT,
        compartment_id=COMPARTMENT_ID,
    )

    # 1. Single async request
    print("1. Single Async Request (ainvoke)")
    print("-" * 40)
    response = await llm.ainvoke("What is the capital of Japan?")
    print(f"Response: {response.content}\n")

    # 2. Async streaming
    print("2. Async Streaming (astream)")
    print("-" * 40)
    print("Response: ", end="")
    async for chunk in llm.astream("Count from 1 to 5"):
        print(chunk.content, end="", flush=True)
    print("\n")

    # 3. Async batch
    print("3. Async Batch (abatch)")
    print("-" * 40)
    questions = [
        "What is Python?",
        "What is Java?",
        "What is Go?",
    ]
    responses = await llm.abatch(questions)
    for q, r in zip(questions, responses):
        print(f"Q: {q}")
        print(f"A: {r.content[:100]}...\n")

    # 4. Concurrent requests with gather
    print("4. Concurrent Requests (asyncio.gather)")
    print("-" * 40)
    results = await asyncio.gather(
        llm.ainvoke("What is 2+2?"),
        llm.ainvoke("What is 3+3?"),
        llm.ainvoke("What is 4+4?"),
    )
    for i, result in enumerate(results, 1):
        print(f"Result {i}: {result.content}")


if __name__ == "__main__":
    asyncio.run(main())
