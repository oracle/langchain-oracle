# Sample 07: Batch Processing Example
# Demonstrates efficient batch processing with rate limiting

import asyncio
import time
from typing import List

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


async def process_with_rate_limit(
    llm: ChatOCIGenAI,
    prompts: List[str],
    max_concurrent: int = 5,
) -> List[str]:
    """Process prompts with concurrency limit."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def limited_invoke(prompt: str, index: int):
        async with semaphore:
            print(f"Processing {index + 1}/{len(prompts)}: {prompt[:30]}...")
            try:
                response = await llm.ainvoke(prompt)
                return (index, response.content)
            except Exception as e:
                return (index, f"Error: {e}")

    # Create tasks for all prompts
    tasks = [limited_invoke(p, i) for i, p in enumerate(prompts)]

    # Process all with limited concurrency
    completed = await asyncio.gather(*tasks)

    # Sort by original index
    completed.sort(key=lambda x: x[0])
    return [content for _, content in completed]


async def main():
    llm = ChatOCIGenAI(
        auth_profile=AUTH_PROFILE,
        model_id="meta.llama-3.3-70b-instruct",
        service_endpoint=SERVICE_ENDPOINT,
        compartment_id=COMPARTMENT_ID,
        model_kwargs={"max_tokens": 100},
    )

    # Generate test prompts
    prompts = [
        f"In one sentence, what is {topic}?"
        for topic in [
            "Python",
            "JavaScript",
            "Rust",
            "Go",
            "TypeScript",
            "Java",
            "C++",
            "Swift",
            "Kotlin",
            "Ruby",
        ]
    ]

    print(f"Processing {len(prompts)} prompts with max 3 concurrent requests")
    print("=" * 60)

    start_time = time.perf_counter()

    # Process with rate limiting
    results = await process_with_rate_limit(llm, prompts, max_concurrent=3)

    elapsed = time.perf_counter() - start_time

    print("\n" + "=" * 60)
    print("Results:")
    print("-" * 60)

    for prompt, result in zip(prompts, results):
        topic = prompt.split("what is ")[1].rstrip("?")
        print(f"{topic}: {result[:80]}...")

    print(f"\nCompleted in {elapsed:.2f} seconds")
    print(f"Average time per request: {elapsed / len(prompts):.2f} seconds")


if __name__ == "__main__":
    asyncio.run(main())
