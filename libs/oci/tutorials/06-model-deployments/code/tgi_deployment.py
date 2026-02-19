# Tutorial 06: TGI Deployment Example
# Demonstrates ChatOCIModelDeploymentTGI for Hugging Face TGI endpoints

import ads
from langchain_core.messages import HumanMessage, SystemMessage

from langchain_oci import ChatOCIModelDeploymentTGI

# Configure authentication
ads.set_auth("api_key")

# Replace with your deployment endpoint
ENDPOINT = "https://modeldeployment.us-ashburn-1.oci.customer-oci.com/<ocid>/predict"


def basic_tgi_chat():
    """Basic chat with TGI deployment."""
    print("Basic TGI Chat")
    print("=" * 50)

    chat = ChatOCIModelDeploymentTGI(
        endpoint=ENDPOINT,
        model="odsc-llm",
        temperature=0.2,
        max_tokens=512,
        top_p=0.9,
    )

    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Explain the difference between REST and GraphQL."),
    ]

    response = chat.invoke(messages)
    print(response.content)


def reproducible_generation():
    """Using seed for reproducible outputs."""
    print("\nReproducible Generation with Seed")
    print("=" * 50)

    chat = ChatOCIModelDeploymentTGI(
        endpoint=ENDPOINT,
        model="odsc-llm",
        temperature=0.5,
        max_tokens=100,
        seed=42,  # Set seed for reproducibility
    )

    # Generate twice with same seed
    prompt = "Generate a random 6-digit code:"

    response1 = chat.invoke(prompt)
    print(f"First generation:  {response1.content.strip()}")

    response2 = chat.invoke(prompt)
    print(f"Second generation: {response2.content.strip()}")

    print("(With same seed, outputs should be identical)")


def logprobs_analysis():
    """Getting log probabilities for token analysis."""
    print("\nLog Probabilities Analysis")
    print("=" * 50)

    chat = ChatOCIModelDeploymentTGI(
        endpoint=ENDPOINT,
        model="odsc-llm",
        temperature=0.0,
        max_tokens=50,
        logprobs=True,
        top_logprobs=3,
    )

    response = chat.invoke("The capital of France is")
    print(f"Response: {response.content}")

    # Access log probabilities from response metadata
    if response.response_metadata.get("logprobs"):
        print("\nTop token probabilities available in response_metadata")


def streaming_tgi():
    """Streaming with TGI deployment."""
    print("\nStreaming TGI Response")
    print("=" * 50)

    chat = ChatOCIModelDeploymentTGI(
        endpoint=ENDPOINT,
        model="odsc-llm",
        streaming=True,
        temperature=0.7,
        max_tokens=200,
    )

    print("Response: ", end="")
    for chunk in chat.stream("Write a haiku about machine learning."):
        print(chunk.content, end="", flush=True)
    print()


if __name__ == "__main__":
    print("TGI Deployment Examples")
    print("Note: Replace ENDPOINT with your actual deployment URL")
    print()

    # Uncomment to run (requires actual deployment):
    # basic_tgi_chat()
    # reproducible_generation()
    # logprobs_analysis()
    # streaming_tgi()

    print("Examples are commented out - configure ENDPOINT and uncomment to run.")
