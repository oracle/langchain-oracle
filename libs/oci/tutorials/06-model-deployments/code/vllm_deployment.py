# Tutorial 06: vLLM Deployment Example
# Demonstrates ChatOCIModelDeploymentVLLM for high-throughput inference

import ads
from langchain_core.messages import HumanMessage, SystemMessage

from langchain_oci import ChatOCIModelDeploymentVLLM

# Configure authentication (uses ~/.oci/config by default)
ads.set_auth("api_key")

# Replace with your deployment endpoint
ENDPOINT = "https://modeldeployment.us-ashburn-1.oci.customer-oci.com/<ocid>/predict"


def basic_chat():
    """Basic chat with vLLM deployment."""
    print("Basic vLLM Chat")
    print("=" * 50)

    chat = ChatOCIModelDeploymentVLLM(
        endpoint=ENDPOINT,
        model="odsc-llm",
        temperature=0.2,
        max_tokens=512,
        top_p=0.95,
    )

    messages = [
        SystemMessage(content="You are a helpful coding assistant."),
        HumanMessage(content="Write a Python function to check if a number is prime."),
    ]

    response = chat.invoke(messages)
    print(response.content)


def streaming_chat():
    """Streaming response from vLLM deployment."""
    print("\nStreaming vLLM Chat")
    print("=" * 50)

    chat = ChatOCIModelDeploymentVLLM(
        endpoint=ENDPOINT,
        model="odsc-llm",
        streaming=True,
        temperature=0.7,
        max_tokens=256,
    )

    print("Response: ", end="")
    for chunk in chat.stream("Tell me a short story about a robot learning to paint."):
        print(chunk.content, end="", flush=True)
    print()


def advanced_parameters():
    """Using advanced vLLM sampling parameters."""
    print("\nAdvanced vLLM Parameters")
    print("=" * 50)

    chat = ChatOCIModelDeploymentVLLM(
        endpoint=ENDPOINT,
        model="odsc-llm",
        # Sampling parameters
        temperature=0.8,
        top_p=0.9,
        top_k=40,
        # Penalties
        frequency_penalty=0.2,
        presence_penalty=0.1,
        repetition_penalty=1.1,
        # Token control
        max_tokens=200,
        min_tokens=50,
        # Output control
        skip_special_tokens=True,
        spaces_between_special_tokens=True,
    )

    response = chat.invoke("Generate a creative product name for a smart water bottle.")
    print(f"Generated name: {response.content}")


def beam_search_generation():
    """Using beam search for more deterministic output."""
    print("\nBeam Search Generation")
    print("=" * 50)

    chat = ChatOCIModelDeploymentVLLM(
        endpoint=ENDPOINT,
        model="odsc-llm",
        use_beam_search=True,
        best_of=3,
        temperature=0.0,  # Usually 0 for beam search
        max_tokens=100,
    )

    prompt = "Translate to French: The quick brown fox jumps over the lazy dog."
    response = chat.invoke(prompt)
    print(f"Translation: {response.content}")


if __name__ == "__main__":
    print("vLLM Deployment Examples")
    print("Note: Replace ENDPOINT with your actual deployment URL")
    print()

    # Uncomment to run (requires actual deployment):
    # basic_chat()
    # streaming_chat()
    # advanced_parameters()
    # beam_search_generation()

    print("Examples are commented out - configure ENDPOINT and uncomment to run.")
