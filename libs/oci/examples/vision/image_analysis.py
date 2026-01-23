# Copyright (c) 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
# ruff: noqa: T201

"""Example: Image Analysis with OCI Vision Models

Demonstrates how to use ChatOCIGenAI with vision-capable models
to analyze images from URLs, local files, and base64 data.

Vision-capable models in OCI Generative AI:
- meta.llama-3.2-90b-vision-instruct
- meta.llama-3.2-11b-vision-instruct
- meta.llama-4-scout-17b-16e-instruct
- meta.llama-4-maverick-17b-128e-instruct-fp8
- openai.gpt-4o (via GenericProvider)

Usage:
    # Set environment variables or use OCI config file
    export OCI_COMPARTMENT_ID="ocid1.compartment.oc1..example"

    python image_analysis.py
"""

from langchain_core.messages import HumanMessage

from langchain_oci import ChatOCIGenAI, is_vision_model, load_image

# Configuration - update these for your environment
COMPARTMENT_ID = "ocid1.compartment.oc1..aaaaaaaaexample"
SERVICE_ENDPOINT = "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"


def get_vision_llm(model_id: str = "meta.llama-3.2-90b-vision-instruct"):
    """Create a vision-capable ChatOCIGenAI instance."""
    if not is_vision_model(model_id):
        print(f"Warning: {model_id} may not support vision inputs")

    return ChatOCIGenAI(
        model_id=model_id,
        compartment_id=COMPARTMENT_ID,
        service_endpoint=SERVICE_ENDPOINT,
    )


def analyze_url_image():
    """Analyze an image from a URL.

    This example uses a public domain image URL to demonstrate
    how to analyze images without downloading them first.
    """
    print("=" * 60)
    print("Example 1: Analyzing image from URL")
    print("=" * 60)

    llm = get_vision_llm()

    # Public domain ant image from Wikipedia
    image_url = (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/"
        "a/a7/Camponotus_flavomarginatus_ant.jpg/"
        "320px-Camponotus_flavomarginatus_ant.jpg"
    )

    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": "What animal is shown in this image? Be specific.",
            },
            {"type": "image_url", "image_url": {"url": image_url}},
        ]
    )

    response = llm.invoke([message])
    print(f"Response: {response.content}\n")
    return response


def analyze_local_image():
    """Analyze an image from a local file.

    This example demonstrates how to load a local image file
    and send it to the vision model as a base64-encoded data URL.
    """
    print("=" * 60)
    print("Example 2: Analyzing local image file")
    print("=" * 60)

    llm = get_vision_llm()

    # Load local image - replace with your actual image path
    try:
        message = HumanMessage(
            content=[
                {"type": "text", "text": "What is shown in this image?"},
                load_image("./sample_image.png"),
            ]
        )

        response = llm.invoke([message])
        print(f"Response: {response.content}\n")
        return response
    except FileNotFoundError:
        print("Note: sample_image.png not found. Skipping local image example.\n")
        return None


def analyze_with_openai_model():
    """Analyze an image using an OpenAI model via OCI.

    This example shows how to use OpenAI models (like GPT-4o)
    through OCI's GenericProvider for vision tasks.
    """
    print("=" * 60)
    print("Example 3: Using OpenAI model for vision")
    print("=" * 60)

    llm = get_vision_llm(model_id="openai.gpt-4o")

    image_url = (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/"
        "4/47/PNG_transparency_demonstration_1.png/"
        "280px-PNG_transparency_demonstration_1.png"
    )

    message = HumanMessage(
        content=[
            {"type": "text", "text": "Describe what you see in this image."},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]
    )

    response = llm.invoke([message])
    print(f"Response: {response.content}\n")
    return response


def compare_images():
    """Compare multiple images in a single request.

    Vision models can analyze multiple images at once, which is
    useful for comparison tasks, before/after analysis, etc.
    """
    print("=" * 60)
    print("Example 4: Comparing multiple images")
    print("=" * 60)

    llm = get_vision_llm(model_id="meta.llama-4-scout-17b-16e-instruct")

    # Two different images for comparison
    image1_url = (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/"
        "a/a7/Camponotus_flavomarginatus_ant.jpg/"
        "320px-Camponotus_flavomarginatus_ant.jpg"
    )
    image2_url = (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/"
        "4/4d/Apis_mellifera_Western_honey_bee.jpg/"
        "320px-Apis_mellifera_Western_honey_bee.jpg"
    )

    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": "Compare these two insects. What are the main differences?",
            },
            {"type": "image_url", "image_url": {"url": image1_url}},
            {"type": "image_url", "image_url": {"url": image2_url}},
        ]
    )

    response = llm.invoke([message])
    print(f"Response: {response.content}\n")
    return response


def streaming_vision():
    """Stream a vision model response.

    Vision models support streaming, which is useful for
    longer responses or real-time feedback.
    """
    print("=" * 60)
    print("Example 5: Streaming vision response")
    print("=" * 60)

    llm = get_vision_llm()

    image_url = (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/"
        "a/a7/Camponotus_flavomarginatus_ant.jpg/"
        "320px-Camponotus_flavomarginatus_ant.jpg"
    )

    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": "Describe this image in detail, including colors and features.",
            },
            {"type": "image_url", "image_url": {"url": image_url}},
        ]
    )

    print("Streaming response: ", end="", flush=True)
    for chunk in llm.stream([message]):
        print(chunk.content, end="", flush=True)
    print("\n")


def main():
    """Run all vision examples."""
    print("\nOCI Vision Model Examples")
    print("=" * 60)
    print()

    # Run examples
    analyze_url_image()
    analyze_local_image()
    # Uncomment if you have access to OpenAI models via OCI:
    # analyze_with_openai_model()
    compare_images()
    streaming_vision()


if __name__ == "__main__":
    main()
