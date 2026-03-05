# Sample 02: Image Analysis Example
# Demonstrates single and multi-image analysis with vision models

from langchain_core.messages import HumanMessage

from langchain_oci import ChatOCIGenAI, load_image
from langchain_oci.utils.vision import VISION_MODELS, is_vision_model

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


def list_vision_models():
    """List all available vision-capable models."""
    print("Vision-capable models:")
    for model in VISION_MODELS:
        print(f"  - {model}")


def check_model_capability(model_id: str):
    """Check if a model supports vision."""
    if is_vision_model(model_id):
        print(f"{model_id} supports vision")
    else:
        print(f"{model_id} does NOT support vision")


def analyze_single_image(image_path: str):
    """Analyze a single image."""
    llm = ChatOCIGenAI(
    auth_profile=AUTH_PROFILE,
    model_id="meta.llama-3.2-90b-vision-instruct",
        service_endpoint=SERVICE_ENDPOINT,
        compartment_id=COMPARTMENT_ID,
    )

    message = HumanMessage(
        content=[
            {"type": "text", "text": "Describe this image in detail."},
            load_image(image_path),
        ]
    )

    response = llm.invoke([message])
    print(f"Analysis: {response.content}")


def compare_images(image_path_1: str, image_path_2: str):
    """Compare two images."""
    llm = ChatOCIGenAI(
    auth_profile=AUTH_PROFILE,
    model_id="meta.llama-3.2-90b-vision-instruct",
        service_endpoint=SERVICE_ENDPOINT,
        compartment_id=COMPARTMENT_ID,
    )

    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": "Compare these two images. What are the key differences?",
            },
            load_image(image_path_1),
            load_image(image_path_2),
        ]
    )

    response = llm.invoke([message])
    print(f"Comparison: {response.content}")


if __name__ == "__main__":
    # List available vision models
    list_vision_models()

    # Check model capability
    check_model_capability("meta.llama-3.2-90b-vision-instruct")
    check_model_capability("meta.llama-3.3-70b-instruct")

    # Uncomment to analyze an image:
    # analyze_single_image("path/to/your/image.jpg")
