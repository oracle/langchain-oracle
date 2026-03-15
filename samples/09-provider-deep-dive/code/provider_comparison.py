# Sample 09: Provider Comparison
# Helps choose the right provider for different use cases

from langchain_oci import ChatOCIGenAI, is_vision_model
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
    """List all vision-capable models."""
    print("Vision-Capable Models")
    print("=" * 50)

    from langchain_oci import VISION_MODELS

    print("Models that support image input:")
    for model in sorted(VISION_MODELS):
        print(f"  - {model}")

    print(f"\nTotal: {len(VISION_MODELS)} vision models")


def check_model_capabilities(model_id: str):
    """Check capabilities for a specific model."""
    print(f"\nModel: {model_id}")
    print("-" * 40)

    # Vision check
    has_vision = is_vision_model(model_id)
    print(f"  Vision: {'Yes' if has_vision else 'No'}")

    # Provider detection
    if model_id.startswith("meta."):
        provider = "meta"
        features = ["Vision (3.2)", "Parallel tools (4+)", "tool_choice"]
    elif model_id.startswith("google."):
        provider = "google"
        features = ["Vision", "PDF", "Video", "Audio"]
    elif model_id.startswith("cohere."):
        provider = "cohere"
        features = ["Citations", "RAG-optimized", "V2 vision (DAC)"]
    elif model_id.startswith("xai."):
        provider = "xai"
        features = ["Vision", "Reasoning content"]
    else:
        provider = "generic"
        features = ["Basic chat"]

    print(f"  Provider: {provider}")
    print(f"  Features: {', '.join(features)}")


def provider_selection_guide():
    """Guide for selecting the right provider."""
    print("\nProvider Selection Guide")
    print("=" * 50)

    guide = """
    Use this decision tree to pick the right model:

    1. Do you need to process PDFs, videos, or audio?
       YES → google.gemini-2.5-flash or google.gemini-2.5-flash

    2. Do you need parallel tool calling (multiple tools at once)?
       YES → meta.llama-4-scout-17b-16e-instruct

    3. Do you need vision (image understanding)?
       - Fast inference → google.gemini-2.5-flash
       - High quality → meta.llama-3.2-90b-vision-instruct
       - Reasoning → xai.grok-4

    4. Do you need RAG with citations?
       YES → cohere.command-r-plus

    5. Do you need step-by-step reasoning exposed?
       YES → xai.grok-4-fast-reasoning

    6. General-purpose, high-quality text generation?
       → meta.llama-3.3-70b-instruct (good balance)
       → cohere.command-r-plus (excellent reasoning)

    7. Need speed and low latency?
       → google.gemini-2.5-flash (fastest)
       → meta.llama-4-scout-17b-16e-instruct (fast + tools)
    """
    print(guide)


def create_task_specific_llm(task: str) -> ChatOCIGenAI:
    """Factory function to create task-appropriate LLM."""
    print(f"\nCreating LLM for task: {task}")

    task_configs = {
        "vision": {
            "model_id": "meta.llama-3.2-90b-vision-instruct",
            "description": "Image analysis and understanding",
        },
        "multimodal": {
            "model_id": "google.gemini-2.5-flash",
            "description": "PDF, video, audio, and image processing",
        },
        "rag": {
            "model_id": "cohere.command-r-plus",
            "description": "RAG with citations",
        },
        "reasoning": {
            "model_id": "xai.grok-4-fast-reasoning",
            "description": "Complex reasoning with chain-of-thought",
        },
        "tools": {
            "model_id": "meta.llama-4-scout-17b-16e-instruct",
            "description": "Parallel tool calling workflows",
        },
        "general": {
            "model_id": "meta.llama-3.3-70b-instruct",
            "description": "General-purpose assistant",
        },
        "fast": {
            "model_id": "google.gemini-2.5-flash",
            "description": "Low-latency responses",
        },
    }

    config = task_configs.get(task, task_configs["general"])
    print(f"  Model: {config['model_id']}")
    print(f"  Purpose: {config['description']}")

    return ChatOCIGenAI(
        auth_profile=AUTH_PROFILE,
        model_id=config["model_id"],
        service_endpoint=SERVICE_ENDPOINT,
        compartment_id=COMPARTMENT_ID,
    )


def feature_matrix():
    """Display complete feature matrix."""
    print("\nComplete Feature Matrix")
    print("=" * 50)

    matrix = """
    | Feature              | Meta  | Gemini | Cohere | xAI  |
    |---------------------|-------|--------|--------|------|
    | Text Generation     |   ✓   |   ✓    |   ✓    |  ✓   |
    | Vision (Images)     |   ✓   |   ✓    |   ✓*   |  ✓   |
    | PDF Processing      |   ✗   |   ✓    |   ✗    |  ✗   |
    | Video Analysis      |   ✗   |   ✓    |   ✗    |  ✗   |
    | Audio Transcription |   ✗   |   ✓    |   ✗    |  ✗   |
    | Tool Calling        |   ✓   |   ✓    |   ✓    |  ✓   |
    | Parallel Tools      |   ✓** |   ✗    |   ✗    |  ✗   |
    | tool_choice         |   ✓   |   ✓    |   ✗    |  ✓   |
    | Citations/RAG       |   ✗   |   ✗    |   ✓    |  ✗   |
    | Reasoning Content   |   ✗   |   ✗    |   ✗    |  ✓   |
    | Streaming           |   ✓   |   ✓    |   ✓    |  ✓   |
    | Async               |   ✓   |   ✓    |   ✓    |  ✓   |

    *  Cohere vision requires DAC (V2 API)
    ** Parallel tools: Llama 4+ only
    """
    print(matrix)


if __name__ == "__main__":
    print("Provider Comparison Utility")
    print("=" * 60)

    # List vision models
    list_vision_models()

    # Check specific models
    models_to_check = [
        "meta.llama-3.3-70b-instruct",
        "google.gemini-2.5-flash",
        "cohere.command-r-plus",
        "xai.grok-4",
    ]

    print("\n\nModel Capability Check")
    print("=" * 50)
    for model in models_to_check:
        check_model_capabilities(model)

    # Selection guide
    provider_selection_guide()

    # Feature matrix
    feature_matrix()

    # Example: Create task-specific LLM
    print("\n\nTask-Specific LLM Factory")
    print("=" * 50)
    print("Example usage:")
    print("""
    # For RAG workflows
    llm = create_task_specific_llm("rag")

    # For multimodal (PDF, video)
    llm = create_task_specific_llm("multimodal")

    # For tool-heavy workflows
    llm = create_task_specific_llm("tools")
    """)
