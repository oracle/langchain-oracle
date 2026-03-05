# Sample 09: Cohere Provider Examples
# Demonstrates Cohere-specific features: RAG, citations, V2 vision

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

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


def basic_cohere_chat():
    """Basic chat with Cohere Command."""
    print("Basic Cohere Command Chat")
    print("=" * 50)

    llm = ChatOCIGenAI(
        auth_profile=AUTH_PROFILE,
        model_id="cohere.command-r-plus",
        service_endpoint=SERVICE_ENDPOINT,
        compartment_id=COMPARTMENT_ID,
    )

    response = llm.invoke("Explain the difference between ML and deep learning.")
    print(response.content)


def rag_with_citations():
    """RAG with citation support."""
    print("\nRAG with Citations")
    print("=" * 50)

    llm = ChatOCIGenAI(
        auth_profile=AUTH_PROFILE,
        model_id="cohere.command-r-plus",
        service_endpoint=SERVICE_ENDPOINT,
        compartment_id=COMPARTMENT_ID,
    )

    # Provide context documents in system message
    messages = [
        SystemMessage(
            content="""Use the following documents to answer the user's question.
Cite the specific document numbers when making claims.

Document 1: Oracle Cloud Infrastructure (OCI) was launched in October 2016.
It provides enterprise-grade cloud services including compute, storage, and networking.

Document 2: OCI Generative AI service provides access to large language models
from multiple providers including Meta, Cohere, and Google.

Document 3: The langchain-oci package enables Python developers to use OCI
Generative AI services with the LangChain framework."""
        ),
        HumanMessage(content="When was OCI launched and what AI does it offer?"),
    ]

    response = llm.invoke(messages)
    print("Response:", response.content)

    # Check for citations in response metadata
    if "citations" in response.response_metadata:
        print("\nCitations:")
        for citation in response.response_metadata["citations"]:
            print(f"  - {citation}")


def cohere_tool_calling():
    """Tool calling with Cohere (limitations apply)."""
    print("\nCohere Tool Calling")
    print("=" * 50)

    @tool
    def search_database(query: str, table: str = "documents") -> str:
        """Search the database for information."""
        return f"Found 5 results for '{query}' in table '{table}'"

    llm = ChatOCIGenAI(
        auth_profile=AUTH_PROFILE,
        model_id="cohere.command-r-plus",
        service_endpoint=SERVICE_ENDPOINT,
        compartment_id=COMPARTMENT_ID,
    )

    # Note: Cohere doesn't support tool_choice or parallel_tool_calls
    _ = llm.bind_tools([search_database])  # Example binding

    print("Cohere tool calling limitations:")
    print("- No tool_choice parameter support")
    print("- No parallel_tool_calls support")
    print("- Sequential tool execution only")
    print("\nPattern:")
    print("""
    llm_with_tools = llm.bind_tools([search_database])
    # Don't pass tool_choice or parallel_tool_calls
    response = llm_with_tools.invoke("Search for Python samples")
    """)


def cohere_v2_vision():
    """Vision with Cohere V2 API (DAC only)."""
    print("\nCohere V2 Vision (DAC Only)")
    print("=" * 50)

    print("Cohere Command A Vision requires:")
    print("1. Dedicated AI Cluster (DAC) deployment")
    print("2. V2 API format (automatically selected for vision models)")
    print()
    print("Model: cohere.command-a-vision-07-2025")
    print()
    print("Usage pattern:")
    print("""
    # For DAC-deployed vision model
    llm = ChatOCIGenAI(
        auth_profile=AUTH_PROFILE,
        model_id="ocid1.generativeaiendpoint.oc1..xxx",  # DAC endpoint
        provider="cohere",  # Explicitly set provider
        service_endpoint=SERVICE_ENDPOINT,
        compartment_id=COMPARTMENT_ID,
    )

    message = HumanMessage(content=[
        {"type": "text", "text": "Describe this image."},
        load_image("image.jpg"),
    ])

    response = llm.invoke([message])
    """)


def cohere_response_metadata():
    """Accessing Cohere-specific response metadata."""
    print("\nCohere Response Metadata")
    print("=" * 50)

    # Example configuration for accessing metadata:
    # llm = ChatOCIGenAI(
    #     model_id="cohere.command-r-plus",
    #     service_endpoint=SERVICE_ENDPOINT,
    #     compartment_id=COMPARTMENT_ID,
    # )

    print("Available in response.response_metadata:")
    print("- citations: Document citations (when RAG context provided)")
    print("- finish_reason: Why generation stopped")
    print("- documents: Referenced documents")
    print("- search_queries: Generated search queries")
    print("- is_search_required: Whether search was needed")
    print("- total_tokens: Token usage")


def cohere_model_comparison():
    """Compare Cohere models."""
    print("\nCohere Model Comparison")
    print("=" * 50)

    comparison = """
    | Model | Best For | Key Features |
    |-------|----------|--------------|
    | cohere.command-r-plus | Complex reasoning | High capability |
    | cohere.command-a-03-2025 | General use | Latest release |
    | cohere.command-a-vision | Vision tasks | V2 API, DAC only |

    Embedding Models:
    | Model | Type | Dimensions |
    |-------|------|------------|
    | cohere.embed-english-v3.0 | Text | 1024 |
    | cohere.embed-multilingual-v3.0 | Text | 1024 |
    | cohere.embed-v4.0 | Text + Image | 256-1536 |
    """
    print(comparison)


if __name__ == "__main__":
    print("Cohere Provider Examples")
    print("=" * 60)

    # Uncomment to run (requires valid credentials):
    # basic_cohere_chat()
    # rag_with_citations()
    # cohere_tool_calling()
    cohere_v2_vision()
    cohere_response_metadata()
    cohere_model_comparison()

    print("\nExamples are commented out - configure credentials and uncomment to run.")
