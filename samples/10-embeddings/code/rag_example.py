# Sample 10: RAG (Retrieval Augmented Generation) Example
# Demonstrates using embeddings for context retrieval

import numpy as np

from langchain_oci import ChatOCIGenAI, OCIGenAIEmbeddings
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


def cosine_similarity(a: list, b: list) -> float:
    """Calculate cosine similarity between two vectors."""
    a_arr = np.array(a)
    b_arr = np.array(b)
    return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)))


def retrieve_context(
    query: str, documents: list, doc_vectors: list, embeddings, top_k: int = 2
) -> list:
    """Retrieve most relevant documents for a query."""
    query_vector = embeddings.embed_query(query)

    # Calculate similarities
    similarities = [
        (i, cosine_similarity(query_vector, dv)) for i, dv in enumerate(doc_vectors)
    ]

    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Return top-k documents
    return [documents[i] for i, _ in similarities[:top_k]]


def main():
    # Create embeddings client
    embeddings = OCIGenAIEmbeddings(
        auth_profile=AUTH_PROFILE,
        model_id="cohere.embed-english-v3.0",
        service_endpoint=SERVICE_ENDPOINT,
        compartment_id=COMPARTMENT_ID,
    )

    # Create LLM
    llm = ChatOCIGenAI(
        auth_profile=AUTH_PROFILE,
        model_id="meta.llama-3.3-70b-instruct",
        service_endpoint=SERVICE_ENDPOINT,
        compartment_id=COMPARTMENT_ID,
    )

    # Knowledge base (in production, this would be much larger)
    knowledge_base = [
        "OCI provides enterprise cloud services including compute and storage.",
        "OCI GenAI offers access to LLMs from Meta, Cohere, and Google.",
        "LangChain is a framework for building apps with LLMs.",
        "The langchain-oci package integrates OCI GenAI with LangChain.",
        "RAG combines retrieval with LLMs for accurate, grounded responses.",
        "Vector embeddings enable semantic similarity search.",
        "OCI offers dedicated AI clusters (DAC) for custom endpoints.",
        "ChatOCIGenAI is the main interface for chat models.",
    ]

    print("RAG Example: Retrieval Augmented Generation")
    print("=" * 50)

    # Index documents
    print("\nIndexing knowledge base...")
    doc_vectors = embeddings.embed_documents(knowledge_base)
    print(f"Indexed {len(doc_vectors)} documents")

    # Example questions
    questions = [
        "What is OCI Generative AI?",
        "How do I use LangChain with Oracle Cloud?",
        "What is RAG and how does it work?",
    ]

    for question in questions:
        print(f"\n{'=' * 50}")
        print(f"Question: {question}")
        print("-" * 50)

        # Retrieve relevant context
        context_docs = retrieve_context(
            question, knowledge_base, doc_vectors, embeddings, top_k=2
        )

        print("Retrieved context:")
        for i, doc in enumerate(context_docs, 1):
            print(f"  {i}. {doc[:80]}...")

        # Generate answer with context
        context = "\n".join(context_docs)
        prompt = f"""Use the following context to answer the question.
Be concise and only use information from the context.

Context:
{context}

Question: {question}

Answer:"""

        response = llm.invoke(prompt)
        print(f"\nAnswer: {response.content}")


if __name__ == "__main__":
    main()
