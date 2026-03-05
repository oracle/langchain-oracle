# Sample 10: Text Embeddings Example
# Demonstrates creating and using text embeddings

import numpy as np

from langchain_oci import OCIGenAIEmbeddings
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


def main():
    # Create embeddings client
    embeddings = OCIGenAIEmbeddings(
        auth_profile=AUTH_PROFILE,
        model_id="cohere.embed-english-v3.0",
        service_endpoint=SERVICE_ENDPOINT,
        compartment_id=COMPARTMENT_ID,
    )

    # Test 1: Single query embedding
    print("Test 1: Single Query Embedding")
    print("-" * 40)
    query = "What is machine learning?"
    query_vector = embeddings.embed_query(query)
    print(f"Query: {query}")
    print(f"Vector dimensions: {len(query_vector)}")
    print(f"First 5 values: {query_vector[:5]}")

    # Test 2: Document embeddings
    print("\nTest 2: Document Embeddings")
    print("-" * 40)
    documents = [
        "Machine learning is a type of artificial intelligence.",
        "Python is a popular programming language.",
        "The weather is nice today.",
        "Neural networks are used in deep learning.",
    ]
    doc_vectors = embeddings.embed_documents(documents)
    print(f"Number of documents: {len(documents)}")
    print(f"Number of vectors: {len(doc_vectors)}")

    # Test 3: Semantic similarity
    print("\nTest 3: Semantic Similarity")
    print("-" * 40)
    print(f"Query: '{query}'")
    print("\nSimilarity scores:")
    for doc, vec in zip(documents, doc_vectors):
        sim = cosine_similarity(query_vector, vec)
        print(f"  {sim:.4f} - {doc[:50]}...")

    # Find most similar document
    similarities = [cosine_similarity(query_vector, v) for v in doc_vectors]
    best_idx = np.argmax(similarities)
    print(f"\nMost similar: '{documents[best_idx]}'")


if __name__ == "__main__":
    main()
