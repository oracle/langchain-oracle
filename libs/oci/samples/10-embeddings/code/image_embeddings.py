# Sample 10: Image Embeddings Example
# Demonstrates multimodal embeddings for cross-modal search

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
    # Create multimodal embeddings client
    embeddings = OCIGenAIEmbeddings(
        auth_profile=AUTH_PROFILE,
        model_id="cohere.embed-v4.0",  # Multimodal model
        service_endpoint=SERVICE_ENDPOINT,
        compartment_id=COMPARTMENT_ID,
    )

    print("Image Embeddings with cohere.embed-v4.0")
    print("=" * 50)

    # Note: Replace these with real image paths
    image_paths = [
        "sunset.jpg",
        "beach.jpg",
        "mountain.jpg",
    ]

    # Check if images exist (demo mode)
    import os

    images_exist = all(os.path.exists(p) for p in image_paths)

    if not images_exist:
        print("\nDemo mode: Image files not found.")
        print("To test image embeddings:")
        print("1. Place image files in the current directory")
        print("2. Update image_paths list")
        print("3. Run this script again")

        # Demo with text embeddings instead
        print("\nShowing cross-modal capability concept:")
        print("-" * 50)

        # Text descriptions that could match images
        descriptions = [
            "A beautiful sunset over the ocean",
            "A sandy beach with palm trees",
            "Snow-capped mountain peaks",
        ]

        print("If you had these images:")
        for i, (img, desc) in enumerate(zip(image_paths, descriptions)):
            print(f"  {img}: '{desc}'")

        print("\nYou could search with text queries like:")
        print("  'Show me sunset photos' -> Would match sunset.jpg")
        print("  'Beach vacation' -> Would match beach.jpg")
        print("  'Mountain scenery' -> Would match mountain.jpg")

    else:
        # Real image embedding
        print("\nEmbedding images...")

        # Embed all images
        image_vectors = embeddings.embed_image_batch(image_paths)
        print(f"Embedded {len(image_vectors)} images")
        print(f"Vector dimensions: {len(image_vectors[0])}")

        # Text query for cross-modal search
        query = "A beautiful sunset over the ocean"
        query_vector = embeddings.embed_query(query)
        print(f"\nSearch query: '{query}'")

        # Find most similar image
        print("\nSimilarity scores:")
        for path, vec in zip(image_paths, image_vectors):
            sim = cosine_similarity(query_vector, vec)
            print(f"  {sim:.4f} - {path}")

        similarities = [cosine_similarity(query_vector, v) for v in image_vectors]
        best_idx = np.argmax(similarities)
        print(f"\nBest match: {image_paths[best_idx]}")


if __name__ == "__main__":
    main()
