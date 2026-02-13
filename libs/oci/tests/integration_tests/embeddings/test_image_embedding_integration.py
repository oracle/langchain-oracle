# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Integration tests for multimodal image embedding with OCI GenAI.

These tests verify embedding capabilities across all supported embedding
models in OCI Generative AI, covering both text-only and multimodal
(text + image) models:

**Text embedding models** (text only):
- Cohere Embed v4.0 (1536-d)
- Cohere Embed English v3.0 (1024-d)
- Cohere Embed English Light v3.0 (384-d)
- Cohere Embed Multilingual v3.0 (1024-d)
- Cohere Embed Multilingual Light v3.0 (384-d)

**Multimodal embedding models** (text + image, same vector space):
- Cohere Embed v4.0 (1536-d)
- Cohere Embed Multilingual Image v3.0 (1024-d)

## Test Organization

Tests are organized into consistent groups:

1. **TestImageEmbedding** - Core image embedding (bytes, file, data URI)
2. **TestCrossModalSimilarity** - Text and image in the same vector space
3. **TestOutputDimensions** - output_dimensions parameter (256-1536)
4. **TestInputType** - All text input_type values
5. **TestMultiModelTextEmbedding** - Text embedding across all 5 models
6. **TestMultiModelImageEmbedding** - Image embedding across multimodal models

## Prerequisites

1. Valid OCI credentials configured
2. Access to OCI Generative AI service
3. A compartment with embedding models enabled

## Setup

    export OCI_COMPARTMENT_ID=<your-compartment-id>
    export OCI_CONFIG_PROFILE=DEFAULT
    export OCI_AUTH_TYPE=SECURITY_TOKEN

## Running the Tests

Run all embedding tests:
    pytest tests/integration_tests/embeddings/ -v

Run only image embedding tests:
    pytest tests/integration_tests/embeddings/ -k "Image" -v

Run multi-model tests:
    pytest tests/integration_tests/embeddings/ -k "MultiModel" -v

Run tests for a specific model:
    pytest tests/integration_tests/embeddings/ -k "embed-v4" -v
"""

import io
import math
import os
from typing import List

import pytest

from langchain_oci.embeddings.oci_generative_ai import OCIGenAIEmbeddings
from langchain_oci.utils.vision import IMAGE_EMBEDDING_MODELS

try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


# =============================================================================
# Configuration
# =============================================================================


def get_config() -> dict:
    """Get test configuration from environment variables."""
    compartment_id = os.environ.get("OCI_COMPARTMENT_ID")
    if not compartment_id:
        pytest.skip("OCI_COMPARTMENT_ID not set")
    return {
        "service_endpoint": os.environ.get(
            "OCI_GENAI_ENDPOINT",
            "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
        ),
        "compartment_id": compartment_id,
        "auth_profile": os.environ.get("OCI_CONFIG_PROFILE", "DEFAULT"),
        "auth_type": os.environ.get("OCI_AUTH_TYPE", "SECURITY_TOKEN"),
    }


# =============================================================================
# Helpers
# =============================================================================


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def create_test_image(color: tuple, size: tuple = (64, 64)) -> bytes:
    """Create a solid-color test image as PNG bytes.

    Args:
        color: RGB tuple (e.g., (255, 0, 0) for red).
        size: Image dimensions as (width, height).

    Returns:
        PNG image as bytes.
    """
    if not PIL_AVAILABLE:
        pytest.skip("PIL not available")

    img = Image.new("RGB", size, color)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def make_embeddings(
    model_id: str = "cohere.embed-v4.0", **kwargs
) -> OCIGenAIEmbeddings:
    """Create an OCIGenAIEmbeddings instance from environment config.

    Args:
        model_id: The embedding model to use.
        **kwargs: Additional parameters passed to OCIGenAIEmbeddings.

    Returns:
        Configured OCIGenAIEmbeddings instance.
    """
    config = get_config()
    return OCIGenAIEmbeddings(
        model_id=model_id,
        compartment_id=config["compartment_id"],
        service_endpoint=config["service_endpoint"],
        auth_type=config["auth_type"],
        auth_profile=config["auth_profile"],
        **kwargs,
    )


# =============================================================================
# Model registries
# =============================================================================

# All available text embedding models and their default dimensions
TEXT_EMBEDDING_MODELS = [
    ("cohere.embed-v4.0", 1536),
    ("cohere.embed-english-v3.0", 1024),
    ("cohere.embed-english-light-v3.0", 384),
    ("cohere.embed-multilingual-v3.0", 1024),
    ("cohere.embed-multilingual-light-v3.0", 384),
]

# Models that support image embedding via input_type="IMAGE".
# These embed text and images into the same vector space.
IMAGE_MODELS_WITH_DIMS = [
    ("cohere.embed-v4.0", 1536),
    ("cohere.embed-multilingual-image-v3.0", 1024),
]


# =============================================================================
# Tests: image embedding basics
# =============================================================================


@pytest.mark.requires("oci")
class TestImageEmbedding:
    """Core image embedding tests with embed-v4.0.

    Verifies that images can be embedded from raw bytes, file paths,
    and data URI strings, producing the expected 1536-d vectors.
    """

    @pytest.fixture
    def embeddings(self) -> OCIGenAIEmbeddings:
        return make_embeddings()

    def test_embed_single_image_bytes(self, embeddings: OCIGenAIEmbeddings) -> None:
        """Embed a single image from raw bytes."""
        red_img = create_test_image((255, 0, 0))
        vector = embeddings.embed_image(red_img)

        assert isinstance(vector, list)
        assert len(vector) > 0
        assert all(isinstance(v, float) for v in vector)

    def test_embed_image_dimensions(self, embeddings: OCIGenAIEmbeddings) -> None:
        """embed-v4.0 produces 1536-dimensional vectors for images."""
        img = create_test_image((0, 128, 255))
        vector = embeddings.embed_image(img)
        assert len(vector) == 1536

    def test_embed_multiple_images(self, embeddings: OCIGenAIEmbeddings) -> None:
        """embed_image_batch handles multiple images."""
        images = [
            create_test_image((255, 0, 0)),
            create_test_image((0, 255, 0)),
            create_test_image((0, 0, 255)),
        ]
        vectors = embeddings.embed_image_batch(images)

        assert len(vectors) == 3
        for v in vectors:
            assert len(v) == 1536

    def test_embed_image_data_uri(self, embeddings: OCIGenAIEmbeddings) -> None:
        """Image can be passed as a data URI string."""
        from langchain_oci.utils.vision import encode_image

        raw = create_test_image((128, 128, 0))
        block = encode_image(raw, mime_type="image/png")
        data_uri = block["image_url"]["url"]

        vector = embeddings.embed_image(data_uri)
        assert len(vector) == 1536

    def test_embed_image_from_file(
        self, embeddings: OCIGenAIEmbeddings, tmp_path
    ) -> None:
        """Image can be loaded from a file path."""
        img_file = tmp_path / "test.png"
        img_file.write_bytes(create_test_image((64, 64, 64)))

        vector = embeddings.embed_image(str(img_file))
        assert len(vector) == 1536


# =============================================================================
# Tests: cross-modal similarity
# =============================================================================


@pytest.mark.requires("oci")
class TestCrossModalSimilarity:
    """Verify that text and images live in the same vector space.

    Multimodal models like embed-v4.0 produce vectors where semantically
    related text and images are closer together, enabling cross-modal
    retrieval (e.g., search images with text queries).
    """

    @pytest.fixture
    def embeddings(self) -> OCIGenAIEmbeddings:
        return make_embeddings()

    def test_red_image_closer_to_red_text(self, embeddings: OCIGenAIEmbeddings) -> None:
        """A red image is more similar to 'red' text than 'database'."""
        red_img = create_test_image((255, 0, 0))

        img_vec = embeddings.embed_image(red_img)
        red_text_vec = embeddings.embed_query("a solid red colored image")
        unrelated_vec = embeddings.embed_query("database connection pool exhaustion")

        sim_related = cosine_similarity(img_vec, red_text_vec)
        sim_unrelated = cosine_similarity(img_vec, unrelated_vec)

        assert sim_related > sim_unrelated, (
            f"Expected red image to be closer to 'red' text "
            f"({sim_related:.4f}) than to unrelated text "
            f"({sim_unrelated:.4f})"
        )

    def test_different_images_different_vectors(
        self, embeddings: OCIGenAIEmbeddings
    ) -> None:
        """Different images produce non-identical embedding vectors."""
        red_img = create_test_image((255, 0, 0))
        blue_img = create_test_image((0, 0, 255))

        red_vec = embeddings.embed_image(red_img)
        blue_vec = embeddings.embed_image(blue_img)

        assert red_vec != blue_vec, "Expected different vectors"
        sim = cosine_similarity(red_vec, blue_vec)
        assert sim < 1.0, f"Expected non-identical vectors, got sim={sim:.4f}"

    def test_text_and_image_vectors_same_space(
        self, embeddings: OCIGenAIEmbeddings
    ) -> None:
        """Text and image vectors have the same dimensionality."""
        text_vec = embeddings.embed_query("a blue sky")
        img_vec = embeddings.embed_image(create_test_image((135, 206, 235)))

        assert len(text_vec) == len(img_vec)
        assert len(text_vec) == 1536


# =============================================================================
# Tests: output_dimensions
# =============================================================================


@pytest.mark.requires("oci")
class TestOutputDimensions:
    """Test output_dimensions parameter with embed-v4.0.

    Embed-v4.0 supports configurable output dimensions: 256, 512,
    1024, or 1536 (default). This works for both text and image inputs.
    """

    @pytest.mark.parametrize("dims", [256, 512, 1024, 1536])
    def test_text_output_dimensions(self, dims: int) -> None:
        """Text embeddings respect output_dimensions."""
        emb = make_embeddings(output_dimensions=dims)
        vector = emb.embed_query("test dimensionality")
        assert len(vector) == dims

    @pytest.mark.parametrize("dims", [256, 1536])
    def test_image_output_dimensions(self, dims: int) -> None:
        """Image embeddings respect output_dimensions."""
        emb = make_embeddings(output_dimensions=dims)
        img = create_test_image((100, 200, 50))
        vector = emb.embed_image(img)
        assert len(vector) == dims


# =============================================================================
# Tests: input_type
# =============================================================================


@pytest.mark.requires("oci")
class TestInputType:
    """Test input_type parameter for text embeddings.

    The OCI embed API supports SEARCH_DOCUMENT, SEARCH_QUERY,
    CLASSIFICATION, and CLUSTERING input types for text.
    IMAGE input type is set automatically by embed_image/embed_image_batch.
    """

    @pytest.mark.parametrize(
        "input_type",
        [
            "SEARCH_DOCUMENT",
            "SEARCH_QUERY",
            "CLASSIFICATION",
            "CLUSTERING",
        ],
    )
    def test_text_input_types(self, input_type: str) -> None:
        """All text input_type values produce valid embeddings."""
        emb = make_embeddings(input_type=input_type)
        vector = emb.embed_query("test input type")
        assert len(vector) == 1536
        assert all(isinstance(v, float) for v in vector)


# =============================================================================
# Tests: multi-model text embedding
# =============================================================================


@pytest.mark.requires("oci")
class TestMultiModelTextEmbedding:
    """Test text embedding across all available models.

    Runs dimension, batch, and semantic relevance tests against
    each of the 5 text embedding models to ensure consistent behavior.
    """

    @pytest.mark.parametrize(
        ("model_id", "expected_dims"),
        TEXT_EMBEDDING_MODELS,
        ids=[m[0] for m in TEXT_EMBEDDING_MODELS],
    )
    def test_text_embedding_dimensions(self, model_id: str, expected_dims: int) -> None:
        """Each model produces the expected vector dimensions."""
        emb = make_embeddings(model_id=model_id)
        vector = emb.embed_query("CPU usage is too high")
        assert len(vector) == expected_dims
        assert all(isinstance(v, float) for v in vector)

    @pytest.mark.parametrize(
        ("model_id", "expected_dims"),
        TEXT_EMBEDDING_MODELS,
        ids=[m[0] for m in TEXT_EMBEDDING_MODELS],
    )
    def test_batch_embedding(self, model_id: str, expected_dims: int) -> None:
        """Each model handles batch embedding correctly."""
        emb = make_embeddings(model_id=model_id)
        texts = [
            "High CPU usage troubleshooting",
            "Memory leak detection",
            "Disk space monitoring",
        ]
        vectors = emb.embed_documents(texts)
        assert len(vectors) == 3
        for v in vectors:
            assert len(v) == expected_dims

    @pytest.mark.parametrize(
        ("model_id", "expected_dims"),
        TEXT_EMBEDDING_MODELS,
        ids=[m[0] for m in TEXT_EMBEDDING_MODELS],
    )
    def test_similarity_relevance(self, model_id: str, expected_dims: int) -> None:
        """Similar texts are closer than unrelated texts."""
        emb = make_embeddings(model_id=model_id)
        query = emb.embed_query("memory leak Java heap dump")
        related = emb.embed_query("JVM out of memory error heap analysis")
        unrelated = emb.embed_query("chocolate cake recipe with vanilla frosting")

        sim_related = cosine_similarity(query, related)
        sim_unrelated = cosine_similarity(query, unrelated)
        assert sim_related > sim_unrelated, (
            f"{model_id}: related={sim_related:.4f} "
            f"should be > unrelated={sim_unrelated:.4f}"
        )


# =============================================================================
# Tests: multi-model image embedding
# =============================================================================


@pytest.mark.requires("oci")
class TestMultiModelImageEmbedding:
    """Test image embedding across all multimodal models.

    Runs dimension, cross-modal, and semantic relevance tests against
    each multimodal model to verify text-image shared vector space.
    """

    @pytest.mark.parametrize(
        ("model_id", "expected_dims"),
        IMAGE_MODELS_WITH_DIMS,
        ids=[m[0] for m in IMAGE_MODELS_WITH_DIMS],
    )
    def test_image_embedding_dimensions(
        self, model_id: str, expected_dims: int
    ) -> None:
        """Each multimodal model produces expected dimensions."""
        emb = make_embeddings(model_id=model_id)
        img = create_test_image((50, 150, 250))
        vector = emb.embed_image(img)
        assert len(vector) == expected_dims

    @pytest.mark.parametrize(
        ("model_id", "expected_dims"),
        IMAGE_MODELS_WITH_DIMS,
        ids=[m[0] for m in IMAGE_MODELS_WITH_DIMS],
    )
    def test_cross_modal_same_dimensions(
        self, model_id: str, expected_dims: int
    ) -> None:
        """Text and image vectors have the same dimensions."""
        emb = make_embeddings(model_id=model_id)
        text_vec = emb.embed_query("a green forest")
        img_vec = emb.embed_image(create_test_image((34, 139, 34)))
        assert len(text_vec) == len(img_vec) == expected_dims

    @pytest.mark.parametrize(
        ("model_id", "expected_dims"),
        IMAGE_MODELS_WITH_DIMS,
        ids=[m[0] for m in IMAGE_MODELS_WITH_DIMS],
    )
    def test_cross_modal_relevance(self, model_id: str, expected_dims: int) -> None:
        """Image is more similar to matching text than unrelated."""
        emb = make_embeddings(model_id=model_id)
        blue_img = create_test_image((0, 0, 255))
        img_vec = emb.embed_image(blue_img)
        blue_vec = emb.embed_query("a solid blue image")
        food_vec = emb.embed_query("pasta carbonara recipe")

        sim_blue = cosine_similarity(img_vec, blue_vec)
        sim_food = cosine_similarity(img_vec, food_vec)
        assert sim_blue > sim_food, (
            f"{model_id}: blue={sim_blue:.4f} should be > food={sim_food:.4f}"
        )

    @pytest.mark.parametrize(
        ("model_id", "expected_dims"),
        IMAGE_MODELS_WITH_DIMS,
        ids=[m[0] for m in IMAGE_MODELS_WITH_DIMS],
    )
    def test_multiple_images(self, model_id: str, expected_dims: int) -> None:
        """embed_image_batch works with each multimodal model."""
        emb = make_embeddings(model_id=model_id)
        images = [
            create_test_image((255, 0, 0)),
            create_test_image((0, 255, 0)),
        ]
        vectors = emb.embed_image_batch(images)
        assert len(vectors) == 2
        for v in vectors:
            assert len(v) == expected_dims


# =============================================================================
# Tests: IMAGE_EMBEDDING_MODELS registry
# =============================================================================


@pytest.mark.requires("oci")
class TestImageEmbeddingModelsRegistry:
    """Verify the IMAGE_EMBEDDING_MODELS registry is accurate.

    Each model listed in the registry should actually support
    image embedding via the OCI API.
    """

    @pytest.mark.parametrize(
        "model_id",
        IMAGE_EMBEDDING_MODELS,
        ids=IMAGE_EMBEDDING_MODELS,
    )
    def test_registry_model_supports_images(self, model_id: str) -> None:
        """Each registered model can embed an image."""
        emb = make_embeddings(model_id=model_id)
        img = create_test_image((128, 64, 192))
        vector = emb.embed_image(img)
        assert isinstance(vector, list)
        assert len(vector) > 0
        assert all(isinstance(v, float) for v in vector)
