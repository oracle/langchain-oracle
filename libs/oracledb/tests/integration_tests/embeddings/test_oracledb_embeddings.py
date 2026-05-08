# Copyright (c) 2024, 2026, Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""
integration_tests/embeddings/test_oracledb_embeddings.py

Integration tests for OracleEmbeddings.

Covers:
- LangChain standard Embeddings interface contract via EmbeddingsIntegrationTests
- embed_documents input validation (missing provider, invalid model, empty params)
- embed_documents output shape, type, dimension, and value bounds
- embed_query output shape, type, and consistency with embed_documents
- Semantic similarity verification (related vs unrelated texts, query-to-doc matching)
- load_onnx_model input validation and Oracle directory error handling
- Full embed -> nearest-neighbor pipeline end to end
- Batch consistency and repeatability across multiple calls

Required environment variables:
    VECDB_HOST   — DSN / host string  (e.g. cdb1_pdb1)
    VECDB_USER   — database username  (e.g. vector_user)
    VECDB_PASS   — database password

Run:
    pytest tests/integration_tests/embeddings/test_oracledb_embeddings.py

Assumption:
    The allminilm model is pre-installed in the database by the infrastructure
    setup script (tkvcvecdb_tpi_loadvectormodel.sql) which runs load_onnx_model
    with all-MiniLM-L6-v2.onnx before tests execute. load_onnx_model itself is
    fully covered by mock-based unit tests in test_oracleai_embeddings.py.

Authors:
    - Diego Ascencio (diegoascencioqa)
"""

from __future__ import annotations

import math
import os
from typing import Type

import oracledb
import pytest
from langchain_tests.integration_tests import EmbeddingsIntegrationTests

from langchain_oracledb import OracleEmbeddings

# ---------------------------------------------------------------------------
# Credentials from environment variables
# ---------------------------------------------------------------------------

USERNAME = os.environ.get("VECDB_USER")
PASSWORD = os.environ.get("VECDB_PASS")
DSN = os.environ.get("VECDB_HOST")

# ---------------------------------------------------------------------------
# Skip entire module if env vars are missing or DB is unreachable
# ---------------------------------------------------------------------------

if not all([USERNAME, PASSWORD, DSN]):
    pytest.skip(
        allow_module_level=True,
        reason="VECDB_USER, VECDB_PASS, VECDB_HOST environment variables not set.",
    )

try:
    oracledb.connect(user=USERNAME, password=PASSWORD, dsn=DSN)
except Exception as e:
    pytest.skip(
        allow_module_level=True,
        reason=f"Database connection failed: {e}",
    )

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def cosine_similarity(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def connection():
    """Fresh connection for each test. Closed after test completes."""
    conn = oracledb.connect(user=USERNAME, password=PASSWORD, dsn=DSN)
    yield conn
    conn.close()


@pytest.fixture
def embedder(connection):
    """OracleEmbeddings instance using the allminilm model."""
    return OracleEmbeddings(
        conn=connection,
        params={"provider": "database", "model": "allminilm"},
    )


# ===========================================================================
# LangChain standard integration tests
# Delegates to EmbeddingsIntegrationTests from langchain_tests — covers
# the standard Embeddings interface contract automatically.
# ===========================================================================


class TestOracleEmbeddingsStandard(EmbeddingsIntegrationTests):
    @property
    def embeddings_class(self) -> Type[OracleEmbeddings]:
        return OracleEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        conn = oracledb.connect(user=USERNAME, password=PASSWORD, dsn=DSN)
        return {
            "conn": conn,
            "params": {"provider": "database", "model": "allminilm"},
        }


# ===========================================================================
# Integration — embed_documents input validation
# Verifies constructor and param handling against a real connection.
# ===========================================================================


class TestEmbedDocumentsValidation:
    def test_missing_provider_uses_oracle_default(self, connection):
        """This documents provider behaviour:  is optional."""
        embedder = OracleEmbeddings(
            conn=connection,
            params={"model": "allminilm"},  # provider omitted
        )
        result = embedder.embed_documents(["hello"])
        assert len(result) == 1
        assert len(result[0]) == 384
        assert all(isinstance(v, float) for v in result[0])

    def test_invalid_model_name_raises(self, connection):
        """A model name that doesn't exist in the DB must raise from Oracle."""
        embedder = OracleEmbeddings(
            conn=connection,
            params={"provider": "database", "model": "this_model_does_not_exist"},
        )
        with pytest.raises(Exception):
            embedder.embed_documents(["hello"])

    def test_empty_params_raises(self, connection):
        """Empty params dict must raise — Oracle needs at least provider and model."""
        embedder = OracleEmbeddings(conn=connection, params={})
        with pytest.raises(Exception):
            embedder.embed_documents(["hello"])


# ===========================================================================
# Integration — embed_documents output shape and type
# ===========================================================================


class TestEmbedDocumentsOutput:
    def test_returns_list_of_lists(self, embedder):
        result = embedder.embed_documents(["hello world"])
        assert isinstance(result, list)
        assert isinstance(result[0], list)

    def test_vector_is_non_empty(self, embedder):
        result = embedder.embed_documents(["hello world"])
        assert len(result[0]) > 0

    def test_vector_contains_floats(self, embedder):
        result = embedder.embed_documents(["hello world"])
        assert all(isinstance(v, float) for v in result[0])

    def test_allminilm_produces_384_dimensions(self, embedder):
        """allminilm is a 384-dim model — verify the known output size."""
        result = embedder.embed_documents(["hello"])
        assert len(result[0]) == 384

    def test_multiple_texts_return_one_vector_each(self, embedder):
        texts = ["hello world", "hi everyone", "greetings"]
        result = embedder.embed_documents(texts)
        assert len(result) == len(texts)

    def test_all_vectors_same_dimension(self, embedder):
        """All vectors from the same model must have identical dimension."""
        texts = ["first sentence", "second sentence", "third sentence"]
        result = embedder.embed_documents(texts)
        dims = [len(v) for v in result]
        assert len(set(dims)) == 1

    def test_empty_input_returns_empty_list(self, embedder):
        result = embedder.embed_documents([])
        assert result == []

    def test_single_character_text_embeds(self, embedder):
        """Minimal input — a single character must still produce a valid vector."""
        result = embedder.embed_documents(["a"])
        assert len(result) == 1
        assert len(result[0]) == 384

    def test_long_text_embeds(self, embedder):
        """A longer text must still produce a single 384-dim vector."""
        long_text = "word " * 200
        result = embedder.embed_documents([long_text.strip()])
        assert len(result) == 1
        assert len(result[0]) == 384

    def test_special_characters_embeds(self, embedder):
        """Punctuation, numbers and symbols must not crash the embedder."""
        result = embedder.embed_documents(["Hello! How are you? 100% fine :)"])
        assert len(result[0]) == 384

    def test_batch_of_ten_texts(self, embedder):
        """Batch embedding of 10 texts must return 10 vectors of equal dimension."""
        texts = [f"sentence number {i}" for i in range(10)]
        result = embedder.embed_documents(texts)
        assert len(result) == 10
        assert all(len(v) == 384 for v in result)

    def test_vector_values_are_bounded(self, embedder):
        """allminilm outputs normalised vectors — values must be in [-1, 1]."""
        result = embedder.embed_documents(["hello world"])
        assert all(-1.0 <= v <= 1.0 for v in result[0])

    def test_different_texts_produce_different_vectors(self, embedder):
        """Semantically different texts must not produce identical vectors."""
        result = embedder.embed_documents(["hello world", "quantum physics"])
        assert result[0] != result[1]

    def test_same_text_twice_produces_identical_vectors(self, embedder):
        """The same text in a batch must produce the same vector both times."""
        result = embedder.embed_documents(["identical text", "identical text"])
        assert result[0] == pytest.approx(result[1], abs=1e-5)


# ===========================================================================
# Integration — embed_query output shape and type
# ===========================================================================


class TestEmbedQueryOutput:
    def test_returns_flat_list_of_floats(self, embedder):
        result = embedder.embed_query("hello world")
        assert isinstance(result, list)
        assert all(isinstance(v, float) for v in result)

    def test_returns_384_dimensions_for_allminilm(self, embedder):
        result = embedder.embed_query("hello world")
        assert len(result) == 384

    def test_query_and_doc_same_dimension(self, embedder):
        """embed_query and embed_documents must return vectors of the same size."""
        doc_vec = embedder.embed_documents(["hello world"])[0]
        query_vec = embedder.embed_query("hello world")
        assert len(doc_vec) == len(query_vec)

    def test_identical_text_as_doc_and_query_match(self, embedder):
        """Same text embedded as doc and as query must produce the same vector."""
        text = "the quick brown fox"
        doc_vec = embedder.embed_documents([text])[0]
        query_vec = embedder.embed_query(text)
        assert doc_vec == pytest.approx(query_vec, abs=1e-5)

    def test_query_values_are_bounded(self, embedder):
        result = embedder.embed_query("hello world")
        assert all(-1.0 <= v <= 1.0 for v in result)


# ===========================================================================
# Functional — semantic similarity (real DB)
# Verifies that Oracle's embeddings reflect actual semantic meaning.
# ===========================================================================


class TestFunctionalSemanticSimilarity:
    def test_similar_texts_closer_than_unrelated(self, embedder):
        """Semantically similar texts must produce vectors closer to each other
        than to an unrelated text."""
        results = embedder.embed_documents(
            [
                "The cat sat on the mat",
                "A cat is sitting on a mat",
                "Quantum mechanics describes subatomic particles",
            ]
        )
        sim_related = cosine_similarity(results[0], results[1])
        sim_unrelated = cosine_similarity(results[0], results[2])
        assert sim_related > sim_unrelated

    def test_query_most_similar_to_matching_doc(self, embedder):
        """A query must be most similar to the document that answers it."""
        docs = [
            "The Eiffel Tower is located in Paris, France.",
            "Python is a popular programming language.",
            "The Amazon river is the largest river by discharge.",
        ]
        doc_vecs = embedder.embed_documents(docs)
        query_vec = embedder.embed_query("Where is the Eiffel Tower?")

        similarities = [cosine_similarity(query_vec, dv) for dv in doc_vecs]
        best_match = similarities.index(max(similarities))
        assert best_match == 0  # Eiffel Tower doc must be most similar

    def test_cosine_similarity_range(self, embedder):
        """Cosine similarity of any two real vectors must be in [-1, 1]."""
        results = embedder.embed_documents(["apple", "orange"])
        sim = cosine_similarity(results[0], results[1])
        assert -1.0 <= sim <= 1.0

    def test_self_similarity_is_one(self, embedder):
        """A vector compared to itself must have cosine similarity of 1.0."""
        result = embedder.embed_documents(["hello world"])[0]
        sim = cosine_similarity(result, result)
        assert sim == pytest.approx(1.0, abs=1e-5)


# ===========================================================================
# Functional — load_onnx_model input validation (real DB, no ONNX file needed)
# ===========================================================================


class TestFunctionalLoadOnnxModelValidation:
    def test_none_conn_raises(self, connection):
        """None conn must raise Exception with 'Invalid input' message."""
        with pytest.raises(Exception, match="Invalid input"):
            OracleEmbeddings.load_onnx_model(None, "MY_DIR", "model.onnx", "MY_MODEL")

    def test_none_dir_raises(self, connection):
        """None dir must raise Exception with 'Invalid input' message."""
        with pytest.raises(Exception, match="Invalid input"):
            OracleEmbeddings.load_onnx_model(connection, None, "model.onnx", "MY_MODEL")

    def test_none_onnx_file_raises(self, connection):
        """None onnx file must raise Exception with 'Invalid input' message."""
        with pytest.raises(Exception, match="Invalid input"):
            OracleEmbeddings.load_onnx_model(connection, "MY_DIR", None, "MY_MODEL")

    def test_none_model_name_raises(self, connection):
        """None model name must raise Exception with 'Invalid input' message."""
        with pytest.raises(Exception, match="Invalid input"):
            OracleEmbeddings.load_onnx_model(connection, "MY_DIR", "model.onnx", None)

    def test_nonexistent_dir_raises_oracle_error(self, connection):
        """A directory that doesn't exist in Oracle must raise an ORA error."""
        with pytest.raises(Exception, match="ORA-"):
            OracleEmbeddings.load_onnx_model(
                connection, "DIR_THAT_DOES_NOT_EXIST", "model.onnx", "MY_MODEL"
            )


# ===========================================================================
# Functional — pipeline (real DB, embedder used end to end)
# ===========================================================================


class TestFunctionalPipeline:
    def test_embed_then_find_nearest(self, embedder):
        """Embed a corpus, embed a query, find the nearest doc by cosine similarity."""
        corpus = [
            "Paris is the capital of France.",
            "Berlin is the capital of Germany.",
            "Tokyo is the capital of Japan.",
            "Python is a programming language.",
        ]
        query = "What is the capital of Germany?"

        corpus_vecs = embedder.embed_documents(corpus)
        query_vec = embedder.embed_query(query)

        similarities = [cosine_similarity(query_vec, cv) for cv in corpus_vecs]
        best = similarities.index(max(similarities))
        assert best == 1  # Berlin doc

    def test_embed_documents_then_embed_query_consistent_dims(self, embedder):
        """Vectors from embed_documents and embed_query must have the same length."""
        doc_vecs = embedder.embed_documents(["first doc", "second doc"])
        query_vec = embedder.embed_query("a query")
        assert all(len(dv) == len(query_vec) for dv in doc_vecs)

    def test_repeated_embed_same_result(self, embedder):
        """Embedding the same text twice must produce identical vectors."""
        text = "repeatable embedding test"
        vec1 = embedder.embed_documents([text])[0]
        vec2 = embedder.embed_documents([text])[0]
        assert vec1 == pytest.approx(vec2, abs=1e-5)

    def test_large_batch_consistent_dimensions(self, embedder):
        """50-text batch must return 50 vectors all at 384 dimensions."""
        texts = [f"document number {i} with some content" for i in range(50)]
        results = embedder.embed_documents(texts)
        assert len(results) == 50
        assert all(len(v) == 384 for v in results)
