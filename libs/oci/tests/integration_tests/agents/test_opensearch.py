# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Integration tests for OpenSearch datastore."""

import os

import pytest

from .conftest import (
    create_embedding_model,
    create_opensearch_store,
    get_opensearch_config,
    opensearch_is_reachable,
)


@pytest.mark.requires("oci", "opensearchpy")
@pytest.mark.skipif(
    not opensearch_is_reachable(), reason="OpenSearch not configured or reachable"
)
class TestOpenSearchDatastore:
    """Tests for OpenSearch datastore with real connection."""

    @pytest.fixture
    def opensearch_store(self):
        """Create OpenSearch store from configuration."""
        return create_opensearch_store()

    @pytest.fixture
    def embedding_model(self):
        """Create embedding model matching OpenSearch index dimensions."""
        config = get_opensearch_config()
        return create_embedding_model(config["embedding_model"])

    def test_connect(self, opensearch_store, embedding_model) -> None:
        """Test connecting to OpenSearch."""
        opensearch_store.connect(embedding_model)
        assert opensearch_store._client is not None

    def test_stats(self, opensearch_store, embedding_model) -> None:
        """Test getting index statistics."""
        opensearch_store.connect(embedding_model)
        stats = opensearch_store.stats()

        config = get_opensearch_config()
        assert stats["store"] == "opensearch"
        assert stats["index"] == config["index_name"]
        assert stats["document_count"] > 0

    def test_search(self, opensearch_store, embedding_model) -> None:
        """Test vector search."""
        opensearch_store.connect(embedding_model)

        query = os.environ.get("OPENSEARCH_TEST_QUERY", "test query")
        query_embedding = embedding_model.embed_query(query)
        results = opensearch_store.search(query, query_embedding, top_k=3)

        assert len(results) > 0

    def test_keyword_search(self, opensearch_store, embedding_model) -> None:
        """Test keyword search."""
        opensearch_store.connect(embedding_model)

        keyword = os.environ.get("OPENSEARCH_TEST_KEYWORD", "error")
        results = opensearch_store.keyword_search(keyword, top_k=3)

        assert isinstance(results, list)

    def test_get_document(self, opensearch_store, embedding_model) -> None:
        """Test getting a document by ID."""
        opensearch_store.connect(embedding_model)

        query = os.environ.get("OPENSEARCH_TEST_QUERY", "test")
        query_embedding = embedding_model.embed_query(query)
        results = opensearch_store.search(query, query_embedding, top_k=1)

        if results:
            doc_id = results[0]["id"]
            doc = opensearch_store.get(doc_id)
            assert doc is not None
            assert doc["id"] == doc_id


@pytest.mark.requires("oci", "opensearchpy", "langgraph")
@pytest.mark.skipif(
    not opensearch_is_reachable(), reason="OpenSearch not configured or reachable"
)
class TestDatastoreToolsWithOpenSearch:
    """Test datastore tools with real OpenSearch connection."""

    @pytest.fixture
    def embedding_model(self):
        """Create embedding model matching OpenSearch index dimensions."""
        config = get_opensearch_config()
        return create_embedding_model(config["embedding_model"])

    @pytest.fixture
    def stores(self) -> dict:
        """Create OpenSearch stores for testing."""
        return {"diagnostics": create_opensearch_store()}

    def test_create_tools(self, stores, embedding_model) -> None:
        """Test creating datastore tools with real OpenSearch."""
        from langchain_oci import create_datastore_tools

        tools = create_datastore_tools(
            stores=stores,
            embedding_model=embedding_model,
        )

        assert len(tools) == 4
        tool_names = {t.name for t in tools}
        assert tool_names == {"search", "keyword_search", "get_document", "stats"}

    def test_search_tool(self, stores, embedding_model) -> None:
        """Test search tool with real OpenSearch."""
        from langchain_oci import create_datastore_tools

        tools = create_datastore_tools(
            stores=stores,
            embedding_model=embedding_model,
            top_k=3,
        )

        search_tool = next(t for t in tools if t.name == "search")
        query = os.environ.get("OPENSEARCH_TEST_QUERY", "test query")
        result = search_tool._run(query=query)

        has_results = (
            "Found" in result or "results" in result.lower() or "No results" in result
        )
        assert has_results

    def test_stats_tool(self, stores, embedding_model) -> None:
        """Test stats tool with real OpenSearch."""
        from langchain_oci import create_datastore_tools

        tools = create_datastore_tools(
            stores=stores,
            embedding_model=embedding_model,
        )

        stats_tool = next(t for t in tools if t.name == "stats")
        result = stats_tool._run(store=None)

        assert "diagnostics" in result

    def test_keyword_search_tool(self, stores, embedding_model) -> None:
        """Test keyword search tool."""
        from langchain_oci import create_datastore_tools

        tools = create_datastore_tools(
            stores=stores,
            embedding_model=embedding_model,
            top_k=3,
        )

        keyword_tool = next(t for t in tools if t.name == "keyword_search")
        keyword = os.environ.get("OPENSEARCH_TEST_KEYWORD", "test")
        result = keyword_tool._run(query=keyword)

        assert result is not None

    def test_get_document_tool(self, stores, embedding_model) -> None:
        """Test get document by ID tool."""
        from langchain_oci import create_datastore_tools

        tools = create_datastore_tools(
            stores=stores,
            embedding_model=embedding_model,
        )

        get_tool = next(t for t in tools if t.name == "get_document")
        result = get_tool._run(document_id="1", store="diagnostics")

        assert result is not None
