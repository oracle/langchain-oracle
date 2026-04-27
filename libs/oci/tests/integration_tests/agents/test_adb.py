# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Integration tests for Oracle ADB datastore."""

import os

import pytest

from .conftest import (
    adb_is_reachable,
    create_adb_store,
    create_embedding_model,
    get_adb_config,
)


@pytest.mark.requires("oci", "oracledb")
@pytest.mark.skipif(not adb_is_reachable(), reason="ADB not configured or reachable")
class TestADBDatastore:
    """Tests for ADB datastore with real connection."""

    @pytest.fixture
    def adb_store(self):
        """Create ADB store from configuration."""
        return create_adb_store()

    @pytest.fixture
    def embedding_model(self):
        """Create embedding model matching ADB index dimensions."""
        config = get_adb_config()
        return create_embedding_model(config["embedding_model"])

    def test_connect(self, adb_store, embedding_model) -> None:
        """Test connecting to ADB."""
        adb_store.connect(embedding_model)
        assert adb_store._connection is not None

    def test_stats(self, adb_store, embedding_model) -> None:
        """Test getting table statistics."""
        adb_store.connect(embedding_model)
        stats = adb_store.stats()

        config = get_adb_config()
        assert stats["store"] == "adb"
        assert stats["table"] == config["table_name"]
        assert stats["document_count"] > 0

    def test_search(self, adb_store, embedding_model) -> None:
        """Test vector search on documents."""
        adb_store.connect(embedding_model)

        query = os.environ.get("ADB_TEST_QUERY", "test query")
        query_embedding = embedding_model.embed_query(query)
        results = adb_store.search(query, query_embedding, top_k=3)

        assert len(results) > 0

    def test_keyword_search(self, adb_store, embedding_model) -> None:
        """Test keyword search."""
        adb_store.connect(embedding_model)

        keyword = os.environ.get("ADB_TEST_KEYWORD", "test")
        results = adb_store.keyword_search(keyword, top_k=3)

        assert len(results) > 0


@pytest.mark.requires("oci", "oracledb", "langgraph")
@pytest.mark.skipif(not adb_is_reachable(), reason="ADB not configured or reachable")
class TestDatastoreToolsWithADB:
    """Test datastore tools with real ADB connection."""

    @pytest.fixture
    def embedding_model(self):
        """Create embedding model matching ADB dimensions."""
        config = get_adb_config()
        return create_embedding_model(config["embedding_model"])

    @pytest.fixture
    def stores(self) -> dict:
        """Create ADB stores for testing."""
        return {"legal": create_adb_store()}

    def test_search_tool(self, stores, embedding_model) -> None:
        """Test search tool with real ADB."""
        from langchain_oci import create_datastore_tools

        tools = create_datastore_tools(
            stores=stores,
            embedding_model=embedding_model,
            top_k=3,
        )

        search_tool = next(t for t in tools if t.name == "search")
        query = os.environ.get("ADB_TEST_QUERY", "test query")
        result = search_tool._run(query=query)

        has_results = (
            "Found" in result or "results" in result.lower() or "No results" in result
        )
        assert has_results
