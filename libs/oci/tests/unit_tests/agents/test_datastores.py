# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Unit tests for datastores and datastore tools."""

import logging
from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.requires("oci")
class TestVectorDataStore:
    """Tests for VectorDataStore abstract base class."""

    def test_is_abstract(self) -> None:
        """Test that VectorDataStore cannot be instantiated directly."""
        from langchain_oci.datastores.vectorstores import VectorDataStore

        with pytest.raises(TypeError, match="abstract"):
            VectorDataStore()  # type: ignore[abstract]

    def test_has_required_methods(self) -> None:
        """Test that VectorDataStore defines required abstract properties/methods."""
        import inspect

        from langchain_oci.datastores.vectorstores import VectorDataStore

        abstract_methods = {
            name
            for name, method in inspect.getmembers(VectorDataStore)
            if getattr(method, "__isabstractmethod__", False)
        }

        expected = {
            "name",
            "connect",
            "vectorstore",
            "get",
            "insert",
            "bulk_insert",
            "update",
            "delete",
            "stats",
        }
        assert expected.issubset(abstract_methods)

    def test_datastore_description_property_has_default(self) -> None:
        """Test that datastore_description property returns empty string by default."""
        from langchain_oci.datastores.vectorstores import VectorDataStore

        # Create a concrete implementation for testing
        class ConcreteStore(VectorDataStore):
            @property
            def name(self) -> str:
                return "test"

            @property
            def vectorstore(self):
                return MagicMock()

            def connect(self, embedding_model):
                pass

            def get(self, document_id):
                return None

            def insert(self, title, content, source, embedding):
                return "1"

            def bulk_insert(self, documents, embeddings):
                return len(documents)

            def update(self, document_id, title, content, source, embedding):
                return True

            def delete(self, document_id):
                return True

            def stats(self):
                return {}

        store = ConcreteStore()
        assert store.datastore_description == ""

    def test_search_documents_uses_vectorstore_standard_contract(self) -> None:
        """Test semantic search delegates to the configured LangChain vector store."""
        from langchain_core.documents import Document

        from langchain_oci.datastores.vectorstores import VectorDataStore

        vectorstore = MagicMock()
        vectorstore.similarity_search_with_score.return_value = [
            (Document(page_content="alpha", metadata={"id": "1"}), 0.9)
        ]

        class ConcreteStore(VectorDataStore):
            @property
            def name(self) -> str:
                return "test"

            @property
            def vectorstore(self):
                return vectorstore

            def connect(self, embedding_model):
                pass

            def get(self, document_id):
                return None

            def insert(self, title, content, source, embedding):
                return "1"

            def bulk_insert(self, documents, embeddings):
                return len(documents)

            def update(self, document_id, title, content, source, embedding):
                return True

            def delete(self, document_id):
                return True

            def stats(self):
                return {}

        store = ConcreteStore()
        results = store.search_documents_with_scores("alpha", 1)

        assert len(results) == 1
        vectorstore.similarity_search_with_score.assert_called_once_with("alpha", k=1)


@pytest.mark.requires("oci")
class TestOpenSearchDataStore:
    """Tests for OpenSearch datastore."""

    def test_initialization(self) -> None:
        """Test OpenSearch datastore can be initialized."""
        from langchain_oci.datastores.vectorstores import OpenSearch

        store = OpenSearch(
            endpoint="https://localhost:9200",
            index_name="test-index",
            username="admin",
            password="admin",
            datastore_description="test documents",
        )

        assert store.endpoint == "https://localhost:9200"
        assert store.index_name == "test-index"
        assert store.username == "admin"
        assert store.name == "opensearch"
        assert store.datastore_description == "test documents"

    def test_connect_requires_opensearchpy(self) -> None:
        """Test that connect raises ImportError if opensearch-py not installed."""
        from langchain_oci.datastores.vectorstores import OpenSearch

        store = OpenSearch(
            endpoint="https://localhost:9200",
            index_name="test-index",
        )

        with patch.dict("sys.modules", {"opensearchpy": None}):
            with pytest.raises(ImportError, match="opensearch-py required"):
                store.connect(MagicMock())

    def test_default_values(self) -> None:
        """Test OpenSearch has sensible defaults."""
        from langchain_oci.datastores.vectorstores import OpenSearch

        store = OpenSearch(
            endpoint="https://localhost:9200",
            index_name="test-index",
        )

        assert store.use_ssl is True
        assert store.verify_certs is True
        assert store.vector_field == "embedding"
        assert store.search_fields == ["title", "content"]

    def test_search_documents_normalizes_text_and_metadata_fields(self) -> None:
        """Search results should use text/metadata-backed OpenSearch documents."""
        from langchain_oci.datastores.vectorstores import OpenSearch
        from langchain_oci.datastores.vectorstores.opensearch import (
            _OpenSearchVectorStore,
        )

        store = OpenSearch(
            endpoint="https://localhost:9200",
            index_name="test-index",
            vector_field="vector_field",
        )
        store._embedding_model = MagicMock()
        store._embedding_model.embed_query.return_value = [0.1, 0.2]
        client = MagicMock()
        client.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_id": "runbook-1",
                        "_score": 0.91,
                        "_source": {
                            "text": "Runbook summary text",
                            "metadata": {
                                "title": "Database Remediation Decision",
                                "content": (
                                    '{"id":"runbook-1","summary":"Large JSON blob"}'
                                ),
                                "source_path": "runbooks/db.yaml",
                                "category": "database",
                            },
                            "vector_field": [0.1, 0.2],
                        },
                    }
                ]
            }
        }

        store._vectorstore = _OpenSearchVectorStore(
            client=client,
            embedding_model=store._embedding_model,
            index_name="test-index",
            vector_field="vector_field",
        )

        docs_and_scores = store.search_documents_with_scores("database", top_k=1)

        assert len(docs_and_scores) == 1
        doc, score = docs_and_scores[0]
        assert doc.page_content == "Runbook summary text"
        assert doc.metadata["id"] == "runbook-1"
        assert doc.metadata["title"] == "Database Remediation Decision"
        assert doc.metadata["source"] == "runbooks/db.yaml"
        assert doc.metadata["category"] == "database"
        assert "content" not in doc.metadata
        assert score == 0.91

    def test_get_normalizes_nested_metadata_content(self) -> None:
        """Document fetch should not expose raw metadata blobs as content."""
        from langchain_oci.datastores.vectorstores import OpenSearch

        store = OpenSearch(
            endpoint="https://localhost:9200",
            index_name="test-index",
            vector_field="vector_field",
        )
        store._client = MagicMock()
        store._client.get.return_value = {
            "found": True,
            "_id": "doc-123",
            "_source": {
                "text": "Clean summary body",
                "metadata": {
                    "title": "Source Track Test",
                    "content": "Original detailed content",
                    "source_path": "runbooks/source.md",
                    "source_id": "source-123",
                },
                "vector_field": [0.1, 0.2],
            },
        }

        doc = store.get("doc-123")

        assert doc == {
            "id": "doc-123",
            "title": "Source Track Test",
            "content": "Clean summary body",
            "source": "runbooks/source.md",
            "source_path": "runbooks/source.md",
            "source_id": "source-123",
        }


@pytest.mark.requires("oci")
class TestADBDataStore:
    """Tests for Oracle ADB datastore."""

    def test_initialization(self) -> None:
        """Test ADB datastore can be initialized."""
        from langchain_oci.datastores.vectorstores import ADB

        store = ADB(
            dsn="mydb_low",
            user="ADMIN",
            password="password123",
            wallet_location="~/.oracle-wallet",
            table_name="MY_VECTORS",
            datastore_description="sales data",
        )

        assert store.dsn == "mydb_low"
        assert store.user == "ADMIN"
        assert store.table_name == "MY_VECTORS"
        assert store.name == "adb"
        assert store.datastore_description == "sales data"

    def test_connect_requires_oracledb(self) -> None:
        """Test that connect raises ImportError if oracledb not installed."""
        from langchain_oci.datastores.vectorstores import ADB

        store = ADB(
            dsn="mydb_low",
            user="ADMIN",
            password="password",
        )

        with patch.dict("sys.modules", {"oracledb": None}):
            with pytest.raises(ImportError, match="oracledb required"):
                store.connect(MagicMock())

    def test_default_values(self) -> None:
        """Test ADB has sensible defaults."""
        from langchain_oci.datastores.vectorstores import ADB

        store = ADB(
            dsn="mydb_low",
            user="ADMIN",
            password="password",
        )

        assert store.table_name == "VECTOR_DOCUMENTS"
        assert store.wallet_location is None
        assert store.wallet_password is None

    def test_search_documents_delegates_to_oraclevs_backend(self) -> None:
        """Test ADB delegates semantic search to OracleVS."""
        from langchain_core.documents import Document

        from langchain_oci.datastores.vectorstores import ADB

        store = ADB(
            dsn="mydb_low",
            user="ADMIN",
            password="password",
        )

        oraclevs = MagicMock()
        oraclevs.similarity_search_with_score.return_value = [
            (
                Document(
                    page_content="alpha content",
                    metadata={"id": "42", "title": "Alpha", "source": "test_source"},
                ),
                0.2,
            )
        ]
        store._oraclevs = oraclevs

        results = store.search_documents_with_scores(query="alpha", top_k=1)

        assert len(results) == 1
        assert results[0][0].metadata["id"] == "42"
        assert results[0][0].metadata["title"] == "Alpha"
        assert results[0][0].metadata["source"] == "test_source"
        assert results[0][1] == 0.2

    @patch("langchain_oci.datastores.vectorstores.adb.uuid.uuid4")
    def test_insert_delegates_to_oraclevs_backend(self, mock_uuid) -> None:
        """Test ADB insert delegates to OracleVS add_documents in chunk mode."""
        from langchain_oci.datastores.vectorstores import ADB

        mock_uuid.return_value = "doc-123"
        store = ADB(dsn="mydb_low", user="ADMIN", password="password")
        store._oraclevs = MagicMock()
        store._write_text_splitter = MagicMock()

        inserted_id = store.insert(
            title="T",
            content="C",
            source="S",
            embedding=[0.1, 0.2],
        )

        assert inserted_id == "doc-123"
        store._oraclevs.add_documents.assert_called_once()
        args, kwargs = store._oraclevs.add_documents.call_args
        assert kwargs["ids"] == ["doc-123"]
        assert kwargs["text_splitter"] is store._write_text_splitter
        assert args[0][0].page_content == "C"
        assert args[0][0].metadata == {
            "id": "doc-123",
            "title": "T",
            "source": "S",
        }

    def test_bulk_insert_delegates_to_oraclevs_backend(self) -> None:
        """Test ADB bulk insert delegates to OracleVS add_documents in chunk mode."""
        from langchain_oci.datastores.vectorstores import ADB

        store = ADB(dsn="mydb_low", user="ADMIN", password="password")
        store._oraclevs = MagicMock()
        store._write_text_splitter = MagicMock()

        count = store.bulk_insert(
            documents=[{"id": "1", "title": "A", "content": "alpha", "source": "src"}],
            embeddings=[[0.1, 0.2]],
        )

        assert count == 1
        store._oraclevs.add_documents.assert_called_once()
        args, kwargs = store._oraclevs.add_documents.call_args
        assert kwargs["ids"] == ["1"]
        assert kwargs["text_splitter"] is store._write_text_splitter
        assert args[0][0].page_content == "alpha"
        assert args[0][0].metadata == {
            "id": "1",
            "title": "A",
            "source": "src",
        }


@pytest.mark.requires("oci")
class TestStoreSelector:
    """Tests for StoreSelector routing logic."""

    def test_single_store_routing(self) -> None:
        """Test that single store always routes to itself."""
        from langchain_oci.datastores.tools import StoreSelector

        mock_embedding = MagicMock()
        mock_embedding.embed_query.return_value = [0.1] * 1024

        mock_store = MagicMock()
        mock_store.datastore_description = "test"

        selector = StoreSelector(
            stores={"only_store": mock_store},
            embedding_model=mock_embedding,
            default_store="only_store",
        )

        assert selector.route("any query") == "only_store"

    def test_multi_store_routing(self) -> None:
        """Test routing between multiple stores."""
        from langchain_oci.datastores.tools import StoreSelector

        mock_embedding = MagicMock()
        # Return different embeddings for different queries
        mock_embedding.embed_query.side_effect = lambda q: (
            [1.0, 0.0, 0.0] if "hr" in q.lower() else [0.0, 1.0, 0.0]
        )

        hr_store = MagicMock()
        hr_store.datastore_description = "HR policies, employee benefits"

        sales_store = MagicMock()
        sales_store.datastore_description = "sales data, revenue"

        selector = StoreSelector(
            stores={"hr": hr_store, "sales": sales_store},
            embedding_model=mock_embedding,
            default_store="hr",
        )

        # The selector pre-computes hint embeddings
        assert "hr" in selector.stores
        assert "sales" in selector.stores


@pytest.mark.requires("oci")
class TestCreateDatastoreTools:
    """Tests for create_datastore_tools factory function."""

    def test_raises_without_stores(self) -> None:
        """Test that empty stores raises ValueError."""
        from langchain_oci.datastores.tools import create_datastore_tools

        with pytest.raises(ValueError, match="At least one datastore"):
            create_datastore_tools(stores={})

    def test_raises_invalid_default_store(self) -> None:
        """Test that invalid default_store raises ValueError."""
        from langchain_oci.datastores.tools import create_datastore_tools

        mock_store = MagicMock()
        mock_store.datastore_description = "test"
        mock_store.vectorstore = MagicMock()
        mock_store.keyword_retriever = MagicMock()

        with pytest.raises(ValueError, match="not found"):
            create_datastore_tools(
                stores={"store1": mock_store},
                default_store="nonexistent",
            )

    def test_requires_compartment_id_for_default_embeddings(self) -> None:
        """Test that compartment_id is required when using default embeddings."""
        from langchain_oci.datastores.tools import create_datastore_tools

        mock_store = MagicMock()
        mock_store.datastore_description = "test"
        mock_store.vectorstore = MagicMock()
        mock_store.keyword_retriever = MagicMock()

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="compartment_id is required"):
                create_datastore_tools(stores={"store1": mock_store})

    def test_creates_four_tools(self) -> None:
        """Test that factory creates all four datastore tools."""
        from langchain_oci.datastores.tools import create_datastore_tools

        mock_store = MagicMock()
        mock_store.datastore_description = "test"
        mock_store.vectorstore = MagicMock()
        mock_store.keyword_retriever = MagicMock()

        mock_embedding = MagicMock()
        mock_embedding.embed_query.return_value = [0.1] * 1024

        tools = create_datastore_tools(
            stores={"test": mock_store},
            embedding_model=mock_embedding,
        )

        assert len(tools) == 4
        tool_names = {t.name for t in tools}
        assert tool_names == {"search", "keyword_search", "get_document", "stats"}

    def test_tools_have_descriptions(self) -> None:
        """Test that all tools have descriptions."""
        from langchain_oci.datastores.tools import create_datastore_tools

        mock_store = MagicMock()
        mock_store.datastore_description = "test documents"
        mock_store.vectorstore = MagicMock()
        mock_store.keyword_retriever = MagicMock()

        mock_embedding = MagicMock()
        mock_embedding.embed_query.return_value = [0.1] * 1024

        tools = create_datastore_tools(
            stores={"test": mock_store},
            embedding_model=mock_embedding,
        )

        for tool in tools:
            assert tool.description, f"Tool {tool.name} should have a description"

    def test_accepts_custom_top_k(self) -> None:
        """Test that custom top_k is passed to search tools."""
        from langchain_oci.datastores.tools import create_datastore_tools

        mock_store = MagicMock()
        mock_store.datastore_description = "test"
        mock_store.vectorstore = MagicMock()
        mock_store.keyword_retriever = MagicMock()

        mock_embedding = MagicMock()
        mock_embedding.embed_query.return_value = [0.1] * 1024

        tools = create_datastore_tools(
            stores={"test": mock_store},
            embedding_model=mock_embedding,
            top_k=20,
        )

        search_tool = next(t for t in tools if t.name == "search")
        assert getattr(search_tool, "top_k") == 20

    def test_search_tool_logs_backend_error(self, caplog) -> None:
        """Test search tool logs backend exceptions with query context."""
        from langchain_oci.datastores.tools.search import SearchTool

        selector = MagicMock()
        selector.route.return_value = "research"
        failing_store = MagicMock()
        failing_store.search_documents_with_scores.side_effect = RuntimeError("boom")
        selector.get_store.return_value = failing_store

        tool = SearchTool(
            selector=selector,
            store_list="research",
            top_k=5,
            description="test search tool",
        )

        with caplog.at_level(logging.ERROR):
            result = tool._run("test query")

        assert result == "Error during semantic search: RuntimeError: boom"
        assert "Datastore semantic search failed" in caplog.text
        assert "test query" in caplog.text
        assert "research" in caplog.text

    def test_keyword_search_tool_logs_backend_error(self, caplog) -> None:
        """Test keyword search tool logs backend exceptions with query context."""
        from langchain_oci.datastores.tools.keyword_search import (
            KeywordSearchTool,
        )

        selector = MagicMock()
        selector.route.return_value = "research"
        failing_store = MagicMock()
        failing_store.keyword_search_documents.side_effect = RuntimeError("boom")
        selector.get_store.return_value = failing_store

        tool = KeywordSearchTool(
            selector=selector,
            store_list="research",
            top_k=5,
            description="test keyword search tool",
        )

        with caplog.at_level(logging.ERROR):
            result = tool._run("exact term")

        assert result == "Error during keyword search: RuntimeError: boom"
        assert "Datastore keyword search failed" in caplog.text
        assert "exact term" in caplog.text
        assert "research" in caplog.text


@pytest.mark.requires("oci")
class TestDatastoreImports:
    """Tests for datastore imports from various paths."""

    def test_import_from_datastores(self) -> None:
        """Test imports from langchain_oci.datastores."""
        from langchain_oci.datastores import (
            ADB,
            OpenSearch,
            VectorDataStore,
            create_datastore_tools,
        )

        assert VectorDataStore is not None
        assert OpenSearch is not None
        assert ADB is not None
        assert create_datastore_tools is not None

    def test_import_from_vectorstores(self) -> None:
        """Test imports from langchain_oci.datastores.vectorstores."""
        from langchain_oci.datastores.vectorstores import (
            ADB,
            OpenSearch,
            VectorDataStore,
        )

        assert VectorDataStore is not None
        assert OpenSearch is not None
        assert ADB is not None

    def test_import_from_top_level(self) -> None:
        """Test imports from langchain_oci top level."""
        from langchain_oci import (
            ADB,
            OpenSearch,
            VectorDataStore,
            create_datastore_tools,
        )

        assert VectorDataStore is not None
        assert OpenSearch is not None
        assert ADB is not None
        assert create_datastore_tools is not None
