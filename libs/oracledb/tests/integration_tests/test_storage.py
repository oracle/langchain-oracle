# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Integration tests for OracleByteStore / OracleDocStore (issue #269)."""

import os
import uuid
from collections.abc import Generator
from typing import Tuple

import oracledb
import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.stores import BaseStore
from langchain_tests.integration_tests import BaseStoreSyncTests

from langchain_oracledb.storage import OracleByteStore, OracleDocStore

username = os.environ.get("VECDB_USER")
password = os.environ.get("VECDB_PASS")
dsn = os.environ.get("VECDB_HOST")

try:
    oracledb.connect(user=username, password=password, dsn=dsn)
except Exception as e:
    pytest.skip(
        allow_module_level=True,
        reason=f"Database connection failed: {e}, skipping tests.",
    )


@pytest.fixture(scope="function")
def connection() -> Generator[oracledb.Connection, None, None]:
    conn = oracledb.connect(user=username, password=password, dsn=dsn)
    try:
        yield conn
    finally:
        try:
            conn.close()
        except Exception:
            pass


class TestOracleByteStoreStandard(BaseStoreSyncTests[bytes]):
    @pytest.fixture
    def kv_store(
        self, connection: oracledb.Connection
    ) -> Generator[BaseStore[str, bytes], None, None]:
        table = f"BYTE_STORE_{uuid.uuid4().hex[:8].upper()}"
        store = OracleByteStore(connection, table_name=table)
        try:
            yield store
        finally:
            OracleByteStore.drop_table(connection, table)

    @pytest.fixture
    def three_values(self) -> Tuple[bytes, bytes, bytes]:
        return b"alpha", b"beta", b"\x00\x01\xff binary"


class TestOracleDocStoreStandard(BaseStoreSyncTests[Document]):
    @pytest.fixture
    def kv_store(
        self, connection: oracledb.Connection
    ) -> Generator[BaseStore[str, Document], None, None]:
        table = f"DOC_STORE_{uuid.uuid4().hex[:8].upper()}"
        store = OracleDocStore(connection, table_name=table)
        try:
            yield store
        finally:
            OracleDocStore.drop_table(connection, table)

    @pytest.fixture
    def three_values(self) -> Tuple[Document, Document, Document]:
        return (
            Document(page_content="alpha", metadata={"n": 1}),
            Document(page_content="beta", metadata={"nested": {"a": [1, 2]}}),
            Document(page_content="gamma", id="doc-3"),
        )


def test_yield_keys_prefix_escapes_wildcards(
    connection: oracledb.Connection,
) -> None:
    table = f"BYTE_STORE_{uuid.uuid4().hex[:8].upper()}"
    store = OracleByteStore(connection, table_name=table)
    try:
        store.mset([("a%b/1", b"x"), ("axb/2", b"y"), ("a_b/3", b"z")])
        # % and _ in the prefix must match literally, not as LIKE wildcards
        assert list(store.yield_keys(prefix="a%b")) == ["a%b/1"]
        assert list(store.yield_keys(prefix="a_b")) == ["a_b/3"]
    finally:
        OracleByteStore.drop_table(connection, table)


def test_cache_backed_embeddings_end_to_end(
    connection: oracledb.Connection,
) -> None:
    """OracleByteStore powers CacheBackedEmbeddings: second run hits the store."""
    try:
        # langchain 1.x moved it to langchain-classic
        from langchain_classic.embeddings import CacheBackedEmbeddings
    except ImportError:
        try:
            from langchain.embeddings import (  # type: ignore[attr-defined,no-redef]
                CacheBackedEmbeddings,
            )
        except ImportError:
            pytest.skip("CacheBackedEmbeddings not available")
    table = f"EMB_CACHE_{uuid.uuid4().hex[:8].upper()}"
    store = OracleByteStore(connection, table_name=table)
    underlying = DeterministicFakeEmbedding(size=8)
    try:
        cached = CacheBackedEmbeddings.from_bytes_store(
            underlying, store, namespace="fake-model"
        )
        first = cached.embed_documents(["hello", "world"])
        assert len(list(store.yield_keys())) == 2

        second = cached.embed_documents(["hello", "world"])
        assert first == second
    finally:
        OracleByteStore.drop_table(connection, table)


def test_parent_document_retriever_end_to_end(
    connection: oracledb.Connection,
) -> None:
    """OracleDocStore + OracleVS power ParentDocumentRetriever."""
    try:
        # langchain 1.x moved it to langchain-classic
        from langchain_classic.retrievers import ParentDocumentRetriever
    except ImportError:
        try:
            from langchain.retrievers import (  # type: ignore[attr-defined,no-redef]
                ParentDocumentRetriever,
            )
        except ImportError:
            pytest.skip("ParentDocumentRetriever not available")
    splitters = pytest.importorskip("langchain_text_splitters")

    from langchain_oracledb.vectorstores import OracleVS
    from langchain_oracledb.vectorstores.oraclevs import drop_table_purge
    from langchain_oracledb.vectorstores.utils import DistanceStrategy

    vs_table = f"PDR_VS_{uuid.uuid4().hex[:8].upper()}"
    doc_table = f"PDR_DOCS_{uuid.uuid4().hex[:8].upper()}"
    vectorstore = OracleVS(
        client=connection,
        embedding_function=DeterministicFakeEmbedding(size=16),
        table_name=vs_table,
        distance_strategy=DistanceStrategy.COSINE,
    )
    docstore = OracleDocStore(connection, table_name=doc_table)
    try:
        retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=docstore,
            child_splitter=splitters.RecursiveCharacterTextSplitter(
                chunk_size=64, chunk_overlap=0
            ),
        )
        parent = Document(
            page_content=(
                "Oracle Database 23ai introduces AI Vector Search. "
                "It supports HNSW and IVF vector indexes. "
                "Bananas are rich in potassium and unrelated to databases."
            ),
            metadata={"source": "doc-1"},
        )
        retriever.add_documents([parent])

        results = retriever.invoke("vector indexes in Oracle")
        assert len(results) == 1
        # The retriever returns the FULL parent, not the matched child chunk.
        assert results[0].page_content == parent.page_content
    finally:
        drop_table_purge(connection, vs_table)
        OracleDocStore.drop_table(connection, doc_table)
