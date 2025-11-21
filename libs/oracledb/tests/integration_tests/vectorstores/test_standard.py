from collections.abc import Generator
from typing import AsyncGenerator

import oracledb
import pytest
import pytest_asyncio
from langchain_core.vectorstores import VectorStore
from langchain_tests.integration_tests.vectorstores import VectorStoreIntegrationTests

from langchain_oracledb import OracleVS
from langchain_oracledb.embeddings import OracleEmbeddings
from langchain_oracledb.vectorstores.oraclevs import adrop_table_purge, drop_table_purge

username = ""
password = ""
dsn = ""

try:
    oracledb.connect(user=username, password=password, dsn=dsn)
except Exception as e:
    pytest.skip(
        allow_module_level=True,
        reason=f"Database connection failed: {e}, skipping tests.",
    )


class TestOracleVSStandardSync(VectorStoreIntegrationTests):
    @property
    def has_async(self) -> bool:
        """Configurable property to enable or disable sync tests."""
        return False

    @pytest.fixture()
    def vectorstore(self) -> Generator[VectorStore, None, None]:  # type: ignore[override]
        """Get an empty vectorstore for unit tests."""
        conn = oracledb.connect(user=username, password=password, dsn=dsn)
        drop_table_purge(conn, "standard_tests")
        store = OracleVS(
            conn,
            embedding_function=self.get_embeddings(),
            table_name="standard_tests",
            mutate_on_duplicate=True,
        )
        yield store


class TestOracleVSOracleEmbeddingsStandardSync(VectorStoreIntegrationTests):
    @property
    def has_async(self) -> bool:
        """Configurable property to enable or disable sync tests."""
        return False

    @pytest.fixture()
    def vectorstore(self) -> Generator[VectorStore, None, None]:  # type: ignore[override]
        """Get an empty vectorstore for unit tests."""
        conn = oracledb.connect(user=username, password=password, dsn=dsn)
        drop_table_purge(conn, "standard_tests")
        embedder_params = {"provider": "database", "model": "allminilm"}
        proxy = ""

        # instance
        model = OracleEmbeddings(conn=conn, params=embedder_params, proxy=proxy)

        store = OracleVS(
            conn,
            embedding_function=model,
            table_name="standard_tests",
            mutate_on_duplicate=True,
        )
        yield store


class TestOracleVSStandardAsync(VectorStoreIntegrationTests):
    @property
    def has_sync(self) -> bool:
        """Configurable property to enable or disable sync tests."""
        return False

    @pytest_asyncio.fixture
    async def vectorstore(self) -> AsyncGenerator[VectorStore, None]:
        """Get an empty vectorstore for unit tests (async version)."""

        conn = await oracledb.connect_async(user=username, password=password, dsn=dsn)
        await adrop_table_purge(conn, "standard_tests")

        store = await OracleVS.acreate(
            conn,
            embedding_function=self.get_embeddings(),
            table_name="standard_tests",
            mutate_on_duplicate=True,
        )
        yield store


class TestOracleVSOracleEmbeddingsStandardAsync(VectorStoreIntegrationTests):
    @property
    def has_sync(self) -> bool:
        """Configurable property to enable or disable sync tests."""
        return False

    @pytest_asyncio.fixture
    async def vectorstore(self) -> AsyncGenerator[VectorStore, None]:
        """Get an empty vectorstore for unit tests (async version)."""

        conn = await oracledb.connect_async(user=username, password=password, dsn=dsn)
        await adrop_table_purge(conn, "standard_tests")
        embedder_params = {"provider": "database", "model": "allminilm"}
        proxy = ""

        # OracleEmbeddings does not support asyncconnection
        conn_syn = oracledb.connect(user=username, password=password, dsn=dsn)
        model = OracleEmbeddings(conn=conn_syn, params=embedder_params, proxy=proxy)

        store = await OracleVS.acreate(
            conn,
            embedding_function=model,
            table_name="standard_tests",
            mutate_on_duplicate=True,
        )
        yield store
