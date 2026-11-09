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

from ._backend_fixtures import async_backend_store, backend_store, native_config


class TestOracleVSStandardSync(VectorStoreIntegrationTests):
    @property
    def has_async(self) -> bool:
        """Configurable property to enable or disable sync tests."""
        return False

    @pytest.fixture()
    def vectorstore(self, oracle_backend) -> Generator[VectorStore, None, None]:
        with backend_store(oracle_backend, self.get_embeddings()) as store:
            yield store


class TestOracleVSOracleEmbeddingsStandardSync(VectorStoreIntegrationTests):
    @property
    def has_async(self) -> bool:
        """Configurable property to enable or disable sync tests."""
        return False

    @pytest.fixture()
    def vectorstore(self) -> Generator[VectorStore, None, None]:
        """Get an empty vectorstore for unit tests."""
        conn = oracledb.connect(**native_config())
        try:
            drop_table_purge(conn, "standard_tests", backend="sql")
            embedder_params = {"provider": "database", "model": "allminilm"}
            proxy = ""

            # instance
            model = OracleEmbeddings(conn=conn, params=embedder_params, proxy=proxy)

            store = OracleVS(
                conn,
                embedding_function=model,
                table_name="standard_tests",
                mutate_on_duplicate=True,
                backend="sql",
            )
            yield store
        finally:
            try:
                drop_table_purge(conn, "standard_tests", backend="sql")
            finally:
                conn.close()


class TestOracleVSStandardAsync(VectorStoreIntegrationTests):
    @property
    def has_sync(self) -> bool:
        """Configurable property to enable or disable sync tests."""
        return False

    @pytest_asyncio.fixture
    async def vectorstore(self, oracle_backend) -> AsyncGenerator[VectorStore, None]:
        async with async_backend_store(oracle_backend, self.get_embeddings()) as store:
            yield store


class TestOracleVSOracleEmbeddingsStandardAsync(VectorStoreIntegrationTests):
    @property
    def has_sync(self) -> bool:
        """Configurable property to enable or disable sync tests."""
        return False

    @pytest_asyncio.fixture
    async def vectorstore(self) -> AsyncGenerator[VectorStore, None]:
        """Get an empty vectorstore for unit tests (async version)."""

        conn = await oracledb.connect_async(**native_config())
        conn_syn = None
        try:
            await adrop_table_purge(conn, "standard_tests", backend="sql")
            embedder_params = {"provider": "database", "model": "allminilm"}
            proxy = ""

            # OracleEmbeddings does not support asyncconnection
            conn_syn = oracledb.connect(**native_config())
            model = OracleEmbeddings(conn=conn_syn, params=embedder_params, proxy=proxy)

            store = await OracleVS.acreate(
                conn,
                embedding_function=model,
                table_name="standard_tests",
                mutate_on_duplicate=True,
                backend="sql",
            )
            yield store
        finally:
            try:
                await adrop_table_purge(conn, "standard_tests", backend="sql")
            finally:
                try:
                    if conn_syn is not None:
                        conn_syn.close()
                finally:
                    await conn.close()
