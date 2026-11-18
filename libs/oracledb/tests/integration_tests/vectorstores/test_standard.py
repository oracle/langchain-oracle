import os
from collections.abc import Generator
from contextlib import contextmanager
from typing import AsyncGenerator

import oracledb
import pytest
import pytest_asyncio
from langchain_core.vectorstores import VectorStore
from langchain_tests.integration_tests.vectorstores import VectorStoreIntegrationTests

from langchain_oracledb.embeddings import OracleEmbeddings

from ._backend_fixtures import (
    async_backend_store,
    backend_store,
    native_config,
    rest_client,
)


@contextmanager
def oracle_embeddings(backend):
    client = (
        rest_client()
        if backend == "rest"
        else oracledb.connect(**native_config(backend))
    )
    model = os.environ.get(
        "ORACLE_REST_EMBEDDING_MODEL"
        if backend == "rest"
        else "ORACLE_EMBEDDINGS_MODEL",
        "ALLMINILM",
    )
    try:
        yield OracleEmbeddings(
            conn=client, params={"provider": "database", "model": model}
        )
    finally:
        if backend != "rest":
            client.close()


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
    def vectorstore(self, oracle_backend) -> Generator[VectorStore, None, None]:
        with oracle_embeddings(oracle_backend) as model:
            with backend_store(oracle_backend, model) as store:
                yield store


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
    async def vectorstore(self, oracle_backend) -> AsyncGenerator[VectorStore, None]:
        # Embeddings use a sync native connection or REST SDK client even when
        # vector-store operations use the async driver.
        with oracle_embeddings(oracle_backend) as model:
            async with async_backend_store(oracle_backend, model) as store:
                yield store
