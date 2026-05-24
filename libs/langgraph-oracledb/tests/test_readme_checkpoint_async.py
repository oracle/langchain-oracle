import uuid

import oracledb
import pytest

from langgraph_oracledb.checkpoint.oracle import AsyncOracleSaver, OracleSaver
from langgraph_oracledb.store.oracle import AsyncOracleStore, OracleStore
from tests.conftest import (
    DEFAULT_CONNECTION_INFO,
    create_connection_string,
    skip_if_no_oracle,
)
from tests.conftest_checkpointer import _cleanup_checkpoint_tables
from tests.embed_test_utils import CharacterEmbeddings


class TestREADME:
    @pytest.fixture(autouse=True)
    def run_after_each_test(self):
        yield
        _cleanup_checkpoint_tables()
        conn_string = create_connection_string(DEFAULT_CONNECTION_INFO)
        with oracledb.connect(conn_string) as connection:
            with connection.cursor() as conn:
                conn.execute("drop table if exists STORE_CONFIGS purge")

    @skip_if_no_oracle()
    @pytest.mark.asyncio
    async def test_readme_checkpoint_async_setup_only(self) -> None:
        """
        README example: AsyncOracleSaver basic setup.
        Ensures that the minimal example in the README executes without error.
        """
        conn_string = create_connection_string(DEFAULT_CONNECTION_INFO)
        async with AsyncOracleSaver.from_conn_string(conn_string) as checkpointer:
            await checkpointer.setup()  # create tables & apply migrations (idempotent)

    @skip_if_no_oracle()
    def test_readme_checkpoint_sync_setup_only(self) -> None:
        """
        README example: OracleSaver (sync) basic setup.
        """
        conn_string = create_connection_string(DEFAULT_CONNECTION_INFO)
        with OracleSaver.from_conn_string(conn_string) as checkpointer:
            checkpointer.setup()  # idempotent

    @skip_if_no_oracle()
    @pytest.mark.asyncio
    async def test_readme_store_async_put_get_search(self) -> None:
        """
        README example: AsyncOracleStore basic put/get/search without vectors.
        """
        conn_string = create_connection_string(DEFAULT_CONNECTION_INFO)
        async with AsyncOracleStore.from_conn_string(conn_string) as store:
            await store.setup()
            ns = ("readme", "example", uuid.uuid4().hex[:8])
            await store.aput(ns, "doc1", {"text": "hello"})
            item = await store.aget(ns, "doc1")
            assert item is not None
            assert item.value == {"text": "hello"}
            # Non-vector search returns latest items
            results = await store.asearch(ns, limit=10)
            assert isinstance(results, list)
            assert any(r.key == "doc1" for r in results)
            # Clean up tables created for this store configuration
            await store.ateardown()

    @skip_if_no_oracle()
    def test_readme_store_sync_put_get_search(self) -> None:
        """
        README example: OracleStore (sync) basic put/get/search without vectors.
        """
        conn_string = create_connection_string(DEFAULT_CONNECTION_INFO)
        with OracleStore.from_conn_string(conn_string) as store:
            store.setup()
            ns = ("readme", "example", uuid.uuid4().hex[:8])
            store.put(ns, "doc1", {"text": "hello"})
            item = store.get(ns, "doc1")
            assert item is not None
            assert item.value == {"text": "hello"}
            results = store.search(ns, limit=10)
            assert isinstance(results, list)
            assert any(r.key == "doc1" for r in results)
            # Clean up tables created for this store configuration
            store.teardown()

    @skip_if_no_oracle()
    @pytest.mark.asyncio
    async def test_readme_vector_async_fake_embeddings(self) -> None:
        """
        README example: AsyncOracleStore vector search with simple embeddings.
        Skips gracefully if Oracle AI Vector Search is not available on the target DB.
        """
        conn_string = create_connection_string(DEFAULT_CONNECTION_INFO)

        # Use a tiny embedding size to keep the example light
        fake = CharacterEmbeddings(dims=8)

        async with AsyncOracleStore.from_conn_string(
            conn_string,
            index={
                "dims": 8,
                "embed": fake,
                "fields": ["text"],
                "index_type": {
                    "type": "hnsw",
                    "neighbors": 16,
                    "efconstruction": 200,
                    "distance_metric": "COSINE",
                },
            },
        ) as store:
            await store.setup()

            ns = ("docs", uuid.uuid4().hex[:8])
            await store.aput(ns, "a", {"text": "alpha"})
            await store.aput(ns, "b", {"text": "beta"})
            results = await store.asearch(ns, query="alphabet", limit=2)
            assert isinstance(results, list)
            # Not asserting on specific ordering/score since embeddings are fake
            await store.ateardown()
