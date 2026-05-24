"""
Test persistence behavior for both sync and async Oracle stores.
Ensures data persists between sessions for both implementations.
"""

import uuid

import oracledb
import pytest

from langgraph_oracledb.store.oracle import AsyncOracleStore, OracleStore
from tests.conftest import DEFAULT_CONNECTION_INFO, create_connection_string
from tests.embed_test_utils import CharacterEmbeddings


class TestStorePersistenceWithMultipleSessions:
    """Test data persistence between store sessions."""

    @pytest.fixture(autouse=True)
    def run_after_each_test(self):
        yield
        conn_string = create_connection_string(DEFAULT_CONNECTION_INFO)
        with oracledb.connect(conn_string) as conn:
            with conn.cursor() as cur:
                cur.execute("drop table if exists STORE_CONFIGS purge")

    def test_sync_store_persistence_without_vector(self):
        """Test that sync OracleStore persists data between sessions without vector indexing."""
        # Create a unique table suffix for this test
        table_suffix = f"test_sync_persist_{uuid.uuid4().hex[:8]}"
        conn_string = create_connection_string(DEFAULT_CONNECTION_INFO)
        test_data = {
            "question": "What is sync persistence?",
            "answer": "Sync persistence means data survives between sessions.",
            "timestamp": 12345,
        }

        store2 = None
        try:
            # Session 1: Write data
            with OracleStore.from_conn_string(
                conn_string, table_suffix=table_suffix
            ) as store1:
                store1.setup()
                store1.put(("test", "persistence"), "sync_key_1", test_data)

                # Verify in same session
                item = store1.get(("test", "persistence"), "sync_key_1")
                assert item is not None
                assert item.value["question"] == test_data["question"]

            # Session 2: Read data (new store instance with same table_suffix)
            with OracleStore.from_conn_string(
                conn_string, table_suffix=table_suffix
            ) as store2:
                store2.setup()

                # Data should persist
                item = store2.get(("test", "persistence"), "sync_key_1")
                assert item is not None, (
                    "Data should persist between sync store sessions"
                )
                assert item.value["question"] == test_data["question"]
                assert item.value["answer"] == test_data["answer"]

                # Clean up after test
                store2.teardown()
        except Exception:
            # Clean up on failure
            if store2:
                store2.teardown()
            raise

    def test_sync_store_persistence_with_vector(self):
        """Test that sync OracleStore persists data between sessions with vector indexing."""
        table_suffix = f"test_sync_vector_{uuid.uuid4().hex[:8]}"
        conn_string = create_connection_string(DEFAULT_CONNECTION_INFO)
        embeddings = CharacterEmbeddings(dims=50)
        test_data = {
            "question": "What is vector persistence?",
            "answer": "Vector persistence includes semantic search capabilities.",
            "content": "Q: What is vector persistence?\nA: Vector persistence includes semantic search capabilities.",
        }

        store2 = None
        try:
            # Session 1: Write data
            with OracleStore.from_conn_string(
                conn_string,
                index={
                    "dims": embeddings.dims,
                    "embed": embeddings,
                    "index_type": {"type": "ivf", "distance_metric": "cosine"},
                    "fields": ["content"],
                },
                table_suffix=table_suffix,
            ) as store1:
                store1.setup()
                store1.put(("test", "vector"), "sync_vector_key_1", test_data)

                # Verify in same session
                item = store1.get(("test", "vector"), "sync_vector_key_1")
                assert item is not None
                assert item.value["question"] == test_data["question"]

            # Session 2: Read data and search
            with OracleStore.from_conn_string(
                conn_string,
                index={
                    "dims": embeddings.dims,
                    "embed": embeddings,
                    "index_type": {"type": "ivf", "distance_metric": "cosine"},
                    "fields": ["content"],
                },
                table_suffix=table_suffix,
            ) as store2:
                store2.setup()

                # Data should persist
                item = store2.get(("test", "vector"), "sync_vector_key_1")
                assert item is not None, (
                    "Vector data should persist between sync store sessions"
                )
                assert item.value["question"] == test_data["question"]

                # Search should work
                results = store2.search(
                    ("test", "vector"), query="vector persistence", limit=5
                )
                assert len(results) >= 1, "Should find the stored item via search"
                found_item = next(
                    (r for r in results if r.key == "sync_vector_key_1"), None
                )
                assert found_item is not None, (
                    "Should find the specific item via search"
                )

                # Clean up after test
                store2.teardown()
        except Exception:
            # Clean up on failure
            if store2:
                store2.teardown()
            raise

    @pytest.mark.asyncio
    async def test_async_store_persistence_without_vector(self):
        """Test that async AsyncOracleStore persists data between sessions without vector indexing."""
        table_suffix = f"test_async_persist_{uuid.uuid4().hex[:8]}"
        conn_string = create_connection_string(DEFAULT_CONNECTION_INFO)
        test_data = {
            "question": "What is async persistence?",
            "answer": "Async persistence means data survives between async sessions.",
            "timestamp": 67890,
        }

        store2 = None
        try:
            # Session 1: Write data
            async with AsyncOracleStore.from_conn_string(
                conn_string, table_suffix=table_suffix
            ) as store1:
                await store1.setup()
                await store1.aput(
                    ("test", "async_persistence"), "async_key_1", test_data
                )

                # Verify in same session
                item = await store1.aget(("test", "async_persistence"), "async_key_1")
                assert item is not None
                assert item.value["question"] == test_data["question"]

            # Session 2: Read data
            async with AsyncOracleStore.from_conn_string(
                conn_string, table_suffix=table_suffix
            ) as store2:
                await store2.setup()

                # Data should persist
                item = await store2.aget(("test", "async_persistence"), "async_key_1")
                assert item is not None, (
                    "Data should persist between async store sessions"
                )
                assert item.value["question"] == test_data["question"]
                assert item.value["answer"] == test_data["answer"]

                # Clean up after test
                await store2.ateardown()
        except Exception:
            # Clean up on failure
            if store2:
                await store2.ateardown()
            raise

    @pytest.mark.asyncio
    async def test_async_store_persistence_with_vector(self):
        """Test that async AsyncOracleStore persists data between sessions with vector indexing."""
        table_suffix = f"test_async_vector_{uuid.uuid4().hex[:8]}"
        conn_string = create_connection_string(DEFAULT_CONNECTION_INFO)
        embeddings = CharacterEmbeddings(dims=50)
        test_data = {
            "question": "What is async vector persistence?",
            "answer": "Async vector persistence includes batched semantic search.",
            "content": "Q: What is async vector persistence?\nA: Async vector persistence includes batched semantic search.",
        }

        store2 = None
        try:
            # Session 1: Write data
            async with AsyncOracleStore.from_conn_string(
                conn_string,
                index={
                    "dims": embeddings.dims,
                    "embed": embeddings,
                    "index_type": {"type": "ivf", "distance_metric": "cosine"},
                    "fields": ["content"],
                },
                table_suffix=table_suffix,
            ) as store1:
                await store1.setup()
                await store1.aput(
                    ("test", "async_vector"), "async_vector_key_1", test_data
                )

                # Verify in same session
                item = await store1.aget(("test", "async_vector"), "async_vector_key_1")
                assert item is not None
                assert item.value["question"] == test_data["question"]

            # Session 2: Read data and search
            async with AsyncOracleStore.from_conn_string(
                conn_string,
                index={
                    "dims": embeddings.dims,
                    "embed": embeddings,
                    "index_type": {"type": "ivf", "distance_metric": "cosine"},
                    "fields": ["content"],
                },
                table_suffix=table_suffix,
            ) as store2:
                await store2.setup()

                # Data should persist
                item = await store2.aget(("test", "async_vector"), "async_vector_key_1")
                assert item is not None, (
                    "Async vector data should persist between sessions"
                )
                assert item.value["question"] == test_data["question"]

                # Search should work
                results = await store2.asearch(
                    ("test", "async_vector"), query="async vector persistence", limit=5
                )
                assert len(results) >= 1, "Should find the stored item via async search"
                found_item = next(
                    (r for r in results if r.key == "async_vector_key_1"), None
                )
                assert found_item is not None, (
                    "Should find the specific item via async search"
                )

                # Clean up after test
                await store2.ateardown()
        except Exception:
            # Clean up on failure
            if store2:
                await store2.ateardown()
            raise

    @pytest.mark.asyncio
    async def test_cross_sync_async_compatibility(self):
        """Test that data written by sync can be read by async and vice versa."""
        table_suffix = f"test_cross_compat_{uuid.uuid4().hex[:8]}"
        conn_string = create_connection_string(DEFAULT_CONNECTION_INFO)

        sync_data = {"source": "sync", "message": "Written by sync store"}
        async_data = {"source": "async", "message": "Written by async store"}

        sync_store_final = None
        try:
            # Sync writes
            with OracleStore.from_conn_string(
                conn_string, table_suffix=table_suffix
            ) as sync_store:
                sync_store.setup()
                sync_store.put(("cross", "test"), "sync_item", sync_data)

            # Async reads sync data and writes async data
            async with AsyncOracleStore.from_conn_string(
                conn_string, table_suffix=table_suffix
            ) as async_store:
                await async_store.setup()

                # Should read sync data
                sync_item = await async_store.aget(("cross", "test"), "sync_item")
                assert sync_item is not None, "Async should read sync-written data"
                assert sync_item.value["source"] == "sync"

                # Write async data
                await async_store.aput(("cross", "test"), "async_item", async_data)

            # Sync reads async data and cleans up
            with OracleStore.from_conn_string(
                conn_string, table_suffix=table_suffix
            ) as sync_store_final:
                sync_store_final.setup()

                async_item = sync_store_final.get(("cross", "test"), "async_item")
                assert async_item is not None, "Sync should read async-written data"
                assert async_item.value["source"] == "async"

                # Clean up after test
                sync_store_final.teardown()
        except Exception:
            # Clean up on failure
            if sync_store_final:
                sync_store_final.teardown()
            raise
