"""
Test persistence behavior for both sync and async Oracle checkpointers.
Ensures checkpoints persist between sessions for both implementations.
"""

import asyncio
import uuid

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    CheckpointMetadata,
    create_checkpoint,
    empty_checkpoint,
)
from langgraph.checkpoint.base.id import uuid6

from langgraph_oracledb.checkpoint.oracle import OracleSaver
from langgraph_oracledb.checkpoint.oracle.aio import (
    AsyncOracleSaver,
)
from tests.conftest import DEFAULT_CONNECTION_INFO, create_connection_string
from tests.conftest_checkpointer import _cleanup_checkpoint_tables


class TestCheckpointWithMultipleSessions:
    """Test checkpoint persistence between checkpointer sessions."""

    @pytest.fixture(autouse=True)
    def run_after_each_test(self):
        yield
        _cleanup_checkpoint_tables()

    def test_sync_checkpoint_persistence(self):
        """Test that sync OracleSaver persists checkpoints between sessions."""
        # Create unique identifiers for this test
        thread_id = f"test_sync_persist_{uuid.uuid4().hex[:8]}"
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
        conn_string = create_connection_string(DEFAULT_CONNECTION_INFO)

        # Create a simple checkpoint using the same pattern as existing tests
        base_checkpoint = empty_checkpoint()
        checkpoint_data = create_checkpoint(base_checkpoint, {}, 1)

        metadata: CheckpointMetadata = {
            "source": "sync_test",
            "step": 1,
            "writes": {"test_key": "Hello sync world"},
            "score": None,
        }

        # Session 1: Save checkpoint
        with OracleSaver.from_conn_string(conn_string) as checkpointer1:
            checkpointer1.setup()
            saved_config = checkpointer1.put(config, checkpoint_data, metadata, {})

            # Verify in same session
            saved_tuple = checkpointer1.get_tuple(saved_config)
            assert saved_tuple is not None
            assert saved_tuple.checkpoint["id"] == checkpoint_data["id"]
            assert saved_tuple.metadata["source"] == "sync_test"

        # Session 2: Load checkpoint (uses same connection string so same tables)
        with OracleSaver.from_conn_string(conn_string) as checkpointer2:
            checkpointer2.setup()

            # Checkpoint should persist (use original config, not saved_config)
            loaded_tuple = checkpointer2.get_tuple(config)
            assert loaded_tuple is not None, (
                "Checkpoint should persist between sync sessions"
            )
            assert loaded_tuple.checkpoint["id"] == checkpoint_data["id"]
            assert loaded_tuple.metadata["source"] == "sync_test"
            assert loaded_tuple.metadata["step"] == 1
            assert loaded_tuple.metadata["writes"]["test_key"] == "Hello sync world"

    @pytest.mark.asyncio
    async def test_async_checkpoint_persistence(self):
        """Test that async AsyncOracleSaver persists checkpoints between sessions."""
        # Create a test checkpoint
        thread_id = f"test_async_persist_{uuid.uuid4().hex[:8]}"
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

        # Create a simple checkpoint using the same pattern as existing tests
        base_checkpoint = empty_checkpoint()
        checkpoint_data = create_checkpoint(base_checkpoint, {}, 1)

        metadata: CheckpointMetadata = {
            "source": "async_test",
            "step": 1,
            "writes": {"test_key": "Hello async world"},
            "score": None,
        }

        # Session 1: Save checkpoint
        conn_string = create_connection_string(DEFAULT_CONNECTION_INFO)
        async with AsyncOracleSaver.from_conn_string(conn_string) as checkpointer1:
            await checkpointer1.setup()
            await checkpointer1.aput(config, checkpoint_data, metadata, {})

            # Verify in same session
            saved_tuple = await checkpointer1.aget_tuple(config)
            assert saved_tuple is not None
            assert saved_tuple.checkpoint["id"] == checkpoint_data["id"]
            assert saved_tuple.metadata["source"] == "async_test"

        # Session 2: Load checkpoint
        async with AsyncOracleSaver.from_conn_string(conn_string) as checkpointer2:
            await checkpointer2.setup()

            # Checkpoint should persist
            loaded_tuple = await checkpointer2.aget_tuple(config)
            assert loaded_tuple is not None, (
                "Checkpoint should persist between async sessions"
            )
            assert loaded_tuple.checkpoint["id"] == checkpoint_data["id"]
            assert loaded_tuple.metadata["source"] == "async_test"
            assert loaded_tuple.metadata["step"] == 1
            assert loaded_tuple.metadata["writes"]["test_key"] == "Hello async world"

    def test_sync_checkpoint_history_persistence(self):
        """Test that sync checkpoint history persists between sessions."""
        thread_id = f"test_sync_history_{uuid.uuid4().hex[:8]}"
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

        # Create multiple checkpoints
        checkpoints = []
        for i in range(3):
            checkpoint = {
                "v": 1,
                "id": f"checkpoint_{i}_{uuid.uuid4().hex}",
                "ts": f"2024-01-01T00:0{i}:00Z",
                "channel_values": {
                    "messages": [{"type": "human", "content": f"Message {i}"}]
                },
                "channel_versions": {"messages": f"v{i}"},
                "versions_seen": {},
            }
            metadata = {"source": "sync_history_test", "step": i}
            checkpoints.append((checkpoint, metadata))

        # Session 1: Save multiple checkpoints
        conn_string = create_connection_string(DEFAULT_CONNECTION_INFO)
        with OracleSaver.from_conn_string(conn_string) as checkpointer1:
            checkpointer1.setup()

            for checkpoint, metadata in checkpoints:
                checkpointer1.put(config, checkpoint, metadata, {})

        # Session 2: Verify history persists
        with OracleSaver.from_conn_string(conn_string) as checkpointer2:
            checkpointer2.setup()

            # Get checkpoint history
            history = list(checkpointer2.list(config, limit=10))
            assert len(history) >= 3, (
                "Checkpoint history should persist between sessions"
            )

            # Verify the content of checkpoints
            checkpoint_ids = {cp.checkpoint["id"] for cp in history}
            expected_ids = {cp[0]["id"] for cp in checkpoints}
            assert expected_ids.issubset(checkpoint_ids), (
                "All saved checkpoints should be in history"
            )

    @pytest.mark.asyncio
    async def test_async_checkpoint_history_persistence(self):
        """Test that async checkpoint history persists between sessions."""
        thread_id = f"test_async_history_{uuid.uuid4().hex[:8]}"
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

        # Create multiple checkpoints
        checkpoints = []
        for i in range(3):
            checkpoint = {
                "v": 1,
                "id": f"checkpoint_{i}_{uuid.uuid4().hex}",
                "ts": f"2024-01-01T00:0{i}:00Z",
                "channel_values": {
                    "messages": [{"type": "human", "content": f"Async message {i}"}]
                },
                "channel_versions": {"messages": f"v{i}"},
                "versions_seen": {},
            }
            metadata = {"source": "async_history_test", "step": i}
            checkpoints.append((checkpoint, metadata))

        # Session 1: Save multiple checkpoints
        conn_string = create_connection_string(DEFAULT_CONNECTION_INFO)
        async with AsyncOracleSaver.from_conn_string(conn_string) as checkpointer1:
            await checkpointer1.setup()

            for checkpoint, metadata in checkpoints:
                await checkpointer1.aput(config, checkpoint, metadata, {})

        # Session 2: Verify history persists
        async with AsyncOracleSaver.from_conn_string(conn_string) as checkpointer2:
            await checkpointer2.setup()

            # Get checkpoint history
            history = []
            async for cp in checkpointer2.alist(config, limit=10):
                history.append(cp)

            assert len(history) >= 3, (
                "Async checkpoint history should persist between sessions"
            )

            # Verify the content of checkpoints
            checkpoint_ids = {cp.checkpoint["id"] for cp in history}
            expected_ids = {cp[0]["id"] for cp in checkpoints}
            assert expected_ids.issubset(checkpoint_ids), (
                "All saved async checkpoints should be in history"
            )

    @pytest.mark.asyncio
    async def test_sync_async_checkpoint_compatibility(self):
        """Test that checkpoints written by sync can be read by async and vice versa."""
        thread_id = f"test_cross_checkpoint_{uuid.uuid4().hex[:8]}"
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

        # Use uuid6 for proper time-ordered checkpoint IDs
        sync_checkpoint = {
            "v": 1,
            "id": str(uuid6(clock_seq=0)),
            "ts": "2024-01-01T00:00:00Z",
            "channel_values": {"source": "sync", "data": "from sync checkpointer"},
            "channel_versions": {"source": "v1"},
            "versions_seen": {},
        }

        async_checkpoint = {
            "v": 1,
            "id": str(uuid6(clock_seq=1)),  # Higher clock_seq for later checkpoint
            "ts": "2024-01-01T00:01:00Z",
            "channel_values": {"source": "async", "data": "from async checkpointer"},
            "channel_versions": {"source": "v2"},
            "versions_seen": {},
        }

        # Sync writes, async reads
        conn_string = create_connection_string(DEFAULT_CONNECTION_INFO)
        with OracleSaver.from_conn_string(conn_string=conn_string) as sync_checkpointer:
            sync_checkpointer.setup()
            sync_checkpointer.put(config, sync_checkpoint, {"writer": "sync"}, {})

        async with AsyncOracleSaver.from_conn_string(
            conn_string=conn_string
        ) as async_checkpointer:
            await async_checkpointer.setup()

            # Should read sync checkpoint
            loaded_tuple = await async_checkpointer.aget_tuple(config)
            assert loaded_tuple is not None, "Async should read sync checkpoint"
            assert loaded_tuple.checkpoint["channel_values"]["source"] == "sync"

            # Async writes
            await async_checkpointer.aput(
                config, async_checkpoint, {"writer": "async"}, {}
            )

        # Sync reads async checkpoint
        with OracleSaver.from_conn_string(conn_string=conn_string) as sync_checkpointer:
            sync_checkpointer.setup()

            # Should read the latest (async) checkpoint
            loaded_tuple = sync_checkpointer.get_tuple(config)
            assert loaded_tuple is not None, "Sync should read async checkpoint"
            assert loaded_tuple.checkpoint["channel_values"]["source"] == "async"

    def test_checkpoint_with_nested_config(self):
        """Test checkpoint persistence with complex nested configurations."""
        thread_id = f"test_nested_{uuid.uuid4().hex[:8]}"
        config: RunnableConfig = {
            "configurable": {"thread_id": thread_id, "checkpoint_ns": "test_namespace"}
        }

        complex_checkpoint = {
            "v": 1,
            "id": f"complex_{uuid.uuid4().hex}",
            "ts": "2024-01-01T00:00:00Z",
            "channel_values": {
                "nested_data": {
                    "level1": {
                        "level2": {
                            "values": [1, 2, 3],
                            "metadata": {"type": "test", "version": "1.0"},
                        }
                    }
                },
                "simple_value": "test_string",
            },
            "channel_versions": {"nested_data": "v1", "simple_value": "v1"},
            "versions_seen": {},
        }

        complex_metadata = {
            "complex_meta": {"nested": True, "items": ["a", "b", "c"]},
            "step": 1,
        }

        # Save with sync
        conn_string = create_connection_string(DEFAULT_CONNECTION_INFO)
        with OracleSaver.from_conn_string(conn_string=conn_string) as checkpointer:
            checkpointer.setup()
            # Pass the channel versions from the checkpoint as new_versions so blobs are stored
            new_versions = complex_checkpoint["channel_versions"]
            checkpointer.put(config, complex_checkpoint, complex_metadata, new_versions)

        # Load with sync to verify persistence
        with OracleSaver.from_conn_string(conn_string=conn_string) as checkpointer:
            checkpointer.setup()

            loaded_tuple = checkpointer.get_tuple(config)
            assert loaded_tuple is not None, "Complex checkpoint should persist"

            # Verify nested structure is preserved
            nested = loaded_tuple.checkpoint["channel_values"]["nested_data"]["level1"][
                "level2"
            ]
            assert nested["values"] == [1, 2, 3]
            assert nested["metadata"]["type"] == "test"

            # Verify metadata structure
            assert loaded_tuple.metadata["complex_meta"]["nested"] is True
            assert loaded_tuple.metadata["complex_meta"]["items"] == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_concurrent_checkpoint_sessions(self):
        """Test that concurrent async checkpoint sessions work correctly."""
        base_thread_id = f"concurrent_{uuid.uuid4().hex[:8]}"
        conn_string = create_connection_string(DEFAULT_CONNECTION_INFO)

        async def write_checkpoint(session_id: int):
            thread_id = f"{base_thread_id}_{session_id}"
            config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

            checkpoint = {
                "v": 1,
                "id": f"concurrent_{session_id}_{uuid.uuid4().hex}",
                "ts": f"2024-01-01T00:0{session_id}:00Z",
                "channel_values": {
                    "session": session_id,
                    "data": f"Session {session_id} data",
                },
                "channel_versions": {"session": f"v{session_id}"},
                "versions_seen": {},
            }
            async with AsyncOracleSaver.from_conn_string(
                conn_string=conn_string
            ) as checkpointer:
                await checkpointer.setup()
                await checkpointer.aput(config, checkpoint, {"session": session_id}, {})
                return thread_id, checkpoint["id"]

        # Run multiple concurrent sessions
        results = await asyncio.gather(*[write_checkpoint(i) for i in range(5)])

        # Verify all checkpoints were saved correctly
        for thread_id, checkpoint_id in results:
            config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

            async with AsyncOracleSaver.from_conn_string(
                conn_string=conn_string
            ) as checkpointer:
                await checkpointer.setup()

                loaded_tuple = await checkpointer.aget_tuple(config)
                assert loaded_tuple is not None, (
                    f"Concurrent checkpoint {checkpoint_id} should persist"
                )
                assert loaded_tuple.checkpoint["id"] == checkpoint_id
