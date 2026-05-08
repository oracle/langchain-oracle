# type: ignore

"""Common fixtures for checkpoint tests."""

from contextlib import asynccontextmanager, contextmanager
from typing import Any

import oracledb
import pytest
from langgraph.checkpoint.base import EXCLUDED_METADATA_KEYS

from langgraph_oracledb.checkpoint.oracle import OracleSaver
from langgraph_oracledb.checkpoint.oracle.aio import AsyncOracleSaver
from tests.conftest import DEFAULT_CONNECTION_INFO, create_connection_string


# Custom object for serialization testing - defined at module level so it can be pickled
# Using underscore prefix to avoid pytest collection warnings
class _TestCustomObject:
    """A custom object for testing serialization."""

    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"_TestCustomObject(value={self.value!r})"


def _exclude_keys(config: dict[str, Any]) -> dict[str, Any]:
    """Exclude metadata keys from config."""
    return {k: v for k, v in config.items() if k not in EXCLUDED_METADATA_KEYS}


def _create_config(
    thread_id: str, checkpoint_ns: str = "", checkpoint_id: str = None
) -> dict:
    """Create a config dict with proper isolation (no shallow copy issues)."""
    config = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
        }
    }
    if checkpoint_id:
        config["configurable"]["checkpoint_id"] = checkpoint_id
    return config


def _cleanup_checkpoint_tables():
    """Clean up checkpoint tables with proper error handling."""
    try:
        with oracledb.connect(**DEFAULT_CONNECTION_INFO) as conn:
            with conn.cursor() as cur:
                # Clear data but keep shared schema objects in place for other tests.
                for table in [
                    "checkpoint_writes",
                    "checkpoint_blobs",
                    "checkpoints",
                    "checkpoint_migrations",
                ]:
                    try:
                        cur.execute(f"DELETE FROM {table}")
                    except oracledb.DatabaseError as e:
                        err = e.args[0]
                        if getattr(err, "code", None) != 942:
                            print(f"Warning: Failed to clear table {table}: {e}")
                conn.commit()
    except Exception as e:
        print(f"Warning: Cleanup failed: {e}")


async def _async_cleanup_checkpoint_tables():
    """Async version of checkpoint table cleanup."""
    try:
        async with await oracledb.connect_async(**DEFAULT_CONNECTION_INFO) as conn:
            async with conn.cursor() as cur:
                # Clear data but keep shared schema objects in place for other tests.
                for table in [
                    "checkpoint_writes",
                    "checkpoint_blobs",
                    "checkpoints",
                    "checkpoint_migrations",
                ]:
                    try:
                        await cur.execute(f"DELETE FROM {table}")
                    except oracledb.DatabaseError as e:
                        err = e.args[0]
                        if getattr(err, "code", None) != 942:
                            print(f"Warning: Failed to clear table {table}: {e}")
                await conn.commit()
    except Exception as e:
        print(f"Warning: Cleanup failed: {e}")


# Sync fixtures
@contextmanager
def _sync_pool_saver():
    """Fixture for pool mode testing."""
    # Create connection string for pool
    conn_string = create_connection_string(DEFAULT_CONNECTION_INFO)

    try:
        # yield checkpointer with pool configuration
        with OracleSaver.from_conn_string(
            conn_string,
            pool_config={"min_size": 2, "max_size": 10},
        ) as checkpointer:
            checkpointer.setup()
            yield checkpointer
    finally:
        _cleanup_checkpoint_tables()


@contextmanager
def _sync_base_saver():
    """Fixture for regular connection mode testing."""
    # Create connection string
    conn_string = create_connection_string(DEFAULT_CONNECTION_INFO)

    try:
        with OracleSaver.from_conn_string(
            conn_string,
        ) as checkpointer:
            checkpointer.setup()
            yield checkpointer
    finally:
        _cleanup_checkpoint_tables()


@contextmanager
def _sync_base_saver_with_serde(serde):
    """Fixture for regular connection mode testing with custom serde."""
    # Create connection string
    conn_string = create_connection_string(DEFAULT_CONNECTION_INFO)

    try:
        with OracleSaver.from_conn_string(
            conn_string,
            serde=serde,
        ) as checkpointer:
            checkpointer.setup()
            yield checkpointer
    finally:
        _cleanup_checkpoint_tables()


@contextmanager
def _sync_pool_saver_with_serde(serde):
    """Fixture for connection pool mode testing with custom serde."""
    # Create connection string
    conn_string = create_connection_string(DEFAULT_CONNECTION_INFO)

    try:
        with OracleSaver.from_conn_string(
            conn_string,
            pool_config={"min_size": 2, "max_size": 10},
            serde=serde,
        ) as checkpointer:
            checkpointer.setup()
            yield checkpointer
    finally:
        _cleanup_checkpoint_tables()


@contextmanager
def _sync_saver(name: str):
    """Wrapper to select between base/pool savers."""
    if name == "base":
        with _sync_base_saver() as saver:
            yield saver
    elif name == "pool":
        with _sync_pool_saver() as saver:
            yield saver


# Async fixtures
@asynccontextmanager
async def _async_pool_saver():
    """Fixture for pool mode testing."""
    # Create connection string for pool
    conn_string = create_connection_string(DEFAULT_CONNECTION_INFO)

    try:
        # yield checkpointer with pool configuration
        async with AsyncOracleSaver.from_conn_string(
            conn_string,
            pool_config={"min_size": 2, "max_size": 10},
        ) as checkpointer:
            await checkpointer.setup()
            yield checkpointer
    finally:
        await _async_cleanup_checkpoint_tables()


@asynccontextmanager
async def _async_base_saver():
    """Fixture for regular connection mode testing."""
    # Create connection string
    conn_string = create_connection_string(DEFAULT_CONNECTION_INFO)

    try:
        async with AsyncOracleSaver.from_conn_string(
            conn_string,
        ) as checkpointer:
            await checkpointer.setup()
            yield checkpointer
    finally:
        await _async_cleanup_checkpoint_tables()


@asynccontextmanager
async def _async_base_saver_with_serde(serde):
    """Fixture for regular connection mode testing with custom serde."""
    # Create connection string
    conn_string = create_connection_string(DEFAULT_CONNECTION_INFO)

    try:
        async with AsyncOracleSaver.from_conn_string(
            conn_string,
            serde=serde,
        ) as checkpointer:
            await checkpointer.setup()
            yield checkpointer
    finally:
        await _async_cleanup_checkpoint_tables()


@asynccontextmanager
async def _async_pool_saver_with_serde(serde):
    """Fixture for connection pool mode testing with custom serde."""
    # Create connection string
    conn_string = create_connection_string(DEFAULT_CONNECTION_INFO)

    try:
        async with AsyncOracleSaver.from_conn_string(
            conn_string,
            pool_config={"min_size": 2, "max_size": 10},
            serde=serde,
        ) as checkpointer:
            await checkpointer.setup()
            yield checkpointer
    finally:
        await _async_cleanup_checkpoint_tables()


@asynccontextmanager
async def _async_saver(name: str):
    """Async wrapper to select between base/pool savers."""
    if name == "base":
        async with _async_base_saver() as saver:
            yield saver
    elif name == "pool":
        async with _async_pool_saver() as saver:
            yield saver


@pytest.fixture
def test_data():
    """Fixture providing test data for checkpoint tests."""
    from langchain_core.runnables import RunnableConfig
    from langgraph.checkpoint.base import (
        Checkpoint,
        CheckpointMetadata,
        create_checkpoint,
        empty_checkpoint,
    )

    config_1: RunnableConfig = {
        "configurable": {
            "thread_id": "thread-1",
            "checkpoint_id": "1",
            "checkpoint_ns": "",
        }
    }
    config_2: RunnableConfig = {
        "configurable": {
            "thread_id": "thread-2",
            "checkpoint_id": "2",
            "checkpoint_ns": "",
        }
    }
    config_3: RunnableConfig = {
        "configurable": {
            "thread_id": "thread-2",
            "checkpoint_id": "2-inner",
            "checkpoint_ns": "inner",
        }
    }

    chkpnt_1: Checkpoint = empty_checkpoint()
    chkpnt_2: Checkpoint = create_checkpoint(chkpnt_1, {}, 1)
    chkpnt_3: Checkpoint = empty_checkpoint()

    metadata_1: CheckpointMetadata = {
        "source": "input",
        "step": 2,
        "writes": {},
        "score": 1,
    }
    metadata_2: CheckpointMetadata = {
        "source": "loop",
        "step": 1,
        "writes": {"foo": "bar"},
        "score": None,
    }
    metadata_3: CheckpointMetadata = {}

    return {
        "configs": [config_1, config_2, config_3],
        "checkpoints": [chkpnt_1, chkpnt_2, chkpnt_3],
        "metadata": [metadata_1, metadata_2, metadata_3],
    }
