"""Store-specific test fixtures and utilities."""

import sys
import uuid
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager

import oracledb
import pytest
from langchain_core.embeddings import Embeddings

from langgraph_oracledb.store.oracle import AsyncOracleStore, OracleStore
from tests.conftest import (
    DEFAULT_CONNECTION_INFO,
    create_connection_string,
)
from tests.embed_test_utils import CharacterEmbeddings


@pytest.fixture
def fake_embeddings() -> CharacterEmbeddings:
    return CharacterEmbeddings(dims=500)


# Oracle-specific constants for store testing
ORACLE_INDEX_TYPES = ["hnsw", "ivf"]
ORACLE_DISTANCE_TYPES = ["EUCLIDEAN", "DOT", "COSINE"]


# Embedding types for testing
def get_embedding_params():
    """Get embedding parameters including both fake embeddings and db model if available."""
    params = [("fake", "CharacterEmbeddings")]
    return params


# TTL configuration for tests
TTL_SECONDS = 6
TTL_MINUTES = TTL_SECONDS / 60


def get_oracle_connection_params():
    """Get Oracle connection parameters for sync tests."""
    import os
    from pathlib import Path

    # Check environment variables first
    dsn = os.getenv("ORACLE_DSN")
    username = os.getenv("ORACLE_USERNAME")
    password = os.getenv("ORACLE_PASSWORD")

    if not all([dsn, username, password]):
        # Try to read from .env file
        env_file = Path(__file__).parent.parent / ".env"
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        key, value = line.split("=", 1)
                        if key == "ORACLE_DSN":
                            dsn = value
                        elif key == "ORACLE_USERNAME":
                            username = value
                        elif key == "ORACLE_PASSWORD":
                            password = value

    if not all([dsn, username, password]):
        pytest.skip("Oracle connection parameters not available")

    return {"user": username, "password": password, "dsn": dsn}


@asynccontextmanager
async def _create_async_vector_store(
    index_type: str,
    distance_type: str,
    embedding_config: CharacterEmbeddings | str,
    text_fields: list[str] | None = None,
) -> AsyncIterator[AsyncOracleStore]:
    """Create an async store with vector search enabled."""
    if sys.version_info < (3, 10):
        pytest.skip("Async Oracle tests require Python 3.10+")

    # Check if Oracle is available
    try:
        # Try to establish a connection
        with oracledb.connect(**DEFAULT_CONNECTION_INFO):
            pass
    except Exception:
        pytest.skip("Oracle database not available")

    # Generate unique table suffix for this vector store
    table_suffix = f"vec_{uuid.uuid4().hex[:8]}"

    distance_mapping = {
        "l2": "EUCLIDEAN",
        "inner_product": "DOT",
        "cosine": "COSINE",
        # Support Oracle native names directly
        "euclidean": "EUCLIDEAN",
        "dot": "DOT",
        "EUCLIDEAN": "EUCLIDEAN",
        "DOT": "DOT",
        "COSINE": "COSINE",
    }

    oracle_distance = distance_mapping.get(distance_type.lower())
    if oracle_distance is None:
        oracle_distance = distance_type.upper()  # Assume it's already an Oracle metric
        if oracle_distance not in ("COSINE", "EUCLIDEAN", "DOT"):
            raise ValueError(
                f"Unsupported distance metric: {distance_type}. "
                f"Supported Oracle distance metrics: COSINE, EUCLIDEAN, DOT"
            )

    index_config = {
        "dims": embedding_config.dims,
        "embed": embedding_config,
        "index_type": {
            "type": index_type,
            "distance_metric": oracle_distance,
        },
        "fields": text_fields,
    }

    conn_string = create_connection_string(DEFAULT_CONNECTION_INFO)
    async with AsyncOracleStore.from_conn_string(
        conn_string,
        index=index_config,
        table_suffix=table_suffix,
    ) as store:
        await store.setup()
        try:
            yield store
        finally:
            await store.ateardown()
            async with oracledb.connect_async(conn_string) as conn:
                await conn.execute("drop table if exists STORE_CONFIGS purge")


@contextmanager
def _create_sync_vector_store(
    index_type: str,
    distance_type: str,
    embedding_config: Embeddings | str,
    text_fields: list[str] | None = None,
    enable_ttl: bool = True,
) -> Iterator[OracleStore]:
    """Create a sync store with vector search enabled."""
    conn_params = get_oracle_connection_params()

    # Create a unique table suffix for this test
    table_suffix = f"test_{uuid.uuid4().hex[:8]}"

    # Use Oracle distance metrics directly
    distance_mapping = {
        "l2": "EUCLIDEAN",
        "inner_product": "DOT",
        "cosine": "COSINE",
        # Support Oracle native names directly
        "euclidean": "EUCLIDEAN",
        "dot": "DOT",
        "EUCLIDEAN": "EUCLIDEAN",
        "DOT": "DOT",
        "COSINE": "COSINE",
    }

    oracle_distance = distance_mapping.get(distance_type.lower())
    if oracle_distance is None:
        oracle_distance = distance_type.upper()  # Assume it's already an Oracle metric
        if oracle_distance not in ("COSINE", "EUCLIDEAN", "DOT"):
            raise ValueError(
                f"Unsupported distance metric: {distance_type}. "
                f"Supported Oracle distance metrics: COSINE, EUCLIDEAN, DOT"
            )

    index_config = {
        "dims": embedding_config.dims,
        "embed": embedding_config,
        "index_type": {
            "type": index_type,
            "distance_metric": oracle_distance,
        },
        "fields": text_fields,
    }

    with oracledb.connect(**conn_params) as conn:
        try:
            store = OracleStore(
                conn,
                index=index_config,
                ttl={"default_ttl": 2, "refresh_on_read": True} if enable_ttl else None,
                table_suffix=table_suffix,
            )

            store.setup()
            yield store
        finally:
            # Clean up using the store's teardown method
            store.teardown()
            with conn.cursor() as cur:
                cur.execute("drop table if exists STORE_CONFIGS purge")


# Async store fixtures


@pytest.fixture(scope="function", params=["default", "pool"])
async def async_store(request) -> AsyncIterator[AsyncOracleStore]:
    """Create an async Oracle store with TTL configuration."""
    if sys.version_info < (3, 10):
        pytest.skip("Async Oracle tests require Python 3.10+")

    # Check if Oracle is available
    try:
        # Try to establish a connection
        with oracledb.connect(**DEFAULT_CONNECTION_INFO):
            pass
    except Exception:
        pytest.skip("Oracle database not available")

    # Generate unique table suffix for this test
    table_suffix = f"test_{uuid.uuid4().hex[:8]}"

    ttl_config = {
        "default_ttl": TTL_MINUTES,
        "refresh_on_read": True,
        "sweep_interval_minutes": TTL_MINUTES / 2,
    }

    conn_string = create_connection_string(DEFAULT_CONNECTION_INFO)

    # Create connection
    if request.param == "pool":
        # Use connection pooling for Oracle with individual parameters

        async with AsyncOracleStore.from_conn_string(
            conn_string,
            pool_config={"min_size": 1, "max_size": 10},
            ttl=ttl_config,
            table_suffix=table_suffix,
        ) as store:
            await store.setup()
            await store.setup()  # Test idempotent setup
            await store.start_ttl_sweeper()
            try:
                yield store
            finally:
                await store.stop_ttl_sweeper()
                await store.ateardown()  # Clean up using store's teardown method
                async with oracledb.connect_async(conn_string) as conn:
                    await conn.execute("drop table if exists STORE_CONFIGS purge")

    elif "dsn" in DEFAULT_CONNECTION_INFO:
        # Use dsn-based connection
        async with AsyncOracleStore.from_conn_string(
            conn_string, ttl=ttl_config, table_suffix=table_suffix
        ) as store:
            await store.setup()
            await store.start_ttl_sweeper()
            try:
                yield store
            finally:
                await store.stop_ttl_sweeper()
                await store.ateardown()
                async with oracledb.connect_async(conn_string) as conn:
                    await conn.execute("drop table if exists STORE_CONFIGS purge")

    else:
        # Use individual parameters
        async with AsyncOracleStore.from_conn_info(
            **DEFAULT_CONNECTION_INFO,
            ttl=ttl_config,
            table_suffix=table_suffix,
        ) as store:
            await store.setup()
            await store.start_ttl_sweeper()
            try:
                yield store
            finally:
                await store.stop_ttl_sweeper()
                await store.ateardown()
                async with oracledb.connect_async(conn_string) as conn:
                    await conn.execute("drop table if exists STORE_CONFIGS purge")


@pytest.fixture(
    scope="function",
    params=[
        (index_type, distance_type, embed_type, embed_value)
        for index_type in ORACLE_INDEX_TYPES
        for distance_type in ORACLE_DISTANCE_TYPES
        for embed_type, embed_value in get_embedding_params()
    ],
    ids=lambda p: f"{p[0]}_{p[1]}_{p[2]}",
)
async def async_vector_store(
    request,
    fake_embeddings: CharacterEmbeddings,
) -> AsyncIterator[AsyncOracleStore]:
    """Create an async store with vector search enabled."""
    index_type, distance_type, embed_type, embed_value = request.param

    # Determine the embedding configuration to use
    if embed_type == "fake":
        embedding_config = fake_embeddings
    else:  # embed_type == "db"
        embedding_config = embed_value  # This is the string "admin.ALL_MINILM_L12_V2"

    async with _create_async_vector_store(
        index_type, distance_type, embedding_config
    ) as store:
        yield store


# Sync store fixtures


@pytest.fixture(scope="function", params=["default", "pool"])
def sync_store(request) -> Iterator[OracleStore]:
    """Create a sync test Oracle store with fresh tables."""
    conn_params = get_oracle_connection_params()

    # Create a unique table suffix for this test
    table_suffix = f"test_{uuid.uuid4().hex[:8]}"

    ttl_config = {
        "default_ttl": TTL_MINUTES,
        "refresh_on_read": True,
        "sweep_interval_minutes": TTL_MINUTES / 2,
    }

    if request.param == "pool":
        # Use connection pooling for Oracle with connection string
        conn_string = create_connection_string(DEFAULT_CONNECTION_INFO)
        try:
            with OracleStore.from_conn_string(
                conn_string,
                ttl=ttl_config,
                table_suffix=table_suffix,
                pool_config={"min_size": 2, "max_size": 10},
            ) as store:
                store.setup()
                store.start_ttl_sweeper()
                yield store
                store.stop_ttl_sweeper()
        finally:
            # Clean up using the store's teardown method and drop config table
            with oracledb.connect(conn_string) as conn:
                with conn.cursor() as cur:
                    cur.execute("drop table if exists STORE_CONFIGS purge")
    else:
        with oracledb.connect(**conn_params) as conn:
            try:
                store = OracleStore(conn, ttl=ttl_config, table_suffix=table_suffix)
                store.setup()
                store.start_ttl_sweeper()
                yield store
                store.stop_ttl_sweeper()
            finally:
                # Clean up using the store's teardown method
                store.teardown()
                with conn.cursor() as cur:
                    cur.execute("drop table if exists STORE_CONFIGS purge")


# Create parameter combinations for Oracle vector store testing
_oracle_vector_params = [
    (index_type, distance_type, True, embed_type, embed_value)
    for index_type in ORACLE_INDEX_TYPES
    for distance_type in ["EUCLIDEAN", "DOT", "COSINE"]
    for embed_type, embed_value in get_embedding_params()
]
# Add one test with TTL disabled for each embedding type
_oracle_vector_params_ttl_disabled = [
    (index_type, distance_type, False, embed_type, embed_value)
    for index_type in ORACLE_INDEX_TYPES[
        :1
    ]  # Just one index type to keep it manageable
    for distance_type in ["COSINE"]  # Just one distance type
    for embed_type, embed_value in get_embedding_params()
]
_oracle_vector_params += _oracle_vector_params_ttl_disabled


@pytest.fixture(
    scope="function",
    params=_oracle_vector_params,
    ids=lambda p: f"{p[0]}_{p[1]}_{p[3]}",
)
def sync_vector_store(
    request,
    fake_embeddings: Embeddings,
) -> Iterator[OracleStore]:
    """Create a sync store with vector search enabled."""
    index_type, distance_type, enable_ttl, embed_type, embed_value = request.param

    # Determine the embedding configuration to use
    if embed_type == "fake":
        embedding_config = fake_embeddings
    else:  # embed_type == "db"
        embedding_config = embed_value  # This is the string "admin.ALL_MINILM_L12_V2"

    with _create_sync_vector_store(
        index_type, distance_type, embedding_config, enable_ttl=enable_ttl
    ) as store:
        yield store


# Compatibility aliases for existing tests
store = async_store  # For async tests that use 'store' fixture name
vector_store = sync_vector_store  # For sync tests that use 'vector_store' fixture name


# Helper functions for custom vector store creation
def create_async_vector_store_with_fields(
    index_type: str,
    distance_type: str,
    embedding_config: CharacterEmbeddings | str,
    text_fields: list[str],
):
    """Create async vector store with custom text fields."""
    return _create_async_vector_store(
        index_type, distance_type, embedding_config, text_fields
    )


def create_sync_vector_store_with_fields(
    index_type: str,
    distance_type: str,
    embedding_config: CharacterEmbeddings | str,
    text_fields: list[str],
):
    """Create sync vector store with custom text fields."""
    return _create_sync_vector_store(
        index_type, distance_type, embedding_config, text_fields
    )
