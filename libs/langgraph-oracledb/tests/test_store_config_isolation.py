"""Test configuration isolation and table suffix functionality."""

from contextlib import asynccontextmanager, contextmanager
from unittest.mock import Mock
from uuid import uuid4

import pytest
from langchain_core.embeddings import Embeddings

from langgraph_oracledb.store.oracle.base import (
    OracleStore,
    _generate_suffix,
    _get_parameters_clause,
)


class _MockEmbeddings(Embeddings):
    dims = 5

    def embed_query(self, text: str):
        return [0.1, 0.2, 0.3, 0.4]

    def embed_documents(self, texts: list[str]):
        return [self.embed_query(text) for text in texts]


class _FakeCursor:
    def __init__(self, row):
        self._row = row

    def execute(self, query, params):
        return None

    def fetchone(self):
        return self._row


class _FakeAsyncCursor:
    def __init__(self, row):
        self._row = row

    async def execute(self, query, params):
        return None

    async def fetchone(self):
        return self._row


class _FakeLob:
    def __init__(self, value: str):
        self._value = value

    def read(self):
        return self._value


def test_suffix_generation():
    """Test that configuration suffix generation is deterministic and unique."""

    embs = _MockEmbeddings()
    config1 = {
        "dims": embs.dims,
        "embed": embs,
        "index_type": {"type": "hnsw", "neighbors": 16, "distance_metric": "COSINE"},
        "fields": ["$"],
    }

    embs = _MockEmbeddings()
    config2 = {
        "dims": embs.dims,
        "embed": embs,
        "index_type": {"type": "hnsw", "neighbors": 16, "distance_metric": "COSINE"},
        "fields": ["$"],
    }

    embs = _MockEmbeddings()
    config3 = {
        "dims": embs.dims,
        "embed": embs,
        "index_type": {
            "type": "hnsw",
            "neighbors": 16,
            "distance_metric": "EUCLIDEAN",
        },  # Different distance
        "fields": ["$"],
    }

    # Same configurations should generate same suffix
    suffix1 = _generate_suffix(config1)
    suffix2 = _generate_suffix(config2)
    assert suffix1 == suffix2, "Same configurations should generate same suffix"

    # Different configurations should generate different suffix
    suffix3 = _generate_suffix(config3)
    assert suffix1 != suffix3, (
        "Different configurations should generate different suffix"
    )


def test_table_suffix_rejects_sql_metacharacters():
    with pytest.raises(ValueError, match="table_suffix"):
        OracleStore(Mock(), table_suffix="safe_suffix;drop_table")


@pytest.mark.asyncio
async def test_async_table_suffix_rejects_sql_metacharacters():
    from langgraph_oracledb.store.oracle.aio import AsyncOracleStore

    with pytest.raises(ValueError, match="table_suffix"):
        AsyncOracleStore(Mock(), table_suffix="safe_suffix;drop_table")


def test_vector_index_config_rejects_untrusted_ddl_fragments():
    config = {
        "dims": 5,
        "embed": _MockEmbeddings(),
        "index_type": {
            "type": "hnsw",
            "distance_metric": "COSINE) PARAMETERS (type IVF",
        },
    }

    with pytest.raises(ValueError, match="distance_metric"):
        OracleStore(Mock(), index=config)


def test_vector_index_config_rejects_non_numeric_parameters():
    config = {
        "dims": 5,
        "embed": _MockEmbeddings(),
        "index_type": {
            "type": "hnsw",
            "distance_metric": "COSINE",
            "neighbors": "16) PARAMETERS (type IVF",
        },
    }

    with pytest.raises(ValueError, match="index_type.neighbors must be an integer"):
        OracleStore(Mock(), index=config)


@pytest.mark.parametrize("legacy_key", ["m", "ef_construction"])
def test_hnsw_index_config_rejects_unsupported_keys(legacy_key):
    config = {
        "dims": 5,
        "embed": _MockEmbeddings(),
        "index_type": {
            "type": "hnsw",
            "distance_metric": "COSINE",
            legacy_key: 16,
        },
    }

    with pytest.raises(ValueError, match=f"unsupported keys: {legacy_key}"):
        OracleStore(Mock(), index=config)


def test_ivf_index_config_rejects_unsupported_keys():
    config = {
        "dims": 5,
        "embed": _MockEmbeddings(),
        "index_type": {
            "type": "ivf",
            "distance_metric": "COSINE",
            "sample_per_partition": 1,
        },
    }

    with pytest.raises(ValueError, match="unsupported keys: sample_per_partition"):
        OracleStore(Mock(), index=config)


def test_vector_index_config_accepts_oracle_26_documented_hnsw_bounds():
    config = {
        "dims": 5,
        "embed": _MockEmbeddings(),
        "accuracy": 1,
        "index_type": {
            "type": "hnsw",
            "distance_metric": "DOT",
            "neighbors": 2048,
            "efconstruction": 65535,
        },
    }

    store = OracleStore(Mock(), index=config)

    assert _get_parameters_clause(store) == (
        "PARAMETERS (type HNSW, neighbors 2048, efconstruction 65535)"
    )


def test_vector_index_config_accepts_oracle_26_documented_ivf_bounds():
    config = {
        "dims": 5,
        "embed": _MockEmbeddings(),
        "accuracy": 100,
        "index_type": {
            "type": "ivf",
            "distance_metric": "EUCLIDEAN",
            "neighbor_partitions": 1,
            "samples_per_partition": 1,
            "min_vectors_per_partition": 0,
        },
    }

    store = OracleStore(Mock(), index=config)

    assert _get_parameters_clause(store) == (
        "PARAMETERS (type IVF, neighbor partitions 1, "
        "samples_per_partition 1, min_vectors_per_partition 0)"
    )


def test_validate_configuration_parses_json_string_params():
    store = OracleStore(Mock())

    @contextmanager
    def fake_cursor():
        yield _FakeCursor(
            (
                1536,
                "COSINE",
                '{"type": "hnsw", "distance_metric": "COSINE", "accuracy": 90}',
            )
        )

    store._cursor = fake_cursor

    with pytest.raises(ValueError, match="Index accuracy type mismatch"):
        store._validate_configuration(
            "test_suffix",
            {
                "dims": 1536,
                "index_type": {"type": "hnsw", "distance_metric": "COSINE"},
                "accuracy": 95,
            },
        )


def test_validate_configuration_detects_dimension_mismatch():
    store = OracleStore(Mock())

    @contextmanager
    def fake_cursor():
        yield _FakeCursor(
            (
                1024,
                "COSINE",
                '{"type": "hnsw", "distance_metric": "COSINE", "accuracy": 95}',
            )
        )

    store._cursor = fake_cursor

    with pytest.raises(ValueError, match="Dimension mismatch"):
        store._validate_configuration(
            "test_suffix",
            {
                "dims": 1536,
                "index_type": {"type": "hnsw", "distance_metric": "COSINE"},
                "accuracy": 95,
            },
        )


def test_validate_configuration_detects_distance_mismatch():
    store = OracleStore(Mock())

    @contextmanager
    def fake_cursor():
        yield _FakeCursor(
            (
                1536,
                "COSINE",
                '{"type": "hnsw", "distance_metric": "COSINE", "accuracy": 95}',
            )
        )

    store._cursor = fake_cursor

    with pytest.raises(ValueError, match="Distance type mismatch"):
        store._validate_configuration(
            "test_suffix",
            {
                "dims": 1536,
                "index_type": {"type": "hnsw", "distance_metric": "DOT"},
                "accuracy": 95,
            },
        )


def test_validate_configuration_detects_index_param_mismatch():
    store = OracleStore(Mock())

    @contextmanager
    def fake_cursor():
        yield _FakeCursor(
            (
                1536,
                "COSINE",
                '{"type": "hnsw", "m": 16, "distance_metric": "COSINE", "accuracy": 95}',
            )
        )

    store._cursor = fake_cursor

    with pytest.raises(ValueError, match="Index parameter mismatch"):
        store._validate_configuration(
            "test_suffix",
            {
                "dims": 1536,
                "index_type": {
                    "type": "hnsw",
                    "m": 32,
                    "distance_metric": "COSINE",
                },
                "accuracy": 95,
            },
        )


def test_validate_configuration_rejects_invalid_json_string_params():
    store = OracleStore(Mock())

    @contextmanager
    def fake_cursor():
        yield _FakeCursor((1536, "COSINE", '{"type": "hnsw"'))

    store._cursor = fake_cursor

    with pytest.raises(ValueError, match="not valid JSON"):
        store._validate_configuration(
            "test_suffix",
            {
                "dims": 1536,
                "index_type": {"type": "hnsw", "distance_metric": "COSINE"},
                "accuracy": None,
            },
        )


@pytest.mark.asyncio
async def test_async_validate_configuration_reads_lob_like_params():
    from langgraph_oracledb.store.oracle.aio import AsyncOracleStore

    store = AsyncOracleStore(Mock())

    @asynccontextmanager
    async def fake_cursor():
        yield _FakeAsyncCursor(
            (
                1536,
                "COSINE",
                _FakeLob(
                    '{"type": "hnsw", "distance_metric": "COSINE", "accuracy": 95}'
                ),
            )
        )

    store._cursor = fake_cursor

    await store._validate_configuration(
        "test_suffix",
        {
            "dims": 1536,
            "index_type": {"type": "hnsw", "distance_metric": "COSINE"},
            "accuracy": 95,
        },
    )


@pytest.mark.asyncio
async def test_async_validate_configuration_detects_distance_mismatch():
    from langgraph_oracledb.store.oracle.aio import AsyncOracleStore

    store = AsyncOracleStore(Mock())

    @asynccontextmanager
    async def fake_cursor():
        yield _FakeAsyncCursor(
            (
                1536,
                "COSINE",
                _FakeLob(
                    '{"type": "hnsw", "distance_metric": "COSINE", "accuracy": 95}'
                ),
            )
        )

    store._cursor = fake_cursor

    with pytest.raises(ValueError, match="Distance type mismatch"):
        await store._validate_configuration(
            "test_suffix",
            {
                "dims": 1536,
                "index_type": {"type": "hnsw", "distance_metric": "DOT"},
                "accuracy": 95,
            },
        )


async def test_configuration_isolation():
    """Test that different configurations create different table sets."""
    import oracledb

    from langgraph_oracledb.store.oracle.aio import AsyncOracleStore
    from tests.conftest import DEFAULT_CONNECTION_INFO, create_connection_string

    # Check if Oracle is available
    try:
        with oracledb.connect(**DEFAULT_CONNECTION_INFO):
            pass
    except Exception:
        pytest.skip("Oracle database not available")

    # Mock embedding functions with different dimensions
    class MockEmbeddings384(Embeddings):
        def embed_query(self, text: str):
            return [0.1] * 384

        def embed_documents(self, texts: list[str]):
            return [self.embed_query(text) for text in texts]

    class MockEmbeddings1536(Embeddings):
        def embed_query(self, text: str):
            return [0.1] * 1536

        def embed_documents(self, texts: list[str]):
            return [self.embed_query(text) for text in texts]

    # Create two stores with different embedding dimensions using proper connection
    conn_string = create_connection_string(DEFAULT_CONNECTION_INFO)
    async with AsyncOracleStore.from_conn_string(
        conn_string,
        index={
            "dims": 384,
            "embed": MockEmbeddings384(),
            "index_type": {"type": "hnsw", "distance_metric": "COSINE"},
        },
    ) as store1:
        await store1.setup()
        async with AsyncOracleStore.from_conn_string(
            conn_string,
            index={
                "dims": 1536,
                "embed": MockEmbeddings1536(),
                "index_type": {"type": "hnsw", "distance_metric": "COSINE"},
            },
        ) as store2:
            await store2.setup()
            # They should have different table suffixes
            assert store1.table_suffix != store2.table_suffix
            assert store1.table_names["store"] != store2.table_names["store"]

            await store2.ateardown()
        await store1.ateardown()
        async with oracledb.connect_async(conn_string) as conn:
            await conn.execute("drop table if exists STORE_CONFIGS purge")


async def test_same_config_same_tables():
    """Test that identical configurations use the same table suffix."""
    import oracledb

    from langgraph_oracledb.store.oracle.aio import AsyncOracleStore
    from tests.conftest import DEFAULT_CONNECTION_INFO, create_connection_string

    # Check if Oracle is available
    try:
        with oracledb.connect(**DEFAULT_CONNECTION_INFO):
            pass
    except Exception:
        pytest.skip("Oracle database not available")

    class MockEmbeddings(Embeddings):
        def embed_query(self, text: str):
            return [0.1] * 1536

        def embed_documents(self, texts: list[str]):
            return [self.embed_query(text) for text in texts]

    config = {
        "dims": 1536,
        "embed": MockEmbeddings(),
        "index_type": {"type": "hnsw", "neighbors": 16, "distance_metric": "COSINE"},
    }

    # Create two stores with identical configuration using proper connection
    conn_string = create_connection_string(DEFAULT_CONNECTION_INFO)
    async with AsyncOracleStore.from_conn_string(
        conn_string, index=config.copy()
    ) as store1:
        await store1.setup()
        async with AsyncOracleStore.from_conn_string(
            conn_string, index=config.copy()
        ) as store2:
            await store2.setup()
            # They should have the same table suffix
            assert store1.table_suffix == store2.table_suffix
            assert store1.table_names["store"] == store2.table_names["store"]
            await store2.ateardown()

        await store1.ateardown()
        async with oracledb.connect_async(conn_string) as conn:
            await conn.execute("drop table if exists STORE_CONFIGS purge")


async def test_explicit_table_suffix():
    """Test that explicit table_suffix parameter works correctly."""
    import oracledb

    from langgraph_oracledb.store.oracle.aio import AsyncOracleStore
    from tests.conftest import DEFAULT_CONNECTION_INFO, create_connection_string

    # Check if Oracle is available
    try:
        with oracledb.connect(**DEFAULT_CONNECTION_INFO):
            pass
    except Exception:
        pytest.skip("Oracle database not available")

    class MockEmbeddings(Embeddings):
        def embed_query(self, text: str):
            return [0.1] * 1536

        def embed_documents(self, texts: list[str]):
            return [self.embed_query(text) for text in texts]

    config = {
        "dims": 1536,
        "embed": MockEmbeddings(),
        "index_type": {"type": "hnsw", "distance_metric": "COSINE"},
    }

    # Create stores with explicit table suffixes using proper connection
    test_suffix = f"test_{uuid4().hex[:8]}"
    conn_string = create_connection_string(DEFAULT_CONNECTION_INFO)
    async with AsyncOracleStore.from_conn_string(
        conn_string, index=config, table_suffix=test_suffix
    ) as store1:
        await store1.setup()
        async with AsyncOracleStore.from_conn_string(
            conn_string, index=config, table_suffix="custom_suffix"
        ) as store2:
            await store2.setup()
            # They should use the specified suffixes
            assert store1.table_suffix == test_suffix
            assert store2.table_suffix == "custom_suffix"
            assert store1.table_names["store"] == f"store_{test_suffix}"
            assert store2.table_names["store"] == "store_custom_suffix"
            await store2.ateardown()

        await store1.ateardown()
        async with oracledb.connect_async(conn_string) as conn:
            await conn.execute("drop table if exists STORE_CONFIGS purge")


async def test_vector_setup_creates_expected_schema_objects():
    """Vector store setup should create the expected tables, indexes, and config row."""
    import oracledb

    from langgraph_oracledb.store.oracle.aio import AsyncOracleStore
    from tests.conftest import DEFAULT_CONNECTION_INFO, create_connection_string

    try:
        with oracledb.connect(**DEFAULT_CONNECTION_INFO):
            pass
    except Exception:
        pytest.skip("Oracle database not available")

    class MockEmbeddings(Embeddings):
        def embed_query(self, text: str):
            return [0.1] * 1536

        def embed_documents(self, texts: list[str]):
            return [self.embed_query(text) for text in texts]

    test_suffix = f"schema_{uuid4().hex[:8]}"
    conn_string = create_connection_string(DEFAULT_CONNECTION_INFO)
    config = {
        "dims": 1536,
        "embed": MockEmbeddings(),
        "index_type": {"type": "hnsw", "distance_metric": "COSINE"},
    }

    async with AsyncOracleStore.from_conn_string(
        conn_string, index=config, table_suffix=test_suffix
    ) as store:
        await store.setup()
        expected_vector_index = (
            f"{store.table_names['store_vectors']}_idx_"
            f"{hash(str(store.index_config)) % 1000000}"
        ).upper()

        try:
            with oracledb.connect(**DEFAULT_CONNECTION_INFO) as conn:
                with conn.cursor() as cur:
                    for table_name in (
                        store.table_names["store"],
                        store.table_names["store_vectors"],
                        store.table_names["store_migrations"],
                        store.table_names["vector_migrations"],
                    ):
                        cur.execute(
                            "SELECT COUNT(*) FROM user_tables WHERE table_name = :1",
                            (table_name.upper(),),
                        )
                        assert cur.fetchone()[0] == 1

                    for index_name in (
                        f"{store.table_names['store']}_prefix_idx",
                        f"idx_{store.table_names['store']}_expires_at",
                        expected_vector_index,
                    ):
                        cur.execute(
                            "SELECT COUNT(*) FROM user_indexes WHERE index_name = :1",
                            (index_name.upper(),),
                        )
                        assert cur.fetchone()[0] == 1

                    cur.execute(
                        """
                        SELECT detected_dims, distance_type, embed_fields
                        FROM store_configs
                        WHERE table_suffix = :1
                        """,
                        (test_suffix,),
                    )
                    row = cur.fetchone()
                    assert row is not None
                    assert int(row[0]) == 1536
                    assert row[1] == "COSINE"
                    assert row[2] == "$"
        finally:
            await store.ateardown()
            async with oracledb.connect_async(conn_string) as conn:
                await conn.execute("drop table if exists STORE_CONFIGS purge")


def test_non_vector_store_suffix():
    """Test table suffix behavior for non-vector stores."""
    import oracledb

    from langgraph_oracledb.store.oracle.base import OracleStore
    from tests.conftest import DEFAULT_CONNECTION_INFO

    # Check if Oracle is available
    try:
        with oracledb.connect(**DEFAULT_CONNECTION_INFO):
            pass
    except Exception:
        pytest.skip("Oracle database not available")

    # Non-vector store with default suffix (novec for non-vector stores)
    with oracledb.connect(**DEFAULT_CONNECTION_INFO) as conn:
        store1 = OracleStore(conn=conn)
        store1.setup()

        assert store1.table_suffix == "novec"
        assert store1.table_names["store"] == "store_novec"

        # Non-vector store with explicit suffix
        store2 = OracleStore(conn=conn, table_suffix="custom")
        store2.setup()
        assert store2.table_suffix == "custom"
        assert store2.table_names["store"] == "store_custom"

        store1.teardown()
        store2.teardown()
        with conn.cursor() as cur:
            cur.execute("drop table if exists STORE_CONFIGS purge")
