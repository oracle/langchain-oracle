from contextlib import asynccontextmanager, contextmanager
from re import search
from unittest.mock import Mock

import pytest
from langchain_core.embeddings import Embeddings

from langgraph_oracledb.store.oracle import AsyncOracleStore, OracleStore


class _MockEmbeddings(Embeddings):
    def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3, 0.4]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]


class _FakeConnection:
    def commit(self) -> None:
        return None


class _FakeAsyncConnection:
    async def commit(self) -> None:
        return None


class _SetupCursor:
    def __init__(self, versions: dict[str, int], existing_tables: set[str]) -> None:
        self.versions = {name.upper(): version for name, version in versions.items()}
        self.existing_tables = {name.upper() for name in existing_tables}
        self.connection = _FakeConnection()
        self._row = None

    def execute(self, query: str, params: tuple[object, ...] | None = None) -> None:
        normalized = " ".join(query.upper().split())
        if "FROM USER_TABLES" in normalized:
            table_name = str(params[0]).upper() if params else ""
            self._row = (1,) if table_name in self.existing_tables else None
            return

        version_match = search(r"SELECT V FROM ([A-Z0-9_]+)", normalized)
        if version_match:
            table_name = version_match.group(1)
            version = self.versions.get(table_name)
            self._row = (version,) if version is not None else None
            return

        self._row = None

    def fetchone(self):
        return self._row


class _AsyncSetupCursor(_SetupCursor):
    def __init__(self, versions: dict[str, int], existing_tables: set[str]) -> None:
        super().__init__(versions, existing_tables)
        self.connection = _FakeAsyncConnection()

    async def execute(
        self, query: str, params: tuple[object, ...] | None = None
    ) -> None:
        super().execute(query, params)

    async def fetchone(self):
        return self._row


def _vector_index_config():
    return {
        "dims": 4,
        "embed": _MockEmbeddings(),
        "index_type": {"type": "hnsw", "distance_metric": "COSINE"},
    }


def test_sync_setup_fails_when_store_table_missing_but_migration_recorded() -> None:
    store = OracleStore(Mock(), table_suffix="demo")

    @contextmanager
    def fake_cursor():
        yield _SetupCursor(
            versions={"store_migrations_demo": 0},
            existing_tables=set(),
        )

    store._cursor = fake_cursor

    with pytest.raises(RuntimeError, match="STORE_DEMO is missing"):
        store.setup()

    assert store.table_names["store"] == "store_demo"


def test_sync_setup_fails_when_vector_table_missing_but_migration_recorded() -> None:
    store = OracleStore(Mock(), index=_vector_index_config(), table_suffix="demo")
    store._validate_configuration = lambda *_args, **_kwargs: None

    @contextmanager
    def fake_cursor():
        yield _SetupCursor(
            versions={
                "store_migrations_demo": 4,
                "vector_migrations_demo": 1,
            },
            existing_tables={"store_demo", "store_configs"},
        )

    store._cursor = fake_cursor

    with pytest.raises(RuntimeError, match="STORE_VECTORS_DEMO is missing"):
        store.setup()

    assert store.table_names["store_vectors"] == "store_vectors_demo"


@pytest.mark.asyncio
async def test_async_setup_fails_when_store_table_missing_but_migration_recorded() -> None:
    store = AsyncOracleStore(Mock(), table_suffix="demo")

    @asynccontextmanager
    async def fake_cursor():
        yield _AsyncSetupCursor(
            versions={"store_migrations_demo": 0},
            existing_tables=set(),
        )

    store._cursor = fake_cursor

    with pytest.raises(RuntimeError, match="STORE_DEMO is missing"):
        await store.setup()

    assert store.table_names["store"] == "store_demo"


@pytest.mark.asyncio
async def test_async_setup_fails_when_vector_table_missing_but_migration_recorded() -> None:
    store = AsyncOracleStore(Mock(), index=_vector_index_config(), table_suffix="demo")

    async def validate_configuration_noop(*_args, **_kwargs) -> None:
        return None

    store._validate_configuration = validate_configuration_noop

    @asynccontextmanager
    async def fake_cursor():
        yield _AsyncSetupCursor(
            versions={
                "store_migrations_demo": 4,
                "vector_migrations_demo": 1,
            },
            existing_tables={"store_demo", "store_configs"},
        )

    store._cursor = fake_cursor

    with pytest.raises(RuntimeError, match="STORE_VECTORS_DEMO is missing"):
        await store.setup()

    assert store.table_names["store_vectors"] == "store_vectors_demo"
