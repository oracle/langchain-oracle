# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""Only sparse-storage rejection of a new auto store permits SQL fallback."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import oracledb
import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from langchain_oracledb import OracleVS
from langchain_oracledb.vectorstores import _vecdb, oraclevs

from . import test_vecdb_pool_sessions as pool_sessions

sessions = pool_sessions.sessions
SPARSE_ERROR = (
    "ORA-00406: COMPATIBLE parameter needs to be 23.6.0.0.0 or greater\n"
    'ORA-00722: Feature "SQL VECTOR SPARSE format"'
)


def database_error(code=406, message=SPARSE_ERROR):
    return oracledb.DatabaseError(SimpleNamespace(code=code, message=message))


@pytest.fixture
def creation(sessions, monkeypatch):
    s = sessions
    pool_sessions.use_one_session(s)
    s.a.columns = set()
    s.error = None
    s.partial_columns = set()
    s.probe_error = None
    s.index_error = None
    s.create_attempts = 0
    original = s.a.execute

    def execute(statement, parameters=None, **binds):
        if statement == _vecdb.VECDB_CAPABILITY_SQL:
            s.a.log.append((statement, binds))
            s.a.one = (1,)
            return
        if "INDEX_BUILD_STATUS" in statement and s.index_error is not None:
            raise s.index_error
        if "create_vector_table(" in statement.lower():
            s.a.log.append((statement, binds))
            s.create_attempts += 1
            if s.error is not None:
                s.a.columns = set(s.partial_columns)
                raise s.error
        if (
            s.create_attempts
            and s.probe_error is not None
            and statement.startswith("SELECT id, text, metadata, embedding FROM ")
        ):
            raise s.probe_error
        return original(statement, parameters, **binds)

    s.a.execute = execute
    if not s.asynchronous:
        s.a.cursor.execute.side_effect = execute
    s.sql_create = AsyncMock() if s.asynchronous else MagicMock()
    s.sql_index = AsyncMock() if s.asynchronous else MagicMock()
    monkeypatch.setattr(
        oraclevs, "_acreate_table" if s.asynchronous else "_create_table", s.sql_create
    )
    monkeypatch.setattr(
        oraclevs,
        "_aensure_sql_constructor_index"
        if s.asynchronous
        else "_ensure_sql_constructor_index",
        s.sql_index,
    )
    return s


async def open_store(s, *, backend="auto", params=None):
    args = (s.pool, DeterministicFakeEmbedding(size=1), "DOCS")
    if s.asynchronous:
        return await OracleVS.acreate(*args, backend=backend, params=params)
    return OracleVS(*args, backend=backend, params=params)


@pytest.mark.parametrize("feature", ["format", "type"])
@pytest.mark.parametrize("code", [406, 45493])
@pytest.mark.asyncio
async def test_sparse_rejection_falls_back_on_the_same_checkout(
    creation, feature, code
):
    s = creation
    message = SPARSE_ERROR.replace("format", feature)
    if code == 45493:
        message = (
            "ORA-45493: feature not enabled by rolling patch\n" + message.split("\n")[1]
        )
    s.error = database_error(code, message)
    store = await open_store(s)
    assert store.backend == "sql"
    assert store._vecdb is None
    assert store.client is s.pool
    assert s.create_attempts == 1
    assert s.pool.acquire.call_count == 1
    assert not s.b.log
    s.sql_create.assert_called_once()
    assert s.sql_create.call_args.args[:3] == (s.a.connection, '"DOCS"', 1)
    assert s.sql_create.call_args.kwargs == {"new_table_only": True}
    s.a.connection.rollback.assert_not_called()
    assert not any("DROP" in sql.upper() for sql, _ in s.a.log)


@pytest.mark.asyncio
async def test_success_uses_vecdb_without_compatibility_or_absence_recheck(creation):
    s = creation
    store = await open_store(s)
    assert store.backend == "vecdb"
    assert store._vecdb.transport.client is s.pool
    assert s.create_attempts == 1
    assert (
        sum(
            sql.startswith("SELECT id, text, metadata, embedding FROM ")
            for sql, _ in s.a.log
        )
        == 1
    )
    assert not any("COMPATIBLE" in sql or "DB_VERSION" in sql for sql, _ in s.a.log)
    s.sql_create.assert_not_called()


@pytest.mark.asyncio
async def test_sql_fallback_keeps_constructor_index_request(creation):
    s = creation
    s.error = database_error()
    store = await open_store(s, params={"vector_index_params": {"idx_type": "HNSW"}})
    assert store.backend == "sql"
    s.sql_index.assert_called_once()
    assert "CREATE VECTOR INDEX" in s.sql_index.call_args.args[2][1].upper()


@pytest.mark.parametrize("backend", ["vecdb", "auto"])
@pytest.mark.asyncio
async def test_existing_vecdb_and_explicit_requests_never_fall_back(creation, backend):
    s = creation
    error = database_error()
    if backend == "auto":
        s.a.columns = set(_vecdb.CURRENT_COLUMNS)
        s.index_error = error
        params = {"vector_index_params": {}}
    else:
        s.error = error
        params = None
    with pytest.raises(RuntimeError) as caught:
        await open_store(s, backend=backend, params=params)
    assert caught.value.__cause__ is error
    s.sql_create.assert_not_called()


@pytest.mark.parametrize(
    "error",
    [
        database_error(1031, "ORA-01031: insufficient privileges"),
        database_error(4031, "ORA-04031: out of shared memory"),
        database_error(6550, "ORA-06550: PLS-00306: wrong arguments"),
        database_error(406, 'ORA-00406\nORA-00722: Feature "SQL VECTOR BINARY type"'),
        database_error(406, "ORA-00406: COMPATIBLE needs to be increased"),
        database_error(722, SPARSE_ERROR),
        database_error(45493, 'ORA-45493\nORA-00722: Feature "another feature"'),
        RuntimeError(SPARSE_ERROR),
    ],
)
@pytest.mark.asyncio
async def test_other_failures_never_trigger_fallback(creation, error):
    s = creation
    s.error = error
    with pytest.raises(RuntimeError) as caught:
        await open_store(s)
    assert caught.value.__cause__ is error
    s.sql_create.assert_not_called()
    assert (
        sum(
            sql.startswith("SELECT id, text, metadata, embedding FROM ")
            for sql, _ in s.a.log
        )
        == 1
    )


@pytest.mark.parametrize(
    "columns",
    [
        _vecdb.CURRENT_COLUMNS,
        {"ID", "TEXT", "METADATA", "EMBEDDING"},
        {"UNRELATED"},
    ],
)
@pytest.mark.asyncio
async def test_partial_or_concurrent_table_is_preserved(creation, columns):
    s = creation
    s.error = database_error()
    s.partial_columns = columns
    with pytest.raises(RuntimeError) as caught:
        await open_store(s)
    assert caught.value.__cause__ is s.error
    assert s.a.columns == columns
    s.sql_create.assert_not_called()
    assert not any("DROP" in sql.upper() for sql, _ in s.a.log)


@pytest.mark.asyncio
async def test_absence_probe_failure_preserves_creation_error(creation):
    s = creation
    s.error = database_error()
    s.probe_error = database_error(3113, "ORA-03113: connection lost")
    with pytest.raises(RuntimeError) as caught:
        await open_store(s)
    assert caught.value.__cause__ is s.error
    s.sql_create.assert_not_called()


@pytest.mark.asyncio
async def test_absence_probe_interruption_never_creates_sql_table(creation):
    s = creation
    s.error = database_error()
    interruption = asyncio.CancelledError() if s.asynchronous else KeyboardInterrupt()
    s.probe_error = interruption
    with pytest.raises(type(interruption)) as caught:
        await open_store(s)
    assert caught.value is interruption
    s.sql_create.assert_not_called()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_atomic_sql_creation_never_adopts_a_concurrent_table(
    monkeypatch, asynchronous
):
    connection = MagicMock()
    exists = AsyncMock() if asynchronous else MagicMock()
    backfill = AsyncMock() if asynchronous else MagicMock()
    collision = database_error(955, "ORA-00955: name is already used")
    create = (
        AsyncMock(side_effect=collision)
        if asynchronous
        else MagicMock(side_effect=collision)
    )
    monkeypatch.setattr(
        oraclevs, "_atable_exists" if asynchronous else "_table_exists", exists
    )
    monkeypatch.setattr(
        oraclevs,
        "abackfill_table_annotation" if asynchronous else "backfill_table_annotation",
        backfill,
    )
    monkeypatch.setattr(
        oraclevs,
        "acreate_table_with_annotations"
        if asynchronous
        else "create_table_with_annotations",
        create,
    )
    with pytest.raises(oracledb.DatabaseError) as caught:
        if asynchronous:
            await oraclevs._acreate_table(connection, '"DOCS"', 3, new_table_only=True)
        else:
            oraclevs._create_table(connection, '"DOCS"', 3, new_table_only=True)
    assert caught.value is collision
    create.assert_called_once()
    exists.assert_not_called()
    backfill.assert_not_called()
