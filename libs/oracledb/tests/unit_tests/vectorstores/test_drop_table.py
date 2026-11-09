# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""Table cleanup follows the backend and never falls back after a package error."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import oracledb
import pytest

from langchain_oracledb.vectorstores._vecdb import (
    AsyncNativeTransport,
    NativeTransport,
    _identifier_parts,
)
from langchain_oracledb.vectorstores.utils import adrop_table_purge, drop_table_purge


@pytest.fixture(params=[False, True], ids=["sync", "async"])
def native(request):
    asynchronous = request.param
    conn = MagicMock(
        spec=oracledb.AsyncConnection if asynchronous else oracledb.Connection,
        version="23.26.4",
        transaction_in_progress=False,
    )
    cursor = conn.cursor.return_value.__enter__.return_value
    cursor.vecdb_registered = True
    cursor.registration_name = "DOCS"
    cursor.registration_owner = "OWNER"
    statement, values = "", {}

    def execute(sql, **binds):
        nonlocal statement, values
        statement, values = sql, binds

    def fetchone():
        if "LIST_VECTOR_TABLES()" in statement:
            entries = (
                [
                    {
                        "owner": cursor.registration_owner,
                        "table_name": cursor.registration_name,
                    }
                ]
                if cursor.vecdb_registered
                else []
            )
            return ({"vector_tables": entries}, "OWNER")
        return (1,)

    cursor.execute.side_effect = execute
    cursor.fetchone.side_effect = fetchone
    if asynchronous:
        cursor.execute = AsyncMock(side_effect=execute)
        cursor.fetchone = AsyncMock(side_effect=fetchone)
    return asynchronous, conn, cursor


async def drop(native, name="DOCS", **kwargs):
    asynchronous, conn, _ = native
    if asynchronous:
        await adrop_table_purge(conn, name, **kwargs)
    else:
        drop_table_purge(conn, name, **kwargs)


@pytest.mark.parametrize("metadata", ["METADATA", "CONTENT_METADATA"])
@pytest.mark.asyncio
async def test_vecdb_drop_uses_package_on_same_checkout(native, metadata):
    asynchronous, conn, cursor = native
    columns = {"ID", "DENSE_VECTOR", metadata}
    transport = AsyncNativeTransport if asynchronous else NativeTransport
    with (
        patch.object(
            transport, "acolumns" if asynchronous else "columns", return_value=columns
        ),
        patch.object(
            transport,
            "acall" if asynchronous else "call",
            autospec=True,
            return_value={},
        ) as call,
    ):
        await drop(native)
    assert call.call_args.args[0].client is conn
    assert call.call_args.args[1] == "drop_vector_table"
    assert call.call_args.kwargs == {"name": '"DOCS"'}
    assert not any(c.args[0].startswith("DROP") for c in cursor.execute.call_args_list)


@pytest.mark.parametrize(
    "columns,backend,expected",
    [
        ({"ID", "TEXT", "METADATA", "EMBEDDING"}, "auto", True),
        ({"PAYLOAD"}, "auto", True),
        ({"ID", "DENSE_VECTOR", "METADATA"}, "sql", True),
        (set(), "auto", False),
    ],
)
@pytest.mark.asyncio
async def test_sql_and_missing_tables(native, columns, backend, expected):
    asynchronous, _, cursor = native
    cursor.vecdb_registered = False
    transport = AsyncNativeTransport if asynchronous else NativeTransport
    with patch.object(
        transport, "acolumns" if asynchronous else "columns", return_value=columns
    ):
        await drop(native, backend=backend)
    if expected:
        assert cursor.execute.call_args.args == ('DROP TABLE "DOCS" PURGE',)
    else:
        cursor.execute.assert_not_called()


@pytest.mark.asyncio
async def test_package_error_never_becomes_raw_drop(native):
    asynchronous, _, cursor = native
    transport = AsyncNativeTransport if asynchronous else NativeTransport
    with (
        patch.object(
            transport,
            "acolumns" if asynchronous else "columns",
            return_value={"ID", "DENSE_VECTOR", "METADATA"},
        ),
        patch.object(
            transport,
            "acall" if asynchronous else "call",
            side_effect=RuntimeError("package failed"),
        ),
        pytest.raises(RuntimeError, match="package failed"),
    ):
        await drop(native)
    assert not any(c.args[0].startswith("DROP") for c in cursor.execute.call_args_list)


@pytest.mark.parametrize("name", ["SCHEMA.DOCS", '"Mixed"'])
@pytest.mark.asyncio
async def test_unsupported_package_name_never_becomes_raw_drop(native, name):
    asynchronous, _, cursor = native
    parts = _identifier_parts(name)
    cursor.registration_name = parts[-1]
    cursor.registration_owner = parts[0] if len(parts) == 2 else "OWNER"
    transport = AsyncNativeTransport if asynchronous else NativeTransport
    with (
        patch.object(
            transport,
            "acolumns" if asynchronous else "columns",
            return_value={"ID", "DENSE_VECTOR", "METADATA"},
        ),
        pytest.raises(ValueError, match="cannot preserve"),
    ):
        await drop(native, name)
    assert not any(c.args[0].startswith("DROP") for c in cursor.execute.call_args_list)


@pytest.mark.asyncio
async def test_pending_work_is_not_committed_by_vecdb_drop(native):
    asynchronous, conn, cursor = native
    conn.transaction_in_progress = True
    transport = AsyncNativeTransport if asynchronous else NativeTransport
    with (
        patch.object(
            transport,
            "acolumns" if asynchronous else "columns",
            return_value={"ID", "DENSE_VECTOR", "METADATA"},
        ),
        pytest.raises(RuntimeError, match="no pending work"),
    ):
        await drop(native)
    assert not any(c.args[0].startswith("DROP") for c in cursor.execute.call_args_list)
    conn.commit.assert_not_called()


@pytest.mark.parametrize("backend", ["vecdb", "invalid"])
@pytest.mark.asyncio
async def test_invalid_backend_or_layout_rejected(native, backend):
    asynchronous, _, cursor = native
    cursor.vecdb_registered = False
    transport = AsyncNativeTransport if asynchronous else NativeTransport
    with (
        patch.object(
            transport,
            "acolumns" if asynchronous else "columns",
            return_value={"PAYLOAD"},
        ),
        pytest.raises(ValueError),
    ):
        await drop(native, backend=backend)
    assert not any(c.args[0].startswith("DROP") for c in cursor.execute.call_args_list)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("status", [None, 404, 403])
@pytest.mark.asyncio
async def test_rest_uses_public_sdk_drop(asynchronous, status):
    cls = type("OracleVecDB", (), {"__module__": "oracle_vecdb.client"})
    client = cls()
    client.config = SimpleNamespace(has_rest_connection=True)
    client.drop_vector_table = MagicMock(return_value={})
    if status:
        error = RuntimeError("REST failed")
        error.status = status
        client.drop_vector_table.side_effect = error

    async def invoke():
        if asynchronous:
            await adrop_table_purge(client, "DOCS")
        else:
            drop_table_purge(client, "DOCS")

    if status == 403:
        with pytest.raises(RuntimeError, match="REST failed"):
            await invoke()
    else:
        await invoke()
    client.drop_vector_table.assert_called_once_with(name="DOCS")


@pytest.mark.asyncio
async def test_pool_drop_retains_one_checkout(native):
    asynchronous, conn, _ = native
    pool = MagicMock(
        spec=oracledb.AsyncConnectionPool if asynchronous else oracledb.ConnectionPool
    )
    pool.acquire.return_value.__enter__.return_value = conn
    pool.acquire.return_value.__aenter__.return_value = conn
    transport = AsyncNativeTransport if asynchronous else NativeTransport
    with (
        patch.object(
            transport,
            "acolumns" if asynchronous else "columns",
            return_value={"ID", "DENSE_VECTOR", "METADATA"},
        ),
        patch.object(
            transport,
            "acall" if asynchronous else "call",
            autospec=True,
            return_value={},
        ) as call,
    ):
        if asynchronous:
            await adrop_table_purge(pool, "DOCS")
        else:
            drop_table_purge(pool, "DOCS")
    pool.acquire.assert_called_once_with()
    assert call.call_args.args[0].client is conn
