# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""Ownership comes from the package, independently of column layout."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import oracledb
import pytest

from langchain_oracledb.vectorstores import _vecdb
from langchain_oracledb.vectorstores.utils import adrop_table_purge, drop_table_purge


@pytest.fixture(params=[False, True], ids=["sync", "async"])
def database(request):
    asynchronous = request.param
    connection = MagicMock(
        spec=oracledb.AsyncConnection if asynchronous else oracledb.Connection,
        version="23.26.4",
        transaction_in_progress=False,
    )
    cursor = connection.cursor.return_value.__enter__.return_value
    state = {
        "available": True,
        "payload": {"vector_tables": []},
        "schema": "OWNER",
        "columns": set(_vecdb.CURRENT_COLUMNS),
        "statement": "",
        "error": None,
    }

    def execute(statement, **binds):
        state["statement"] = statement
        if "SELECT SYS.DBMS_VECTOR_DATABASE." in statement and state["error"]:
            raise state["error"]

    def fetchone():
        if state["statement"] == _vecdb.VECDB_RECOGNITION_SQL:
            return (int(state["available"]),)
        if "SELECT SYS.DBMS_VECTOR_DATABASE." in state["statement"]:
            return state["payload"], state["schema"]
        return (1,)

    cursor.execute.side_effect = execute
    cursor.fetchone.side_effect = fetchone
    cursor.fetchall.side_effect = lambda: [(name,) for name in state["columns"]]
    if asynchronous:
        cursor.execute = AsyncMock(side_effect=execute)
        cursor.fetchone = AsyncMock(side_effect=fetchone)
        cursor.fetchall = AsyncMock(side_effect=cursor.fetchall.side_effect)
    return asynchronous, connection, cursor, state


async def recognize(database, name="DOCS"):
    asynchronous, _, cursor, _ = database
    return (
        await _vecdb._ais_vecdb_table(cursor, name)
        if asynchronous
        else _vecdb._is_vecdb_table(cursor, name)
    )


@pytest.mark.parametrize("registered", [False, True])
@pytest.mark.asyncio
async def test_registry_membership_identifies_package_tables(registered, database):
    _, _, cursor, state = database
    state["payload"] = json.dumps(
        {
            "vector_tables": [{"owner": "OWNER", "table_name": "DOCS"}]
            if registered
            else []
        }
    )
    assert await recognize(database) is registered
    assert cursor.execute.call_args.args == (_vecdb.VECDB_TABLES_SQL,)
    assert cursor.execute.call_count == 2


@pytest.mark.asyncio
async def test_absent_package_does_not_execute_package_code(database):
    _, _, cursor, state = database
    state["available"] = False
    assert not await recognize(database)
    cursor.execute.assert_called_once_with(_vecdb.VECDB_RECOGNITION_SQL)


@pytest.mark.asyncio
async def test_existing_sql_store_selects_sql_without_vecdb_package(database):
    asynchronous, connection, cursor, state = database
    state.update(available=False, columns={"ID", "EMBEDDING", "METADATA", "TEXT"})
    selected = (
        await _vecdb.anative_selection(connection, "DOCS", "auto")
        if asynchronous
        else _vecdb.native_selection(connection, "DOCS", "auto")
    )
    assert selected[0] == "sql"
    assert not any(
        "SELECT SYS.DBMS_VECTOR_DATABASE." in c.args[0]
        for c in cursor.execute.call_args_list
    )


@pytest.mark.parametrize(
    "entry",
    [
        {"owner": "OTHER", "table_name": "DOCS"},
        {"owner": "OWNER", "table_name": "OTHER"},
    ],
)
@pytest.mark.asyncio
async def test_unqualified_registry_matching_ignores_other_objects(database, entry):
    database[3]["payload"] = {"vector_tables": [entry]}
    assert not await recognize(database)


@pytest.mark.parametrize("code", [942, 1031, 6550, 3113])
@pytest.mark.asyncio
async def test_recognition_errors_never_become_nonmembership(database, code):
    _, _, cursor, state = database
    error = oracledb.DatabaseError(
        SimpleNamespace(code=code, message=f"ORA-{code:05d}")
    )
    state["error"] = error
    with pytest.raises(oracledb.DatabaseError) as caught:
        await recognize(database)
    assert caught.value is error
    assert cursor.execute.call_count == 2


@pytest.mark.parametrize(
    "owner,table,expected",
    [
        ("OWNER", "DOCS", True),
        ("OTHER", "DOCS", False),
        ("OWNER", "OTHER", False),
    ],
)
@pytest.mark.asyncio
async def test_qualified_registry_matching_keeps_owner_and_name(
    database, owner, table, expected
):
    _, _, cursor, state = database
    state["payload"] = {"vector_tables": [{"owner": owner, "table_name": table}]}
    assert await recognize(database, '"OWNER"."DOCS"') is expected
    assert "LIST_VECTOR_TABLES()" in cursor.execute.call_args.args[0]
    assert cursor.execute.call_args.kwargs == {}


@pytest.mark.parametrize("name", ['"Mixed"', '"LITERAL.DOT"', '"O\'BRIEN"'])
@pytest.mark.asyncio
async def test_registry_matching_preserves_exact_identifier(database, name):
    _, _, cursor, state = database
    state["payload"] = {"vector_tables": [{"owner": "OWNER", "table_name": name[1:-1]}]}
    assert await recognize(database, name)
    assert cursor.execute.call_args.kwargs == {}
    assert name not in cursor.execute.call_args.args[0]


@pytest.mark.parametrize(
    "payload", [{}, {"owner": "OWNER"}, {"table_name": "DOCS"}, []]
)
@pytest.mark.asyncio
async def test_malformed_response_does_not_authorize_sql_fallback(database, payload):
    database[3]["payload"] = payload
    with pytest.raises(ValueError):
        await recognize(database)


@pytest.mark.parametrize(
    "payload", [None, {}, {"vector_tables": None}, {"vector_tables": [{}]}]
)
@pytest.mark.asyncio
async def test_malformed_registry_does_not_authorize_sql_fallback(database, payload):
    database[3]["payload"] = payload
    with pytest.raises(ValueError):
        await recognize(database, '"OWNER"."DOCS"')


@pytest.mark.parametrize(
    "columns",
    [
        {"ID", "DENSE_VECTOR"},
        {"ID", "DENSE_VECTOR", "METADATA"},
        set(_vecdb.CURRENT_COLUMNS),
        set(_vecdb.CURRENT_COLUMNS) | {"METADATA"},
    ],
)
@pytest.mark.parametrize("available", [False, True])
@pytest.mark.asyncio
async def test_unregistered_lookalikes_use_sql_cleanup(database, columns, available):
    asynchronous, connection, cursor, state = database
    state.update(columns=columns, available=available)
    if asynchronous:
        await adrop_table_purge(connection, "DOCS")
    else:
        drop_table_purge(connection, "DOCS")
    assert cursor.execute.call_args.args == ('DROP TABLE "DOCS" PURGE',)
    assert not any(
        "DROP_VECTOR_TABLE(" in c.args[0] for c in cursor.execute.call_args_list
    )


@pytest.mark.asyncio
async def test_cleanup_recognition_failure_never_drops(database):
    asynchronous, connection, cursor, state = database
    state["error"] = oracledb.DatabaseError(
        SimpleNamespace(code=1031, message="ORA-01031")
    )
    with pytest.raises(RuntimeError, match="ORA-01031"):
        if asynchronous:
            await adrop_table_purge(connection, "DOCS")
        else:
            drop_table_purge(connection, "DOCS")
    assert not any(c.args[0].startswith("DROP") for c in cursor.execute.call_args_list)


@pytest.mark.parametrize("registered", [False, True])
@pytest.mark.asyncio
async def test_backend_selection_distinguishes_same_columns_by_registration(
    database, registered
):
    asynchronous, connection, cursor, state = database
    state["payload"] = {
        "vector_tables": [{"owner": "OWNER", "table_name": "DOCS"}]
        if registered
        else []
    }
    if registered:
        cursor.var.side_effect = lambda *a, **kw: MagicMock(getvalue=lambda: "23.6.0")
        selected = (
            await _vecdb.anative_selection(connection, "DOCS", "auto")
            if asynchronous
            else _vecdb.native_selection(connection, "DOCS", "auto")
        )
        assert selected[0] == "vecdb"
    else:
        with pytest.raises(ValueError, match="not registered"):
            if asynchronous:
                await _vecdb.anative_selection(connection, "DOCS", "auto")
            else:
                _vecdb.native_selection(connection, "DOCS", "auto")
    assert not any(
        c.args[0].startswith(("ALTER", "CREATE", "DROP"))
        for c in cursor.execute.call_args_list
    )
