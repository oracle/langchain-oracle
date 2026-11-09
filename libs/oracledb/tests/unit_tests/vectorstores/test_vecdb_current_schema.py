# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""Catalog lookups follow the server's CURRENT_SCHEMA, not the login user."""

from contextlib import asynccontextmanager, contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import oracledb
import pytest

from langchain_oracledb.vectorstores import _vecdb
from langchain_oracledb.vectorstores._vecdb import (
    CURRENT_COLUMNS,
    AsyncNativeTransport,
    NativeTransport,
    VecDBBackend,
    anative_selection,
    native_selection,
)
from langchain_oracledb.vectorstores.utils import DistanceStrategy

SQL_COLUMNS = {"ID", "TEXT", "METADATA", "EMBEDDING"}
OTHER_COLUMNS = {"ID", "PAYLOAD"}
GUIDE = (
    "SIMPLIFIED_JSON dataguide on change "
    "sys.dbms_vector_database.handle_mvi_index search_on none"
)


@pytest.fixture(params=[False, True], ids=["sync", "async"])
def schemas(request, monkeypatch):
    asynchronous = request.param
    # The driver property may be stale after ALTER SESSION executed as SQL.
    conn = MagicMock(
        username="LOGIN",
        current_schema="LOGIN",
        version="26.1.0.24.0",
        transaction_in_progress=False,
    )
    cursor = conn.cursor.return_value.__enter__.return_value
    state = {
        "schema": "ACTIVE",
        "tables": {"LOGIN": set(OTHER_COLUMNS), "ACTIVE": set(CURRENT_COLUMNS)},
        "indexes": {
            "LOGIN": [("LOGIN_IDX", GUIDE, "LOGIN")],
            "ACTIVE": [("ACTIVE_IDX", GUIDE, "ACTIVE")],
        },
        "collision": False,
        "statement": "",
        "binds": {},
    }

    def owner():
        if state["statement"].find("user_") >= 0:
            return "LOGIN"
        return state["binds"].get("owner", state["schema"])

    def execute(statement, **binds):
        state.update(statement=statement, binds=binds)
        if statement.startswith("SELECT *"):
            columns = state["tables"].get(state["schema"], set())
            if not columns:
                raise oracledb.DatabaseError(SimpleNamespace(code=942))
            cursor.description = [(c,) for c in columns]
        elif statement.startswith("ALTER TABLE"):
            columns = state["tables"][state["schema"]]
            if "RENAME COLUMN" in statement:
                columns.remove("METADATA")
                columns.add("CONTENT_METADATA")
            else:
                columns.add(statement.split(" ADD ")[1].split()[0])

    def fetchall():
        if (
            "SELECT object_name, procedure_name FROM all_procedures"
            in state["statement"]
        ):
            return [
                ("DBMS_VECTOR_DATABASE", "HANDLE_MVI_INDEX"),
                ("DBMS_HYBRID_VECTOR", "SEARCH"),
            ]
        if "column_name FROM" in state["statement"]:
            return [(c,) for c in state["tables"].get(owner(), set())]
        if "i.parameters" in state["statement"]:
            return state["indexes"].get(owner(), [])
        raise AssertionError(state["statement"])

    def fetchone():
        if state["statement"] == _vecdb.VECDB_CAPABILITY_SQL:
            return 1, "23.6.0.0.0"
        if "SYS.DBMS_VECTOR_DATABASE.DESCRIBE_VECTOR_TABLE" in state["statement"]:
            parts = _vecdb._identifier_parts(state["binds"]["name"])
            schema = parts[0] if len(parts) == 2 else state["schema"]
            return (
                {"owner": schema, "table_name": parts[-1]}
                if "DENSE_VECTOR" in state["tables"].get(schema, set())
                else None,
                state["schema"],
            )
        if "all_procedures" in state["statement"]:
            return (1,)
        if "all_synonyms" in state["statement"]:
            return None
        if "indexes" in state["statement"]:
            return ("REQUESTED",) if owner() == "LOGIN" or state["collision"] else None
        raise AssertionError(state["statement"])

    cursor.execute.side_effect = execute
    cursor.fetchall.side_effect = fetchall
    cursor.fetchone.side_effect = fetchone
    cursor.var.side_effect = lambda *a, **kw: MagicMock(
        getvalue=MagicMock(return_value="23.6.0.0.0")
    )
    if asynchronous:
        cursor.execute = AsyncMock(side_effect=execute)
        cursor.fetchall = AsyncMock(side_effect=fetchall)
        cursor.fetchone = AsyncMock(side_effect=fetchone)

    @contextmanager
    def get(_):
        yield conn

    @asynccontextmanager
    async def aget(_):
        yield conn

    monkeypatch.setattr(_vecdb, "_get_connection", get)
    monkeypatch.setattr(_vecdb, "_aget_connection", aget)
    transport = (AsyncNativeTransport if asynchronous else NativeTransport)(
        conn, (26, 1, 0)
    )
    return asynchronous, conn, cursor, state, transport


async def select(schemas, name="DOCS"):
    asynchronous, conn, _, _, _ = schemas
    return (
        await anative_selection(conn, name, "auto")
        if asynchronous
        else native_selection(conn, name, "auto")
    )[0]


@pytest.mark.parametrize("login_columns", [SQL_COLUMNS, CURRENT_COLUMNS, OTHER_COLUMNS])
@pytest.mark.parametrize(
    "active_columns, expected",
    [(SQL_COLUMNS, "sql"), (CURRENT_COLUMNS, "vecdb"), (OTHER_COLUMNS, None)],
)
@pytest.mark.asyncio
async def test_backend_classifies_active_schema_even_when_login_object_has_same_name(
    schemas, login_columns, active_columns, expected
):
    _, _, cursor, state, _ = schemas
    state["tables"] = {"LOGIN": set(login_columns), "ACTIVE": set(active_columns)}
    if expected is None:
        with pytest.raises(ValueError, match="Incompatible"):
            await select(schemas)
    else:
        assert await select(schemas) == expected
    query = cursor.execute.call_args_list[0]
    assert "owner = SYS_CONTEXT('USERENV', 'CURRENT_SCHEMA')" in query.args[0]
    assert query.kwargs == {"name": "DOCS"}
    assert not any(
        c.args[0].startswith("ALTER SESSION") for c in cursor.execute.call_args_list
    )


@pytest.mark.asyncio
async def test_explicit_schema_overrides_current_schema(schemas):
    schemas[3]["tables"] = {"LOGIN": SQL_COLUMNS, "ACTIVE": CURRENT_COLUMNS}
    assert await select(schemas, '"LOGIN"."DOCS"') == "sql"
    assert schemas[2].execute.call_args_list[0].kwargs == {
        "owner": "LOGIN",
        "name": "DOCS",
    }
    assert "owner = :owner" in schemas[2].execute.call_args_list[0].args[0]


@pytest.mark.asyncio
async def test_current_schema_is_evaluated_each_time(schemas):
    state = schemas[3]
    state["tables"] = {"ACTIVE": SQL_COLUMNS, "SECOND": CURRENT_COLUMNS}
    assert await select(schemas) == "sql"
    state["schema"] = "SECOND"
    assert await select(schemas) == "vecdb"
    assert schemas[1].current_schema == "LOGIN"


@pytest.mark.asyncio
async def test_prepare_finds_active_vecdb_table_without_creation(schemas):
    asynchronous, _, cursor, _, transport = schemas
    backend = VecDBBackend(transport, "DOCS", DistanceStrategy.COSINE)
    if asynchronous:
        await backend.run_async("prepare")
    else:
        backend.prepare()
    assert cursor.execute.call_count == 2
    assert "all_tab_columns" in cursor.execute.call_args_list[0].args[0]
    assert "all_indexes" in cursor.execute.call_args_list[1].args[0]


@pytest.mark.asyncio
async def test_migration_inspects_and_alters_only_active_schema(schemas):
    asynchronous, _, cursor, state, transport = schemas
    login = set(state["tables"]["LOGIN"])
    state["tables"]["ACTIVE"] = {"ID", "DENSE_VECTOR", "METADATA"}
    if asynchronous:
        await transport.amigrate("DOCS")
    else:
        transport.migrate("DOCS")
    assert state["tables"]["ACTIVE"] == CURRENT_COLUMNS
    assert state["tables"]["LOGIN"] == login
    assert all(
        "CURRENT_SCHEMA" in c.args[0]
        for c in cursor.execute.call_args_list
        if "column_name FROM" in c.args[0]
    )


@pytest.mark.asyncio
async def test_index_replacement_uses_discovered_owner_in_active_schema(schemas):
    asynchronous, _, cursor, _, transport = schemas
    if asynchronous:
        await transport.acreate_text_index("DOCS", "REQUESTED")
    else:
        transport.create_text_index("DOCS", "REQUESTED")
    statements = [c.args[0] for c in cursor.execute.call_args_list]
    assert 'DROP INDEX "ACTIVE"."ACTIVE_IDX"' in statements
    assert any(
        s.startswith('CREATE SEARCH INDEX "ACTIVE"."ACTIVE_IDX" ON "DOCS"')
        for s in statements
    )
    assert not any("LOGIN_IDX" in s for s in statements)
    assert "i.table_owner = SYS_CONTEXT('USERENV', 'CURRENT_SCHEMA')" in statements[1]


@pytest.mark.parametrize("collision", [False, True])
@pytest.mark.asyncio
async def test_new_index_collision_check_uses_current_schema(schemas, collision):
    asynchronous, _, cursor, state, transport = schemas
    state["indexes"]["ACTIVE"] = []
    state["collision"] = collision
    if collision:
        with pytest.raises(ValueError, match="already used"):
            if asynchronous:
                await transport.acreate_text_index("DOCS", "REQUESTED")
            else:
                transport.create_text_index("DOCS", "REQUESTED")
    elif asynchronous:
        await transport.acreate_text_index("DOCS", "REQUESTED")
    else:
        transport.create_text_index("DOCS", "REQUESTED")
    statements = [c.args[0] for c in cursor.execute.call_args_list]
    assert "owner = SYS_CONTEXT('USERENV', 'CURRENT_SCHEMA')" in statements[2]
    assert any(s.startswith("CREATE SEARCH INDEX") for s in statements) is not collision
