# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""Missing-name fallback and SQL object resolution before backend selection."""

from contextlib import asynccontextmanager, contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import oracledb
import pytest
from langchain_core.embeddings import Embeddings

from langchain_oracledb import OracleVS
from langchain_oracledb.vectorstores import _vecdb, oraclevs
from langchain_oracledb.vectorstores._vecdb import (
    CURRENT_COLUMNS,
    anative_selection,
    native_selection,
)

SQL_COLUMNS = {"ID", "TEXT", "METADATA", "EMBEDDING"}


class Embedding(Embeddings):
    def embed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0]


def database_error(code):
    return oracledb.DatabaseError(SimpleNamespace(code=code, message=f"ORA-{code:05d}"))


@pytest.fixture(params=[False, True], ids=["sync", "async"])
def database(request, monkeypatch):
    asynchronous = request.param
    conn = MagicMock(version="26.1.0.24.0", thin=True)
    cursor = conn.cursor.return_value.__enter__.return_value
    state = {
        "catalog": set(),
        "resolved": None,
        "synonym": False,
        "error": None,
        "last": None,
        "exists": False,
    }
    cursor.description = None
    cursor.var.side_effect = lambda *a, **kw: MagicMock(
        getvalue=MagicMock(return_value="23.6.0.0.0")
    )

    def execute(statement, **binds):
        state["last"] = statement
        if "column_name FROM" in statement:
            state["name"] = binds["name"]
        if statement.startswith("SELECT *"):
            if state["error"]:
                raise state["error"]
            if state["resolved"] is None:
                raise database_error(942)
            cursor.description = [(c,) for c in sorted(state["resolved"])]
        elif statement.startswith("SELECT 1 FROM") and "ROWNUM" in statement:
            if not state["exists"]:
                raise database_error(942)
        elif statement.startswith("CREATE TABLE"):
            state["exists"] = True

    def fetchone():
        if "SYS.DBMS_VECTOR_DATABASE.LIST_VECTOR_TABLES" in state["last"]:
            registered = "DENSE_VECTOR" in state["catalog"]
            return (
                {
                    "vector_tables": [{"owner": "OWNER", "table_name": state["name"]}]
                    if registered
                    else []
                },
                "OWNER",
            )
        if "all_synonyms" in state["last"]:
            return (1,) if state["synonym"] else None
        if "all_procedures" in state["last"]:
            return (1,)
        raise AssertionError(f"Unexpected fetchone: {state['last']}")

    cursor.execute.side_effect = execute
    cursor.fetchall.side_effect = lambda: [(c,) for c in state["catalog"]]
    cursor.fetchone.side_effect = fetchone
    cursor.getbatcherrors.return_value = []
    if asynchronous:
        cursor.execute = AsyncMock(side_effect=execute)
        cursor.fetchall = AsyncMock(
            side_effect=lambda: [(c,) for c in state["catalog"]]
        )
        cursor.fetchone = AsyncMock(side_effect=fetchone)
        cursor.executemany = AsyncMock()
        conn.commit = AsyncMock()

    @contextmanager
    def acquire(_):
        yield conn

    @asynccontextmanager
    async def aacquire(_):
        yield conn

    for module in (_vecdb, oraclevs):
        monkeypatch.setattr(module, "_get_connection", acquire)
        monkeypatch.setattr(module, "_aget_connection", aacquire)
    return asynchronous, conn, cursor, state


async def select(database, name, backend="auto"):
    asynchronous, conn, _, _ = database
    return (
        await anative_selection(conn, name, backend)
        if asynchronous
        else native_selection(conn, name, backend)
    )[0]


@pytest.mark.parametrize(
    "name", ["langchain", '"literal.dot"', '"Mixed"', '"OWNER"."DOCS"']
)
@pytest.mark.asyncio
async def test_missing_unrepresentable_names_use_sql_only_in_auto(database, name):
    assert await select(database, name) == "sql"
    with pytest.raises(ValueError, match="Native VecDB package operations"):
        await select(database, name, "vecdb")
    assert not any(
        "all_procedures" in c.args[0] for c in database[2].execute.call_args_list
    )


@pytest.mark.asyncio
async def test_missing_representable_name_still_selects_vecdb(database):
    assert await select(database, "DOCS") == "vecdb"
    assert any(
        "all_procedures" in c.args[0] for c in database[2].execute.call_args_list
    )
    assert any(
        c.args[0] == _vecdb.COMPATIBILITY_SQL
        for c in database[2].execute.call_args_list
    )


@pytest.mark.asyncio
async def test_default_factory_creates_sql_table_on_vecdb_capable_database(database):
    asynchronous, conn, cursor, state = database
    if asynchronous:
        store = await OracleVS.afrom_texts(
            ["seed"], Embedding(), client=conn, ids=["seed"]
        )
    else:
        store = OracleVS.from_texts(["seed"], Embedding(), client=conn, ids=["seed"])
    assert store.backend == "sql" and store._vecdb is None
    assert store.table_name == '"langchain"'
    ddl = [
        c.args[0]
        for c in cursor.execute.call_args_list
        if c.args[0].startswith("CREATE")
    ]
    assert len(ddl) == 1 and ddl[0].startswith('CREATE TABLE "langchain"')
    assert cursor.executemany.call_args.args[0].startswith('INSERT INTO "langchain"')
    assert state["exists"]
    assert not any(
        "DBMS_VECTOR_DATABASE" in c.args[0] for c in cursor.execute.call_args_list
    )


@pytest.mark.parametrize("name", ["ALIAS", '"alias.dot"', '"OWNER"."ALIAS"'])
@pytest.mark.asyncio
async def test_resolved_sql_layout_precedes_capability_and_name_checks(database, name):
    _, _, cursor, state = database
    state.update(resolved=SQL_COLUMNS, synonym=True, exists=True)
    assert await select(database, name) == "sql"
    assert await select(database, name, "sql") == "sql"
    with pytest.raises(ValueError, match="ordinary OracleVS"):
        await select(database, name, "vecdb")
    assert any(c.args[0].startswith("SELECT *") for c in cursor.execute.call_args_list)
    assert not any(
        "procedure_name = 'SEARCH'" in c.args[0] for c in cursor.execute.call_args_list
    )


@pytest.mark.asyncio
async def test_factory_writes_through_sql_synonym_without_creation(database):
    asynchronous, conn, cursor, state = database
    state.update(resolved=SQL_COLUMNS, synonym=True, exists=True)
    kwargs = {"client": conn, "table_name": '"alias.dot"', "ids": ["seed"]}
    if asynchronous:
        store = await OracleVS.afrom_texts(["seed"], Embedding(), **kwargs)
    else:
        store = OracleVS.from_texts(["seed"], Embedding(), **kwargs)
    assert store.backend == "sql"
    assert cursor.executemany.call_args.args[0].startswith('INSERT INTO "alias.dot"')
    assert not any(
        c.args[0].startswith("CREATE") for c in cursor.execute.call_args_list
    )


@pytest.mark.parametrize("columns", [CURRENT_COLUMNS, {"ID", "PAYLOAD"}])
@pytest.mark.asyncio
async def test_resolved_incompatible_layout_is_not_an_ordinary_sql_table(
    database, columns
):
    database[3].update(resolved=columns, synonym=True)
    with pytest.raises(ValueError, match="SQL backend|Incompatible"):
        await select(database, "ALIAS", "sql")


@pytest.mark.asyncio
async def test_existing_vecdb_table_with_unsupported_name_never_falls_back(database):
    database[3]["catalog"] = CURRENT_COLUMNS
    assert await select(database, '"Mixed"') == "vecdb"


@pytest.mark.parametrize("code", [942, 980, 1775, 1031, 2019, 12514])
@pytest.mark.asyncio
async def test_synonym_access_and_resolution_errors_never_become_missing(
    database, code
):
    _, _, cursor, state = database
    error = database_error(code)
    state.update(error=error, synonym=True)
    with pytest.raises(oracledb.DatabaseError) as caught:
        await select(database, "ALIAS")
    assert caught.value is error
    assert not any("all_procedures" in c.args[0] for c in cursor.execute.call_args_list)
    assert not any(
        c.args[0].startswith("CREATE") for c in cursor.execute.call_args_list
    )
