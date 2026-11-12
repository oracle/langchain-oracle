# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""A package description failure must not break readable SQL synonyms."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import oracledb
import pytest

from langchain_oracledb.vectorstores import _vecdb


@pytest.fixture(params=[False, True], ids=["sync", "async"])
def synonym(request):
    asynchronous = request.param
    connection = MagicMock(version="26.1.0.24.0")
    cursor = connection.cursor.return_value.__enter__.return_value
    state = {
        "statement": "",
        "direct_columns": set(),
        "columns": {"ID", "TEXT", "METADATA", "EMBEDDING"},
        "synonym": True,
        "code": 942,
        "read_error": None,
    }

    def execute(statement, **binds):
        state["statement"] = statement
        if "DESCRIBE_VECTOR_TABLE(" in statement:
            raise oracledb.DatabaseError(SimpleNamespace(code=state["code"]))
        if statement.startswith("SELECT *"):
            if state["read_error"]:
                raise state["read_error"]
            cursor.description = [(column,) for column in state["columns"]]

    def fetchone():
        if state["statement"] == _vecdb.VECDB_RECOGNITION_SQL:
            return (1,)
        if "all_synonyms" in state["statement"]:
            return (1,) if state["synonym"] else None
        raise AssertionError(state["statement"])

    cursor.execute.side_effect = execute
    cursor.fetchone.side_effect = fetchone
    cursor.fetchall.side_effect = lambda: [(c,) for c in state["direct_columns"]]
    if asynchronous:
        cursor.execute = AsyncMock(side_effect=execute)
        cursor.fetchone = AsyncMock(side_effect=fetchone)
        cursor.fetchall = AsyncMock(side_effect=cursor.fetchall.side_effect)
    return asynchronous, connection, cursor, state


@pytest.mark.parametrize("name", ["ALIAS", '"OWNER"."ALIAS"', '"literal.dot"'])
@pytest.mark.parametrize("backend", ["auto", "sql", "vecdb"])
@pytest.mark.asyncio
async def test_readable_sql_synonym_retains_sql_backend(synonym, name, backend):
    asynchronous, connection, _, _ = synonym

    async def select():
        return (
            await _vecdb.anative_selection(connection, name, backend)
            if asynchronous
            else _vecdb.native_selection(connection, name, backend)
        )

    if backend == "vecdb":
        with pytest.raises(ValueError, match="ordinary OracleVS"):
            await select()
    else:
        assert (await select())[0] == "sql"


@pytest.mark.parametrize(
    "case", ["direct-table", "not-a-synonym", "vecdb-layout", "unreadable"]
)
@pytest.mark.asyncio
async def test_description_942_requires_a_readable_sql_synonym(synonym, case):
    asynchronous, _, cursor, state = synonym
    if case == "direct-table":
        state["direct_columns"] = state["columns"]
    elif case == "not-a-synonym":
        state["synonym"] = False
    elif case == "vecdb-layout":
        state["columns"] = set(_vecdb.CURRENT_COLUMNS)
    else:
        state["read_error"] = oracledb.DatabaseError(SimpleNamespace(code=980))
    with pytest.raises(oracledb.DatabaseError):
        if asynchronous:
            await _vecdb._ais_vecdb_table(cursor, "ALIAS")
        else:
            _vecdb._is_vecdb_table(cursor, "ALIAS")


@pytest.mark.asyncio
async def test_synonym_does_not_hide_permission_errors(synonym):
    asynchronous, _, cursor, state = synonym
    state["code"] = 1031
    with pytest.raises(oracledb.DatabaseError):
        if asynchronous:
            await _vecdb._ais_vecdb_table(cursor, "ALIAS")
        else:
            _vecdb._is_vecdb_table(cursor, "ALIAS")
    assert not any("all_synonyms" in c.args[0] for c in cursor.execute.call_args_list)
