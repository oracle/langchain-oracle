# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""Backend selection leaves compatibility enforcement to the database."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from langchain_oracledb.vectorstores._vecdb import (
    CURRENT_COLUMNS,
    VECDB_CAPABILITY_SQL,
    anative_selection,
    native_selection,
)

from .._sql_probe import sql_probe


def connection(version, compatibility, columns=(), supported=True, asynchronous=False):
    conn = MagicMock(version=version)
    cursor = conn.cursor.return_value.__enter__.return_value
    cursor.fetchall.return_value = [(c,) for c in columns]
    cursor.fetchone.return_value = (int(supported),)
    statement = ""

    def execute(sql, **binds):
        nonlocal statement
        statement = sql
        if sql_probe(sql, columns):
            return

    def fetchone():
        if "all_synonyms" in statement:
            return None
        if statement == VECDB_CAPABILITY_SQL:
            return (int(supported),)
        if "SYS.DBMS_VECTOR_DATABASE.DESCRIBE_VECTOR_TABLE" in statement:
            return (
                {"owner": "OWNER", "table_name": "DOCS"}
                if "DENSE_VECTOR" in columns
                else None,
                "OWNER",
            )
        return (int(supported),)

    cursor.execute.side_effect = execute
    cursor.fetchone.side_effect = fetchone
    cursor.var.side_effect = lambda *args, **kwargs: MagicMock(
        getvalue=MagicMock(return_value=compatibility)
    )
    if asynchronous:
        cursor.execute = AsyncMock(side_effect=execute)
        cursor.fetchall = AsyncMock(return_value=[(c,) for c in columns])
        cursor.fetchone = AsyncMock(side_effect=fetchone)
    return conn, cursor


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("name", ['"हिन्दी"', "U1.TB16"])
@pytest.mark.asyncio
async def test_auto_fallback_accepts_existing_sql_identifiers(asynchronous, name):
    conn, _ = connection(
        "26.1.0.24.0",
        "23.5.0.0.0",
        columns={"ID", "TEXT", "METADATA", "EMBEDDING"},
        asynchronous=asynchronous,
    )
    result = (
        await anative_selection(conn, name, "auto")
        if asynchronous
        else native_selection(conn, name, "auto")
    )
    assert result[0] == "sql"


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "version, compatible, expected",
    [
        ("26.1.0.24.0", "23.5.0.0.0", "vecdb"),
        ("26.1.0.24.0", "23.6.0.0.0", "vecdb"),
        ("23.26.3.0.0", "23.5.0.0.0", "vecdb"),
        ("23.26.3.0.0", "23.6.0.0.0", "vecdb"),
        ("23.26.2.0.0", "23.3.0.0.0", "vecdb"),
        ("23.26.2.0.0", "23.4.0.0.0", "vecdb"),
        ("23.26.1.0.0", "23.5.0.0.0", "vecdb"),
    ],
)
@pytest.mark.asyncio
async def test_auto_selects_a_vecdb_candidate_without_reading_compatible(
    asynchronous, version, compatible, expected
):
    conn, cursor = connection(version, compatible, asynchronous=asynchronous)
    result = (
        await anative_selection(conn, "DOCS", "auto")
        if asynchronous
        else native_selection(conn, "DOCS", "auto")
    )
    assert result[0] == expected
    assert any(
        call.args[0] == VECDB_CAPABILITY_SQL for call in cursor.execute.call_args_list
    )
    assert all(
        "v$parameter" not in call.args[0].lower()
        for call in cursor.execute.call_args_list
    )
    assert (
        sum(
            call.args[0] == VECDB_CAPABILITY_SQL
            for call in cursor.execute.call_args_list
        )
        == 1
    )
    assert not any(
        call.args[0].startswith("BEGIN DBMS_UTILITY.DB_VERSION")
        for call in cursor.execute.call_args_list
    )


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "backend, columns",
    [
        ("vecdb", ()),
        ("auto", CURRENT_COLUMNS),
        ("vecdb", CURRENT_COLUMNS),
    ],
)
@pytest.mark.asyncio
async def test_explicit_vecdb_and_existing_vecdb_tables_never_fall_back_to_sql(
    asynchronous, backend, columns
):
    conn, _ = connection(
        "26.1.0.24.0", "23.5.0.0.0", columns=columns, asynchronous=asynchronous
    )
    result = (
        await anative_selection(conn, "DOCS", backend)
        if asynchronous
        else native_selection(conn, "DOCS", backend)
    )
    assert result[0] == "vecdb"


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "backend, columns, supported",
    [
        ("sql", (), True),
        ("auto", {"ID", "EMBEDDING", "METADATA", "TEXT"}, True),
        ("auto", (), False),
    ],
)
@pytest.mark.asyncio
async def test_sql_tables_and_unavailable_package_do_not_probe_compatible(
    asynchronous, backend, columns, supported
):
    conn, cursor = connection(
        "26.1.0.24.0",
        "invalid",
        columns=columns,
        supported=supported,
        asynchronous=asynchronous,
    )
    result = (
        await anative_selection(conn, "DOCS", backend)
        if asynchronous
        else native_selection(conn, "DOCS", backend)
    )
    assert result[0] == "sql"


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_package_capability_errors_are_not_treated_as_sql_fallback(asynchronous):
    conn, cursor = connection("26.1.0.24.0", "23.6.0.0.0", asynchronous=asynchronous)

    original = cursor.execute.side_effect

    def execute(sql, **kwargs):
        if sql == VECDB_CAPABILITY_SQL:
            raise RuntimeError("Permission denied")
        return original(sql, **kwargs)

    cursor.execute.side_effect = execute
    with pytest.raises(RuntimeError, match="Permission denied"):
        if asynchronous:
            await anative_selection(conn, "DOCS", "auto")
        else:
            native_selection(conn, "DOCS", "auto")


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_selection_does_not_depend_on_a_reported_compatible_value(asynchronous):
    conn, cursor = connection("26.1.0.24.0", "unknown", asynchronous=asynchronous)
    result = (
        await anative_selection(conn, "DOCS", "auto")
        if asynchronous
        else native_selection(conn, "DOCS", "auto")
    )
    assert result[0] == "vecdb"
    assert not any(
        "DB_VERSION" in call.args[0] or "COMPATIBLE" in call.args[0]
        for call in cursor.execute.call_args_list
    )


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "columns", [CURRENT_COLUMNS, {"ID", "DENSE_VECTOR", "METADATA"}]
)
@pytest.mark.asyncio
async def test_explicit_sql_rejects_package_recognized_vecdb_before_mutation(
    asynchronous, columns
):
    conn, cursor = connection(
        "23.26.4", "23.6.0", columns=columns, asynchronous=asynchronous
    )
    with pytest.raises(ValueError, match="Incompatible OracleVS table layout"):
        if asynchronous:
            await anative_selection(conn, "DOCS", "sql")
        else:
            native_selection(conn, "DOCS", "sql")
    assert cursor.execute.call_count == 1
    assert (
        cursor.execute.call_args_list[0]
        .args[0]
        .startswith("SELECT id, text, metadata, embedding FROM ")
    )
    assert not any(" FROM all_" in c.args[0] for c in cursor.execute.call_args_list)
    assert not any(
        "DBMS_VECTOR_DATABASE" in c.args[0] for c in cursor.execute.call_args_list
    )
    conn.commit.assert_not_called()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_qualified_sql_cannot_bypass_vecdb_layout_guard(asynchronous):
    conn, cursor = connection(
        "23.26.4", "23.6.0", columns=CURRENT_COLUMNS, asynchronous=asynchronous
    )
    with pytest.raises(ValueError, match="Incompatible OracleVS table layout"):
        if asynchronous:
            await anative_selection(conn, '"OWNER"."DOCS"', "sql")
        else:
            native_selection(conn, '"OWNER"."DOCS"', "sql")
    assert (
        cursor.execute.call_args_list[0]
        .args[0]
        .startswith("SELECT id, text, metadata, embedding FROM ")
    )
    assert cursor.execute.call_args_list[0].kwargs == {}
    assert not any(
        "DESCRIBE_VECTOR_TABLE(" in c.args[0] for c in cursor.execute.call_args_list
    )
