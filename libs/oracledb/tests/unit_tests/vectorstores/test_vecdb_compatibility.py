# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""Native backend capability selection includes the database COMPATIBLE value."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from langchain_oracledb.vectorstores._vecdb import (
    CURRENT_COLUMNS,
    VECDB_CAPABILITY_SQL,
    anative_selection,
    native_selection,
)


def connection(version, compatibility, columns=(), supported=True, asynchronous=False):
    conn = MagicMock(version=version)
    cursor = conn.cursor.return_value.__enter__.return_value
    cursor.fetchall.return_value = [(c,) for c in columns]
    cursor.fetchone.return_value = (int(supported),)
    statement = ""

    def execute(sql, **binds):
        nonlocal statement
        statement = sql

    def fetchone():
        if statement == VECDB_CAPABILITY_SQL:
            return int(supported), compatibility if supported else None
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
    conn, _ = connection("26.1.0.24.0", "23.5.0.0.0", asynchronous=asynchronous)
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
        ("26.1.0.24.0", "23.5.0.0.0", "sql"),
        ("26.1.0.24.0", "23.6.0.0.0", "vecdb"),
        ("23.26.3.0.0", "23.5.0.0.0", "sql"),
        ("23.26.3.0.0", "23.6.0.0.0", "vecdb"),
        ("23.26.2.0.0", "23.3.0.0.0", "sql"),
        ("23.26.2.0.0", "23.4.0.0.0", "vecdb"),
        ("23.26.1.0.0", "23.5.0.0.0", "vecdb"),
    ],
)
@pytest.mark.asyncio
async def test_auto_checks_compatible_before_selecting_new_table_backend(
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
    cursor.var.assert_not_called()


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
    with pytest.raises(ValueError, match=r"COMPATIBLE >= 23\.6\.0.*23\.5"):
        if asynchronous:
            await anative_selection(conn, "DOCS", backend)
        else:
            native_selection(conn, "DOCS", backend)


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
    cursor.var.side_effect = AssertionError("Unexpected compatibility probe")
    result = (
        await anative_selection(conn, "DOCS", backend)
        if asynchronous
        else native_selection(conn, "DOCS", backend)
    )
    assert result[0] == "sql"


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_compatibility_probe_errors_are_not_treated_as_sql_fallback(asynchronous):
    conn, cursor = connection("26.1.0.24.0", "23.6.0.0.0", asynchronous=asynchronous)

    def execute(sql, **kwargs):
        if sql == VECDB_CAPABILITY_SQL:
            raise RuntimeError("Permission denied")

    cursor.execute.side_effect = execute
    with pytest.raises(RuntimeError, match="Permission denied"):
        if asynchronous:
            await anative_selection(conn, "DOCS", "auto")
        else:
            native_selection(conn, "DOCS", "auto")


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_unknown_compatible_value_is_not_silently_accepted(asynchronous):
    conn, _ = connection("26.1.0.24.0", "unknown", asynchronous=asynchronous)
    with pytest.raises(ValueError, match="Cannot determine"):
        if asynchronous:
            await anative_selection(conn, "DOCS", "auto")
        else:
            native_selection(conn, "DOCS", "auto")


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
    with pytest.raises(ValueError, match="SQL backend cannot use a VecDB table layout"):
        if asynchronous:
            await anative_selection(conn, "DOCS", "sql")
        else:
            native_selection(conn, "DOCS", "sql")
    assert cursor.execute.call_count == 2
    assert "all_tab_columns" in cursor.execute.call_args_list[0].args[0]
    assert (
        "owner = SYS_CONTEXT('USERENV', 'CURRENT_SCHEMA')"
        in cursor.execute.call_args_list[0].args[0]
    )
    cursor.var.assert_not_called()
    conn.commit.assert_not_called()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_qualified_sql_cannot_bypass_vecdb_layout_guard(asynchronous):
    conn, cursor = connection(
        "23.26.4", "23.6.0", columns=CURRENT_COLUMNS, asynchronous=asynchronous
    )
    with pytest.raises(ValueError, match="SQL backend"):
        if asynchronous:
            await anative_selection(conn, '"OWNER"."DOCS"', "sql")
        else:
            native_selection(conn, '"OWNER"."DOCS"', "sql")
    assert "all_tab_columns" in cursor.execute.call_args_list[0].args[0]
    assert cursor.execute.call_args_list[0].kwargs == {"owner": "OWNER", "name": "DOCS"}
    assert "DESCRIBE_VECTOR_TABLE(" in cursor.execute.call_args.args[0]
