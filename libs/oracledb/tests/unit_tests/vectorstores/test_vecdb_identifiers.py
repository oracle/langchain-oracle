# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""Native identifiers retain quoted component boundaries across every layer."""

import json
from contextlib import asynccontextmanager, contextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest

from langchain_oracledb.vectorstores import _vecdb
from langchain_oracledb.vectorstores._vecdb import (
    CURRENT_COLUMNS,
    AsyncNativeTransport,
    NativeTransport,
    VecDBBackend,
)
from langchain_oracledb.vectorstores.utils import DistanceStrategy

NAMES = [
    ('"literal.dot"', '"literal.dot"', {"name": "literal.dot"}),
    ('"OWNER"."DOCS"', '"OWNER"."DOCS"', {"owner": "OWNER", "name": "DOCS"}),
    (
        '"owner.dot"."literal.dot"',
        '"owner.dot"."literal.dot"',
        {"owner": "owner.dot", "name": "literal.dot"},
    ),
]
GUIDE = (
    "SIMPLIFIED_JSON dataguide on change "
    "sys.dbms_vector_database.handle_mvi_index search_on none"
)


@pytest.fixture(params=[False, True], ids=["sync", "async"])
def native(request, monkeypatch):
    asynchronous = request.param
    conn = MagicMock(version="23.26.4", transaction_in_progress=False)
    cursor = conn.cursor.return_value.__enter__.return_value
    state = {"columns": set(CURRENT_COLUMNS), "rows": [], "one": None}

    def execute(statement, **binds):
        if "SELECT object_name, procedure_name FROM all_procedures" in statement:
            state["rows"] = [
                ("DBMS_VECTOR_DATABASE", "HANDLE_MVI_INDEX"),
                ("DBMS_HYBRID_VECTOR", "SEARCH"),
            ]
        elif "DBMS_UTILITY.DB_VERSION" in statement:
            binds["compatibility"].getvalue.return_value = "23.6.0"
        elif "column_name FROM" in statement:
            state["rows"] = [(column,) for column in state["columns"]]
        elif statement.startswith("ALTER TABLE"):
            if "RENAME COLUMN" in statement:
                state["columns"].remove("METADATA")
                state["columns"].add("CONTENT_METADATA")
            else:
                state["columns"].add(statement.split(" ADD ")[1].split()[0])
        elif statement.startswith("SELECT"):
            state["rows"] = []

    cursor.execute.side_effect = execute
    cursor.fetchall.side_effect = lambda: state["rows"]
    cursor.fetchone.side_effect = lambda: state["one"]
    cursor.getbatcherrors.return_value = []
    cursor.var.return_value.getvalue.return_value = '{"items": []}'
    if asynchronous:
        cursor.execute = AsyncMock(side_effect=execute)
        cursor.fetchall = AsyncMock(side_effect=lambda: state["rows"])
        cursor.fetchone = AsyncMock(side_effect=lambda: state["one"])
        cursor.executemany = AsyncMock()
        conn.commit = AsyncMock()

    @contextmanager
    def get(_):
        yield conn

    @asynccontextmanager
    async def aget(_):
        yield conn

    monkeypatch.setattr(_vecdb, "_get_connection", get)
    monkeypatch.setattr(_vecdb, "_aget_connection", aget)
    transport = (AsyncNativeTransport if asynchronous else NativeTransport)(
        conn, (23, 26, 4)
    )
    return asynchronous, transport, cursor, state


async def run(native, name, method, *args):
    asynchronous, transport, _, _ = native
    backend = VecDBBackend(transport, name, DistanceStrategy.COSINE)
    if asynchronous:
        return await backend.run_async(method, *args)
    return getattr(backend, method)(*args)


def assert_lookup(cursor, binds):
    lookups = [
        c for c in cursor.execute.call_args_list if "column_name FROM" in c.args[0]
    ]
    assert lookups
    for call in lookups:
        assert call.kwargs == binds
        assert "all_tab_columns" in call.args[0]
        owner = (
            ":owner" if "owner" in binds else "SYS_CONTEXT('USERENV', 'CURRENT_SCHEMA')"
        )
        assert f"owner = {owner}" in call.args[0]


@pytest.mark.parametrize("name, quoted, binds", NAMES)
@pytest.mark.asyncio
async def test_existing_prepare_inspects_exact_object_without_creating(
    native, name, quoted, binds
):
    await run(native, name, "prepare")
    cursor = native[2]
    assert_lookup(cursor, binds)
    assert len(cursor.execute.call_args_list) == 2


@pytest.mark.parametrize("name, quoted, binds", NAMES)
@pytest.mark.asyncio
async def test_insert_only_never_splits_a_literal_dot(native, name, quoted, binds):
    result = await run(native, name, "add", ["seed"], [[1.0]], [{}], ["id"], False)
    assert result == ["id"]
    cursor = native[2]
    assert_lookup(cursor, binds)
    assert cursor.executemany.call_args.args[0].startswith(
        f"INSERT INTO {quoted} (ID, DENSE_VECTOR, CONTENT_METADATA)"
    )


@pytest.mark.parametrize("name, quoted, binds", NAMES)
@pytest.mark.asyncio
async def test_text_search_keeps_sql_identifier(native, name, quoted, binds):
    await run(native, name, "text_search", "oracle", 3)
    cursor = native[2]
    assert_lookup(cursor, binds)
    assert f"FROM {quoted} WHERE JSON_TEXTCONTAINS" in cursor.execute.call_args.args[0]


@pytest.mark.parametrize("name, quoted, binds", NAMES)
@pytest.mark.asyncio
async def test_migration_reinspects_same_owner_and_table(native, name, quoted, binds):
    asynchronous, transport, cursor, state = native
    state["columns"] = {"ID", "DENSE_VECTOR", "METADATA"}
    if asynchronous:
        await transport.amigrate(quoted)
    else:
        transport.migrate(quoted)
    assert state["columns"] == CURRENT_COLUMNS
    assert_lookup(cursor, binds)
    ddl = [
        c.args[0]
        for c in cursor.execute.call_args_list
        if c.args[0].startswith("ALTER")
    ]
    assert len(ddl) == 4
    assert all(s.startswith(f"ALTER TABLE {quoted} ") for s in ddl)


@pytest.mark.asyncio
async def test_package_calls_preserve_native_quoted_name(native):
    name = '"LITERAL.DOT"'
    for method, args in [
        ("add", (["seed"], [[1.0]], [{}], ["id"], True)),
        ("get", (["id"],)),
        ("delete", (["id"],)),
        ("search", ([1.0], 3, None)),
        ("create_index", ({},)),
    ]:
        await run(native, name, method, *args)
        assert native[2].execute.call_args.kwargs["table_name"] == '"LITERAL.DOT"'


@pytest.mark.parametrize(
    "name, message",
    [
        ('"literal.dot"', "normalize table names to uppercase"),
        ('"OWNER"."DOCS"', "do not support schema-qualified table names"),
    ],
)
@pytest.mark.parametrize(
    "operation",
    [
        "create_vector_table",
        "upsert_vectors",
        "list_vectors",
        "delete_vectors",
        "search",
        "create_index",
    ],
)
@pytest.mark.asyncio
async def test_unrepresentable_package_names_fail_before_database_access(
    native, name, message, operation
):
    asynchronous, transport, cursor, _ = native
    key = "name" if operation == "create_vector_table" else "table_name"
    with pytest.raises(ValueError, match=message) as error:
        if asynchronous:
            await transport.acall(operation, **{key: name})
        else:
            transport.call(operation, **{key: name})
    assert repr(name) in str(error.value)
    assert "cannot preserve" in str(error.value)
    assert "underlying VecDB package" in str(error.value)
    cursor.execute.assert_not_called()


@pytest.mark.parametrize("name, quoted, binds", NAMES)
@pytest.mark.asyncio
async def test_metadata_index_discovery_reuses_exact_existing_index(
    native, name, quoted, binds
):
    asynchronous, transport, cursor, state = native
    existing = ("actual.index", "SIMPLIFIED_JSON_HYBRID")
    if "owner" in binds:
        existing += (binds["owner"],)
    original = cursor.execute.side_effect

    def execute(statement, **params):
        original(statement, **params)
        if "i.parameters" in statement:
            state["rows"] = [existing]

    cursor.execute.side_effect = execute
    if asynchronous:
        await transport.acreate_text_index(quoted, '"requested.index"')
    else:
        transport.create_text_index(quoted, '"requested.index"')
    assert_lookup(cursor, binds)
    lookup = cursor.execute.call_args
    assert lookup.kwargs == {**binds, "metadata_column": "CONTENT_METADATA"}
    assert "all_indexes" in lookup.args[0]
    owner = ":owner" if "owner" in binds else "SYS_CONTEXT('USERENV', 'CURRENT_SCHEMA')"
    assert f"i.table_owner = {owner}" in lookup.args[0]
    assert not any(
        c.args[0].startswith(("DROP", "CREATE")) for c in cursor.execute.call_args_list
    )


@pytest.mark.asyncio
async def test_qualified_index_replacement_and_recovery_keep_catalog_owner(native):
    asynchronous, transport, cursor, state = native
    name = '"table.owner"."literal.dot"'
    existing = ("actual.index", GUIDE, "index.owner")
    original = cursor.execute.side_effect
    failure = RuntimeError("replacement failed")
    failed = False

    def execute(statement, **params):
        nonlocal failed
        original(statement, **params)
        if "i.parameters" in statement:
            state["rows"] = [existing]
        elif "FROM all_indexes" in statement:
            state["one"] = ("actual.index",)
        elif statement.startswith("CREATE SEARCH") and not failed:
            failed = True
            raise failure

    cursor.execute.side_effect = execute
    with pytest.raises(RuntimeError) as caught:
        if asynchronous:
            await transport.acreate_text_index(name, '"requested.index"')
        else:
            transport.create_text_index(name, '"requested.index"')
    assert caught.value is failure
    statements = [c.args[0] for c in cursor.execute.call_args_list]
    assert 'DROP INDEX "index.owner"."actual.index"' in statements
    assert 'DROP INDEX "index.owner"."actual.index" FORCE' in statements
    assert statements[-1].startswith(
        'CREATE SEARCH INDEX "index.owner"."actual.index" '
        'ON "table.owner"."literal.dot"'
    )
    recovery = [
        c
        for c in cursor.execute.call_args_list
        if "FROM all_indexes WHERE" in c.args[0]
    ]
    assert recovery[0].kwargs == {"owner": "index.owner", "name": "actual.index"}


@pytest.mark.asyncio
async def test_new_index_inherits_table_owner_and_checks_that_schema(native):
    asynchronous, transport, cursor, state = native
    original = cursor.execute.side_effect

    def execute(statement, **params):
        original(statement, **params)
        if "i.parameters" in statement:
            state["rows"] = []

    cursor.execute.side_effect = execute
    if asynchronous:
        await transport.acreate_text_index('"OWNER"."DOCS"', '"new.index"')
    else:
        transport.create_text_index('"OWNER"."DOCS"', '"new.index"')
    statements = cursor.execute.call_args_list
    collision = next(
        call
        for call in statements
        if "SELECT index_name FROM all_indexes" in call.args[0]
    )
    assert collision.kwargs == {"owner": "OWNER", "name": "new.index"}
    assert "FROM all_indexes" in collision.args[0]
    assert (
        statements[-1]
        .args[0]
        .startswith('CREATE SEARCH INDEX "OWNER"."new.index" ON "OWNER"."DOCS"')
    )


@pytest.mark.parametrize("name, quoted, binds", NAMES)
@pytest.mark.asyncio
async def test_hybrid_search_keeps_table_but_does_not_add_index_owner(
    native, name, quoted, binds
):
    asynchronous, transport, cursor, state = native
    existing = ("actual.index", "SIMPLIFIED_JSON_HYBRID")
    if "owner" in binds:
        existing += (binds["owner"],)
    state["one"] = existing
    original = cursor.execute.side_effect

    def execute(statement, **params):
        original(statement, **params)
        if "DBMS_HYBRID_VECTOR.SEARCH" in statement:
            state["rows"] = []

    cursor.execute.side_effect = execute
    await run(
        native, name, "hybrid_search", {"vector": {"search_vector": [1.0]}}, False
    )
    assert_lookup(cursor, binds)
    calls = cursor.execute.call_args_list
    assert calls[-2].kwargs == {**binds, "metadata_column": "CONTENT_METADATA"}
    assert f"JOIN {quoted} t" in calls[-1].args[0]
    index = json.loads(calls[-1].kwargs["search_params"])["hybrid_index_name"]
    assert index == '"actual.index"'
