# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""A checkout owns native discovery, execution, retries and DDL recovery."""

import asyncio
import json
from itertools import cycle
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import oracledb
import pytest

from langchain_oracledb import OracleVS
from langchain_oracledb.embeddings import OracleEmbeddings
from langchain_oracledb.retrievers.hybrid_search import (
    acreate_hybrid_index,
    create_hybrid_index,
)
from langchain_oracledb.vectorstores._vecdb import (
    CURRENT_COLUMNS,
    AsyncNativeTransport,
    NativeTransport,
    VecDBBackend,
    _TableInfo,
)
from langchain_oracledb.vectorstores.utils import DistanceStrategy

GUIDE = (
    "SIMPLIFIED_JSON dataguide on change "
    "sys.dbms_vector_database.handle_mvi_index "
    'sync (every "freq=secondly; interval=30")'
    " search_on none"
)


class Variable:
    value = None

    def getvalue(self):
        return self.value


class Session:
    def __init__(self, schema, asynchronous):
        self.schema = schema
        self.columns = {"ID", "DENSE_VECTOR", "METADATA"}
        self.parameters = GUIDE
        self.index_exists = True
        self.index_name = "GUIDE"
        self.search_errors = []
        self.preference_exists = False
        self.fail_create = False
        self.fail_query = False
        self.log = []
        self.rows = []
        self.one = None
        self.leased = False
        self.connection = MagicMock(
            spec=oracledb.AsyncConnection if asynchronous else oracledb.Connection,
            version="23.26.4",
            transaction_in_progress=False,
            autocommit=False,
        )
        if asynchronous:
            self.connection.__aenter__.side_effect = self.enter
            self.connection.__aexit__.side_effect = self.exit
        else:
            self.connection.__enter__.return_value = self.connection
            self.connection.__exit__.side_effect = self.exit
        self.cursor = MagicMock()
        self.cursor.__enter__.return_value = self.cursor
        self.cursor.__aenter__.return_value = self.cursor
        self.connection.cursor.return_value = self.cursor
        self.cursor.var.side_effect = lambda *args, **kwargs: Variable()
        self.cursor.getbatcherrors.return_value = []
        if asynchronous:
            self.cursor.execute = AsyncMock(side_effect=self.aexecute)
            self.cursor.executemany = AsyncMock(side_effect=self.executemany)
            self.cursor.fetchall = AsyncMock(side_effect=lambda: self.rows)
            self.cursor.fetchone = AsyncMock(side_effect=lambda: self.one)
            self.connection.commit = AsyncMock()
            self.connection.rollback = AsyncMock()
        else:
            self.cursor.execute.side_effect = self.execute
            self.cursor.executemany.side_effect = self.executemany
            self.cursor.fetchall.side_effect = lambda: self.rows
            self.cursor.fetchone.side_effect = lambda: self.one

    async def enter(self):
        assert not self.leased, "pool attempted to lend an already-held session"
        self.leased = True
        return self.connection

    def exit(self, *args):
        self.leased = False

    async def aexecute(self, *args, **kwargs):
        await asyncio.sleep(0)  # Interleave simultaneous operation-local bindings.
        self.execute(*args, **kwargs)

    def execute(self, statement, parameters=None, **binds):
        self.log.append((statement, binds))
        if statement.startswith("BEGIN IF FALSE THEN"):
            return
        if "SELECT object_name, procedure_name FROM all_procedures" in statement:
            self.rows = [
                ("DBMS_VECTOR_DATABASE", "HANDLE_MVI_INDEX"),
                ("DBMS_HYBRID_VECTOR", "SEARCH"),
            ]
        elif "column_name FROM all_tab_columns" in statement:
            self.rows = [(column,) for column in self.columns]
        elif "DBMS_UTILITY.DB_VERSION" in statement:
            binds["compatibility"].value = "23.6.0"
        elif statement.startswith("ALTER TABLE"):
            if "RENAME COLUMN" in statement:
                self.columns.remove("METADATA")
                self.columns.add("CONTENT_METADATA")
            else:
                self.columns.add(statement.split(" ADD ")[1].split()[0])
        elif "i.parameters" in statement:
            self.rows = [(self.index_name, self.parameters, self.schema)]
            self.one = self.rows[0]
        elif "SELECT owner, index_name, status" in statement:
            self.rows = [(self.schema, "GUIDE", "VALID", "VALID", "VALID")]
        elif "SELECT index_name FROM all_indexes" in statement:
            self.one = ("GUIDE",) if self.index_exists else None
        elif statement.startswith("DROP INDEX"):
            self.index_exists = False
        elif statement.startswith("CREATE HYBRID VECTOR INDEX"):
            if self.fail_create:
                raise RuntimeError("build failed")
            self.index_exists = True
            self.parameters = "SIMPLIFIED_JSON_HYBRID " + GUIDE
        elif statement.startswith("CREATE SEARCH INDEX"):
            self.index_exists = True
            self.parameters = GUIDE
        elif "CREATE_PREFERENCE" in statement:
            self.preference_exists = True
        elif "DROP_PREFERENCE" in statement:
            self.preference_exists = False
        elif "JSON_TABLE(DBMS_HYBRID_VECTOR.SEARCH" in statement:
            self.check_metadata_search(statement, binds)
            self.rows = [
                (self.schema, {"__orcl_internal_content": self.schema}, 1.0, 0.5, 0.5)
            ]
        elif "VECTOR_DISTANCE(dense_vector" in statement:
            self.check_metadata_search(statement, binds)
            self.rows = [
                (self.schema, {"__orcl_internal_content": self.schema}, 0.25, [1.0])
            ]
        elif "JSON_TEXTCONTAINS" in statement:
            self.check_metadata_search(statement, binds)
            self.rows = [(self.schema, {"__orcl_internal_content": self.schema}, 17)]
        elif "result" in binds:
            if ".search(" in statement.lower() and self.fail_query:
                self.fail_query = False
                raise RuntimeError('ORA-00904: "CONTENT_METADATA": invalid identifier')
            binds["result"].value = json.dumps(
                {
                    "items": [
                        {
                            "id": self.schema,
                            "metadata": {"__orcl_internal_content": self.schema},
                            "distance": 0.0,
                        }
                    ]
                }
            )
            if "create_vector_table" in statement.lower():
                self.columns = set(CURRENT_COLUMNS)

    def check_metadata_search(self, statement, binds):
        meta = "CONTENT_METADATA" if "CONTENT_METADATA" in statement else "METADATA"
        if meta not in self.columns:
            raise RuntimeError(f'ORA-00904: "{meta}": invalid identifier')
        if self.search_errors:
            raise self.search_errors.pop(0)
        if "search_params" in binds:
            index = json.loads(binds["search_params"])["hybrid_index_name"]
            if index not in {
                f'"{self.index_name}"',
                f'"{self.schema}"."{self.index_name}"',
            }:
                raise RuntimeError(
                    "ORA-20000: Oracle Text error: DRG-10502: index does not exist"
                )

    def executemany(self, statement, values, **kwargs):
        self.log.append((statement, {}))


@pytest.fixture(params=[False, True], ids=["sync", "async"])
def sessions(request):
    asynchronous = request.param
    a, b = Session("SCHEMA_A", asynchronous), Session("SCHEMA_B", asynchronous)
    b.columns = set(CURRENT_COLUMNS)
    pool = MagicMock(
        spec=oracledb.AsyncConnectionPool if asynchronous else oracledb.ConnectionPool
    )
    pool.acquire = MagicMock(side_effect=cycle([a.connection, b.connection]))
    transport = (AsyncNativeTransport if asynchronous else NativeTransport)(
        pool, (23, 26, 4)
    )
    backend = VecDBBackend(transport, "DOCS", DistanceStrategy.COSINE)
    return SimpleNamespace(
        asynchronous=asynchronous,
        a=a,
        b=b,
        pool=pool,
        transport=transport,
        backend=backend,
    )


async def invoke(s, method, *args):
    if s.asynchronous:
        return await s.backend.run_async(method, *args)
    return getattr(s.backend, method)(*args)


def assert_one_session(s):
    assert s.pool.acquire.call_count == 1
    assert s.a.log
    assert not s.b.log
    assert s.transport.client is s.pool
    assert s.backend.transport is s.transport


@pytest.mark.parametrize(
    "operation", ["text_search", "hybrid_search", "add", "create_text_index"]
)
@pytest.mark.asyncio
async def test_native_operation_discovers_and_executes_on_one_checkout(
    sessions, operation
):
    s = sessions
    if operation == "text_search":
        result = await invoke(s, operation, "oracle", 1)
        assert result[0].page_content == "SCHEMA_A"
    elif operation == "hybrid_search":
        s.a.parameters = "SIMPLIFIED_JSON_HYBRID " + GUIDE
        result = await invoke(s, operation, {"text": {"search_text": "oracle"}}, False)
        assert result[0].page_content == "SCHEMA_A"
    elif operation == "add":
        assert await invoke(
            s, operation, ["oracle"], [[1.0]], [{}], ["one"], False
        ) == ["one"]
        assert "DENSE_VECTOR, METADATA)" in s.a.cursor.executemany.call_args.args[0]
    else:
        await invoke(s, operation, "REQUESTED")
        assert any('DROP INDEX "SCHEMA_A"."GUIDE"' == sql for sql, _ in s.a.log)
        assert any(
            "CREATE SEARCH INDEX" in sql and "(METADATA)" in sql for sql, _ in s.a.log
        )
    assert_one_session(s)


@pytest.mark.asyncio
async def test_query_layout_migration_and_retry_keep_the_failed_queries_session(
    sessions,
):
    s = sessions
    s.a.fail_query = True
    result = await invoke(s, "search", [1.0], 1, None)
    assert result[0][0].page_content == "SCHEMA_A"
    assert s.a.columns == CURRENT_COLUMNS
    assert len([sql for sql, _ in s.a.log if sql.startswith("ALTER TABLE")]) == 4
    assert (
        len(
            [
                sql
                for sql, _ in s.a.log
                if ".search(" in sql.lower()
                and not sql.startswith("BEGIN IF FALSE THEN")
            ]
        )
        == 2
    )
    assert_one_session(s)


@pytest.mark.asyncio
async def test_get_batches_do_not_switch_schema_between_batches(sessions):
    s = sessions
    result = await invoke(s, "get", [str(i) for i in range(101)])
    assert [doc.page_content for doc in result] == ["SCHEMA_A", "SCHEMA_A"]
    assert_one_session(s)


@pytest.mark.asyncio
async def test_prepare_uses_each_checkouts_version_without_changing_shared_snapshot(
    sessions,
):
    s = sessions
    s.a.connection.version = "23.26.1"
    s.a.columns = set()
    s.b.columns = set()
    await invoke(s, "prepare")
    await invoke(s, "prepare")
    assert s.pool.acquire.call_count == 2
    a_create = next(
        binds for sql, binds in s.a.log if "create_vector_table" in sql.lower()
    )
    b_create = next(
        binds for sql, binds in s.b.log if "create_vector_table" in sql.lower()
    )
    assert a_create["auto_generate_id"] is False
    assert "table_params" not in a_create
    assert json.loads(b_create["table_params"]) == {"auto_generate_id": False}
    assert s.transport.version == (23, 26, 4)
    assert s.transport.client is s.pool


@pytest.mark.asyncio
async def test_index_params_use_the_operation_session_version(sessions):
    s = sessions
    s.a.connection.version = "23.26.1"
    if s.asynchronous:
        s.transport.acall = AsyncMock(return_value={"job": "submitted"})
    else:
        s.transport.call = MagicMock(return_value={"job": "submitted"})
    assert await invoke(s, "create_index", {"idx_type": "HNSW"}) == {"job": "submitted"}
    call = s.transport.acall.call_args if s.asynchronous else s.transport.call.call_args
    assert call.kwargs["index_params"]["indexing"] == "auto"
    assert "vector_index_params" not in call.kwargs["index_params"]
    assert s.pool.acquire.call_count == 1
    assert s.transport.version == (23, 26, 4)


@pytest.mark.parametrize("failure", [False, True])
@pytest.mark.asyncio
async def test_public_hybrid_preference_ddl_recovery_and_cleanup_share_session(
    sessions, failure
):
    s = sessions
    s.a.fail_create = failure
    store = OracleVS.__new__(OracleVS)
    store._vecdb = s.backend
    store.client = s.pool
    store.table_name = s.backend.quoted
    store.embedding_function = MagicMock(spec=OracleEmbeddings)
    store.embedding_function.params = {"provider": "database", "model": "MODEL"}

    async def build():
        if s.asynchronous:
            await acreate_hybrid_index(s.pool, "REQUESTED", vector_store=store)
        else:
            create_hybrid_index(s.pool, "REQUESTED", vector_store=store)

    if failure:
        with pytest.raises(RuntimeError, match="build failed"):
            await build()
        assert any(sql.startswith("CREATE SEARCH INDEX") for sql, _ in s.a.log)
    else:
        await build()
    assert s.a.index_exists
    assert not s.a.preference_exists
    assert any("CREATE_PREFERENCE" in sql for sql, _ in s.a.log)
    assert any("DROP_PREFERENCE" in sql for sql, _ in s.a.log)
    assert_one_session(s)
    assert store.client is s.pool


@pytest.mark.parametrize("sessions", [True], indirect=True, ids=["async"])
@pytest.mark.asyncio
async def test_concurrent_async_pool_operations_keep_private_session_bindings(sessions):
    s = sessions
    first, second = await asyncio.gather(
        invoke(s, "text_search", "oracle", 1), invoke(s, "text_search", "oracle", 1)
    )
    assert [first[0].page_content, second[0].page_content] == ["SCHEMA_A", "SCHEMA_B"]
    assert s.pool.acquire.call_count == 2
    assert s.transport.client is s.pool
    assert s.transport.version == (23, 26, 4)
    assert all(not state.leased for state in [s.a, s.b])
    assert any("JSON_TEXTCONTAINS(METADATA," in sql for sql, _ in s.a.log)
    assert any("JSON_TEXTCONTAINS(CONTENT_METADATA," in sql for sql, _ in s.b.log)


@pytest.mark.parametrize("sessions", [True], indirect=True, ids=["async"])
@pytest.mark.asyncio
async def test_direct_async_query_retry_also_retains_the_checkout(sessions):
    s = sessions
    s.a.fail_query = True
    result = await s.backend.aquery(
        "search", table_name=s.backend.name, query_by={"vector": [1.0]}, top_k=1
    )
    assert result["items"][0]["id"] == "SCHEMA_A"
    assert s.a.columns == CURRENT_COLUMNS
    assert_one_session(s)


@pytest.mark.asyncio
async def test_index_status_uses_one_private_pool_checkout(sessions):
    s = sessions
    store = OracleVS.__new__(OracleVS)
    store._vecdb = s.backend
    result = (
        await store.adescribe_vecdb_index(kind="metadata")
        if s.asynchronous
        else store.describe_vecdb_index(kind="metadata")
    )
    assert result["ready"] is True
    assert result["details"]["indexes"][0]["owner"] == "SCHEMA_A"
    assert_one_session(s)
    assert s.transport.client is s.pool


@pytest.mark.parametrize("sessions", [True], indirect=True, ids=["async"])
@pytest.mark.asyncio
async def test_preference_cleanup_finishes_on_pinned_session_after_cancellation(
    sessions,
):
    s = sessions
    entered, release = asyncio.Event(), asyncio.Event()
    original = s.a.cursor.execute.side_effect

    async def execute(statement, *args, **kwargs):
        if "DROP_PREFERENCE" in statement:
            entered.set()
            await release.wait()
        await original(statement, *args, **kwargs)

    s.a.cursor.execute.side_effect = execute
    store = OracleVS.__new__(OracleVS)
    store._vecdb = s.backend
    store.client = s.pool
    store.table_name = s.backend.quoted
    store.embedding_function = MagicMock(spec=OracleEmbeddings)
    store.embedding_function.params = {"provider": "database", "model": "MODEL"}
    task = asyncio.create_task(
        acreate_hybrid_index(s.pool, "REQUESTED", vector_store=store)
    )
    try:
        await asyncio.wait_for(entered.wait(), 2)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        assert s.a.leased
        assert s.a.preference_exists
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        release.set()
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    assert not s.a.leased
    assert not s.a.preference_exists
    assert_one_session(s)


@pytest.mark.parametrize("operation", ["text_search", "hybrid_search"])
@pytest.mark.asyncio
async def test_prepare_discovers_once_and_searches_resolve_unqualified_names(
    sessions, operation
):
    s = sessions
    for session in (s.a, s.b):
        session.columns = set(CURRENT_COLUMNS)
        session.parameters = "SIMPLIFIED_JSON_HYBRID " + GUIDE
        session.connection.current_schema = "STALE_DRIVER_VALUE"
    await invoke(s, "prepare")
    assert len([sql for sql, _ in s.a.log if "column_name FROM" in sql]) == 1
    assert len([sql for sql, _ in s.a.log if "i.parameters" in sql]) == 1
    s.a.log.clear()
    s.b.log.clear()
    for expected in ("SCHEMA_B", "SCHEMA_A"):
        result = (
            await invoke(s, "text_search", "oracle", 1)
            if operation == "text_search"
            else await invoke(
                s, "hybrid_search", {"text": {"search_text": "oracle"}}, False
            )
        )
        assert result[0].page_content == expected
    for session in (s.a, s.b):
        assert len(session.log) == 1
        sql, binds = session.log[0]
        assert "all_tab_columns" not in sql and "all_indexes" not in sql
        assert "CURRENT_SCHEMA" not in sql
        if operation == "hybrid_search":
            assert json.loads(binds["search_params"])["hybrid_index_name"] == '"GUIDE"'


def use_one_session(s):
    s.pool.acquire.side_effect = None
    s.pool.acquire.return_value = s.a.connection


@pytest.mark.asyncio
async def test_search_refreshes_when_another_session_has_different_layout(sessions):
    s = sessions
    await invoke(s, "prepare")  # SCHEMA_A uses the legacy metadata column.
    s.a.log.clear()
    result = await invoke(s, "text_search", "oracle", 1)  # SCHEMA_B uses the new one.
    assert result[0].page_content == "SCHEMA_B"
    assert not s.a.log
    assert len([sql for sql, _ in s.b.log if "JSON_TEXTCONTAINS" in sql]) == 2
    assert len([sql for sql, _ in s.b.log if "column_name FROM" in sql]) == 1
    s.b.log.clear()
    s.pool.acquire.side_effect = None
    s.pool.acquire.return_value = s.b.connection
    await invoke(s, "text_search", "oracle", 1)
    assert len(s.b.log) == 1


@pytest.mark.asyncio
async def test_hybrid_search_refreshes_a_replaced_external_index_once(sessions):
    s = sessions
    use_one_session(s)
    s.a.parameters = "SIMPLIFIED_JSON_HYBRID " + GUIDE
    await invoke(s, "prepare")
    s.a.index_name = "REPLACEMENT"
    s.a.log.clear()
    result = await invoke(
        s, "hybrid_search", {"text": {"search_text": "oracle"}}, False
    )
    assert result[0].page_content == "SCHEMA_A"
    searches = [
        binds for sql, binds in s.a.log if "JSON_TABLE(DBMS_HYBRID_VECTOR.SEARCH" in sql
    ]
    assert [json.loads(b["search_params"])["hybrid_index_name"] for b in searches] == [
        '"GUIDE"',
        '"REPLACEMENT"',
    ]
    assert len([sql for sql, _ in s.a.log if "i.parameters" in sql]) == 1
    s.a.log.clear()
    await invoke(s, "hybrid_search", {"text": {"search_text": "oracle"}}, False)
    assert len(s.a.log) == 1


@pytest.mark.asyncio
async def test_local_index_creation_updates_snapshot_for_next_search(sessions):
    s = sessions
    use_one_session(s)
    await invoke(s, "prepare")

    def build(physical, params, meta):
        return f"CREATE HYBRID VECTOR INDEX {physical} ON DOCS({meta})"

    if s.asynchronous:
        await s.transport.acreate_metadata_index(
            "DOCS", "REQUESTED", hybrid=True, build=build
        )
    else:
        s.transport.create_metadata_index("DOCS", "REQUESTED", hybrid=True, build=build)
    s.a.log.clear()
    result = await invoke(
        s, "hybrid_search", {"text": {"search_text": "oracle"}}, False
    )
    assert result[0].page_content == "SCHEMA_A"
    assert len(s.a.log) == 1
    assert json.loads(s.a.log[0][1]["search_params"])["hybrid_index_name"] == '"GUIDE"'


@pytest.mark.asyncio
async def test_migration_invalidates_layout_before_next_search(sessions):
    s = sessions
    use_one_session(s)
    await invoke(s, "prepare")
    if s.asynchronous:
        await s.transport.amigrate("DOCS")
    else:
        s.transport.migrate("DOCS")
    s.a.log.clear()
    result = await invoke(s, "text_search", "oracle", 1)
    assert result[0].page_content == "SCHEMA_A"
    assert s.a.log[-1][0].startswith("SELECT /* lc_oracledb mode=text backend=vecdb")
    assert ' id, CONTENT_METADATA, SCORE(1) FROM "DOCS"' in s.a.log[-1][0]
    assert len([sql for sql, _ in s.a.log if "column_name FROM" in sql]) == 1
    s.a.log.clear()
    await invoke(s, "text_search", "oracle", 1)
    assert len(s.a.log) == 1


@pytest.mark.parametrize("operation", ["text_search", "hybrid_search"])
@pytest.mark.asyncio
async def test_unrelated_search_errors_do_not_trigger_discovery(sessions, operation):
    s = sessions
    use_one_session(s)
    s.a.parameters = "SIMPLIFIED_JSON_HYBRID " + GUIDE
    await invoke(s, "prepare")
    error = oracledb.DatabaseError(
        SimpleNamespace(code=1031, message="ORA-01031: insufficient privileges")
    )
    s.a.search_errors = [error]
    s.a.log.clear()
    with pytest.raises(oracledb.DatabaseError) as caught:
        if operation == "text_search":
            await invoke(s, operation, "oracle", 1)
        else:
            await invoke(s, operation, {"text": {"search_text": "oracle"}}, False)
    assert caught.value is error
    assert len(s.a.log) == 1


@pytest.mark.asyncio
async def test_stale_layout_refresh_retries_only_once(sessions):
    s = sessions
    use_one_session(s)
    await invoke(s, "prepare")
    error = RuntimeError('ORA-00904: "METADATA": invalid identifier')
    s.a.search_errors = [error, error]
    s.a.log.clear()
    with pytest.raises(RuntimeError) as caught:
        await invoke(s, "text_search", "oracle", 1)
    assert caught.value is error
    assert len([sql for sql, _ in s.a.log if "JSON_TEXTCONTAINS" in sql]) == 2
    assert len([sql for sql, _ in s.a.log if "column_name FROM" in sql]) == 1


@pytest.mark.asyncio
async def test_vecdb_ignores_qualified_requested_hybrid_index(sessions):
    s = sessions
    use_one_session(s)
    s.a.parameters = "SIMPLIFIED_JSON_HYBRID " + GUIDE
    await invoke(s, "prepare")
    s.a.log.clear()
    qualified = '"OTHER_SCHEMA"."REQUESTED"'
    result = await invoke(
        s,
        "hybrid_search",
        {"text": {"search_text": "oracle"}, "hybrid_index_name": qualified},
        False,
    )
    assert result[0].page_content == "SCHEMA_A"
    assert len(s.a.log) == 1
    assert json.loads(s.a.log[0][1]["search_params"])["hybrid_index_name"] == '"GUIDE"'


@pytest.mark.asyncio
async def test_failed_index_replacement_discards_snapshot(sessions):
    s = sessions
    use_one_session(s)
    await invoke(s, "prepare")
    s.a.fail_create = True

    def build(physical, params, meta):
        return f"CREATE HYBRID VECTOR INDEX {physical} ON DOCS({meta})"

    with pytest.raises(RuntimeError, match="build failed"):
        if s.asynchronous:
            await s.transport.acreate_metadata_index(
                "DOCS", "REQUESTED", hybrid=True, build=build
            )
        else:
            s.transport.create_metadata_index(
                "DOCS", "REQUESTED", hybrid=True, build=build
            )
    assert s.a.index_exists
    s.a.log.clear()
    result = await invoke(s, "text_search", "oracle", 1)
    assert result[0].page_content == "SCHEMA_A"
    assert len([sql for sql, _ in s.a.log if "column_name FROM" in sql]) == 1
    assert len([sql for sql, _ in s.a.log if "i.parameters" in sql]) == 1


@pytest.mark.parametrize(
    "operation,mode",
    [
        ("text_search", "text"),
        ("hybrid_search", "hybrid"),
        ("hybrid_search", "keyword"),
        ("hybrid_search", "semantic"),
    ],
)
@pytest.mark.parametrize("current_layout", [False, True], ids=["legacy", "current"])
@pytest.mark.parametrize("program", ["other-program", "devrel-developerhub-langchain"])
@pytest.mark.asyncio
async def test_native_metadata_search_execution_markers(
    sessions, operation, mode, current_layout, program, monkeypatch
):
    s = sessions
    monkeypatch.setattr(oracledb.defaults, "program", program)
    if current_layout:
        s.a.columns = set(CURRENT_COLUMNS)
    s.a.parameters = "SIMPLIFIED_JSON_HYBRID " + GUIDE
    query = "private_request_123"
    if operation == "text_search":
        result = await invoke(s, operation, query, 2)
    else:
        params = {}
        if mode in {"hybrid", "keyword"}:
            params["text"] = {"search_text": query}
        if mode in {"hybrid", "semantic"}:
            params["vector"] = {"search_text": query}
        result = await invoke(s, operation, params, False)
    assert result[0].page_content == "SCHEMA_A"
    searches = [
        (sql, binds)
        for sql, binds in s.a.log
        if "JSON_TEXTCONTAINS" in sql or "JSON_TABLE(DBMS_HYBRID_VECTOR.SEARCH" in sql
    ]
    assert len(searches) == 1
    sql, binds = searches[0]
    source = (
        " source=devrel-developerhub-langchain"
        if program == "devrel-developerhub-langchain"
        else ""
    )
    marker = f"/* lc_oracledb mode={mode} backend=vecdb{source} */"
    assert sql.startswith("SELECT " + marker + " ")
    assert sql.count("lc_oracledb") == 1
    assert query not in sql
    assert query in json.dumps(binds)
    assert all(
        "lc_oracledb" not in statement for statement, _ in s.a.log if statement != sql
    )
    assert_one_session(s)


@pytest.mark.parametrize("scores", [False, True])
@pytest.mark.asyncio
async def test_text_search_ranks_by_database_score(sessions, scores):
    result = await invoke(sessions, "text_search", "oracle", 1, scores)
    assert result[0].metadata == ({"score": 17} if scores else {})
    statements = [sql for sql, _ in sessions.a.log if "JSON_TEXTCONTAINS" in sql]
    assert statements
    assert "SCORE(1)" in statements[-1]
    assert ":query, 1)" in statements[-1]
    assert "ORDER BY SCORE(1) DESC" in statements[-1]


@pytest.mark.asyncio
async def test_sql_filter_fallback_uses_one_pool_checkout(sessions):
    s = sessions
    result = await invoke(s, "search", [1.0], 1, {"payload": {"$eq": {"a": 1}}}, True)
    assert result[0][0].id == "SCHEMA_A"
    assert result[0][0].page_content == "SCHEMA_A"
    assert result[0][1] == 0.25
    assert list(result[0][2]) == [1.0]
    assert_one_session(s)
    queries = [
        (sql, binds) for sql, binds in s.a.log if "VECTOR_DISTANCE(dense_vector" in sql
    ]
    assert len(queries) == 1
    sql, binds = queries[0]
    assert "METADATA AS metadata" in sql
    assert (
        sql.index("WHERE JSON_EQUAL")
        < sql.index("ORDER BY distance")
        < sql.index("FETCH FIRST")
    )
    assert binds["value0"] == '{"a": 1}'
    assert not any(".search(" in sql.lower() for sql, _ in s.a.log)


@pytest.mark.asyncio
async def test_sql_filter_fallback_refreshes_layout_on_same_session(sessions):
    s = sessions
    s.transport._remember_table("DOCS", _TableInfo(frozenset(CURRENT_COLUMNS), None))
    result = await invoke(s, "search", [1.0], 1, {"payload": {"$eq": {"a": 1}}}, True)
    assert result[0][0].id == "SCHEMA_A"
    assert_one_session(s)
    queries = [sql for sql, _ in s.a.log if "VECTOR_DISTANCE(dense_vector" in sql]
    assert len(queries) == 2
    assert "CONTENT_METADATA AS metadata" in queries[0]
    assert "METADATA AS metadata" in queries[1]


@pytest.mark.parametrize("requested", ["REQUESTED", '"REQUESTED"', '"index.with.dot"'])
@pytest.mark.parametrize("discovered_hybrid", [False, True])
@pytest.mark.asyncio
async def test_vecdb_search_uses_physical_index_instead_of_requested_name(
    sessions, requested, discovered_hybrid
):
    s = sessions
    use_one_session(s)
    if discovered_hybrid:
        s.a.parameters = "SIMPLIFIED_JSON_HYBRID " + GUIDE
    await invoke(s, "prepare")
    s.a.log.clear()
    params = {"text": {"search_text": "oracle"}, "hybrid_index_name": requested}
    if not discovered_hybrid:
        with pytest.raises(ValueError, match="Create a hybrid vector index"):
            await invoke(s, "hybrid_search", params, False)
        assert not any("search_params" in binds for _, binds in s.a.log)
        return
    result = await invoke(s, "hybrid_search", params, False)
    assert result[0].page_content == "SCHEMA_A"
    assert len(s.a.log) == 1
    assert json.loads(s.a.log[0][1]["search_params"])["hybrid_index_name"] == '"GUIDE"'
    assert params["hybrid_index_name"] == requested


@pytest.mark.asyncio
async def test_vecdb_creation_and_retrieval_share_the_retained_physical_index(sessions):
    s = sessions
    use_one_session(s)
    store = OracleVS.__new__(OracleVS)
    store._vecdb = s.backend
    store.client = s.pool
    store.table_name = s.backend.quoted
    store.embedding_function = MagicMock(spec=OracleEmbeddings)
    store.embedding_function.params = {"provider": "database", "model": "MODEL"}
    if s.asynchronous:
        assert (
            await acreate_hybrid_index(s.pool, "MY_INDEX", vector_store=store) is None
        )
    else:
        assert create_hybrid_index(s.pool, "MY_INDEX", vector_store=store) is None
    s.a.log.clear()
    from langchain_oracledb.retrievers.hybrid_search import OracleHybridSearchRetriever

    retriever = OracleHybridSearchRetriever(
        vector_store=store, idx_name="MY_INDEX", search_mode="keyword"
    )
    result = (
        await retriever.ainvoke("oracle")
        if s.asynchronous
        else retriever.invoke("oracle")
    )
    assert result[0].page_content == "SCHEMA_A"
    assert len(s.a.log) == 1
    assert json.loads(s.a.log[0][1]["search_params"])["hybrid_index_name"] == '"GUIDE"'
