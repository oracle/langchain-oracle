# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""Operation-local preflight and explicit index readiness contracts."""

import json
from contextlib import asynccontextmanager, contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import oracledb
import pytest

from langchain_oracledb import OracleVS
from langchain_oracledb.vectorstores import _vecdb
from langchain_oracledb.vectorstores._vecdb_capabilities import (
    INDEX_FEATURES_SQL,
    function_preflight,
)
from langchain_oracledb.vectorstores.utils import DistanceStrategy


@pytest.fixture(params=[False, True], ids=["sync", "async"])
def native(request, monkeypatch):
    asynchronous = request.param
    connection = MagicMock(version="23.26.4", transaction_in_progress=False)
    cursor = connection.cursor.return_value.__enter__.return_value
    cursor.var.return_value.getvalue.return_value = '{"items": []}'
    if asynchronous:
        cursor.execute = AsyncMock()
        cursor.fetchall = AsyncMock()
        cursor.fetchone = AsyncMock()

    @contextmanager
    def get(_):
        yield connection

    @asynccontextmanager
    async def aget(_):
        yield connection

    monkeypatch.setattr(_vecdb, "_get_connection", get)
    monkeypatch.setattr(_vecdb, "_aget_connection", aget)
    transport = (
        _vecdb.AsyncNativeTransport if asynchronous else _vecdb.NativeTransport
    )(connection, (23, 26, 4))
    return asynchronous, connection, cursor, transport


@pytest.mark.parametrize(
    "statement",
    ["CREATE TABLE DOCS(ID NUMBER)", "SELECT 1 FROM DUAL", "", "DROP INDEX IDX"],
)
def test_preflight_never_accepts_ddl_or_non_adapter_sql(statement):
    with pytest.raises(ValueError, match="PL/SQL"):
        function_preflight(statement)


@pytest.mark.parametrize(
    "operation,params",
    [
        ("create_vector_table", {"name": "DOCS"}),
        ("describe_vector_table", {"name": "DOCS"}),
        ("drop_vector_table", {"name": "DOCS"}),
        (
            "upsert_vectors",
            {
                "table_name": "DOCS",
                "vectors": [
                    {
                        "id": "one",
                        "dense_vector": [1.0],
                        "metadata": {"__orcl_internal_content": "one"},
                    }
                ],
            },
        ),
        ("delete_vectors", {"table_name": "DOCS", "ids": ["one"]}),
        (
            "create_index",
            {
                "table_name": "DOCS",
                "index_params": {"metadata_index_params": {"auto_index": True}},
            },
        ),
    ],
)
@pytest.mark.asyncio
async def test_other_package_operations_compile_exact_statement_first(
    native, operation, params
):
    asynchronous, _, cursor, transport = native
    if asynchronous:
        await transport.acall(operation, **params)
    else:
        transport.call(operation, **params)
    preflight, execution = cursor.execute.call_args_list
    assert preflight.args[0] == function_preflight(execution.args[0])
    assert preflight.kwargs == execution.kwargs
    if asynchronous:
        assert cursor.execute.await_count == 2


@pytest.mark.parametrize("operation", ["search", "query", "list_vectors"])
@pytest.mark.parametrize("version", [(23, 26, 1), (23, 26, 3)])
@pytest.mark.asyncio
async def test_read_executes_once_without_compile_preflight(native, operation, version):
    asynchronous, _, cursor, transport = native
    transport.version = version
    params = {"table_name": "DOCS"}
    if operation == "list_vectors":
        params.update(ids=["one"], limit=1, offset=0)
    else:
        params.update(query_by={"vector": [1.0]}, top_k=1)
    statement, binds = transport._statement(operation, params)
    result = (
        await transport.acall(operation, **params)
        if asynchronous
        else transport.call(operation, **params)
    )
    assert result == {"items": []}
    cursor.execute.assert_called_once()
    execution = cursor.execute.call_args
    assert execution.args[0] == statement
    assert execution.kwargs == {"result": cursor.var.return_value, **binds}
    if asynchronous:
        assert cursor.execute.await_count == 1


@pytest.mark.parametrize("operation", ["search", "query", "list_vectors"])
@pytest.mark.parametrize("code", [6550, 1031, 3113])
@pytest.mark.asyncio
async def test_read_error_propagates_without_preflight_or_retry(
    native, operation, code
):
    asynchronous, _, cursor, transport = native
    error = oracledb.DatabaseError(SimpleNamespace(code=code))
    cursor.execute.side_effect = error
    params = {"table_name": "DOCS"}
    if operation == "list_vectors":
        params.update(ids=["one"], limit=1, offset=0)
    else:
        params.update(query_by={"vector": [1.0]}, top_k=1)
    with pytest.raises(oracledb.DatabaseError) as caught:
        if asynchronous:
            await transport.acall(operation, **params)
        else:
            transport.call(operation, **params)
    assert caught.value is error
    cursor.execute.assert_called_once()
    assert "IF FALSE" not in cursor.execute.call_args.args[0]
    if asynchronous:
        assert cursor.execute.await_count == 1


@pytest.mark.asyncio
async def test_get_by_ids_executes_one_statement_per_batch(native):
    asynchronous, _, cursor, transport = native
    cursor.var.return_value.getvalue.return_value = '{"vectors": []}'
    backend = _vecdb.VecDBBackend(transport, "DOCS", DistanceStrategy.COSINE)
    ids = [str(i) for i in range(101)]
    result = await backend.run_async("get", ids) if asynchronous else backend.get(ids)
    assert result == []
    assert cursor.execute.call_count == 2
    assert [
        json.loads(call.kwargs["ids_json"]) for call in cursor.execute.call_args_list
    ] == [
        [_vecdb.storage_id(id_) for id_ in ids[:100]],
        [_vecdb.storage_id(id_) for id_ in ids[100:]],
    ]
    assert all("IF FALSE" not in call.args[0] for call in cursor.execute.call_args_list)
    if asynchronous:
        assert cursor.execute.await_count == 2


@pytest.mark.parametrize("code", [6550, 1031, 3113])
@pytest.mark.asyncio
async def test_compile_failure_blocks_operation_and_retains_unknown_errors(
    native, code
):
    asynchronous, _, cursor, transport = native
    error = oracledb.DatabaseError(SimpleNamespace(code=code))
    cursor.execute.side_effect = error
    with pytest.raises(
        ValueError if code == 6550 else oracledb.DatabaseError
    ) as caught:
        if asynchronous:
            await transport.acall("drop_vector_table", name="DOCS")
        else:
            transport.call("drop_vector_table", name="DOCS")
    if code == 6550:
        assert caught.value.__cause__ is error
        assert "drop_vector_table" in str(caught.value)
        assert "No operation was executed" in str(caught.value)
    else:
        assert caught.value is error
    cursor.execute.assert_called_once()
    assert cursor.execute.call_args.args[0].startswith("BEGIN IF FALSE THEN")


@pytest.mark.asyncio
async def test_execution_error_has_no_signature_retry(native):
    asynchronous, _, cursor, transport = native
    error = oracledb.DatabaseError(SimpleNamespace(code=3113))
    cursor.execute.side_effect = [None, error]
    with pytest.raises(oracledb.DatabaseError) as caught:
        if asynchronous:
            await transport.acall("drop_vector_table", name="DOCS")
        else:
            transport.call("drop_vector_table", name="DOCS")
    assert caught.value is error
    assert cursor.execute.call_count == 2


@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize("available", [False, True])
@pytest.mark.asyncio
async def test_feature_preflight_before_any_index_ddl(native, hybrid, available):
    asynchronous, _, cursor, transport = native
    cursor.fetchall.return_value = [("DBMS_VECTOR_DATABASE", "HANDLE_MVI_INDEX")]
    if hybrid and available:
        cursor.fetchall.return_value.append(("DBMS_HYBRID_VECTOR", "SEARCH"))
    if not hybrid and not available:
        cursor.fetchall.return_value = []
    if available:
        if asynchronous:
            await transport.apreflight_index(hybrid)
        else:
            transport.preflight_index(hybrid)
    else:
        with pytest.raises(ValueError, match="No index DDL"):
            if asynchronous:
                await transport.apreflight_index(hybrid)
            else:
                transport.preflight_index(hybrid)
    cursor.execute.assert_called_once_with(INDEX_FEATURES_SQL)


@pytest.mark.parametrize("operation", ["create_index", "rebuild_index"])
@pytest.mark.asyncio
async def test_pending_work_blocks_preference_or_index_mutation(native, operation):
    asynchronous, connection, cursor, transport = native
    connection.transaction_in_progress = True
    with pytest.raises(RuntimeError, match="pending work"):
        if asynchronous:
            await transport.apreflight_index(True)
        else:
            transport.preflight_index(True)
    cursor.execute.assert_not_called()
    with pytest.raises(RuntimeError, match="pending work"):
        if asynchronous:
            await transport.acall(operation, table_name="DOCS", index_params={})
        else:
            transport.call(operation, table_name="DOCS", index_params={})
    cursor.execute.assert_not_called()


@pytest.mark.parametrize(
    "rows,ready",
    [
        ([], False),
        ([("OWNER", "IDX", "VALID", None, None)], True),
        ([("OWNER", "IDX", "VALID", "VALID", "VALID")], True),
        ([("OWNER", "IDX", "UNUSABLE", None, None)], False),
        ([("OWNER", "IDX", "VALID", "VALID", "IN_PROGRESS")], False),
        ([("OWNER", "IDX", "VALID", "VALID", "FAILED")], False),
        ([("OWNER", "IDX", "N/A", None, None)], None),
        ([("OWNER", "IDX", "VALID", "VALID", "FUTURE_STATUS")], None),
    ],
)
@pytest.mark.parametrize("kind", ["vector", "metadata"])
@pytest.mark.asyncio
async def test_public_native_readiness_snapshot_is_owner_aware(
    native, rows, ready, kind
):
    asynchronous, _, cursor, transport = native
    cursor.fetchall.return_value = rows
    store = OracleVS.__new__(OracleVS)
    store._vecdb = _vecdb.VecDBBackend(
        transport, '"OWNER"."DOCS"', DistanceStrategy.COSINE
    )
    result = (
        await store.adescribe_vecdb_index(kind=kind)
        if asynchronous
        else store.describe_vecdb_index(kind=kind)
    )
    assert result["ready"] is ready
    assert len(result["details"]["indexes"]) == len(rows)
    cursor.execute.assert_called_once()
    assert cursor.execute.call_args.kwargs == {"owner": "OWNER", "name": "DOCS"}
    statement = cursor.execute.call_args.args[0]
    assert ("index_type='VECTOR'" in statement) is (kind == "vector")


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "status,ready",
    [
        ("READY", True),
        ("Index Creation Completed", True),
        ("BUILDING", False),
        ("FAILED", False),
        ("future state", None),
    ],
)
@pytest.mark.asyncio
async def test_sdk_index_status_preserves_provider_details(asynchronous, status, ready):
    client = SimpleNamespace(
        config=SimpleNamespace(has_rest_connection=True),
        describe_index=MagicMock(
            return_value={"index_status": status, "extra": "kept"}
        ),
    )
    store = OracleVS.__new__(OracleVS)
    store._vecdb = _vecdb.VecDBBackend(
        _vecdb.RestTransport(client), "DOCS", DistanceStrategy.COSINE
    )
    result = (
        await store.adescribe_vecdb_index()
        if asynchronous
        else store.describe_vecdb_index()
    )
    assert result == {
        "ready": ready,
        "details": {"index_status": status, "extra": "kept"},
    }
    client.describe_index.assert_called_once_with(table_name="DOCS")


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("state", ["SCHEDULED", "RUNNING", "FAILED", "SUCCEEDED"])
@pytest.mark.asyncio
async def test_completed_job_does_not_certify_index_readiness(asynchronous, state):
    client = SimpleNamespace(
        config=SimpleNamespace(has_rest_connection=True),
        describe_index_job=MagicMock(return_value={"state": state, "job_name": "job"}),
    )
    store = OracleVS.__new__(OracleVS)
    store._vecdb = _vecdb.VecDBBackend(
        _vecdb.RestTransport(client), "DOCS", DistanceStrategy.COSINE
    )
    result = (
        await store.adescribe_vecdb_index(job_name="job")
        if asynchronous
        else store.describe_vecdb_index(job_name="job")
    )
    assert result["ready"] is None
    assert result["details"]["state"] == state
    client.describe_index_job.assert_called_once_with(index_job_name="job")


@pytest.mark.asyncio
async def test_sql_backend_status_is_explicitly_unsupported():
    store = OracleVS.__new__(OracleVS)
    store._vecdb = None
    with pytest.raises(ValueError, match="VecDB backend"):
        store.describe_vecdb_index()
    with pytest.raises(ValueError, match="VecDB backend"):
        await store.adescribe_vecdb_index()


@pytest.mark.asyncio
async def test_public_hybrid_gate_runs_before_temporary_preference_creation(
    native, monkeypatch
):
    from langchain_oracledb.retrievers.hybrid_search import (
        OracleVectorizerPreference,
        acreate_hybrid_index,
        create_hybrid_index,
    )

    asynchronous, connection, cursor, transport = native
    cursor.fetchall.return_value = [("DBMS_VECTOR_DATABASE", "HANDLE_MVI_INDEX")]
    create = MagicMock()
    acreate = AsyncMock()
    monkeypatch.setattr(OracleVectorizerPreference, "create_preference", create)
    monkeypatch.setattr(OracleVectorizerPreference, "acreate_preference", acreate)
    store = OracleVS.__new__(OracleVS)
    store._vecdb = _vecdb.VecDBBackend(transport, "DOCS", DistanceStrategy.COSINE)
    with pytest.raises(ValueError, match="DBMS_HYBRID_VECTOR.SEARCH"):
        if asynchronous:
            await acreate_hybrid_index(connection, "IDX", vector_store=store)
        else:
            create_hybrid_index(connection, "IDX", vector_store=store)
    create.assert_not_called()
    acreate.assert_not_awaited()
    cursor.execute.assert_called_once_with(INDEX_FEATURES_SQL)


@pytest.mark.parametrize("kind,job", [("invalid", None), ("vector", "job")])
@pytest.mark.asyncio
async def test_unsupported_native_status_does_not_submit_anything(native, kind, job):
    asynchronous, _, cursor, transport = native
    store = OracleVS.__new__(OracleVS)
    store._vecdb = _vecdb.VecDBBackend(transport, "DOCS", DistanceStrategy.COSINE)
    with pytest.raises(ValueError):
        if asynchronous:
            await store.adescribe_vecdb_index(kind=kind, job_name=job)
        else:
            store.describe_vecdb_index(kind=kind, job_name=job)
    cursor.execute.assert_not_called()


@pytest.mark.parametrize(
    "operation", ["search", "query", "list_vectors", "upsert_vectors"]
)
@pytest.mark.parametrize("version", [(23, 26, 1), (23, 26, 3)])
@pytest.mark.parametrize("program", ["other-program", "devrel-developerhub-langchain"])
@pytest.mark.asyncio
async def test_native_vector_execution_markers(
    native, operation, version, program, monkeypatch
):
    asynchronous, _, cursor, transport = native
    monkeypatch.setattr(oracledb.defaults, "program", program)
    transport.version = version
    params = {"table_name": "DOCS"}
    if operation in {"search", "query"}:
        params.update(query_by={"vector": [0.12, 0.34]}, top_k=2)
    elif operation == "list_vectors":
        params.update(ids=["private-id"], limit=1, offset=0)
    else:
        params["vectors"] = [{"id": "one", "vector": [0.12, 0.34], "metadata": {}}]
    plain, expected_binds = _vecdb.SQLNetAdapter(version).statement(
        operation, {**params, "table_name": '"DOCS"'}
    )
    if asynchronous:
        await transport.acall(operation, **params)
    else:
        transport.call(operation, **params)
    executed = cursor.execute.call_args.args[0]
    if operation in {"search", "query"}:
        source = (
            " source=devrel-developerhub-langchain"
            if program == "devrel-developerhub-langchain"
            else ""
        )
        marker = f"/* lc_oracledb mode=vector backend=vecdb{source} */"
        assert executed.startswith("BEGIN " + marker + " ")
        assert executed.count(marker) == 1
        assert executed.replace(" " + marker, "", 1) == plain
        assert cursor.execute.call_count == 1
    else:
        assert executed == plain
        assert all(
            "lc_oracledb" not in call.args[0] for call in cursor.execute.call_args_list
        )
    assert cursor.execute.call_args.kwargs == {
        "result": cursor.var.return_value,
        **expected_binds,
    }
