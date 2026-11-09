# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""Native VecDB signature expectations and transaction failure recovery."""

import asyncio
import json
from contextlib import asynccontextmanager, contextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest

from langchain_oracledb.vectorstores import _vecdb
from langchain_oracledb.vectorstores.utils import DistanceStrategy


@pytest.mark.parametrize(
    "version", [(23, 26, 1), (23, 26, 2), (23, 26, 3), (23, 26, 4)]
)
@pytest.mark.parametrize(
    "operation", ["create_vector_table", "drop_vector_table", "upsert_vectors"]
)
def test_emitted_function_contract(version, operation):
    params = {"name": "DOCS"}
    if operation == "create_vector_table":
        params["table_params"] = {"auto_generate_id": False}
    elif operation == "upsert_vectors":
        params = {
            "table_name": "DOCS",
            "vectors": [
                {"id": "one", "dense_vector": [1.0], "metadata": {"text": "one"}}
            ],
        }
    sql, binds = _vecdb.NativeTransport(None, version)._statement(operation, params)
    if operation == "upsert_vectors":
        arguments = "table_name => :table_name, vectors => JSON(:vectors)"
        assert binds == {
            "table_name": '"DOCS"',
            "vectors": json.dumps(params["vectors"]),
        }
    elif version < (23, 26, 3):
        arguments = "table_name => :table_name"
        expected = {"table_name": '"DOCS"'}
        if operation == "create_vector_table":
            arguments += (
                ", auto_generate_id => :auto_generate_id, "
                "vector_type => :vector_type, debug_flags => JSON(:debug_flags), "
                "request_id => :request_id"
            )
            expected.update(
                auto_generate_id=False,
                vector_type="dense",
                debug_flags=None,
                request_id=None,
            )
        assert binds == expected
    else:
        arguments = "name => :name"
        expected = {"name": '"DOCS"'}
        if operation == "create_vector_table":
            arguments += ", table_params => JSON(:table_params)"
            expected["table_params"] = json.dumps({"auto_generate_id": False})
        assert binds == expected
    assert (
        sql == f"BEGIN :result := DBMS_VECTOR_DATABASE.{operation}({arguments}); END;"
    )


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "operation",
    ["create_vector_table", "drop_vector_table", "upsert_vectors", "create_index"],
)
@pytest.mark.asyncio
async def test_signature_errors_propagate_without_mutation_retry(
    asynchronous, operation
):
    error = RuntimeError("ORA-06550: PLS-00306: wrong number or types of arguments")
    transport = MagicMock()
    transport.call.side_effect = error
    transport.acall = AsyncMock(side_effect=error)
    backend = _vecdb.VecDBBackend(transport, "DOCS", DistanceStrategy.COSINE)
    with pytest.raises(RuntimeError) as caught:
        if asynchronous:
            await backend.aquery(operation, table_name="DOCS")
        else:
            backend.query(operation, table_name="DOCS")
    assert caught.value is error
    (transport.acall if asynchronous else transport.call).assert_called_once_with(
        operation, table_name="DOCS"
    )
    transport.migrate.assert_not_called()
    transport.amigrate.assert_not_called()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("stage", ["execute", "commit"])
@pytest.mark.parametrize("rollback_fails", [False, True])
@pytest.mark.parametrize("interrupted", [False, True])
@pytest.mark.asyncio
async def test_insert_failure_preserves_primary_error(
    monkeypatch, asynchronous, stage, rollback_fails, interrupted
):
    connection = MagicMock(autocommit=False, transaction_in_progress=False)
    cursor = connection.cursor.return_value.__enter__.return_value
    cursor.getbatcherrors.return_value = []
    if asynchronous:
        cursor.executemany = AsyncMock()
        connection.commit = AsyncMock()
        connection.rollback = AsyncMock()
    error = (
        (asyncio.CancelledError() if asynchronous else KeyboardInterrupt())
        if interrupted
        else RuntimeError("connection lost")
    )
    (
        cursor.executemany if stage == "execute" else connection.commit
    ).side_effect = error
    rollback_error = RuntimeError("rollback connection lost")
    if rollback_fails:
        connection.rollback.side_effect = rollback_error

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
    )(None, (23, 26, 4))
    transport.columns = MagicMock(return_value=_vecdb.CURRENT_COLUMNS)
    transport.acolumns = AsyncMock(return_value=_vecdb.CURRENT_COLUMNS)
    records = [{"id": "one", "dense_vector": [1.0], "metadata": {"text": "one"}}]
    with pytest.raises(type(error)) as caught:
        if asynchronous:
            await transport.ainsert("DOCS", records)
        else:
            transport.insert("DOCS", records)
    assert caught.value is error
    assert caught.value.__cause__ is (rollback_error if rollback_fails else None)
    cursor.executemany.assert_called_once()
    connection.rollback.assert_called_once()
    assert connection.commit.call_count == (1 if stage == "commit" else 0)
    if asynchronous:
        connection.rollback.assert_awaited_once()
