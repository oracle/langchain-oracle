# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Annotation syntax fallback without suppressing ordinary DDL failures."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import oracledb
import pytest

from langchain_oracledb._annotations import (
    acreate_table_with_annotations,
    create_table_with_annotations,
)
from langchain_oracledb.cache import OracleCache
from langchain_oracledb.chat_message_histories import OracleChatMessageHistory
from langchain_oracledb.storage import OracleByteStore, OracleDocStore
from langchain_oracledb.vectorstores.oraclevs import _acreate_table, _create_table

DDL = 'CREATE TABLE "DOCS" (id NUMBER)'


def db_error(code, offset=0):
    return oracledb.DatabaseError(SimpleNamespace(code=code, offset=offset))


@pytest.fixture(params=[False, True], ids=["sync", "async"])
def execute(request):
    cursor = MagicMock()
    if request.param:
        cursor.execute = AsyncMock()

    async def run(ddl=DDL, kind="vector_store"):
        if request.param:
            await acreate_table_with_annotations(cursor, ddl, kind)
        else:
            create_table_with_annotations(cursor, ddl, kind)

    return cursor, run


@pytest.mark.asyncio
async def test_supported_creation_executes_only_the_annotated_ddl(execute):
    cursor, run = execute
    await run()
    cursor.execute.assert_called_once()
    assert cursor.execute.call_args.args[0].startswith(DDL + " ANNOTATIONS (")


@pytest.mark.parametrize("code", [922, 933, 3049])
@pytest.mark.parametrize("offset", [0, len(DDL) + 1])
@pytest.mark.parametrize(
    "kind",
    [
        "vector_store",
        "semantic_cache",
        "exact_cache",
        "kv_store",
        "chat_history",
    ],
)
@pytest.mark.asyncio
async def test_unsupported_annotation_syntax_retries_exact_plain_ddl(
    execute, code, offset, kind
):
    cursor, run = execute
    cursor.execute.side_effect = [db_error(code, offset), None]
    await run(kind=kind)
    first, second = [c.args[0] for c in cursor.execute.call_args_list]
    assert first.startswith(DDL + " ANNOTATIONS (")
    assert f"type={kind};backend=sql" in first
    assert second == DDL


@pytest.mark.parametrize("code", [922, 933, 3049])
@pytest.mark.asyncio
async def test_syntax_error_before_annotations_is_not_retried(execute, code):
    cursor, run = execute
    error = db_error(code, offset=7)
    cursor.execute.side_effect = error
    with pytest.raises(oracledb.DatabaseError) as raised:
        await run()
    assert raised.value is error
    cursor.execute.assert_called_once()


@pytest.mark.parametrize("code", [904, 942, 955, 1031, 1653, 3113])
@pytest.mark.asyncio
async def test_non_syntax_errors_propagate_without_retry(execute, code):
    cursor, run = execute
    error = db_error(code, len(DDL) + 1)
    cursor.execute.side_effect = error
    with pytest.raises(oracledb.DatabaseError) as raised:
        await run()
    assert raised.value is error
    cursor.execute.assert_called_once()


@pytest.mark.parametrize("code", [922, 933, 1031, 955])
@pytest.mark.asyncio
async def test_plain_retry_failure_propagates_and_does_not_retry_again(execute, code):
    cursor, run = execute
    error = db_error(code)
    cursor.execute.side_effect = [db_error(922), error]
    with pytest.raises(oracledb.DatabaseError) as raised:
        await run()
    assert raised.value is error
    assert cursor.execute.call_count == 2


@pytest.mark.parametrize(
    "factory,module,kind",
    [
        (OracleCache, "cache", "exact_cache"),
        (OracleByteStore, "storage", "kv_store"),
        (OracleDocStore, "storage", "kv_store"),
        (OracleChatMessageHistory, "chat_message_histories", "chat_history"),
    ],
)
def test_scalar_creation_paths_use_the_fallback(factory, module, kind):
    conn = MagicMock(spec=oracledb.Connection)
    cursor = conn.cursor.return_value.__enter__.return_value
    cursor.execute.side_effect = [db_error(922), None]
    with patch(f"langchain_oracledb.{module}._table_exists", return_value=False):
        if factory is OracleChatMessageHistory:
            factory.create_tables(conn, "DOCS", create_index=False)
        else:
            factory(conn, "DOCS")
    first, second = [c.args[0] for c in cursor.execute.call_args_list]
    assert f"type={kind};backend=sql" in first
    assert "ANNOTATIONS" not in second
    conn.commit.assert_called_once()


@pytest.mark.parametrize("kind", ["vector_store", "semantic_cache"])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_vector_creation_paths_use_the_fallback(kind, asynchronous):
    conn = MagicMock()
    cursor = conn.cursor.return_value.__enter__.return_value
    if asynchronous:
        cursor.execute = AsyncMock(side_effect=[db_error(922), None])
        with patch(
            "langchain_oracledb.vectorstores.oraclevs._atable_exists",
            new=AsyncMock(return_value=False),
        ):
            await _acreate_table(conn, '"DOCS"', 3, kind)
    else:
        cursor.execute.side_effect = [db_error(922), None]
        with patch(
            "langchain_oracledb.vectorstores.oraclevs._table_exists", return_value=False
        ):
            _create_table(conn, '"DOCS"', 3, kind)
    first, second = [c.args[0] for c in cursor.execute.call_args_list]
    assert f"type={kind};backend=sql" in first
    assert "ANNOTATIONS" not in second
