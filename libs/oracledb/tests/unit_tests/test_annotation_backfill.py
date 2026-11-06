# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""SQL backfill respects transactions, existing metadata, and identifier scope."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import oracledb
import pytest

from langchain_oracledb._annotations import (
    abackfill_table_annotation,
    backfill_table_annotation,
)
from langchain_oracledb.cache import OracleCache
from langchain_oracledb.chat_message_histories import OracleChatMessageHistory
from langchain_oracledb.storage import OracleByteStore, OracleDocStore


@pytest.fixture(params=[False, True], ids=["sync", "async"])
def database(request):
    asynchronous = request.param
    conn = MagicMock(
        spec=oracledb.AsyncConnection if asynchronous else oracledb.Connection
    )
    conn.transaction_in_progress = False
    cursor = conn.cursor.return_value.__enter__.return_value
    cursor.fetchone.return_value = ("OWNER", "DOCS")
    if asynchronous:
        cursor.execute = AsyncMock()
        cursor.fetchone = AsyncMock(return_value=("OWNER", "DOCS"))

    async def run(name='"DOCS"', kind="vector_store"):
        if asynchronous:
            await abackfill_table_annotation(conn, name, kind)
        else:
            backfill_table_annotation(conn, name, kind)

    return conn, cursor, run


@pytest.mark.asyncio
async def test_backfill_unknown_creator_and_add_without_replacing(database):
    conn, cursor, run = database
    await run()
    assert cursor.execute.call_count == 2
    ddl = cursor.execute.call_args.args[0]
    assert ddl.startswith('ALTER TABLE "OWNER"."DOCS" ANNOTATIONS (ADD IF NOT EXISTS ')
    assert ";version=unknown;type=vector_store;backend=sql" in ddl
    conn.commit.assert_not_called()
    conn.rollback.assert_not_called()


@pytest.mark.parametrize("state", [True, None])
@pytest.mark.asyncio
async def test_pending_or_unknown_transaction_does_not_even_probe(database, state):
    conn, cursor, run = database
    conn.transaction_in_progress = state
    await run()
    cursor.execute.assert_not_called()
    conn.commit.assert_not_called()
    conn.rollback.assert_not_called()


@pytest.mark.asyncio
async def test_existing_marker_or_indirect_target_produces_no_ddl(database):
    conn, cursor, run = database
    cursor.fetchone.return_value = None
    await run()
    assert cursor.execute.call_count == 1
    conn.commit.assert_not_called()


@pytest.mark.asyncio
async def test_recheck_transaction_before_alter(database):
    conn, cursor, run = database
    type(conn).transaction_in_progress = PropertyMock(side_effect=[False, True])
    await run()
    assert cursor.execute.call_count == 1
    conn.commit.assert_not_called()


@pytest.mark.parametrize(
    "name,owner,table",
    [
        ('"Mixed.Table"', None, "Mixed.Table"),
        ('"Owner.Name"."Table.Name"', "Owner.Name", "Table.Name"),
    ],
)
@pytest.mark.asyncio
async def test_quoted_names_are_bound_without_splitting_literal_dots(
    database, name, owner, table
):
    _, cursor, run = database
    cursor.fetchone.return_value = None
    await run(name)
    assert cursor.execute.call_args.kwargs == {
        "owner": owner,
        "name": table,
        "annotation_name": "LANGCHAIN_ORACLE",
    }


@pytest.mark.parametrize("code", [942, 1031, 54, 3113])
@pytest.mark.parametrize("during_ddl", [False, True])
@pytest.mark.asyncio
async def test_expected_backfill_failures_skip_but_connection_errors_propagate(
    database, code, during_ddl
):
    conn, cursor, run = database
    error = oracledb.DatabaseError(SimpleNamespace(code=code))
    cursor.execute.side_effect = [None, error] if during_ddl else [error]
    if code == 3113:
        with pytest.raises(oracledb.DatabaseError) as raised:
            await run()
        assert raised.value is error
    else:
        await run()
    conn.commit.assert_not_called()


@pytest.mark.parametrize("code", [922, 933, 3049])
@pytest.mark.parametrize("stage", ["query", "before_clause", "annotation_clause"])
@pytest.mark.asyncio
async def test_only_annotation_ddl_syntax_errors_are_skipped(database, code, stage):
    conn, cursor, run = database
    error = oracledb.DatabaseError(
        SimpleNamespace(code=code, offset=100 if stage == "annotation_clause" else 5)
    )
    cursor.execute.side_effect = [error] if stage == "query" else [None, error]
    if stage == "annotation_clause":
        await run()
    else:
        with pytest.raises(oracledb.DatabaseError) as raised:
            await run()
        assert raised.value is error
    conn.commit.assert_not_called()
    conn.rollback.assert_not_called()
    conn.rollback.assert_not_called()


@pytest.mark.parametrize("factory", [OracleCache, OracleByteStore, OracleDocStore])
def test_existing_cache_storage_initialization_does_not_commit_pending_work(factory):
    conn = MagicMock(spec=oracledb.Connection)
    conn.transaction_in_progress = True
    module = "cache" if factory is OracleCache else "storage"
    with patch(f"langchain_oracledb.{module}._table_exists", return_value=True):
        factory(conn, "DOCS")
    conn.cursor.return_value.__enter__.return_value.execute.assert_not_called()
    conn.commit.assert_not_called()


def test_existing_chat_initialization_does_not_commit_pending_work():
    conn = MagicMock(spec=oracledb.Connection)
    conn.transaction_in_progress = True
    with patch(
        "langchain_oracledb.chat_message_histories._table_exists", return_value=True
    ):
        OracleChatMessageHistory.create_tables(conn, "DOCS", create_index=False)
    conn.cursor.return_value.__enter__.return_value.execute.assert_not_called()
    conn.commit.assert_not_called()
