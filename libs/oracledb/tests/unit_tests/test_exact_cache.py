# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Unit tests for the exact-match OracleCache (mocked connection)."""

from typing import Any
from unittest.mock import MagicMock

import oracledb
import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, Generation

from langchain_oracledb.cache import (
    OracleCache,
    _cache_entry_id,
    _dumps_generations,
)


def _mock_connection() -> MagicMock:
    # spec makes isinstance(conn, oracledb.Connection) pass in _get_connection
    conn = MagicMock(spec=oracledb.Connection)
    cursor = MagicMock()
    cursor_cm = MagicMock()
    cursor_cm.__enter__ = MagicMock(return_value=cursor)
    cursor_cm.__exit__ = MagicMock(return_value=False)
    conn.cursor = MagicMock(return_value=cursor_cm)
    conn._mock_cursor = cursor
    return conn


@pytest.fixture
def conn() -> MagicMock:
    return _mock_connection()


def test_init_creates_table(conn: MagicMock) -> None:
    OracleCache(client=conn, table_name="MY_CACHE")

    ddl = conn._mock_cursor.execute.call_args.args[0]
    assert "CREATE TABLE IF NOT EXISTS" in ddl
    assert '"MY_CACHE"' in ddl
    conn.commit.assert_called()


def test_init_rejects_missing_client() -> None:
    with pytest.raises(ValueError, match="client must be provided"):
        OracleCache(client=None)


def test_lookup_miss_returns_none(conn: MagicMock) -> None:
    cache = OracleCache(client=conn)
    conn._mock_cursor.fetchone.return_value = None

    assert cache.lookup("p", "l") is None
    sql, binds = conn._mock_cursor.execute.call_args.args
    assert "WHERE id = :id" in sql
    assert binds == {"id": _cache_entry_id("p", "l")}


def test_lookup_hit_deserializes_and_resets_ids(conn: MagicMock) -> None:
    cache = OracleCache(client=conn)
    gen = ChatGeneration(message=AIMessage(content="hi", id="lc_run--x"))
    conn._mock_cursor.fetchone.return_value = (_dumps_generations([gen]),)

    hit = cache.lookup("p", "l")
    assert hit is not None
    assert hit[0].message.id is None  # type: ignore[union-attr]
    assert hit[0].message.content == "hi"  # type: ignore[union-attr]


def test_lookup_reads_lob(conn: MagicMock) -> None:
    cache = OracleCache(client=conn)
    lob = MagicMock()
    lob.read.return_value = _dumps_generations([Generation(text="v")])
    conn._mock_cursor.fetchone.return_value = (lob,)

    hit = cache.lookup("p", "l")
    assert hit is not None and hit[0].text == "v"


def test_update_merges_with_hashed_binds(conn: MagicMock) -> None:
    cache = OracleCache(client=conn)
    cache.update("p", "l", [Generation(text="v")])

    sql, binds = conn._mock_cursor.execute.call_args.args
    assert "MERGE INTO" in sql
    assert binds["id"] == _cache_entry_id("p", "l")
    assert "generations" in binds


def test_update_skips_tool_calls(conn: MagicMock) -> None:
    cache = OracleCache(client=conn)
    calls_before = conn._mock_cursor.execute.call_count
    gen = ChatGeneration(
        message=AIMessage(
            content="",
            tool_calls=[{"name": "t", "args": {}, "id": "x", "type": "tool_call"}],
        )
    )
    cache.update("p", "l", [gen])
    assert conn._mock_cursor.execute.call_count == calls_before


def test_clear_filters(conn: MagicMock) -> None:
    cache = OracleCache(client=conn)
    cache.clear(llm_string="l")

    sql, binds = conn._mock_cursor.execute.call_args.args
    assert "DELETE FROM" in sql and "llm_string_hash = :llm_hash" in sql

    cache.clear()
    sql2 = conn._mock_cursor.execute.call_args.args[0]
    assert "WHERE" not in sql2


def test_clear_rejects_unknown_filters(conn: MagicMock) -> None:
    cache = OracleCache(client=conn)
    with pytest.raises(ValueError, match="Unsupported clear filters"):
        cache.clear(nope=1)


def test_table_name_validated() -> None:
    conn: Any = _mock_connection()
    with pytest.raises(ValueError, match="not valid"):
        OracleCache(client=conn, table_name='bad"name')
