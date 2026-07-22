# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Unit tests for OracleByteStore / OracleDocStore (mocked connection)."""

import json
from unittest.mock import MagicMock

import oracledb
import pytest
from langchain_core.documents import Document

from langchain_oracledb.storage import OracleByteStore, OracleDocStore


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


def test_init_creates_table_blob(conn: MagicMock) -> None:
    OracleByteStore(conn, table_name="BYTES")

    ddl = conn._mock_cursor.execute.call_args.args[0]
    assert "CREATE TABLE IF NOT EXISTS" in ddl and '"BYTES"' in ddl
    assert "BLOB" in ddl


def test_init_creates_table_clob(conn: MagicMock) -> None:
    OracleDocStore(conn, table_name="DOCS")

    ddl = conn._mock_cursor.execute.call_args.args[0]
    assert "CLOB" in ddl


def test_init_rejects_missing_client() -> None:
    with pytest.raises(ValueError, match="client must be provided"):
        OracleByteStore(None)


def test_mget_preserves_order_and_missing(conn: MagicMock) -> None:
    store = OracleByteStore(conn)
    conn._mock_cursor.fetchall.return_value = [("b", b"2"), ("a", b"1")]

    assert store.mget(["a", "missing", "b"]) == [b"1", None, b"2"]
    sql, binds = conn._mock_cursor.execute.call_args.args
    assert "WHERE k IN (:k0, :k1, :k2)" in sql
    assert binds == {"k0": "a", "k1": "missing", "k2": "b"}


def test_mget_empty(conn: MagicMock) -> None:
    store = OracleByteStore(conn)
    calls_before = conn._mock_cursor.execute.call_count
    assert store.mget([]) == []
    assert conn._mock_cursor.execute.call_count == calls_before


def test_mset_merges(conn: MagicMock) -> None:
    store = OracleByteStore(conn)
    store.mset([("a", b"1"), ("b", b"2")])

    sql, rows = conn._mock_cursor.executemany.call_args.args
    assert "MERGE INTO" in sql
    assert rows == [{"k": "a", "v": b"1"}, {"k": "b", "v": b"2"}]
    conn.commit.assert_called()


def test_bytestore_rejects_non_bytes(conn: MagicMock) -> None:
    store = OracleByteStore(conn)
    with pytest.raises(TypeError, match="must be bytes"):
        store.mset([("a", "not-bytes")])  # type: ignore[list-item]


def test_docstore_round_trips_document(conn: MagicMock) -> None:
    store = OracleDocStore(conn)
    doc = Document(page_content="hello", metadata={"n": {"a": 1}}, id="d1")

    serialized = store._serialize(doc)
    restored = store._deserialize(serialized)
    assert restored == doc
    assert json.loads(serialized)["id"] == "d1"


def test_docstore_rejects_non_document(conn: MagicMock) -> None:
    store = OracleDocStore(conn)
    with pytest.raises(TypeError, match="must be Documents"):
        store.mset([("a", "text")])  # type: ignore[list-item]


def test_mdelete(conn: MagicMock) -> None:
    store = OracleByteStore(conn)
    store.mdelete(["a", "b"])

    sql, rows = conn._mock_cursor.executemany.call_args.args
    assert "DELETE FROM" in sql
    assert rows == [{"k": "a"}, {"k": "b"}]


def test_yield_keys_prefix_uses_escaped_like(conn: MagicMock) -> None:
    store = OracleByteStore(conn)
    conn._mock_cursor.__iter__ = MagicMock(return_value=iter([("a%b/1",)]))

    keys = list(store.yield_keys(prefix="a%b"))
    assert keys == ["a%b/1"]
    sql, binds = conn._mock_cursor.execute.call_args.args
    assert "LIKE :prefix ESCAPE" in sql
    assert binds == {"prefix": "a\\%b%"}


def test_yield_keys_without_prefix(conn: MagicMock) -> None:
    store = OracleByteStore(conn)
    conn._mock_cursor.__iter__ = MagicMock(return_value=iter([("k1",), ("k2",)]))

    assert list(store.yield_keys()) == ["k1", "k2"]
    sql = conn._mock_cursor.execute.call_args.args[0]
    assert "WHERE" not in sql


def test_table_name_validated() -> None:
    with pytest.raises(ValueError, match="not valid"):
        OracleByteStore(_mock_connection(), table_name='bad"name')
