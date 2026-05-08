# Copyright (c) 2026, Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from __future__ import annotations

import json
from contextlib import asynccontextmanager, contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from langchain_oracledb.embeddings.oracleai import OracleEmbeddings
from langchain_oracledb.utilities.oracleai import OracleSummary
from langchain_oracledb.vectorstores.oraclevs import OracleVS

PROXY = "http://proxy.example:80"


def _embedding_row(vector: list[float]) -> tuple[str]:
    return (json.dumps({"embed_vector": json.dumps(vector)}),)


def _proxy_bind_values(cursor: MagicMock) -> list[str | None]:
    return [
        call.kwargs.get("proxy")
        for call in cursor.execute.call_args_list
        if "utl_http.set_proxy" in call.args[0]
    ]


def test_embed_documents_clears_session_proxy_after_failure() -> None:
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value = cursor
    conn.gettype.return_value.newobject.return_value = MagicMock()
    cursor.execute.side_effect = [
        None,
        RuntimeError("embedding failed"),
        None,
    ]

    embeddings = OracleEmbeddings(
        conn=conn,
        params={"provider": "database", "model": "demo"},
        proxy=PROXY,
    )

    with patch("oracledb.defaults"):
        with pytest.raises(RuntimeError, match="embedding failed"):
            embeddings.embed_documents(["text"])

    assert _proxy_bind_values(cursor) == [PROXY, None]


def test_embed_documents_preserves_original_failure_if_proxy_cleanup_fails() -> None:
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value = cursor
    conn.gettype.return_value.newobject.return_value = MagicMock()
    cursor.execute.side_effect = [
        None,
        RuntimeError("embedding failed"),
        RuntimeError("cleanup failed"),
    ]

    embeddings = OracleEmbeddings(
        conn=conn,
        params={"provider": "database", "model": "demo"},
        proxy=PROXY,
    )

    with patch("oracledb.defaults"):
        with pytest.raises(RuntimeError, match="embedding failed"):
            embeddings.embed_documents(["text"])

    assert _proxy_bind_values(cursor) == [PROXY, None]
    cursor.close.assert_called_once()


def test_get_summary_clears_session_proxy_after_failure() -> None:
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value = cursor
    cursor.var.return_value = MagicMock()
    cursor.execute.side_effect = [
        None,
        RuntimeError("summary failed"),
        None,
    ]

    summary = OracleSummary(conn=conn, params={"provider": "database"}, proxy=PROXY)

    with pytest.raises(RuntimeError, match="summary failed"):
        summary.get_summary("text")

    assert _proxy_bind_values(cursor) == [PROXY, None]


def test_get_summary_preserves_original_failure_if_proxy_cleanup_fails() -> None:
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value = cursor
    cursor.var.return_value = MagicMock()
    cursor.execute.side_effect = [
        None,
        RuntimeError("summary failed"),
        RuntimeError("cleanup failed"),
    ]

    summary = OracleSummary(conn=conn, params={"provider": "database"}, proxy=PROXY)

    with pytest.raises(RuntimeError, match="summary failed"):
        summary.get_summary("text")

    assert _proxy_bind_values(cursor) == [PROXY, None]
    cursor.close.assert_called_once()


def test_oraclevs_add_texts_clears_session_proxy_after_failure() -> None:
    connection = MagicMock()
    cursor = MagicMock()
    connection.cursor.return_value.__enter__.return_value = cursor
    cursor.executemany.side_effect = RuntimeError("insert failed")

    store = OracleVS.__new__(OracleVS)
    store.client = object()
    store.embedding_function = OracleEmbeddings(
        conn=MagicMock(),
        params={"provider": "database", "model": "demo"},
        proxy=PROXY,
    )
    store.table_name = '"DOCS"'
    store.mutate_on_duplicate = False

    @contextmanager
    def fake_get_connection(_client):
        yield connection

    with patch(
        "langchain_oracledb.vectorstores.oraclevs._get_connection",
        fake_get_connection,
    ):
        with pytest.raises(RuntimeError, match="insert failed"):
            store.add_texts(["text"], ids=["doc-1"])

    assert _proxy_bind_values(cursor) == [PROXY, None]


def test_oraclevs_add_texts_preserves_original_failure_if_proxy_cleanup_fails() -> None:
    connection = MagicMock()
    cursor = MagicMock()
    connection.cursor.return_value.__enter__.return_value = cursor
    cursor.execute.side_effect = [None, RuntimeError("cleanup failed")]
    cursor.executemany.side_effect = RuntimeError("insert failed")

    store = OracleVS.__new__(OracleVS)
    store.client = object()
    store.embedding_function = OracleEmbeddings(
        conn=MagicMock(),
        params={"provider": "database", "model": "demo"},
        proxy=PROXY,
    )
    store.table_name = '"DOCS"'
    store.mutate_on_duplicate = False

    @contextmanager
    def fake_get_connection(_client):
        yield connection

    with patch(
        "langchain_oracledb.vectorstores.oraclevs._get_connection",
        fake_get_connection,
    ):
        with pytest.raises(RuntimeError, match="insert failed"):
            store.add_texts(["text"], ids=["doc-1"])

    assert _proxy_bind_values(cursor) == [PROXY, None]


def test_oraclevs_add_texts_fails_explicitly_if_success_cleanup_fails() -> None:
    connection = MagicMock()
    cursor = MagicMock()
    connection.cursor.return_value.__enter__.return_value = cursor
    cursor.execute.side_effect = [None, RuntimeError("cleanup failed")]
    cursor.getbatcherrors.return_value = []

    store = OracleVS.__new__(OracleVS)
    store.client = object()
    store.embedding_function = OracleEmbeddings(
        conn=MagicMock(),
        params={"provider": "database", "model": "demo"},
        proxy=PROXY,
    )
    store.table_name = '"DOCS"'
    store.mutate_on_duplicate = False

    @contextmanager
    def fake_get_connection(_client):
        yield connection

    with patch(
        "langchain_oracledb.vectorstores.oraclevs._get_connection",
        fake_get_connection,
    ):
        with pytest.raises(
            RuntimeError,
            match="proxy cleanup failed after successful add_texts",
        ):
            store.add_texts(["text"], ids=["doc-1"])

    connection.commit.assert_not_called()
    assert _proxy_bind_values(cursor) == [PROXY, None]


@pytest.mark.asyncio
async def test_oraclevs_aadd_texts_clears_session_proxy_after_failure() -> None:
    connection = MagicMock()
    connection.commit = AsyncMock()
    cursor = MagicMock()
    cursor.execute = AsyncMock()
    cursor.executemany = AsyncMock(side_effect=RuntimeError("insert failed"))
    connection.cursor.return_value.__enter__.return_value = cursor

    store = OracleVS.__new__(OracleVS)
    store.client = object()
    store.embedding_function = OracleEmbeddings(
        conn=MagicMock(),
        params={"provider": "database", "model": "demo"},
        proxy=PROXY,
    )
    store.table_name = '"DOCS"'
    store.mutate_on_duplicate = False

    @asynccontextmanager
    async def fake_aget_connection(_client):
        yield connection

    with patch(
        "langchain_oracledb.vectorstores.oraclevs._aget_connection",
        fake_aget_connection,
    ):
        with pytest.raises(RuntimeError, match="insert failed"):
            await store.aadd_texts(["text"], ids=["doc-1"])

    proxy_values = [
        call.kwargs.get("proxy")
        for call in cursor.execute.await_args_list
        if "utl_http.set_proxy" in call.args[0]
    ]
    assert proxy_values == [PROXY, None]


@pytest.mark.asyncio
async def test_oraclevs_aadd_texts_fails_explicitly_if_success_cleanup_fails() -> None:
    connection = MagicMock()
    connection.commit = AsyncMock()
    cursor = MagicMock()
    cursor.execute = AsyncMock(side_effect=[None, RuntimeError("cleanup failed")])
    cursor.executemany = AsyncMock()
    cursor.getbatcherrors.return_value = []
    connection.cursor.return_value.__enter__.return_value = cursor

    store = OracleVS.__new__(OracleVS)
    store.client = object()
    store.embedding_function = OracleEmbeddings(
        conn=MagicMock(),
        params={"provider": "database", "model": "demo"},
        proxy=PROXY,
    )
    store.table_name = '"DOCS"'
    store.mutate_on_duplicate = False

    @asynccontextmanager
    async def fake_aget_connection(_client):
        yield connection

    with patch(
        "langchain_oracledb.vectorstores.oraclevs._aget_connection",
        fake_aget_connection,
    ):
        with pytest.raises(
            RuntimeError,
            match="proxy cleanup failed after successful aadd_texts",
        ):
            await store.aadd_texts(["text"], ids=["doc-1"])

    connection.commit.assert_not_awaited()
    proxy_values = [
        call.kwargs.get("proxy")
        for call in cursor.execute.await_args_list
        if "utl_http.set_proxy" in call.args[0]
    ]
    assert proxy_values == [PROXY, None]


@pytest.mark.asyncio
async def test_aadd_texts_keeps_original_error_on_cleanup_failure() -> None:
    connection = MagicMock()
    connection.commit = AsyncMock()
    cursor = MagicMock()
    cursor.execute = AsyncMock(side_effect=[None, RuntimeError("cleanup failed")])
    cursor.executemany = AsyncMock(side_effect=RuntimeError("insert failed"))
    connection.cursor.return_value.__enter__.return_value = cursor

    store = OracleVS.__new__(OracleVS)
    store.client = object()
    store.embedding_function = OracleEmbeddings(
        conn=MagicMock(),
        params={"provider": "database", "model": "demo"},
        proxy=PROXY,
    )
    store.table_name = '"DOCS"'
    store.mutate_on_duplicate = False

    @asynccontextmanager
    async def fake_aget_connection(_client):
        yield connection

    with patch(
        "langchain_oracledb.vectorstores.oraclevs._aget_connection",
        fake_aget_connection,
    ):
        with pytest.raises(RuntimeError, match="insert failed"):
            await store.aadd_texts(["text"], ids=["doc-1"])

    proxy_values = [
        call.kwargs.get("proxy")
        for call in cursor.execute.await_args_list
        if "utl_http.set_proxy" in call.args[0]
    ]
    assert proxy_values == [PROXY, None]
