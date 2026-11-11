# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Annotation format and SQL creation paths."""

from importlib.metadata import PackageNotFoundError
from unittest.mock import AsyncMock, MagicMock, patch

import oracledb
import pytest

from langchain_oracledb import _annotations
from langchain_oracledb.cache import (
    OracleCache,
    OracleSemanticCache,
    _SemanticCacheVectorStore,
)
from langchain_oracledb.chat_message_histories import OracleChatMessageHistory
from langchain_oracledb.storage import OracleByteStore, OracleDocStore
from langchain_oracledb.vectorstores.oraclevs import (
    OracleVS,
    _acreate_table,
    _create_table,
)


@pytest.fixture(autouse=True)
def package_version():
    with patch.object(_annotations, "version", return_value="1.6.0"):
        yield


@pytest.mark.parametrize("table_type", sorted(_annotations._TABLE_TYPES))
@pytest.mark.parametrize("backend", ["sql", "vecdb"])
def test_annotation_contract(table_type, backend):
    annotations = _annotations.table_annotations(table_type, backend)
    assert annotations == {
        "LANGCHAIN_ORACLE": (
            "format=1;package=langchain-oracledb;version=1.6.0;"
            f"type={table_type};backend={backend}"
        )
    }
    assert _annotations.table_annotation_clause(table_type, backend) == (
        f"ANNOTATIONS (LANGCHAIN_ORACLE '{annotations['LANGCHAIN_ORACLE']}')"
    )


def test_missing_distribution_has_explicit_unknown_version():
    with patch.object(_annotations, "version", side_effect=PackageNotFoundError):
        assert (
            ";version=unknown;"
            in _annotations.table_annotations("kv_store")["LANGCHAIN_ORACLE"]
        )


@pytest.mark.parametrize("version", ["", "1;type=other", "1=2", "1\n2", "1\r2"])
def test_version_cannot_inject_annotation_fields(version):
    with patch.object(_annotations, "version", return_value=version):
        with pytest.raises(ValueError, match="delimiter"):
            _annotations.table_annotations("vector_store")


def test_sql_literal_escapes_package_metadata():
    with patch.object(_annotations, "version", return_value="1.6.0'quoted"):
        assert "version=1.6.0''quoted;" in _annotations.table_annotation_clause(
            "vector_store"
        )


@pytest.mark.parametrize(
    "table_type,backend", [("wrong", "sql"), ("vector_store", "wrong")]
)
def test_invalid_classification_is_rejected(table_type, backend):
    with pytest.raises(ValueError, match="Unsupported LangChain"):
        _annotations.table_annotations(table_type, backend)


def connection():
    conn = MagicMock(spec=oracledb.Connection)
    return conn, conn.cursor.return_value.__enter__.return_value


@pytest.mark.parametrize("table_type", ["vector_store", "semantic_cache"])
@pytest.mark.parametrize("exists", [False, True])
@pytest.mark.asyncio
async def test_sql_vector_creation_sync_and_async_match(table_type, exists):
    conn, cursor = connection()
    with patch(
        "langchain_oracledb.vectorstores.oraclevs._table_exists", return_value=exists
    ):
        _create_table(conn, '"Mixed.Table"', 3, table_type)
    statements = [c.args[0] for c in cursor.execute.call_args_list]
    cursor.execute = AsyncMock()
    with patch(
        "langchain_oracledb.vectorstores.oraclevs._atable_exists",
        new=AsyncMock(return_value=exists),
    ):
        await _acreate_table(conn, '"Mixed.Table"', 3, table_type)
    assert [c.args[0] for c in cursor.execute.call_args_list] == statements
    if exists:
        assert statements == []
    else:
        assert len(statements) == 1
        assert statements[0].startswith('CREATE TABLE "Mixed.Table"')
        assert f"type={table_type};backend=sql" in statements[0]
        assert statements[0].count("ANNOTATIONS (") == 1


@pytest.mark.parametrize(
    "factory,table_type",
    [
        (OracleCache, "exact_cache"),
        (OracleByteStore, "kv_store"),
        (OracleDocStore, "kv_store"),
    ],
)
def test_cache_and_storage_create_with_annotation(factory, table_type):
    conn, cursor = connection()
    module = "cache" if factory is OracleCache else "storage"
    with patch(f"langchain_oracledb.{module}._table_exists", return_value=False):
        factory(conn, table_name="DOCS")
    sql = cursor.execute.call_args.args[0]
    assert 'CREATE TABLE IF NOT EXISTS "DOCS"' in sql
    assert f"type={table_type};backend=sql" in sql
    assert cursor.execute.call_count == 1


@pytest.mark.parametrize("exists", [False, True])
def test_chat_history_annotates_only_new_tables(exists):
    conn, cursor = connection()
    with patch(
        "langchain_oracledb.chat_message_histories._table_exists", return_value=exists
    ):
        OracleChatMessageHistory.create_tables(conn, "DOCS", create_index=False)
    if exists:
        cursor.execute.assert_not_called()
    else:
        assert "type=chat_history;backend=sql" in cursor.execute.call_args.args[0]


def test_semantic_cache_uses_its_vector_store_subclass():
    with patch("langchain_oracledb.cache._SemanticCacheVectorStore") as store:
        OracleSemanticCache(object(), object())
    store.assert_called_once()
    assert "_table_type" not in store.call_args.kwargs


@pytest.mark.parametrize(
    "store_type,kind",
    [
        (OracleVS, "vector_store"),
        (_SemanticCacheVectorStore, "semantic_cache"),
    ],
)
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_store_subclass_selects_annotation_type_during_creation(
    store_type, kind, asynchronous
):
    conn = MagicMock(
        spec=oracledb.AsyncConnection if asynchronous else oracledb.Connection
    )

    def initialize(self, *args):
        self.table_name = '"DOCS"'

    module = "langchain_oracledb.vectorstores.oraclevs"
    with (
        patch.object(OracleVS, "_initialize", autospec=True, side_effect=initialize),
        patch(
            "langchain_oracledb.vectorstores._vecdb.native_selection",
            return_value=("sql", (23, 4, 0)),
        ),
        patch(
            "langchain_oracledb.vectorstores._vecdb.anative_selection",
            new=AsyncMock(return_value=("sql", (23, 4, 0))),
        ),
    ):
        if asynchronous:
            with (
                patch(
                    "langchain_oracledb.vectorstores._vecdb._ainspect_columns",
                    new=AsyncMock(return_value=set()),
                ),
                patch.object(
                    OracleVS, "aget_embedding_dimension", new=AsyncMock(return_value=3)
                ),
                patch(module + "._acreate_table", new=AsyncMock()) as create,
            ):
                store = await store_type.acreate(conn, object(), "DOCS")
                create.assert_awaited_once_with(conn, '"DOCS"', 3, kind)
        else:
            with (
                patch.object(OracleVS, "get_embedding_dimension", return_value=3),
                patch(module + "._create_table") as create,
            ):
                store = store_type(conn, object(), "DOCS")
                create.assert_called_once_with(conn, '"DOCS"', 3, kind)
    assert isinstance(store, store_type)
