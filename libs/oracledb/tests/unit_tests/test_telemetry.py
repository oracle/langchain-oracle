# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Executed search attribution, hint preservation, and non-search exclusions."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import oracledb
import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from langchain_oracledb._telemetry import SQL_COMMENTS, tag_sql
from langchain_oracledb.embeddings import OracleEmbeddings
from langchain_oracledb.retrievers.hybrid_search import OracleHybridSearchRetriever
from langchain_oracledb.retrievers.text_search import OracleTextSearchRetriever
from langchain_oracledb.vectorstores.oraclevs import (
    OracleVS,
    _get_similarity_search_query,
)
from langchain_oracledb.vectorstores.utils import DistanceStrategy


@pytest.mark.parametrize("backend", ["sql", "vecdb"])
@pytest.mark.parametrize("operation", SQL_COMMENTS)
@pytest.mark.parametrize(
    "sql",
    [
        "SELECT :value FROM dual",
        "BEGIN :r := f(:p); END;",
        "DECLARE n NUMBER; BEGIN NULL; END;",
    ],
)
def test_tag_changes_only_a_stable_comment(sql, operation, backend):
    tagged = tag_sql(sql, operation, backend=backend)
    comment = SQL_COMMENTS[operation].replace(" */", f" backend={backend} */")
    assert tagged.replace(" " + comment, "") == sql
    assert tagged.count(comment) == 1


def test_vector_hint_remains_first_and_search_shape_unchanged():
    sql, binds = _get_similarity_search_query(
        '"Mixed.Table"',
        DistanceStrategy.COSINE,
        3,
        filter={"tenant": {"$eq": "one"}},
        return_embeddings=True,
    )
    assert (
        'SELECT /*+ VECTOR_INDEX_TRANSFORM("Mixed.Table") */ '
        + (SQL_COMMENTS["vector"].replace(" */", " backend=sql */"))
        in sql
    )
    assert binds and "FETCH APPROX FIRST 3" in sql
    assert "one" not in sql  # User filter data remains bound, not in the marker.


def test_unknown_operation_cannot_inject_sql_comments():
    with pytest.raises(ValueError):
        tag_sql("SELECT 1 FROM dual", "vector */ DELETE FROM docs --")


@pytest.mark.parametrize("mode", ["keyword", "semantic", "hybrid"])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_sql_hybrid_tags_search_but_not_rowid_fetch(mode, asynchronous):
    store = MagicMock(spec=OracleVS)
    store.client = object()
    store.table_name = '"DOCS"'
    store.embedding_function = DeterministicFakeEmbedding(size=3)
    retriever = OracleHybridSearchRetriever(
        vector_store=store, idx_name="HVI", search_mode=mode
    )
    conn, cursor = MagicMock(), MagicMock()
    conn.cursor.return_value.__enter__.return_value = cursor
    conn.cursor.return_value.__aenter__.return_value = cursor
    hits = json.dumps(
        [{"rowid": "rid", "score": 1, "text_score": 1, "vector_score": 1}]
    )
    results = [[(hits,)], [("document", {})]]
    if asynchronous:
        cursor.execute = AsyncMock()
        cursor.fetchall = AsyncMock(side_effect=results)
        with patch(
            "langchain_oracledb.retrievers.hybrid_search._aget_connection"
        ) as get:
            get.return_value.__aenter__.return_value = conn
            await retriever._aget_relevant_documents("oracle")
    else:
        cursor.fetchall.side_effect = results
        with patch(
            "langchain_oracledb.retrievers.hybrid_search._get_connection"
        ) as get:
            get.return_value.__enter__.return_value = conn
            retriever._get_relevant_documents("oracle")
    search, fetch = cursor.execute.call_args_list
    assert SQL_COMMENTS[mode].replace(" */", " backend=sql */") in search.args[0]
    assert "lc_oracledb" not in fetch.args[0]


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_sql_text_sync_async_execute_text_marker(asynchronous):
    conn, cursor = MagicMock(), MagicMock()
    conn.cursor.return_value.__enter__.return_value = cursor
    conn.cursor.return_value.__aenter__.return_value = cursor
    cursor.description = [("TEXT",)]
    retriever = OracleTextSearchRetriever.model_construct(
        client=conn, table_name='"DOCS"', column_name="TEXT", vector_store=None
    )
    if asynchronous:
        cursor.execute = AsyncMock()
        cursor.fetchall = AsyncMock(return_value=[])
        with patch("langchain_oracledb.retrievers.text_search._aget_connection") as get:
            get.return_value.__aenter__.return_value = conn
            await retriever._aget_relevant_documents("oracle")
    else:
        cursor.fetchall.return_value = []
        with patch("langchain_oracledb.retrievers.text_search._get_connection") as get:
            get.return_value.__enter__.return_value = conn
            retriever._get_relevant_documents("oracle")
    assert (
        SQL_COMMENTS["text"].replace(" */", " backend=sql */")
        in cursor.execute.call_args.args[0]
    )


def test_embedding_batch_has_one_marker_and_proxy_statements_have_none():
    conn, cursor = MagicMock(), MagicMock()
    conn.cursor.return_value = cursor
    cursor.__iter__.return_value = iter([])
    conn.gettype.return_value.element_type = oracledb.DB_TYPE_VARCHAR
    embedder = OracleEmbeddings(
        conn=conn, params={"provider": "database", "model": "M"}, proxy="http://proxy"
    )
    embedder.embed_documents(["first", "second"])
    statements = [c.args[0] for c in cursor.execute.call_args_list]
    assert len(statements) == 3
    assert ["lc_oracledb" in s for s in statements] == [False, True, False]
    assert SQL_COMMENTS["embedding"].replace(" */", " backend=sql */") in statements[1]
    assert "mode=" not in statements[1]


@pytest.mark.parametrize("backend", ["auto", "unknown", "sql */ DELETE FROM docs --"])
def test_backend_must_be_a_resolved_execution_path(backend):
    with pytest.raises(ValueError, match="backend"):
        tag_sql("SELECT 1 FROM dual", "vector", backend=backend)


@pytest.mark.parametrize("operation", SQL_COMMENTS)
@pytest.mark.parametrize("backend", ["sql", "vecdb"])
@pytest.mark.parametrize(
    "program",
    [
        "devrel-developerhub-langchain",
        "DEVREL-DEVELOPERHUB-LANGCHAIN",
        "devrel-developerhub-langchain ",
        "devrel-developerhub-langchain-other",
        "other-program",
        "",
        None,
        "*/ DELETE FROM docs --",
    ],
)
def test_only_exact_allowlisted_program_adds_source(operation, backend, program):
    with patch(
        "langchain_oracledb._telemetry.oracledb",
        SimpleNamespace(defaults=SimpleNamespace(program=program)),
    ):
        sql = tag_sql("SELECT :value FROM dual", operation, backend=backend)
    fields = f" backend={backend}"
    if program == "devrel-developerhub-langchain":
        fields += " source=devrel-developerhub-langchain"
    comment = SQL_COMMENTS[operation].replace(" */", fields + " */")
    assert sql == "SELECT " + comment + " :value FROM dual"


def test_driver_without_program_property_remains_supported():
    with patch(
        "langchain_oracledb._telemetry.oracledb",
        SimpleNamespace(defaults=SimpleNamespace()),
    ):
        assert tag_sql("SELECT 1 FROM dual", "vector") == (
            "SELECT /* lc_oracledb mode=vector backend=sql */ 1 FROM dual"
        )


def test_program_is_checked_when_the_statement_is_generated(monkeypatch):
    monkeypatch.setattr(oracledb.defaults, "program", "devrel-developerhub-langchain")
    sql, _ = _get_similarity_search_query('"DOCS"', DistanceStrategy.COSINE, 3)
    assert "backend=sql source=devrel-developerhub-langchain */" in sql
    monkeypatch.setattr(oracledb.defaults, "program", "other-program")
    sql, _ = _get_similarity_search_query('"DOCS"', DistanceStrategy.COSINE, 3)
    assert "backend=sql */" in sql
    assert "source=" not in sql
