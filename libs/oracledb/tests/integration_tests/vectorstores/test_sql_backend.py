# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""Live SQL-backend regression checks independent of VecDB auto-selection."""

import json
import os
import uuid

import oracledb
import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from langchain_oracledb import OracleVS
from langchain_oracledb.embeddings import OracleEmbeddings
from langchain_oracledb.retrievers.hybrid_search import (
    OracleHybridSearchRetriever,
    acreate_hybrid_index,
    create_hybrid_index,
)
from langchain_oracledb.vectorstores.oraclevs import acreate_index, create_index
from langchain_oracledb.vectorstores.utils import (
    DistanceStrategy,
    output_type_string_handler,
)


@pytest.fixture
def sql_database():
    if not os.environ.get("VECDB_HOST"):
        pytest.skip("Set VECDB_HOST, VECDB_USER and VECDB_PASS for live SQL checks")
    config = {
        "user": os.environ["VECDB_USER"],
        "password": os.environ["VECDB_PASS"],
        "dsn": os.environ["VECDB_HOST"],
    }
    with oracledb.connect(**config) as connection:
        model = os.environ.get("ORACLE_SQL_HYBRID_MODEL", "ALLMINILM")
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT COUNT(*) FROM user_mining_models WHERE model_name=:model",
                model=model,
            )
            if not cursor.fetchone()[0]:
                pytest.skip(
                    "Set ORACLE_SQL_HYBRID_MODEL to an installed embedding model"
                )
        table = "LCS_" + uuid.uuid4().hex[:12].upper()
        try:
            yield connection, config, model, table
        finally:
            connection.rollback()
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT COUNT(*) FROM user_tables WHERE table_name=:name",
                    name=table,
                )
                if cursor.fetchone()[0]:
                    cursor.execute(f'DROP TABLE "{table}" PURGE')


def assert_sql_store(store):
    assert store.backend == "sql"
    assert store._vecdb is None


def test_explicit_sql_vector_index_and_hybrid_modes(sql_database):
    connection, _, model_name, table = sql_database
    model = OracleEmbeddings(
        conn=connection, params={"provider": "database", "model": model_name}
    )
    store = OracleVS(
        connection,
        model,
        table,
        backend="sql",
        mutate_on_duplicate=True,
        distance_strategy=DistanceStrategy.COSINE,
    )
    assert_sql_store(store)
    store.add_texts(
        ["oracle database backup", "banana fruit orchard"],
        [{"tenant": "one"}, {"tenant": "two"}],
        ids=["one", "two"],
    )
    assert store.get_by_ids(["one"])[0].page_content == "oracle database backup"
    create_index(connection, store, {"idx_type": "HNSW", "idx_name": table + "_V"})
    assert [
        doc.id
        for doc in store.similarity_search("oracle", filter={"tenant": {"$eq": "one"}})
    ] == ["one"]
    create_hybrid_index(connection, table + "_H", vector_store=store)
    for mode in ("keyword", "semantic", "hybrid"):
        docs = OracleHybridSearchRetriever(
            vector_store=store,
            idx_name=table + "_H",
            search_mode=mode,
            return_scores=True,
            k=2,
        ).invoke("oracle")
        assert docs and all(
            doc.page_content and "score" in doc.metadata for doc in docs
        )
        if mode == "keyword":
            assert [doc.id for doc in docs] == ["one"]
    store.delete(["two"])
    assert store.get_by_ids(["two"]) == []


@pytest.mark.asyncio
async def test_explicit_sql_async_vector_index_and_hybrid_modes(sql_database):
    connection, config, model_name, table = sql_database
    # OracleEmbeddings uses a separate synchronous connection for model calls;
    # the store and index/retriever operations use the native async driver.
    model = OracleEmbeddings(
        conn=connection, params={"provider": "database", "model": model_name}
    )
    async with await oracledb.connect_async(**config) as client:
        store = await OracleVS.acreate(
            client,
            model,
            table,
            backend="sql",
            mutate_on_duplicate=True,
            distance_strategy=DistanceStrategy.COSINE,
        )
        assert_sql_store(store)
        await store.aadd_texts(
            ["oracle database backup", "banana fruit orchard"],
            [{"tenant": "one"}, {"tenant": "two"}],
            ids=["one", "two"],
        )
        assert (await store.aget_by_ids(["one"]))[
            0
        ].page_content == "oracle database backup"
        await acreate_index(
            client, store, {"idx_type": "HNSW", "idx_name": table + "_V"}
        )
        await acreate_hybrid_index(client, table + "_H", vector_store=store)
        for mode in ("keyword", "semantic", "hybrid"):
            docs = await OracleHybridSearchRetriever(
                vector_store=store,
                idx_name=table + "_H",
                search_mode=mode,
                return_scores=True,
                k=2,
            ).ainvoke("oracle")
            assert docs and all(
                doc.page_content and "score" in doc.metadata for doc in docs
            )
            if mode == "keyword":
                assert [doc.id for doc in docs] == ["one"]
        await store.adelete(["two"])
        assert await store.aget_by_ids(["two"]) == []


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.asyncio
async def test_sql_hybrid_document_batch_preserves_hit_identity(
    sql_database, asynchronous
):
    connection, config, _, table = sql_database
    store = OracleVS(
        connection,
        DeterministicFakeEmbedding(size=3),
        table,
        backend="sql",
        mutate_on_duplicate=True,
    )
    long_text = "long document " * 4000
    texts = ["first", "deleted", long_text]
    store.add_texts(
        texts, [{"tenant": "one"} for _ in texts], ids=["one", "gone", "long"]
    )
    with connection.cursor() as cursor:
        cursor.outputtypehandler = output_type_string_handler
        cursor.execute(f"SELECT ROWIDTOCHAR(ROWID), text FROM {store.table_name}")
        physical = {str(text): rowid for rowid, text in cursor.fetchall()}
    store.delete(["gone"])
    # Include a deleted hit, duplicate hits, a CLOB document, and over 32 KB of
    # bound rowids. This exercises the real JSON_TABLE join without IN-list limits.
    rowids = [
        physical[long_text],
        physical["deleted"],
        physical["first"],
        physical[long_text],
    ]
    rowids.extend([physical["first"]] * 2001)
    assert len(json.dumps(rowids)) > 32767
    hits = [
        {"rowid": rowid, "score": i + 1, "text_score": i + 2, "vector_score": i + 3}
        for i, rowid in enumerate(rowids)
    ]
    retriever = OracleHybridSearchRetriever(
        vector_store=store, idx_name=table + "_H", return_scores=True
    )
    if asynchronous:
        async with await oracledb.connect_async(**config) as client:
            async with client.cursor() as cursor:
                cursor.outputtypehandler = output_type_string_handler
                cursor.setinputsizes(rowids=oracledb.DB_TYPE_JSON)
                await cursor.execute(retriever._get_documents_query(), rowids=rowids)
                rows = await cursor.fetchall()
    else:
        with connection.cursor() as cursor:
            cursor.outputtypehandler = output_type_string_handler
            cursor.setinputsizes(rowids=oracledb.DB_TYPE_JSON)
            cursor.execute(retriever._get_documents_query(), rowids=rowids)
            rows = cursor.fetchall()
    docs = retriever._get_result_documents(rows, hits)
    positions = [i for i in range(len(hits)) if i != 1]
    assert [doc.id for doc in docs] == ["long", "one", "long"] + ["one"] * 2001
    assert [doc.page_content for doc in docs] == [long_text, "first", long_text] + [
        "first"
    ] * 2001
    assert [doc.metadata["score"] for doc in docs] == [
        hits[i]["score"] for i in positions
    ]
    assert [doc.metadata["text_score"] for doc in docs] == [
        hits[i]["text_score"] for i in positions
    ]
    assert [doc.metadata["vector_score"] for doc in docs] == [
        hits[i]["vector_score"] for i in positions
    ]
