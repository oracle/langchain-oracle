"""Real ORDS SQL hybrid lifecycle using the native SQL plans."""

import asyncio
import os
import uuid

import pytest

from langchain_oracledb import OracleVS
from langchain_oracledb.embeddings import OracleEmbeddings
from langchain_oracledb.retrievers.hybrid_search import (
    OracleHybridSearchRetriever,
    OracleVectorizerPreference,
    acreate_hybrid_index,
    adrop_preference,
    create_hybrid_index,
)
from langchain_oracledb.vectorstores._vecdb_rest_sql import ORDSSQLConnection
from langchain_oracledb.vectorstores.utils import (
    adrop_index,
    drop_index,
    drop_table_purge,
)

from ._backend_fixtures import rest_client


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.asyncio
async def test_rest_hybrid_sql_lifecycle(oracle_backend, asynchronous):
    if oracle_backend != "rest":
        pytest.skip("ORDS SQL transport contract")
    client = rest_client()

    # Fail before creating anything if this deployment does not expose SQL.
    def probe():
        with ORDSSQLConnection(client).cursor() as cursor:
            cursor.execute("SELECT 1 AS probe FROM dual")
            assert cursor.fetchone() == (1,)

    await asyncio.to_thread(probe)
    table = "LCRH_" + uuid.uuid4().hex[:12].upper()
    embedding = OracleEmbeddings(
        conn=client,
        params={
            "provider": "database",
            "model": os.environ.get("ORACLE_REST_EMBEDDING_MODEL", "ALL_MINILM_L12_V2"),
        },
    )
    pref = None
    created = False
    try:
        store = await asyncio.to_thread(
            OracleVS, client, embedding, table, mutate_on_duplicate=True
        )
        created = True
        await store.aadd_texts(
            ["A volcano erupts molten lava.", "A bicycle has two wheels."],
            metadatas=[{"topic": "geology"}, {"topic": "transport"}],
            ids=["volcano", "bicycle"],
        )
        if asynchronous:
            pref = await OracleVectorizerPreference.acreate_preference(store)
            await acreate_hybrid_index(
                client, table + "_HVI", vectorizer_preference=pref
            )
        else:
            pref = await asyncio.to_thread(
                OracleVectorizerPreference.create_preference, store
            )
            await asyncio.to_thread(
                create_hybrid_index, client, table + "_HVI", vectorizer_preference=pref
            )
        # Repeated creation must retain the same physical index.
        await acreate_hybrid_index(client, table + "_OTHER", vectorizer_preference=pref)
        status = await store.adescribe_vecdb_index(kind="metadata")
        assert status["details"]["indexes"]
        for mode in ("keyword", "semantic", "hybrid"):
            retriever = OracleHybridSearchRetriever(
                vector_store=store,
                idx_name="IGNORED",
                search_mode=mode,
                return_scores=True,
                k=2,
            )
            docs = (
                await retriever.ainvoke("volcano")
                if asynchronous
                else await asyncio.to_thread(retriever.invoke, "volcano")
            )
            assert docs
            assert all(doc.id in {"volcano", "bicycle"} for doc in docs)
            assert all("score" in doc.metadata for doc in docs)
            if mode != "semantic":
                assert docs[0].id == "volcano"
                assert docs[0].metadata["topic"] == "geology"
            if mode == "keyword":
                assert all(doc.metadata["vector_score"] == 0 for doc in docs)
        physical = status["details"]["indexes"][0]
        index_name = f'"{physical["owner"]}"."{physical["index_name"]}"'
        if asynchronous:
            await adrop_index(client, index_name)
        else:
            await asyncio.to_thread(drop_index, client, index_name)
        assert not (await store.adescribe_vecdb_index(kind="metadata"))["details"][
            "indexes"
        ]
    finally:
        try:
            if created:
                await asyncio.to_thread(
                    drop_table_purge, client, table, backend="vecdb"
                )
        finally:
            if pref is not None:
                await adrop_preference(client, pref.preference_name)
