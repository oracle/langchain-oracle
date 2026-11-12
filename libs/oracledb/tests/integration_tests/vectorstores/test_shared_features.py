"""Portable filters, factories, and vector-backed cache behavior by backend."""

# Test setup: a DBA must run GRANT EXECUTE ON CTXSYS.CTX_DDL TO test_user;
# for the native test schema. Direct CTX_DDL.SYNC_INDEX calls make writes
# searchable before assertions; without the grant they fail with PLS-00201.
# This privilege is required by these tests, not by the library.

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, Generation

from langchain_oracledb import OracleVS
from langchain_oracledb.cache import OracleSemanticCache

from ._backend_fixtures import async_backend_store, backend_store
from .test_backend_contract import ContractEmbeddings


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.asyncio
async def test_portable_filters_and_document_factories(oracle_backend, asynchronous):
    embeddings = ContractEmbeddings()
    docs = [
        Document(
            id="a",
            page_content="oracle alpha",
            metadata={"tenant": "a", "rank": 1, "address": {"city": "NYC"}},
        ),
        Document(
            id="b",
            page_content="orchard beta",
            metadata={"tenant": "b", "rank": 2, "address": {"city": "LA"}},
        ),
        Document(
            id="c",
            page_content="oracle gamma",
            metadata={"tenant": "a", "rank": 3, "address": {"city": "NYC"}},
        ),
    ]
    cases = [
        ({"rank": {"$gte": 2}}, {"b", "c"}),
        ({"tenant": {"$in": ["a"]}}, {"a", "c"}),
        ({"$and": [{"tenant": {"$eq": "a"}}, {"rank": {"$gt": 2}}]}, {"c"}),
        ({"$or": [{"rank": {"$eq": 1}}, {"tenant": {"$eq": "b"}}]}, {"a", "b"}),
        ({"address.city": {"$eq": "NYC"}}, {"a", "c"}),
    ]

    async def check(store):
        args = dict(
            client=store.client,
            table_name=store.table_name,
            backend=store.backend,
            mutate_on_duplicate=True,
        )
        if asynchronous:
            reopened = await OracleVS.afrom_documents(docs, embeddings, **args)
            assert {
                d.id for d in await reopened.aget_by_ids(["a", "b", "c", "missing"])
            } == {"a", "b", "c"}
        else:
            reopened = OracleVS.from_documents(docs, embeddings, **args)
            assert {d.id for d in reopened.get_by_ids(["a", "b", "c", "missing"])} == {
                "a",
                "b",
                "c",
            }
        assert reopened.backend == store.backend
        for filter_, expected in cases:
            hits = (
                await reopened.asimilarity_search("oracle", k=4, filter=filter_)
                if asynchronous
                else reopened.similarity_search("oracle", k=4, filter=filter_)
            )
            assert {d.id for d in hits} == expected, filter_
        retriever = reopened.as_retriever(
            search_kwargs={"k": 1, "filter": {"tenant": {"$eq": "b"}}}
        )
        hits = (
            await retriever.ainvoke("orchard")
            if asynchronous
            else retriever.invoke("orchard")
        )
        assert [doc.id for doc in hits] == ["b"]

    if asynchronous:
        async with async_backend_store(oracle_backend, embeddings) as store:
            await check(store)
    else:
        with backend_store(oracle_backend, embeddings) as store:
            await check(store)


def test_semantic_cache_model_isolation_and_clear(oracle_backend):
    with backend_store(oracle_backend, ContractEmbeddings()) as store:
        cache = OracleSemanticCache(
            store.client,
            ContractEmbeddings(),
            store.table_name,
            distance_strategy=store.distance_strategy,
            score_threshold=0,
        )
        assert cache._vector_store.backend == store.backend
        cache.update("oracle", "model-a", [Generation(text="answer a")])
        cache.update("oracle", "model-b", [Generation(text="answer b")])
        assert cache.lookup("oracle", "model-a") == [Generation(text="answer a")]
        assert cache.lookup("oracle", "model-b") == [Generation(text="answer b")]
        procedural = ChatGeneration(
            message=AIMessage(
                content="", tool_calls=[{"id": "call", "name": "tool", "args": {}}]
            )
        )
        cache.update("oracle", "model-a", [procedural])
        assert cache.lookup("oracle", "model-a") == [Generation(text="answer a")]
        if oracle_backend == "rest":
            with pytest.raises(ValueError, match="semantic cache clearing"):
                cache.clear(prompt="oracle", llm_string="model-a")
            return
        cache.clear(prompt="oracle", llm_string="model-a")
        assert cache.lookup("oracle", "model-a") is None
        assert cache.lookup("oracle", "model-b") == [Generation(text="answer b")]
        cache.clear(llm_string="model-b")
        assert cache.lookup("oracle", "model-b") is None
        cache.update("oracle", "model-a", [Generation(text="again")])
        cache.clear()
        assert cache.lookup("oracle", "model-a") is None


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.asyncio
async def test_delete_validation_and_numeric_metadata(oracle_backend, asynchronous):
    from decimal import Decimal

    async def check(store):
        metadata = {"integer": 7, "nested": [1.25, True, None]}
        if asynchronous:
            await store.aadd_texts(["oracle"], [metadata], ids=["numeric"])
            documents = await store.aget_by_ids(["numeric"])
            hits = await store.asimilarity_search("oracle", k=1)
        else:
            store.add_texts(["oracle"], [metadata], ids=["numeric"])
            documents = store.get_by_ids(["numeric"])
            hits = store.similarity_search("oracle", k=1)
        for doc in [*documents, *hits]:
            assert type(doc.metadata["integer"]) is Decimal
            assert type(doc.metadata["nested"][0]) is Decimal
            assert doc.metadata["nested"][1] is True
            assert doc.metadata["nested"][2] is None
        for ids in (None, []):
            with pytest.raises(ValueError, match="No ids provided to delete"):
                if asynchronous:
                    await store.adelete(ids)
                else:
                    store.delete(ids)
        remaining = (
            await store.aget_by_ids(["numeric"])
            if asynchronous
            else store.get_by_ids(["numeric"])
        )
        assert len(remaining) == 1

    if asynchronous:
        async with async_backend_store(oracle_backend, ContractEmbeddings()) as store:
            await check(store)
    else:
        with backend_store(oracle_backend, ContractEmbeddings()) as store:
            await check(store)


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.asyncio
async def test_keyword_relevance_scores_and_order(oracle_backend, asynchronous):
    import uuid

    from langchain_oracledb.retrievers.text_search import (
        OracleTextSearchRetriever,
        acreate_text_index,
        create_text_index,
    )
    from langchain_oracledb.vectorstores.utils import _aget_connection, _get_connection

    if oracle_backend == "rest":
        pytest.skip("REST SDK does not expose keyword retrieval or text indexing")
    texts = ["oracle " + "fruit " * 99, "oracle " * 50 + "fruit " * 50]

    async def check(store):
        index = "LCP_" + uuid.uuid4().hex[:16].upper()
        if asynchronous:
            await store.aadd_texts(texts, ids=["a", "z"])
            await acreate_text_index(store.client, index, vector_store=store)
            async with _aget_connection(store.client) as connection:
                with connection.cursor() as cursor:
                    await cursor.execute(
                        "SELECT index_name FROM user_indexes "
                        "WHERE table_name=:name AND index_type='DOMAIN'",
                        name=store.table_name.strip('"'),
                    )
                    actual = (await cursor.fetchone())[0]
                    await cursor.execute(
                        "BEGIN CTX_DDL.SYNC_INDEX(:name); END;", name=actual
                    )
        else:
            store.add_texts(texts, ids=["a", "z"])
            create_text_index(store.client, index, vector_store=store)
            with _get_connection(store.client) as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        "SELECT index_name FROM user_indexes "
                        "WHERE table_name=:name AND index_type='DOMAIN'",
                        name=store.table_name.strip('"'),
                    )
                    actual = cursor.fetchone()[0]
                    cursor.execute("BEGIN CTX_DDL.SYNC_INDEX(:name); END;", name=actual)
        retriever = OracleTextSearchRetriever(
            vector_store=store, k=2, return_scores=True
        )
        docs = (
            await retriever.ainvoke("oracle")
            if asynchronous
            else retriever.invoke("oracle")
        )
        assert [doc.page_content for doc in docs] == [texts[1], texts[0]]
        assert docs[0].metadata["score"] > docs[1].metadata["score"] > 0
        retriever.return_scores = False
        docs = (
            await retriever.ainvoke("oracle")
            if asynchronous
            else retriever.invoke("oracle")
        )
        assert [doc.page_content for doc in docs] == [texts[1], texts[0]]
        assert all("score" not in doc.metadata for doc in docs)

    if asynchronous:
        async with async_backend_store(oracle_backend, ContractEmbeddings()) as store:
            await check(store)
    else:
        with backend_store(oracle_backend, ContractEmbeddings()) as store:
            await check(store)
