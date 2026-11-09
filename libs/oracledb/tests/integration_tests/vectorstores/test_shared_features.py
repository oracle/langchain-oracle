"""Portable filters, factories, and vector-backed cache behavior by backend."""

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
