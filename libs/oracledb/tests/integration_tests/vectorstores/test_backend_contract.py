# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""One public behavior contract exercised against each real backend."""

import pytest
from langchain_core.embeddings import Embeddings

from langchain_oracledb import OracleVS
from langchain_oracledb.vectorstores.utils import adrop_table_purge, drop_table_purge

from ._backend_fixtures import async_backend_store, backend_store


class ContractEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [
            [1.0, 0.0, 0.0] if "oracle" in text else [0.0, 1.0, 0.0] for text in texts
        ]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.asyncio
async def test_scores_vectors_filters_and_long_text(oracle_backend, asynchronous):
    long_text = "oracle " + "document payload " * 4000
    embeddings = ContractEmbeddings()

    async def check(store):
        if asynchronous:
            ids = await store.aadd_texts(
                [long_text, "orchard"],
                [{"tenant": "a"}, {"tenant": "b"}],
                ids=["one", "two"],
            )
            hits = await store.asimilarity_search_by_vector_returning_embeddings(
                [1.0, 0.0, 0.0], k=2, filter={"tenant": {"$eq": "a"}}
            )
        else:
            ids = store.add_texts(
                [long_text, "orchard"],
                [{"tenant": "a"}, {"tenant": "b"}],
                ids=["one", "two"],
            )
            hits = store.similarity_search_by_vector_returning_embeddings(
                [1.0, 0.0, 0.0], k=2, filter={"tenant": {"$eq": "a"}}
            )
        assert ids == ["one", "two"]
        assert len(hits) == 1
        doc, score, vector = hits[0]
        assert doc.id == "one" and doc.page_content == long_text
        assert doc.metadata == {"tenant": "a"}
        assert score == pytest.approx(0)
        assert list(vector) == [1.0, 0.0, 0.0]
        if asynchronous:
            mmr = await store.amax_marginal_relevance_search("oracle", k=1, fetch_k=2)
        else:
            mmr = store.max_marginal_relevance_search("oracle", k=1, fetch_k=2)
        assert [doc.id for doc in mmr] == ["one"]
        if asynchronous:
            reopened = await OracleVS.acreate(
                store.client,
                embeddings,
                store.table_name,
                backend="auto",
                mutate_on_duplicate=True,
            )
            assert (await reopened.aget_by_ids(["one"]))[0].page_content == long_text
        else:
            reopened = OracleVS(
                store.client,
                embeddings,
                store.table_name,
                backend="auto",
                mutate_on_duplicate=True,
            )
            assert reopened.get_by_ids(["one"])[0].page_content == long_text
        assert reopened.backend == store.backend

    if asynchronous:
        async with async_backend_store(oracle_backend, embeddings) as store:
            await check(store)
    else:
        with backend_store(oracle_backend, embeddings) as store:
            await check(store)


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.asyncio
async def test_drop_api_is_repeatable(oracle_backend, asynchronous):
    embeddings = ContractEmbeddings()
    # Fixture teardown drops again. Repeating a drop of an absent table is safe.
    if asynchronous:
        async with async_backend_store(oracle_backend, embeddings) as store:
            await store.aadd_texts(["oracle"], ids=["one"])
            await adrop_table_purge(store.client, store.table_name)
            replacement = await OracleVS.acreate(
                store.client, embeddings, store.table_name, backend=store.backend
            )
            assert await replacement.aget_by_ids(["one"]) == []
            await adrop_table_purge(store.client, store.table_name)
    else:
        with backend_store(oracle_backend, embeddings) as store:
            store.add_texts(["oracle"], ids=["one"])
            drop_table_purge(store.client, store.table_name)
            replacement = OracleVS(
                store.client, embeddings, store.table_name, backend=store.backend
            )
            assert replacement.get_by_ids(["one"]) == []
            drop_table_purge(store.client, store.table_name)


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.asyncio
async def test_insert_only_preserves_existing_documents(oracle_backend, asynchronous):
    embeddings = ContractEmbeddings()
    original = "oracle " + "long text " * 4000

    async def check(store):
        store.mutate_on_duplicate = False
        if oracle_backend == "rest":
            with pytest.raises(ValueError, match="does not support insert-only"):
                if asynchronous:
                    await store.aadd_texts([original], ids=["one"])
                else:
                    store.add_texts([original], ids=["one"])
            return
        if asynchronous:
            assert await store.aadd_texts([original], ids=["one"]) == ["one"]
            assert await store.aadd_texts(["changed"], ids=["one"]) == []
            assert (await store.aget_by_ids(["one"]))[0].page_content == original
        else:
            assert store.add_texts([original], ids=["one"]) == ["one"]
            assert store.add_texts(["changed"], ids=["one"]) == []
            assert store.get_by_ids(["one"])[0].page_content == original

    if asynchronous:
        async with async_backend_store(oracle_backend, embeddings) as store:
            await check(store)
    else:
        with backend_store(oracle_backend, embeddings) as store:
            await check(store)
