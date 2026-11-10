# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""Managed VecDB ingestion generates and replaces vectors inside the database."""

import os
from unittest.mock import patch

import numpy as np
import oracledb
import pytest

from langchain_oracledb import OracleVS
from langchain_oracledb.embeddings import OracleEmbeddings

from ._backend_fixtures import (
    async_backend_store,
    backend_store,
    native_config,
    rest_client,
)


@pytest.fixture
def managed_model(oracle_backend):
    if oracle_backend.startswith("sql"):
        pytest.skip("Managed embedding ingestion requires VecDB")
    rest = oracle_backend == "rest"
    client = (
        rest_client() if rest else oracledb.connect(**native_config(oracle_backend))
    )
    model = os.environ.get(
        "ORACLE_REST_EMBEDDING_MODEL" if rest else "ORACLE_EMBEDDINGS_MODEL",
        "ALLMINILM",
    )
    try:
        if not rest and tuple(map(int, client.version.split(".")[:3])) < (23, 26, 3):
            pytest.skip("Managed embedding ingestion requires native 23.26.3+")
        yield OracleEmbeddings(
            conn=client, params={"provider": "database", "model": model}
        )
    finally:
        if not rest:
            client.close()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_managed_upsert_completes_embeddings_and_replaces_duplicate(
    oracle_backend, managed_model, asynchronous
):
    original = "Oracle database backup and recovery"
    replacement = "Growing tomatoes in a sunny garden"
    # Reference vectors are deliberately computed outside the ingestion guard.
    before, after = managed_model.embed_documents([original, replacement])
    assert not np.allclose(before, after)

    async def verify(store):
        assert store._vecdb.managed_embeddings
        for text, expected in [(original, before), (replacement, after)]:
            with patch.object(
                OracleEmbeddings,
                "embed_documents",
                side_effect=AssertionError("Ingestion fetched embeddings into Python"),
            ):
                if asynchronous:
                    assert await store.aadd_texts([text], ids=["same"]) == ["same"]
                else:
                    assert store.add_texts([text], ids=["same"]) == ["same"]
            if asynchronous:
                hits = await store.asimilarity_search_by_vector_returning_embeddings(
                    expected, k=2
                )
            else:
                hits = store.similarity_search_by_vector_returning_embeddings(
                    expected, k=2
                )
            assert len(hits) == 1
            assert hits[0][0].id == "same"
            assert hits[0][0].page_content == text
            np.testing.assert_allclose(hits[0][2], expected, rtol=1e-5, atol=1e-6)

        if asynchronous:
            reopened = await OracleVS.acreate(
                store.client,
                managed_model,
                store.table_name,
                backend="vecdb",
                mutate_on_duplicate=True,
            )
            assert (await reopened.aget_by_ids(["same"]))[0].page_content == replacement
        else:
            reopened = OracleVS(
                store.client,
                managed_model,
                store.table_name,
                backend="vecdb",
                mutate_on_duplicate=True,
            )
            assert reopened.get_by_ids(["same"])[0].page_content == replacement

    if asynchronous:
        async with async_backend_store(oracle_backend, managed_model) as store:
            await verify(store)
    else:
        with backend_store(oracle_backend, managed_model) as store:
            await verify(store)
