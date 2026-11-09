# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""Every OracleVS metadata filter has one expectation across all backends."""

import oracledb
import pytest

from langchain_oracledb import OracleVS
from langchain_oracledb.vectorstores._vecdb_filters import (
    SQLFilterRequired,
    translate_filter,
)

from ._backend_fixtures import backend_store, native_config
from ._filter_cases import FILTER_CASES, FILTER_METADATA, FILTER_TEXTS
from .test_backend_contract import ContractEmbeddings


@pytest.fixture(scope="module")
def filter_seed(oracle_backend):
    # Seed once per backend, but reopen async connections on each test's loop.
    # No case mutates the seed rows or suppresses backend-specific failures.
    with backend_store(oracle_backend, ContractEmbeddings()) as store:
        store.add_texts(FILTER_TEXTS, FILTER_METADATA, ids=["1", "2", "3"])
        yield store


@pytest.fixture
async def filter_store(filter_seed, oracle_backend, asynchronous):
    if not asynchronous:
        yield filter_seed
        return
    client = filter_seed.client
    if oracle_backend != "rest":
        config = native_config(oracle_backend)
        client = (
            oracledb.create_pool_async(min=1, max=1, **config)
            if oracle_backend.endswith("-pool")
            else await oracledb.connect_async(**config)
        )
    try:
        store = await OracleVS.acreate(
            client,
            filter_seed.embedding_function,
            filter_seed.table_name,
            backend=filter_seed.backend,
        )
        yield store
    finally:
        if oracle_backend != "rest":
            await client.close()


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("filter_,expected", FILTER_CASES)
@pytest.mark.asyncio
async def test_filter_results(
    filter_store, asynchronous, filter_, expected, oracle_backend
):
    store = filter_store
    if oracle_backend == "rest":
        try:
            translate_filter(filter_)
        except SQLFilterRequired:
            with pytest.raises(ValueError, match="Unsupported VecDB filter"):
                if asynchronous:
                    await store.asimilarity_search("oracle", k=3, filter=filter_)
                else:
                    store.similarity_search("oracle", k=3, filter=filter_)
            return
    hits = (
        await store.asimilarity_search("oracle", k=3, filter=filter_)
        if asynchronous
        else store.similarity_search("oracle", k=3, filter=filter_)
    )
    assert {doc.id for doc in hits} == set(expected), filter_


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    "filter_,message",
    [
        ({"ss')--": "HELLOE"}, "Invalid metadata key"),
        ({"drinks": {"$neq": ["soda", "tea"]}}, "Invalid operator"),
        ({"drinks": {"$contains": "tea"}}, "Invalid operator"),
    ],
)
@pytest.mark.asyncio
async def test_filter_validation(filter_store, asynchronous, filter_, message):
    with pytest.raises(ValueError, match=message):
        if asynchronous:
            await filter_store.asimilarity_search("oracle", k=3, filter=filter_)
        else:
            filter_store.similarity_search("oracle", k=3, filter=filter_)
