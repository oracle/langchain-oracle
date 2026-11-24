# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""Live OracleVS -> contextual compression -> Oracle reranker pipeline.

Native profiles use VECDB_USER/PASS/HOST and ORACLE_EMBEDDINGS_MODEL (ALLMINILM
by default). Set ORACLE_RERANK_MODEL to an installed regression model. REST
uses ORACLE_VECDB_REST_URL and SDK authentication variables, plus
ORACLE_REST_EMBEDDING_MODEL and ORACLE_REST_RERANK_MODEL. Models are caller-owned;
each test creates and removes its own vector table.
"""

import os
import uuid

import oracledb
import pytest

from langchain_oracledb import OracleEmbeddings, OracleReranker, OracleVS
from langchain_oracledb.vectorstores.utils import (
    DistanceStrategy,
    adrop_table_purge,
    drop_table_purge,
)

from ..vectorstores._backend_fixtures import (
    check_native_backend,
    native_config,
    rest_client,
)

QUERY = "How do I restore an Oracle database backup?"
TEXTS = [
    "Bananas are a yellow fruit.",
    "Restore the Oracle database using an RMAN backup.",
    "Restore the Oracle database using an RMAN backup.",
    "Football matches are played on a pitch.",
]
IDS = ["fruit", "backup1", "backup2", "sport"]
METADATA = [
    {"topic": "backup" if i in (1, 2) else "other", "source": i} for i in range(4)
]


@pytest.fixture(params=["sql", "vecdb", "rest"])
def pipeline_config(request):
    backend = request.param
    table = "LC_PIPE_" + uuid.uuid4().hex[:16].upper()
    if backend == "rest":
        model = os.environ.get("ORACLE_REST_RERANK_MODEL")
        if not model:
            pytest.skip("Set ORACLE_REST_RERANK_MODEL for the REST pipeline")
        client = rest_client()
        embed_model = os.environ.get("ORACLE_REST_EMBEDDING_MODEL", "ALLMINILM")
        config = None
    else:
        model = os.environ.get("ORACLE_RERANK_MODEL")
        if not model:
            pytest.skip("Set ORACLE_RERANK_MODEL for the native pipeline")
        config = native_config(backend)
        check_native_backend(config, backend, table)
        client = oracledb.connect(**config)
        embed_model = os.environ.get("ORACLE_EMBEDDINGS_MODEL", "ALLMINILM")
    try:
        yield backend, client, config, table, embed_model, model
    finally:
        if backend != "rest":
            client.close()


def assert_results(documents):
    assert len(documents) == 2
    assert {doc.id for doc in documents} == {"backup1", "backup2"}
    assert all(doc.page_content == TEXTS[1] for doc in documents)
    assert all(doc.metadata["topic"] == "backup" for doc in documents)
    assert {doc.metadata["source"] for doc in documents} == {1, 2}


def test_contextual_compression_pipeline(pipeline_config):
    classic = pytest.importorskip("langchain_classic.retrievers")
    backend, client, _, table, embed_model, model = pipeline_config
    selected = "vecdb" if backend == "rest" else backend
    embeddings = OracleEmbeddings(
        conn=client, params={"provider": "database", "model": embed_model}
    )
    try:
        store = OracleVS(
            client,
            embeddings,
            table,
            backend=selected,
            mutate_on_duplicate=True,
            distance_strategy=DistanceStrategy.COSINE,
        )
        store.add_texts(TEXTS, metadatas=METADATA, ids=IDS)
        pipeline = classic.ContextualCompressionRetriever(
            base_retriever=store.as_retriever(search_kwargs={"k": 4}),
            base_compressor=OracleReranker(
                conn=client, params={"provider": "database", "model": model}, top_n=2
            ),
        )
        assert_results(pipeline.invoke(QUERY))
    finally:
        drop_table_purge(client, table, backend=selected)


@pytest.mark.asyncio
async def test_contextual_compression_pipeline_async(pipeline_config):
    classic = pytest.importorskip("langchain_classic.retrievers")
    backend, client, config, table, embed_model, model = pipeline_config
    selected = "vecdb" if backend == "rest" else backend
    async_client = (
        client if backend == "rest" else await oracledb.connect_async(**config)
    )
    embeddings = OracleEmbeddings(
        conn=client, params={"provider": "database", "model": embed_model}
    )
    try:
        store = await OracleVS.acreate(
            async_client,
            embeddings,
            table,
            backend=selected,
            mutate_on_duplicate=True,
            distance_strategy=DistanceStrategy.COSINE,
        )
        await store.aadd_texts(TEXTS, metadatas=METADATA, ids=IDS)
        pipeline = classic.ContextualCompressionRetriever(
            base_retriever=store.as_retriever(search_kwargs={"k": 4}),
            base_compressor=OracleReranker(
                conn=client, params={"provider": "database", "model": model}, top_n=2
            ),
        )
        assert_results(await pipeline.ainvoke(QUERY))
    finally:
        try:
            await adrop_table_purge(async_client, table, backend=selected)
        finally:
            if backend != "rest":
                await async_client.close()
