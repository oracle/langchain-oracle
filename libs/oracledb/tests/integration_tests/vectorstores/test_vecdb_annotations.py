# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""Live annotation persistence and CRUD across native and REST backends."""

import asyncio
import uuid
from unittest.mock import patch

import oracledb
import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from langchain_oracledb import OracleVS
from langchain_oracledb._annotations import table_annotations
from langchain_oracledb.cache import _SemanticCacheVectorStore
from langchain_oracledb.vectorstores._vecdb_adapter import as_dict
from langchain_oracledb.vectorstores.utils import adrop_table_purge, drop_table_purge

from ._backend_fixtures import check_native_backend, native_config, rest_client


@pytest.mark.parametrize(
    "store_type,kind",
    [(OracleVS, "vector_store"), (_SemanticCacheVectorStore, "semantic_cache")],
)
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.asyncio
async def test_creation_reopen_and_crud_annotations(
    oracle_backend, store_type, kind, asynchronous
):
    table = "LCCA_" + uuid.uuid4().hex[:14].upper()
    backend = "sql" if oracle_backend.startswith("sql") else "vecdb"
    config = None
    if oracle_backend == "rest":
        client = rest_client()
    else:
        config = native_config(oracle_backend)
        await asyncio.to_thread(check_native_backend, config, oracle_backend, table)
        if asynchronous:
            client = (
                oracledb.create_pool_async(min=1, max=2, increment=1, **config)
                if oracle_backend.endswith("-pool")
                else await oracledb.connect_async(**config)
            )
        else:
            client = (
                oracledb.create_pool(min=1, max=2, increment=1, **config)
                if oracle_backend.endswith("-pool")
                else oracledb.connect(**config)
            )
    expected = table_annotations(kind, backend)

    def annotation():
        if oracle_backend == "rest":
            return as_dict(client.describe_vector_table(name=table))["annotations"]
        with oracledb.connect(**config) as connection, connection.cursor() as cursor:
            cursor.execute(
                "SELECT annotation_name, annotation_value FROM user_annotations_usage "
                "WHERE object_name=:name AND object_type='TABLE' "
                "AND column_name IS NULL AND domain_name IS NULL "
                "AND annotation_name='LANGCHAIN_ORACLE'",
                name=table,
            )
            return dict(cursor.fetchall())

    embedding = DeterministicFakeEmbedding(size=3)
    try:
        if asynchronous:
            store = await store_type.acreate(
                client, embedding, table, backend=backend, mutate_on_duplicate=True
            )
            await store.aadd_texts(["oracle annotation document"], ids=["one"])
            assert (await store.aget_by_ids(["one"]))[
                0
            ].page_content == "oracle annotation document"
        else:
            store = store_type(
                client, embedding, table, backend=backend, mutate_on_duplicate=True
            )
            store.add_texts(["oracle annotation document"], ids=["one"])
            assert (
                store.get_by_ids(["one"])[0].page_content
                == "oracle annotation document"
            )
        actual = await asyncio.to_thread(annotation)
        assert {key: actual[key] for key in expected} == expected
        with patch("langchain_oracledb._annotations.version", return_value="999.0"):
            if asynchronous:
                reopened = await store_type.acreate(
                    client, embedding, table, backend=backend, mutate_on_duplicate=True
                )
                assert (await reopened.aget_by_ids(["one"]))[
                    0
                ].page_content == "oracle annotation document"
            else:
                reopened = store_type(
                    client, embedding, table, backend=backend, mutate_on_duplicate=True
                )
                assert (
                    reopened.get_by_ids(["one"])[0].page_content
                    == "oracle annotation document"
                )
        assert await asyncio.to_thread(annotation) == actual
    finally:
        try:
            if asynchronous:
                await adrop_table_purge(client, table, backend=backend)
            else:
                drop_table_purge(client, table, backend=backend)
        finally:
            if oracle_backend != "rest":
                if asynchronous:
                    await client.close()
                else:
                    client.close()
