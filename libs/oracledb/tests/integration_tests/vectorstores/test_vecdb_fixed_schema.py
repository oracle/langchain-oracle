# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""A fixed CURRENT_SCHEMA can differ from the login user, including in pools."""

import os
import uuid

import oracledb
import pytest

from langchain_oracledb import OracleVS
from langchain_oracledb.retrievers.text_search import (
    OracleTextSearchRetriever,
    create_text_index,
)
from langchain_oracledb.vectorstores.utils import (
    _quote_indentifier,
    adrop_table_purge,
    drop_table_purge,
)

from ._backend_fixtures import native_config
from .test_backend_contract import ContractEmbeddings
from .test_vecdb import metadata_index, sync_metadata_index


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("pooled", [False, True], ids=["direct", "pool"])
@pytest.mark.asyncio
async def test_fixed_schema_recognition_search_and_cleanup(asynchronous, pooled):
    owner = os.environ.get("ORACLE_VECDB_TEST_OWNER")
    owner_password = os.environ.get("ORACLE_VECDB_TEST_OWNER_PASSWORD")
    if not owner or not owner_password:
        pytest.skip("Set ORACLE_VECDB_TEST_OWNER and ORACLE_VECDB_TEST_OWNER_PASSWORD")
    config = native_config("vecdb")
    if owner == config["user"]:
        pytest.fail("Fixed-schema test requires a different login user and owner")
    owner_config = {**config, "user": owner, "password": owner_password}
    owner_config.pop("mode", None)
    table = "LCFIX_" + uuid.uuid4().hex[:12].upper()
    embeddings = ContractEmbeddings()
    client = None
    with (
        oracledb.connect(**owner_config) as owner_connection,
        oracledb.connect(**config) as login_connection,
    ):
        try:
            owner_store = OracleVS(
                owner_connection,
                embeddings,
                table,
                backend="vecdb",
                mutate_on_duplicate=True,
                params={"vector_index_params": {"idx_type": "IVF", "parallel": 1}},
            )
            owner_store.add_texts(["oracle owner"], ids=["one"])
            create_text_index(owner_connection, "REQUESTED", vector_store=owner_store)
            sync_metadata_index(
                owner_connection, metadata_index(owner_connection, owner_store)[0]
            )
            login_store = OracleVS(login_connection, embeddings, table, backend="sql")
            login_store.add_texts(["oracle login"], ids=["one"])
            with owner_connection.cursor() as cursor:
                # Both identifiers come from the fixture configuration / generated name.
                cursor.execute(
                    f'GRANT SELECT,INSERT,UPDATE,DELETE ON "{table}" TO '
                    f"{_quote_indentifier(config['user'])}"
                )
            statement = f"ALTER SESSION SET CURRENT_SCHEMA={_quote_indentifier(owner)}"
            if asynchronous:
                client = (
                    oracledb.create_pool_async(min=2, max=2, **config)
                    if pooled
                    else await oracledb.connect_async(**config)
                )
                connections = (
                    [await client.acquire(), await client.acquire()]
                    if pooled
                    else [client]
                )
                try:
                    for connection in connections:
                        with connection.cursor() as cursor:
                            await cursor.execute(statement)
                finally:
                    if pooled:
                        for connection in connections:
                            await client.release(connection)
                store = await OracleVS.acreate(
                    client, embeddings, table, mutate_on_duplicate=True
                )
                assert store.backend == "vecdb"
                assert (await store.aget_by_ids(["one"]))[
                    0
                ].page_content == "oracle owner"
                assert await store.aadd_texts(["oracle second"], ids=["two"]) == ["two"]
                assert {
                    doc.id for doc in await store.asimilarity_search("oracle", k=2)
                } == {"one", "two"}
                assert [
                    doc.id
                    for doc in await OracleTextSearchRetriever(
                        vector_store=store
                    ).ainvoke("owner")
                ] == ["one"]
                with pytest.raises(RuntimeError, match="ORA-57715"):
                    await adrop_table_purge(client, table)
            else:
                client = (
                    oracledb.create_pool(min=2, max=2, **config)
                    if pooled
                    else oracledb.connect(**config)
                )
                connections = (
                    [client.acquire(), client.acquire()] if pooled else [client]
                )
                try:
                    for connection in connections:
                        with connection.cursor() as cursor:
                            cursor.execute(statement)
                finally:
                    if pooled:
                        for connection in connections:
                            client.release(connection)
                store = OracleVS(client, embeddings, table, mutate_on_duplicate=True)
                assert store.backend == "vecdb"
                assert store.get_by_ids(["one"])[0].page_content == "oracle owner"
                assert store.add_texts(["oracle second"], ids=["two"]) == ["two"]
                assert {doc.id for doc in store.similarity_search("oracle", k=2)} == {
                    "one",
                    "two",
                }
                assert [
                    doc.id
                    for doc in OracleTextSearchRetriever(vector_store=store).invoke(
                        "owner"
                    )
                ] == ["one"]
                with pytest.raises(RuntimeError, match="ORA-57715"):
                    drop_table_purge(client, table)
            # Cleanup rejection must not drop the owner table or fall back to
            # dropping the same-named SQL table in the login user's schema.
            assert owner_store.get_by_ids(["one"])[0].page_content == "oracle owner"
            assert login_store.get_by_ids(["one"])[0].page_content == "oracle login"
        finally:
            if client is not None:
                if asynchronous:
                    await client.close()
                else:
                    client.close()
            drop_table_purge(owner_connection, table)
            drop_table_purge(login_connection, table, backend="sql")
