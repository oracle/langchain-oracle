# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""SQL creation and reopen must work when VecDB rejects the session schema."""

import uuid

import oracledb
import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from langchain_oracledb import OracleVS
from langchain_oracledb.vectorstores.utils import (
    _aget_connection,
    _get_connection,
    _quote_indentifier,
    drop_table_purge,
)

from ._backend_fixtures import native_config


@pytest.mark.parametrize(
    "asynchronous,pooled", [(False, False), (False, True), (True, False), (True, True)]
)
@pytest.mark.parametrize("backend", ["auto", "sql"])
@pytest.mark.parametrize("existing", [False, True], ids=["create", "reopen"])
@pytest.mark.asyncio
async def test_sql_uses_supplied_connection_when_session_schema_differs(
    asynchronous, pooled, backend, existing
):
    config = native_config("sql")
    name = "LCRES_" + uuid.uuid4().hex[:12].upper()
    embedding = DeterministicFakeEmbedding(size=3)
    with oracledb.connect(**config) as setup:
        owner = _quote_indentifier(setup.username)
        qualified = owner + "." + _quote_indentifier(name)
        if existing:
            original = OracleVS(setup, embedding, name, backend="sql")
            original.add_texts(["original document"], ids=["seed"])
        if asynchronous:
            client = (
                oracledb.create_pool_async(min=1, max=1, **config)
                if pooled
                else await oracledb.connect_async(**config)
            )
        else:
            client = (
                oracledb.create_pool(min=1, max=1, **config)
                if pooled
                else oracledb.connect(**config)
            )
        try:
            if asynchronous:
                async with _aget_connection(client) as connection:
                    with connection.cursor() as cursor:
                        await cursor.execute("ALTER SESSION SET CURRENT_SCHEMA=SYS")
                if existing:
                    store = await OracleVS.acreate(
                        client, embedding, qualified, backend=backend
                    )
                    assert (await store.aget_by_ids(["seed"]))[
                        0
                    ].page_content == "original document"
                else:
                    store = await OracleVS.afrom_texts(
                        ["original document"],
                        embedding,
                        ids=["seed"],
                        client=client,
                        table_name=qualified,
                        backend=backend,
                    )
                await store.aadd_texts(["new document"], ids=["new"])
                assert (await store.aget_by_ids(["new"]))[
                    0
                ].page_content == "new document"
                await store.adelete(["new"])
                assert await store.aget_by_ids(["new"]) == []
            else:
                with _get_connection(client) as connection:
                    with connection.cursor() as cursor:
                        cursor.execute("ALTER SESSION SET CURRENT_SCHEMA=SYS")
                if existing:
                    store = OracleVS(client, embedding, qualified, backend=backend)
                    assert (
                        store.get_by_ids(["seed"])[0].page_content
                        == "original document"
                    )
                else:
                    store = OracleVS.from_texts(
                        ["original document"],
                        embedding,
                        ids=["seed"],
                        client=client,
                        table_name=qualified,
                        backend=backend,
                    )
                store.add_texts(["new document"], ids=["new"])
                assert store.get_by_ids(["new"])[0].page_content == "new document"
                store.delete(["new"])
                assert store.get_by_ids(["new"]) == []
            assert store.backend == "sql"
        finally:
            if asynchronous:
                async with _aget_connection(client) as connection:
                    with connection.cursor() as cursor:
                        await cursor.execute(
                            "ALTER SESSION SET CURRENT_SCHEMA=" + owner
                        )
                await client.close()
            else:
                with _get_connection(client) as connection:
                    with connection.cursor() as cursor:
                        cursor.execute("ALTER SESSION SET CURRENT_SCHEMA=" + owner)
                client.close()
            drop_table_purge(setup, name, backend="sql")


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.asyncio
async def test_sql_creation_preserves_a_dangling_synonym(asynchronous):
    config = native_config("sql")
    name = "LCALIAS_" + uuid.uuid4().hex[:12].upper()
    embedding = DeterministicFakeEmbedding(size=3)
    with oracledb.connect(**config) as setup:
        with setup.cursor() as cursor:
            cursor.execute(f"CREATE SYNONYM {name} FOR LC_MISSING_SYNONYM_TARGET")
        try:
            with pytest.raises(Exception, match="ORA-00980"):
                if asynchronous:
                    async with await oracledb.connect_async(**config) as client:
                        await OracleVS.acreate(client, embedding, name, backend="sql")
                else:
                    OracleVS(setup, embedding, name, backend="sql")
            with setup.cursor() as cursor:
                cursor.execute(
                    "SELECT table_name FROM user_synonyms WHERE synonym_name=:name",
                    name=name,
                )
                assert cursor.fetchone() == ("LC_MISSING_SYNONYM_TARGET",)
                cursor.execute(
                    "SELECT COUNT(*) FROM user_tables WHERE table_name=:name", name=name
                )
                assert cursor.fetchone() == (0,)
        finally:
            with setup.cursor() as cursor:
                cursor.execute(f"DROP SYNONYM {name}")
