# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""Repeated SQL long-text MERGEs keep bound LOBs and the session usable."""

import uuid

import oracledb
import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from langchain_oracledb import OracleVS

from ._backend_fixtures import native_config


@pytest.fixture
def sql_lob_table():
    config = native_config("sql")
    table = "LCLB_" + uuid.uuid4().hex[:12].upper()
    try:
        yield config, table
    finally:
        # A separate connection can clean up even if the regression kills the
        # connection under test, preserving the original failure as evidence.
        with oracledb.connect(**config) as connection, connection.cursor() as cursor:
            cursor.execute(
                "SELECT COUNT(*) FROM user_tables WHERE table_name=:name", name=table
            )
            if cursor.fetchone()[0]:
                cursor.execute(f'DROP TABLE "{table}" PURGE')


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("layout", ["long-short", "short-long", "long-long"])
@pytest.mark.parametrize(
    "long_text", ["x" * 64000, "界" * 12000], ids=["ascii", "multibyte"]
)
@pytest.mark.asyncio
async def test_long_batch_then_long_update_preserves_connection(
    sql_lob_table, asynchronous, layout, long_text
):
    config, table = sql_lob_table
    client = (
        await oracledb.connect_async(**config)
        if asynchronous
        else oracledb.connect(**config)
    )
    try:
        embedding = DeterministicFakeEmbedding(size=3)
        store = (
            await OracleVS.acreate(
                client, embedding, table, backend="sql", mutate_on_duplicate=True
            )
            if asynchronous
            else OracleVS(
                client, embedding, table, backend="sql", mutate_on_duplicate=True
            )
        )
        assert store.backend == "sql"
        texts = [long_text if part == "long" else "short" for part in layout.split("-")]
        ids = ["one", "two"]
        target = layout.split("-").index("long")

        async def upsert(values, keys):
            result = (
                await store.aadd_texts(values, ids=keys)
                if asynchronous
                else store.add_texts(values, ids=keys)
            )
            assert result == keys

        async def verify():
            docs = (
                await store.aget_by_ids(ids) if asynchronous else store.get_by_ids(ids)
            )
            assert {doc.id: doc.page_content for doc in docs} == dict(zip(ids, texts))
            with client.cursor() as cursor:
                sql = f'SELECT DBMS_LOB.GETLENGTH(TEXT) FROM "{table}"'
                if asynchronous:
                    await cursor.execute(sql)
                    lengths = await cursor.fetchall()
                    await cursor.execute("SELECT 1 FROM dual")
                    assert await cursor.fetchone() == (1,)
                else:
                    cursor.execute(sql)
                    lengths = cursor.fetchall()
                    cursor.execute("SELECT 1 FROM dual")
                    assert cursor.fetchone() == (1,)
                assert sorted(row[0] for row in lengths) == sorted(map(len, texts))

        await upsert(texts, ids)
        await verify()
        for suffix in (" updated", " updated again"):
            texts[target] = long_text + suffix
            await upsert([texts[target]], [ids[target]])
            await verify()
    finally:
        if asynchronous:
            await client.close()
        else:
            client.close()
