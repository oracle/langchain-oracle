# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""SQL constructor options create missing indexes and preserve existing ones."""

import asyncio
import uuid

import oracledb
import pytest

from langchain_oracledb import OracleVS
from langchain_oracledb.vectorstores.oraclevs import acreate_index, create_index
from langchain_oracledb.vectorstores.utils import (
    _aget_connection,
    adrop_table_purge,
    drop_table_purge,
)

from ._backend_fixtures import native_config
from .test_backend_contract import ContractEmbeddings


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("kind", ["HNSW", "IVF"])
@pytest.mark.asyncio
async def test_sql_constructor_index_lifecycle(oracle_backend, asynchronous, kind):
    if not oracle_backend.startswith("sql"):
        pytest.skip("SQL constructor contract")
    config = native_config(oracle_backend)
    table = "LCIX_" + uuid.uuid4().hex[:12].upper()
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

    def indexes():
        with oracledb.connect(**config) as conn, conn.cursor() as cursor:
            cursor.execute(
                "SELECT index_name FROM user_indexes "
                "WHERE table_name=:name AND index_type='VECTOR' ORDER BY index_name",
                name=table,
            )
            return [row[0] for row in cursor.fetchall()]

    kwargs = dict(
        client=client,
        embedding_function=ContractEmbeddings(),
        table_name=table,
        backend="sql",
    )
    options = {
        "organization": "INMEMORY GRAPH" if kind == "HNSW" else "PARTITIONS",
        "parallel_creation": 1,
        "advanced_params": {"neighbors": 16, "efConstruction": 100}
        if kind == "HNSW"
        else {"partitions": 4},
    }
    try:
        # Omission leaves SQL tables unindexed.
        store = await OracleVS.acreate(**kwargs) if asynchronous else OracleVS(**kwargs)
        assert await asyncio.to_thread(indexes) == []
        kwargs["params"] = {"vector_index_params": options}
        store = await OracleVS.acreate(**kwargs) if asynchronous else OracleVS(**kwargs)
        original = await asyncio.to_thread(indexes)
        assert len(original) == 1
        assert original[0].startswith(kind + "_")
        # A different requested name and algorithm still preserve the existing index.
        kwargs["params"] = {
            "vector_index_params": {
                "idx_type": "IVF" if kind == "HNSW" else "HNSW",
                "idx_name": "UNUSED_" + uuid.uuid4().hex[:8].upper(),
            }
        }
        if asynchronous:
            await OracleVS.acreate(**kwargs)
        else:
            OracleVS(**kwargs)
        assert await asyncio.to_thread(indexes) == original
        # The same aliases work through the explicit index API as well.
        if asynchronous:
            async with _aget_connection(client) as conn:
                with conn.cursor() as cursor:
                    await cursor.execute(f'DROP INDEX "{original[0]}"')
            await acreate_index(client, store, options)
        else:
            if oracle_backend.endswith("-pool"):
                with client.acquire() as conn, conn.cursor() as cursor:
                    cursor.execute(f'DROP INDEX "{original[0]}"')
            else:
                with client.cursor() as cursor:
                    cursor.execute(f'DROP INDEX "{original[0]}"')
            create_index(client, store, options)
        rebuilt = await asyncio.to_thread(indexes)
        assert len(rebuilt) == 1
        assert rebuilt[0].startswith(kind + "_")
    finally:
        if asynchronous:
            await adrop_table_purge(client, table, backend="sql")
            await client.close()
        else:
            drop_table_purge(client, table, backend="sql")
            client.close()


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("quoted", [False, True], ids=["ordinary", "quoted"])
@pytest.mark.asyncio
async def test_sql_constructor_resolves_tables_and_synonyms(
    oracle_backend, asynchronous, quoted
):
    if not oracle_backend.startswith("sql"):
        pytest.skip("SQL constructor name resolution")
    from langchain_oracledb.vectorstores.oraclevs import _SQL_VECTOR_INDEX_QUERY

    config = native_config(oracle_backend)
    suffix = uuid.uuid4().hex[:12].upper()
    table_name = ("LcResolve_" if quoted else "LCRESOLVE_") + suffix
    table = f'"{table_name}"'
    synonym = "LCALIAS_" + suffix
    sequence = "LCSEQ_" + suffix
    # Use a separate connection for fixture DDL and dictionary assertions so
    # pooled and async constructors exercise their actual connection handling.
    with oracledb.connect(**config) as control, control.cursor() as cursor:
        cursor.execute("SELECT USER FROM dual")
        schema = cursor.fetchone()[0]
        OracleVS(control, ContractEmbeddings(), table, backend="sql")
        client = None
        created = []
        try:
            cursor.execute(f"CREATE SYNONYM {synonym} FOR {table}")
            created.append(("SYNONYM", synonym))
            cursor.execute(f"CREATE SEQUENCE {sequence}")
            created.append(("SEQUENCE", sequence))
            names = [table if quoted else table_name, f'"{schema}".{table}', synonym]
            for name in names:
                cursor.execute(_SQL_VECTOR_INDEX_QUERY, table_name=name)
                assert cursor.fetchone() is None
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
            original = None
            for name in names:
                kwargs = dict(
                    client=client,
                    embedding_function=ContractEmbeddings(),
                    table_name=name,
                    backend="sql",
                    params={"vector_index_params": {"idx_type": "IVF", "parallel": 1}},
                )
                if asynchronous:
                    await OracleVS.acreate(**kwargs)
                else:
                    OracleVS(**kwargs)
                cursor.execute(
                    "SELECT index_name FROM user_indexes "
                    "WHERE table_name=:name AND index_type='VECTOR'",
                    name=table_name,
                )
                indexes = cursor.fetchall()
                assert len(indexes) == 1
                if original is None:
                    original = indexes
                assert indexes == original
                cursor.execute(_SQL_VECTOR_INDEX_QUERY, table_name=name)
                assert cursor.fetchone() == original[0]
            # Context 2 accepts sequences and returns no matching vector index.
            # Context 0 must reject them, pinning the documented table namespace.
            with pytest.raises(oracledb.DatabaseError) as error:
                cursor.execute(_SQL_VECTOR_INDEX_QUERY, table_name=sequence)
                cursor.fetchone()
            assert error.value.args[0].code == 4047

        finally:
            if client is not None:
                if asynchronous:
                    await client.close()
                else:
                    client.close()
            for kind, name in reversed(created):
                cursor.execute(f"DROP {kind} {name}")
            drop_table_purge(control, table, backend="sql")
