# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""Package ownership versus SQL lookalikes on configured native databases."""

import uuid

import oracledb
import pytest

from langchain_oracledb import OracleVS
from langchain_oracledb.vectorstores import _vecdb
from langchain_oracledb.vectorstores.utils import (
    _aget_connection,
    _get_connection,
    adrop_table_purge,
    drop_table_purge,
)

from ._backend_fixtures import native_config
from .test_vecdb import FixedEmbeddings


@pytest.fixture(
    params=[(False, False), (False, True), (True, False), (True, True)],
    ids=["sync", "sync-pool", "async", "async-pool"],
)
async def native(request):
    asynchronous, pooled = request.param
    config = native_config("vecdb")
    with oracledb.connect(**config) as setup:
        if asynchronous:
            client = (
                oracledb.create_pool_async(min=1, max=1, **config)
                if pooled
                else await oracledb.connect_async(**config)
            )
        else:
            client = oracledb.create_pool(min=1, max=1, **config) if pooled else setup
        try:
            yield asynchronous, setup, client
        finally:
            if asynchronous:
                await client.close()
            elif pooled:
                client.close()


async def registered(native, table):
    asynchronous, _, client = native
    if asynchronous:
        async with _aget_connection(client) as connection:
            with connection.cursor() as cursor:
                return await _vecdb._ais_vecdb_table(cursor, table)
    with _get_connection(client) as connection:
        with connection.cursor() as cursor:
            return _vecdb._is_vecdb_table(cursor, table)


async def drop(native, table):
    asynchronous, _, client = native
    if asynchronous:
        await adrop_table_purge(client, table)
    else:
        drop_table_purge(client, table)


@pytest.mark.parametrize("metadata", ["", ", METADATA JSON", ", CONTENT_METADATA JSON"])
@pytest.mark.asyncio
async def test_sql_lookalikes_are_not_vecdb_and_auto_cleanup_uses_sql(native, metadata):
    _, setup, _ = native
    table = "LCOWN_" + uuid.uuid4().hex[:12].upper()
    with setup.cursor() as cursor:
        cursor.execute(
            f'CREATE TABLE "{table}" (ID VARCHAR2(40), '
            f"DENSE_VECTOR VECTOR(3, FLOAT32){metadata})"
        )
        cursor.execute("SELECT SYS_CONTEXT('USERENV', 'CURRENT_SCHEMA') FROM SYS.DUAL")
        owner = cursor.fetchone()[0]
    try:
        assert not await registered(native, table)
        assert not await registered(native, f'"{owner}"."{table}"')
        await drop(native, table)
        with setup.cursor() as cursor:
            cursor.execute(
                "SELECT COUNT(*) FROM user_tables WHERE table_name=:name", name=table
            )
            assert cursor.fetchone()[0] == 0
    finally:
        drop_table_purge(setup, table, backend="sql")


@pytest.mark.asyncio
async def test_existing_sql_store_reopens_in_auto_and_cleans_up(native):
    asynchronous, setup, client = native
    table = "LCOWN_" + uuid.uuid4().hex[:12].upper()
    try:
        store = OracleVS(setup, FixedEmbeddings(), table, backend="sql")
        store.add_texts(["oracle"], ids=["one"])
        reopened = (
            await OracleVS.acreate(client, FixedEmbeddings(), table)
            if asynchronous
            else OracleVS(client, FixedEmbeddings(), table)
        )
        assert reopened.backend == "sql"
        docs = (
            await reopened.aget_by_ids(["one"])
            if asynchronous
            else reopened.get_by_ids(["one"])
        )
        assert docs[0].page_content == "oracle"
        await drop(native, table)
    finally:
        drop_table_purge(setup, table, backend="sql")


@pytest.mark.asyncio
async def test_package_table_is_recognized_reopened_and_dropped(native):
    asynchronous, setup, client = native
    table = "LCOWN_" + uuid.uuid4().hex[:12].upper()
    # An unsupported server must not count as a passing VecDB case.
    try:
        _vecdb.native_selection(setup, table, "vecdb")
    except ValueError as error:
        pytest.skip(str(error))
    transport = _vecdb.NativeTransport(setup, _vecdb.parse_version(setup.version))
    transport.call(
        "create_vector_table",
        name=table,
        table_params={"auto_generate_id": False},
        index_params={
            "vector_index_params": {"auto_index": False},
            "metadata_index_params": {"auto_index": False},
        },
    )
    try:
        with setup.cursor() as cursor:
            cursor.execute(
                "SELECT SYS_CONTEXT('USERENV', 'CURRENT_SCHEMA') FROM SYS.DUAL"
            )
            owner = cursor.fetchone()[0]
        assert await registered(native, table)
        assert await registered(native, f'"{owner}"."{table}"')
        reopened = (
            await OracleVS.acreate(client, FixedEmbeddings(), table)
            if asynchronous
            else OracleVS(client, FixedEmbeddings(), table)
        )
        assert reopened.backend == "vecdb"
        await drop(native, table)
        with setup.cursor() as cursor:
            cursor.execute(
                "SELECT COUNT(*) FROM user_tables WHERE table_name=:name", name=table
            )
            assert cursor.fetchone()[0] == 0
    finally:
        drop_table_purge(setup, table)
