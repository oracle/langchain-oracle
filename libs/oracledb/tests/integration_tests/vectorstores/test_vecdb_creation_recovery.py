# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""Verify real VecDB creation errors and table absence before SQL fallback."""

import re
import uuid
from unittest.mock import patch

import oracledb
import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from langchain_oracledb import OracleVS
from langchain_oracledb.vectorstores._vecdb import (
    AsyncNativeTransport,
    NativeTransport,
    parse_version,
)
from langchain_oracledb.vectorstores.oraclevs import _aget_connection, _get_connection
from langchain_oracledb.vectorstores.utils import adrop_table_purge, drop_table_purge

from ._backend_fixtures import (
    aretry_ddl,
    native_config,
    retry_ddl,
    sparse_storage_available,
)


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("pooled", [False, True], ids=["connection", "pool"])
@pytest.mark.parametrize("backend", ["auto", "vecdb"])
@pytest.mark.asyncio
async def test_native_creation_error_signature_and_absence_before_fallback(
    asynchronous, pooled, backend, request
):
    config = native_config("vecdb")
    with oracledb.connect(**config) as reference:
        if parse_version(reference.version) < (23, 26, 3):
            pytest.skip("This regression tests VecDB with sparse storage columns")
        supported = sparse_storage_available(reference)
        request.node.user_properties.append(("database_version", reference.version))
    name = "LCCR_" + uuid.uuid4().hex[:16].upper()
    failures = []
    original_call = NativeTransport.call
    original_acall = AsyncNativeTransport.acall

    def capture_failure(error, count):
        failures.append(error)
        detail = error.args[0]
        message = detail.message
        assert detail.code in {406, 45493}
        assert "ORA-00722" in message
        assert re.search(r"SQL VECTOR SPARSE (?:format|type)", message, re.I)
        assert count == 0, "Failed creation left a table; SQL fallback is unsafe"
        request.node.user_properties.extend(
            [
                ("creation_error_code", detail.code),
                ("creation_error_message", message),
                ("tables_immediately_after_creation_error", count),
            ]
        )

    def call(transport, operation, **params):
        try:
            return original_call(transport, operation, **params)
        except oracledb.DatabaseError as error:
            if operation == "create_vector_table":
                # This runs before OracleVS catches the error and starts fallback.
                with _get_connection(transport.client) as connection:
                    with connection.cursor() as cursor:
                        cursor.execute(
                            "SELECT COUNT(*) FROM user_tables WHERE table_name=:name",
                            name=name,
                        )
                        capture_failure(error, cursor.fetchone()[0])
            raise

    async def acall(transport, operation, **params):
        try:
            return await original_acall(transport, operation, **params)
        except oracledb.DatabaseError as error:
            if operation == "create_vector_table":
                async with _aget_connection(transport.client) as connection:
                    with connection.cursor() as cursor:
                        await cursor.execute(
                            "SELECT COUNT(*) FROM user_tables WHERE table_name=:name",
                            name=name,
                        )
                        capture_failure(error, (await cursor.fetchone())[0])
            raise

    if asynchronous:
        client = (
            oracledb.create_pool_async(**config, min=1, max=1, increment=1)
            if pooled
            else await oracledb.connect_async(**config)
        )
    else:
        client = (
            oracledb.create_pool(**config, min=1, max=1, increment=1)
            if pooled
            else oracledb.connect(**config)
        )
    store = None
    try:
        with (
            patch.object(NativeTransport, "call", call),
            patch.object(AsyncNativeTransport, "acall", acall),
        ):
            args = (client, DeterministicFakeEmbedding(size=3), name)
            if backend == "vecdb" and not supported:
                with pytest.raises(RuntimeError) as caught:
                    if asynchronous:
                        await OracleVS.acreate(*args, backend=backend)
                    else:
                        OracleVS(*args, backend=backend)
                assert len(failures) == 1
                assert caught.value.__cause__ is failures[0]
            else:
                store = (
                    await OracleVS.acreate(*args, backend=backend)
                    if asynchronous
                    else OracleVS(*args, backend=backend)
                )
                assert store.backend == ("vecdb" if supported else "sql")
                assert len(failures) == (0 if supported else 1)
                request.node.user_properties.append(("selected_backend", store.backend))
        with oracledb.connect(**config) as reference, reference.cursor() as cursor:
            cursor.execute(
                "SELECT column_name FROM user_tab_columns WHERE table_name=:name",
                name=name,
            )
            columns = {row[0] for row in cursor.fetchall()}
        if store is None:
            assert columns == set()
        elif store.backend == "sql":
            assert {"ID", "TEXT", "METADATA", "EMBEDDING"} <= columns
            assert "DENSE_VECTOR" not in columns
        else:
            assert {
                "ID",
                "DENSE_VECTOR",
                "CONTENT_METADATA",
                "SPARSE_VECTOR",
            } <= columns
    finally:
        try:
            if store is not None:
                if asynchronous:
                    await aretry_ddl(
                        lambda: adrop_table_purge(client, name, backend=store.backend),
                        54,
                    )
                else:
                    retry_ddl(
                        lambda: drop_table_purge(client, name, backend=store.backend),
                        54,
                    )
        finally:
            if asynchronous:
                await client.close()
            else:
                client.close()
