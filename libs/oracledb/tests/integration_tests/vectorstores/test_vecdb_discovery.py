# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""Live discovery/text SQL checks without requiring native package creation."""

import array
import os
import uuid
from unittest.mock import patch

import oracledb
import pytest

from langchain_oracledb.vectorstores import _vecdb
from langchain_oracledb.vectorstores.utils import DistanceStrategy


@pytest.mark.parametrize("metadata_column", ["METADATA", "CONTENT_METADATA"])
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.asyncio
async def test_discovery_is_reused_and_external_column_change_refreshes(
    metadata_column, asynchronous
):
    if not os.environ.get("VECDB_HOST"):
        pytest.skip("Set VECDB_HOST, VECDB_USER and VECDB_PASS for live SQL checks")
    config = {
        "user": os.environ["VECDB_USER"],
        "password": os.environ["VECDB_PASS"],
        "dsn": os.environ["VECDB_HOST"],
    }
    table = "LCDISC_" + uuid.uuid4().hex[:12].upper()
    index = table + "_T"
    other_column = "CONTENT_METADATA" if metadata_column == "METADATA" else "METADATA"
    async_connection = None
    with oracledb.connect(**config) as connection:
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"CREATE TABLE {table} (ID VARCHAR2(30), "
                    f"DENSE_VECTOR VECTOR(3, FLOAT32), {metadata_column} JSON)"
                )
                cursor.setinputsizes(
                    None, oracledb.DB_TYPE_VECTOR, oracledb.DB_TYPE_JSON
                )
                cursor.execute(
                    f"INSERT INTO {table} VALUES (:1, :2, :3)",
                    [
                        "one",
                        array.array("f", [1.0, 0.0, 0.0]),
                        {
                            "__orcl_internal_content": "oracle database",
                            "text": "userlabel",
                        },
                    ],
                )
                connection.commit()
                cursor.execute(
                    f"CREATE SEARCH INDEX {index} ON {table} ({metadata_column}) "
                    "FOR JSON PARAMETERS ('SEARCH_ON TEXT SYNC (ON COMMIT)')"
                )
            if asynchronous:
                async_connection = await oracledb.connect_async(**config)
                transport = _vecdb.AsyncNativeTransport(
                    async_connection, _vecdb.parse_version(async_connection.version)
                )
            else:
                transport = _vecdb.NativeTransport(
                    connection, _vecdb.parse_version(connection.version)
                )
            backend = _vecdb.VecDBBackend(transport, table, DistanceStrategy.COSINE)

            async def invoke(method, *args):
                if asynchronous:
                    return await backend.run_async(method, *args)
                return getattr(backend, method)(*args)

            with (
                patch.object(
                    _vecdb, "_layout_query", wraps=_vecdb._layout_query
                ) as layout,
                patch.object(
                    _vecdb, "_metadata_index_query", wraps=_vecdb._metadata_index_query
                ) as indexes,
            ):
                await invoke("prepare")
                assert (layout.call_count, indexes.call_count) == (1, 1)
                for _ in range(3):
                    docs = await invoke("text_search", "oracle", 1)
                    assert [doc.page_content for doc in docs] == ["oracle database"]
                assert docs[0].metadata == {"text": "userlabel"}
                assert await invoke("text_search", "userlabel", 1) == []
                assert (layout.call_count, indexes.call_count) == (1, 1)
                # External DDL is intentionally outside the transport lifecycle.
                with connection.cursor() as cursor:
                    cursor.execute(f"DROP INDEX {index}")
                    cursor.execute(
                        f"ALTER TABLE {table} RENAME COLUMN {metadata_column} "
                        f"TO {other_column}"
                    )
                    cursor.execute(
                        f"CREATE SEARCH INDEX {index} ON {table} ({other_column}) "
                        "FOR JSON PARAMETERS ('SEARCH_ON TEXT SYNC (ON COMMIT)')"
                    )
                docs = await invoke("text_search", "oracle", 1)
                assert [doc.page_content for doc in docs] == ["oracle database"]
                assert (layout.call_count, indexes.call_count) == (2, 2)
                await invoke("text_search", "oracle", 1)
                assert (layout.call_count, indexes.call_count) == (2, 2)
        finally:
            if async_connection is not None:
                await async_connection.close()
            connection.rollback()
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT COUNT(*) FROM user_tables WHERE table_name=:name",
                    name=table,
                )
                if cursor.fetchone()[0]:
                    cursor.execute(f"DROP TABLE {table} PURGE")
