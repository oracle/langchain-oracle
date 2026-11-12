# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""Creation options and explicit index changes preserve VecDB-owned metadata."""

import asyncio
import json
import uuid

import oracledb
import pytest

from langchain_oracledb import OracleVS
from langchain_oracledb.vectorstores._vecdb_adapter import as_dict
from langchain_oracledb.vectorstores.oraclevs import acreate_index, create_index
from langchain_oracledb.vectorstores.utils import (
    DistanceStrategy,
    adrop_table_purge,
    drop_table_purge,
)

from ._backend_fixtures import check_native_backend, native_config, rest_client
from .test_backend_contract import ContractEmbeddings


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("kind", ["HNSW", "IVF"])
@pytest.mark.asyncio
async def test_vecdb_index_creation_and_reconfiguration(
    oracle_backend, asynchronous, kind
):
    if oracle_backend.startswith("sql"):
        pytest.skip("VecDB configuration contract; SQL index behavior is separate")
    table = "LCIC_" + uuid.uuid4().hex[:12].upper()
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

    def configuration():
        indexes = {}
        if oracle_backend == "rest":
            details = as_dict(client.describe_vector_table(name=table))
        else:
            with oracledb.connect(**config) as conn, conn.cursor() as cursor:
                cursor.execute(
                    "SELECT DBMS_VECTOR_DATABASE.DESCRIBE_VECTOR_TABLE(:name) "
                    "FROM dual",
                    name=table,
                )
                details = as_dict(cursor.fetchone()[0])
                cursor.execute(
                    "SELECT index_name FROM all_indexes "
                    "WHERE table_owner=SYS_CONTEXT('USERENV','CURRENT_SCHEMA') "
                    "AND table_name=:name AND index_type='VECTOR'",
                    name=table,
                )
                indexes["name"] = cursor.fetchone()[0]
                cursor.execute(
                    "SELECT index_name,parameters FROM all_indexes "
                    "WHERE table_owner=SYS_CONTEXT('USERENV','CURRENT_SCHEMA') "
                    "AND table_name=:name AND ityp_owner='CTXSYS' ORDER BY index_name",
                    name=table,
                )
                indexes["metadata"] = cursor.fetchall()
        params = details["index_params"]
        if isinstance(params, str):
            params = json.loads(params)
        vector = params["vector_index_params"]
        return {
            "organization": vector.get("organization", ""),
            "accuracy": vector.get("accuracy"),
            "metric": vector.get("distance_metric"),
            **indexes,
        }

    async def assert_configuration(expected_kind, accuracy):
        # REST may submit background work. Wait here in the test, not in the API.
        for _ in range(30):
            found = await asyncio.to_thread(configuration)
            right_kind = (
                ("GRAPH" in found["organization"])
                if expected_kind == "HNSW"
                else ("PARTITIONS" in found["organization"])
            )
            if right_kind and found["accuracy"] == accuracy:
                assert found["metric"] == "COSINE"
                return found
            await asyncio.sleep(1)
        pytest.fail(f"VecDB did not apply index configuration: {found}")

    options = {
        "organization": "INMEMORY GRAPH" if kind == "HNSW" else "PARTITIONS",
        "accuracy": 85,
        "advanced_params": (
            {"neighbors": 16, "efConstruction": 100}
            if kind == "HNSW"
            else {"partitions": 4}
        ),
    }
    kwargs = {
        "client": client,
        "embedding_function": ContractEmbeddings(),
        "table_name": table,
        "backend": "vecdb",
        "distance_strategy": DistanceStrategy.COSINE,
        "mutate_on_duplicate": True,
        "params": {"vector_index_params": options},
    }
    try:
        store = await OracleVS.acreate(**kwargs) if asynchronous else OracleVS(**kwargs)
        initial = await assert_configuration(kind, 85)
        for params in (None, {}):
            result = (
                await acreate_index(client, store, params)
                if asynchronous
                else create_index(client, store, params)
            )
            assert result is None
        assert await asyncio.to_thread(configuration) == initial
        if asynchronous:
            await store.aadd_texts(["oracle index configuration"], ids=["one"])
        else:
            store.add_texts(["oracle index configuration"], ids=["one"])
        other_kind = "IVF" if kind == "HNSW" else "HNSW"
        for target in (kind, other_kind):
            # Minimal LangChain requests exercise the compatibility defaults.
            result = (
                await acreate_index(client, store, {"idx_type": target})
                if asynchronous
                else create_index(client, store, {"idx_type": target})
            )
            assert isinstance(result, dict)
            changed = await assert_configuration(target, 90)
            if oracle_backend != "rest":
                assert changed["name"] == initial["name"]
                assert changed["metadata"] == initial["metadata"]
        # Creation-only options must not undo the explicit reconfiguration.
        reopened = (
            await OracleVS.acreate(**kwargs) if asynchronous else OracleVS(**kwargs)
        )
        await assert_configuration(other_kind, 90)
        docs = (
            await reopened.aget_by_ids(["one"])
            if asynchronous
            else reopened.get_by_ids(["one"])
        )
        assert docs[0].page_content == "oracle index configuration"
    finally:
        try:
            if asynchronous:
                await adrop_table_purge(client, table, backend="vecdb")
            else:
                drop_table_purge(client, table, backend="vecdb")
        finally:
            if oracle_backend != "rest":
                if asynchronous:
                    await client.close()
                else:
                    client.close()


def test_missing_native_vector_index_uses_package_defaults(oracle_backend):
    if oracle_backend != "vecdb":
        pytest.skip("Direct native package setup for a deliberately missing index")
    from ._backend_fixtures import backend_store

    with backend_store("vecdb", ContractEmbeddings()) as store:
        with store.client.cursor() as cursor:
            output = cursor.var(oracledb.DB_TYPE_CLOB)
            cursor.execute(
                "BEGIN :output := DBMS_VECTOR_DATABASE.DROP_INDEX("
                "table_name=>:name,index_params=>JSON(:params)); END;",
                output=output,
                name=store.table_name,
                params='{"index_type":"vector"}',
            )
        assert store.describe_vecdb_index()["details"]["indexes"] == []
        result = create_index(store.client, store)
        assert isinstance(result, dict)
        assert store.describe_vecdb_index()["ready"] is True
        assert create_index(store.client, store) is None
