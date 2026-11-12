# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""Stored constructor index policies and reopen behavior on real backends."""

import asyncio
import json
import uuid

import oracledb
import pytest

from langchain_oracledb import OracleVS
from langchain_oracledb.vectorstores._vecdb_adapter import as_dict
from langchain_oracledb.vectorstores.utils import adrop_table_purge, drop_table_purge

from ._backend_fixtures import (
    aretry_ddl,
    check_native_backend,
    native_config,
    rest_client,
    retry_ddl,
)
from .test_backend_contract import ContractEmbeddings


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    "request_kind",
    [
        "omitted",
        "outer-empty",
        "nested-null",
        "nested-empty",
        "manual",
        "auto",
        "options",
    ],
)
@pytest.mark.asyncio
async def test_stored_constructor_policy_and_reopen(
    oracle_backend, asynchronous, request_kind, record_property
):
    table = "LCPOL_" + uuid.uuid4().hex[:12].upper()
    selected = "sql" if oracle_backend.startswith("sql") else "vecdb"
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

    options = {"idx_type": "HNSW", "accuracy": 85}
    params = {
        "outer-empty": {},
        "nested-null": {"vector_index_params": None},
        "nested-empty": {"vector_index_params": {}},
        "manual": {"vector_index_params": {"auto_index": False}},
        "auto": {"vector_index_params": {"auto_index": True}},
        "options": {"vector_index_params": options},
    }
    kwargs = dict(
        client=client,
        embedding_function=ContractEmbeddings(),
        table_name=table,
        backend=selected,
        mutate_on_duplicate=True,
    )
    if request_kind != "omitted":
        kwargs["params"] = params[request_kind]

    async def construct(**overrides):
        arguments = {**kwargs, **overrides}
        return (
            await OracleVS.acreate(**arguments)
            if asynchronous
            else OracleVS(**arguments)
        )

    def snapshot():
        found = {}
        if oracle_backend == "rest":
            description = as_dict(client.describe_vector_table(name=table))
        else:
            with (
                oracledb.connect(**config) as connection,
                connection.cursor() as cursor,
            ):
                # Effective options/build SCNs require this test-only grant:
                # GRANT SELECT ON SYS.ALL_VECTOR_INDEXES TO test_user;
                cursor.execute(
                    "SELECT index_name FROM all_indexes "
                    "WHERE table_owner=SYS_CONTEXT('USERENV','CURRENT_SCHEMA') "
                    "AND table_name=:name AND index_type='VECTOR' ORDER BY index_name",
                    name=table,
                )
                names = [row[0] for row in cursor.fetchall()]
                found["physical_indexes"] = names
                cursor.execute(
                    "SELECT COUNT(*) FROM all_views WHERE owner='SYS' "
                    "AND view_name='ALL_VECTOR_INDEXES'"
                )
                if cursor.fetchone()[0]:
                    cursor.execute(
                        "SELECT index_name,index_organization,distance_metric,"
                        "index_accuracy,build_scn FROM SYS.ALL_VECTOR_INDEXES "
                        "WHERE table_owner=SYS_CONTEXT('USERENV','CURRENT_SCHEMA') "
                        "AND table_name=:name ORDER BY index_name",
                        name=table,
                    )
                    found["effective_vector_indexes"] = cursor.fetchall()
                else:
                    found["effective_vector_indexes"] = None
                found["ddl"] = []
                for name in names:
                    cursor.execute(
                        "SELECT DBMS_METADATA.GET_DDL('INDEX', :name) FROM dual",
                        name=name,
                    )
                    ddl = cursor.fetchone()[0]
                    found["ddl"].append(ddl.read() if hasattr(ddl, "read") else ddl)
                if selected == "sql":
                    return found
                cursor.execute(
                    "SELECT DBMS_VECTOR_DATABASE.DESCRIBE_VECTOR_TABLE(:name) "
                    "FROM dual",
                    name=table,
                )
                description = as_dict(cursor.fetchone()[0])
        stored = description["index_params"]
        found["stored_options"] = (
            json.loads(stored) if isinstance(stored, str) else stored
        )
        return found

    created = False
    try:
        if selected == "sql" and request_kind in {"manual", "auto"}:
            with pytest.raises(ValueError, match="Invalid parameter: auto_index"):
                await construct()
            with (
                oracledb.connect(**config) as connection,
                connection.cursor() as cursor,
            ):
                cursor.execute(
                    "SELECT COUNT(*) FROM user_tables WHERE table_name=:name",
                    name=table,
                )
                assert cursor.fetchone()[0] == 0
            return
        created = True
        await construct()
        initial = await asyncio.to_thread(snapshot)
        record_property("initial_index_state", json.dumps(initial, default=str))
        if selected == "sql":
            expected = request_kind in {"nested-empty", "options"}
            assert bool(initial["physical_indexes"]) is expected
            if (
                request_kind == "options"
                and initial["effective_vector_indexes"] is not None
            ):
                assert initial["effective_vector_indexes"][0][3] == 85
        else:
            vector = initial["stored_options"]["vector_index_params"]
            if request_kind in {"omitted", "outer-empty", "nested-null", "manual"}:
                assert vector["auto_index"] is False
                if oracle_backend != "rest":
                    assert initial["physical_indexes"] == []
            elif request_kind == "auto":
                assert vector["auto_index"] is True
            elif request_kind == "options":
                assert vector["accuracy"] == 85
        # Omitted, empty, and explicit constructor requests must all preserve
        # existing options and physical index identities, including manual tables.
        for reopen_params in (
            None,
            {"vector_index_params": {}},
            {"vector_index_params": options},
        ):
            await construct(params=reopen_params)
            current = await asyncio.to_thread(snapshot)
            if (
                selected == "sql"
                and not initial["physical_indexes"]
                and reopen_params is not None
            ):
                assert current["physical_indexes"]
                initial = current
            else:
                assert current == initial
    finally:
        try:
            if created:
                if asynchronous:
                    await aretry_ddl(
                        lambda: adrop_table_purge(client, table, backend=selected), 54
                    )
                else:
                    retry_ddl(
                        lambda: drop_table_purge(client, table, backend=selected), 54
                    )
        finally:
            if oracle_backend != "rest":
                if asynchronous:
                    await client.close()
                else:
                    client.close()
