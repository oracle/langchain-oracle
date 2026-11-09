"""Isolated stores for the same tests on each explicitly selected backend."""

import asyncio
import os
import uuid
from contextlib import asynccontextmanager, contextmanager

import oracledb
import pytest

from langchain_oracledb import OracleVS
from langchain_oracledb.vectorstores._vecdb import native_selection
from langchain_oracledb.vectorstores.utils import adrop_table_purge, drop_table_purge


def native_config(backend="sql"):
    if backend.startswith("vecdb") and os.environ.get("ORACLE_VECDB_NATIVE_DSN"):
        keys = (
            "ORACLE_VECDB_USERNAME",
            "ORACLE_VECDB_PASSWORD",
            "ORACLE_VECDB_NATIVE_DSN",
        )
    else:
        keys = ("VECDB_USER", "VECDB_PASS", "VECDB_HOST")
    if not all(os.environ.get(key) for key in keys):
        pytest.skip("Missing native configuration: " + ", ".join(keys))
    config = dict(zip(("user", "password", "dsn"), (os.environ[k] for k in keys)))
    if config["user"].upper() == "SYS":
        config["mode"] = oracledb.AUTH_MODE_SYSDBA
    return config


def rest_client():
    if not os.environ.get("ORACLE_VECDB_REST_URL"):
        pytest.skip("Missing ORACLE_VECDB_REST_URL")
    sdk = pytest.importorskip("oracle_vecdb")
    auth = {}
    if os.environ.get("ORACLE_VECDB_ACCESS_TOKEN"):
        auth["access_token"] = os.environ["ORACLE_VECDB_ACCESS_TOKEN"]
    else:
        keys = ("ORACLE_VECDB_USERNAME", "ORACLE_VECDB_PASSWORD")
        if not all(os.environ.get(key) for key in keys):
            pytest.skip("Missing REST credentials")
        auth.update(username=os.environ[keys[0]], password=os.environ[keys[1]])
    return sdk.OracleVecDB(
        sdk.Configuration(
            rest_url=os.environ["ORACLE_VECDB_REST_URL"],
            ssl_ca_cert=os.environ.get("ORACLE_VECDB_CA_CERT"),
            timeout=30,
            **auth,
        )
    )


def check_native_backend(config, backend, table):
    if backend.startswith("vecdb"):
        with oracledb.connect(**config) as connection:
            try:
                native_selection(connection, table, "vecdb")
            except ValueError as exc:
                if any(
                    text in str(exc)
                    for text in (
                        "VecDB requires",
                        "DBMS_VECTOR_DATABASE.SEARCH is unavailable",
                    )
                ):
                    pytest.skip("Native VecDB unavailable: " + str(exc))
                raise


@contextmanager
def backend_store(backend, embeddings, *, table_name=None):
    table = (
        table_name
        if table_name is not None
        else "LCMX_" + uuid.uuid4().hex[:16].upper()
    )
    selected = "sql" if backend.startswith("sql") else "vecdb"
    if backend == "rest":
        client = rest_client()
    else:
        config = native_config(backend)
        check_native_backend(config, backend, table)
        client = (
            oracledb.create_pool(min=1, max=2, increment=1, **config)
            if backend.endswith("-pool")
            else oracledb.connect(**config)
        )
    try:
        store = OracleVS(
            client, embeddings, table, backend=selected, mutate_on_duplicate=True
        )
        assert store.backend == selected
        yield store
    finally:
        try:
            drop_table_purge(client, table, backend=selected)
        finally:
            if backend != "rest":
                client.close()


@asynccontextmanager
async def async_backend_store(backend, embeddings, *, table_name=None):
    table = (
        table_name
        if table_name is not None
        else "LCMX_" + uuid.uuid4().hex[:16].upper()
    )
    selected = "sql" if backend.startswith("sql") else "vecdb"
    if backend == "rest":
        client = rest_client()
    else:
        config = native_config(backend)
        await asyncio.to_thread(check_native_backend, config, backend, table)
        client = (
            oracledb.create_pool_async(min=1, max=2, increment=1, **config)
            if backend.endswith("-pool")
            else await oracledb.connect_async(**config)
        )
    try:
        store = await OracleVS.acreate(
            client, embeddings, table, backend=selected, mutate_on_duplicate=True
        )
        assert store.backend == selected
        yield store
    finally:
        try:
            await adrop_table_purge(client, table, backend=selected)
        finally:
            if backend != "rest":
                await client.close()
