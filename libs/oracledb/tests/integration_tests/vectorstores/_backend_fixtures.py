"""Isolated stores for the same tests on each explicitly selected backend."""

import asyncio
import os
import re
import time
import uuid
from contextlib import asynccontextmanager, contextmanager

import oracledb
import pytest

from langchain_oracledb import OracleVS
from langchain_oracledb.vectorstores._vecdb import (
    _sparse_compatibility_error,
    native_selection,
    parse_version,
)
from langchain_oracledb.vectorstores.utils import (
    DistanceStrategy,
    adrop_table_purge,
    drop_table_purge,
)


def _ddl_error_code(error):
    # Inspect chained diagnostics without parsing rendered request parameters.
    while error is not None:
        if isinstance(error, oracledb.DatabaseError) and error.args:
            return getattr(error.args[0], "code", None)
        # The REST SDK keeps server diagnostics in its structured error message.
        message = getattr(error, "error_message", None)
        if isinstance(message, str):
            match = re.search(r"\bORA-(\d{5}):", message)
            if match:
                return int(match.group(1))
        error = error.__cause__
    return None


def retry_ddl(action, code):
    """Allow five seconds for a scheduled Text sync to release test DDL."""
    for attempt in range(11):
        try:
            return action()
        except Exception as error:
            if _ddl_error_code(error) != code or attempt == 10:
                raise
            time.sleep(0.5)


async def aretry_ddl(action, code):
    """Async test-only DDL retry; cancellation propagates immediately."""
    for attempt in range(11):
        try:
            return await action()
        except Exception as error:
            if _ddl_error_code(error) != code or attempt == 10:
                raise
            await asyncio.sleep(0.5)


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


_SPARSE_STORAGE = {}


def sparse_storage_available(connection):
    """Probe actual storage support once per test endpoint, including Autonomous."""
    key = (connection.dsn, connection.username, connection.version)
    if key not in _SPARSE_STORAGE:
        name = "LCSP_" + uuid.uuid4().hex[:16].upper()
        with connection.cursor() as cursor:
            try:
                cursor.execute(f"CREATE TABLE {name} (V VECTOR(*, FLOAT32, SPARSE))")
            except oracledb.DatabaseError as error:
                if not _sparse_compatibility_error(error):
                    raise
                _SPARSE_STORAGE[key] = False
            else:
                cursor.execute(f"DROP TABLE {name} PURGE")
                _SPARSE_STORAGE[key] = True
    return _SPARSE_STORAGE[key]


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
            if parse_version(connection.version) >= (23, 26, 3):
                if not sparse_storage_available(connection):
                    pytest.skip("Database rejects sparse vector storage")


@contextmanager
def backend_store(
    backend,
    embeddings,
    *,
    table_name=None,
    distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
):
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
            client,
            embeddings,
            table,
            distance_strategy=distance_strategy,
            backend=selected,
            mutate_on_duplicate=True,
        )
        assert store.backend == selected
        yield store
    finally:
        try:
            retry_ddl(lambda: drop_table_purge(client, table, backend=selected), 54)
        finally:
            if backend != "rest":
                client.close()


@asynccontextmanager
async def async_backend_store(
    backend,
    embeddings,
    *,
    table_name=None,
    distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
):
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
            client,
            embeddings,
            table,
            distance_strategy=distance_strategy,
            backend=selected,
            mutate_on_duplicate=True,
        )
        assert store.backend == selected
        yield store
    finally:
        try:
            await aretry_ddl(
                lambda: adrop_table_purge(client, table, backend=selected), 54
            )
        finally:
            if backend != "rest":
                await client.close()
