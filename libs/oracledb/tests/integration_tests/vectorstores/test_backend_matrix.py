# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""Live existing-table and backend-selection matrix on the native SQL database.

SQL-created VecDB-shaped fixtures are unregistered and cannot select VecDB.
True VecDB tables are covered separately against the VecDB service.
"""

import array
import hashlib
import json
import os
import uuid

import oracledb
import pytest
from langchain_core.embeddings import Embeddings

from langchain_oracledb import OracleVS
from langchain_oracledb.vectorstores.oraclevs import INTERNAL_ID_KEY
from langchain_oracledb.vectorstores.utils import DistanceStrategy


class MatrixEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [
            [1.0, 0.0, 0.0] if "oracle" in text else [0.0, 1.0, 0.0] for text in texts
        ]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


ORIGINS = (
    "library_sql",
    "external_sql",
    "quoted_sql",
    "legacy_vecdb_shape",
    "current_vecdb_shape",
    "unrelated",
    "ambiguous",
)
SQL_ORIGINS = {"library_sql", "external_sql", "quoted_sql"}


@pytest.fixture(scope="module")
def database_matrix():
    if not os.environ.get("VECDB_HOST"):
        pytest.skip("Set VECDB_HOST, VECDB_USER and VECDB_PASS for the live matrix")
    config = {
        "user": os.environ["VECDB_USER"],
        "password": os.environ["VECDB_PASS"],
        "dsn": os.environ["VECDB_HOST"],
    }
    if config["user"].upper() == "SYS":
        config["mode"] = oracledb.AUTH_MODE_SYSDBA
    suffix = uuid.uuid4().hex[:10].upper()
    tables = {origin: f"LCM_{suffix}_{n}" for n, origin in enumerate(ORIGINS)}
    tables["quoted_sql"] = f"LCM_{suffix}_Mixed"
    with oracledb.connect(**config) as connection:
        with connection.cursor() as cursor:
            version, compatible = cursor.var(str), cursor.var(str)
            cursor.execute(
                "BEGIN DBMS_UTILITY.DB_VERSION(:version,:compatible); END;",
                version=version,
                compatible=compatible,
            )
            if tuple(map(int, compatible.getvalue().split(".")[:2])) >= (23, 6):
                pytest.skip("This matrix targets SQL fallback with COMPATIBLE <23.6")
            if tuple(map(int, connection.version.split(".")[:3])) < (23, 26, 3):
                pytest.skip("This matrix targets the current VecDB generation")
        try:
            for origin, name in tables.items():
                quoted = f'"{name}"'
                with connection.cursor() as cursor:
                    if origin == "library_sql":
                        store = OracleVS(
                            connection, MatrixEmbeddings(), name, backend="sql"
                        )
                        store.add_texts(
                            ["oracle existing seed"], [{"tenant": "base"}], ids=["seed"]
                        )
                        continue
                    if origin in SQL_ORIGINS:
                        cursor.execute(
                            f"CREATE TABLE {quoted} (ID RAW(16) PRIMARY KEY, "
                            "TEXT CLOB, METADATA JSON, EMBEDDING VECTOR(3,FLOAT32))"
                        )
                        cursor.setinputsizes(
                            None, None, oracledb.DB_TYPE_JSON, oracledb.DB_TYPE_VECTOR
                        )
                        cursor.execute(
                            f"INSERT INTO {quoted} VALUES (:1,:2,:3,:4)",
                            [
                                bytes.fromhex(hashlib.sha256(b"seed").hexdigest()[:16]),
                                "oracle existing seed",
                                {INTERNAL_ID_KEY: "seed", "tenant": "base"},
                                array.array("f", [1, 0, 0]),
                            ],
                        )
                    elif origin == "unrelated":
                        cursor.execute(
                            f"CREATE TABLE {quoted} (ID NUMBER PRIMARY KEY, "
                            "PAYLOAD VARCHAR2(100))"
                        )
                        cursor.execute(f"INSERT INTO {quoted} VALUES (1,'unchanged')")
                    else:
                        metadata = (
                            "METADATA"
                            if origin == "legacy_vecdb_shape"
                            else "CONTENT_METADATA"
                        )
                        extra = ", METADATA JSON" if origin == "ambiguous" else ""
                        cursor.execute(
                            f"CREATE TABLE {quoted} (ID VARCHAR2(64) PRIMARY KEY, "
                            f"DENSE_VECTOR VECTOR(3,FLOAT32), {metadata} JSON{extra})"
                        )
                        cursor.setinputsizes(
                            None, oracledb.DB_TYPE_VECTOR, oracledb.DB_TYPE_JSON
                        )
                        cursor.execute(
                            f"INSERT INTO {quoted} (ID,DENSE_VECTOR,{metadata}) "
                            "VALUES (:1,:2,:3)",
                            [
                                "seed",
                                array.array("f", [1, 0, 0]),
                                {"text": "oracle existing seed", "tenant": "base"},
                            ],
                        )
                connection.commit()
            pool = oracledb.create_pool(**config, min=1, max=2, increment=1)
            try:
                baseline = {
                    name: table_snapshot(connection, name) for name in tables.values()
                }
                yield connection, pool, config, tables, baseline
            finally:
                pool.close()
        finally:
            connection.rollback()
            with connection.cursor() as cursor:
                for name in tables.values():
                    cursor.execute(
                        "SELECT COUNT(*) FROM user_tables WHERE table_name=:name",
                        name=name,
                    )
                    if cursor.fetchone()[0]:
                        cursor.execute(f'DROP TABLE "{name}" PURGE')


def expected_error(origin, backend):
    if origin in SQL_ORIGINS:
        return "ordinary OracleVS" if backend == "vecdb" else None
    return "Incompatible"


def table_snapshot(connection, name):
    def normalize(value):
        if hasattr(value, "read"):
            return value.read()
        if isinstance(value, array.array):
            return value.tolist()
        if isinstance(value, bytes):
            return value.hex()
        return value

    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT column_name,data_type FROM user_tab_columns "
            "WHERE table_name=:name ORDER BY column_id",
            name=name,
        )
        columns = cursor.fetchall()
        cursor.execute(f'SELECT * FROM "{name}"')
        data = [[normalize(value) for value in row] for row in cursor.fetchall()]
    return json.dumps([columns, data], sort_keys=True)


def assert_preserved(connection, name, baseline):
    assert table_snapshot(connection, name) == baseline


@pytest.mark.parametrize("origin", ORIGINS)
@pytest.mark.parametrize("backend", ["auto", "sql", "vecdb"])
@pytest.mark.parametrize("pooled", [False, True])
def test_existing_table_sync(database_matrix, origin, backend, pooled):
    connection, pool, _, tables, baseline = database_matrix
    name = tables[origin]
    client = pool if pooled else connection
    error = expected_error(origin, backend)
    if error:
        with pytest.raises(ValueError, match=error):
            OracleVS(client, MatrixEmbeddings(), name, backend=backend)
        assert_preserved(connection, name, baseline[name])
        return
    store = OracleVS(
        client,
        MatrixEmbeddings(),
        name,
        backend=backend,
        mutate_on_duplicate=True,
        distance_strategy=DistanceStrategy.COSINE,
    )
    assert store.backend == "sql" and store._vecdb is None
    assert store.get_by_ids(["seed"])[0].page_content == "oracle existing seed"
    record = uuid.uuid4().hex
    try:
        store.add_texts(["oracle matrix"], [{"tenant": record}], ids=[record])
        assert store.get_by_ids([record])[0].metadata == {"tenant": record}
        assert (
            store.similarity_search("oracle", filter={"tenant": {"$eq": record}})[0].id
            == record
        )
        store.add_texts(["oracle updated"], [{"tenant": record}], ids=[record])
        assert store.get_by_ids([record])[0].page_content == "oracle updated"
    finally:
        store.delete([record])
    assert_preserved(connection, name, baseline[name])


@pytest.mark.parametrize("origin", ORIGINS)
@pytest.mark.parametrize("backend", ["auto", "sql", "vecdb"])
@pytest.mark.parametrize("pooled", [False, True])
@pytest.mark.asyncio
async def test_existing_table_async(database_matrix, origin, backend, pooled):
    connection, _, config, tables, baseline = database_matrix
    name = tables[origin]
    client = (
        oracledb.create_pool_async(**config, min=1, max=2, increment=1)
        if pooled
        else await oracledb.connect_async(**config)
    )
    try:
        error = expected_error(origin, backend)
        if error:
            with pytest.raises(ValueError, match=error):
                await OracleVS.acreate(
                    client, MatrixEmbeddings(), name, backend=backend
                )
            assert_preserved(connection, name, baseline[name])
            return
        store = await OracleVS.acreate(
            client,
            MatrixEmbeddings(),
            name,
            backend=backend,
            mutate_on_duplicate=True,
            distance_strategy=DistanceStrategy.COSINE,
        )
        assert store.backend == "sql" and store._vecdb is None
        assert (await store.aget_by_ids(["seed"]))[
            0
        ].page_content == "oracle existing seed"
        record = uuid.uuid4().hex
        try:
            await store.aadd_texts(
                ["oracle matrix"], [{"tenant": record}], ids=[record]
            )
            assert (await store.aget_by_ids([record]))[0].metadata == {"tenant": record}
            assert (
                await store.asimilarity_search(
                    "oracle", filter={"tenant": {"$eq": record}}
                )
            )[0].id == record
            await store.aadd_texts(
                ["oracle updated"], [{"tenant": record}], ids=[record]
            )
            assert (await store.aget_by_ids([record]))[
                0
            ].page_content == "oracle updated"
        finally:
            await store.adelete([record])
        assert_preserved(connection, name, baseline[name])
    finally:
        await client.close()


@pytest.mark.parametrize("backend", ["auto", "sql", "vecdb"])
@pytest.mark.parametrize("pooled", [False, True])
def test_missing_table_sync(database_matrix, backend, pooled):
    connection, pool, _, _, _ = database_matrix
    name = "LCM_NEW_" + uuid.uuid4().hex[:10].upper()
    client = pool if pooled else connection
    try:
        if backend == "vecdb":
            with pytest.raises(ValueError, match="COMPATIBLE"):
                OracleVS(client, MatrixEmbeddings(), name, backend=backend)
        else:
            store = OracleVS.from_texts(
                ["oracle fresh"],
                MatrixEmbeddings(),
                [{"tenant": "fresh"}],
                client=client,
                table_name=name,
                backend=backend,
                ids=["fresh"],
                mutate_on_duplicate=True,
            )
            assert store.backend == "sql"
            reopened = OracleVS(client, MatrixEmbeddings(), name, backend="auto")
            assert reopened.get_by_ids(["fresh"])[0].page_content == "oracle fresh"
    finally:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT COUNT(*) FROM user_tables WHERE table_name=:name", name=name
            )
            exists = cursor.fetchone()[0]
            if backend == "vecdb":
                assert exists == 0
            if exists:
                cursor.execute(f'DROP TABLE "{name}" PURGE')


@pytest.mark.parametrize("backend", ["auto", "sql", "vecdb"])
@pytest.mark.parametrize("pooled", [False, True])
@pytest.mark.asyncio
async def test_missing_table_async(database_matrix, backend, pooled):
    connection, _, config, _, _ = database_matrix
    name = "LCM_NEW_" + uuid.uuid4().hex[:10].upper()
    client = (
        oracledb.create_pool_async(**config, min=1, max=2, increment=1)
        if pooled
        else await oracledb.connect_async(**config)
    )
    try:
        if backend == "vecdb":
            with pytest.raises(ValueError, match="COMPATIBLE"):
                await OracleVS.acreate(
                    client, MatrixEmbeddings(), name, backend=backend
                )
        else:
            store = await OracleVS.afrom_texts(
                ["oracle fresh"],
                MatrixEmbeddings(),
                [{"tenant": "fresh"}],
                client=client,
                table_name=name,
                backend=backend,
                ids=["fresh"],
                mutate_on_duplicate=True,
            )
            assert store.backend == "sql"
            reopened = await OracleVS.acreate(
                client, MatrixEmbeddings(), name, backend="auto"
            )
            assert (await reopened.aget_by_ids(["fresh"]))[
                0
            ].page_content == "oracle fresh"
    finally:
        await client.close()
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT COUNT(*) FROM user_tables WHERE table_name=:name", name=name
            )
            exists = cursor.fetchone()[0]
            if backend == "vecdb":
                assert exists == 0
            if exists:
                cursor.execute(f'DROP TABLE "{name}" PURGE')


@pytest.mark.parametrize("origin", sorted(SQL_ORIGINS))
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_existing_sql_insert_only_preserves_original(
    database_matrix, origin, asynchronous
):
    connection, _, config, tables, baseline = database_matrix
    name = tables[origin]
    if asynchronous:
        async with await oracledb.connect_async(**config) as client:
            store = await OracleVS.acreate(
                client, MatrixEmbeddings(), name, backend="auto"
            )
            assert (
                await store.aadd_texts(
                    ["must not replace seed"], [{"tenant": "changed"}], ids=["seed"]
                )
                == []
            )
    else:
        store = OracleVS(connection, MatrixEmbeddings(), name, backend="auto")
        assert (
            store.add_texts(
                ["must not replace seed"], [{"tenant": "changed"}], ids=["seed"]
            )
            == []
        )
    assert_preserved(connection, name, baseline[name])


@pytest.mark.parametrize(
    "origin", ["library_sql", "legacy_vecdb_shape", "current_vecdb_shape"]
)
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_schema_qualified_sql_access_checks_real_layout(
    database_matrix, origin, asynchronous
):
    connection, _, config, tables, baseline = database_matrix
    name = tables[origin]
    with connection.cursor() as cursor:
        cursor.execute("SELECT SYS_CONTEXT('USERENV','CURRENT_SCHEMA') FROM dual")
        owner = cursor.fetchone()[0]
    qualified = f'"{owner}"."{name}"'

    async def open_async(client):
        return await OracleVS.acreate(
            client, MatrixEmbeddings(), qualified, backend="sql"
        )

    if asynchronous:
        async with await oracledb.connect_async(**config) as client:
            if origin in SQL_ORIGINS:
                store = await open_async(client)
                assert (await store.aget_by_ids(["seed"]))[
                    0
                ].page_content == "oracle existing seed"
            else:
                with pytest.raises(ValueError, match="not registered in VecDB"):
                    await open_async(client)
    elif origin in SQL_ORIGINS:
        store = OracleVS(connection, MatrixEmbeddings(), qualified, backend="sql")
        assert store.get_by_ids(["seed"])[0].page_content == "oracle existing seed"
    else:
        with pytest.raises(ValueError, match="not registered in VecDB"):
            OracleVS(connection, MatrixEmbeddings(), qualified, backend="sql")
    assert_preserved(connection, name, baseline[name])


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("literal_dot", [False, True])
@pytest.mark.asyncio
async def test_native_vecdb_sql_identifiers_preserve_target(
    database_matrix, asynchronous, literal_dot
):
    """Exercise SQL paths without claiming public VecDB support here.

    These VecDB-shaped fixtures test identifier resolution rather than package
    APIs. The public backend remains guarded by this database's COMPATIBLE.
    """
    from langchain_oracledb.vectorstores._vecdb import (
        AsyncNativeTransport,
        NativeTransport,
        VecDBBackend,
    )

    connection, _, config, _, _ = database_matrix
    with connection.cursor() as cursor:
        cursor.execute("SELECT USER FROM dual")
        owner = cursor.fetchone()[0]
    stem = "LCM_ID_" + uuid.uuid4().hex[:10].upper()
    # Both objects are writable. Losing the literal name's quotes would write
    # into the decoy OWNER.STEM rather than the intended "OWNER.STEM" table.
    literal = f'"{owner}.{stem}"'
    qualified = f'"{owner}"."{stem}"'
    target = literal if literal_dot else qualified
    created = []
    client = None
    try:
        with connection.cursor() as cursor:
            for name in (f'"{stem}"', literal):
                cursor.execute(
                    f"CREATE TABLE {name} (ID VARCHAR2(64) PRIMARY KEY, "
                    "DENSE_VECTOR VECTOR(3,FLOAT32), CONTENT_METADATA JSON)"
                )
                created.append(name)
                cursor.setinputsizes(
                    None, oracledb.DB_TYPE_VECTOR, oracledb.DB_TYPE_JSON
                )
                cursor.execute(
                    f"INSERT INTO {name} VALUES (:1,:2,:3)",
                    ["seed", array.array("f", [1, 0, 0]), {"text": "seed"}],
                )
        connection.commit()
        client = await oracledb.connect_async(**config) if asynchronous else connection
        transport = (AsyncNativeTransport if asynchronous else NativeTransport)(
            client, (26, 1, 0)
        )
        backend = VecDBBackend(transport, target, DistanceStrategy.COSINE)
        if asynchronous:
            await backend.run_async("prepare")
            result = await backend.run_async(
                "add", ["oracle target"], [[1, 0, 0]], [{}], ["added"], False
            )
        else:
            backend.prepare()
            result = backend.add(["oracle target"], [[1, 0, 0]], [{}], ["added"], False)
        assert result == ["added"]
        with connection.cursor() as cursor:
            for name in (literal, qualified):
                cursor.execute(f"SELECT ID FROM {name} ORDER BY ID")
                assert cursor.fetchall() == (
                    [("added",), ("seed",)] if name == target else [("seed",)]
                )
            cursor.execute(
                f"SELECT JSON_VALUE(CONTENT_METADATA, '$.text') "
                f"FROM {target} WHERE ID='added'"
            )
            assert cursor.fetchone()[0] == "oracle target"
    finally:
        if asynchronous and client is not None:
            await client.close()
        connection.rollback()
        with connection.cursor() as cursor:
            for name in reversed(created):
                cursor.execute(f"DROP TABLE {name} PURGE")


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("backend", ["auto", "sql", "vecdb"])
@pytest.mark.parametrize(
    "kind", ["private", "chain", "qualified", "public", "quoted_dot"]
)
@pytest.mark.asyncio
async def test_sql_synonym_factory_preserves_backend_and_rows(
    database_matrix, asynchronous, backend, kind
):
    connection, _, config, tables, baseline = database_matrix
    target = tables["library_sql"]
    alias = "LCM_SYN_" + uuid.uuid4().hex[:10].upper()
    if kind == "quoted_dot":
        alias += ".Dot"
    with connection.cursor() as cursor:
        cursor.execute("SELECT privilege FROM session_privs")
        privileges = {r[0] for r in cursor.fetchall()}
        required = (
            {"CREATE PUBLIC SYNONYM", "DROP PUBLIC SYNONYM"}
            if kind == "public"
            else {"CREATE SYNONYM"}
        )
        if not required <= privileges:
            pytest.skip(
                f"Synonym privileges unavailable: {sorted(required - privileges)}"
            )
        cursor.execute("SELECT USER FROM dual")
        owner = cursor.fetchone()[0]
    public = "PUBLIC " if kind == "public" else ""
    access = f'"{owner}"."{alias}"' if kind == "qualified" else f'"{alias}"'
    created = []
    record = uuid.uuid4().hex
    try:
        with connection.cursor() as cursor:
            referent = f'"{owner}"."{target}"'
            if kind == "chain":
                link = alias + "_LINK"
                cursor.execute(f'CREATE SYNONYM "{link}" FOR {referent}')
                created.append(("", link))
                referent = f'"{link}"'
            cursor.execute(f'CREATE {public}SYNONYM "{alias}" FOR {referent}')
            created.append((public, alias))
        if asynchronous:
            async with await oracledb.connect_async(**config) as client:
                if backend == "vecdb":
                    with pytest.raises(ValueError, match="ordinary OracleVS"):
                        await OracleVS.acreate(
                            client, MatrixEmbeddings(), access, backend=backend
                        )
                else:
                    store = await OracleVS.afrom_texts(
                        ["oracle synonym"],
                        MatrixEmbeddings(),
                        [{"tenant": record}],
                        client=client,
                        table_name=access,
                        ids=[record],
                        backend=backend,
                        mutate_on_duplicate=True,
                    )
                    try:
                        assert store.backend == "sql" and store._vecdb is None
                        assert (await store.aget_by_ids(["seed"]))[
                            0
                        ].page_content == "oracle existing seed"
                        assert (await store.aget_by_ids([record]))[
                            0
                        ].page_content == "oracle synonym"
                        assert (
                            await store.asimilarity_search(
                                "oracle", filter={"tenant": {"$eq": record}}
                            )
                        )[0].id == record
                        await store.aadd_texts(
                            ["oracle updated"], [{"tenant": record}], ids=[record]
                        )
                        assert (await store.aget_by_ids([record]))[
                            0
                        ].page_content == "oracle updated"
                    finally:
                        await store.adelete([record])
        elif backend == "vecdb":
            with pytest.raises(ValueError, match="ordinary OracleVS"):
                OracleVS(connection, MatrixEmbeddings(), access, backend=backend)
        else:
            store = OracleVS.from_texts(
                ["oracle synonym"],
                MatrixEmbeddings(),
                [{"tenant": record}],
                client=connection,
                table_name=access,
                ids=[record],
                backend=backend,
                mutate_on_duplicate=True,
            )
            try:
                assert store.backend == "sql" and store._vecdb is None
                assert (
                    store.get_by_ids(["seed"])[0].page_content == "oracle existing seed"
                )
                assert store.get_by_ids([record])[0].page_content == "oracle synonym"
                assert (
                    store.similarity_search(
                        "oracle", filter={"tenant": {"$eq": record}}
                    )[0].id
                    == record
                )
                store.add_texts(["oracle updated"], [{"tenant": record}], ids=[record])
                assert store.get_by_ids([record])[0].page_content == "oracle updated"
            finally:
                store.delete([record])
        assert_preserved(connection, target, baseline[target])
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT COUNT(*) FROM user_tables WHERE table_name=:name", name=alias
            )
            assert cursor.fetchone()[0] == 0
    finally:
        with connection.cursor() as cursor:
            for visibility, name in reversed(created):
                cursor.execute(f'DROP {visibility}SYNONYM "{name}"')


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("cycle", [False, True])
@pytest.mark.asyncio
async def test_broken_synonym_never_creates_a_table(
    database_matrix, asynchronous, cycle
):
    connection, _, config, _, _ = database_matrix
    alias = "LCM_BAD_" + uuid.uuid4().hex[:10].upper()
    link = alias + "_LINK"
    created = []
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT privilege FROM session_privs WHERE privilege='CREATE SYNONYM'"
            )
            if not cursor.fetchone():
                pytest.skip("CREATE SYNONYM is unavailable")
            cursor.execute(f'CREATE SYNONYM "{alias}" FOR "{link}"')
            created.append(alias)
            if cycle:
                cursor.execute(f'CREATE SYNONYM "{link}" FOR "{alias}"')
                created.append(link)
        if asynchronous:
            async with await oracledb.connect_async(**config) as client:
                with pytest.raises(RuntimeError, match="ORA-00942|ORA-00980|ORA-01775"):
                    await OracleVS.acreate(client, MatrixEmbeddings(), alias)
        else:
            with pytest.raises(RuntimeError, match="ORA-00942|ORA-00980|ORA-01775"):
                OracleVS(connection, MatrixEmbeddings(), alias)
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT COUNT(*) FROM user_tables WHERE table_name=:name", name=alias
            )
            assert cursor.fetchone()[0] == 0
    finally:
        with connection.cursor() as cursor:
            for name in reversed(created):
                cursor.execute(f'DROP SYNONYM "{name}"')


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("origin", ["library_sql", "current_vecdb_shape", "unrelated"])
@pytest.mark.asyncio
async def test_current_schema_does_not_inspect_login_users_table(
    database_matrix, asynchronous, origin
):
    from langchain_oracledb.vectorstores._vecdb import (
        AsyncNativeTransport,
        NativeTransport,
        anative_selection,
        native_selection,
    )

    connection, _, config, tables, baseline = database_matrix
    name = tables[origin]
    client = await oracledb.connect_async(**config) if asynchronous else connection
    original = None
    try:
        transport = (AsyncNativeTransport if asynchronous else NativeTransport)(
            client, (26, 1, 0)
        )
        with client.cursor() as cursor:
            if asynchronous:
                await cursor.execute(
                    "SELECT SYS_CONTEXT('USERENV','CURRENT_SCHEMA') FROM SYS.DUAL"
                )
                original = (await cursor.fetchone())[0]
                before = await transport.acolumns(name)
                await cursor.execute("ALTER SESSION SET CURRENT_SCHEMA=SYS")
            else:
                cursor.execute(
                    "SELECT SYS_CONTEXT('USERENV','CURRENT_SCHEMA') FROM SYS.DUAL"
                )
                original = cursor.fetchone()[0]
                before = transport.columns(name)
                cursor.execute("ALTER SESSION SET CURRENT_SCHEMA=SYS")
            assert before
            if asynchronous:
                assert await transport.acolumns(name) == set()
                assert (await anative_selection(client, name, "auto"))[0] == "sql"
                assert await transport.acolumns(f'"{original}"."{name}"') == before
                await cursor.execute(
                    "SELECT SYS_CONTEXT('USERENV','CURRENT_SCHEMA') FROM SYS.DUAL"
                )
                assert (await cursor.fetchone())[0] == "SYS"
            else:
                assert transport.columns(name) == set()
                assert native_selection(client, name, "auto")[0] == "sql"
                assert transport.columns(f'"{original}"."{name}"') == before
                cursor.execute(
                    "SELECT SYS_CONTEXT('USERENV','CURRENT_SCHEMA') FROM SYS.DUAL"
                )
                assert cursor.fetchone()[0] == "SYS"
    finally:
        if original is not None:
            with client.cursor() as cursor:
                if asynchronous:
                    await cursor.execute(
                        f'ALTER SESSION SET CURRENT_SCHEMA="{original}"'
                    )
                else:
                    cursor.execute(f'ALTER SESSION SET CURRENT_SCHEMA="{original}"')
        if asynchronous:
            await client.close()
    assert_preserved(connection, name, baseline[name])


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_current_schema_index_lookup_does_not_find_login_users_index(
    database_matrix, asynchronous
):
    from langchain_oracledb.vectorstores._vecdb import (
        AsyncNativeTransport,
        NativeTransport,
        _index_query,
        _metadata_index_query,
    )

    connection, _, config, tables, baseline = database_matrix
    name = tables["current_vecdb_shape"]
    index = "LCM_CSIDX_" + uuid.uuid4().hex[:10].upper()
    with connection.cursor() as cursor:
        cursor.execute("SELECT SYS_CONTEXT('USERENV','CURRENT_SCHEMA') FROM SYS.DUAL")
        original = cursor.fetchone()[0]
    created = False
    client = None
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                f'CREATE SEARCH INDEX "{index}" ON "{name}" (CONTENT_METADATA) '
                "FOR JSON PARAMETERS ('SEARCH_ON TEXT')"
            )
            created = True
        client = await oracledb.connect_async(**config) if asynchronous else connection
        transport = (AsyncNativeTransport if asynchronous else NativeTransport)(
            client, (26, 1, 0)
        )
        with client.cursor() as cursor:

            async def aquery(statement, binds):
                await cursor.execute(statement, **binds)
                return await cursor.fetchall()

            def query(statement, binds):
                cursor.execute(statement, **binds)
                return cursor.fetchall()

            for factory, args in [
                (_metadata_index_query, (name, "CONTENT_METADATA")),
                (_index_query, (index,)),
            ]:
                statement, binds = factory(*args)
                found = (
                    await aquery(statement, binds)
                    if asynchronous
                    else query(statement, binds)
                )
                assert len(found) == 1 and found[0][0] == index
            if asynchronous:
                await cursor.execute("ALTER SESSION SET CURRENT_SCHEMA=SYS")
            else:
                cursor.execute("ALTER SESSION SET CURRENT_SCHEMA=SYS")
            for factory, args in [
                (_metadata_index_query, (name, "CONTENT_METADATA")),
                (_index_query, (index,)),
            ]:
                statement, binds = factory(*args)
                found = (
                    await aquery(statement, binds)
                    if asynchronous
                    else query(statement, binds)
                )
                assert found == []
            with pytest.raises(ValueError, match="recognized VecDB table"):
                if asynchronous:
                    await transport.acreate_text_index(name, "DO_NOT_CREATE")
                else:
                    transport.create_text_index(name, "DO_NOT_CREATE")
    finally:
        if client is not None:
            with client.cursor() as cursor:
                if asynchronous:
                    await cursor.execute(
                        f'ALTER SESSION SET CURRENT_SCHEMA="{original}"'
                    )
                else:
                    cursor.execute(f'ALTER SESSION SET CURRENT_SCHEMA="{original}"')
            if asynchronous:
                await client.close()
        if created:
            with connection.cursor() as cursor:
                cursor.execute(f'DROP INDEX "{original}"."{index}"')
    assert_preserved(connection, name, baseline[name])


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("pending", [False, True])
@pytest.mark.asyncio
async def test_native_vecdb_failed_batch_preserves_transaction_boundary(
    database_matrix, asynchronous, pending
):
    from langchain_oracledb.vectorstores._vecdb import (
        AsyncNativeTransport,
        NativeTransport,
    )

    _, _, config, tables, baseline = database_matrix
    name = tables["current_vecdb_shape"]
    valid = "atomic_" + uuid.uuid4().hex[:12]
    caller = "caller_" + uuid.uuid4().hex[:12]
    records = [
        {"id": valid, "dense_vector": [1.0, 0.0, 0.0], "metadata": {"text": "valid"}},
        {
            "id": "seed",
            "dense_vector": [1.0, 0.0, 0.0],
            "metadata": {"text": "duplicate"},
        },
        {
            "id": "fatal",
            "dense_vector": [1.0, 0.0],
            "metadata": {"text": "wrong dimension"},
        },
    ]
    if asynchronous:
        async with oracledb.connect_async(**config) as connection:
            with connection.cursor() as cursor:
                if pending:
                    await cursor.execute(
                        f'INSERT INTO "{name}" (ID) VALUES (:id)', id=caller
                    )
                transport = AsyncNativeTransport(connection, (23, 26, 4))
                with pytest.raises(
                    RuntimeError, match="pending work" if pending else "ORA-"
                ):
                    await transport.ainsert(name, records)
                await cursor.execute(
                    f'SELECT ID FROM "{name}" WHERE ID IN (:a,:b)', a=valid, b=caller
                )
                assert await cursor.fetchall() == ([(caller,)] if pending else [])
                if pending:
                    assert connection.transaction_in_progress
                    await connection.rollback()
                else:
                    await connection.commit()
    else:
        with oracledb.connect(**config) as connection:
            with connection.cursor() as cursor:
                if pending:
                    cursor.execute(f'INSERT INTO "{name}" (ID) VALUES (:id)', id=caller)
                transport = NativeTransport(connection, (23, 26, 4))
                with pytest.raises(
                    RuntimeError, match="pending work" if pending else "ORA-"
                ):
                    transport.insert(name, records)
                cursor.execute(
                    f'SELECT ID FROM "{name}" WHERE ID IN (:a,:b)', a=valid, b=caller
                )
                assert cursor.fetchall() == ([(caller,)] if pending else [])
                if pending:
                    assert connection.transaction_in_progress
                    connection.rollback()
                else:
                    connection.commit()
    with oracledb.connect(**config) as connection:
        assert_preserved(connection, name, baseline[name])


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_native_migration_rejects_sparse_ddl_below_compatible(
    database_matrix, asynchronous
):
    from langchain_oracledb.vectorstores._vecdb import (
        AsyncNativeTransport,
        NativeTransport,
    )

    _, _, config, tables, baseline = database_matrix
    name = tables["legacy_vecdb_shape"]
    if asynchronous:
        async with oracledb.connect_async(**config) as connection:
            transport = AsyncNativeTransport(connection, (23, 26, 1))
            with pytest.raises(ValueError, match="COMPATIBLE >= 23.6"):
                await transport.amigrate(name)
    else:
        with oracledb.connect(**config) as connection:
            transport = NativeTransport(connection, (23, 26, 1))
            with pytest.raises(ValueError, match="COMPATIBLE >= 23.6"):
                transport.migrate(name)
    with oracledb.connect(**config) as connection:
        assert_preserved(connection, name, baseline[name])


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("operation", ["text", "insert"])
@pytest.mark.asyncio
async def test_native_pool_operation_keeps_checkout_schema(
    database_matrix, asynchronous, operation
):
    """Real driver checkouts alternate owning schema/SYS; no cross-tenant claim.

    The VecDB-shaped fixture exercises native SQL, not package capability gates.
    SYS has no table with this unique fixture name, making a second checkout
    inside the first operation observable without creating another user's table.
    """
    from contextlib import asynccontextmanager, contextmanager

    from langchain_oracledb.vectorstores._vecdb import (
        AsyncNativeTransport,
        NativeTransport,
    )
    from langchain_oracledb.vectorstores.utils import _quote_indentifier

    connection, _, config, tables, baseline = database_matrix
    name = tables["current_vecdb_shape"]
    index = "LCM_F8_" + uuid.uuid4().hex[:10].upper()
    id_ = "pool-f8-" + uuid.uuid4().hex
    with connection.cursor() as cursor:
        cursor.execute("SELECT SYS_CONTEXT('USERENV','CURRENT_SCHEMA') FROM dual")
        owner = _quote_indentifier(cursor.fetchone()[0])
        if operation == "text":
            cursor.execute(
                f'CREATE SEARCH INDEX "{index}" ON "{name}" (CONTENT_METADATA) FOR JSON'
            )

    class AlternatingPool(oracledb.ConnectionPool):
        checkouts = 0

        @contextmanager
        def acquire(self):
            self.checkouts += 1
            schema = owner if self.checkouts % 2 else '"SYS"'
            with super().acquire() as client:
                with client.cursor() as cursor:
                    cursor.execute("ALTER SESSION SET CURRENT_SCHEMA=" + schema)
                try:
                    yield client
                finally:
                    with client.cursor() as cursor:
                        cursor.execute("ALTER SESSION SET CURRENT_SCHEMA=" + owner)

    class AlternatingAsyncPool(oracledb.AsyncConnectionPool):
        checkouts = 0

        @asynccontextmanager
        async def acquire(self):
            self.checkouts += 1
            schema = owner if self.checkouts % 2 else '"SYS"'
            async with super().acquire() as client:
                with client.cursor() as cursor:
                    await cursor.execute("ALTER SESSION SET CURRENT_SCHEMA=" + schema)
                try:
                    yield client
                finally:
                    with client.cursor() as cursor:
                        await cursor.execute(
                            "ALTER SESSION SET CURRENT_SCHEMA=" + owner
                        )

    records = [
        {
            "id": id_,
            "dense_vector": [1.0, 0.0, 0.0],
            "metadata": {"text": "oracle pool insert"},
        }
    ]
    pool = (AlternatingAsyncPool if asynchronous else AlternatingPool)(
        **config, min=1, max=1, increment=0
    )
    try:
        transport = (AsyncNativeTransport if asynchronous else NativeTransport)(
            pool, (23, 26, 1)
        )
        if asynchronous:
            result = (
                await transport.atext_search(name, "oracle", 1)
                if operation == "text"
                else await transport.ainsert(name, records)
            )
        else:
            result = (
                transport.text_search(name, "oracle", 1)
                if operation == "text"
                else transport.insert(name, records)
            )
        assert pool.checkouts == 1
        assert transport.client is pool
        assert transport.version == (23, 26, 1)
        if operation == "text":
            assert [doc.page_content for doc in result] == ["oracle existing seed"]
        else:
            assert result == [id_]
        # With cached text discovery, SQL rejects the missing SYS table. After
        # insert-only writes there is no text snapshot, so discovery rejects it
        # before search SQL. Both paths must fail in the second checkout.
        error_type = oracledb.DatabaseError if operation == "text" else ValueError
        message = "ORA-00942" if operation == "text" else "recognized VecDB table"
        with pytest.raises(error_type, match=message):
            if asynchronous:
                await transport.atext_search(name, "oracle", 1)
            else:
                transport.text_search(name, "oracle", 1)
        assert pool.checkouts == 2
    finally:
        if asynchronous:
            await pool.close()
        else:
            pool.close()
        with connection.cursor() as cursor:
            cursor.execute(f'DELETE FROM "{name}" WHERE ID=:id', id=id_)
            connection.commit()
            if operation == "text":
                cursor.execute(f'DROP INDEX "{index}"')
    assert_preserved(connection, name, baseline[name])


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "operation", ["create_vector_table", "drop_vector_table", "upsert_vectors"]
)
@pytest.mark.asyncio
async def test_native_function_signatures_compile_without_execution(
    database_matrix, asynchronous, operation
):
    """Qualify emitted signatures without running package mutations."""
    from langchain_oracledb.vectorstores._vecdb import NativeTransport, parse_version

    connection, _, config, _, _ = database_matrix
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT COUNT(*) FROM all_procedures WHERE owner='SYS' "
            "AND object_name='DBMS_VECTOR_DATABASE' AND procedure_name=:operation",
            operation=operation.upper(),
        )
        if not cursor.fetchone()[0]:
            pytest.skip("Native package operation unavailable on this database")
    params = {"name": "LCM_NEVER_CREATE"}
    if operation == "create_vector_table":
        params["table_params"] = {"auto_generate_id": False}
    elif operation == "upsert_vectors":
        params = {
            "table_name": "LCM_NEVER_CREATE",
            "vectors": [
                {
                    "id": "never",
                    "dense_vector": [1.0, 0.0, 0.0],
                    "metadata": {"text": "never"},
                }
            ],
        }
    statement, binds = NativeTransport(
        connection, parse_version(connection.version)
    )._statement(operation, params)
    # Oracle still resolves the function signature in an unreachable PL/SQL branch.
    statement = (
        statement.replace(
            "BEGIN :result :=", "DECLARE result CLOB; BEGIN IF FALSE THEN result :=", 1
        ).removesuffix(" END;")
        + " END IF; END;"
    )
    if asynchronous:
        async with oracledb.connect_async(**config) as client:
            with client.cursor() as cursor:
                await cursor.execute(statement, binds)
    else:
        with connection.cursor() as cursor:
            cursor.execute(statement, binds)
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT COUNT(*) FROM user_tables WHERE table_name='LCM_NEVER_CREATE'"
        )
        assert cursor.fetchone()[0] == 0


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "failure",
    [
        "interruption",
        "connection_loss",
        "rollback_failure",
        "commit_before",
        "commit_after",
    ],
)
@pytest.mark.asyncio
async def test_native_insert_real_driver_failure_outcomes(
    database_matrix, monkeypatch, asynchronous, failure
):
    """Inject failures around real driver DML/commit; no network-loss claim.

    commit_after models a lost acknowledgment: rollback cannot undo that commit.
    Async interruption uses actual task cancellation after real DML completes.
    """
    import asyncio
    from contextlib import asynccontextmanager, contextmanager

    from langchain_oracledb.vectorstores import _vecdb

    _, _, config, tables, baseline = database_matrix
    name = tables["current_vecdb_shape"]
    identifier = "failure_" + uuid.uuid4().hex[:12]
    records = [
        {
            "id": identifier,
            "dense_vector": [1.0, 0.0, 0.0],
            "metadata": {"text": "failure fixture"},
        }
    ]
    connection = (
        await oracledb.connect_async(**config)
        if asynchronous
        else oracledb.connect(**config)
    )
    error = RuntimeError("injected failure after real driver operation")
    entered = asyncio.Event()
    closed = False

    class CursorProxy:
        def __init__(self):
            self.cursor = connection.cursor()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self.cursor.__exit__(*args)

        def __getattr__(self, key):
            return getattr(self.cursor, key)

        def executemany(self, *args, **kwargs):
            self.cursor.executemany(*args, **kwargs)
            if failure == "interruption":
                raise KeyboardInterrupt()
            if failure in {"connection_loss", "rollback_failure"}:
                if failure == "connection_loss":
                    proxy.close()
                raise error

        async def aexecutemany(self, *args, **kwargs):
            await self.cursor.executemany(*args, **kwargs)
            if failure == "interruption":
                entered.set()
                await asyncio.Event().wait()
            if failure in {"connection_loss", "rollback_failure"}:
                if failure == "connection_loss":
                    await proxy.aclose()
                raise error

    class ConnectionProxy:
        def __getattr__(self, key):
            return getattr(connection, key)

        def cursor(self):
            cursor = CursorProxy()
            if asynchronous:
                cursor.executemany = cursor.aexecutemany
            return cursor

        def close(self):
            nonlocal closed
            connection.close()
            closed = True

        async def aclose(self):
            nonlocal closed
            await connection.close()
            closed = True

        def commit(self):
            if failure == "commit_before":
                raise error
            connection.commit()
            if failure == "commit_after":
                raise error

        async def acommit(self):
            if failure == "commit_before":
                raise error
            await connection.commit()
            if failure == "commit_after":
                raise error

        def rollback(self):
            if failure == "rollback_failure":
                self.close()
            connection.rollback()

        async def arollback(self):
            if failure == "rollback_failure":
                await self.aclose()
            await connection.rollback()

    proxy = ConnectionProxy()
    if asynchronous:
        proxy.commit = proxy.acommit
        proxy.rollback = proxy.arollback

    @contextmanager
    def get(_):
        yield proxy

    @asynccontextmanager
    async def aget(_):
        yield proxy

    monkeypatch.setattr(_vecdb, "_get_connection", get)
    monkeypatch.setattr(_vecdb, "_aget_connection", aget)
    transport = (
        _vecdb.AsyncNativeTransport if asynchronous else _vecdb.NativeTransport
    )(None, (23, 26, 4))
    try:
        expected = (
            oracledb.Error
            if failure == "connection_loss"
            else asyncio.CancelledError
            if asynchronous and failure == "interruption"
            else KeyboardInterrupt
            if failure == "interruption"
            else RuntimeError
        )
        with pytest.raises(expected) as caught:
            if asynchronous:
                task = asyncio.create_task(transport.ainsert(name, records))
                if failure == "interruption":
                    await asyncio.wait_for(entered.wait(), timeout=10)
                    task.cancel()
                await task
            else:
                transport.insert(name, records)
        if failure not in {"interruption", "connection_loss"}:
            assert caught.value is error
        if failure in {"connection_loss", "rollback_failure"}:
            assert isinstance(caught.value.__cause__, oracledb.Error)
        # Detect leaked uncommitted rows by committing any still-open session.
        if not closed:
            if asynchronous:
                await connection.commit()
            else:
                connection.commit()
        with oracledb.connect(**config) as observer:
            with observer.cursor() as cursor:
                cursor.execute(f'SELECT ID FROM "{name}" WHERE ID=:id', id=identifier)
                assert cursor.fetchall() == (
                    [(identifier,)] if failure == "commit_after" else []
                )
    finally:
        if not closed:
            if asynchronous:
                await connection.close()
            else:
                connection.close()
        with oracledb.connect(**config) as cleanup:
            with cleanup.cursor() as cursor:
                cursor.execute(f'DELETE FROM "{name}" WHERE ID=:id', id=identifier)
            cleanup.commit()
            assert_preserved(cleanup, name, baseline[name])


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_vecdb_index_readiness_snapshot_tracks_real_metadata_index(
    database_matrix, asynchronous
):
    from langchain_oracledb.vectorstores._vecdb import (
        AsyncNativeTransport,
        NativeTransport,
        VecDBBackend,
    )

    connection, _, config, tables, baseline = database_matrix
    name = tables["current_vecdb_shape"]
    index = "LCM_READY_" + uuid.uuid4().hex[:12].upper()
    client = await oracledb.connect_async(**config) if asynchronous else connection
    transport = (AsyncNativeTransport if asynchronous else NativeTransport)(
        client, (23, 26, 4)
    )
    store = OracleVS.__new__(OracleVS)
    store._vecdb = VecDBBackend(transport, name, DistanceStrategy.COSINE)

    async def status():
        return (
            await store.adescribe_vecdb_index(kind="metadata")
            if asynchronous
            else store.describe_vecdb_index(kind="metadata")
        )

    try:
        assert (await status())["ready"] is False
        with connection.cursor() as cursor:
            cursor.execute(
                f'CREATE SEARCH INDEX "{index}" ON "{name}" (CONTENT_METADATA) '
                "FOR JSON PARAMETERS ('DATAGUIDE OFF SEARCH_ON TEXT SYNC (MANUAL)')"
            )
        result = await status()
        assert result["ready"] is True
        assert result["details"]["indexes"][0]["index_name"] == index
        with connection.cursor() as cursor:
            cursor.execute(f'DROP INDEX "{index}"')
        assert (await status())["ready"] is False
    finally:
        if asynchronous:
            await client.close()
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT COUNT(*) FROM user_indexes WHERE index_name=:name", name=index
            )
            if cursor.fetchone()[0]:
                cursor.execute(f'DROP INDEX "{index}" FORCE')
    assert_preserved(connection, name, baseline[name])


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_native_metadata_replacement_recovers_real_index_and_ingestion(
    database_matrix, asynchronous
):
    """Real DROP/failed CREATE/restore, followed by insert and metadata text query."""
    from langchain_oracledb.vectorstores._vecdb import (
        AsyncNativeTransport,
        NativeTransport,
        _metadata_index_query,
    )

    connection, _, config, tables, baseline = database_matrix
    name = tables["current_vecdb_shape"]
    index = "LCM_RECOVER_" + uuid.uuid4().hex[:10].upper()
    identifier = "recovered_" + uuid.uuid4().hex[:12]
    client = await oracledb.connect_async(**config) if asynchronous else connection
    transport = (AsyncNativeTransport if asynchronous else NativeTransport)(
        client, (23, 26, 4)
    )
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                f'CREATE SEARCH INDEX "{index}" ON "{name}" (CONTENT_METADATA) '
                "FOR JSON PARAMETERS ('DATAGUIDE ON CHANGE "
                "SYS.DBMS_VECTOR_DATABASE.HANDLE_MVI_INDEX "
                "SEARCH_ON TEXT SYNC (MANUAL)')"
            )
            statement, binds = _metadata_index_query(name, "CONTENT_METADATA")
            cursor.execute(statement, binds)
            original = cursor.fetchall()
        assert len(original) == 1
        with pytest.raises(oracledb.DatabaseError):
            if asynchronous:
                await transport.acreate_metadata_index(
                    name,
                    "IGNORED",
                    hybrid=True,
                    build=lambda *_: "CREATE HYBRID VECTOR INDEX",
                )
            else:
                transport.create_metadata_index(
                    name,
                    "IGNORED",
                    hybrid=True,
                    build=lambda *_: "CREATE HYBRID VECTOR INDEX",
                )
        with connection.cursor() as cursor:
            cursor.execute(statement, binds)
            assert cursor.fetchall() == original
        records = [
            {
                "id": identifier,
                "dense_vector": [1.0, 0.0, 0.0],
                "metadata": {"text": "recoveryprobe", "tenant": "afterrestore"},
            }
        ]
        if asynchronous:
            assert await transport.ainsert(name, records) == [identifier]
        else:
            assert transport.insert(name, records) == [identifier]
        with connection.cursor() as cursor:
            cursor.execute("BEGIN CTX_DDL.SYNC_INDEX(:name); END;", name=index)
        hits = (
            await transport.atext_search(name, "recoveryprobe", 10)
            if asynchronous
            else transport.text_search(name, "recoveryprobe", 10)
        )
        assert [doc.id for doc in hits] == [identifier]
        assert hits[0].metadata["tenant"] == "afterrestore"
        readiness = (
            await transport.adescribe_index(name, "metadata")
            if asynchronous
            else transport.describe_index(name, "metadata")
        )
        assert readiness["ready"] is True
    finally:
        if asynchronous:
            await client.close()
        with connection.cursor() as cursor:
            cursor.execute(f'DELETE FROM "{name}" WHERE ID=:id', id=identifier)
            connection.commit()
            cursor.execute(
                "SELECT COUNT(*) FROM user_indexes WHERE index_name=:name", name=index
            )
            if cursor.fetchone()[0]:
                cursor.execute(f'DROP INDEX "{index}" FORCE')
    assert_preserved(connection, name, baseline[name])


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_native_signature_preflight_rejects_before_package_execution(
    database_matrix, asynchronous
):
    from langchain_oracledb.vectorstores._vecdb import (
        AsyncNativeTransport,
        NativeTransport,
    )

    connection, _, config, tables, baseline = database_matrix
    client = await oracledb.connect_async(**config) if asynchronous else connection
    transport = (AsyncNativeTransport if asynchronous else NativeTransport)(
        client, (23, 26, 4)
    )
    # Use a nonexistent table, even though the unreachable block cannot invoke DROP.
    statement, binds = transport._statement(
        "drop_vector_table", {"name": "LCM_NEVER_DROP"}
    )
    invalid = statement.replace("name =>", "unsupported_argument =>", 1)
    transport._statement = lambda *args, **kwargs: (invalid, binds)
    try:
        with pytest.raises(ValueError, match="cannot compile its expected signature"):
            if asynchronous:
                await transport.acall("drop_vector_table", name="LCM_NEVER_DROP")
            else:
                transport.call("drop_vector_table", name="LCM_NEVER_DROP")
        for name in tables.values():
            assert_preserved(connection, name, baseline[name])
    finally:
        if asynchronous:
            await client.close()
