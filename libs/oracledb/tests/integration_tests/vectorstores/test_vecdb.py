# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""Opt-in REST VecDB checks using isolated, automatically removed tables."""

import os
import uuid

import pytest
from langchain_core.embeddings import Embeddings

from langchain_oracledb import OracleVS
from langchain_oracledb.vectorstores.utils import DistanceStrategy, drop_table_purge


class FixedEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [
            [1.0, 0.0, 0.0] if "oracle" in text.lower() else [0.0, 1.0, 0.0]
            for text in texts
        ]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


@pytest.fixture
def vecdb_store():
    url = os.environ.get("ORACLE_VECDB_REST_URL")
    if not url:
        pytest.skip("Set ORACLE_VECDB_REST_URL for REST VecDB integration tests")
    sdk = pytest.importorskip("oracle_vecdb")
    auth = {}
    if os.environ.get("ORACLE_VECDB_ACCESS_TOKEN"):
        auth["access_token"] = os.environ["ORACLE_VECDB_ACCESS_TOKEN"]
    else:
        auth["username"] = os.environ["ORACLE_VECDB_USERNAME"]
        auth["password"] = os.environ["ORACLE_VECDB_PASSWORD"]
    client = sdk.OracleVecDB(
        sdk.Configuration(
            rest_url=url,
            ssl_ca_cert=os.environ.get("ORACLE_VECDB_CA_CERT"),
            timeout=30,
            **auth,
        )
    )
    table = "LCV_TEST_" + uuid.uuid4().hex[:12].upper()
    try:
        store = OracleVS(
            client,
            FixedEmbeddings(),
            table,
            distance_strategy=DistanceStrategy.COSINE,
            mutate_on_duplicate=True,
        )
        yield store
    finally:
        try:
            drop_table_purge(client, table, backend="vecdb")
        except Exception as exc:
            if getattr(exc, "status", None) != 404:
                raise


def test_rest_crud_filters_and_full_text(vecdb_store):
    store = vecdb_store
    long_text = "oracle " + "long document payload " * 4000
    assert store.backend == "vecdb"
    assert store.add_texts(
        [long_text, "other subject"],
        [{"tenant": "a"}, {"tenant": "b"}],
        ids=["one", "two"],
    ) == ["one", "two"]
    doc = store.get_by_ids(["one"])[0]
    assert doc.page_content == long_text
    assert doc.metadata == {"tenant": "a"}
    hits = store.similarity_search_with_score("oracle", k=2)
    assert hits[0][0].id == "one"
    assert hits[0][1] == pytest.approx(0)
    assert [
        doc.id
        for doc in store.similarity_search(
            "oracle", k=2, filter={"tenant": {"$eq": "b"}}
        )
    ] == ["two"]
    store.add_texts(["oracle changed"], [{"tenant": "a"}], ids=["one"])
    assert store.get_by_ids(["one"])[0].page_content == "oracle changed"
    store.delete(["one", "two"])
    assert store.get_by_ids(["one", "two"]) == []


@pytest.mark.asyncio
async def test_rest_async_and_returned_embeddings(vecdb_store):
    store = vecdb_store
    assert await store.aadd_texts(["oracle"], ids=["async-one"]) == ["async-one"]
    hits = await store.asimilarity_search_by_vector_returning_embeddings(
        [1.0, 0.0, 0.0], k=1
    )
    assert hits[0][0].id == "async-one"
    assert hits[0][2].tolist() == [1.0, 0.0, 0.0]
    assert (await store.aget_by_ids(["async-one"]))[0].page_content == "oracle"
    await store.adelete(["async-one"])
    assert await store.aget_by_ids(["async-one"]) == []


@pytest.fixture
def native_index_store(vecdb_store):
    """Use a native client for operations outside the REST SDK API."""
    dsn = os.environ.get("ORACLE_VECDB_NATIVE_DSN")
    if not dsn:
        pytest.skip("Set ORACLE_VECDB_NATIVE_DSN to test native index operations")
    import oracledb

    with oracledb.connect(
        user=os.environ["ORACLE_VECDB_USERNAME"],
        password=os.environ["ORACLE_VECDB_PASSWORD"],
        dsn=dsn,
    ) as connection:
        store = OracleVS(
            connection,
            FixedEmbeddings(),
            vecdb_store.table_name,
            distance_strategy=DistanceStrategy.COSINE,
            mutate_on_duplicate=True,
        )
        yield store, connection


def metadata_index(connection, store):
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT index_name, parameters FROM user_indexes "
            "WHERE table_name = :name AND index_type = 'DOMAIN'",
            name=store.table_name.strip('"'),
        )
        return cursor.fetchone()


def sync_metadata_index(connection, index):
    with connection.cursor() as cursor:
        cursor.execute("BEGIN CTX_DDL.SYNC_INDEX(:name); END;", name=index)


def test_text_replacement_preserves_vecdb_settings(native_index_store):
    from langchain_oracledb.retrievers.text_search import (
        OracleTextSearchRetriever,
        create_text_index,
    )

    store, connection = native_index_store
    original_name, original_parameters = metadata_index(connection, store)
    store.add_texts(["oracle database"], [{"tenant": "one"}], ids=["one"])
    create_text_index(connection, "REQUESTED_TEXT_NAME", vector_store=store)
    actual_name, parameters = metadata_index(connection, store)
    assert actual_name == original_name
    assert "SEARCH_ON TEXT" in parameters.upper()
    assert "HANDLE_MVI_INDEX" in parameters.upper()
    assert "interval=30" in original_parameters and "interval=30" in parameters
    retriever = OracleTextSearchRetriever(vector_store=store)
    assert [doc.id for doc in retriever.invoke("oracle")] == ["one"]
    store.add_texts(["kiwi orchard"], [{"new_text_path": "introduced"}], ids=["one"])
    sync_metadata_index(connection, actual_name)
    assert [doc.id for doc in retriever.invoke("kiwi")] == ["one"]
    assert retriever.invoke("oracle") == []
    assert [
        doc.id
        for doc in store.similarity_search(
            "kiwi", filter={"new_text_path": {"$eq": "introduced"}}
        )
    ] == ["one"]


def test_text_to_hybrid_generates_mvi_for_new_metadata_path(native_index_store):
    from langchain_oracledb.embeddings import OracleEmbeddings
    from langchain_oracledb.retrievers.hybrid_search import (
        OracleHybridSearchRetriever,
        create_hybrid_index,
    )
    from langchain_oracledb.retrievers.text_search import (
        OracleTextSearchRetriever,
        create_text_index,
    )

    model = os.environ.get("ORACLE_VECDB_HYBRID_MODEL")
    if not model:
        pytest.skip("Set ORACLE_VECDB_HYBRID_MODEL to an installed embedding model")
    store, connection = native_index_store
    store.embedding_function = OracleEmbeddings(
        conn=connection, params={"provider": "database", "model": model}
    )
    store.add_texts(["oracle database"], [{"tenant": "one"}], ids=["one"])
    original_name, _ = metadata_index(connection, store)
    create_text_index(connection, "REQUESTED_TEXT_NAME", vector_store=store)
    create_hybrid_index(connection, "REQUESTED_HYBRID_NAME", vector_store=store)
    actual_name, parameters = metadata_index(connection, store)
    assert actual_name == original_name
    assert "SIMPLIFIED_JSON_HYBRID" in parameters.upper()
    assert "HANDLE_MVI_INDEX" in parameters.upper()
    assert "interval=30" in parameters
    store.add_texts(
        ["oracle newpathmarker"], [{"new_hvi_path": "introduced"}], ids=["new"]
    )
    sync_metadata_index(connection, actual_name)
    # Inspect before filtering, so filter execution cannot account for the new MVI.
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT column_expression FROM user_ind_expressions "
            "WHERE table_name = :name",
            name=store.table_name.strip('"'),
        )
        assert any("new_hvi_path" in str(row[0]) for row in cursor.fetchall())
    assert [
        doc.id
        for doc in store.similarity_search(
            "oracle", filter={"new_hvi_path": {"$eq": "introduced"}}
        )
    ] == ["new"]
    text = OracleTextSearchRetriever(vector_store=store)
    assert [doc.id for doc in text.invoke("newpathmarker")] == ["new"]
    hybrid = OracleHybridSearchRetriever(
        vector_store=store, idx_name="REQUESTED_HYBRID_NAME", return_scores=True
    )
    hits = hybrid.invoke("oracle")
    assert hits
    assert any(doc.metadata["text_score"] > 0 for doc in hits)
    assert any(doc.metadata["vector_score"] > 0 for doc in hits)
    create_text_index(connection, "REQUESTED_TEXT_NAME", vector_store=store)
    create_hybrid_index(connection, "REQUESTED_HYBRID_NAME", vector_store=store)
    assert metadata_index(connection, store) == (actual_name, parameters)


def test_rest_sdk_reports_unsupported_sql_operations(vecdb_store):
    from langchain_oracledb.retrievers.hybrid_search import (
        OracleHybridSearchRetriever,
        create_hybrid_index,
    )
    from langchain_oracledb.retrievers.text_search import (
        OracleTextSearchRetriever,
        create_text_index,
    )

    store = vecdb_store
    operations = [
        lambda: create_text_index(store.client, "TEXT_IDX", vector_store=store),
        lambda: create_hybrid_index(store.client, "HYBRID_IDX", vector_store=store),
        lambda: OracleTextSearchRetriever(vector_store=store).invoke("oracle"),
        lambda: OracleHybridSearchRetriever(
            vector_store=store, idx_name="HYBRID_IDX"
        ).invoke("oracle"),
    ]
    for operation in operations:
        with pytest.raises(ValueError, match="oracle-vecdb REST SDK does not support"):
            operation()


def test_native_auto_falls_back_when_compatible_is_below_sparse_requirement():
    """A new SQL store is usable even when the VecDB package is installed."""
    import oracledb

    from langchain_oracledb.vectorstores._vecdb import parse_version

    dsn = os.environ.get("VECDB_HOST")
    if not dsn:
        pytest.skip(
            "Set VECDB_HOST, VECDB_USER and VECDB_PASS for native fallback checks"
        )
    with oracledb.connect(
        user=os.environ["VECDB_USER"], password=os.environ["VECDB_PASS"], dsn=dsn
    ) as connection:
        with connection.cursor() as cursor:
            db_version = cursor.var(str, size=128)
            compatible = cursor.var(str, size=128)
            cursor.execute(
                "BEGIN DBMS_UTILITY.DB_VERSION(:version, :compatible); END;",
                version=db_version,
                compatible=compatible,
            )
        if parse_version(connection.version) < (23, 26, 3) or parse_version(
            compatible.getvalue()
        ) >= (23, 6, 0):
            pytest.skip("This regression requires current VecDB with COMPATIBLE <23.6")
        table = "LCN_COMP_" + uuid.uuid4().hex[:12].upper()
        try:
            with pytest.raises(ValueError, match=r"COMPATIBLE >= 23\.6\.0"):
                OracleVS(connection, FixedEmbeddings(), table, backend="vecdb")
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT COUNT(*) FROM user_tables WHERE table_name=:name",
                    name=table,
                )
                assert cursor.fetchone()[0] == 0
            store = OracleVS(
                connection,
                FixedEmbeddings(),
                table,
                backend="auto",
                mutate_on_duplicate=True,
            )
            assert store.backend == "sql" and store._vecdb is None
            store.add_texts(["oracle database"], ids=["one"])
            reopened = OracleVS(connection, FixedEmbeddings(), table, backend="auto")
            assert reopened.backend == "sql"
            assert reopened.get_by_ids(["one"])[0].page_content == "oracle database"
            assert reopened.similarity_search("oracle")[0].id == "one"
            reopened.delete(["one"])
            assert reopened.get_by_ids(["one"]) == []
        finally:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT table_name FROM user_tables WHERE table_name=:name",
                    name=table,
                )
                if cursor.fetchone():
                    cursor.execute(f'DROP TABLE "{table}" PURGE')


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_rest_projected_metadata_and_query_options(vecdb_store, asynchronous):
    store = vecdb_store
    store.add_texts(
        ["oracle projection"],
        [{"tenant": "one", "omitted": "value"}],
        ids=["projection"],
    )
    options = {"output_selector": ["tenant"], "advanced_options": {"accuracy": 90}}
    if asynchronous:
        results = await store.asimilarity_search("oracle", k=1, **options)
    else:
        results = store.similarity_search("oracle", k=1, **options)
    assert results[0].id == "projection"
    assert results[0].page_content == "oracle projection"
    assert results[0].metadata == {"tenant": "one"}
