# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""Backend selection, SDK contracts, and bounded VecDB layout repair."""

import json
import subprocess
import sys
from contextlib import contextmanager
from copy import deepcopy
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from langchain_oracledb import OracleVS
from langchain_oracledb._annotations import table_annotations
from langchain_oracledb.retrievers.hybrid_search import (
    OracleHybridSearchRetriever,
)
from langchain_oracledb.retrievers.text_search import OracleTextSearchRetriever
from langchain_oracledb.vectorstores._vecdb import (
    CURRENT_COLUMNS,
    AsyncNativeTransport,
    NativeTransport,
    VecDBBackend,
    _layout,
    _migration_actions,
    layout_error,
    native_selection,
    parse_version,
    table_identifier,
)
from langchain_oracledb.vectorstores._vecdb_adapter import LOGICAL_ID_KEY, storage_id
from langchain_oracledb.vectorstores.utils import DistanceStrategy


class Embedding(Embeddings):
    def embed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0]


@pytest.fixture
def sdk():
    oracle_vecdb = pytest.importorskip("oracle_vecdb")
    client = oracle_vecdb.OracleVecDB(
        oracle_vecdb.Configuration(
            rest_url="https://example.com/ords/test/_/db-api/stable/vecdb/",
            access_token="test",
        )
    )
    client.describe_vector_table = MagicMock(return_value={"table_name": "DOCS"})
    client.create_vector_table = MagicMock(return_value={"table_name": "DOCS"})
    client.describe_index = MagicMock(return_value={"index_status": "READY"})
    client.rebuild_index = MagicMock(return_value={"message": "rebuilt"})
    client.upsert_vectors = MagicMock(return_value={"upserted_count": 1})
    client.delete_vectors = MagicMock(return_value={"message": "deleted"})
    client.list_vectors = MagicMock(return_value={"items": []})
    client.query = MagicMock(
        return_value={
            "items": [
                {
                    "id": "a",
                    "metadata": {"__orcl_internal_content": "hello", "tenant": "one"},
                    "distance": 0.25,
                    "vector": [1.0, 0.0],
                }
            ]
        }
    )
    return client


@pytest.fixture
def store(sdk):
    return OracleVS(sdk, Embedding(), "DOCS", mutate_on_duplicate=True)


@pytest.mark.parametrize(
    "version, expected",
    [("23.26.1.0.0", (23, 26, 1)), ("Release 23.0.0 Version 23.26.3.0.0", (23, 26, 3))],
)
def test_version(version, expected):
    assert parse_version(version) == expected


@pytest.mark.parametrize("name", ['x"y', " ", "schema..docs", '"unclosed'])
def test_identifiers_reject_malformed_names(name):
    with pytest.raises(ValueError):
        table_identifier(name)


@pytest.mark.parametrize(
    "name, quoted, api_name",
    [
        ('"हिन्दी"', '"हिन्दी"', "हिन्दी"),
        ("schema.docs", '"schema"."docs"', "schema.docs"),
        ('"literal.dot"', '"literal.dot"', "literal.dot"),
        ("DOCS; DROP TABLE x", '"DOCS; DROP TABLE x"', "DOCS; DROP TABLE x"),
        ("DOCS) --", '"DOCS) --"', "DOCS) --"),
    ],
)
def test_identifiers_preserve_shared_quoting(name, quoted, api_name):
    assert table_identifier(name) == (quoted, api_name)


def test_optional_sdk_not_imported_by_sql_package():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import langchain_oracledb; "
            "assert 'oracle_vecdb' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_factory_reports_missing_sdk():
    with patch.dict(sys.modules, {"oracle_vecdb": None}):
        with pytest.raises(ImportError, match=r"langchain-oracledb\[vecdb\]"):
            OracleVS.from_vecdb_config(object(), "DOCS", Embedding())


@pytest.mark.parametrize("kwargs", [{}, {"mutate_on_duplicate": True}])
def test_factory_defaults_to_upserts(sdk, kwargs):
    with patch("oracle_vecdb.OracleVecDB", return_value=sdk):
        store = OracleVS.from_vecdb_config(object(), "DOCS", Embedding(), **kwargs)

    assert store.add_texts(["hello"], ids=["a"]) == ["a"]
    sdk.upsert_vectors.assert_called_once()


def test_factory_preserves_explicit_insert_only(sdk):
    with (
        patch("oracle_vecdb.OracleVecDB", return_value=sdk),
        pytest.raises(ValueError, match="does not support insert-only"),
    ):
        OracleVS.from_vecdb_config(
            object(), "DOCS", Embedding(), mutate_on_duplicate=False
        )
    sdk.describe_vector_table.assert_not_called()
    sdk.create_vector_table.assert_not_called()
    sdk.upsert_vectors.assert_not_called()


def selection_connection(version, columns, supported=True, compatible="23.6.0.0.0"):
    connection = MagicMock(version=version)
    cursor = connection.cursor.return_value.__enter__.return_value
    cursor.fetchall.return_value = [(column,) for column in columns]
    cursor.fetchone.return_value = (int(supported),)
    statement = ""

    def execute(sql, **binds):
        nonlocal statement
        statement = sql

    def fetchone():
        if "WITH FUNCTION lc_compatible" in statement:
            return int(supported), compatible if supported else None
        if "SYS.DBMS_VECTOR_DATABASE.DESCRIBE_VECTOR_TABLE" in statement:
            return (
                {"owner": "OWNER", "table_name": "DOCS"}
                if "DENSE_VECTOR" in columns
                else None,
                "OWNER",
            )
        return (int(supported),)

    cursor.execute.side_effect = execute
    cursor.fetchone.side_effect = fetchone
    cursor.var.side_effect = lambda *args, **kwargs: MagicMock(
        getvalue=MagicMock(return_value=compatible)
    )
    return connection, cursor


@pytest.mark.parametrize(
    "version, expected",
    [
        ("23.4.0.0.0", "sql"),
        ("23.26.1.0.0", "vecdb"),
        ("23.26.2.0.0", "vecdb"),
        ("23.26.3.0.0", "vecdb"),
    ],
)
def test_auto_new_table(version, expected):
    conn, _ = selection_connection(version, [])
    assert native_selection(conn, "DOCS", "auto")[0] == expected


def test_auto_preserves_existing_sql_even_on_current_database():
    conn, cursor = selection_connection(
        "23.26.3", {"ID", "EMBEDDING", "TEXT", "METADATA"}
    )
    assert native_selection(conn, "DOCS", "auto")[0] == "sql"
    assert cursor.execute.call_count == 3
    with pytest.raises(ValueError) as error:
        native_selection(conn, "DOCS", "vecdb")
    assert str(error.value) == (
        "The VecDB backend cannot use an ordinary OracleVS table layout; "
        "use backend='auto' or backend='sql'"
    )


def test_auto_missing_package_uses_sql_but_explicit_vecdb_fails():
    conn, _ = selection_connection("23.26.3", [], False)
    assert native_selection(conn, "DOCS", "auto")[0] == "sql"
    with pytest.raises(ValueError, match="unavailable"):
        native_selection(conn, "DOCS", "vecdb")


def test_explicit_sql_does_not_probe_capabilities():
    conn, cursor = selection_connection("23.26.3", [])
    assert native_selection(conn, "DOCS", "sql")[0] == "sql"
    assert [c.args[0] for c in cursor.execute.call_args_list] == [
        "SELECT column_name FROM all_tab_columns "
        "WHERE owner = SYS_CONTEXT('USERENV', 'CURRENT_SCHEMA') AND table_name = :name",
        'SELECT * FROM "DOCS" WHERE 1 = 0',
    ]


def test_current_table_rejected_on_legacy_database():
    conn, _ = selection_connection("23.26.1", CURRENT_COLUMNS)
    with pytest.raises(ValueError, match="legacy database"):
        native_selection(conn, "DOCS", "auto")


def test_rest_store_roundtrip_mapping_without_metadata_mutation(store, sdk):
    metadata = {"tenant": "one"}
    assert store.backend == "vecdb"
    assert store.add_texts(["hello"], [metadata], ids=["a"]) == ["a"]
    assert metadata == {"tenant": "one"}
    records = sdk.upsert_vectors.call_args.kwargs["vectors"]
    assert records == [
        {
            "id": storage_id("a"),
            "dense_vector": [1.0, 0.0],
            "metadata": {
                "tenant": "one",
                "__orcl_internal_content": "hello",
                LOGICAL_ID_KEY: "a",
            },
        }
    ]
    hits = store.similarity_search_with_score(
        "query", k=3, filter={"tenant": {"$eq": "one"}}
    )
    assert hits[0][0] == Document(
        id="a", page_content="hello", metadata={"tenant": "one"}
    )
    assert hits[0][1] == 0.25  # lower-is-better distance, same as SQL OracleVS
    assert sdk.query.call_args.kwargs["filters"] == {
        "tenant": {"$string": {"$eq": "one"}}
    }
    assert sdk.query.call_args.kwargs["advanced_options"] == {
        "distance_metric": "EUCLIDEAN"
    }


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_rest_numeric_metadata_roundtrip_uses_real_sdk_serializer(
    store, sdk, asynchronous
):
    from oracle_vecdb.services.ords.api_client import ApiClient
    from oracle_vecdb.services.ords.models.upsert_vectors_request import (
        UpsertVectorsRequest,
    )
    from oracle_vecdb.services.ords.models.upsert_vectors_request_vectors_inner import (
        UpsertVectorsRequestVectorsInner,
    )

    metadata = {
        "count": 1,
        "large": 9007199254740993,
        "nested": {"values": [1.25, 0.1, True, False, None, "1"]},
    }
    sdk.query.return_value["items"][0]["metadata"].update(metadata)
    doc = store.similarity_search("query")[0]
    assert isinstance(doc.metadata["count"], Decimal)
    original = deepcopy(doc.metadata)
    payloads = []

    def serialize_request(**kwargs):
        request = UpsertVectorsRequest(
            vectors=[
                UpsertVectorsRequestVectorsInner(**record)
                for record in kwargs["vectors"]
            ]
        )
        payloads.append(
            json.loads(json.dumps(ApiClient().sanitize_for_serialization(request)))
        )
        return {"upserted_count": 1}

    sdk.upsert_vectors.side_effect = serialize_request
    if asynchronous:
        await store.aadd_documents([doc])
    else:
        store.add_documents([doc])
    written = payloads[0]["vectors"][0]["metadata"]
    assert type(written["count"]) is int
    assert written["large"] == 9007199254740993
    assert type(written["large"]) is int
    assert type(written["nested"]["values"][0]) is float
    assert written["nested"]["values"][2] is True
    assert written["nested"]["values"][3] is False
    assert {key: written[key] for key in metadata} == metadata
    assert doc.metadata == original
    assert isinstance(doc.metadata["count"], Decimal)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "number",
    [
        Decimal("123456789.123456789"),
        Decimal("1e-400"),
        Decimal("1e400"),
        Decimal("NaN"),
        Decimal("Infinity"),
        Decimal("-Infinity"),
    ],
)
@pytest.mark.asyncio
async def test_rest_unrepresentable_metadata_numbers_do_not_write(
    store, sdk, asynchronous, number
):
    with pytest.raises(ValueError, match="precision|not JSON compliant"):
        if asynchronous:
            await store.aadd_texts(["hello"], [{"nested": [number]}], ids=["a"])
        else:
            store.add_texts(["hello"], [{"nested": [number]}], ids=["a"])
    sdk.upsert_vectors.assert_not_called()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_reserved_content_key_and_bad_batch_do_not_write(
    store, sdk, asynchronous
):
    with pytest.raises(ValueError, match="__orcl_internal_content.*reserved"):
        if asynchronous:
            await store.aadd_texts(
                ["hello"], [{"__orcl_internal_content": "user value"}]
            )
        else:
            store.add_texts(["hello"], [{"__orcl_internal_content": "user value"}])
    with pytest.raises(ValueError, match="Length mismatch"):
        store.add_texts(["hello"], ids=["one", "two"])
    sdk.upsert_vectors.assert_not_called()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_user_text_metadata_roundtrip_filter_and_projection(
    store, sdk, asynchronous
):
    metadata = {"text": "user label", "metadata": {"text": "nested label"}}
    if asynchronous:
        await store.aadd_texts(["document body"], [metadata], ids=["a"])
    else:
        store.add_texts(["document body"], [metadata], ids=["a"])
    record = sdk.upsert_vectors.call_args.kwargs["vectors"][0]
    assert record["metadata"] == {
        **metadata,
        "__orcl_internal_content": "document body",
        LOGICAL_ID_KEY: "a",
    }
    sdk.query.return_value = {"items": [{**record, "distance": 0.25}]}
    sdk.list_vectors.return_value = {"items": [record]}
    options = {"filter": {"text": {"$eq": "user label"}}, "output_selector": ["text"]}
    if asynchronous:
        hits = await store.asimilarity_search("query", **options)
        fetched = await store.aget_by_ids(["a"])
    else:
        hits = store.similarity_search("query", **options)
        fetched = store.get_by_ids(["a"])
    expected = Document(id="a", page_content="document body", metadata=metadata)
    assert hits == fetched == [expected]
    assert sdk.query.call_args.kwargs["filters"] == {
        "text": {"$string": {"$eq": "user label"}}
    }
    assert sdk.query.call_args.kwargs["output_selector"] == [
        "text",
        "__orcl_internal_content",
        LOGICAL_ID_KEY,
    ]
    assert metadata == {"text": "user label", "metadata": {"text": "nested label"}}


def test_empty_batch_does_not_write(store, sdk):
    assert store.add_texts([]) == []
    sdk.upsert_vectors.assert_not_called()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_vecdb_uses_shared_empty_id_and_metadata_defaults(
    store, sdk, asynchronous
):
    ids = (
        await store.aadd_texts(["hello"], metadatas=[], ids=[])
        if asynchronous
        else store.add_texts(["hello"], metadatas=[], ids=[])
    )
    assert len(ids) == 1 and ids[0]
    assert sdk.upsert_vectors.call_args.kwargs["vectors"][0]["id"] == storage_id(ids[0])


@pytest.mark.parametrize("kwargs", [{}, {"mutate_on_duplicate": False}])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_insert_only_rest_rejected_before_preparing_table(
    sdk, kwargs, asynchronous
):
    with pytest.raises(ValueError, match="REST SDK does not support"):
        if asynchronous:
            await OracleVS.acreate(sdk, Embedding(), "DOCS", **kwargs)
        else:
            OracleVS(sdk, Embedding(), "DOCS", **kwargs)
    sdk.describe_vector_table.assert_not_called()
    sdk.create_vector_table.assert_not_called()
    sdk.upsert_vectors.assert_not_called()


def test_get_pages_id_batches_and_delete(store, sdk):
    sdk.list_vectors.return_value = {
        "items": [{"id": "a", "metadata": {"__orcl_internal_content": "hello"}}]
    }
    assert store.get_by_ids([str(i) for i in range(205)])[0].page_content == "hello"
    assert [call.kwargs["limit"] for call in sdk.list_vectors.call_args_list] == [
        100,
        100,
        5,
    ]
    store.delete(["a"])
    sdk.delete_vectors.assert_called_once_with(table_name="DOCS", ids=[storage_id("a")])


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_delete_rejects_missing_ids_before_sdk(store, sdk, asynchronous):
    for ids in (None, []):
        with pytest.raises(ValueError, match="No ids provided to delete"):
            if asynchronous:
                await store.adelete(ids)
            else:
                store.delete(ids)
    sdk.delete_vectors.assert_not_called()


def test_return_embeddings(store):
    hit = store.similarity_search_by_vector_returning_embeddings([1.0, 0.0])[0]
    assert hit[0].id == "a"
    assert hit[2].tolist() == [1.0, 0.0]


@pytest.mark.asyncio
async def test_rest_async_paths(store, sdk):
    assert await store.aadd_texts(["hello"], ids=["a"]) == ["a"]
    assert (await store.asimilarity_search("q"))[0].id == "a"
    assert (await store.asimilarity_search_by_vector_returning_embeddings([1.0, 0.0]))[
        0
    ][2].tolist() == [1.0, 0.0]
    await store.adelete(["a"])
    sdk.delete_vectors.assert_called_once()


def test_only_404_causes_rest_creation(sdk):
    for status in [401, 403, 500]:
        error = RuntimeError("request failed")
        error.status = status
        sdk.describe_vector_table.side_effect = error
        with pytest.raises(RuntimeError):
            OracleVS(sdk, Embedding(), "DOCS", mutate_on_duplicate=True)
    sdk.create_vector_table.assert_not_called()
    error = RuntimeError("not found")
    error.status = 404
    sdk.describe_vector_table.side_effect = error
    OracleVS(sdk, Embedding(), "DOCS", mutate_on_duplicate=True)
    sdk.create_vector_table.assert_called_once_with(
        name="DOCS",
        table_params={"auto_generate_id": False},
        annotations=table_annotations("vector_store", "vecdb"),
    )


def test_migration_is_error_triggered_and_retry_is_bounded():
    transport = MagicMock()
    mismatch = RuntimeError('ORA-00904: "CONTENT_METADATA": invalid identifier')
    transport.call.side_effect = [mismatch, {"results": []}]
    backend = VecDBBackend(transport, "DOCS", DistanceStrategy.COSINE)
    assert backend.search([1.0], 2, None) == []
    transport.migrate.assert_called_once_with("DOCS")
    assert transport.call.call_count == 2
    transport.reset_mock()
    transport.call.side_effect = [mismatch, mismatch]
    with pytest.raises(RuntimeError):
        backend.search([1.0], 2, None)
    assert transport.call.call_count == 2
    transport.migrate.assert_called_once()


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("HTTP 401"),
        RuntimeError("ORA-00904: tenant"),
        RuntimeError("ORA-01031: insufficient privileges"),
    ],
)
def test_unrelated_errors_never_migrate(error):
    transport = MagicMock()
    transport.call.side_effect = error
    backend = VecDBBackend(transport, "DOCS", DistanceStrategy.COSINE)
    with pytest.raises(RuntimeError):
        backend.search([1.0], 2, None)
    transport.migrate.assert_not_called()


def test_migration_steps_are_resumable_and_reject_ambiguous_layout():
    assert len(_migration_actions({"ID", "DENSE_VECTOR", "METADATA"}, '"DOCS"')) == 4
    assert (
        len(
            _migration_actions(
                {"ID", "DENSE_VECTOR", "CONTENT_METADATA", "CONTENT"}, '"DOCS"'
            )
        )
        == 2
    )
    assert _migration_actions(CURRENT_COLUMNS, '"DOCS"') == []
    with pytest.raises(ValueError):
        _migration_actions(CURRENT_COLUMNS | {"METADATA"}, '"DOCS"')
    with pytest.raises(ValueError):
        _layout({"ID", "TEXT"})


def test_pending_transaction_prevents_migration():
    connection = SimpleNamespace(transaction_in_progress=True, version="23.26.3")

    @contextmanager
    def get(_):
        yield connection

    with patch("langchain_oracledb.vectorstores._vecdb._get_connection", get):
        with pytest.raises(RuntimeError, match="pending work"):
            NativeTransport(object(), (23, 26, 3)).migrate("DOCS")


def test_native_package_signatures():
    legacy = NativeTransport(object(), (23, 26, 1))
    current = NativeTransport(object(), (23, 26, 3))
    old_sql, old_binds = legacy._statement(
        "create_vector_table",
        {"name": "DOCS", "table_params": {"auto_generate_id": False}},
    )
    new_sql, new_binds = current._statement(
        "create_vector_table",
        {"name": "DOCS", "table_params": {"auto_generate_id": False}},
    )
    assert "table_name => :table_name" in old_sql
    assert old_binds["auto_generate_id"] is False
    assert "name => :name" in new_sql
    assert "table_params => JSON(:table_params)" in new_sql
    assert json.loads(new_binds["table_params"]) == {"auto_generate_id": False}
    assert "table_params" not in old_binds


def test_generation_specific_index_json():
    old = VecDBBackend(
        NativeTransport(None, (23, 26, 1)), "DOCS", DistanceStrategy.COSINE
    )
    new = VecDBBackend(
        NativeTransport(None, (23, 26, 3)), "DOCS", DistanceStrategy.COSINE
    )
    assert old.index_params({"idx_type": "HNSW"})["indexing"] == "auto"
    assert (
        new.index_params({"idx_type": "HNSW"})["vector_index_params"]["auto_index"]
        is True
    )
    assert (
        new.index_params({"efConstruction": 20})["vector_index_params"][
            "advanced_params"
        ]["efConstruction"]
        == 20
    )
    with pytest.raises(ValueError):
        new.index_params({"accuracy": 101})


@pytest.mark.parametrize(
    "kind, invalid",
    [
        ("HNSW", {"accuracy": 0}),
        ("HNSW", {"accuracy": True}),
        ("HNSW", {"neighbors": 1}),
        ("HNSW", {"efConstruction": 65536}),
        ("IVF", {"neighbor_part": 0}),
        ("IVF", {"parallel": 0}),
    ],
)
def test_index_parameter_validation_matches_existing_sql(kind, invalid):
    from langchain_oracledb.vectorstores.oraclevs import (
        _get_hnsw_index_ddl,
        _get_ivf_index_ddl,
    )

    params = {"idx_type": kind, **invalid}
    sql_builder = _get_hnsw_index_ddl if kind == "HNSW" else _get_ivf_index_ddl
    backend = VecDBBackend(
        NativeTransport(None, (23, 26, 3)), "DOCS", DistanceStrategy.COSINE
    )
    with pytest.raises(ValueError) as sql_error:
        sql_builder('"DOCS"', DistanceStrategy.COSINE, params)
    with pytest.raises(ValueError) as vecdb_error:
        backend.index_params(params)
    assert str(vecdb_error.value) == str(sql_error.value)


def test_existing_index_options_are_translated_for_the_sdk(store, sdk):
    sdk.rebuild_index = MagicMock(return_value={"message": "rebuilt"})
    store._vecdb.create_index({"idx_type": "ivf", "neighbor_part": 32, "parallel": 2})
    mapped = sdk.rebuild_index.call_args.kwargs["index_params"]
    assert mapped["parallel_creation"] == 2
    assert mapped["vector_index_params"]["organization"] == "PARTITIONS"
    assert mapped["vector_index_params"]["advanced_params"] == {"partitions": 32}


def test_additional_index_option_validation_is_delegated_to_the_sdk(store, sdk):
    with pytest.raises(ValueError, match="unknown_option"):
        store._vecdb.create_index({"unknown_option": True})
    sdk.rebuild_index.assert_not_called()


@pytest.mark.parametrize(
    "filters",
    [
        {"address": {"city": "NYC"}},
        {"address.city": "NYC"},
        {"tags[*]": {"$all": ["a", "b"]}},
        {"size": {"$between": [1, 10]}},
        {"$nor": [{"tenant": "one"}, {"size": {"$gt": 2}}]},
    ],
)
def test_vecdb_filters_are_translated(store, sdk, filters):
    store.similarity_search("query", filter=filters)
    from langchain_oracledb.vectorstores._vecdb_filters import translate_filter

    assert sdk.query.call_args.kwargs["filters"] == translate_filter(filters)


def test_vecdb_filter_errors_come_from_the_sdk(store, sdk):
    with pytest.raises(ValueError, match="not a recognized logical operator"):
        store.similarity_search("query", filter={"$unknown": "x"})
    sdk.query.assert_not_called()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_native_text_limit_is_bound_instead_of_interpolated(asynchronous):
    transport = (
        AsyncNativeTransport(None, (23, 26, 3))
        if asynchronous
        else NativeTransport(None, (23, 26, 3))
    )
    connection = MagicMock()
    cursor = connection.cursor.return_value.__enter__.return_value
    limit = "1; DROP TABLE DOCS"
    if asynchronous:
        transport.acolumns = AsyncMock(return_value=CURRENT_COLUMNS)
        cursor.execute = AsyncMock()
        cursor.fetchall = AsyncMock(return_value=[])
        cursor.fetchone = AsyncMock(return_value=None)
        with patch(
            "langchain_oracledb.vectorstores._vecdb._aget_connection"
        ) as acquire:
            acquire.return_value.__aenter__.return_value = connection
            await transport.atext_search("DOCS", "oracle", limit)
    else:
        transport.columns = MagicMock(return_value=CURRENT_COLUMNS)
        cursor.fetchall.return_value = []
        with patch("langchain_oracledb.vectorstores._vecdb._get_connection") as acquire:
            acquire.return_value.__enter__.return_value = connection
            transport.text_search("DOCS", "oracle", limit)
    sql = cursor.execute.call_args.args[0]
    assert "FETCH FIRST :k ROWS ONLY" in sql
    assert limit not in sql
    assert cursor.execute.call_args.kwargs["k"] == limit


def test_text_retriever_uses_metadata_backend(store):
    store._vecdb.text_search = MagicMock(
        return_value=[Document(id="a", page_content="hello")]
    )
    retriever = OracleTextSearchRetriever(vector_store=store)
    assert retriever.invoke("hello")[0].id == "a"
    store._vecdb.text_search.assert_called_once_with('"hello"', 4, False)


@pytest.mark.parametrize("mode", ["keyword", "semantic", "hybrid"])
def test_hybrid_retriever_calls_native_search_without_application_fusion(store, mode):
    store._vecdb.transport = NativeTransport(object(), (23, 26, 4))
    store.embedding_function.embed_query = MagicMock(
        side_effect=AssertionError("separate query embedding")
    )
    expected = [
        Document(id="native", page_content="native result", metadata={"score": 7})
    ]
    store._vecdb.hybrid_search = MagicMock(return_value=expected)
    store._vecdb.text_search = MagicMock(
        side_effect=AssertionError("separate text query")
    )
    store.similarity_search = MagicMock(
        side_effect=AssertionError("separate vector query")
    )
    retriever = OracleHybridSearchRetriever(
        vector_store=store, idx_name="HVI", search_mode=mode, return_scores=True
    )
    assert retriever.invoke("hello") == expected
    params, scores = store._vecdb.hybrid_search.call_args.args
    assert scores is True
    assert params["hybrid_index_name"] == '"HVI"'
    assert ("text" in params) == (mode != "semantic")
    assert ("vector" in params) == (mode != "keyword")
    if mode != "semantic":
        assert params["text"] == {
            "json_textcontains": ["$.__orcl_internal_content", '"hello"']
        }
    if mode != "keyword":
        assert params["vector"] == {"search_text": "hello"}
    store.embedding_function.embed_query.assert_not_called()


@pytest.mark.parametrize("mode", ["keyword", "semantic", "hybrid"])
@pytest.mark.asyncio
async def test_async_hybrid_uses_database_query_embedding(store, mode):
    store._vecdb.transport = NativeTransport(object(), (23, 26, 4))
    store.embedding_function.aembed_query = AsyncMock(
        side_effect=AssertionError("separate async query embedding")
    )
    store.embedding_function.embed_query = MagicMock(
        side_effect=AssertionError("sync embed")
    )
    expected = [Document(id="native", page_content="hello")]
    store._vecdb.run_async = AsyncMock(return_value=expected)
    retriever = OracleHybridSearchRetriever(
        vector_store=store, idx_name="HVI", search_mode=mode
    )
    assert await retriever.ainvoke("hello") == expected
    method, params, scores = store._vecdb.run_async.call_args.args
    assert method == "hybrid_search"
    assert ("vector" in params) == (mode != "keyword")
    if mode != "keyword":
        assert params["vector"] == {"search_text": "hello"}
    store.embedding_function.embed_query.assert_not_called()
    store.embedding_function.aembed_query.assert_not_called()


def test_native_hybrid_transport_binds_query_and_maps_database_scores():
    transport = NativeTransport(object(), (23, 26, 4))
    transport.columns = MagicMock(return_value=CURRENT_COLUMNS)
    connection = MagicMock()
    cursor = connection.cursor.return_value.__enter__.return_value
    original = {"__orcl_internal_content": "hello", "tenant": "one"}
    cursor.fetchall.return_value = [("a", original, 9, 2, 7)]
    cursor.fetchone.return_value = ("HVI", "SIMPLIFIED_JSON_HYBRID")
    params = {
        "hybrid_index_name": "HVI",
        "text": {"json_textcontains": ["$.__orcl_internal_content", "x' OR 1=1"]},
    }

    @contextmanager
    def get(_):
        yield connection

    with patch("langchain_oracledb.vectorstores._vecdb._get_connection", get):
        docs = transport.hybrid_search("DOCS", params, True)
    assert docs[0].id == "a"
    assert docs[0].page_content == "hello"
    assert docs[0].metadata == {
        "tenant": "one",
        "score": 9,
        "text_score": 2,
        "vector_score": 0,
    }
    assert original == {"__orcl_internal_content": "hello", "tenant": "one"}
    statement = cursor.execute.call_args.args[0]
    assert "DBMS_HYBRID_VECTOR.SEARCH(JSON(:search_params))" in statement
    assert "ORDER BY h.position" in statement
    assert "x' OR 1=1" not in statement
    request = json.loads(cursor.execute.call_args.kwargs["search_params"])
    assert request["hybrid_index_name"] == '"HVI"'
    assert request["text"]["json_textcontains"] == params["text"]["json_textcontains"]
    assert request["text"]["inpath"] == ["$.__orcl_internal_content"]
    assert request["return"]["format"] == "JSON"


def test_rest_hybrid_reports_unsupported_operation(store):
    with pytest.raises(ValueError, match="REST SDK does not support"):
        OracleHybridSearchRetriever(vector_store=store, idx_name="HVI").invoke("hello")


@pytest.mark.parametrize("params", [{"rrf_k": 60}, {"fetch_k": 20}])
def test_hybrid_rejects_previous_application_fusion_parameters(store, params):
    store._vecdb.transport = NativeTransport(object(), (23, 26, 4))
    with pytest.raises(ValueError, match="DBMS_HYBRID_VECTOR"):
        OracleHybridSearchRetriever(
            vector_store=store, idx_name="HVI", params=params
        ).invoke("hello")


@pytest.mark.asyncio
async def test_async_native_query_retries_once():
    transport = AsyncNativeTransport(None, (23, 26, 3))
    transport.acall = AsyncMock(
        side_effect=[RuntimeError('ORA-00904: "CONTENT_METADATA"'), {"results": []}]
    )
    transport.amigrate = AsyncMock()
    backend = VecDBBackend(transport, "DOCS", DistanceStrategy.COSINE)
    assert await backend.run_async("search", [1.0], 3, None) == []
    transport.amigrate.assert_awaited_once_with('"DOCS"')
    assert transport.acall.await_count == 2


def test_native_text_index_replaces_guide_preserving_settings():
    transport = NativeTransport(object(), (23, 26, 3))
    transport.preflight_index = MagicMock()
    transport.columns = MagicMock(return_value=CURRENT_COLUMNS)
    connection = MagicMock()
    connection.transaction_in_progress = False
    cursor = connection.cursor.return_value.__enter__.return_value
    original = (
        "SIMPLIFIED_JSON dataguide on change "
        "sys.dbms_vector_database.handle_mvi_index search_on none "
        'sync (every "freq=secondly; interval=30" )'
    )
    cursor.fetchall.return_value = [("VEC_METADATA_IDX", original)]

    @contextmanager
    def get(_):
        yield connection

    with patch("langchain_oracledb.vectorstores._vecdb._get_connection", get):
        transport.create_text_index("DOCS", "REQUESTED_NAME")
    statements = [call.args[0] for call in cursor.execute.call_args_list]
    assert statements[1] == 'DROP INDEX "VEC_METADATA_IDX"'
    assert 'CREATE SEARCH INDEX "VEC_METADATA_IDX"' in statements[2]
    assert "SEARCH_ON TEXT" in statements[2]
    assert "search_on none" not in statements[2]
    assert 'sync (every "freq=secondly; interval=30" )' in statements[2]
    assert "sys.dbms_vector_database.handle_mvi_index" in statements[2]
    assert "MEMORY" not in statements[2] and "ON COMMIT" not in statements[2]


def test_native_text_index_reuses_already_enabled_index():
    transport = NativeTransport(object(), (23, 26, 3))
    transport.columns = MagicMock(return_value=CURRENT_COLUMNS)
    connection = MagicMock()
    cursor = connection.cursor.return_value.__enter__.return_value
    cursor.fetchall.return_value = [("VEC_METADATA_IDX", "search_on text_value")]

    @contextmanager
    def get(_):
        yield connection

    with patch("langchain_oracledb.vectorstores._vecdb._get_connection", get):
        transport.create_text_index("DOCS", "REQUESTED_NAME")
    assert cursor.execute.call_count == 1


def test_native_insert_only_skips_duplicate_rows():
    transport = NativeTransport(object(), (23, 26, 3))
    transport.columns = MagicMock(return_value=CURRENT_COLUMNS)
    connection = MagicMock(transaction_in_progress=False)
    cursor = connection.cursor.return_value.__enter__.return_value
    cursor.getbatcherrors.return_value = [
        SimpleNamespace(code=1, offset=0, message="duplicate ID")
    ]
    records = [
        {"id": id_, "dense_vector": [1.0], "metadata": {"__orcl_internal_content": "x"}}
        for id_ in ["a", "b"]
    ]

    @contextmanager
    def get(_):
        yield connection

    with patch("langchain_oracledb.vectorstores._vecdb._get_connection", get):
        assert transport.insert("DOCS", records) == ["b"]
    assert "INSERT INTO" in cursor.executemany.call_args.args[0]
    assert "CONTENT_METADATA" in cursor.executemany.call_args.args[0]
    assert cursor.executemany.call_args.kwargs["batcherrors"] is True
    connection.commit.assert_called_once()


@pytest.mark.asyncio
async def test_native_async_crud_uses_async_driver_not_thread_wrapping():
    transport = AsyncNativeTransport(None, (23, 26, 3))
    transport.acall = AsyncMock(return_value={"items": []})
    transport.ainsert = AsyncMock(return_value=[storage_id("a")])
    backend = VecDBBackend(transport, "DOCS", DistanceStrategy.COSINE)
    assert await backend.run_async("add", ["x"], [[1.0]], [{}], ["a"], False) == ["a"]
    transport.ainsert.assert_awaited_once()
    assert await backend.run_async("get", ["a"]) == []
    await backend.run_async("delete", ["a"])
    assert transport.acall.await_count == 2


def test_migration_applies_missing_actions_without_table_drop():
    import re

    from langchain_oracledb.vectorstores import _vecdb

    columns = {"ID", "DENSE_VECTOR", "CONTENT_METADATA", "CONTENT"}
    connection = MagicMock(transaction_in_progress=False, version="23.26.3")
    cursor = connection.cursor.return_value.__enter__.return_value
    cursor.var.return_value.getvalue.return_value = "23.6.0"

    def execute(statement, **binds):
        if " ADD " in statement:
            columns.add(re.search(r" ADD (\w+)", statement).group(1))

    cursor.execute.side_effect = execute
    cursor.fetchall.side_effect = lambda: [(column,) for column in columns]

    @contextmanager
    def get(_):
        yield connection

    with patch.object(_vecdb, "_get_connection", get):
        NativeTransport(None, (23, 26, 3)).migrate("DOCS")
    assert columns == CURRENT_COLUMNS
    actions = [
        call.args[0]
        for call in cursor.execute.call_args_list
        if "ALTER TABLE" in call.args[0]
    ]
    assert len(actions) == 2
    assert not any("RENAME" in action or "DROP" in action for action in actions)


def test_from_texts_propagates_backend_ids_and_mutation_mode(sdk):
    store = OracleVS.from_texts(
        ["hello"],
        Embedding(),
        client=sdk,
        table_name="DOCS",
        backend="vecdb",
        mutate_on_duplicate=True,
        ids=["custom"],
    )
    assert store.backend == "vecdb"
    assert sdk.upsert_vectors.call_args.kwargs["vectors"][0]["id"] == storage_id(
        "custom"
    )


def test_layout_error_checks_actual_missing_column_not_request_context():
    assert layout_error(
        RuntimeError('ORA-00904: "T"."CONTENT_METADATA": invalid identifier')
    )
    assert not layout_error(
        RuntimeError(
            "Request table_name CONTENT_METADATA; "
            'ORA-00904: "TENANT": invalid identifier'
        )
    )
    assert not layout_error(
        RuntimeError('ORA-00904: "CONTENT_METADATA_OTHER": invalid identifier')
    )


def test_native_pool_initialization_does_not_acquire_a_second_connection():
    from langchain_oracledb.vectorstores import _vecdb, oraclevs

    pool = object()
    connection = MagicMock(version="23.26.3", thin=True)
    cursor = connection.cursor.return_value.__enter__.return_value
    cursor.fetchall.return_value = []
    cursor.fetchone.return_value = (1,)
    created = False

    def execute(statement, **binds):
        nonlocal created
        if statement == _vecdb.VECDB_CAPABILITY_SQL:
            cursor.fetchone.return_value = (1, "23.6.0.0.0")
        if "create_vector_table" in statement.lower() and not statement.startswith(
            "BEGIN IF FALSE THEN"
        ):
            created = True
        if "column_name FROM" in statement:
            cursor.fetchall.return_value = (
                [(col,) for col in CURRENT_COLUMNS] if created else []
            )
        if "i.parameters" in statement:
            cursor.fetchone.return_value = None

    cursor.execute.side_effect = execute
    cursor.var.side_effect = lambda kind, **kwargs: MagicMock(
        getvalue=MagicMock(
            return_value=("23.6.0.0.0" if kind is str else '{"table_name":"DOCS"}')
        )
    )
    acquired = False

    @contextmanager
    def get(client):
        nonlocal acquired
        if client is pool:
            assert not acquired, "Nested pool acquisition would deadlock max=1 pools"
            acquired = True
            try:
                yield connection
            finally:
                acquired = False
        else:
            assert client is connection
            yield connection

    with (
        patch.object(oraclevs, "_get_connection", get),
        patch.object(_vecdb, "_get_connection", get),
    ):
        store = OracleVS(pool, Embedding(), "DOCS")
    assert store.backend == "vecdb"
    assert store._vecdb.transport.client is pool


def test_migration_refreshes_version_after_pool_database_upgrade():
    from langchain_oracledb.vectorstores import _vecdb

    connection = MagicMock(version="23.26.4", transaction_in_progress=False)
    cursor = connection.cursor.return_value.__enter__.return_value
    cursor.fetchall.return_value = [(column,) for column in CURRENT_COLUMNS]
    cursor.var.return_value.getvalue.return_value = "23.6.0"

    @contextmanager
    def get(_):
        yield connection

    transport = NativeTransport(None, (23, 26, 1))
    with patch.object(_vecdb, "_get_connection", get):
        transport.migrate("DOCS")
    assert transport.version == (23, 26, 4)


@pytest.mark.asyncio
async def test_async_native_hybrid_transport_returns_database_scores():
    from contextlib import asynccontextmanager

    transport = AsyncNativeTransport(object(), (23, 26, 4))
    transport.acolumns = AsyncMock(return_value=CURRENT_COLUMNS)
    connection = MagicMock()
    cursor = connection.cursor.return_value.__enter__.return_value
    cursor.execute = AsyncMock()
    cursor.fetchone = AsyncMock(return_value=("HVI", "SIMPLIFIED_JSON_HYBRID"))
    cursor.fetchall = AsyncMock(
        return_value=[
            ("a", {"__orcl_internal_content": "hello", "tenant": "one"}, 9, 2, 7)
        ]
    )

    @asynccontextmanager
    async def get(_):
        yield connection

    backend = VecDBBackend(transport, "DOCS", DistanceStrategy.COSINE)
    params = {"hybrid_index_name": "HVI", "vector": {"search_vector": [1.0, 0.0]}}
    with patch("langchain_oracledb.vectorstores._vecdb._aget_connection", get):
        docs = await backend.run_async("hybrid_search", params, False)
    assert docs == [Document(id="a", page_content="hello", metadata={"tenant": "one"})]
    request = json.loads(cursor.execute.call_args.kwargs["search_params"])
    assert request["hybrid_index_name"] == '"HVI"'
    assert json.loads(request["vector"]["search_vector"]) == [1.0, 0.0]
    assert params["vector"]["search_vector"] == [1.0, 0.0]
    assert request["return"]["format"] == "JSON"
    assert "DBMS_HYBRID_VECTOR.SEARCH" in cursor.execute.call_args.args[0]


@pytest.mark.parametrize(
    "version, column", [((23, 26, 1), "METADATA"), ((23, 26, 4), "CONTENT_METADATA")]
)
def test_vecdb_hybrid_ddl_uses_metadata_column(store, version, column):
    from langchain_oracledb.retrievers.hybrid_search import (
        OracleVectorizerPreference,
        _get_hybrid_index_ddl,
    )

    store._vecdb.transport = NativeTransport(object(), version)
    pref = OracleVectorizerPreference.__new__(OracleVectorizerPreference)
    pref.vs = store
    pref.preference_name = "MY_PREF"
    ddl = _get_hybrid_index_ddl(pref, '"HVI"', {})
    assert f'"DOCS"({column})' in ddl
    assert "CREATE HYBRID VECTOR INDEX" in ddl


@pytest.mark.parametrize(
    "operation", ["text", "hybrid", "text_index", "hybrid_index", "preference"]
)
def test_rest_unsupported_operations_fail_without_native_fallback(store, operation):
    from langchain_oracledb.retrievers.hybrid_search import (
        OracleVectorizerPreference,
        create_hybrid_index,
    )
    from langchain_oracledb.retrievers.text_search import create_text_index

    store._embed_query = MagicMock(side_effect=AssertionError("must not embed"))
    operations = {
        "text": lambda: OracleTextSearchRetriever(vector_store=store).invoke("hello"),
        "hybrid": lambda: OracleHybridSearchRetriever(
            vector_store=store, idx_name="IDX"
        ).invoke("hello"),
        "text_index": lambda: create_text_index(
            store.client, "IDX", vector_store=store
        ),
        "hybrid_index": lambda: create_hybrid_index(
            store.client, "IDX", vector_store=store
        ),
        "preference": lambda: OracleVectorizerPreference.create_preference(store),
    }
    with patch("langchain_oracledb.vectorstores._vecdb._get_connection") as connect:
        with pytest.raises(
            ValueError, match="oracle-vecdb REST SDK does not support"
        ) as error:
            operations[operation]()
    assert "native_client" not in str(error.value)
    connect.assert_not_called()
    store._embed_query.assert_not_called()
    store.client.query.assert_not_called()
    store.client.upsert_vectors.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operation", ["text", "hybrid", "text_index", "hybrid_index", "preference"]
)
async def test_async_rest_unsupported_operations_fail_without_native_fallback(
    store, operation
):
    from langchain_oracledb.retrievers.hybrid_search import (
        OracleVectorizerPreference,
        acreate_hybrid_index,
    )
    from langchain_oracledb.retrievers.text_search import acreate_text_index

    store._aembed_query = AsyncMock(side_effect=AssertionError("must not embed"))
    operations = {
        "text": lambda: OracleTextSearchRetriever(vector_store=store).ainvoke("hello"),
        "hybrid": lambda: OracleHybridSearchRetriever(
            vector_store=store, idx_name="IDX"
        ).ainvoke("hello"),
        "text_index": lambda: acreate_text_index(
            store.client, "IDX", vector_store=store
        ),
        "hybrid_index": lambda: acreate_hybrid_index(
            store.client, "IDX", vector_store=store
        ),
        "preference": lambda: OracleVectorizerPreference.acreate_preference(store),
    }
    with patch("langchain_oracledb.vectorstores._vecdb._aget_connection") as connect:
        with pytest.raises(
            ValueError, match="oracle-vecdb REST SDK does not support"
        ) as error:
            await operations[operation]()
    assert "native_client" not in str(error.value)
    connect.assert_not_called()
    store._aembed_query.assert_not_called()
    store.client.query.assert_not_called()
    store.client.upsert_vectors.assert_not_called()


def test_rest_layout_error_reports_unsupported_migration_without_retry(store):
    store.client.query.side_effect = RuntimeError(
        'ORA-00904: "CONTENT_METADATA": invalid identifier'
    )
    with pytest.raises(
        RuntimeError, match="REST SDK does not support schema migration"
    ):
        store.similarity_search("hello")
    assert store.client.query.call_count == 1


@pytest.mark.asyncio
async def test_async_rest_layout_error_reports_unsupported_migration_without_retry(
    store,
):
    store.client.query.side_effect = RuntimeError(
        'ORA-00904: "CONTENT_METADATA": invalid identifier'
    )
    with pytest.raises(
        RuntimeError, match="REST SDK does not support schema migration"
    ):
        await store.asimilarity_search("hello")
    assert store.client.query.call_count == 1


def test_public_store_factories_have_no_native_companion_parameter():
    import inspect

    assert "native_client" not in inspect.signature(OracleVS).parameters
    assert "native_client" not in inspect.signature(OracleVS.acreate).parameters


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_facade_forwards_vecdb_options_without_mutating_them(
    store, sdk, asynchronous
):
    options = {"accuracy": 90, "advanced_params": {"rescore_factor": 2}}
    selector = ["tenant"]
    method = (
        store.asimilarity_search_by_vector_with_relevance_scores
        if asynchronous
        else store.similarity_search_by_vector_with_relevance_scores
    )
    result = method(
        [1.0, 0.0],
        k=3,
        db_filter={"tenant": "one"},
        output_selector=selector,
        advanced_options=options,
    )
    if asynchronous:
        result = await result
    assert result[0][0].page_content == "hello"
    assert sdk.query.call_args.kwargs == {
        "table_name": "DOCS",
        "query_by": {"vector": [1.0, 0.0]},
        "top_k": 3,
        "filters": {"tenant": {"$string": {"$eq": "one"}}},
        "include_vectors": False,
        "advanced_options": {**options, "distance_metric": store._vecdb.metric},
        "output_selector": ["tenant", "__orcl_internal_content", LOGICAL_ID_KEY],
    }
    assert options == {"accuracy": 90, "advanced_params": {"rescore_factor": 2}}
    assert selector == ["tenant"]


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "options, message",
    [
        ({"typo": True}, "Unsupported VecDB search options"),
        ({"advanced_options": {"accuracy": 101}}, "accuracy"),
        ({"advanced_options": {"distance_metric": "WRONG"}}, "distance_strategy"),
        ({"output_selector": "tenant"}, "output_selector"),
    ],
)
@pytest.mark.asyncio
async def test_facade_rejects_unsupported_vecdb_options_before_query(
    store, sdk, asynchronous, options, message
):
    with pytest.raises(ValueError, match=message):
        if asynchronous:
            await store.asimilarity_search_by_vector([1.0, 0.0], **options)
        else:
            store.similarity_search_by_vector([1.0, 0.0], **options)
    sdk.query.assert_not_called()


def test_sqlnet_sdk_rejected_before_preparing_table(sdk):
    sdk.config = SimpleNamespace(has_sqlnet_connection=True, has_rest_connection=False)
    with pytest.raises(ValueError, match="SQLNet-configured.*oracledb connection"):
        OracleVS(sdk, Embedding(), "DOCS")
    sdk.describe_vector_table.assert_not_called()
    sdk.create_vector_table.assert_not_called()


@pytest.mark.asyncio
async def test_async_sqlnet_sdk_rejected_before_preparing_table(sdk):
    sdk.config = SimpleNamespace(has_sqlnet_connection=True, has_rest_connection=False)
    with pytest.raises(ValueError, match="SQLNet-configured"):
        await OracleVS.acreate(sdk, Embedding(), "DOCS")
    sdk.describe_vector_table.assert_not_called()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_native_facade_submits_projected_search_request(store, asynchronous):
    transport = (AsyncNativeTransport if asynchronous else NativeTransport)(
        None, (23, 26, 4)
    )
    transport.call = MagicMock(return_value={"items": []})
    transport.acall = AsyncMock(return_value={"items": []})
    store._vecdb.transport = transport
    if asynchronous:
        await store.asimilarity_search_by_vector(
            [1.0, 0.0], output_selector=[], advanced_options={"accuracy": 75}
        )
        submitted = transport.acall.call_args.kwargs
    else:
        store.similarity_search_by_vector(
            [1.0, 0.0], output_selector=[], advanced_options={"accuracy": 75}
        )
        submitted = transport.call.call_args.kwargs
    assert submitted == {
        "table_name": "DOCS",
        "query_by": {"vector": [1.0, 0.0]},
        "top_k": 4,
        "filters": None,
        "include_vectors": False,
        "output_selector": ["__orcl_internal_content", LOGICAL_ID_KEY],
        "advanced_options": {"accuracy": 75, "distance_metric": store._vecdb.metric},
    }


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "failure", ["fatal", "driver", "pending", "autocommit", "duplicates"]
)
@pytest.mark.asyncio
async def test_vecdb_insert_matches_sql(monkeypatch, asynchronous, failure, caplog):
    from contextlib import asynccontextmanager

    from langchain_oracledb.vectorstores import _vecdb

    connection = MagicMock(
        transaction_in_progress=failure == "pending", autocommit=failure == "autocommit"
    )
    cursor = connection.cursor.return_value.__enter__.return_value
    errors = [SimpleNamespace(code=1, offset=1, message="duplicate ID")]
    if failure == "fatal":
        errors.append(SimpleNamespace(code=51803, offset=2, message="wrong dimension"))
    cursor.getbatcherrors.return_value = errors
    if asynchronous:
        cursor.executemany = AsyncMock()
        connection.commit = AsyncMock()
        connection.rollback = AsyncMock()
    if failure == "driver":
        cursor.executemany.side_effect = RuntimeError("driver failure")

    @contextmanager
    def get(_):
        yield connection

    @asynccontextmanager
    async def aget(_):
        yield connection

    monkeypatch.setattr(_vecdb, "_get_connection", get)
    monkeypatch.setattr(_vecdb, "_aget_connection", aget)
    transport = (AsyncNativeTransport if asynchronous else NativeTransport)(
        None, (23, 26, 4)
    )
    transport.columns = MagicMock(return_value=CURRENT_COLUMNS)
    transport.acolumns = AsyncMock(return_value=CURRENT_COLUMNS)
    records = [
        {"id": key, "dense_vector": [1.0], "metadata": {"__orcl_internal_content": key}}
        for key in ["valid", "duplicate", "other"]
    ]

    async def insert():
        if asynchronous:
            return await transport.ainsert("DOCS", records)
        return transport.insert("DOCS", records)

    if failure == "driver":
        with pytest.raises(RuntimeError, match="driver failure"):
            await insert()
        connection.commit.assert_not_called()
    else:
        assert await insert() == (
            ["valid"] if failure == "fatal" else ["valid", "other"]
        )
        connection.commit.assert_called_once()
        assert "duplicate ID" in caplog.text
        if failure == "fatal":
            assert "wrong dimension" in caplog.text
    cursor.executemany.assert_called_once()
    connection.rollback.assert_not_called()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_migration_rechecks_compatible_before_any_ddl(monkeypatch, asynchronous):
    from contextlib import asynccontextmanager

    from langchain_oracledb.vectorstores import _vecdb

    connection = MagicMock(version="23.26.4", transaction_in_progress=False)
    cursor = connection.cursor.return_value.__enter__.return_value
    cursor.var.return_value.getvalue.return_value = "23.5.0"
    if asynchronous:
        cursor.execute = AsyncMock()

    @contextmanager
    def get(_):
        yield connection

    @asynccontextmanager
    async def aget(_):
        yield connection

    monkeypatch.setattr(_vecdb, "_get_connection", get)
    monkeypatch.setattr(_vecdb, "_aget_connection", aget)
    transport = (AsyncNativeTransport if asynchronous else NativeTransport)(
        None, (23, 26, 1)
    )
    with pytest.raises(ValueError, match="COMPATIBLE >= 23.6"):
        if asynchronous:
            await transport.amigrate("DOCS")
        else:
            transport.migrate("DOCS")
    assert [call.args[0] for call in cursor.execute.call_args_list] == [
        _vecdb.COMPATIBILITY_SQL
    ]


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("params", [None, {}])
@pytest.mark.asyncio
async def test_public_rest_index_creation_is_noop(store, sdk, asynchronous, params):
    from langchain_oracledb.vectorstores.oraclevs import acreate_index, create_index

    sdk.create_index = MagicMock(
        side_effect=AssertionError("VecDB owns vector indexes")
    )
    if asynchronous:
        result = await acreate_index(sdk, store, params)
    else:
        result = create_index(sdk, store, params)
    assert result is None
    sdk.query.assert_not_called()


@pytest.mark.parametrize("include_vectors", [False, True])
def test_public_sdk_and_native_provider_share_query_and_result_contract(
    sdk, include_vectors
):
    """Run the same fixture through the public SDK and adapted native provider."""
    from langchain_oracledb.vectorstores._vecdb_adapter import (
        SQLNetAdapter,
        search_request,
    )

    payload = search_request(
        "DOCS",
        "COSINE",
        [1.0, 0.0],
        2,
        {"tenant": "one"},
        include_vectors,
        {"output_selector": ["tenant"], "advanced_options": {"accuracy": 90}},
    )
    hit = {
        "id": "fixture",
        "metadata": {"__orcl_internal_content": "Oracle λ", "tenant": "one"},
        "distance": 0.25,
    }
    if include_vectors:
        hit["vector"] = [1.0, 0.0]
    service = MagicMock()
    service.query.return_value = {"items": [hit]}
    # Restore the public method that the SDK fixture normally records directly.
    with patch.object(sdk, "_get_active_service", return_value=service):
        public_response = type(sdk).query(sdk, **payload)
    assert service.query.call_args.kwargs == {**payload, "debug_flags": None}
    _, binds = SQLNetAdapter((23, 26, 3)).statement("search", payload)
    for field in ("query_by", "filters", "advanced_options", "output_selector"):
        assert json.loads(binds[field + "_json"]) == payload[field]
    backend = VecDBBackend(
        NativeTransport(None, (23, 26, 3)), "DOCS", DistanceStrategy.COSINE
    )
    public_results = backend.search_results(public_response, include_vectors)
    native_results = backend.search_results({"results": [hit]}, include_vectors)
    assert public_results[0][:2] == native_results[0][:2]
    if include_vectors:
        assert public_results[0][2].tolist() == native_results[0][2].tolist()


def test_sdk_public_ords_configuration_contract(sdk):
    sdk.config = SimpleNamespace(has_sqlnet_connection=False, has_rest_connection=True)
    store = OracleVS(sdk, Embedding(), "DOCS", mutate_on_duplicate=True)
    assert store.backend == "vecdb"
    sdk.describe_vector_table.assert_called_once()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("params", [None, {}])
@pytest.mark.asyncio
async def test_public_native_index_creation_is_noop(store, asynchronous, params):
    from langchain_oracledb.vectorstores.oraclevs import acreate_index, create_index

    transport = (AsyncNativeTransport if asynchronous else NativeTransport)(
        None, (23, 26, 4)
    )
    transport.call = MagicMock(side_effect=AssertionError("VecDB owns vector indexes"))
    transport.acall = AsyncMock(side_effect=AssertionError("VecDB owns vector indexes"))
    transport.describe_index = MagicMock(
        return_value={"ready": True, "details": {"indexes": [{"index_name": "AUTO"}]}}
    )
    transport.adescribe_index = AsyncMock(
        return_value={"ready": True, "details": {"indexes": [{"index_name": "AUTO"}]}}
    )
    store._vecdb.transport = transport
    if asynchronous:
        result = await acreate_index(None, store, params)
    else:
        result = create_index(None, store, params)
    assert result is None
    transport.call.assert_not_called()
    transport.acall.assert_not_called()


def test_sdk_without_public_transport_configuration_is_rejected(sdk):
    sdk.config = SimpleNamespace(has_sqlnet_connection=False, has_rest_connection=False)
    with pytest.raises(ValueError, match="ORDS-configured"):
        OracleVS(sdk, Embedding(), "DOCS")
    sdk.describe_vector_table.assert_not_called()
    sdk.create_vector_table.assert_not_called()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("factory", [False, True], ids=["constructor", "from_texts"])
@pytest.mark.asyncio
async def test_public_vecdb_creation_options_forwarded(sdk, asynchronous, factory):
    error = RuntimeError("missing")
    error.status = 404
    sdk.describe_vector_table.side_effect = error
    options = {"idx_type": "IVF", "neighbor_part": 4, "accuracy": 85}
    kwargs = {
        "client": sdk,
        "table_name": "DOCS",
        "backend": "vecdb",
        "mutate_on_duplicate": True,
        "params": {"vector_index_params": options},
    }
    if factory:
        if asynchronous:
            store = await OracleVS.afrom_texts(["oracle"], Embedding(), **kwargs)
        else:
            store = OracleVS.from_texts(["oracle"], Embedding(), **kwargs)
    elif asynchronous:
        store = await OracleVS.acreate(embedding_function=Embedding(), **kwargs)
    else:
        store = OracleVS(embedding_function=Embedding(), **kwargs)
    assert store.backend == "vecdb"
    sdk.create_vector_table.assert_called_once()
    mapped = sdk.create_vector_table.call_args.kwargs["index_params"][
        "vector_index_params"
    ]
    assert mapped["organization"] == "PARTITIONS"
    assert mapped["accuracy"] == 85
    assert mapped["advanced_params"] == {"partitions": 4}
    assert "distribute_params" not in mapped
    assert options == {"idx_type": "IVF", "neighbor_part": 4, "accuracy": 85}


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "filters",
    [
        {"payload": {"$eq": {"a": 1}}},
        {"drinks": {"$ne": ["tea"]}},
        {"items[*].tags": {"$eq": ["tea", "soda"]}},
    ],
)
@pytest.mark.asyncio
async def test_unsupported_filters_fail_before_sdk_query(
    store, sdk, asynchronous, filters
):
    with pytest.raises(ValueError, match="Unsupported VecDB filter"):
        if asynchronous:
            await store.asimilarity_search("query", filter=filters)
        else:
            store.similarity_search("query", filter=filters)
    sdk.query.assert_not_called()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("return_scores", [False, True])
@pytest.mark.asyncio
async def test_text_retriever_preserves_provider_scores(
    store, asynchronous, return_scores
):
    expected = [Document(id="z", page_content="oracle", metadata={"score": 37})]
    store._vecdb.text_search = MagicMock(return_value=expected)
    store._vecdb.run_async = AsyncMock(return_value=expected)
    retriever = OracleTextSearchRetriever(
        vector_store=store, return_scores=return_scores
    )
    hits = (
        await retriever.ainvoke("oracle")
        if asynchronous
        else retriever.invoke("oracle")
    )
    assert hits[0].metadata["score"] == 37
    if asynchronous:
        store._vecdb.run_async.assert_awaited_once_with(
            "text_search", '"oracle"', 4, return_scores
        )
    else:
        store._vecdb.text_search.assert_called_once_with('"oracle"', 4, return_scores)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "arguments,message",
    [
        ({"table_name": "OTHER", "column_name": "metadata"}, "Only give one"),
        ({"column_name": "metadata"}, "column_name must be 'text'"),
    ],
)
@pytest.mark.asyncio
async def test_vecdb_text_index_validates_common_arguments_first(
    store, sdk, asynchronous, arguments, message
):
    from langchain_oracledb.retrievers.text_search import (
        acreate_text_index,
        create_text_index,
    )

    with pytest.raises(ValueError, match=message):
        if asynchronous:
            await acreate_text_index(sdk, "IDX", vector_store=store, **arguments)
        else:
            create_text_index(sdk, "IDX", vector_store=store, **arguments)
    sdk.query.assert_not_called()


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_uuid_chunks_preserve_logical_ids_through_crud(
    store, sdk, native, asynchronous
):
    """Exercise public chunking with a transport enforcing VecDB's ID limit."""
    from copy import deepcopy
    from uuid import UUID

    records = {}

    def execute(operation, **params):
        if operation == "upsert_vectors":
            for record in params["vectors"]:
                assert len(record["id"].encode("utf-8")) == 40
                records[record["id"]] = deepcopy(record)
            return {}
        if operation == "delete_vectors":
            for id_ in params["ids"]:
                records.pop(id_, None)
            return {}
        if operation == "list_vectors":
            return {"items": [records[id_] for id_ in params["ids"] if id_ in records]}
        if operation == "search":
            selector = params.get("output_selector")
            return {
                "items": [
                    {
                        **record,
                        "distance": 0.0,
                        "metadata": {
                            key: value
                            for key, value in record["metadata"].items()
                            if selector is None or key in selector
                        },
                    }
                    for record in records.values()
                ]
            }
        raise AssertionError(operation)

    if native:
        transport = (AsyncNativeTransport if asynchronous else NativeTransport)(
            None, (23, 26, 3)
        )
        transport.call = MagicMock(side_effect=execute)
        transport.acall = AsyncMock(side_effect=execute)
        store._vecdb.transport = transport
    else:
        sdk.upsert_vectors.side_effect = lambda **params: execute(
            "upsert_vectors", **params
        )
        sdk.list_vectors.side_effect = lambda **params: execute(
            "list_vectors", **params
        )
        sdk.delete_vectors.side_effect = lambda **params: execute(
            "delete_vectors", **params
        )
        sdk.query.side_effect = lambda **params: execute("search", **params)

    class Splitter:
        def split_text(self, text):
            return text.split("|")

    parent_id = str(UUID("12345678-1234-5678-1234-567812345678"))
    source = Document(
        id=parent_id, page_content="first|second", metadata={"tenant": "one"}
    )
    expected_ids = [f"{parent_id}#chunk-{i}" for i in range(2)]
    if asynchronous:
        ids = await store.aadd_documents([source], text_splitter=Splitter())
        fetched = await store.aget_by_ids(ids)
        hits = await store.asimilarity_search("query", output_selector=["tenant"])
    else:
        ids = store.add_documents([source], text_splitter=Splitter())
        fetched = store.get_by_ids(ids)
        hits = store.similarity_search("query", output_selector=["tenant"])
    assert ids == expected_ids
    assert len(records) == 2
    assert list(records) == [storage_id(id_) for id_ in expected_ids]
    assert [doc.id for doc in fetched] == expected_ids
    assert [doc.id for doc in hits] == expected_ids
    assert [doc.metadata for doc in fetched] == [
        {"tenant": "one", "source_doc_index": 0, "chunk_index": i} for i in range(2)
    ]
    assert all(doc.metadata == {"tenant": "one"} for doc in hits)
    assert [doc.page_content for doc in fetched] == ["first", "second"]
    fetched[0].page_content = "updated"
    if asynchronous:
        assert await store.aadd_documents(fetched) == expected_ids
        updated = await store.aget_by_ids(ids)
        await store.adelete(ids)
        assert await store.aget_by_ids(ids) == []
    else:
        assert store.add_documents(fetched) == expected_ids
        updated = store.get_by_ids(ids)
        store.delete(ids)
        assert store.get_by_ids(ids) == []
    assert updated[0].page_content == "updated"
    assert records == {}
    assert source.metadata == {"tenant": "one"}


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_insert_only_returns_successful_logical_ids(asynchronous):
    transport = (AsyncNativeTransport if asynchronous else NativeTransport)(
        None, (23, 26, 3)
    )
    transport.insert = MagicMock(return_value=[storage_id("second")])
    transport.ainsert = AsyncMock(return_value=[storage_id("second")])
    backend = VecDBBackend(transport, "DOCS", DistanceStrategy.COSINE)
    args = (["one", "two"], [[1.0], [2.0]], [{}, {}], ["first", "second"], False)
    result = (
        await backend.run_async("add", *args) if asynchronous else backend.add(*args)
    )
    assert result == ["second"]
    call = transport.ainsert.call_args if asynchronous else transport.insert.call_args
    assert [r["id"] for r in call.args[1]] == [
        storage_id("first"),
        storage_id("second"),
    ]


@pytest.mark.parametrize(
    "logical_id", ["a", "12345678-1234-5678-1234-567812345678#chunk-0", "文档"]
)
def test_storage_ids_hash_every_logical_id(logical_id):
    import hashlib

    assert (
        storage_id(logical_id)
        == hashlib.sha256(logical_id.encode("utf-8")).hexdigest()[:40]
    )
    assert len(storage_id(logical_id)) == 40


def test_backend_rejects_reserved_logical_id_metadata():
    transport = MagicMock()
    backend = VecDBBackend(transport, "DOCS", DistanceStrategy.COSINE)
    with pytest.raises(ValueError, match="reserved"):
        backend.add(["text"], [[1.0]], [{LOGICAL_ID_KEY: "spoofed"}], ["a"], True)
    transport.call.assert_not_called()
