"""Boundary behavior for serialized responses, empty stores, and cache messages."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import oracledb
import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, Generation

from langchain_oracledb import cache
from langchain_oracledb.storage import OracleByteStore
from langchain_oracledb.utilities.oracleai import output_type_handler
from langchain_oracledb.vectorstores import _vecdb_adapter as adapter


@pytest.mark.parametrize(
    "source,target",
    [
        (oracledb.DB_TYPE_CLOB, oracledb.DB_TYPE_LONG),
        (oracledb.DB_TYPE_BLOB, oracledb.DB_TYPE_LONG_RAW),
        (oracledb.DB_TYPE_NCLOB, oracledb.DB_TYPE_LONG_NVARCHAR),
    ],
)
def test_lob_output_conversion_preserves_fetch_batch(source, target):
    cursor = MagicMock(arraysize=37)
    assert (
        output_type_handler(cursor, SimpleNamespace(type_code=source))
        is cursor.var.return_value
    )
    cursor.var.assert_called_once_with(target, arraysize=37)


def test_non_lob_output_uses_driver_default():
    cursor = MagicMock()
    assert (
        output_type_handler(cursor, SimpleNamespace(type_code=oracledb.DB_TYPE_NUMBER))
        is None
    )
    cursor.var.assert_not_called()


def test_empty_store_writes_do_not_acquire_connection():
    store = OracleByteStore.__new__(OracleByteStore)
    store._client = object()
    assert store.mget([]) == []
    assert store.mset([]) is None
    assert store.mdelete([]) is None


def test_invalid_tool_call_response_is_not_cacheable():
    generation = ChatGeneration(
        message=AIMessage(
            content="",
            invalid_tool_calls=[
                {"id": "call", "name": "tool", "args": "{", "error": "invalid JSON"}
            ],
        )
    )
    assert cache._has_tool_calls([generation])
    assert not cache._has_tool_calls([Generation(text="answer")])


def test_cache_legacy_loader_signature(monkeypatch):
    monkeypatch.setattr(cache, "_LOADS_SUPPORTS_ALLOWED_OBJECTS", False)
    original = [Generation(text="answer")]
    assert cache._loads_generations(cache._dumps_generations(original)) == original


@pytest.mark.parametrize("kind", ["attribute", "type"])
def test_cache_read_only_message_id_is_preserved(kind):
    class Message:
        @property
        def id(self):
            return "original"

        @id.setter
        def id(self, value):
            raise AttributeError() if kind == "attribute" else TypeError()

    message = Message()
    cache._reset_generation_ids([SimpleNamespace(message=message)])
    assert message.id == "original"


@pytest.mark.parametrize("response", [None, [], 1, "[]"])
def test_non_object_vecdb_response_is_rejected(response):
    with pytest.raises(ValueError, match="JSON object"):
        adapter.as_dict(response)


@pytest.mark.parametrize("response", [{}, {"items": None}, {"vectors": {}}])
def test_vecdb_response_requires_result_array(response):
    with pytest.raises(ValueError, match="array"):
        adapter.rows(response, "vectors")


@pytest.mark.parametrize(
    "row",
    [
        {"id": "x", "metadata": {}},
        {"metadata": {"text": "body"}},
        {"id": 1, "metadata": {"text": "body"}},
        {"id": "x", "metadata": {"text": 1}},
    ],
)
def test_vecdb_document_requires_string_id_and_text(row):
    with pytest.raises(ValueError, match="missing its ID or metadata text"):
        adapter.document(row)


@pytest.mark.parametrize(
    "params", [[], {"vector": []}, {"text": "oracle"}, {"return": []}]
)
def test_hybrid_sections_require_objects(params):
    with pytest.raises(TypeError, match="dictionary"):
        adapter.validate_hybrid_request(params)


def test_hybrid_document_mapper_requires_json_format():
    with pytest.raises(ValueError, match="return.format=JSON"):
        adapter.validate_hybrid_request({"return": {"format": "XML"}})


@pytest.mark.parametrize("params", [[], {"vector": ["x"]}, {"text": "oracle"}])
def test_retriever_options_require_objects(params):
    with pytest.raises(TypeError, match="dictionary"):
        adapter.validate_retriever_options(params)


def test_hybrid_json_vector_is_decoded_once():
    result = adapter.hybrid_search_request(
        "DOCS", {"hybrid_index_name": "DOCS_HVI", "vector": {"search_vector": "[1, 0]"}}
    )
    assert result["vector"]["search_vector"] == "[1, 0]"


@pytest.mark.parametrize("backend,length", [("sql", 64), ("vecdb", 40)])
def test_semantic_cache_ids_fit_backend_and_preserve_sql_ids(backend, length):
    instance = cache.OracleSemanticCache.__new__(cache.OracleSemanticCache)
    instance._vector_store = MagicMock(backend=backend)
    instance.update("oracle", "model-a", [Generation(text="answer")])
    first_id = instance._vector_store.add_texts.call_args.kwargs["ids"][0]
    assert first_id == cache._cache_entry_id("oracle", "model-a")[:length]
    instance.update("oracle", "model-b", [Generation(text="another")])
    assert instance._vector_store.add_texts.call_args.kwargs["ids"][0] != first_id


@pytest.mark.parametrize("pool_available", [False, True])
def test_invalid_native_client_has_actionable_type_error(monkeypatch, pool_available):
    from langchain_oracledb.vectorstores import utils

    if not pool_available:
        monkeypatch.delattr(oracledb, "ConnectionPool")
    with pytest.raises(TypeError, match="Expected client of type oracledb.Connection"):
        with utils._get_connection(object()):
            pytest.fail("invalid client was accepted")


@pytest.mark.asyncio
@pytest.mark.parametrize("pool_available", [False, True])
async def test_invalid_async_client_has_actionable_type_error(
    monkeypatch, pool_available
):
    from langchain_oracledb.vectorstores import utils

    if not pool_available:
        monkeypatch.delattr(oracledb, "AsyncConnectionPool")
    with pytest.raises(
        TypeError, match="Expected client of type oracledb.AsyncConnection"
    ):
        async with utils._aget_connection(object()):
            pytest.fail("invalid client was accepted")


@pytest.mark.parametrize("kind", ["database", "runtime", "validation", "unexpected"])
def test_sync_exception_wrapper_retains_original_cause(kind):
    from langchain_oracledb.vectorstores.utils import _handle_exceptions

    cause = {
        "database": oracledb.DatabaseError("database failed"),
        "runtime": RuntimeError("runtime failed"),
        "validation": ValueError("validation failed"),
        "unexpected": TypeError("unexpected failure"),
    }[kind]

    @_handle_exceptions
    def fail():
        raise cause

    expected = ValueError if kind == "validation" else RuntimeError
    with pytest.raises(expected) as caught:
        fail()
    assert caught.value.__cause__ is cause
    assert str(cause) in str(caught.value)


@pytest.mark.parametrize(
    "source,target",
    [
        (oracledb.DB_TYPE_CLOB, oracledb.DB_TYPE_LONG),
        (oracledb.DB_TYPE_NCLOB, oracledb.DB_TYPE_LONG_NVARCHAR),
    ],
)
def test_string_lob_conversion_preserves_fetch_batch(source, target):
    from langchain_oracledb.vectorstores.utils import output_type_string_handler

    cursor = MagicMock(arraysize=17)
    assert (
        output_type_string_handler(cursor, SimpleNamespace(type_code=source))
        is cursor.var.return_value
    )
    cursor.var.assert_called_once_with(target, arraysize=17)


@pytest.mark.parametrize(
    "operator,value",
    [
        ("$between", [1]),
        ("$between", [None, None]),
        ("$in", "a"),
        ("$nin", 1),
        ("$all", None),
        ("$not", "x"),
        ("$unknown", 1),
    ],
)
def test_invalid_sql_comparison_rejected_before_execution(operator, value):
    from langchain_oracledb.vectorstores.oraclevs import _get_comparison_string

    with pytest.raises(ValueError):
        _get_comparison_string(operator, value, [])


@pytest.mark.parametrize(
    "value", [[], {"$unknown": []}, {"$and": {}}, {"field": {"$exists": "true"}}]
)
def test_invalid_sql_filter_rejected_before_execution(value):
    from langchain_oracledb.vectorstores.oraclevs import _generate_where_clause

    with pytest.raises(ValueError):
        _generate_where_clause(value, [])


def test_nested_sql_filter_groups_sibling_conditions():
    from langchain_oracledb.vectorstores.oraclevs import _generate_where_clause

    binds = []
    result = _generate_where_clause({"address": {"city": "NYC", "zip": "10001"}}, binds)
    assert binds == ["NYC", "10001"]
    assert "$.address.city" in result and "$.address.zip" in result
    assert " AND " in result and result.startswith("(") and result.endswith(")")


@pytest.mark.parametrize("name", [None, "", "x;DROP TABLE t", "schema.column"])
def test_text_index_column_requires_simple_identifier(name):
    from langchain_oracledb.retrievers.text_search import _quote_simple_identifier

    with pytest.raises(ValueError):
        _quote_simple_identifier(name, "column")


@pytest.mark.parametrize("value", [True, 0, -1, 1.5, "1"])
def test_text_search_limit_must_be_positive_integer(value):
    from langchain_oracledb.retrievers.text_search import _positive_int

    with pytest.raises(ValueError, match="positive integer"):
        _positive_int(value, "k")


@pytest.mark.parametrize("columns", [set(), {"ID", "TEXT", "EMBEDDING", "METADATA"}])
def test_migration_requires_vecdb_layout(columns):
    from langchain_oracledb.vectorstores._vecdb import _migration_actions

    with pytest.raises(ValueError, match="recognized VecDB layouts"):
        _migration_actions(columns, '"DOCS"')


def test_vecdb_layout_requires_metadata_column():
    from langchain_oracledb.vectorstores._vecdb import BASE_COLUMNS, _layout

    with pytest.raises(ValueError, match="no recognized metadata column"):
        _layout(set(BASE_COLUMNS))


@pytest.mark.parametrize("value", [None, [], "", 1])
def test_hybrid_textcontains_requires_nonempty_array(value):
    from langchain_oracledb._vecdb_sdk.services.ords.hybrid_search import HybridSearch

    with pytest.raises(ValueError, match="non-empty list"):
        HybridSearch._hybrid_json_textcontains(value)


@pytest.mark.parametrize("value", [None, [], {}, 1])
def test_hybrid_filter_expression_requires_nonempty_object(value):
    from langchain_oracledb._vecdb_sdk.services.ords.hybrid_search import HybridSearch

    with pytest.raises(ValueError, match="non-empty dictionaries"):
        HybridSearch._hybrid_filter_expr(value)


@pytest.mark.parametrize("operation", ["execute", "drop_table", ""])
def test_sqlnet_adapter_rejects_unknown_operations(operation):
    with pytest.raises(ValueError, match="Unsupported VecDB operation"):
        adapter.SQLNetAdapter((26, 1)).statement(operation, {})


def test_sqlnet_adapter_rejects_internal_parameter_injection():
    with pytest.raises(ValueError, match="Invalid internal VecDB parameter"):
        adapter.SQLNetAdapter((26, 1))._function_statement(
            "query", {"x); END;--": "value"}
        )


def test_advanced_search_options_require_mapping():
    with pytest.raises(ValueError, match="dictionary"):
        adapter.search_request(
            "DOCS", "COSINE", [1, 0], 2, None, False, {"advanced_options": []}
        )


@pytest.mark.parametrize("ids_function", ["_get_delete_ddl", "_get_by_ids_select"])
def test_sql_id_queries_require_ids(ids_function):
    from langchain_oracledb.vectorstores import oraclevs

    with pytest.raises(ValueError, match="No ids provided"):
        getattr(oraclevs, ids_function)("DOCS", None)


@pytest.mark.parametrize("operation", ["embedding", "summary", "splitter"])
def test_cursor_cleanup_failure_preserves_operation_error(
    operation, monkeypatch, caplog
):
    from langchain_oracledb.document_loaders.oracleai import OracleTextSplitter
    from langchain_oracledb.embeddings.oracleai import OracleEmbeddings
    from langchain_oracledb.utilities.oracleai import OracleSummary

    connection = MagicMock()
    cursor = connection.cursor.return_value
    original = RuntimeError("operation failed")
    cursor.execute.side_effect = original
    cursor.close.side_effect = RuntimeError("cleanup failed")
    monkeypatch.setattr(oracledb, "defaults", MagicMock())
    actions = {
        "embedding": lambda: OracleEmbeddings(
            conn=connection, params={}
        ).embed_documents(["text"]),
        "summary": lambda: OracleSummary(connection, {}).get_summary("text"),
        "splitter": lambda: OracleTextSplitter(connection, {}).split_text("text"),
    }
    with pytest.raises(RuntimeError) as caught:
        actions[operation]()
    assert caught.value is original
    cursor.close.assert_called_once()
    assert "Failed to close Oracle" in caplog.text


@pytest.mark.asyncio
async def test_async_unexpected_exception_retains_cause():
    from langchain_oracledb.vectorstores.utils import _ahandle_exceptions

    cause = TypeError("unexpected")

    @_ahandle_exceptions
    async def fail():
        raise cause

    with pytest.raises(RuntimeError) as caught:
        await fail()
    assert caught.value.__cause__ is cause


@pytest.mark.asyncio
@pytest.mark.parametrize("exists", [False, True])
async def test_async_index_drop_is_idempotent(exists, monkeypatch):
    from unittest.mock import AsyncMock

    from langchain_oracledb.vectorstores import utils

    connection = MagicMock(spec=oracledb.AsyncConnection)
    cursor = connection.cursor.return_value.__enter__.return_value
    cursor.execute = AsyncMock()
    lookup = AsyncMock(return_value=exists)
    monkeypatch.setattr(utils, "_aindex_exists", lookup)
    await utils.adrop_index(connection, "DOCS_IDX")
    lookup.assert_awaited_once_with(connection, '"DOCS_IDX"')
    if exists:
        cursor.execute.assert_awaited_once_with('DROP INDEX "DOCS_IDX"')
    else:
        cursor.execute.assert_not_awaited()
