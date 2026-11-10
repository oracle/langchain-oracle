# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""Oracle reranking contracts, parameter forwarding, and document identity."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import oracledb
import pytest
from langchain_core.documents import Document
from pydantic import ValidationError

from langchain_oracledb import OracleReranker


@pytest.fixture
def documents():
    return [
        Document(page_content="same text", metadata={"position": i}, id=str(i))
        for i in range(3)
    ]


@pytest.fixture
def native():
    connection = MagicMock(spec=oracledb.Connection)
    cursor = connection.cursor.return_value.__enter__.return_value
    cursor.fetchone.return_value = (
        json.dumps(
            [
                {"index": "2", "score": "0.9", "content": "ignored"},
                {"index": "0", "score": "0.1"},
                {"index": "1", "score": "0.9"},
            ]
        ),
    )
    return connection, cursor


@pytest.fixture
def rest():
    sdk = pytest.importorskip("oracle_vecdb")
    client = sdk.OracleVecDB(
        sdk.Configuration(
            rest_url="https://example.com/ords/test/_/db-api/stable/vecdb/",
            access_token="test",
        )
    )
    client.rerank = MagicMock(
        return_value=SimpleNamespace(
            to_dict=MagicMock(
                return_value={
                    "items": [{"index": 2, "score": 0.9}, {"index": 0, "score": 0.1}]
                }
            )
        )
    )
    client.close = MagicMock()
    return client


def make_reranker(connection, **kwargs):
    return OracleReranker(
        conn=connection,
        params=kwargs.pop("params", {"provider": "database", "model": "RERANK_MODEL"}),
        **kwargs,
    )


def test_native_orders_preserves_identity_and_ties(native, documents):
    connection, cursor = native
    original_metadata = [dict(doc.metadata) for doc in documents]
    result = make_reranker(connection, top_n=2).compress_documents(documents, "query")
    assert len(result) == 2
    assert result[0] is documents[1]
    assert result[1] is documents[2]
    assert [doc.metadata for doc in documents] == original_metadata
    assert [doc.id for doc in result] == ["1", "2"]
    connection.cursor.return_value.__exit__.assert_called_once()
    connection.close.assert_not_called()


@pytest.mark.parametrize(
    "provider, limit_name",
    [
        ("database", "top_n"),
        ("cohere", "top_n"),
        ("vertexai", "topN"),
        ("ocigenai", "topN"),
    ],
)
def test_native_binds_long_inputs_and_normalizes_limit(
    native, documents, provider, limit_name
):
    connection, cursor = native
    params = {
        "provider": provider,
        "model": "MODEL_'_QUOTE",
        "top_n": 100,
        "topN": 200,
        "credential_name": "MY_CRED",
    }
    before = dict(params)
    documents[0].page_content = "long text " * 10000
    query = "query's long text " * 10000
    make_reranker(connection, params=params, top_n=10).compress_documents(
        documents, query
    )
    cursor.execute.assert_called_once()
    (sql,), binds = cursor.execute.call_args
    assert "DBMS_VECTOR_CHAIN.UTL_TO_RERANK" in sql
    assert "RETURNING CLOB" in sql
    assert query not in sql and params["model"] not in sql
    assert binds["query"] == query
    assert json.loads(binds["documents"]) == {
        "documents": [doc.page_content for doc in documents]
    }
    forwarded = json.loads(binds["params"])
    assert forwarded[limit_name] == 3
    assert ("topN" if limit_name == "top_n" else "top_n") not in forwarded
    assert forwarded["credential_name"] == "MY_CRED"
    assert params == before
    cursor.setinputsizes.assert_called_once_with(
        query=oracledb.DB_TYPE_CLOB,
        documents=oracledb.DB_TYPE_CLOB,
        params=oracledb.DB_TYPE_CLOB,
    )


def test_native_reads_lob_before_cursor_closes(native, documents):
    connection, cursor = native
    lob = MagicMock()
    lob.read.return_value = '[{"index": 0, "score": 1}]'
    cursor.fetchone.return_value = (lob,)
    events = []
    lob.read.side_effect = lambda: events.append("read") or '[{"index": 0, "score": 1}]'
    connection.cursor.return_value.__exit__.side_effect = lambda *args: events.append(
        "close"
    )
    assert (
        make_reranker(connection).compress_documents(documents, "query")
        == documents[:1]
    )
    assert events == ["read", "close"]


def test_native_pool_releases_checkout(native, documents):
    connection, _ = native
    pool = MagicMock(spec=oracledb.ConnectionPool)
    checkout = pool.acquire.return_value
    checkout.__enter__.return_value = connection
    make_reranker(pool).compress_documents(documents, "query")
    checkout.__exit__.assert_called_once()
    pool.close.assert_not_called()


@pytest.mark.parametrize("use_rest", [False, True])
def test_empty_input_has_no_side_effects(native, use_rest, request):
    client = request.getfixturevalue("rest") if use_rest else native[0]
    assert make_reranker(client).compress_documents([], "query") == []
    if use_rest:
        client.rerank.assert_not_called()
    else:
        client.cursor.assert_not_called()


def test_rest_maps_sdk_results(rest, documents):
    result = make_reranker(rest, top_n=1).compress_documents(documents, "query")
    assert len(result) == 1 and result[0] is documents[2]
    rest.rerank.assert_called_once_with(
        query="query",
        documents=[doc.page_content for doc in documents],
        model_name="RERANK_MODEL",
        model_params={"top_n": 3},
    )
    rest.close.assert_not_called()


@pytest.mark.parametrize("use_rest", [False, True])
def test_errors_propagate_without_retry(native, documents, use_rest, request):
    client = request.getfixturevalue("rest") if use_rest else native[0]
    error = RuntimeError("model unavailable")
    call = client.rerank if use_rest else native[1].execute
    call.side_effect = error
    with pytest.raises(RuntimeError) as raised:
        make_reranker(client).compress_documents(documents, "query")
    assert raised.value is error
    call.assert_called_once()


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {},
        "invalid json",
        [{"index": -1, "score": 1}],
        [{"index": 3, "score": 1}],
        [{"index": True, "score": 1}],
        [{"index": 0.5, "score": 1}],
        [{"index": "0.5", "score": 1}],
        [{"index": 0, "score": 1}, {"index": 0, "score": 2}],
        [{"index": 0}],
        [{"index": 0, "score": True}],
        [{"index": 0, "score": "NaN"}],
        [{"index": 0, "score": "Infinity"}],
        [{"score": 1}],
        ["not an object"],
    ],
)
@pytest.mark.parametrize("use_rest", [False, True])
def test_malformed_results_fail(native, documents, payload, use_rest, request):
    client = request.getfixturevalue("rest") if use_rest else native[0]
    if use_rest:
        client.rerank.return_value.to_dict.return_value = {"items": payload}
    else:
        native[1].fetchone.return_value = (json.dumps(payload),)
    with pytest.raises(ValueError):
        make_reranker(client).compress_documents(documents, "query")


def test_missing_database_row_fails(native, documents):
    native[1].fetchone.return_value = None
    with pytest.raises(ValueError, match="missing result"):
        make_reranker(native[0]).compress_documents(documents, "query")


@pytest.mark.parametrize("top_n", [0, -1, True, 1.5])
def test_invalid_top_n_rejected(native, top_n):
    with pytest.raises(ValidationError):
        make_reranker(native[0], top_n=top_n)


@pytest.mark.parametrize(
    "params",
    [{}, {"provider": "database"}, {"model": "M"}, {"provider": "", "model": "M"}],
)
def test_missing_provider_or_model_rejected(native, params):
    with pytest.raises(ValueError):
        make_reranker(native[0], params=params)


def test_rest_rejects_external_provider(rest):
    with pytest.raises(ValueError, match="REST reranking requires"):
        make_reranker(rest, params={"provider": "cohere", "model": "external-model"})


@pytest.mark.asyncio
async def test_async_compression(native, documents):
    result = await make_reranker(native[0], top_n=1).acompress_documents(
        documents, "query"
    )
    assert len(result) == 1 and result[0] is documents[1]


def test_reranker_directory_loading_uses_regression_metadata(native):
    connection, _ = native
    cursor = connection.cursor.return_value
    OracleReranker.load_onnx_model(
        connection, "MODEL_DIR", "model.onnx", "RERANK_MODEL"
    )
    (sql,), binds = cursor.execute.call_args
    assert "drop_model" in sql
    assert binds["path"] == "MODEL_DIR"
    assert binds["filename"] == "model.onnx"
    assert json.loads(binds["metadata"]) == {
        "function": "regression",
        "regressionOutput": "output",
        "input": {"first_input": ["FIRST_INPUT"], "second_input": ["SECOND_INPUT"]},
    }
    cursor.close.assert_called_once()


def test_reranker_native_url_loading_uses_regression_metadata(native):
    connection, _ = native
    cursor = connection.cursor.return_value
    OracleReranker.load_onnx_model(
        connection, model_name="RERANK_MODEL", url="https://example.com/model.onnx"
    )
    (sql,), binds = cursor.execute.call_args
    assert "DBMS_VECTOR_DATABASE.LOAD_MODEL" in sql
    assert "drop_model" not in sql
    assert json.loads(binds["model_params"]) == {"metadata": {"function": "regression"}}


@pytest.mark.parametrize(
    "params",
    [
        None,
        {"credential": "CRED"},
        {"metadata": {"function": "regression", "regressionOutput": "score"}},
    ],
)
def test_reranker_rest_loading_preserves_custom_metadata(rest, params):
    rest.load_model = MagicMock()
    before = json.dumps(params)
    assert (
        OracleReranker.load_onnx_model(
            rest,
            model_name="RERANK_MODEL",
            url="https://example.com/model.onnx",
            model_params=params,
        )
        is None
    )
    expected = dict(params or {})
    expected.setdefault("metadata", {"function": "regression"})
    rest.load_model.assert_called_once_with(
        model_name="RERANK_MODEL",
        url="https://example.com/model.onnx",
        model_params=expected,
    )
    assert json.dumps(params) == before
    rest.close.assert_not_called()


def test_reranker_rest_directory_loading_rejected(rest):
    rest.load_model = MagicMock()
    with pytest.raises(ValueError, match="REST SDK model loading requires url"):
        OracleReranker.load_onnx_model(rest, "MODEL_DIR", "model.onnx", "RERANK_MODEL")
    rest.load_model.assert_not_called()


def test_reranker_load_error_propagates(rest):
    error = RuntimeError("download failed")
    rest.load_model = MagicMock(side_effect=error)
    with pytest.raises(RuntimeError) as raised:
        OracleReranker.load_onnx_model(
            rest, model_name="RERANK_MODEL", url="https://example.com/model.onnx"
        )
    assert raised.value is error
    rest.load_model.assert_called_once()


def test_contextual_compression_retriever(native, documents):
    classic = pytest.importorskip("langchain_classic.retrievers")
    from langchain_core.retrievers import BaseRetriever

    class Retriever(BaseRetriever):
        def _get_relevant_documents(self, query, *, run_manager):
            return documents

    retriever = classic.ContextualCompressionRetriever(
        base_retriever=Retriever(), base_compressor=make_reranker(native[0], top_n=1)
    )
    result = retriever.invoke("query")
    assert len(result) == 1 and result[0] is documents[1]
