# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""Branch-derived validation, SQL binding and LangChain output contracts."""

import json
from contextlib import asynccontextmanager, contextmanager
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from langchain_oracledb.vectorstores._vecdb import (
    AsyncNativeTransport,
    NativeTransport,
    VecDBBackend,
)
from langchain_oracledb.vectorstores._vecdb_adapter import (
    SQLNetAdapter,
    as_dict,
    document,
    hybrid_search_request,
    rows,
)
from langchain_oracledb.vectorstores.utils import DistanceStrategy


@pytest.mark.parametrize("version", [(23, 26, 1), (23, 26, 3)])
def test_query_uses_sdk_template_with_bound_payloads(version):
    params = {
        "table_name": "DOCS",
        "query_by": {"vector": [1.0, 0.0]},
        "top_k": 5,
        "filters": {"tenant": {"$eq": "x' OR 1=1"}},
        "include_vectors": True,
        "output_selector": ["text", "tenant"],
        "advanced_options": {"distance_metric": "COSINE"},
    }
    original = deepcopy(params)
    sql, binds = SQLNetAdapter(version).statement("search", params)
    assert "dbms_vector_database.search(" in sql
    assert "output_selector => JSON(:output_selector_json)" in sql
    assert "x' OR 1=1" not in sql
    assert json.loads(binds["filters_json"]) == params["filters"]
    assert json.loads(binds["query_by_json"]) == params["query_by"]
    assert json.loads(binds["output_selector_json"]) == ["text", "tenant"]
    assert binds["debug_flags_json"] is None
    assert params == original


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_sync_async_driver_gets_same_copied_query_contract(asynchronous):
    connection = MagicMock()
    cursor = connection.cursor.return_value.__enter__.return_value
    cursor.var.return_value.getvalue.return_value = '{"items": []}'

    @contextmanager
    def get(_):
        yield connection

    @asynccontextmanager
    async def aget(_):
        yield connection

    args = {"table_name": "DOCS", "query_by": {"vector": [1.0]}, "top_k": 2}
    if asynchronous:
        cursor.execute = AsyncMock()
        transport = AsyncNativeTransport(object(), (23, 26, 3))
        with patch("langchain_oracledb.vectorstores._vecdb._aget_connection", aget):
            assert await transport.acall("search", **args) == {"items": []}
        assert cursor.execute.await_count == 1
    else:
        transport = NativeTransport(object(), (23, 26, 3))
        with patch("langchain_oracledb.vectorstores._vecdb._get_connection", get):
            assert transport.call("search", **args) == {"items": []}
    cursor.execute.assert_called_once()
    sql = cursor.execute.call_args.args[0]
    assert sql.startswith("BEGIN /* lc_oracledb mode=vector backend=vecdb")
    assert " :result := dbms_vector_database.search(" in sql
    assert cursor.execute.call_args.kwargs["query_by_json"] == '{"vector": [1.0]}'
    assert cursor.setinputsizes.call_args.kwargs["query_by_json"] is not None


def test_index_template_captures_real_package_result():
    sql, binds = SQLNetAdapter((23, 26, 3)).statement(
        "create_index",
        {
            "table_name": "DOCS",
            "index_params": {"metadata_index_params": {"auto_index": True}},
        },
    )
    assert "l_output := dbms_vector_database.create_index(" in sql
    assert ":result := l_output; END;" in sql
    assert json.loads(binds["index_params_json"]) == {
        "metadata_index_params": {"auto_index": True}
    }


@pytest.mark.parametrize(
    "operation, params, message",
    [
        ("search", {"query_by": {"id": "1", "text": "x"}, "top_k": 1}, "exactly one"),
        ("search", {"query_by": {"vector": [1.0]}, "top_k": False}, "top_k"),
        ("list_vectors", {"limit": 0}, "limit"),
        (
            "create_index",
            {
                "index_params": {
                    "vector_index_params": {
                        "organization": "INMEMORY GRAPH",
                        "distribute_params": {"distribute_method": "invalid"},
                    }
                }
            },
            "distribute_params",
        ),
        (
            "create_index",
            {
                "index_params": {
                    "vector_index_params": {
                        "organization": "PARTITIONS",
                        "online_build": True,
                    }
                }
            },
            "online_build",
        ),
        ("create_index", {"index_params": {"unknown": True}}, "Extra inputs"),
    ],
)
def test_native_request_validation_precedes_connection_acquisition(
    operation, params, message
):
    with patch("langchain_oracledb.vectorstores._vecdb._get_connection") as get:
        with pytest.raises(ValueError, match=message):
            NativeTransport(object(), (23, 26, 3)).call(operation, **params)
    get.assert_not_called()


@pytest.mark.parametrize("version", [(23, 26, 1), (23, 26, 3)])
@pytest.mark.parametrize("kind", ["HNSW", "IVF"])
def test_index_translation_satisfies_copied_validation(version, kind):
    backend = VecDBBackend(
        NativeTransport(None, version), "DOCS", DistanceStrategy.COSINE
    )
    params = backend.index_params({"idx_type": kind})
    # Exercise request validation as well as template construction.
    sql, _ = SQLNetAdapter(version).statement(
        "create_index", {"table_name": "DOCS", "index_params": params}
    )
    assert "dbms_vector_database.create_index(" in sql
    vector = params.get("vector_index_params", params)
    assert "distribute_params" not in vector


def test_explicit_distribution_is_preserved():
    backend = VecDBBackend(
        NativeTransport(None, (23, 26, 3)), "DOCS", DistanceStrategy.COSINE
    )
    mapped = backend.index_params(
        {"distribute_params": {"distribute_method": "ROWID RANGE"}}
    )
    assert mapped["vector_index_params"]["distribute_params"] == {
        "distribute_method": "ROWID RANGE"
    }


def test_hybrid_uses_branch_request_shape_and_preserves_extended_options():
    params = {
        "hybrid_index_name": "HVI",
        "vector": {"search_vector": [1.0, 0.0], "aggregator_params": ["AVGN", 5, 50]},
        "text": {"json_textcontains": ["$.text", '"oracle"']},
        "search_fusion": "UNION",
        "return": {"topN": 7},
    }
    original = deepcopy(params)
    result = hybrid_search_request("DOCS", params)
    assert result["vector"]["search_vector"] == "[1.0, 0.0]"
    assert result["vector"]["aggregator_params"] == ["AVGN", 5, 50]
    assert result["text"]["inpath"] == ["$.text"]
    assert result["text"]["json_textcontains"] == ["$.text", '"oracle"']
    assert result["return"]["topN"] == 7
    assert result["return"]["format"] == "JSON"
    assert "rowid" in result["return"]["values"]
    assert params == original


@pytest.mark.parametrize("vector", [[], [True], [float("nan")], [float("inf")], ["1"]])
def test_hybrid_rejects_invalid_vectors_without_native_access(vector):
    with patch("langchain_oracledb.vectorstores._vecdb._get_connection") as get:
        with pytest.raises(ValueError):
            NativeTransport(object(), (23, 26, 3)).hybrid_search(
                "DOCS", {"vector": {"search_vector": vector}}, False
            )
    get.assert_not_called()


@pytest.mark.parametrize("mode", ["VECTOR_ONLY", "TEXT_ONLY", "UNION"])
def test_hybrid_mode_survives_request_adaptation(mode):
    params = {"hybrid_index_name": "HVI", "search_fusion": mode}
    if mode == "TEXT_ONLY":
        params["text"] = {"json_textcontains": ["$.text", "hello"]}
    else:
        params["vector"] = {"search_vector": [1.0]}
    assert hybrid_search_request("DOCS", params)["search_fusion"] == mode


@pytest.mark.parametrize("key", ["items", "results"])
def test_response_adapter_keeps_full_text_id_and_metadata_without_mutation(key):
    item = {
        "id": "a",
        "metadata": {"text": "full document", "tenant": "one"},
        "score": 9,
    }
    original = deepcopy(item)
    response = SimpleNamespace(to_dict=lambda: {key: [item]})
    doc = document(rows(response, "results")[0])
    assert doc.id == "a"
    assert doc.page_content == "full document"
    assert doc.metadata == {"tenant": "one"}
    assert item == original


def test_response_adapter_accepts_pydantic_shape():
    class Response(BaseModel):
        items: list[dict]

    response = Response(items=[{"id": "a", "metadata": {"text": "hello"}}])
    assert as_dict(response)["items"][0]["id"] == "a"


@pytest.mark.parametrize("params", [{"vector": None}, {"text": []}, {"return": None}])
def test_malformed_hybrid_sections_fail_before_native_access(params):
    with patch("langchain_oracledb.vectorstores._vecdb._get_connection") as get:
        with pytest.raises(TypeError, match="dictionary"):
            NativeTransport(object(), (23, 26, 3)).hybrid_search("DOCS", params, False)
    get.assert_not_called()


@pytest.mark.parametrize("index", ["VECIDX$_1_SEARCH_IDX", "MixedCase_HVI", "HVI#DOCS"])
def test_hybrid_request_preserves_physical_catalog_name(index):
    request = hybrid_search_request(
        "DOCS",
        {"hybrid_index_name": index, "vector": {"search_vector": [1.0]}},
    )
    assert request["hybrid_index_name"] == index


@pytest.mark.parametrize(
    "mode, fusion", [("semantic", "VECTOR_ONLY"), ("keyword", "TEXT_ONLY")]
)
def test_retriever_mode_maps_to_explicit_sdk_fusion(mode, fusion):
    from langchain_oracledb.vectorstores._vecdb_adapter import retriever_hybrid_options

    params = {"text": {"snippet": 250}, "vector": {"accuracy": 95}}
    original = deepcopy(params)
    options = retriever_hybrid_options("oracle", mode, "HVI", 3, params, {})
    if mode == "semantic":
        options["vector"]["search_vector"] = [1.0]
    request = hybrid_search_request("DOCS", options)
    assert request["search_fusion"] == fusion
    assert request["return"]["topN"] == 3
    assert request["return"]["format"] == "JSON"
    assert params == original


def test_retriever_overrides_preserve_options_and_do_not_mutate_inputs():
    from langchain_oracledb.vectorstores._vecdb_adapter import retriever_hybrid_options

    params = {"vector": {"accuracy": 90}, "search_fusion": "INTERSECT"}
    overrides = {"vector": {"aggregator_params": ["AVGN", 5, 50]}}
    original = deepcopy(overrides)
    options = retriever_hybrid_options("oracle", "hybrid", "HVI", 7, params, overrides)
    options["vector"]["search_vector"] = [1.0]
    request = hybrid_search_request("DOCS", options)
    assert request["vector"]["aggregator_params"] == ["AVGN", 5, 50]
    assert request["search_fusion"] == "INTERSECT"
    assert overrides == original
    assert params["vector"] == {"accuracy": 90}


def test_retriever_rejects_derived_fields_in_runtime_overrides():
    from langchain_oracledb.vectorstores._vecdb_adapter import retriever_hybrid_options

    with pytest.raises(ValueError, match="derived internally"):
        retriever_hybrid_options(
            "oracle",
            "hybrid",
            "HVI",
            3,
            {},
            {"text": {"json_textcontains": ["$.other", "secret"]}},
        )


def test_hnsw_defaults_include_construction_settings_required_by_live_database():
    backend = VecDBBackend(
        NativeTransport(None, (23, 26, 3)), "DOCS", DistanceStrategy.COSINE
    )
    params = backend.index_params({"idx_type": "HNSW"})["vector_index_params"]
    assert params["advanced_params"] == {"neighbors": 32, "efConstruction": 200}


def test_explicit_sdk_hnsw_settings_override_mapped_defaults():
    backend = VecDBBackend(
        NativeTransport(None, (23, 26, 3)), "DOCS", DistanceStrategy.COSINE
    )
    requested = {"advanced_params": {"neighbors": 16}}
    assert backend.index_params(requested)["vector_index_params"][
        "advanced_params"
    ] == {
        "neighbors": 16,
        "efConstruction": 200,
    }
    assert requested == {"advanced_params": {"neighbors": 16}}
