"""Pin copied SDK request contracts independently of transport execution."""

from copy import deepcopy

import pytest

from langchain_oracledb._vecdb_sdk import parameter_validation as validation
from langchain_oracledb._vecdb_sdk.services.ords.hybrid_search import HybridSearch
from langchain_oracledb._vecdb_sdk.services.sqlnet.query_mapper import SQLNetQueryMapper


class CatalogHybrid(HybridSearch):
    def _describe_table_embed_params(self, table_name):
        return {"model": "EMBED_MODEL", "embed_metadata_jsonpath": "$.body.text"}

    def _hybrid_index_rows_for_table(self, table_name):
        return [{"idx_name": "docs_hvi"}]


def request(**changes):
    args = dict(
        table_name="DOCS",
        query_kind="text",
        query_value="oracle",
        top_k=3,
        filters=None,
        advanced_options=None,
        hybrid_options={},
    )
    args.update(changes)
    return CatalogHybrid()._hybrid_search_request(**args)


def test_hybrid_catalog_discovery_and_request_do_not_mutate_caller_options():
    options = {"contains": "oracle", "vector": {"aggregator_params": ["AVGN", 4, 20]}}
    original = deepcopy(options)
    result = request(hybrid_options=options, filters={"tenant": "a"})
    assert options == original
    assert result["hybrid_index_name"] == "DOCS_HVI"
    assert result["text"] == {"inpath": ["$.body.text"], "contains": "oracle"}
    assert result["vector"]["search_text"] == "oracle"
    assert result["filter_by"] == {
        "op": "=",
        "type": "string",
        "path": "CONTENT_METADATA.tenant",
        "args": ["a"],
    }
    assert result["return"]["topN"] == 3


def test_hybrid_cached_configuration_and_vector_source():
    service = CatalogHybrid()
    embed = {"model": "MODEL", "embed_metadata_jsonpath": "text"}
    service._hybrid_index_configs["DOCS"] = {
        "embed_params": embed,
        "hybrid_index_name": "cached_hvi",
    }
    result = service._hybrid_search_request("docs", "vector", [1, 0], 2, None, None, {})
    assert result["hybrid_index_name"] == "CACHED_HVI"
    assert result["vector"] == {"search_vector": "[1, 0]"}
    assert result["text"] == {"inpath": ["$.text"]}
    assert embed == {"model": "MODEL", "embed_metadata_jsonpath": "text"}


@pytest.mark.parametrize(
    "options,message",
    [
        ([], "hybrid_options"),
        ({"return": []}, "return"),
        ({"return": {"format": "XML"}}, "format=JSON"),
        ({"return": {"topN": 0}}, "positive integer"),
        ({"text": []}, "text and vector"),
        ({"vector": []}, "text and vector"),
        ({"text": {"inpath": []}}, "JSON paths"),
        ({"contains": "a", "text": {"search_text": "b"}}, "Conflicting text"),
        ({"filter_by": {}}, "not both"),
    ],
)
def test_hybrid_request_rejects_ambiguous_or_unmappable_inputs(options, message):
    with pytest.raises((TypeError, ValueError), match=message):
        request(hybrid_options=options, filters={"tenant": "a"})


@pytest.mark.parametrize(
    "options",
    [
        {"search_text": "x"},
        {"vector": {"search_text": "x"}},
        {"vector": {"search_vector": [0, 1]}},
    ],
)
def test_vector_query_has_exactly_one_source(options):
    with pytest.raises(ValueError, match="Conflicting vector"):
        request(query_kind="vector", query_value=[1, 0], hybrid_options=options)


def test_top_level_search_text_is_preserved_and_advanced_options_are_separate():
    result = request(hybrid_options={"search_text": "chosen"})
    assert result["search_text"] == "chosen"
    assert "search_text" not in result["text"]
    with pytest.raises(ValueError, match="hybrid_options"):
        request(advanced_options={"accuracy": 90})


def test_index_discovery_requires_an_unambiguous_index(monkeypatch):
    service = CatalogHybrid()
    monkeypatch.setattr(service, "_hybrid_index_rows_for_table", lambda _: [])
    with pytest.raises(ValueError, match="exactly one HVI"):
        service._hybrid_search_request("DOCS", "text", "oracle", 1, None, None, {})
    for method in [
        HybridSearch()._describe_table_embed_params,
        HybridSearch()._hybrid_index_rows_for_table,
    ]:
        with pytest.raises(NotImplementedError, match="adapter must supply"):
            method("DOCS")


@pytest.mark.parametrize(
    "value",
    [None, {}, {"model": " "}, {"model": "MODEL", "embed_metadata_jsonpath": " "}],
)
def test_embed_configuration_requires_model_and_path(value):
    with pytest.raises(ValueError, match="requires"):
        HybridSearch._require_hybrid_embed_params(value, "creation")


def test_embed_configuration_is_a_defensive_copy():
    original = {"model": "MODEL", "embed_metadata_jsonpath": "text", "other": [1]}
    result = HybridSearch._require_hybrid_embed_params(original, "creation")
    result["other"].append(2)
    assert original["other"] == [1]


@pytest.mark.parametrize(
    "query", [None, {"text": " "}, {}, {"text": "x", "vector": [1]}]
)
def test_query_source_must_be_unique_and_nonempty(query):
    with pytest.raises((TypeError, ValueError)):
        HybridSearch._hybrid_search_input(query)


@pytest.mark.parametrize("path", [None, "", "$[0]", "$.bad-key", "$.a..b"])
def test_metadata_paths_are_not_sql_fragments(path):
    with pytest.raises(ValueError):
        HybridSearch._json_path_literal(path)


@pytest.mark.parametrize("value", [None, "", "two words", "A;DROP_TABLE", "SCHEMA.IDX"])
def test_hybrid_index_name_is_a_simple_identifier(value):
    with pytest.raises(ValueError):
        HybridSearch._identifier(value, "index")


@pytest.mark.parametrize("value", [[], ["text"], [3]])
def test_text_inpath_must_contain_json_paths(value):
    with pytest.raises(ValueError):
        HybridSearch._validate_json_path_list(value, "inpath")


@pytest.mark.parametrize("value", [0, True, 1.5, "2"])
def test_top_k_is_a_positive_integer(value):
    with pytest.raises(ValueError):
        HybridSearch._positive_integer(value, "top_k")


@pytest.mark.parametrize(
    "filter_,expected",
    [
        (
            {"age": {"$between": [18, 65]}},
            {
                "op": "BETWEEN",
                "type": "number",
                "path": "CONTENT_METADATA.age",
                "args": [18, 65],
            },
        ),
        (
            {"tenant": {"$in": ["a", "b"]}},
            {
                "op": "IN",
                "type": "string",
                "path": "CONTENT_METADATA.tenant",
                "args": ["a", "b"],
            },
        ),
        (
            {"flag": True},
            {
                "op": "=",
                "type": "string",
                "path": "CONTENT_METADATA.flag",
                "args": [True],
            },
        ),
        (
            {"field": {"$exists": True}},
            {"op": "EXISTS", "path": "CONTENT_METADATA.field", "args": []},
        ),
        (
            {"field": {"$exists": False}},
            {
                "op": "NOT",
                "args": [
                    {"op": "EXISTS", "path": "CONTENT_METADATA.field", "args": []}
                ],
            },
        ),
    ],
)
def test_hybrid_filter_operators_have_stable_json_contracts(filter_, expected):
    assert HybridSearch._hybrid_filter(filter_) == expected


def test_nested_hybrid_filters_preserve_logic_and_numeric_bounds():
    result = HybridSearch._hybrid_filter(
        {"$or": [{"age": {"$gte": 18, "$lt": 65}}, {"$not": {"address.city": "NYC"}}]}
    )
    assert result["op"] == "OR"
    bounds, negation = result["args"]
    assert bounds["op"] == "AND"
    assert [entry["op"] for entry in bounds["args"]] == [">=", "<"]
    assert negation == {
        "op": "NOT",
        "args": [
            {
                "op": "=",
                "type": "string",
                "path": "CONTENT_METADATA.address.city",
                "args": ["NYC"],
            }
        ],
    }
    assert HybridSearch._hybrid_filter({"a": 1, "b": "x"})["op"] == "AND"


@pytest.mark.parametrize(
    "value",
    [
        None,
        {},
        {"$unknown": []},
        {"$and": []},
        {"$or": {}},
        {"bad-key": 1},
        {"a": {}},
        {"a": {"$unknown": 1}},
        {"a": {"$exists": 1}},
        {"a": {"$between": [1]}},
        {"a": {"$in": []}},
    ],
)
def test_malformed_filters_fail_before_transport_execution(value):
    with pytest.raises(ValueError):
        HybridSearch._hybrid_filter(value)


@pytest.mark.parametrize(
    "args",
    [
        {"include_paths": [" "]},
        {"exclude_paths": ["tags[*]"]},
        {"include_paths": ["*"], "exclude_paths": ["*"]},
    ],
)
def test_metadata_index_paths_reject_ambiguous_or_array_policies(args):
    with pytest.raises(ValueError):
        validation.MetadataIndexParams(**args)


@pytest.mark.parametrize(
    "params",
    [
        {
            "organization": "PARTITIONS",
            "distribute_params": {"distribute_method": "AUTO"},
        },
        {"online_build": True},
        {"compression_ratio": 4},
        {"quantization_type": "SCALAR"},
        {"quantization_type": "SCALAR", "compression_ratio": 3},
        {"advanced_params": {"partitions": 0}},
        {"advanced_params": {"neighbors": 5}},
        {
            "organization": "INMEMORY GRAPH",
            "distribute_params": {"distribute_method": "AUTO"},
            "advanced_params": {"algorithm": "unknown"},
        },
    ],
)
def test_vector_index_option_dependencies_are_enforced(params):
    with pytest.raises(ValueError):
        validation.validate_index_params({"vector_index_params": params})


def test_supported_quantized_hnsw_and_ivf_parameters_are_not_rewritten():
    for params in [
        {"advanced_params": {"partitions": 16}},
        {
            "organization": "INMEMORY GRAPH",
            "distribute_params": {"distribute_method": "AUTO"},
            "advanced_params": {
                "neighbors": 32,
                "efConstruction": 200,
                "algorithm": "uniform_quantization",
            },
        },
    ]:
        params.update(quantization_type="SCALAR", compression_ratio=4)
        original = deepcopy(params)
        validation.validate_index_params({"vector_index_params": params})
        assert params == original
    validation.validate_embed_params(
        {"model": "MODEL", "embed_metadata_jsonpath": "text"}
    )
    validation.validate_index_params(
        {"indexing": "AUTO", "distance": "COSINE", "parallel_creation": 2}
    )


@pytest.mark.parametrize(
    "params",
    [{"indexing": "sometimes"}, {"distance": "COSINE", "distance_metric": "DOT"}],
)
def test_legacy_index_options_cannot_disagree(params):
    with pytest.raises(ValueError):
        validation.validate_index_params(params)


@pytest.mark.parametrize(
    "operation,params,error",
    [
        ("create_vector_table", {"comment": 123}, TypeError),
        ("create_vector_table", {"table_params": {"auto_generate_id": 1}}, TypeError),
        ("create_index", {"index_params": []}, ValueError),
        ("list_vectors", {"limit": 1, "offset": -1}, ValueError),
        ("list_vectors", {"limit": 1, "ids": [1]}, ValueError),
    ],
)
def test_operation_validation_rejects_wrong_public_parameter_types(
    operation, params, error
):
    with pytest.raises(error):
        validation.validate_operation_arguments(operation, params)


def test_mapper_reports_unknown_operations_and_formats_only_explicit_templates(
    monkeypatch,
):
    with pytest.raises(KeyError, match="No SQLNet query"):
        SQLNetQueryMapper.get_query("missing")
    monkeypatch.setitem(
        SQLNetQueryMapper._QUERIES, "format_fixture", "SELECT {expression}\n FROM dual"
    )
    assert (
        SQLNetQueryMapper.get_query("format_fixture", expression="1")
        == "SELECT 1 FROM dual"
    )
