# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""OracleVS filters must never silently acquire QBE semantics."""

from copy import deepcopy

import pytest

from langchain_oracledb.vectorstores._vecdb_adapter import search_request
from langchain_oracledb.vectorstores._vecdb_filters import (
    SQLFilterRequired,
    translate_filter,
)


@pytest.mark.parametrize(
    "predicate",
    [
        {"drinks": {"$ne": ["soda", "tea"]}},
        {"items[*].tags": {"$eq": ["soda", "tea"]}},
        {"address[0]": {"$ne": {"city": "Mono Vista"}}},
        {"$or": [{"age": 65}, {"payload": {"$eq": {"a": 1}}}]},
    ],
)
def test_unsupported_filters_rejected_at_request_boundary(predicate):
    with pytest.raises(SQLFilterRequired, match="Unsupported VecDB filter"):
        search_request("DOCS", "COSINE", [1.0], 3, predicate, False, {})


def test_nested_paths_and_alias_are_translated_without_mutating_input():
    predicate = {"address": {"city": "NYC"}, "name": {"$instr": "ar"}}
    original = deepcopy(predicate)
    assert translate_filter(predicate) == {
        "$and": [
            {"address.city": {"$string": {"$eq": "NYC"}}},
            {"name": {"$string": {"$hasSubstring": "ar"}}},
        ]
    }
    assert predicate == original


@pytest.mark.parametrize(
    "predicate,message",
    [
        ({"ss')--": "x"}, "Invalid metadata key"),
        ({"drinks": {"$contains": "tea"}}, "Invalid operator"),
        ({"age": {"$between": [1]}}, "Invalid value for"),
        ({"$or": {}}, "Logical operators require"),
    ],
)
def test_existing_validation_errors_precede_package_calls(predicate, message):
    with pytest.raises(ValueError, match=message):
        translate_filter(predicate)


def test_filter_values_are_copied():
    predicate = {"name": {"$in": ["Mary", "Mark"]}}
    result = translate_filter(predicate)
    result["name"]["$string"]["$in"].append("Jason")
    assert predicate["name"]["$in"] == ["Mary", "Mark"]


@pytest.mark.parametrize(
    "operand,method,normalized",
    [
        ("tea", "$string", "tea"),
        (1, "$number", 1),
        (True, "$boolean", True),
        (False, "$boolean", False),
        (None, "$string", ""),
        ("", "$string", ""),
    ],
)
@pytest.mark.parametrize("operator", ["$eq", "$ne"])
def test_scalar_comparisons_use_explicit_item_context(
    operand, method, normalized, operator
):
    assert translate_filter({"x": {operator: operand}}) == {
        "x": {method: {operator: normalized}}
    }


def test_not_negates_the_complete_positive_predicate():
    assert translate_filter({"age": {"$not": {"$gt": 46, "$lt": 65}}}) == {
        "$nor": [{"age": {"$number": {"$gt": 46, "$lt": 65}}}]
    }


def test_regex_uses_substring_matching_and_conversion():
    assert translate_filter({"x": {"$regex": "ar"}}) == {
        "x": {"$string": {"$likeRegex": "ar"}}
    }
    assert translate_filter({"x": {"$regex": ""}}) == {
        "$and": [{"x": {"$exists": True}}, {"x": {"$exists": False}}]
    }


def test_array_equality_guards_size_order_and_element_types():
    assert translate_filter({"x": {"$eq": ["tea", None]}}) == {
        "$and": [
            {"x": {"$type": "array"}},
            {"x": {"$size": 2}},
            {"x[0]": {"$type": "string"}},
            {"x[0]": {"$eq": "tea"}},
            {"x[1]": {"$type": "null"}},
        ]
    }


def test_mixed_types_keep_one_sql_item_context():
    with pytest.raises(SQLFilterRequired):
        translate_filter({"x": {"$gt": 1, "$lt": "5"}})


def test_membership_null_operands_follow_the_sql_bind_context():
    assert translate_filter({"x": {"$in": [None, "tea"]}}) == {
        "x": {"$string": {"$in": ["", "tea"]}}
    }
    with pytest.raises(SQLFilterRequired):
        translate_filter({"x": {"$in": [None, 1]}})


def test_native_sql_types_are_not_sent_as_qbe_json():
    from datetime import datetime
    from decimal import Decimal

    for value in [Decimal("1.25"), datetime(2026, 1, 1)]:
        with pytest.raises(SQLFilterRequired):
            translate_filter({"x": {"$eq": value}})
