# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""OracleVS filters must never silently acquire QBE semantics."""

from copy import deepcopy

import pytest

from langchain_oracledb.vectorstores._vecdb_adapter import search_request
from langchain_oracledb.vectorstores._vecdb_filters import translate_filter


@pytest.mark.parametrize(
    "predicate",
    [
        {"drinks": {"$ne": "tea"}},
        {"drinks": {"$eq": ["soda", "tea"]}},
        {"address[0]": {"$ne": {"city": "Mono Vista"}}},
        {"address.zip": {"$not": {"$eq": "90001"}}},
        {"name": {"$regex": "*rk"}},
        {"$or": [{"age": 65}, {"drinks": {"$ne": "tea"}}]},
    ],
)
def test_unsupported_filters_rejected_at_request_boundary(predicate):
    with pytest.raises(ValueError, match="Unsupported VecDB filter"):
        search_request("DOCS", "COSINE", [1.0], 3, predicate, False, {})


def test_nested_paths_and_alias_are_translated_without_mutating_input():
    predicate = {"address": {"city": "NYC"}, "name": {"$instr": "ar"}}
    original = deepcopy(predicate)
    assert translate_filter(predicate) == {
        "$and": [
            {"address.city": "NYC"},
            {"name": {"$hasSubstring": "ar"}},
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
    result["name"]["$in"].append("Jason")
    assert predicate["name"]["$in"] == ["Mary", "Mark"]
