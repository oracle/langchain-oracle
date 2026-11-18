# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""SQL and QBE consume one nested-field and logical-expression grammar."""

import pytest

from langchain_oracledb.vectorstores._vecdb_filters import translate_filter
from langchain_oracledb.vectorstores.oraclevs import _generate_where_clause


@pytest.mark.parametrize(
    "render", [translate_filter, lambda value: _generate_where_clause(value, [])]
)
@pytest.mark.parametrize(
    "predicate,message",
    [
        ([], "Must be a dictionary"),
        ({"$unknown": []}, "not a recognized logical operator"),
        ({"$and": {}}, "Logical operators require"),
        ({"address": {"city": "NYC", "$eq": "x"}}, "cannot be mixed"),
        ({"x": {"$not": {"city": "NYC", "$eq": "x"}}}, "cannot be mixed"),
    ],
)
def test_shared_validation_errors(render, predicate, message):
    with pytest.raises(ValueError, match=message):
        render(predicate)


@pytest.mark.parametrize(
    "render", [translate_filter, lambda value: _generate_where_clause(value, [])]
)
def test_nested_and_dotted_fields_have_the_same_rendering(render):
    nested = {
        "$or": [{"address": {"city": "NYC", "zip": {"$ne": "10001"}}}, {"age": 21}]
    }
    dotted = {
        "$or": [{"address.city": "NYC", "address.zip": {"$ne": "10001"}}, {"age": 21}]
    }
    assert render(nested) == render(dotted)


@pytest.mark.parametrize("value", [True, False])
@pytest.mark.parametrize(
    "predicate",
    [lambda v: {"x": v}, lambda v: {"x": {"$eq": v}}, lambda v: {"x": {"$in": [v]}}],
)
def test_sql_comparison_preserves_boolean_values_in_cached_statements(value, predicate):
    values = []
    sql = _generate_where_clause(predicate(value), values)
    assert "@.boolean()" in sql
    assert ("true" if value else "false") in sql
    assert "PASSING" not in sql
    assert values == []
