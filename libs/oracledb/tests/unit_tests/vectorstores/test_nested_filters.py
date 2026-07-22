# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Unit tests for nested metadata filters in OracleVS (issue #272).

Nested dict filters like ``{"address": {"city": "NYC"}}`` are translated
into dotted JSON paths (``$.address.city``), equivalent to the already
supported ``{"address.city": "NYC"}`` spelling. Operator dicts and scalars
terminate the recursion, so nesting depth is arbitrary.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from langchain_oracledb.vectorstores.oraclevs import _generate_where_clause


def _clause(filter: Dict[str, Any]) -> tuple[str, List[Any]]:
    binds: List[Any] = []
    return _generate_where_clause(filter, binds), binds


def test_nested_path_scalar() -> None:
    sql, binds = _clause({"address": {"city": "NYC"}})

    assert "'$.address.city?(@ == $val)'" in sql
    assert binds == ["NYC"]


def test_nested_path_equivalent_to_dotted_key() -> None:
    nested_sql, nested_binds = _clause({"address": {"city": "NYC"}})
    dotted_sql, dotted_binds = _clause({"address.city": "NYC"})

    assert nested_sql == dotted_sql
    assert nested_binds == dotted_binds


def test_nested_path_with_operator_leaf() -> None:
    sql, binds = _clause({"address": {"zip": {"$gte": 10000}}})

    assert "'$.address.zip?(@ >= $val0)'" in sql
    assert binds == [10000]


def test_deeply_nested_path() -> None:
    sql, binds = _clause({"a": {"b": {"c": {"d": "deep"}}}})

    assert "'$.a.b.c.d?(@ == $val)'" in sql
    assert binds == ["deep"]


def test_nested_siblings_combine_with_and() -> None:
    sql, binds = _clause({"address": {"city": {"$ne": "LA"}, "zip": "10001"}})

    assert sql.startswith("(") and sql.endswith(")")
    assert " AND " in sql
    assert "'$.address.city?(@ != $val0)'" in sql
    assert "'$.address.zip?(@ == $val)'" in sql
    assert binds == ["LA", "10001"]


def test_nested_inside_logical_operators() -> None:
    sql, binds = _clause(
        {"$or": [{"address": {"city": "NYC"}}, {"profile": {"level": 3}}]}
    )

    assert " OR " in sql
    assert "'$.address.city?(@ == $val)'" in sql
    assert "'$.profile.level?(@ == $val)'" in sql
    assert binds == ["NYC", 3]


def test_logical_nesting_still_supported() -> None:
    """Pre-existing behavior pinned: $and/$or compose recursively."""
    sql, binds = _clause(
        {"$and": [{"$or": [{"a": 1}, {"b": 2}]}, {"$nor": [{"c": 3}]}]}
    )

    assert " OR " in sql and " AND " in sql and "NOT" in sql
    assert binds == [1, 2, 3]


def test_mixed_operator_and_nested_keys_raise() -> None:
    with pytest.raises(ValueError, match="cannot be mixed"):
        _clause({"a": {"$eq": 1, "b": 2}})


def test_bind_variables_stay_positional() -> None:
    """Bind names must track the shared bind list across nested branches."""
    sql, binds = _clause({"$and": [{"x": {"y": "1"}}, {"z": {"$in": ["a", "b"]}}]})

    assert ":value0" in sql
    assert binds[0] == "1"
    assert len(binds) == 3  # "1" + two $in values
