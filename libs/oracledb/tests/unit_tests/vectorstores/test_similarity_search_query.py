# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Unit tests for the SQL emitted by ``_get_similarity_search_query``.

The vector-index hint check (issue #130) keeps the vector index in the plan
even when a JSON Search Index is also defined on the table — without the
hint, the optimizer picks the JSON Search Index for the metadata filter and
skips the vector index entirely, regressing similarity-search latency.
"""

from __future__ import annotations

from langchain_community.vectorstores.utils import DistanceStrategy

from langchain_oracledb.vectorstores.oraclevs import _get_similarity_search_query

_TABLE = "ORAVS_DOCUMENTS"


def test_query_contains_vector_index_transform_hint_unfiltered() -> None:
    sql, _ = _get_similarity_search_query(
        table_name=_TABLE,
        distance_strategy=DistanceStrategy.COSINE,
        k=8,
    )
    assert f"/*+ VECTOR_INDEX_TRANSFORM({_TABLE}) */" in sql


def test_query_contains_vector_index_transform_hint_with_filter() -> None:
    """The filter path is the actual scenario from issue #130 — once the
    optimizer sees `JSON_EXISTS` and a JSON Search Index, it would otherwise
    drop the vector index. The hint has to be present here too.
    """
    sql, _ = _get_similarity_search_query(
        table_name=_TABLE,
        distance_strategy=DistanceStrategy.COSINE,
        k=8,
        filter={"category": "research"},
    )
    assert f"/*+ VECTOR_INDEX_TRANSFORM({_TABLE}) */" in sql
    assert "JSON_EXISTS" in sql


def test_hint_uses_caller_supplied_table_identifier() -> None:
    """Whatever identifier flows into ``FROM`` must also flow into the hint
    so the two reference the same object — including any quoting the caller
    chose to apply.
    """
    quoted = '"My Vector Table"'
    sql, _ = _get_similarity_search_query(
        table_name=quoted,
        distance_strategy=DistanceStrategy.COSINE,
        k=4,
    )
    assert f"/*+ VECTOR_INDEX_TRANSFORM({quoted}) */" in sql
    assert f"FROM {quoted}" in sql
