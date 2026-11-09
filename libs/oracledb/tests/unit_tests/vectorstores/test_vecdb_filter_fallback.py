# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""SQL fallback preserves predicates, validation and the VecDB result contract."""

from unittest.mock import MagicMock

import pytest

from langchain_oracledb.vectorstores._vecdb import (
    NativeTransport,
    VecDBBackend,
    _filter_query,
    _filter_response,
)
from langchain_oracledb.vectorstores._vecdb_adapter import search_request
from langchain_oracledb.vectorstores.utils import DistanceStrategy


@pytest.mark.parametrize("metadata_column", ["METADATA", "CONTENT_METADATA"])
@pytest.mark.parametrize("metric", ["COSINE", "EUCLIDEAN", "DOT"])
def test_query_uses_original_predicates_before_distance_limit(metadata_column, metric):
    request = search_request('"literal.dot"', metric, [1.0, 0.0], 2, None, True, {})
    sql, binds = _filter_query(
        '"literal.dot"',
        metadata_column,
        request,
        {"metadata": {"$ne": {"value": "'); DROP TABLE x--"}}},
    )
    assert f'{metadata_column} AS metadata FROM "literal.dot"' in sql
    assert "JSON_QUERY(metadata, '$.metadata' )" in sql
    assert "DROP TABLE" not in sql
    assert '"value"' in binds["value0"]
    assert (
        sql.index("WHERE")
        < sql.index("ORDER BY distance")
        < sql.index("FETCH FIRST :k")
    )
    assert list(binds["embedding"]) == [1.0, 0.0]
    assert binds["k"] == 2


def test_fallback_projection_keeps_id_score_and_requested_embeddings():
    request = search_request(
        "DOCS", "COSINE", [1.0], 1, None, True, {"output_selector": ["tenant"]}
    )
    metadata = {"tenant": "a", "other": "hidden", "lc_oracle_content": "body"}
    response = _filter_response([("a", metadata, 0.25, [1.0])], request)
    assert response == {
        "results": [
            {
                "id": "a",
                "distance": 0.25,
                "vector": [1.0],
                "metadata": {"tenant": "a", "lc_oracle_content": "body"},
            }
        ]
    }
    assert metadata["other"] == "hidden"


@pytest.mark.parametrize(
    "options",
    [
        {"unknown": 1},
        {"output_selector": 1},
        {"advanced_options": {"distance_metric": "DOT"}},
    ],
)
def test_fallback_does_not_bypass_option_validation(options):
    transport = NativeTransport(None, (23, 26, 4))
    transport.filter_search = MagicMock()
    backend = VecDBBackend(transport, "DOCS", DistanceStrategy.COSINE)
    with pytest.raises(ValueError):
        backend.search([1.0], 1, {"x": {"$eq": {"a": 1}}}, **options)
    transport.filter_search.assert_not_called()


def test_invalid_filter_is_not_a_fallback_signal():
    transport = NativeTransport(None, (23, 26, 4))
    transport.filter_search = MagicMock()
    backend = VecDBBackend(transport, "DOCS", DistanceStrategy.COSINE)
    with pytest.raises(ValueError, match="Invalid operator"):
        backend.search([1.0], 1, {"x": {"$invalid": 1}})
    transport.filter_search.assert_not_called()


def test_search_execution_failures_do_not_select_sql_fallback():
    transport = NativeTransport(None, (23, 26, 4))
    transport.call = MagicMock(side_effect=RuntimeError("search failed"))
    transport.filter_search = MagicMock()
    backend = VecDBBackend(transport, "DOCS", DistanceStrategy.COSINE)
    with pytest.raises(RuntimeError, match="search failed"):
        backend.search([1.0], 1, {"x": "a"})
    transport.filter_search.assert_not_called()


def test_invalid_later_branch_precedes_fallback_for_an_earlier_branch():
    transport = NativeTransport(None, (23, 26, 4))
    transport.filter_search = MagicMock()
    backend = VecDBBackend(transport, "DOCS", DistanceStrategy.COSINE)
    with pytest.raises(ValueError, match="Invalid operator"):
        backend.search(
            [1.0], 1, {"$or": [{"x": {"$eq": {"a": 1}}}, {"x": {"$invalid": 1}}]}
        )
    transport.filter_search.assert_not_called()
