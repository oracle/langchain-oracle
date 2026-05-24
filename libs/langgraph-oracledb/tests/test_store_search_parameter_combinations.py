# type: ignore

"""Oracle store search parameter coverage."""

from contextlib import contextmanager

import pytest
from langgraph.store.base import SearchOp

from langgraph_oracledb.store.oracle import OracleStore
from langgraph_oracledb.store.oracle.base import _namespace_to_text


@contextmanager
def _test_store():
    """Create a mock store for query generation checks."""
    from langgraph_oracledb.store.oracle.base import OracleStore as BaseOracleStore

    store = BaseOracleStore.__new__(BaseOracleStore)
    store.conn = None
    store.table_name = "store_param_test"
    store.vector_table_name = None
    store.table_names = {
        "store": "store_param_test",
        "store_vectors": None,
        "store_migrations": "store_migrations_param_test",
        "vector_migrations": "vector_migrations_param_test",
    }
    store.index_config = None
    store.embeddings = None
    store.detected_dims = None
    store.distance_metric = None
    store.index_type = None

    def mock_validate_configuration(suffix, config, dims):
        return None

    store._validate_configuration = mock_validate_configuration
    yield store


def generate_search_parameter_combinations():
    """Generate a small, high-signal set of store search parameter combinations."""
    namespace_prefix_cases = [
        ("docs",),
        ("docs", "python"),
        ("docs", "python", "tutorial"),
        # ("deep", "a", "b", "c", "d"),
    ]

    filter_cases = [
        None,
        {"status": "active"},
        {"count": {"$gt": 10}},
        {"rating": {"$lte": 5}},
        {"status": "active", "priority": {"$gte": 3}},
        # {"score": {"$gte": 3.5}},
        # {"views": {"$lt": 1000}},
        # {"type": {"$ne": "archived"}},
        # {"type": "article", "stats": {"$gt": 100}, "rating": {"$lte": 5}},
    ]

    limit_offset_cases = [
        (10, 0),
        (5, 0),
        (20, 10),
        # (100, 0),
        # (10, 100),
    ]

    combinations = []

    for ns_prefix in namespace_prefix_cases:
        combinations.append(
            {
                "namespace_prefix": ns_prefix,
                "filter": None,
                "limit": 10,
                "offset": 0,
            }
        )

    for filter_dict in filter_cases:
        if filter_dict is not None:
            combinations.append(
                {
                    "namespace_prefix": ("test",),
                    "filter": filter_dict,
                    "limit": 10,
                    "offset": 0,
                }
            )

    for limit, offset in limit_offset_cases:
        combinations.append(
            {
                "namespace_prefix": ("items",),
                "filter": None,
                "limit": limit,
                "offset": offset,
            }
        )

    combinations.extend(
        [
            {
                "namespace_prefix": ("docs",),
                "filter": {"status": "published"},
                "limit": 5,
                "offset": 0,
            },
            {
                "namespace_prefix": ("articles",),
                "filter": {"views": {"$gte": 100}, "rating": {"$gt": 3}},
                "limit": 10,
                "offset": 5,
            },
        ]
    )

    return combinations


def _expected_filter_param_count(filter_dict) -> int:
    if not filter_dict:
        return 0
    count = 0
    for value in filter_dict.values():
        if isinstance(value, dict):
            count += len(value)
        else:
            count += 1
    return count


def _assert_prepared_search_query(store: OracleStore, params: dict) -> None:
    search_op = SearchOp(
        namespace_prefix=params["namespace_prefix"],
        filter=params.get("filter"),
        limit=params["limit"],
        offset=params["offset"],
    )

    queries, embedding_requests = store._prepare_batch_search_queries([(0, search_op)])

    assert len(queries) == 1
    assert embedding_requests == []

    query_sql, query_params = queries[0]
    assert "FROM store_param_test st" in query_sql
    assert "OFFSET :offset ROWS FETCH NEXT :limit ROWS ONLY" in query_sql
    assert query_params["limit"] == params["limit"]
    assert query_params["offset"] == params["offset"]
    if params["namespace_prefix"]:
        assert (
            query_params["ns_prefix"]
            == f"{_namespace_to_text(params['namespace_prefix'])}%"
        )
    else:
        assert "ns_prefix" not in query_params

    filter_dict = params.get("filter")
    filter_params = [key for key in query_params if key.startswith("filter_")]
    assert len(filter_params) == _expected_filter_param_count(filter_dict)


class TestStoreSearchParamCombinations:
    @pytest.mark.parametrize("params", generate_search_parameter_combinations())
    def test_store_search_parameter_combinations(self, params) -> None:
        """Validate representative store search query generation directly."""
        with _test_store() as store:
            _assert_prepared_search_query(store, params)

    def test_store_search_edge_cases(self) -> None:
        """Exercise risky generation-only edge cases without snapshots."""
        edge_cases = [
            {
                "namespace_prefix": (),
                "filter": None,
                "limit": 10,
                "offset": 0,
            },
            {
                "namespace_prefix": ("test",),
                "filter": {"desc": "'; DROP TABLE --"},
                "limit": 10,
                "offset": 0,
            },
            {
                "namespace_prefix": ("test",),
                "filter": {"text": "测试中文🎉"},
                "limit": 10,
                "offset": 0,
            },
            {
                "namespace_prefix": ("data",),
                "filter": None,
                "limit": 10,
                "offset": 1000000,
            },
            {
                "namespace_prefix": ("test",),
                "filter": None,
                "limit": 0,
                "offset": 0,
            },
            {
                "namespace_prefix": ("test",),
                "filter": {"value": {"$lt": -100}},
                "limit": 10,
                "offset": 0,
            },
            {
                "namespace_prefix": ("test",),
                "filter": {"field": None},
                "limit": 10,
                "offset": 0,
            },
            {
                "namespace_prefix": ("a", "b", "c", "d", "e", "f", "g", "h"),
                "filter": None,
                "limit": 10,
                "offset": 0,
            },
            {
                "namespace_prefix": ("test",),
                "filter": {"price": {"$gte": 19.99}},
                "limit": 10,
                "offset": 0,
            },
        ]

        with _test_store() as store:
            for case in edge_cases:
                _assert_prepared_search_query(store, case)
