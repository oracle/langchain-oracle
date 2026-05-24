# type: ignore

"""Oracle store search key coverage."""

import uuid
from contextlib import contextmanager

import oracledb
import pytest
from langgraph.store.base import SearchOp

from langgraph_oracledb.store.oracle import OracleStore
from langgraph_oracledb.store.oracle.base import _namespace_to_text
from tests.conftest import (
    DEFAULT_CONNECTION_INFO,
    create_connection_string,
    skip_if_no_oracle,
)

_CONNECTION_POOL = None


def get_connection_pool():
    """Get or create a connection pool for real execution checks."""
    global _CONNECTION_POOL
    if _CONNECTION_POOL is None:
        _CONNECTION_POOL = oracledb.create_pool(
            **DEFAULT_CONNECTION_INFO,
            min=5,
            max=20,
            increment=2,
        )
    return _CONNECTION_POOL


@contextmanager
def _test_store(use_db: bool = False):
    """Create a test store for SQL generation and validation."""
    if not use_db:
        from langgraph_oracledb.store.oracle.base import OracleStore as BaseOracleStore

        table_suffix = "key_test"
        store = BaseOracleStore.__new__(BaseOracleStore)
        store.conn = None
        store.table_name = f"store_{table_suffix}"
        store.vector_table_name = None
        store.table_names = {
            "store": f"store_{table_suffix}",
            "store_vectors": None,
            "store_migrations": f"store_migrations_{table_suffix}",
            "vector_migrations": f"vector_migrations_{table_suffix}",
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
        return

    pool = get_connection_pool()
    table_suffix = f"key_test_{uuid.uuid4().hex[:8]}"

    conn = pool.acquire()
    try:
        store = OracleStore(conn=conn, table_suffix=table_suffix)
        store.setup()
        yield store
    finally:
        try:
            store.teardown()
        except Exception:
            with conn.cursor() as cur:
                for table in [
                    f"store_{table_suffix}",
                    f"store_vectors_{table_suffix}",
                    f"store_migrations_{table_suffix}",
                    f"vector_migrations_{table_suffix}",
                ]:
                    try:
                        cur.execute(f"DROP TABLE {table} CASCADE CONSTRAINTS")
                    except oracledb.DatabaseError:
                        pass
                conn.commit()
        pool.release(conn)


def generate_key_search_combinations():
    """Generate high-signal search combinations."""
    combinations = []

    combinations.extend(
        [
            {"namespace": ("simple",), "filter": None},
            {"namespace": ("multi", "level", "namespace"), "filter": None},
        ]
    )

    combinations.extend(
        [
            {"namespace": ("test",), "filter": {"active": True}},
            {"namespace": ("test",), "filter": {"active": False}},
            {"namespace": ("test",), "filter": {"count": 42}},
            {"namespace": ("test",), "filter": {"price": 19.99}},
            {"namespace": ("test",), "filter": {"name": "test"}},
        ]
    )

    combinations.extend(
        [
            {"namespace": ("data",), "filter": {"value": {"$eq": 100}}},
            {"namespace": ("data",), "filter": {"value": {"$ne": 100}}},
            {"namespace": ("data",), "filter": {"value": {"$gt": 50}}},
            {"namespace": ("data",), "filter": {"value": {"$gte": 50}}},
            {"namespace": ("data",), "filter": {"value": {"$lt": 100}}},
            {"namespace": ("data",), "filter": {"value": {"$lte": 100}}},
        ]
    )

    combinations.extend(
        [
            {
                "namespace": ("complex",),
                "filter": {"status": "active", "priority": {"$gte": 3}},
            },
            {
                "namespace": ("complex",),
                "filter": {
                    "type": "article",
                    "views": {"$gt": 1000},
                    "rating": {"$lte": 5},
                },
            },
            # {
            #     "namespace": ("complex",),
            #     "filter": {
            #         "enabled": True,
            #         "score": {"$gte": 0},
            #         "score": {"$lte": 100},  # noqa: F601
            #     },
            # },
        ]
    )

    combinations.extend(
        [
            {"namespace_prefix": ("docs",), "filter": None},
            {"namespace_prefix": ("docs", "api"), "filter": {"version": "v2"}},
            {
                "namespace_prefix": ("users",),
                "filter": {"role": "admin", "active": True},
            },
        ]
    )

    combinations.extend(
        [
            {"namespace": ("special",), "filter": {"text": "Hello 'World'"}},
            {"namespace": ("special",), "filter": {"text": 'Test "quotes"'}},
            {"namespace": ("special",), "filter": {"path": "/usr/local/bin"}},
            {"namespace": ("special",), "filter": {"query": "SELECT * FROM table"}},
            # {"namespace": ("unicode",), "filter": {"text": "测试中文"}},
            # {"namespace": ("unicode",), "filter": {"emoji": "🚀🎉"}},
        ]
    )

    combinations.extend(
        [
            {"namespace": ("paginated",), "filter": None, "limit": 1, "offset": 0},
            {
                "namespace": ("paginated",),
                "filter": {"type": "item"},
                "limit": 50,
                "offset": 100,
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


def _assert_store_search_query_shape(store, params) -> None:
    namespace = params.get("namespace")
    namespace_prefix = params.get("namespace_prefix")
    filter_dict = params.get("filter")
    limit = params.get("limit", 10)
    offset = params.get("offset", 0)

    search_namespace_prefix = namespace if namespace is not None else namespace_prefix
    search_op = SearchOp(
        namespace_prefix=search_namespace_prefix,
        filter=filter_dict,
        limit=limit,
        offset=offset,
    )

    queries, embedding_requests = store._prepare_batch_search_queries([(0, search_op)])

    assert len(queries) == 1
    assert embedding_requests == []

    query_sql, query_params = queries[0]
    assert "OFFSET :offset ROWS FETCH NEXT :limit ROWS ONLY" in query_sql
    assert query_params["limit"] == limit
    assert query_params["offset"] == offset
    assert (
        query_params["ns_prefix"] == f"{_namespace_to_text(search_namespace_prefix)}%"
    )

    filter_params = [key for key in query_params if key.startswith("filter_")]
    assert len(filter_params) == _expected_filter_param_count(filter_dict)


class TestStoreSearchKeyCombinations:
    @pytest.mark.parametrize("params", generate_key_search_combinations())
    def test_store_search_key_combinations(self, params) -> None:
        """Validate representative key combinations directly."""
        with _test_store() as store:
            _assert_store_search_query_shape(store, params)

    @skip_if_no_oracle()
    def test_store_search_problematic_cases(self) -> None:
        """Test specific cases that were identified as potentially problematic."""
        problematic_cases = [
            {
                "name": "boolean_true_filter",
                "namespace": ("test",),
                "filter": {"enabled": True},
            },
            {
                "name": "boolean_false_filter",
                "namespace": ("test",),
                "filter": {"enabled": False},
            },
            {
                "name": "multiple_booleans",
                "namespace": ("test",),
                "filter": {"active": True, "archived": False, "visible": True},
            },
            {"name": "zero_value", "namespace": ("test",), "filter": {"count": 0}},
            {
                "name": "negative_value",
                "namespace": ("test",),
                "filter": {"balance": -100.50},
            },
            {
                "name": "very_large_number",
                "namespace": ("test",),
                "filter": {"value": {"$gt": 999999999999}},
            },
            {
                "name": "empty_string",
                "namespace": ("test",),
                "filter": {"description": ""},
            },
            {
                "name": "whitespace_only",
                "namespace": ("test",),
                "filter": {"text": "   "},
            },
            {
                "name": "sql_injection_attempt",
                "namespace": ("test",),
                "filter": {"input": "'; DROP TABLE store; --"},
            },
            {
                "name": "all_operators",
                "namespace": ("test",),
                "filter": {
                    "exact": "value",
                    "gt_field": {"$gt": 10},
                    "gte_field": {"$gte": 20},
                    "lt_field": {"$lt": 30},
                    "lte_field": {"$lte": 40},
                    "ne_field": {"$ne": "excluded"},
                },
            },
        ]

        with _test_store(use_db=True) as store:
            for case in problematic_cases:
                search_op = SearchOp(
                    namespace_prefix=case["namespace"],
                    filter=case["filter"],
                    limit=10,
                    offset=0,
                )
                queries, _ = store._prepare_batch_search_queries([(0, search_op)])
                assert len(queries) == 1

                query_sql, query_params = queries[0]
                with store.conn.cursor() as cur:
                    cur.execute(query_sql, query_params)
                    cur.fetchall()
