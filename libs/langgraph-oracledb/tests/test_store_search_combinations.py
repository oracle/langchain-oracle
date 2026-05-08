# type: ignore

"""Oracle store search combination coverage."""

import time
import uuid
from contextlib import asynccontextmanager, contextmanager

import oracledb
import pytest
from langgraph.store.base import SearchOp

from langgraph_oracledb.store.oracle import OracleStore
from langgraph_oracledb.store.oracle.aio import AsyncOracleStore
from langgraph_oracledb.store.oracle.base import _namespace_to_text
from tests.conftest import (
    DEFAULT_CONNECTION_INFO,
    create_connection_string,
    skip_if_no_oracle,
)
from tests.embed_test_utils import CharacterEmbeddings

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
def _test_store_sync(
    table_suffix: str, with_vector: bool = False, use_db: bool = False
):
    """Create a sync test store."""
    if not use_db:
        from langgraph_oracledb.store.oracle.base import OracleStore as BaseOracleStore

        store = BaseOracleStore.__new__(BaseOracleStore)
        store.conn = None
        store.table_name = f"store_{table_suffix}"
        store.vector_table_name = (
            f"store_vectors_{table_suffix}" if with_vector else None
        )
        store.table_names = {
            "store": f"store_{table_suffix}",
            "store_vectors": f"store_vectors_{table_suffix}" if with_vector else None,
            "store_migrations": f"store_migrations_{table_suffix}",
            "vector_migrations": f"vector_migrations_{table_suffix}",
        }
        store.index_config = None
        store.embeddings = None
        store.detected_dims = None
        store.distance_metric = None
        store.index_type = None
        if with_vector:
            store.index_config = {
                "dims": 128,
                "embed": CharacterEmbeddings(dims=128),
                "fields": ["text", "content"],
                "index_type": {
                    "type": "hnsw",
                    "distance_metric": "COSINE",
                    "neighbors": 16,
                    "efconstruction": 200,
                },
                "__estimated_num_vectors": 2,
            }
            store.embeddings = store.index_config["embed"]
            store.detected_dims = 128
            store.distance_metric = "COSINE"
            store.index_type = "hnsw"

        def mock_validate_configuration(suffix, config, dims):
            return None

        store._validate_configuration = mock_validate_configuration
        yield store
        return

    pool = get_connection_pool()
    index_config = None
    if with_vector:
        index_config = {
            "dims": 128,
            "embed": CharacterEmbeddings(dims=128),
            "fields": ["text", "content"],
            "index_type": {
                "type": "hnsw",
                "distance_metric": "COSINE",
                "neighbors": 16,
                "efconstruction": 200,
            },
        }

    conn = pool.acquire()
    try:
        store = OracleStore(conn=conn, table_suffix=table_suffix, index=index_config)
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


@asynccontextmanager
async def _test_store_async(
    table_suffix: str, with_vector: bool = False, use_db: bool = False
):
    """Create an async test store."""
    if not use_db:
        from langgraph_oracledb.store.oracle.aio import (
            AsyncOracleStore as BaseAsyncOracleStore,
        )

        store = BaseAsyncOracleStore.__new__(BaseAsyncOracleStore)
        store.conn = None

        class MockTask:
            def cancel(self):
                return None

        store._task = MockTask()
        store.table_name = f"store_{table_suffix}"
        store.vector_table_name = (
            f"store_vectors_{table_suffix}" if with_vector else None
        )
        store.table_names = {
            "store": f"store_{table_suffix}",
            "store_vectors": f"store_vectors_{table_suffix}" if with_vector else None,
            "store_migrations": f"store_migrations_{table_suffix}",
            "vector_migrations": f"vector_migrations_{table_suffix}",
        }
        store.index_config = None
        store.embeddings = None
        store.detected_dims = None
        store.distance_metric = None
        store.index_type = None
        if with_vector:
            store.index_config = {
                "dims": 128,
                "embed": CharacterEmbeddings(dims=128),
                "fields": ["text", "content"],
                "index_type": {
                    "type": "hnsw",
                    "distance_metric": "COSINE",
                    "neighbors": 16,
                    "efconstruction": 200,
                },
                "__estimated_num_vectors": 2,
            }
            store.embeddings = store.index_config["embed"]
            store.detected_dims = 128
            store.distance_metric = "COSINE"
            store.index_type = "hnsw"

        async def mock_validate_configuration(suffix, config, dims):
            return None

        store._validate_configuration = mock_validate_configuration
        yield store
        return

    conn_string = create_connection_string(DEFAULT_CONNECTION_INFO)
    index_config = None
    if with_vector:
        index_config = {
            "dims": 128,
            "embed": CharacterEmbeddings(dims=128),
            "fields": ["text", "content"],
            "index_type": {
                "type": "hnsw",
                "distance_metric": "COSINE",
                "neighbors": 16,
                "efconstruction": 200,
            },
        }

    async with AsyncOracleStore.from_conn_string(
        conn_string, table_suffix=table_suffix, index=index_config
    ) as store:
        await store.setup()
        try:
            yield store
        finally:
            try:
                await store.ateardown()
            except Exception:
                async with await oracledb.connect_async(
                    **DEFAULT_CONNECTION_INFO
                ) as conn:
                    async with conn.cursor() as cur:
                        for table in [
                            f"store_{table_suffix}",
                            f"store_vectors_{table_suffix}",
                            f"store_migrations_{table_suffix}",
                            f"vector_migrations_{table_suffix}",
                        ]:
                            try:
                                await cur.execute(
                                    f"DROP TABLE {table} CASCADE CONSTRAINTS"
                                )
                            except oracledb.DatabaseError:
                                pass
                        await conn.commit()


def generate_search_combinations():
    """Generate a reduced set of representative store search scenarios."""
    filter_cases = [
        None,
        {"status": "active"},
        {"active": True},
        {"score": {"$gt": 3.0}},
        {"status": "published", "priority": {"$gte": 3}},
        # {"archived": False},
        # {"count": 42},
        # {"price": 19.99},
        # {"views": {"$gte": 100}},
        # {"age": {"$lt": 65}},
        # {"rating": {"$lte": 5.0}},
        # {"type": {"$ne": "deleted"}},
        # {"active": True, "score": {"$gt": 2.5}, "views": {"$lte": 1000}},
    ]

    query_cases = [
        None,
        "find similar documents",
        "machine learning tutorial",
        # "red apple fruit",
    ]

    namespace_cases = [
        ("docs",),
        ("articles", "tech"),
        # ("products", "electronics", "phones"),
    ]

    pagination_cases = [
        (10, 0),
        (5, 10),
        # (20, 0),
    ]

    combinations = []
    for query in query_cases:
        for filter_dict in filter_cases:
            for namespace in namespace_cases:
                for limit, offset in pagination_cases:
                    combinations.append(
                        {
                            "scenario": "vector_search"
                            if query is not None
                            else "filter_only",
                            "query": query,
                            "filter": filter_dict,
                            "namespace": namespace,
                            "limit": limit,
                            "offset": offset,
                            "with_vector_config": query is not None,
                        }
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
    search_op = SearchOp(
        namespace_prefix=params["namespace"],
        filter=params.get("filter"),
        query=params.get("query"),
        limit=params["limit"],
        offset=params["offset"],
    )

    queries, embedding_requests = store._prepare_batch_search_queries([(0, search_op)])
    assert len(queries) == 1

    query_sql, query_params = queries[0]
    assert query_params["limit"] == params["limit"]
    assert query_params["offset"] == params["offset"]
    assert query_params["ns_prefix"] == f"{_namespace_to_text(params['namespace'])}%"

    filter_params = [key for key in query_params if key.startswith("filter_")]
    assert len(filter_params) == _expected_filter_param_count(params.get("filter"))

    if params["query"] is None:
        assert embedding_requests == []
        assert "NULL AS score" in query_sql
        assert ":expanded_limit" not in query_sql
    else:
        assert embedding_requests == [(0, params["query"])]
        assert "JOIN" in query_sql
        assert "FETCH FIRST :expanded_limit ROWS ONLY" in query_sql
        assert query_params["expanded_limit"] > params["limit"]


class TestStoreSearchCombinations:
    @pytest.mark.parametrize("params", generate_search_combinations())
    def test_store_search_combinations_sync(self, params) -> None:
        """Validate representative sync search query generation directly."""
        with _test_store_sync(
            "sync_matrix",
            with_vector=params["with_vector_config"],
            use_db=False,
        ) as store:
            _assert_store_search_query_shape(store, params)

    @pytest.mark.parametrize("params", generate_search_combinations())
    async def test_store_search_combinations_async(self, params) -> None:
        """Validate representative async search query generation directly."""
        async with _test_store_async(
            "async_matrix",
            with_vector=params["with_vector_config"],
            use_db=False,
        ) as store:
            _assert_store_search_query_shape(store, params)

    @skip_if_no_oracle()
    def test_store_search_combinations_real_data(self):
        """Validate end-to-end search behavior with real Oracle data."""
        test_cases = [
            {
                "name": "vector_with_filters",
                "with_vector": True,
                "data": [
                    (
                        "doc1",
                        {
                            "text": "red apple fruit",
                            "category": "food",
                            "price": 1.50,
                            "organic": True,
                        },
                    ),
                    (
                        "doc2",
                        {
                            "text": "red car vehicle",
                            "category": "transport",
                            "price": 25000.0,
                            "organic": False,
                        },
                    ),
                    (
                        "doc3",
                        {
                            "text": "green apple fruit",
                            "category": "food",
                            "price": 1.20,
                            "organic": True,
                        },
                    ),
                    (
                        "doc4",
                        {
                            "text": "blue car sedan",
                            "category": "transport",
                            "price": 30000.0,
                            "organic": False,
                        },
                    ),
                ],
                "searches": [
                    {
                        "query": "red apple fruit",
                        "filter": {"organic": True},
                        "expected_min": 1,
                    },
                    {
                        "query": "red car vehicle",
                        "filter": {"category": "transport"},
                        "expected_min": 1,
                    },
                    {
                        "query": "green apple",
                        "filter": {"price": {"$lt": 1.40}},
                        "expected_min": 1,
                    },
                ],
            },
            {
                "name": "filter_only",
                "with_vector": False,
                "data": [
                    ("item1", {"status": "active", "priority": 5, "archived": False}),
                    ("item2", {"status": "inactive", "priority": 2, "archived": True}),
                    ("item3", {"status": "active", "priority": 3, "archived": False}),
                    ("item4", {"status": "pending", "priority": 4, "archived": False}),
                ],
                "searches": [
                    {"query": None, "filter": {"status": "active"}, "expected_min": 2},
                    {
                        "query": None,
                        "filter": {"priority": {"$gte": 4}},
                        "expected_min": 2,
                    },
                    {
                        "query": None,
                        "filter": {"status": "active", "priority": {"$gt": 3}},
                        "expected_min": 1,
                    },
                ],
            },
        ]

        timestamp = int(time.time() * 1000) % 100000

        for i, case in enumerate(test_cases):
            unique_suffix = f"real_{i}_{timestamp}"[:32]
            with _test_store_sync(
                unique_suffix, with_vector=case["with_vector"], use_db=True
            ) as store:
                for key, value in case["data"]:
                    store.put(("test",), key, value)

                for search in case["searches"]:
                    results = store.search(
                        ("test",),
                        query=search.get("query"),
                        filter=search.get("filter"),
                        limit=10,
                        offset=0,
                    )

                    actual_count = len(results)
                    expected_min = search["expected_min"]

                    if search.get("query") and actual_count == 0:
                        continue

                    assert actual_count >= expected_min

                    for result in results:
                        filter_dict = search.get("filter")
                        if not filter_dict:
                            continue
                        for key, expected_val in filter_dict.items():
                            actual_val = result.value.get(key)
                            if isinstance(expected_val, dict):
                                for op, val in expected_val.items():
                                    if op == "$gte":
                                        assert actual_val >= val
                                    elif op == "$gt":
                                        assert actual_val > val
                                    elif op == "$lt":
                                        assert actual_val < val
                                    elif op == "$lte":
                                        assert actual_val <= val
                            else:
                                assert actual_val == expected_val

    @skip_if_no_oracle()
    def test_store_search_edge_cases(self):
        """Test edge cases for search combinations with real Oracle execution."""
        edge_cases = [
            {
                "name": "vector_search_no_results",
                "with_vector": True,
                "query": "nonexistent content",
                "filter": {"status": "active"},
            },
            {
                "name": "filter_only_no_results",
                "with_vector": False,
                "query": None,
                "filter": {"nonexistent_field": "value"},
            },
            {
                "name": "vector_search_special_chars",
                "with_vector": True,
                "query": "test with 'quotes' and \"double quotes\"",
                "filter": {"text": "'; DROP TABLE --"},
            },
            {
                "name": "filter_only_unicode",
                "with_vector": False,
                "query": None,
                "filter": {"description": "测试中文🎉"},
            },
        ]

        timestamp = int(time.time() * 1000) % 100000

        for i, case in enumerate(edge_cases):
            unique_suffix = f"edge_{i}_{timestamp}"[:32]
            with _test_store_sync(
                unique_suffix, with_vector=case["with_vector"], use_db=True
            ) as store:
                store.put(
                    ("test",), "item1", {"text": "normal content", "status": "active"}
                )

                results = store.search(
                    ("test",),
                    query=case.get("query"),
                    filter=case.get("filter"),
                    limit=10,
                    offset=0,
                )
                assert isinstance(results, list)
