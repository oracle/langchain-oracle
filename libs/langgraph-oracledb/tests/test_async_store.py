# type: ignore
from __future__ import annotations

import asyncio
import datetime
import itertools
from contextlib import suppress
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import oracledb
import pytest
from langchain_core.embeddings import Embeddings
from langgraph.store.base import (
    GetOp,
    Item,
    ListNamespacesOp,
    PutOp,
    SearchOp,
)

from langgraph_oracledb.store.oracle import AsyncOracleStore
from langgraph_oracledb.store.oracle import aio as oracle_store_aio
from tests.conftest import (
    DEFAULT_CONNECTION_INFO,
    ORACLE_DISTANCE_TYPES,
    ORACLE_INDEX_TYPES,
)
from tests.conftest_store import (
    TTL_MINUTES,
    TTL_SECONDS,
    create_async_vector_store_with_fields,
)
from tests.conftest_store import (
    async_store as store,
)
from tests.conftest_store import (
    async_vector_store as vector_store,
)
from tests.embed_test_utils import CharacterEmbeddings

# NOTE: These tests are modeled after the PostgreSQL store tests in
# libs/checkpoint-postgres/tests. PostgreSQL has a "pipeline" mode that
# pipelines SQL statements for better performance. Oracle does not have
# an equivalent pipeline mode, so we only test "default" and "pool" modes.


# Store fixture is imported from conftest_store.py


class _FakeAsyncSearchCursor:
    def __init__(self, rows, update_error=None):
        self._rows = rows
        self._update_error = update_error
        self.description = [
            ("PREFIX",),
            ("KEY",),
            ("VALUE",),
            ("CREATED_AT",),
            ("UPDATED_AT",),
            ("SCORE",),
        ]
        self.execute_calls = 0

    async def execute(self, query, params):
        self.execute_calls += 1
        if (
            query.lstrip().upper().startswith("UPDATE")
            and self._update_error is not None
        ):
            raise self._update_error

    async def fetchall(self):
        return self._rows


async def test_no_running_loop(store: AsyncOracleStore) -> None:
    with pytest.raises(asyncio.InvalidStateError):
        store.put(("foo", "bar"), "baz", {"val": "baz"})
    with pytest.raises(asyncio.InvalidStateError):
        store.get(("foo", "bar"), "baz")
    with pytest.raises(asyncio.InvalidStateError):
        store.delete(("foo", "bar"), "baz")
    with pytest.raises(asyncio.InvalidStateError):
        store.search(("foo", "bar"))
    with pytest.raises(asyncio.InvalidStateError):
        store.list_namespaces(prefix=("foo",))
    with pytest.raises(asyncio.InvalidStateError):
        store.batch([PutOp(namespace=("foo", "bar"), key="baz", value={"val": "baz"})])
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(store.put, ("foo", "bar"), "baz", {"val": "baz"})
        result = await asyncio.wrap_future(future)
        assert result is None
        future = executor.submit(store.get, ("foo", "bar"), "baz")
        result = await asyncio.wrap_future(future)
        assert result.value == {"val": "baz"}
        result = await asyncio.wrap_future(
            executor.submit(store.list_namespaces, prefix=("foo",))
        )


async def test_start_ttl_sweeper_without_ttl_returns_completed_task() -> None:
    store = AsyncOracleStore(Mock())

    task = await store.start_ttl_sweeper()

    await task
    assert task.done()
    assert store._ttl_sweeper_task is None


async def test_stop_ttl_sweeper_when_not_running_returns_true() -> None:
    store = AsyncOracleStore(Mock(), ttl={"default_ttl": 1})

    assert await store.stop_ttl_sweeper() is True


async def test_start_ttl_sweeper_reuses_running_task() -> None:
    store = AsyncOracleStore(
        Mock(),
        ttl={"default_ttl": 1, "refresh_on_read": True, "sweep_interval_minutes": 1},
    )

    task_1 = await store.start_ttl_sweeper()
    task_2 = await store.start_ttl_sweeper()

    assert task_1 is task_2
    assert not task_1.done()

    assert await store.stop_ttl_sweeper(timeout=1.0) is True
    assert store._ttl_sweeper_task is None


async def test_stop_ttl_sweeper_timeout_returns_false() -> None:
    store = AsyncOracleStore(Mock(), ttl={"default_ttl": 1})

    async def blocked() -> None:
        await asyncio.Future()

    task = asyncio.create_task(blocked())
    store._ttl_sweeper_task = task

    try:
        assert await store.stop_ttl_sweeper(timeout=0.01) is False
        assert store._ttl_sweeper_task is task
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
        store._ttl_sweeper_task = None


async def test_large_batches(request: Any, store: AsyncOracleStore) -> None:
    N = 10  # Oracle conservative limit for all modes, PostgreSQL uses 100
    M = 2  # Oracle conservative limit for all modes, PostgreSQL uses 10

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for m in range(M):
            for i in range(N):
                futures += [
                    executor.submit(
                        store.put,
                        ("test", "foo", "bar", "baz", str(m % 2)),
                        f"key{i}",
                        value={"foo": "bar" + str(i)},
                    ),
                    executor.submit(
                        store.get,
                        ("test", "foo", "bar", "baz", str(m % 2)),
                        f"key{i}",
                    ),
                    executor.submit(
                        store.list_namespaces,
                        prefix=None,
                        max_depth=m + 1,
                    ),
                    executor.submit(
                        store.search,
                        ("test",),
                    ),
                    executor.submit(
                        store.put,
                        ("test", "foo", "bar", "baz", str(m % 2)),
                        f"key{i}",
                        value={"foo": "bar" + str(i)},
                    ),
                    executor.submit(
                        store.put,
                        ("test", "foo", "bar", "baz", str(m % 2)),
                        f"key{i}",
                        None,
                    ),
                ]

        results = await asyncio.gather(
            *(asyncio.wrap_future(future) for future in futures)
        )
    assert len(results) == M * N * 6


async def test_large_batches_async(store: AsyncOracleStore) -> None:
    N = 100  # Reduced from 1000 for Oracle performance
    M = 5  # Reduced from 10 for Oracle performance
    coros = []
    for m in range(M):
        for i in range(N):
            coros.append(
                store.aput(
                    ("test", "foo", "bar", "baz", str(m % 2)),
                    f"key{i}",
                    value={"foo": "bar" + str(i)},
                )
            )
            coros.append(
                store.aget(
                    ("test", "foo", "bar", "baz", str(m % 2)),
                    f"key{i}",
                )
            )
            coros.append(
                store.alist_namespaces(
                    prefix=None,
                    max_depth=m + 1,
                )
            )
            coros.append(
                store.asearch(
                    ("test",),
                )
            )
            coros.append(
                store.aput(
                    ("test", "foo", "bar", "baz", str(m % 2)),
                    f"key{i}",
                    value={"foo": "bar" + str(i)},
                )
            )
            coros.append(
                store.adelete(
                    ("test", "foo", "bar", "baz", str(m % 2)),
                    f"key{i}",
                )
            )

    results = await asyncio.gather(*coros)
    assert len(results) == M * N * 6


async def test_abatch_order(store: AsyncOracleStore) -> None:
    # Setup test data
    await store.aput(("test", "foo"), "key1", {"data": "value1"})
    await store.aput(("test", "bar"), "key2", {"data": "value2"})

    ops = [
        GetOp(namespace=("test", "foo"), key="key1"),
        PutOp(namespace=("test", "bar"), key="key2", value={"data": "value2"}),
        SearchOp(
            namespace_prefix=("test",), filter={"data": "value1"}, limit=10, offset=0
        ),
        ListNamespacesOp(match_conditions=None, max_depth=None, limit=10, offset=0),
        GetOp(namespace=("test",), key="key3"),
    ]

    results = await store.abatch(ops)
    assert len(results) == 5
    assert isinstance(results[0], Item)
    assert isinstance(results[0].value, dict)
    assert results[0].value == {"data": "value1"}
    assert results[0].key == "key1"
    assert results[1] is None
    assert isinstance(results[2], list)
    assert len(results[2]) == 1
    assert isinstance(results[3], list)
    assert ("test", "foo") in results[3] and ("test", "bar") in results[3]
    assert results[4] is None

    ops_reordered = [
        SearchOp(namespace_prefix=("test",), filter=None, limit=5, offset=0),
        GetOp(namespace=("test", "bar"), key="key2"),
        ListNamespacesOp(match_conditions=None, max_depth=None, limit=5, offset=0),
        PutOp(namespace=("test",), key="key3", value={"data": "value3"}),
        GetOp(namespace=("test", "foo"), key="key1"),
    ]

    results_reordered = await store.abatch(ops_reordered)
    assert len(results_reordered) == 5
    assert isinstance(results_reordered[0], list)
    assert len(results_reordered[0]) == 2
    assert isinstance(results_reordered[1], Item)
    assert results_reordered[1].value == {"data": "value2"}
    assert results_reordered[1].key == "key2"
    assert isinstance(results_reordered[2], list)
    assert ("test", "foo") in results_reordered[2] and (
        "test",
        "bar",
    ) in results_reordered[2]
    assert results_reordered[3] is None
    assert isinstance(results_reordered[4], Item)
    assert results_reordered[4].value == {"data": "value1"}
    assert results_reordered[4].key == "key1"


async def test_batch_get_ops(store: AsyncOracleStore) -> None:
    # Setup test data
    await store.aput(("test",), "key1", {"data": "value1"})
    await store.aput(("test",), "key2", {"data": "value2"})

    ops = [
        GetOp(namespace=("test",), key="key1"),
        GetOp(namespace=("test",), key="key2"),
        GetOp(namespace=("test",), key="key3"),
    ]

    results = await store.abatch(ops)

    assert len(results) == 3
    assert results[0] is not None
    assert results[1] is not None
    assert results[2] is None
    assert results[0].key == "key1"
    assert results[1].key == "key2"


async def test_batch_put_ops(store: AsyncOracleStore) -> None:
    ops = [
        PutOp(namespace=("test",), key="key1", value={"data": "value1"}),
        PutOp(namespace=("test",), key="key2", value={"data": "value2"}),
        PutOp(namespace=("test",), key="key3", value=None),
    ]

    results = await store.abatch(ops)

    assert len(results) == 3
    assert all(result is None for result in results)

    # Verify the puts worked
    items = await store.asearch(["test"], limit=10)
    assert len(items) == 2  # key3 had None value so wasn't stored


async def test_batch_search_ops(store: AsyncOracleStore) -> None:
    # Setup test data
    await store.aput(("test", "foo"), "key1", {"data": "value1"})
    await store.aput(("test", "bar"), "key2", {"data": "value2"})

    ops = [
        SearchOp(
            namespace_prefix=("test",), filter={"data": "value1"}, limit=10, offset=0
        ),
        SearchOp(namespace_prefix=("test",), filter=None, limit=5, offset=0),
    ]

    results = await store.abatch(ops)

    assert len(results) == 2
    assert len(results[0]) == 1  # Filtered results
    assert len(results[1]) == 2  # All results


async def test_batch_list_namespaces_ops(store: AsyncOracleStore) -> None:
    # Setup test data
    await store.aput(("test", "namespace1"), "key1", {"data": "value1"})
    await store.aput(("test", "namespace2"), "key2", {"data": "value2"})

    ops = [ListNamespacesOp(match_conditions=None, max_depth=None, limit=10, offset=0)]

    results = await store.abatch(ops)

    assert len(results) == 1
    assert len(results[0]) == 2
    assert ("test", "namespace1") in results[0]
    assert ("test", "namespace2") in results[0]


# Vector store fixtures are imported from conftest_store.py


async def test_vector_store_initialization(
    vector_store: AsyncOracleStore, fake_embeddings: CharacterEmbeddings
) -> None:
    """Test store initialization with embedding config."""
    assert vector_store.index_config is not None
    assert vector_store.index_config["dims"] == fake_embeddings.dims
    if isinstance(vector_store.index_config["embed"], Embeddings):
        assert vector_store.index_config["embed"] == fake_embeddings


async def test_vector_insert_with_auto_embedding(
    vector_store: AsyncOracleStore,
) -> None:
    """Test inserting items that get auto-embedded."""
    docs = [
        ("doc1", {"text": "short text"}),
        ("doc2", {"text": "longer text document"}),
        ("doc3", {"text": "longest text document here"}),
        ("doc4", {"description": "text in description field"}),
        ("doc5", {"content": "text in content field"}),
        ("doc6", {"body": "text in body field"}),
    ]

    for key, value in docs:
        await vector_store.aput(("test",), key, value)

    results = await vector_store.asearch(("test",), query="long text")
    assert len(results) > 0

    doc_order = [r.key for r in results]
    assert "doc2" in doc_order
    assert "doc3" in doc_order


async def test_vector_update_with_embedding(vector_store: AsyncOracleStore) -> None:
    """Test that updating items properly updates their embeddings."""
    await vector_store.aput(("test",), "doc1", {"text": "zany zebra Xerxes"})
    await vector_store.aput(("test",), "doc2", {"text": "something about dogs"})
    await vector_store.aput(("test",), "doc3", {"text": "text about birds"})

    results_initial = await vector_store.asearch(("test",), query="Zany Xerxes")
    assert len(results_initial) > 0
    assert results_initial[0].key == "doc1"
    initial_score = results_initial[0].score

    await vector_store.aput(("test",), "doc1", {"text": "new text about dogs"})

    results_after = await vector_store.asearch(("test",), query="Zany Xerxes")
    after_score = next((r.score for r in results_after if r.key == "doc1"), 0.0)
    assert after_score < initial_score

    results_new = await vector_store.asearch(("test",), query="new text about dogs")
    for r in results_new:
        if r.key == "doc1":
            assert r.score > after_score

    # Don't index this one
    await vector_store.aput(
        ("test",), "doc4", {"text": "new text about dogs"}, index=False
    )
    results_new = await vector_store.asearch(
        ("test",), query="new text about dogs", limit=3
    )
    assert not any(r.key == "doc4" for r in results_new)


async def test_vector_store_concurrent_updates_keep_single_rows(
    vector_store: AsyncOracleStore,
) -> None:
    """Concurrent updates to the same key should not leave duplicate store rows."""
    payloads = [{"text": f"concurrent value {i}"} for i in range(8)]

    await asyncio.gather(
        *(vector_store.aput(("test",), "shared-doc", payload) for payload in payloads)
    )

    item = await vector_store.aget(("test",), "shared-doc")
    assert item is not None
    assert item.value in payloads

    with oracledb.connect(**DEFAULT_CONNECTION_INFO) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT COUNT(*) FROM {vector_store.table_names['store']} "
                "WHERE prefix = :1 AND key = :2",
                ("test", "shared-doc"),
            )
            assert cur.fetchone()[0] == 1

            cur.execute(
                f"SELECT COUNT(*) FROM {vector_store.table_names['store_vectors']} "
                "WHERE prefix = :1 AND key = :2",
                ("test", "shared-doc"),
            )
            assert cur.fetchone()[0] == 1


async def test_vector_search_with_filters(vector_store: AsyncOracleStore) -> None:
    """Test combining vector search with filters."""
    docs = [
        ("doc1", {"text": "red apple", "color": "red", "score": 4.5}),
        ("doc2", {"text": "red car", "color": "red", "score": 3.0}),
        ("doc3", {"text": "green apple", "color": "green", "score": 4.0}),
        ("doc4", {"text": "blue car", "color": "blue", "score": 3.5}),
    ]

    for key, value in docs:
        await vector_store.aput(("test",), key, value)

    results = await vector_store.asearch(
        ("test",), query="apple", filter={"color": "red"}
    )
    assert len(results) == 2
    assert results[0].key == "doc1"

    results = await vector_store.asearch(
        ("test",), query="car", filter={"color": "red"}
    )
    assert len(results) == 2
    assert results[0].key == "doc2"

    results = await vector_store.asearch(
        ("test",), query="bbbbluuu", filter={"score": {"$gt": 3.2}}
    )
    assert len(results) == 3
    assert results[0].key == "doc4"

    results = await vector_store.asearch(
        ("test",), query="apple", filter={"score": {"$gte": 4.0}, "color": "green"}
    )
    assert len(results) == 1
    assert results[0].key == "doc3"


async def test_vector_search_pagination(vector_store: AsyncOracleStore) -> None:
    """Test pagination with vector search."""
    for i in range(5):
        await vector_store.aput(
            ("test",), f"doc{i}", {"text": f"test document number {i}"}
        )

    results_page1 = await vector_store.asearch(("test",), query="test", limit=2)
    results_page2 = await vector_store.asearch(
        ("test",), query="test", limit=2, offset=2
    )

    assert len(results_page1) == 2
    assert len(results_page2) == 2
    assert results_page1[0].key != results_page2[0].key

    all_results = await vector_store.asearch(("test",), query="test", limit=10)
    assert len(all_results) == 5


async def test_vector_search_edge_cases(vector_store: AsyncOracleStore) -> None:
    """Test edge cases in vector search."""
    await vector_store.aput(("test",), "doc1", {"text": "test document"})

    perfect_match = await vector_store.asearch(("test",), query="text test document")
    perfect_score = perfect_match[0].score

    results = await vector_store.asearch(("test",), query="")
    assert len(results) == 1
    assert results[0].score is None

    results = await vector_store.asearch(("test",), query=None)
    assert len(results) == 1
    assert results[0].score is None

    long_query = "foo " * 100
    results = await vector_store.asearch(("test",), query=long_query)
    assert len(results) == 1
    assert results[0].score < perfect_score

    special_query = "test!@#$%^&*()"
    results = await vector_store.asearch(("test",), query=special_query)
    assert len(results) == 1
    assert results[0].score < perfect_score


@pytest.mark.parametrize(
    "index_type,distance_type",
    [
        *itertools.product(ORACLE_INDEX_TYPES, ORACLE_DISTANCE_TYPES),
    ],
)
async def test_embed_with_path(
    request: Any,
    fake_embeddings: CharacterEmbeddings,
    index_type: str,
    distance_type: str,
) -> None:
    """Test vector search with specific text fields in Oracle store."""
    async with create_async_vector_store_with_fields(
        index_type,
        distance_type,
        fake_embeddings,
        text_fields=["key0", "key1", "key3"],
    ) as store:
        # This will have 2 vectors representing it
        doc1 = {
            # Omit key0 - check it doesn't raise an error
            "key1": "xxx",
            "key2": "yyy",
            "key3": "zzz",
        }
        # This will have 3 vectors representing it
        doc2 = {
            "key0": "uuu",
            "key1": "vvv",
            "key2": "www",
            "key3": "xxx",
        }
        await store.aput(("test",), "doc1", doc1)
        await store.aput(("test",), "doc2", doc2)

        # doc2.key3 and doc1.key1 both would have the highest score
        results = await store.asearch(("test",), query="xxx")
        assert len(results) == 2
        assert results[0].key != results[1].key
        ascore = results[0].score
        bscore = results[1].score
        assert ascore == pytest.approx(bscore, abs=1e-3)

        results = await store.asearch(("test",), query="uuu")
        assert len(results) == 2
        assert results[0].key != results[1].key
        assert results[0].key == "doc2"
        assert results[0].score > results[1].score
        assert ascore == pytest.approx(results[0].score, abs=1e-3)

        # Un-indexed - will have low results for both. Not zero (because we're projecting)
        # but less than the above.
        results = await store.asearch(("test",), query="www")
        assert len(results) == 2
        assert results[0].score < ascore
        assert results[1].score < ascore


@pytest.mark.parametrize(
    "index_type,distance_type",
    [
        *itertools.product(ORACLE_INDEX_TYPES, ORACLE_DISTANCE_TYPES),
    ],
)
async def test_search_sorting(
    request: Any,
    fake_embeddings: CharacterEmbeddings,
    index_type: str,
    distance_type: str,
) -> None:
    """Test operation-level field configuration for vector search."""
    async with create_async_vector_store_with_fields(
        index_type,
        distance_type,
        fake_embeddings,
        text_fields=["key1"],  # Default fields that won't match our test data
    ) as store:
        amatch = {
            "key1": "mmm",
        }

        await store.aput(("test", "M"), "M", amatch)
        N = 100
        for i in range(N):
            await store.aput(("test", "A"), f"A{i}", {"key1": "no"})
        for i in range(N):
            await store.aput(("test", "Z"), f"Z{i}", {"key1": "no"})

        results = await store.asearch(("test",), query="mmm", limit=10)
        assert len(results) == 10
        assert len(set(r.key for r in results)) == 10
        assert results[0].key == "M"
        assert results[0].score > results[1].score


async def test_store_ttl(store):
    # Assumes a TTL of 1 minute = 60 seconds
    ns = ("foo",)
    await store.start_ttl_sweeper()
    await store.aput(
        ns,
        key="item1",
        value={"foo": "bar"},
        ttl=TTL_MINUTES,  # type: ignore
    )
    await asyncio.sleep(TTL_SECONDS - 2)
    res = await store.aget(ns, key="item1", refresh_ttl=True)
    assert res is not None
    await asyncio.sleep(TTL_SECONDS - 2)
    results = await store.asearch(ns, query="foo", refresh_ttl=True)
    assert len(results) == 1
    await asyncio.sleep(TTL_SECONDS - 2)
    res = await store.aget(ns, key="item1", refresh_ttl=False)
    assert res is not None
    await asyncio.sleep(TTL_SECONDS - 1)
    # Now has been more than TTL_SECONDS
    results = await store.asearch(ns, query="bar", refresh_ttl=False)
    assert len(results) == 0


async def test_asearch_ttl_refresh_errors_are_best_effort(monkeypatch):
    class FakeDatabaseError(Exception):
        pass

    monkeypatch.setattr(oracle_store_aio.oracledb, "DatabaseError", FakeDatabaseError)

    store = AsyncOracleStore(Mock(), ttl={"default_ttl": 1, "refresh_on_read": True})
    store.table_names = {"store": "store_test"}

    now = datetime.datetime.now(datetime.timezone.utc)
    cursor = _FakeAsyncSearchCursor(
        [("foo", "item1", {"foo": "bar"}, now, now, None)],
        update_error=FakeDatabaseError(SimpleNamespace(code=54)),
    )

    results = [None]
    prepared_statements = (
        [
            (
                "SELECT prefix, key, value, created_at, updated_at, NULL AS score FROM store_test",
                {},
            )
        ],
        [],
    )
    search_ops = [
        (
            0,
            SearchOp(
                namespace_prefix=("foo",),
                filter=None,
                limit=10,
                offset=0,
                query=None,
                refresh_ttl=True,
            ),
        )
    ]

    await store._batch_search_ops(prepared_statements, search_ops, results, cursor)

    assert len(results[0]) == 1
    assert results[0][0].key == "item1"
    assert cursor.execute_calls == 2
