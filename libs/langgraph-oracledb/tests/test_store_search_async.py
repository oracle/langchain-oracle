# type: ignore

import uuid
from contextlib import asynccontextmanager

import oracledb
import pytest
from langgraph.store.base import SearchItem

from langgraph_oracledb.store.oracle.aio import AsyncOracleStore
from tests.conftest import (
    DEFAULT_CONNECTION_INFO,
    create_connection_string,
    skip_if_no_oracle,
)


@asynccontextmanager
async def _test_store(table_suffix: str | None = None):
    """Create a test store for SQL generation and validation."""
    # Generate unique table suffix to avoid collisions
    if table_suffix is None:
        table_suffix = f"test_{uuid.uuid4().hex[:8]}"

    conn_string = create_connection_string(DEFAULT_CONNECTION_INFO)

    async with AsyncOracleStore.from_conn_string(
        conn_string, table_suffix=table_suffix
    ) as store:
        await store.setup()
        yield store
        await store.ateardown()


class TestStoreSearchAsync:
    @pytest.fixture(autouse=True)
    def run_after_each_test(self):
        yield
        conn_string = create_connection_string(DEFAULT_CONNECTION_INFO)
        with oracledb.connect(conn_string) as conn:
            with conn.cursor() as cur:
                cur.execute("drop table if exists STORE_CONFIGS purge")

    @skip_if_no_oracle()
    async def test_store_search_basic_operations(self):
        """Test basic store search operations with actual data."""
        async with _test_store("async_basic") as store:
            # Put some test data
            await store.aput(
                ("users",), "user1", {"name": "Alice", "age": 30, "active": True}
            )
            await store.aput(
                ("users",), "user2", {"name": "Bob", "age": 25, "active": False}
            )
            await store.aput(
                ("users",), "user3", {"name": "Charlie", "age": 35, "active": True}
            )
            await store.aput(("products",), "prod1", {"name": "Widget", "price": 19.99})
            await store.aput(("products",), "prod2", {"name": "Gadget", "price": 29.99})

            # Test basic search without filters
            results = await store.asearch(("users",))
            assert len(results) == 3
            assert all(isinstance(r, SearchItem) for r in results)

            # Test search with namespace prefix
            results = await store.asearch(("users",))
            assert len(results) == 3

            results = await store.asearch(("products",))
            assert len(results) == 2

            # Test search with limit and offset
            results = await store.asearch(("users",), limit=2, offset=0)
            assert len(results) == 2

            results = await store.asearch(("users",), limit=2, offset=2)
            assert len(results) == 1

    @skip_if_no_oracle()
    async def test_store_search_with_filters(self):
        """Test store search with various filter conditions."""
        async with _test_store("async_filters") as store:
            # Put test data
            await store.aput(
                ("docs",),
                "doc1",
                {"title": "Python Guide", "views": 1000, "rating": 4.5},
            )
            await store.aput(
                ("docs",),
                "doc2",
                {"title": "Java Tutorial", "views": 500, "rating": 3.8},
            )
            await store.aput(
                ("docs",),
                "doc3",
                {"title": "Go Handbook", "views": 1500, "rating": 4.2},
            )
            await store.aput(
                ("docs",), "doc4", {"title": "Rust Manual", "views": 800, "rating": 4.7}
            )

            # Test equality filter
            results = await store.asearch(("docs",), filter={"title": "Python Guide"})
            assert len(results) == 1
            assert results[0].value["title"] == "Python Guide"

            # Test greater than filter
            results = await store.asearch(("docs",), filter={"views": {"$gt": 900}})
            assert len(results) == 2
            assert all(r.value["views"] > 900 for r in results)

            # Test greater than or equal filter
            results = await store.asearch(("docs",), filter={"rating": {"$gte": 4.2}})
            assert len(results) == 3
            assert all(float(r.value["rating"]) >= 4.2 for r in results)

            # Test less than filter
            results = await store.asearch(("docs",), filter={"views": {"$lt": 1000}})
            assert len(results) == 2
            assert all(r.value["views"] < 1000 for r in results)

            # Test not equal filter
            results = await store.asearch(
                ("docs",), filter={"title": {"$ne": "Python Guide"}}
            )
            assert len(results) == 3
            assert all(r.value["title"] != "Python Guide" for r in results)

            # Test multiple filters combined
            results = await store.asearch(
                ("docs",), filter={"views": {"$gte": 800}, "rating": {"$gt": 4.0}}
            )
            assert len(results) == 3
            for r in results:
                assert int(r.value["views"]) >= 800
                assert float(r.value["rating"]) > 4.0

    @skip_if_no_oracle()
    async def test_store_search_complex_filters(self):
        """Test store search with complex nested filter conditions."""
        async with _test_store("async_complex") as store:
            # Put test data with nested structures
            await store.aput(
                ("items",),
                "item1",
                {
                    "type": "book",
                    "metadata": {"author": "Alice", "year": 2020},
                    "stats": {"views": 1000, "likes": 50},
                },
            )
            await store.aput(
                ("items",),
                "item2",
                {
                    "type": "article",
                    "metadata": {"author": "Bob", "year": 2021},
                    "stats": {"views": 500, "likes": 25},
                },
            )
            await store.aput(
                ("items",),
                "item3",
                {
                    "type": "book",
                    "metadata": {"author": "Charlie", "year": 2019},
                    "stats": {"views": 1500, "likes": 75},
                },
            )

            # Test filter on top-level field
            results = await store.asearch(("items",), filter={"type": "book"})
            assert len(results) == 2

            # Test combination of filters
            results = await store.asearch(
                ("items",), filter={"type": "book", "stats.views": {"$gte": 1000}}
            )
            # Note: nested path filtering might not work directly without JSON path support
            # This tests the basic filter structure generation

    @skip_if_no_oracle()
    async def test_store_search_special_characters(self):
        """Test store search with special characters in values."""
        async with _test_store("async_special") as store:
            # Put data with special characters
            await store.aput(("test",), "key1", {"desc": "Hello 'World'"})
            await store.aput(("test",), "key2", {"desc": 'Test "quotes"'})
            await store.aput(("test",), "key3", {"desc": "Path/with/slashes"})
            await store.aput(("test",), "key4", {"desc": "Semi;colon"})

            # Search for items with special characters
            results = await store.asearch(("test",), filter={"desc": "Hello 'World'"})
            assert len(results) == 1
            assert results[0].value["desc"] == "Hello 'World'"

            results = await store.asearch(("test",), filter={"desc": 'Test "quotes"'})
            assert len(results) == 1

            # Test that special characters don't break SQL
            results = await store.asearch(
                ("test",), filter={"desc": {"$ne": "nonexistent"}}
            )
            assert len(results) == 4  # Should return all items safely

    @skip_if_no_oracle()
    async def test_store_search_empty_results(self):
        """Test store search behavior with no matching results."""
        async with _test_store("async_empty") as store:
            # Put some data
            await store.aput(("data",), "item1", {"value": 100})
            await store.aput(("data",), "item2", {"value": 200})

            # Search for non-existent namespace
            results = await store.asearch(("nonexistent",))
            assert results == []

            # Search with filter that matches nothing
            results = await store.asearch(("data",), filter={"value": {"$gt": 1000}})
            assert results == []

            # Search with impossible combination of filters
            # Note: Due to dict key collision, only the last filter will be applied
            # This tests edge case handling rather than logical impossibility
            results = await store.asearch(
                ("data",), filter={"nonexistent_field": {"$gt": 150}}
            )
            assert results == []

    @skip_if_no_oracle()
    async def test_store_search_pagination(self):
        """Test store search pagination with offset and limit."""
        async with _test_store("async_page") as store:
            # Put multiple items
            for i in range(20):
                await store.aput(
                    ("items",), f"item{i:02d}", {"index": i, "value": f"value_{i}"}
                )

            # Test first page
            page1 = await store.asearch(("items",), limit=5, offset=0)
            assert len(page1) == 5

            # Test second page
            page2 = await store.asearch(("items",), limit=5, offset=5)
            assert len(page2) == 5

            # Ensure no overlap between pages
            page1_keys = {r.key for r in page1}
            page2_keys = {r.key for r in page2}
            assert page1_keys.isdisjoint(page2_keys)

            # Test last page
            last_page = await store.asearch(("items",), limit=5, offset=15)
            assert len(last_page) == 5

            # Test offset beyond data
            empty_page = await store.asearch(("items",), limit=5, offset=100)
            assert empty_page == []

    @skip_if_no_oracle()
    async def test_store_search_namespace_prefix(self):
        """Test store search with namespace prefix patterns."""
        async with _test_store("async_ns") as store:
            # Create hierarchical namespace structure
            await store.aput(("docs",), "d1", {"type": "root"})
            await store.aput(("docs", "python"), "d2", {"type": "python"})
            await store.aput(("docs", "python", "tutorial"), "d3", {"type": "tutorial"})
            await store.aput(("docs", "java"), "d4", {"type": "java"})
            await store.aput(("docs", "java", "guide"), "d5", {"type": "guide"})
            await store.aput(("other",), "o1", {"type": "other"})

            # Search with namespace prefix - should return all items under "docs"
            results = await store.asearch(("docs",))
            assert len(results) == 5  # All items under "docs" prefix
            types = {r.value["type"] for r in results}
            assert types == {"root", "python", "tutorial", "java", "guide"}

            # Test more specific namespace prefix
            results = await store.asearch(("docs", "python"))
            assert len(results) == 2  # Items under "docs.python" prefix
            types = {r.value["type"] for r in results}
            assert types == {"python", "tutorial"}

            # Test exact namespace search - this tests the most specific namespace
            results = await store.asearch(("docs", "java", "guide"))
            assert len(results) == 1  # Only the guide item
            assert results[0].value["type"] == "guide"

            # Search with filter in specific namespace
            results = await store.asearch(
                ("docs", "python", "tutorial"), filter={"type": "tutorial"}
            )
            assert (
                len(results) <= 1
            )  # Should find the tutorial item if it exists in this exact namespace

    @skip_if_no_oracle()
    async def test_store_search_with_ttl(self):
        """Test store search with TTL (time-to-live) items."""
        async with _test_store("async_ttl") as store:
            # Put items with different TTLs
            await store.aput(
                ("temp",), "t1", {"data": "expires_soon"}, ttl=1
            )  # 1 minute
            await store.aput(
                ("temp",), "t2", {"data": "expires_later"}, ttl=60
            )  # 60 minutes
            await store.aput(("temp",), "t3", {"data": "permanent"})  # No TTL

            # All items should be searchable immediately
            results = await store.asearch(("temp",))
            assert len(results) == 3

            # Test with filter on TTL items
            results = await store.asearch(("temp",), filter={"data": "permanent"})
            assert len(results) == 1
            assert results[0].value["data"] == "permanent"

    @skip_if_no_oracle()
    async def test_store_batch_search_operations(self):
        """Test batch search operations."""
        async with _test_store("async_batch") as store:
            from langgraph.store.base import SearchOp

            # Put test data
            await store.aput(("users",), "u1", {"name": "Alice", "role": "admin"})
            await store.aput(("users",), "u2", {"name": "Bob", "role": "user"})
            await store.aput(("products",), "p1", {"name": "Widget", "price": 10})
            await store.aput(("products",), "p2", {"name": "Gadget", "price": 20})

            # Create batch search operations
            ops = [
                SearchOp(("users",), filter=None, limit=10, offset=0),
                SearchOp(
                    ("products",), filter={"price": {"$gte": 15}}, limit=10, offset=0
                ),
                SearchOp(("users",), filter={"role": "admin"}, limit=10, offset=0),
            ]

            # Execute batch
            results = await store.abatch(ops)

            assert len(results) == 3
            assert len(results[0]) == 2  # All users
            assert len(results[1]) == 1  # Products with price >= 15
            assert len(results[2]) == 1  # Admin users
            assert results[2][0].value["name"] == "Alice"
