# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Integration tests for the exact-match OracleCache (issue #267)."""

import os
import uuid
from collections.abc import Generator

import oracledb
import pytest
from langchain_core.caches import BaseCache
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, Generation
from langchain_tests.integration_tests import SyncCacheTestSuite

from langchain_oracledb.cache import OracleCache

username = os.environ.get("VECDB_USER")
password = os.environ.get("VECDB_PASS")
dsn = os.environ.get("VECDB_HOST")

try:
    oracledb.connect(user=username, password=password, dsn=dsn)
except Exception as e:
    pytest.skip(
        allow_module_level=True,
        reason=f"Database connection failed: {e}, skipping tests.",
    )


@pytest.fixture(scope="function")
def connection() -> Generator[oracledb.Connection, None, None]:
    conn = oracledb.connect(user=username, password=password, dsn=dsn)
    try:
        yield conn
    finally:
        try:
            conn.close()
        except Exception:
            pass


@pytest.fixture(scope="function")
def cache_table_name() -> Generator[str, None, None]:
    yield f"EXACT_CACHE_{uuid.uuid4().hex[:8].upper()}"


@pytest.fixture(scope="function")
def exact_cache(
    connection: oracledb.Connection, cache_table_name: str
) -> Generator[OracleCache, None, None]:
    cache = OracleCache(client=connection, table_name=cache_table_name)
    try:
        yield cache
    finally:
        OracleCache.drop_table(connection, cache_table_name)


class TestOracleCacheStandard(SyncCacheTestSuite):
    @pytest.fixture
    def cache(
        self,
        connection: oracledb.Connection,
        cache_table_name: str,
    ) -> Generator[BaseCache, None, None]:
        cache = OracleCache(client=connection, table_name=cache_table_name)
        try:
            yield cache
        finally:
            OracleCache.drop_table(connection, cache_table_name)


def test_exact_cache_is_exact(exact_cache: OracleCache) -> None:
    """Near-identical prompts must NOT hit (unlike the semantic cache)."""
    exact_cache.update("what is 2+2?", "llm-a", [Generation(text="4")])

    hit = exact_cache.lookup("what is 2+2?", "llm-a")
    assert hit is not None and hit[0].text == "4"
    assert exact_cache.lookup("what is 2 + 2?", "llm-a") is None
    assert exact_cache.lookup("what is 2+2?", "llm-b") is None


def test_exact_cache_update_is_idempotent(exact_cache: OracleCache) -> None:
    exact_cache.update("p", "l", [Generation(text="v1")])
    exact_cache.update("p", "l", [Generation(text="v2")])

    hit = exact_cache.lookup("p", "l")
    assert hit is not None and hit[0].text == "v2"


def test_exact_cache_clear_filters(exact_cache: OracleCache) -> None:
    exact_cache.update("p1", "l1", [Generation(text="a")])
    exact_cache.update("p1", "l2", [Generation(text="b")])
    exact_cache.update("p2", "l1", [Generation(text="c")])

    exact_cache.clear(llm_string="l1")
    assert exact_cache.lookup("p1", "l1") is None
    assert exact_cache.lookup("p2", "l1") is None
    assert exact_cache.lookup("p1", "l2") is not None

    exact_cache.clear(prompt="p1")
    assert exact_cache.lookup("p1", "l2") is None


def test_exact_cache_clear_rejects_unsupported_filters(
    exact_cache: OracleCache,
) -> None:
    with pytest.raises(ValueError, match="Unsupported clear filters"):
        exact_cache.clear(bogus=True)


def test_exact_cache_skips_tool_call_generations(exact_cache: OracleCache) -> None:
    gen = ChatGeneration(
        message=AIMessage(
            content="",
            tool_calls=[{"name": "t", "args": {}, "id": "x", "type": "tool_call"}],
        )
    )
    exact_cache.update("p", "l", [gen])
    assert exact_cache.lookup("p", "l") is None


def test_exact_cache_resets_cached_message_ids(exact_cache: OracleCache) -> None:
    gen = ChatGeneration(message=AIMessage(content="hi", id="lc_run--original"))
    exact_cache.update("p", "l", [gen])

    hit = exact_cache.lookup("p", "l")
    assert hit is not None
    assert hit[0].message.id is None  # type: ignore[union-attr]
