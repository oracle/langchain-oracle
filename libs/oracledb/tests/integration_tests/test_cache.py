# Copyright (c) 2025, 2026, Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
import uuid
from collections.abc import Generator

import oracledb
import pytest
from langchain_core.caches import BaseCache
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.outputs import Generation
from langchain_tests.integration_tests import SyncCacheTestSuite

from langchain_oracledb.cache import OracleSemanticCache, _cache_entry_id, _hash_value
from langchain_oracledb.vectorstores.utils import _index_exists, _quote_indentifier

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
    yield f"SEM_CACHE_{uuid.uuid4().hex[:8].upper()}"


class TestOracleSemanticCacheStandard(SyncCacheTestSuite):
    @pytest.fixture
    def cache(
        self,
        connection: oracledb.Connection,
        cache_table_name: str,
    ) -> Generator[BaseCache, None, None]:
        cache = OracleSemanticCache(
            client=connection,
            embedding=DeterministicFakeEmbedding(size=6),
            table_name=cache_table_name,
        )
        try:
            yield cache
        finally:
            OracleSemanticCache.drop_table(connection, cache_table_name)


def test_semantic_cache_isolated_by_llm_string(
    connection: oracledb.Connection, cache_table_name: str
) -> None:
    cache = OracleSemanticCache(
        client=connection,
        embedding=DeterministicFakeEmbedding(size=6),
        table_name=cache_table_name,
    )

    try:
        cache.update(
            "What is the capital of France?",
            "model-a",
            [Generation(text="Paris from A")],
        )
        cache.update(
            "What is the capital of France?",
            "model-b",
            [Generation(text="Paris from B")],
        )

        assert cache.lookup("What is the capital of France?", "model-a") == [
            Generation(text="Paris from A")
        ]
        assert cache.lookup("What is the capital of France?", "model-b") == [
            Generation(text="Paris from B")
        ]
    finally:
        OracleSemanticCache.drop_table(connection, cache_table_name)


def test_semantic_cache_clear_supports_llm_filter(
    connection: oracledb.Connection, cache_table_name: str
) -> None:
    cache = OracleSemanticCache(
        client=connection,
        embedding=DeterministicFakeEmbedding(size=6),
        table_name=cache_table_name,
    )

    try:
        cache.update("prompt one", "model-a", [Generation(text="A1")])
        cache.update("prompt two", "model-b", [Generation(text="B1")])

        cache.clear(llm_string="model-a")

        assert cache.lookup("prompt one", "model-a") is None
        assert cache.lookup("prompt two", "model-b") == [Generation(text="B1")]
    finally:
        OracleSemanticCache.drop_table(connection, cache_table_name)


def test_semantic_cache_score_threshold_uses_max_distance(
    connection: oracledb.Connection, cache_table_name: str
) -> None:
    cache = OracleSemanticCache(
        client=connection,
        embedding=DeterministicFakeEmbedding(size=6),
        table_name=cache_table_name,
        score_threshold=0.0,
    )

    try:
        cache.update(
            "oracle database semantic cache",
            "model-a",
            [Generation(text="cached")],
        )

        assert cache.lookup("oracle database semantic cache", "model-a") == [
            Generation(text="cached")
        ]
        assert cache.lookup("completely different question", "model-a") is None
    finally:
        OracleSemanticCache.drop_table(connection, cache_table_name)


def test_semantic_cache_clear_supports_prompt_filter(
    connection: oracledb.Connection, cache_table_name: str
) -> None:
    cache = OracleSemanticCache(
        client=connection,
        embedding=DeterministicFakeEmbedding(size=6),
        table_name=cache_table_name,
    )

    try:
        cache.update("prompt one", "model-a", [Generation(text="A1")])
        cache.update("prompt two", "model-a", [Generation(text="A2")])

        cache.clear(prompt="prompt one")

        with connection.cursor() as cursor:
            cursor.execute(
                f"SELECT COUNT(*) FROM {_quote_indentifier(cache_table_name)} "
                "WHERE JSON_VALUE(metadata, '$.prompt_hash') = :prompt_hash",
                prompt_hash=_hash_value("prompt one"),
            )
            removed_count = cursor.fetchone()[0]
            cursor.execute(
                f"SELECT COUNT(*) FROM {_quote_indentifier(cache_table_name)} "
                "WHERE JSON_VALUE(metadata, '$.prompt_hash') = :prompt_hash",
                prompt_hash=_hash_value("prompt two"),
            )
            remaining_count = cursor.fetchone()[0]

        assert removed_count == 0
        assert remaining_count == 1
    finally:
        OracleSemanticCache.drop_table(connection, cache_table_name)


def test_semantic_cache_lookup_returns_none_for_non_string_return_val(
    connection: oracledb.Connection, cache_table_name: str
) -> None:
    cache = OracleSemanticCache(
        client=connection,
        embedding=DeterministicFakeEmbedding(size=6),
        table_name=cache_table_name,
    )

    try:
        cache._vector_store.add_texts(
            ["prompt one"],
            [
                {
                    cache.PROMPT_HASH: _hash_value("prompt one"),
                    cache.LLM_HASH: _hash_value("model-a"),
                    cache.RETURN_VAL: ["not", "a", "string"],
                }
            ],
            ids=[_cache_entry_id("prompt one", "model-a")],
        )

        assert cache.lookup("prompt one", "model-a") is None
    finally:
        OracleSemanticCache.drop_table(connection, cache_table_name)


def test_semantic_cache_clear_rejects_unsupported_filters(
    connection: oracledb.Connection, cache_table_name: str
) -> None:
    cache = OracleSemanticCache(
        client=connection,
        embedding=DeterministicFakeEmbedding(size=6),
        table_name=cache_table_name,
    )

    try:
        with pytest.raises(ValueError, match="Unsupported clear filters: unknown"):
            cache.clear(unknown="value")
    finally:
        OracleSemanticCache.drop_table(connection, cache_table_name)


def test_semantic_cache_can_create_vector_index_on_init(
    connection: oracledb.Connection, cache_table_name: str
) -> None:
    index_name = f"IDX_{cache_table_name}"
    OracleSemanticCache(
        client=connection,
        embedding=DeterministicFakeEmbedding(size=6),
        table_name=cache_table_name,
        create_index_if_missing=True,
        index_name=index_name,
    )

    try:
        assert _index_exists(
            connection,
            _quote_indentifier(index_name),
            cache_table_name,
        )
    finally:
        OracleSemanticCache.drop_table(connection, cache_table_name)
