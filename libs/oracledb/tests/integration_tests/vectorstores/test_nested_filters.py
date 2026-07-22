# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Integration tests for nested metadata filters in OracleVS (issue #272).

Requires a reachable Oracle Database 23ai+ via the usual env vars:
VECDB_USER, VECDB_PASS, VECDB_HOST (dsn).
"""

from __future__ import annotations

import os
from typing import Any, List

import oracledb
import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from langchain_oracledb.vectorstores import DistanceStrategy, OracleVS
from langchain_oracledb.vectorstores.oraclevs import drop_table_purge

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

TABLE = "ORAVS_NESTED_FILTER_IT"

DOCS = [
    Document(
        page_content="alpha",
        metadata={"address": {"city": "NYC", "zip": "10001"}, "level": 1},
    ),
    Document(
        page_content="beta",
        metadata={"address": {"city": "LA", "zip": "90001"}, "level": 2},
    ),
    Document(
        page_content="gamma",
        metadata={"address": {"city": "NYC", "zip": "11201"}, "level": 3},
    ),
]


@pytest.fixture(scope="module")
def vectorstore() -> Any:
    conn = oracledb.connect(user=username, password=password, dsn=dsn)
    drop_table_purge(conn, TABLE)
    vs = OracleVS.from_documents(
        DOCS,
        DeterministicFakeEmbedding(size=64),
        client=conn,
        table_name=TABLE,
        distance_strategy=DistanceStrategy.COSINE,
    )
    yield vs
    drop_table_purge(conn, TABLE)


def _contents(results: List[Document]) -> List[str]:
    return sorted(d.page_content for d in results)


def test_nested_path_scalar(vectorstore: OracleVS) -> None:
    r = vectorstore.similarity_search("x", k=5, filter={"address": {"city": "NYC"}})
    assert _contents(r) == ["alpha", "gamma"]


def test_nested_siblings_with_operator(vectorstore: OracleVS) -> None:
    r = vectorstore.similarity_search(
        "x", k=5, filter={"address": {"city": "NYC", "zip": {"$ne": "10001"}}}
    )
    assert _contents(r) == ["gamma"]


def test_or_with_nested_branch(vectorstore: OracleVS) -> None:
    r = vectorstore.similarity_search(
        "x",
        k=5,
        filter={"$or": [{"address": {"city": "LA"}}, {"level": {"$gte": 3}}]},
    )
    assert _contents(r) == ["beta", "gamma"]


def test_logical_nesting_with_nested_path(vectorstore: OracleVS) -> None:
    r = vectorstore.similarity_search(
        "x",
        k=5,
        filter={
            "$and": [
                {"$or": [{"level": 1}, {"level": 2}]},
                {"address": {"zip": "90001"}},
            ]
        },
    )
    assert _contents(r) == ["beta"]


def test_nested_equals_dotted_spelling(vectorstore: OracleVS) -> None:
    nested = vectorstore.similarity_search(
        "x", k=5, filter={"address": {"city": "NYC"}}
    )
    dotted = vectorstore.similarity_search("x", k=5, filter={"address.city": "NYC"})
    assert _contents(nested) == _contents(dotted)
