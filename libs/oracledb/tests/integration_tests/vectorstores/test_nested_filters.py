# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Integration tests for nested metadata filters in OracleVS (issue #272).

Run the same nested-path behavior on every configured SQL, VecDB, and REST backend.
"""

from __future__ import annotations

from typing import Any, List

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from langchain_oracledb.vectorstores import OracleVS

from ._backend_fixtures import backend_store

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
def filter_seed(oracle_backend) -> Any:
    with backend_store(oracle_backend, DeterministicFakeEmbedding(size=64)) as store:
        store.add_documents(DOCS)
        yield store


@pytest.fixture
def vectorstore(filter_seed) -> OracleVS:
    return filter_seed


def _contents(results: List[Document]) -> List[str]:
    return sorted(d.page_content for d in results)


def test_nested_path_scalar(vectorstore: OracleVS) -> None:
    r = vectorstore.similarity_search("x", k=5, filter={"address": {"city": "NYC"}})
    assert _contents(r) == ["alpha", "gamma"]


def test_nested_siblings_with_operator(vectorstore: OracleVS) -> None:
    if vectorstore.backend == "vecdb":
        with pytest.raises(
            ValueError, match=r"Unsupported VecDB filter address.zip \$ne"
        ):
            vectorstore.similarity_search(
                "x", k=5, filter={"address": {"city": "NYC", "zip": {"$ne": "10001"}}}
            )
        return
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
