# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""Native backends share filter results, including SQL-only predicates and MMR."""

import math

import pytest
from langchain_core.embeddings import Embeddings

from langchain_oracledb.vectorstores._vecdb_filters import (
    SQLFilterRequired,
    translate_filter,
)

from . import test_filter_contract
from ._backend_fixtures import backend_store

filter_store = test_filter_contract.filter_store

VALUES = [
    None,
    "",
    False,
    True,
    1,
    "1",
    "tea",
    "soda",
    ["tea", "soda"],
    [],
    {},
    {"a": 1},
    {"a": 1, "b": 2},
    ["1", 2],
    ["tea", None],
    [{"a": 1}],
    ["soda", "tea"],
]
METADATA = [{"group": "edge"}] + [{"group": "edge", "x": v} for v in VALUES]
TEXTS = [f"edge {i}" for i in range(len(METADATA))]
# More near, nonmatching rows than either k or fetch_k. A post-filtered top-k
# cannot find the two valid rows at greater distances.
for i in range(10):
    TEXTS.append(f"near {i}")
    METADATA.append({"payload": {"skip": True}, "tag": "skip"})
TEXTS += ["middle", "far"]
METADATA += [{"payload": {"keep": True}, "tag": "keep"} for _ in range(2)]


class FilterEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [
            [0.8, 0.6, 0.0]
            if t == "middle"
            else [0.0, 1.0, 0.0]
            if t == "far"
            else [1.0, 0.0, 0.0]
            for t in texts
        ]

    def embed_query(self, text):
        return [1.0, 0.0, 0.0]


@pytest.fixture(scope="module")
def filter_seed(oracle_backend):
    with backend_store(oracle_backend, FilterEmbeddings()) as store:
        store.add_texts(TEXTS, METADATA, ids=TEXTS)
        yield store


def expected_ids(indices):
    return {f"edge {i}" for i in indices}


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    "predicate,indices",
    [
        ({"x": {"$ne": "tea"}}, [1, 2, 3, 4, 5, 6, 8, 9, 11, 12, 13, 14, 15, 16, 17]),
        ({"x": {"$eq": 1}}, [5, 6, 14]),
        ({"x": {"$eq": True}}, [4]),
        ({"x": {"$eq": False}}, [3]),
        ({"$or": [{"x": {"$eq": True}}, {"x": {"$eq": {"never": "matches"}}}]}, [4]),
        ({"$or": [{"x": {"$eq": False}}, {"x": {"$eq": {"never": "matches"}}}]}, [3]),
        ({"x": {"$eq": None}}, [2]),
        ({"x": {"$eq": ""}}, [2]),
        ({"x": {"$ne": None}}, [1, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17]),
        ({"x": {"$regex": "[0-9]+"}}, [5, 6, 14]),
        ({"x": {"$regex": ""}}, []),
        (
            {"x": {"$not": {"$eq": "tea"}}},
            [0, 1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 16],
        ),
        ({"x": {"$eq": ["tea", "soda"]}}, [9]),
        ({"x": {"$eq": {"a": 1}}}, [12]),
        (
            {"x": {"$ne": {"a": 1}}},
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17],
        ),
        ({"$or": [{"x": {"$eq": {"a": 1}}}, {"x": {"$eq": ["tea", "soda"]}}]}, [9, 12]),
    ],
)
@pytest.mark.asyncio
async def test_filter_edge_cases(
    filter_store, oracle_backend, asynchronous, predicate, indices
):
    predicate = {"group": "edge", **predicate}
    if oracle_backend == "rest":
        try:
            translate_filter(predicate)
        except SQLFilterRequired:
            with pytest.raises(ValueError, match="Unsupported VecDB filter"):
                if asynchronous:
                    await filter_store.asimilarity_search(
                        "query", k=100, filter=predicate
                    )
                else:
                    filter_store.similarity_search("query", k=100, filter=predicate)
            return
    hits = (
        await filter_store.asimilarity_search("query", k=100, filter=predicate)
        if asynchronous
        else filter_store.similarity_search("query", k=100, filter=predicate)
    )
    assert {d.id for d in hits} == expected_ids(indices)
    for doc in hits:
        assert doc.metadata == METADATA[TEXTS.index(doc.id)]
        assert doc.page_content == doc.id


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    "predicate",
    [
        {"payload": {"$eq": {"keep": True}}},
        {"$or": [{"payload": {"$eq": {"keep": True}}}, {"tag": "never"}]},
        {"tag": "keep", "payload": {"$ne": {"skip": True}}},
    ],
)
@pytest.mark.asyncio
async def test_sql_filter_precedes_limit_and_mmr(
    filter_store, oracle_backend, asynchronous, predicate
):
    store = filter_store
    if oracle_backend == "rest":
        with pytest.raises(ValueError, match="Unsupported VecDB filter"):
            if asynchronous:
                await store.asimilarity_search("query", k=1, filter=predicate)
            else:
                store.similarity_search("query", k=1, filter=predicate)
        return
    if asynchronous:
        hits = await store.asimilarity_search_by_vector_returning_embeddings(
            [1.0, 0.0, 0.0], k=2, filter=predicate
        )
        mmr = await store.amax_marginal_relevance_search(
            "query", k=1, fetch_k=1, filter=predicate
        )
        limited = await store.asimilarity_search("query", k=1, db_filter=predicate)
    else:
        hits = store.similarity_search_by_vector_returning_embeddings(
            [1.0, 0.0, 0.0], k=2, filter=predicate
        )
        mmr = store.max_marginal_relevance_search(
            "query", k=1, fetch_k=1, filter=predicate
        )
        limited = store.similarity_search("query", k=1, db_filter=predicate)
    assert [h[0].id for h in hits] == ["middle", "far"]
    assert [h[1] for h in hits] == pytest.approx([math.sqrt(0.4), math.sqrt(2)])
    assert list(hits[0][2]) == pytest.approx([0.8, 0.6, 0.0])
    assert list(hits[1][2]) == [0.0, 1.0, 0.0]
    assert [d.id for d in mmr] == ["middle"]
    assert [d.id for d in limited] == ["middle"]
    assert hits[0][0].metadata == {"payload": {"keep": True}, "tag": "keep"}
