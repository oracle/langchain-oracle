# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""SQL hybrid retrieval batches documents without losing rank or score identity."""

import json
from unittest.mock import AsyncMock, MagicMock

import oracledb
import pytest

from langchain_oracledb.retrievers.hybrid_search import OracleHybridSearchRetriever
from langchain_oracledb.vectorstores.oraclevs import INTERNAL_ID_KEY, OracleVS


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("scores", [False, True])
@pytest.mark.parametrize("case", ["empty", "ranked", "missing", "duplicate", "large"])
@pytest.mark.asyncio
async def test_sql_hybrid_batches_document_fetches(asynchronous, scores, case):
    count = 0 if case == "empty" else 2001 if case == "large" else 3
    hits = [
        {
            "rowid": f"RID{i:015d}",
            "score": 100 - i / 10,
            "text_score": i / 10,
            "vector_score": 10 + i / 10,
        }
        for i in range(count)
    ]
    if case == "duplicate":
        hits[2]["rowid"] = hits[0]["rowid"]
    expected_positions = [i for i in range(count) if case != "missing" or i != 1]
    documents = {
        hit["rowid"]: (hit["rowid"], {INTERNAL_ID_KEY: hit["rowid"], "tenant": "one"})
        for hit in hits
    }
    cursor = MagicMock()
    fetched = [
        (i + 1, documents[hits[i]["rowid"]][0], documents[hits[i]["rowid"]][1])
        for i in expected_positions
    ]
    rows = []

    def execute(sql, *args, **kwargs):
        nonlocal rows
        if "DBMS_HYBRID_VECTOR.SEARCH" in sql:
            rows = [(json.dumps(hits),)]
        elif "JSON_TABLE" in sql:
            rows = fetched
        else:
            rowid = args[0][0]
            position = next(i for i, hit in enumerate(hits) if hit["rowid"] == rowid)
            rows = [documents[rowid]] if position in expected_positions else []

    cursor.execute = (AsyncMock if asynchronous else MagicMock)(side_effect=execute)
    cursor.fetchall = (AsyncMock if asynchronous else MagicMock)(
        side_effect=lambda: rows
    )
    client = MagicMock(
        spec=oracledb.AsyncConnection if asynchronous else oracledb.Connection
    )
    client.cursor.return_value.__enter__.return_value = cursor
    client.cursor.return_value.__aenter__.return_value = cursor
    store = MagicMock(spec=OracleVS)
    store.client = client
    store.table_name = '"OWNER"."DOCS"'
    store._vecdb = None
    retriever = OracleHybridSearchRetriever(
        vector_store=store, idx_name="HYBRID_IDX", return_scores=scores, k=max(count, 1)
    )
    result = (
        await retriever.ainvoke("oracle")
        if asynchronous
        else retriever.invoke("oracle")
    )
    assert [doc.id for doc in result] == [hits[i]["rowid"] for i in expected_positions]
    assert [doc.page_content for doc in result] == [doc.id for doc in result]
    for doc, position in zip(result, expected_positions):
        expected = {"tenant": "one"}
        if scores:
            expected.update(
                {
                    key: hits[position][key]
                    for key in ("score", "text_score", "vector_score")
                }
            )
        assert doc.metadata == expected
    assert all(INTERNAL_ID_KEY in metadata for _, metadata in documents.values())
    assert cursor.execute.call_count == (2 if hits else 1)
    if hits:
        sql = cursor.execute.call_args.args[0]
        assert "JSON_TABLE" in sql and "ORDER BY h.position" in sql
        assert 'JOIN "OWNER"."DOCS"' in sql
        assert hits[0]["rowid"] not in sql
        assert cursor.execute.call_args.kwargs["rowids"] == [
            hit["rowid"] for hit in hits
        ]
        assert cursor.setinputsizes.call_args.kwargs == {
            "rowids": oracledb.DB_TYPE_JSON
        }
