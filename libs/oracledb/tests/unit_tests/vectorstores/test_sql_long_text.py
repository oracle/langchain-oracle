"""Large text input locators are cleaned up even when an upsert fails."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import oracledb
import pytest
from langchain_core.embeddings import Embeddings

from langchain_oracledb import OracleVS
from langchain_oracledb.vectorstores.oraclevs import (
    INTERNAL_ID_KEY,
    _aclose_bound_lob,
    _close_bound_lob,
)


class Embedding(Embeddings):
    def embed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0]


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_failed_upsert_releases_all_input_lobs(asynchronous):
    conn = MagicMock(
        spec=oracledb.AsyncConnection if asynchronous else oracledb.Connection
    )
    cursor = conn.cursor.return_value.__enter__.return_value
    lobs = [MagicMock(), MagicMock()]
    if asynchronous:
        for lob in lobs:
            lob.write = AsyncMock()
            lob.close = AsyncMock()
        conn.createlob = AsyncMock(side_effect=lobs)
        cursor.executemany = AsyncMock(side_effect=RuntimeError("upsert failed"))
    else:
        conn.createlob.side_effect = lobs
        cursor.executemany.side_effect = RuntimeError("upsert failed")
    store = OracleVS.__new__(OracleVS)
    store.client, store.embedding_function = conn, Embedding()
    store._vecdb, store.table_name, store.mutate_on_duplicate = None, '"DOCS"', True
    metadatas = [{"tenant": "a"}, {"tenant": "b"}]
    with pytest.raises(RuntimeError, match="upsert failed"):
        if asynchronous:
            await store.aadd_texts(["x" * 64000, "short"], metadatas, ids=["a", "b"])
        else:
            store.add_texts(["x" * 64000, "short"], metadatas, ids=["a", "b"])
    for lob in lobs:
        lob.close.assert_called_once_with()
    assert all(INTERNAL_ID_KEY not in metadata for metadata in metadatas)
    conn.commit.assert_not_called()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("code", [64219, 3113])
@pytest.mark.asyncio
async def test_only_consumed_locator_errors_are_ignored(asynchronous, code):
    error = oracledb.DatabaseError(SimpleNamespace(code=code))
    lob = MagicMock()
    lob.close = (
        AsyncMock(side_effect=error) if asynchronous else MagicMock(side_effect=error)
    )

    async def close():
        if asynchronous:
            await _aclose_bound_lob(lob)
        else:
            _close_bound_lob(lob)

    if code == 64219:
        await close()
    else:
        with pytest.raises(oracledb.DatabaseError):
            await close()
