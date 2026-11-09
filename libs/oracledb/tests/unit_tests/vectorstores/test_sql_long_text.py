"""The driver owns temporary LOBs bound to SQL upserts, including on failure."""

from unittest.mock import AsyncMock, MagicMock

import oracledb
import pytest
from langchain_core.embeddings import Embeddings

from langchain_oracledb import OracleVS
from langchain_oracledb.vectorstores.oraclevs import INTERNAL_ID_KEY


class Embedding(Embeddings):
    def embed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0]


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("failed", [False, True], ids=["success", "failure"])
@pytest.mark.asyncio
async def test_bound_lob_lifetime_is_owned_by_driver(asynchronous, failed):
    conn = MagicMock(
        spec=oracledb.AsyncConnection if asynchronous else oracledb.Connection
    )
    cursor = conn.cursor.return_value.__enter__.return_value
    cursor.getbatcherrors.return_value = []
    lobs = [MagicMock(), MagicMock()]
    if asynchronous:
        for lob in lobs:
            lob.write = AsyncMock()
            lob.close = AsyncMock()
        conn.createlob = AsyncMock(side_effect=lobs)
        conn.commit = AsyncMock()
        cursor.executemany = AsyncMock()
    else:
        conn.createlob.side_effect = lobs
    if failed:
        cursor.executemany.side_effect = RuntimeError("upsert failed")
    store = OracleVS.__new__(OracleVS)
    store.client, store.embedding_function = conn, Embedding()
    store._vecdb, store.table_name, store.mutate_on_duplicate = None, '"DOCS"', True
    metadatas = [{"tenant": "a"}, {"tenant": "b"}]
    texts = ["x" * 64000, "short"]

    async def upsert():
        if asynchronous:
            return await store.aadd_texts(texts, metadatas, ids=["a", "b"])
        return store.add_texts(texts, metadatas, ids=["a", "b"])

    if failed:
        with pytest.raises(RuntimeError, match="upsert failed"):
            await upsert()
        conn.commit.assert_not_called()
    else:
        assert await upsert() == ["a", "b"]
        conn.commit.assert_called_once_with()
    for lob, text in zip(lobs, texts):
        lob.write.assert_called_once_with(text)
        lob.close.assert_not_called()
    assert all(INTERNAL_ID_KEY not in metadata for metadata in metadatas)
