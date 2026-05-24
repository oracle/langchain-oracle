# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Integration tests for the ADB datastore review-fix behaviours.

These tests run against a live Oracle Autonomous Database (Vector / 23ai+).
They are skipped unless the following environment variables are set:

- ``ADB_DSN``                 e.g. ``deepresearch_low``
- ``ADB_USER``                e.g. ``ADMIN``
- ``ADB_PASSWORD``
- ``ADB_WALLET_LOCATION``     path to the unzipped wallet directory
- ``ADB_WALLET_PASSWORD``     (optional, defaults to ``ADB_PASSWORD``)

Run:

    cd libs/oci
    poetry run pytest tests/integration_tests/datastores/test_adb_review_fixes.py -v
"""

from __future__ import annotations

import os
import uuid
from typing import Iterator

import pytest
from langchain_core.embeddings import Embeddings

from langchain_oci.datastores.vectorstores.adb import ADB

REQUIRED_ENV = ("ADB_DSN", "ADB_USER", "ADB_PASSWORD", "ADB_WALLET_LOCATION")


def _missing_env() -> bool:
    return any(not os.environ.get(k) for k in REQUIRED_ENV)


pytestmark = [
    pytest.mark.requires("oracledb", "langchain_oracledb"),
    pytest.mark.skipif(
        _missing_env(),
        reason=f"ADB env vars not set: {', '.join(REQUIRED_ENV)}",
    ),
]


class _StubEmbeddings(Embeddings):
    """Deterministic embeddings — avoids hitting an LLM provider in CI."""

    DIM = 8

    def embed_query(self, text: str) -> list[float]:
        seed = (len(text) % 7) + 1
        return [float(seed) / (i + 1) for i in range(self.DIM)]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]


def _make_store(table: str, *, chunk_on_write: bool = False) -> ADB:
    return ADB(
        dsn=os.environ["ADB_DSN"],
        user=os.environ["ADB_USER"],
        password=os.environ["ADB_PASSWORD"],
        wallet_location=os.environ["ADB_WALLET_LOCATION"],
        wallet_password=os.environ.get("ADB_WALLET_PASSWORD"),
        table_name=table,
        chunk_on_write=chunk_on_write,
    )


def _drop_table(table: str) -> None:
    """Connect with a fresh handle and drop the test table if it exists."""
    cleanup = _make_store(table)
    try:
        cleanup.connect(_StubEmbeddings())
        cur = cleanup._connection.cursor()
        try:
            cur.execute(f"DROP TABLE {table} PURGE")
        except Exception:
            pass
        finally:
            cur.close()
    finally:
        cleanup.close()


@pytest.fixture
def temp_table() -> Iterator[str]:
    name = f"PR192_T_{uuid.uuid4().hex[:8].upper()}"
    yield name
    _drop_table(name)


@pytest.fixture
def store(temp_table: str) -> Iterator[ADB]:
    s = _make_store(temp_table)
    s.connect(_StubEmbeddings())
    try:
        yield s
    finally:
        s.close()


# ---------------------------------------------------------------------------
# 1. close() / __exit__ release the connection
# ---------------------------------------------------------------------------


def test_close_clears_connection_and_is_idempotent(temp_table: str) -> None:
    s = _make_store(temp_table)
    s.connect(_StubEmbeddings())
    handle = s._connection
    assert handle is not None

    s.close()
    assert s._connection is None
    with pytest.raises(RuntimeError, match="not connected"):
        _ = s.vectorstore

    # Underlying oracledb connection is closed — cursor() must fail.
    with pytest.raises(Exception):
        handle.cursor()

    # Idempotent.
    s.close()


def test_context_manager_closes_connection(temp_table: str) -> None:
    s = _make_store(temp_table)
    s.connect(_StubEmbeddings())
    with s as ctx:
        assert ctx is s
        assert s._connection is not None
    assert s._connection is None


# ---------------------------------------------------------------------------
# 2. bulk_insert is batched (one underlying add_* call for N docs)
# ---------------------------------------------------------------------------


def test_bulk_insert_batches_into_single_add_call(store: ADB) -> None:
    calls: list[dict[str, int]] = []
    original = store._oraclevs.add_texts

    def counting_add_texts(*args, **kwargs):
        ids = kwargs.get("ids")
        texts = kwargs.get("texts")
        if ids is None and len(args) >= 3:
            ids = args[2]
        if texts is None and len(args) >= 1:
            texts = args[0]
        calls.append({"ids": len(ids or []), "texts": len(texts or [])})
        return original(*args, **kwargs)

    store._oraclevs.add_texts = counting_add_texts

    docs = [
        {"title": f"t{i}", "content": f"content {i}", "source": "bulk-test"}
        for i in range(5)
    ]
    n = store.bulk_insert(docs, embeddings=[])
    assert n == 5
    assert len(calls) == 1, f"expected 1 batched call, got {calls}"
    assert calls[0] == {"ids": 5, "texts": 5}


# ---------------------------------------------------------------------------
# 3. update() restores the snapshot on ingestion failure
# ---------------------------------------------------------------------------


def test_update_restores_snapshot_on_ingestion_failure(store: ADB) -> None:
    doc_id = store.insert(
        title="Original",
        content="original content body",
        source="update-test",
        embedding=[],
    )
    original_doc = store.get(doc_id)
    assert original_doc is not None
    assert original_doc["title"] == "Original"

    real_ingest = store._ingest_document
    call_count = {"n": 0}

    def flaky_ingest(document, doc_id):
        call_count["n"] += 1
        # First call is the new-content ingest → fail.
        # Second call is the snapshot restore → succeed.
        if call_count["n"] == 1:
            raise RuntimeError("simulated ingestion failure")
        return real_ingest(document, doc_id)

    store._ingest_document = flaky_ingest  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="simulated ingestion failure"):
        store.update(
            doc_id,
            title="Replaced",
            content="replaced content",
            source=None,
            embedding=None,
        )

    # Restore the real ingest before reading.
    store._ingest_document = real_ingest  # type: ignore[method-assign]

    recovered = store.get(doc_id)
    assert recovered is not None, "original document was lost on failed update"
    assert recovered["title"] == "Original"
    assert "original content" in recovered["content"]


# ---------------------------------------------------------------------------
# 4. metadata.id function-based index is created automatically
# ---------------------------------------------------------------------------


def test_metadata_id_index_is_created(store: ADB, temp_table: str) -> None:
    cur = store._connection.cursor()
    cur.execute(
        "SELECT index_name FROM user_indexes WHERE table_name = :t",
        {"t": temp_table.upper()},
    )
    index_names = [row[0] for row in cur.fetchall()]
    cur.close()
    matching = [n for n in index_names if n.startswith("IDX_") and n.endswith("_MID")]
    assert matching, f"expected metadata.id index on {temp_table}, found {index_names}"

    # Calling again must not raise (ORA-00955 path).
    store._ensure_metadata_id_index()
