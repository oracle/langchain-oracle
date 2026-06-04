# Copyright (c) 2024, 2026, Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""
test_text.py

Integration tests for Oracle Text full-text search with OracleTextSearchRetriever.
Covers:
- correct match and ranking
- fuzzy text search
- input combinations and validation
- both OracleVS vector store-backed tables and regular user tables
- sync and async code paths
"""

import os
import uuid
from typing import Any, Dict, Tuple

import oracledb
import pytest

from langchain_oracledb.embeddings import OracleEmbeddings
from langchain_oracledb.retrievers.text_search import (
    OracleTextSearchRetriever,
    acreate_text_index,
    create_text_index,
)
from langchain_oracledb.vectorstores import DistanceStrategy, OracleVS
from langchain_oracledb.vectorstores.utils import (
    drop_index,
    drop_table_purge,
)

username = os.environ.get("VECDB_USER")
password = os.environ.get("VECDB_PASS")
dsn = os.environ.get("VECDB_HOST")

# Attempt a quick connection to determine whether to skip all tests
try:
    oracledb.connect(user=username, password=password, dsn=dsn)
except Exception as e:
    pytest.skip(
        allow_module_level=True,
        reason=f"Database connection failed: {e}, skipping tests.",
    )


# -------------------------
# Fixtures for setup/teardown and common data
# -------------------------
@pytest.fixture(scope="function")
def resource_names() -> Dict[str, str]:
    """Generate unique resource names to avoid collisions."""
    suffix = uuid.uuid4().hex[:8]
    return {
        # Vector store-backed table and index
        "table_vs": f"TB_FT_VS_{suffix}",
        "index_vs_text": f"IDX_FT_VS_TX_{suffix}",
        "index_vs_meta": f"IDX_FT_VS_MD_{suffix}",
        # Raw user table and index
        "table_raw": f"TB_FT_RAW_{suffix}",
        "index_raw": f"IDX_FT_RAW_{suffix}",
    }


@pytest.fixture(scope="function")
def connection():
    """Sync connection fixture."""
    conn = oracledb.connect(user=username, password=password, dsn=dsn)
    try:
        yield conn
    finally:
        try:
            conn.close()
        except Exception:
            pass


@pytest.fixture(scope="function")
async def aconnection():
    """Async connection fixture."""
    conn = await oracledb.connect_async(user=username, password=password, dsn=dsn)
    try:
        yield conn
    finally:
        try:
            await conn.close()
        except Exception:
            pass


@pytest.fixture(scope="function")
def cleanup(connection, resource_names):
    """
    Ensure tables and indexes are dropped before and after a test.
    Uses sync connection for cleanup (works for async tests too).
    """
    # pre-clean
    for _ in range(2):
        for idx in ("index_vs_text", "index_vs_meta", "index_raw"):
            try:
                drop_index(connection, resource_names[idx])
            except Exception:
                pass
        for tbl in ("table_vs", "table_raw"):
            try:
                drop_table_purge(connection, resource_names[tbl])
            except Exception:
                pass
    yield
    # post-clean
    for _ in range(2):
        for idx in ("index_vs_text", "index_vs_meta", "index_raw"):
            try:
                drop_index(connection, resource_names[idx])
            except Exception:
                pass
        for tbl in ("table_vs", "table_raw"):
            try:
                drop_table_purge(connection, resource_names[tbl])
            except Exception:
                pass


@pytest.fixture(scope="function")
def db_embedder_params() -> Dict[str, Any]:
    # Use the same default DB vectorizer model as hybrid tests
    return {"provider": "database", "model": "allminilm"}


@pytest.fixture(scope="function")
def sample_texts_and_metadatas() -> Tuple[list[str], list[dict]]:
    texts = [
        "If the answer to any preceding questions about database is yes, then say yes",
        (
            "A tablespace can be online (accessible) or offline "
            "(not accessible) whenever the database is open."
        ),
    ]
    metadatas = [
        {
            "id": "cncpt_15.5.3.2.2_P4",
            "link": "https://docs.oracle.com/en/database/oracle/oracle-database/23/cncpt/logical-storage-structures.html#GUID-5387D7B2-C0CA-4C1E-811B-C7EB9B636442",
        },
        {
            "id": "cncpt_15.5.5_P1",
            "link": "https://docs.oracle.com/en/database/oracle/oracle-database/23/cncpt/logical-storage-structures.html#GUID-D02B2220-E6F5-40D9-AFB5-BC69BCEF6CD4",
        },
    ]
    return texts, metadatas


@pytest.fixture(scope="function")
def sample_docstring_texts_and_metadatas() -> Tuple[list[str], list[dict]]:
    texts = [
        "Our refund policy for premium plan allows refunds within 30 days.",
        "Please review the latest SLA describing uptime commitments.",
    ]
    metadatas = [
        {"id": "doc_refund", "customer_id": "CUST_A"},
        {"id": "doc_sla", "customer_id": "CUST_B"},
    ]
    return texts, metadatas


# -------------------------
# Helper for raw user table (non-OracleVS)
# -------------------------
def _create_raw_table_and_data(connection, table_name: str):
    with connection.cursor() as cur:
        # Create a simple table with a body CLOB and a title column
        cur.execute(
            f"BEGIN EXECUTE IMMEDIATE 'DROP TABLE {table_name} PURGE';"
            "EXCEPTION WHEN OTHERS THEN NULL; END;"
        )
        cur.execute(f"CREATE TABLE {table_name} (title VARCHAR2(200), body CLOB)")
        cur.execute(
            f"INSERT INTO {table_name}(title, body) VALUES (:1, :2)",
            [
                "Refund",
                "Our refund policy for premium plan allows refunds within 30 days.",
            ],
        )
        cur.execute(
            f"INSERT INTO {table_name}(title, body) VALUES (:1, :2)",
            ["SLA", "Please review the latest SLA describing uptime commitments."],
        )
        connection.commit()


# -------------------------
# Sync tests - OracleVS-backed table
# -------------------------
def test_text_vs_sync_exact_and_scores_and_returned_columns_default(
    connection, cleanup, resource_names, db_embedder_params, sample_texts_and_metadatas
) -> None:
    """
    - Create OracleVS from texts
    - Create Oracle Text SEARCH INDEX on 'text'
    - Run exact search and validate top document, metadata, and scores
    - Validate default returned_columns behavior (other column auto-included)
    """
    proxy = ""
    model = OracleEmbeddings(conn=connection, params=db_embedder_params, proxy=proxy)

    texts, metadatas = sample_texts_and_metadatas
    drop_table_purge(connection, resource_names["table_vs"])

    vs = OracleVS.from_texts(
        texts,
        model,
        metadatas,
        client=connection,
        table_name=resource_names["table_vs"],
        distance_strategy=DistanceStrategy.COSINE,
    )

    # Create Oracle Text index on text column
    create_text_index(
        connection,
        idx_name=resource_names["index_vs_text"],
        vector_store=vs,
    )

    # Build retriever; returned_columns defaults to ["metadata"] for vs+text
    retriever = OracleTextSearchRetriever(
        vector_store=vs,
        k=1,
        return_scores=True,
    )
    docs = retriever.invoke("tablespace")
    assert len(docs) == 1
    # Ensure the 'tablespace' document ranks first
    assert docs[0].metadata["id"] == "cncpt_15.5.5_P1"
    # Score is present
    assert "score" in docs[0].metadata
    # Ensure user metadata fields are preserved
    assert "link" in docs[0].metadata

    # Override k at call time, expect 1 result
    docs2 = retriever.invoke("database", k=1)
    assert len(docs2) == 1


def test_text_vs_sync_fuzzy_on_text(
    connection, cleanup, resource_names, db_embedder_params, sample_texts_and_metadatas
) -> None:
    """
    - Create index on 'text'
    - Use fuzzy search to match misspelled term
    """
    proxy = ""
    model = OracleEmbeddings(conn=connection, params=db_embedder_params, proxy=proxy)

    texts, metadatas = sample_texts_and_metadatas
    drop_table_purge(connection, resource_names["table_vs"])

    vs = OracleVS.from_texts(
        texts,
        model,
        metadatas,
        client=connection,
        table_name=resource_names["table_vs"],
        distance_strategy=DistanceStrategy.COSINE,
    )

    create_text_index(
        connection,
        idx_name=resource_names["index_vs_text"],
        vector_store=vs,
        column_name="text",
    )

    retriever = OracleTextSearchRetriever(
        vector_store=vs,
        column_name="text",
        fuzzy=True,
        k=1,
        return_scores=True,
    )
    # Misspelled "tablespace"
    docs = retriever.invoke("tabespace")
    assert len(docs) == 1
    assert docs[0].metadata["id"] == "cncpt_15.5.5_P1"
    assert docs[0].metadata["score"] > 0


def test_text_vs_sync_index_on_metadata_not_allowed(
    connection,
    cleanup,
    resource_names,
    db_embedder_params,
    sample_docstring_texts_and_metadatas,
) -> None:
    """
    - Create VS with docstring-like samples
    - Verify that indexing/searching 'metadata' with vector_store is not allowed
    """
    proxy = ""
    model = OracleEmbeddings(conn=connection, params=db_embedder_params, proxy=proxy)

    texts, metadatas = sample_docstring_texts_and_metadatas
    drop_table_purge(connection, resource_names["table_vs"])

    vs = OracleVS.from_texts(
        texts,
        model,
        metadatas,
        client=connection,
        table_name=resource_names["table_vs"],
        distance_strategy=DistanceStrategy.COSINE,
    )

    with pytest.raises(ValueError, match="column_name must be 'text'"):
        create_text_index(
            connection,
            idx_name=resource_names["index_vs_meta"],
            vector_store=vs,
            column_name="metadata",
        )

    with pytest.raises(ValueError, match="column_name must be 'text'"):
        OracleTextSearchRetriever(
            vector_store=vs,
            column_name="metadata",
            k=1,
            return_scores=True,
        )


# -------------------------
# Sync tests - Raw user table (non-OracleVS)
# -------------------------
def test_text_raw_table_sync_exact_and_scores_and_returned_columns(
    connection, cleanup, resource_names
) -> None:
    """
    - Create a user table with (title, body)
    - Create Oracle Text SEARCH INDEX on 'body'
    - Build retriever using client+table_name+column_name
    - Validate ranking, returned_columns and score
    """
    _create_raw_table_and_data(connection, resource_names["table_raw"])

    create_text_index(
        connection,
        idx_name=resource_names["index_raw"],
        table_name=resource_names["table_raw"],
        column_name="body",
    )

    retriever = OracleTextSearchRetriever(
        client=connection,
        table_name=resource_names["table_raw"],
        column_name="body",
        k=1,
        return_scores=True,
        returned_columns=["title"],
    )
    docs = retriever.invoke("refund policy")
    assert len(docs) == 1
    assert "refund policy" in docs[0].page_content.lower()
    assert docs[0].metadata.get("title") == "Refund"
    assert docs[0].metadata.get("score", 0) > 0


def test_text_raw_table_sync_fuzzy_search(connection, cleanup, resource_names) -> None:
    """
    - Create raw table and index
    - Use fuzzy search to match misspelled query
    """
    _create_raw_table_and_data(connection, resource_names["table_raw"])

    create_text_index(
        connection,
        idx_name=resource_names["index_raw"],
        table_name=resource_names["table_raw"],
        column_name="body",
    )

    retriever = OracleTextSearchRetriever(
        client=connection,
        table_name=resource_names["table_raw"],
        column_name="body",
        fuzzy=True,
        k=1,
        return_scores=True,
        returned_columns=["title"],
    )
    docs = retriever.invoke("refnd polciy")
    assert len(docs) == 1
    assert docs[0].metadata["title"] == "Refund"
    assert docs[0].metadata["score"] > 0


def test_text_input_validation_errors_sync(
    connection, cleanup, resource_names, db_embedder_params, sample_texts_and_metadatas
) -> None:
    """
    Validate errors:
    - create_text_index with both vector_store and table_name
    - create_text_index with neither
    - invalid column_name with vector_store
    - retriever init with both vector_store and table_name
    - retriever init with table_name but no client
    """
    proxy = ""
    model = OracleEmbeddings(conn=connection, params=db_embedder_params, proxy=proxy)

    texts, metadatas = sample_texts_and_metadatas
    drop_table_purge(connection, resource_names["table_vs"])

    vs = OracleVS.from_texts(
        texts,
        model,
        metadatas,
        client=connection,
        table_name=resource_names["table_vs"],
        distance_strategy=DistanceStrategy.COSINE,
    )

    # Both vector_store and table_name -> error
    with pytest.raises(ValueError, match="Only give one of vector_store or table_name"):
        create_text_index(
            connection,
            idx_name=resource_names["index_vs_text"],
            vector_store=vs,
            table_name=resource_names["table_raw"],
            column_name="text",
        )

    # Neither vector_store nor table_name -> error
    with pytest.raises(ValueError, match="Provide either vector_store or table_name"):
        create_text_index(connection, idx_name=resource_names["index_vs_text"])

    # Invalid column when vector_store is used
    with pytest.raises(ValueError, match="column_name must be 'text'"):
        create_text_index(
            connection,
            idx_name=resource_names["index_vs_text"],
            vector_store=vs,
            column_name="title",
        )

    # Retriever both vector_store and table_name -> error
    with pytest.raises(ValueError, match="Only give one of vector_store or table_name"):
        OracleTextSearchRetriever(
            vector_store=vs,
            client=connection,
            table_name=resource_names["table_vs"],
            column_name="text",
        )

    # Retriever table_name but no client -> error
    with pytest.raises(ValueError, match="client must be provided"):
        OracleTextSearchRetriever(
            table_name=resource_names["table_raw"],
            column_name="body",
        )


def test_text_returned_columns_dedup_sync(
    connection, cleanup, resource_names, db_embedder_params, sample_texts_and_metadatas
) -> None:
    """
    Ensure returned_columns does not duplicate the main column:
    - For OracleVS/text, passing returned_columns=['text'] should not duplicate content
    - For raw table/body, passing returned_columns including 'body' should not duplicate
    """
    proxy = ""
    model = OracleEmbeddings(conn=connection, params=db_embedder_params, proxy=proxy)
    texts, metadatas = sample_texts_and_metadatas
    drop_table_purge(connection, resource_names["table_vs"])

    vs = OracleVS.from_texts(
        texts,
        model,
        metadatas,
        client=connection,
        table_name=resource_names["table_vs"],
        distance_strategy=DistanceStrategy.COSINE,
    )
    create_text_index(
        connection,
        idx_name=resource_names["index_vs_text"],
        vector_store=vs,
        column_name="text",
    )

    retriever_vs = OracleTextSearchRetriever(
        vector_store=vs,
        column_name="text",
        returned_columns=["text"],  # should be de-duplicated away
        k=1,
    )
    docs_vs = retriever_vs.invoke("tablespace")
    assert len(docs_vs) == 1
    # metadata should not contain a duplicate 'text' key
    assert "text" not in docs_vs[0].metadata

    # Raw table case
    _create_raw_table_and_data(connection, resource_names["table_raw"])
    create_text_index(
        connection,
        idx_name=resource_names["index_raw"],
        table_name=resource_names["table_raw"],
        column_name="body",
    )
    retriever_raw = OracleTextSearchRetriever(
        client=connection,
        table_name=resource_names["table_raw"],
        column_name="body",
        returned_columns=["title", "body"],  # 'body' should be de-duplicated
        k=1,
    )
    docs_raw = retriever_raw.invoke("refund")
    assert len(docs_raw) == 1
    assert "body" not in docs_raw[0].metadata
    assert "title" in docs_raw[0].metadata


# -------------------------
# Async tests - OracleVS-backed table
# -------------------------
@pytest.mark.asyncio
async def test_text_vs_async_exact_and_scores(
    aconnection,
    connection,
    cleanup,
    resource_names,
    db_embedder_params,
    sample_texts_and_metadatas,
) -> None:
    proxy = ""
    model = OracleEmbeddings(conn=connection, params=db_embedder_params, proxy=proxy)

    texts, metadatas = sample_texts_and_metadatas
    drop_table_purge(connection, resource_names["table_vs"])

    vs = await OracleVS.afrom_texts(
        texts,
        model,
        metadatas,
        client=aconnection,
        table_name=resource_names["table_vs"],
        distance_strategy=DistanceStrategy.COSINE,
    )

    await acreate_text_index(
        aconnection,
        idx_name=resource_names["index_vs_text"],
        vector_store=vs,
    )

    retriever = OracleTextSearchRetriever(
        vector_store=vs,
        k=1,
        return_scores=True,
    )
    docs = await retriever.ainvoke("tablespace")
    assert len(docs) == 1
    assert docs[0].metadata["id"] == "cncpt_15.5.5_P1"
    assert docs[0].metadata["score"] > 0


@pytest.mark.asyncio
async def test_text_vs_async_fuzzy_on_text(
    aconnection,
    connection,
    cleanup,
    resource_names,
    db_embedder_params,
    sample_texts_and_metadatas,
) -> None:
    proxy = ""
    model = OracleEmbeddings(conn=connection, params=db_embedder_params, proxy=proxy)

    texts, metadatas = sample_texts_and_metadatas
    drop_table_purge(connection, resource_names["table_vs"])

    vs = await OracleVS.afrom_texts(
        texts,
        model,
        metadatas,
        client=aconnection,
        table_name=resource_names["table_vs"],
        distance_strategy=DistanceStrategy.COSINE,
    )

    await acreate_text_index(
        aconnection,
        idx_name=resource_names["index_vs_text"],
        vector_store=vs,
        column_name="text",
    )

    retriever = OracleTextSearchRetriever(
        vector_store=vs,
        column_name="text",
        fuzzy=True,
        k=1,
        return_scores=True,
    )
    docs = await retriever.ainvoke("tabespace")
    assert len(docs) == 1
    assert docs[0].metadata["id"] == "cncpt_15.5.5_P1"
    assert docs[0].metadata["score"] > 0


# -------------------------
# Async tests - Raw user table
# -------------------------
@pytest.mark.asyncio
async def test_text_raw_table_async_exact_and_scores(
    aconnection, connection, cleanup, resource_names
) -> None:
    _create_raw_table_and_data(connection, resource_names["table_raw"])

    await acreate_text_index(
        aconnection,
        idx_name=resource_names["index_raw"],
        table_name=resource_names["table_raw"],
        column_name="body",
    )

    retriever = OracleTextSearchRetriever(
        client=aconnection,
        table_name=resource_names["table_raw"],
        column_name="body",
        k=1,
        return_scores=True,
        returned_columns=["title"],
    )
    docs = await retriever.ainvoke("SLA")
    assert len(docs) == 1
    assert docs[0].metadata["title"] == "SLA"
    assert docs[0].metadata["score"] > 0


@pytest.mark.asyncio
async def test_text_raw_table_async_fuzzy(
    aconnection, connection, cleanup, resource_names
) -> None:
    _create_raw_table_and_data(connection, resource_names["table_raw"])

    await acreate_text_index(
        aconnection,
        idx_name=resource_names["index_raw"],
        table_name=resource_names["table_raw"],
        column_name="body",
    )

    retriever = OracleTextSearchRetriever(
        client=aconnection,
        table_name=resource_names["table_raw"],
        column_name="body",
        fuzzy=True,
        k=1,
        return_scores=True,
        returned_columns=["title"],
    )
    docs = await retriever.ainvoke("refnd polciy")
    assert len(docs) == 1
    assert docs[0].metadata["title"] == "Refund"
    assert docs[0].metadata["score"] > 0


@pytest.mark.asyncio
async def test_text_input_validation_errors_async(
    aconnection,
    connection,
    cleanup,
    resource_names,
    db_embedder_params,
    sample_texts_and_metadatas,
) -> None:
    """
    Async counterparts for input validation where applicable.
    """
    proxy = ""
    model = OracleEmbeddings(conn=connection, params=db_embedder_params, proxy=proxy)

    texts, metadatas = sample_texts_and_metadatas
    drop_table_purge(connection, resource_names["table_vs"])

    vs = await OracleVS.afrom_texts(
        texts,
        model,
        metadatas,
        client=aconnection,
        table_name=resource_names["table_vs"],
        distance_strategy=DistanceStrategy.COSINE,
    )

    with pytest.raises(ValueError, match="Only give one of vector_store or table_name"):
        await acreate_text_index(
            aconnection,
            idx_name=resource_names["index_vs_text"],
            vector_store=vs,
            table_name=resource_names["table_vs"],
            column_name="text",
        )

    with pytest.raises(ValueError, match="Provide either vector_store or table_name"):
        await acreate_text_index(
            aconnection,
            idx_name=resource_names["index_vs_text"],
        )

    with pytest.raises(ValueError, match="column_name must be 'text'"):
        await acreate_text_index(
            aconnection,
            idx_name=resource_names["index_vs_text"],
            vector_store=vs,
            column_name="badcol",
        )

    # Per-model validator checks for retriever on async path too
    with pytest.raises(ValueError, match="client must be provided"):
        OracleTextSearchRetriever(
            table_name=resource_names["table_raw"],
            column_name="body",
        )

    with pytest.raises(ValueError, match="Only give one of vector_store or table_name"):
        OracleTextSearchRetriever(
            vector_store=vs, client=aconnection, table_name=resource_names["table_vs"]
        )


def _build_vs_with_texts(
    connection, table_name: str, idx_name: str, texts: list[str], metadatas: list[dict]
) -> OracleVS:
    proxy = ""
    model = OracleEmbeddings(
        conn=connection,
        params={"provider": "database", "model": "allminilm"},
        proxy=proxy,
    )
    drop_table_purge(connection, table_name)
    vs = OracleVS.from_texts(
        texts,
        model,
        metadatas,
        client=connection,
        table_name=table_name,
        distance_strategy=DistanceStrategy.COSINE,
    )
    create_text_index(
        connection,
        idx_name=idx_name,
        vector_store=vs,
        column_name="text",
    )
    return vs


def _assert_same_doc_sync(
    retriever: OracleTextSearchRetriever, q1: str, q2: str, expected_id: str
) -> None:
    d1 = retriever.invoke(q1)
    d2 = retriever.invoke(q2)
    assert len(d1) == 1 and len(d2) == 1
    assert d1[0].metadata.get("id") == expected_id
    assert d2[0].metadata.get("id") == expected_id


async def _assert_same_doc_async(
    retriever: OracleTextSearchRetriever, q1: str, q2: str, expected_id: str
) -> None:
    d1 = await retriever.ainvoke(q1)
    d2 = await retriever.ainvoke(q2)
    assert len(d1) == 1 and len(d2) == 1
    assert d1[0].metadata.get("id") == expected_id
    assert d2[0].metadata.get("id") == expected_id


def test_text_vs_sync_literal_and_fuzzy_punctuation_grid(
    connection, cleanup, resource_names
) -> None:
    """
    Validate that punctuation in either the query or the indexed text does not change
    retrieval results.
    Uses an OracleVS-backed table.
    """
    # Dataset A: punctuation in the text, queries with and without punctuation
    texts_a = [
        "The quick, brown fox jumps over the lazy dog!",
        "Completely unrelated sentence.",
    ]
    metas_a = [
        {"id": "doc_fox"},
        {"id": "doc_other"},
    ]
    vs_a = _build_vs_with_texts(
        connection,
        resource_names["table_vs"],
        resource_names["index_vs_text"],
        texts_a,
        metas_a,
    )

    q_plain = "quick brown fox"
    q_punct = "quick, brown fox!!!"

    for fuzzy in [True, False]:
        retr = OracleTextSearchRetriever(
            vector_store=vs_a,
            column_name="text",
            fuzzy=fuzzy,
            k=1,
            return_scores=True,
        )
        _assert_same_doc_sync(retr, q_plain, q_punct, expected_id="doc_fox")

    # Dataset B: no punctuation in the text, queries contain punctuation variants
    texts_b = [
        "Refund policy for premium plan allows refunds within 30 days",
        "Please review the latest SLA describing uptime commitments",
    ]
    metas_b = [
        {"id": "doc_refund"},
        {"id": "doc_sla"},
    ]
    vs_b = _build_vs_with_texts(
        connection,
        resource_names["table_vs"],
        resource_names["index_vs_text"],
        texts_b,
        metas_b,
    )

    q_plain_b = "refund policy for premium"
    q_punct_b = "refund, policy for (premium) ???"

    for fuzzy in [True, False]:
        retr_b = OracleTextSearchRetriever(
            vector_store=vs_b,
            column_name="text",
            fuzzy=fuzzy,
            k=1,
            return_scores=True,
        )
        _assert_same_doc_sync(retr_b, q_plain_b, q_punct_b, expected_id="doc_refund")


@pytest.mark.asyncio
async def test_text_vs_async_literal_and_fuzzy_punctuation_grid(
    aconnection, connection, cleanup, resource_names
) -> None:
    """
    Async counterparts of the punctuation-insensitive literal/fuzzy grid tests.
    """
    # Dataset A: punctuation in the text
    proxy = ""
    model = OracleEmbeddings(
        conn=connection,
        params={"provider": "database", "model": "allminilm"},
        proxy=proxy,
    )
    texts_a = [
        "The quick, brown fox jumps over the lazy dog!",
        "Completely unrelated sentence.",
    ]
    metas_a = [
        {"id": "doc_fox"},
        {"id": "doc_other"},
    ]
    drop_table_purge(connection, resource_names["table_vs"])
    vs_a = await OracleVS.afrom_texts(
        texts_a,
        model,
        metas_a,
        client=aconnection,
        table_name=resource_names["table_vs"],
        distance_strategy=DistanceStrategy.COSINE,
    )
    await acreate_text_index(
        aconnection,
        idx_name=resource_names["index_vs_text"],
        vector_store=vs_a,
        column_name="text",
    )

    q_plain = "quick brown fox"
    q_punct = "quick, brown fox!!!"

    for fuzzy in [True, False]:
        retr = OracleTextSearchRetriever(
            vector_store=vs_a,
            column_name="text",
            fuzzy=fuzzy,
            k=1,
            return_scores=True,
        )
        await _assert_same_doc_async(retr, q_plain, q_punct, expected_id="doc_fox")

    # Dataset B: no punctuation in the text, queries contain punctuation
    texts_b = [
        "Refund policy for premium plan allows refunds within 30 days",
        "Please review the latest SLA describing uptime commitments",
    ]
    metas_b = [
        {"id": "doc_refund"},
        {"id": "doc_sla"},
    ]
    drop_table_purge(connection, resource_names["table_vs"])
    vs_b = await OracleVS.afrom_texts(
        texts_b,
        model,
        metas_b,
        client=aconnection,
        table_name=resource_names["table_vs"],
        distance_strategy=DistanceStrategy.COSINE,
    )
    await acreate_text_index(
        aconnection,
        idx_name=resource_names["index_vs_text"],
        vector_store=vs_b,
        column_name="text",
    )

    q_plain_b = "refund policy for premium"
    q_punct_b = "refund, policy for (premium) ???"

    for fuzzy in [True, False]:
        retr_b = OracleTextSearchRetriever(
            vector_store=vs_b,
            column_name="text",
            fuzzy=fuzzy,
            k=1,
            return_scores=True,
        )
        await _assert_same_doc_async(
            retr_b, q_plain_b, q_punct_b, expected_id="doc_refund"
        )


@pytest.mark.asyncio
async def test_text_vs_async_literal_true_vs_false_operator_semantics(
    aconnection, connection, cleanup, resource_names
) -> None:
    """
    Ensure Oracle Text operators are applied only when operator_search=True.
    Using NEAR(...) should:
      - return a match when operator_search=True
      - return no matches when operator_search=False (treated as literal text)
    """
    proxy = ""
    model = OracleEmbeddings(
        conn=connection,
        params={"provider": "database", "model": "allminilm"},
        proxy=proxy,
    )

    texts = [
        "Refund policy for premium plan allows refunds within 30 days",
        "Completely unrelated sentence",
    ]
    metas = [
        {"id": "doc_refund"},
        {"id": "doc_other"},
    ]

    # Build VS and index
    drop_table_purge(connection, resource_names["table_vs"])
    vs = await OracleVS.afrom_texts(
        texts,
        model,
        metas,
        client=aconnection,
        table_name=resource_names["table_vs"],
        distance_strategy=DistanceStrategy.COSINE,
    )
    await acreate_text_index(
        aconnection,
        idx_name=resource_names["index_vs_text"],
        vector_store=vs,
        column_name="text",
    )

    query = "NEAR((policy, refund), 2, TRUE)"
    retr_true = OracleTextSearchRetriever(
        vector_store=vs,
        column_name="text",
        operator_search=False,
        k=1,
        return_scores=True,
    )
    docs_false = await retr_true.ainvoke(query)
    assert len(docs_false) == 1
    assert docs_false[0].metadata.get("id") == "doc_refund"

    # operator_search=True -> operator semantics applied, expect no doc
    retr_false = OracleTextSearchRetriever(
        vector_store=vs,
        column_name="text",
        operator_search=True,
        k=1,
        return_scores=True,
    )
    docs_true = await retr_false.ainvoke(query)
    assert len(docs_true) == 0


def test_text_vs_sync_literal_true_vs_false_operator_semantics(
    connection, cleanup, resource_names
) -> None:
    """
    Sync counterpart for operator_search True vs False behavior using NEAR(...).
    """
    texts = [
        "Refund policy for premium plan allows refunds within 30 days",
        "Completely unrelated sentence",
    ]
    metas = [
        {"id": "doc_refund"},
        {"id": "doc_other"},
    ]
    vs = _build_vs_with_texts(
        connection,
        resource_names["table_vs"],
        resource_names["index_vs_text"],
        texts,
        metas,
    )

    query = "NEAR((policy, refund), 2, TRUE)"
    retr_true = OracleTextSearchRetriever(
        vector_store=vs,
        column_name="text",
        operator_search=False,
        k=1,
        return_scores=True,
    )
    docs_false = retr_true.invoke(query)
    assert len(docs_false) == 1
    assert docs_false[0].metadata.get("id") == "doc_refund"

    retr_false = OracleTextSearchRetriever(
        vector_store=vs,
        column_name="text",
        operator_search=True,
        k=1,
        return_scores=True,
    )
    docs_true = retr_false.invoke(query)
    assert len(docs_true) == 0


def test_text_vs_fuzzy_word(connection, cleanup, resource_names) -> None:
    """
    Test fuzzy search
    """
    texts = [
        "Refund policy for premium plan allows refunds within 30 days",
        "Completely unrelated sentence",
    ]
    metas = [
        {"id": "doc_refund"},
        {"id": "doc_other"},
    ]
    vs = _build_vs_with_texts(
        connection,
        resource_names["table_vs"],
        resource_names["index_vs_text"],
        texts,
        metas,
    )

    retr_true = OracleTextSearchRetriever(
        vector_store=vs,
        column_name="text",
        operator_search=False,
        k=1,
        return_scores=True,
        fuzzy=True,
    )

    query = "policy premium plan near"
    docs_true = retr_true.invoke(query)
    assert len(docs_true) == 1

    query = ""
    docs_true = retr_true.invoke(query)
    assert len(docs_true) == 0


# -------------------------
# Additional fixtures
# -------------------------


@pytest.fixture(scope="function")
def pool():
    """Connection pool fixture — exercises the pool code path in _get_connection."""
    p = oracledb.create_pool(
        user=username,
        password=password,
        dsn=dsn,
        min=1,
        max=3,
        increment=1,
    )
    try:
        yield p
    finally:
        try:
            p.close()
        except Exception:
            pass


@pytest.fixture(scope="function")
def three_doc_texts_and_metadatas() -> Tuple[list[str], list[dict]]:
    """Three documents with clearly distinct topics for score ordering tests."""
    texts = [
        "The tablespace can be online or offline whenever the database is open.",
        "If the answer to any preceding questions about database is yes, say yes.",
        "Completely unrelated content about cooking recipes and kitchen tools.",
    ]
    metadatas = [
        {"id": "doc_tablespace"},
        {"id": "doc_questions"},
        {"id": "doc_cooking"},
    ]
    return texts, metadatas


# -------------------------
# Additional helpers
# -------------------------


def _build_vs_with_index_3doc(
    connection, resource_names, db_embedder_params, texts, metadatas
):
    """Helper mirroring _build_vs_with_texts but using resource_names['table_vs']
    and resource_names['index_vs'] for the new three-doc tests."""
    proxy = ""
    model = OracleEmbeddings(conn=connection, params=db_embedder_params, proxy=proxy)
    drop_table_purge(connection, resource_names["table_vs"])
    vs = OracleVS.from_texts(
        texts,
        model,
        metadatas,
        client=connection,
        table_name=resource_names["table_vs"],
        distance_strategy=DistanceStrategy.COSINE,
    )
    create_text_index(
        connection, idx_name=resource_names["index_vs_text"], vector_store=vs
    )
    return vs


def _create_raw_table_and_index_3doc(connection, resource_names):
    """Create a 3-row raw table and text index for ordering / pool tests."""
    with connection.cursor() as cur:
        cur.execute(
            f"BEGIN EXECUTE IMMEDIATE 'DROP TABLE {resource_names['table_raw']} PURGE';"
            "EXCEPTION WHEN OTHERS THEN NULL; END;"
        )
        cur.execute(
            f"CREATE TABLE {resource_names['table_raw']} "
            "(title VARCHAR2(200), body CLOB)"
        )
        cur.execute(
            f"INSERT INTO {resource_names['table_raw']}(title, body) VALUES (:1, :2)",
            [
                "Tablespace",
                "The tablespace can be online or offline when the database is open.",
            ],
        )
        cur.execute(
            f"INSERT INTO {resource_names['table_raw']}(title, body) VALUES (:1, :2)",
            [
                "Questions",
                "If the answer to preceding questions about database is yes.",
            ],
        )
        cur.execute(
            f"INSERT INTO {resource_names['table_raw']}(title, body) VALUES (:1, :2)",
            ["Cooking", "Completely unrelated content about cooking recipes."],
        )
        connection.commit()
    create_text_index(
        connection,
        idx_name=resource_names["index_raw"],
        table_name=resource_names["table_raw"],
        column_name="body",
    )


# -------------------------
# operator_search=True with valid Oracle Text expressions
# -------------------------


def test_operator_search_near_returns_correct_doc(
    connection,
    cleanup,
    resource_names,
    db_embedder_params,
    three_doc_texts_and_metadatas,
) -> None:
    """operator_search=True with NEAR(...) must return the document where
    both terms appear close together."""
    texts, metadatas = three_doc_texts_and_metadatas
    vs = _build_vs_with_index_3doc(
        connection, resource_names, db_embedder_params, texts, metadatas
    )

    retriever = OracleTextSearchRetriever(
        vector_store=vs,
        column_name="text",
        operator_search=True,
        k=1,
        return_scores=True,
    )
    docs = retriever.invoke("NEAR((tablespace, database), 15, TRUE)")
    assert len(docs) == 1
    assert docs[0].metadata["id"] == "doc_tablespace"
    assert docs[0].metadata["score"] > 0


def test_operator_search_and_returns_correct_doc(
    connection,
    cleanup,
    resource_names,
    db_embedder_params,
    three_doc_texts_and_metadatas,
) -> None:
    """operator_search=True with AND must return documents containing both terms."""
    texts, metadatas = three_doc_texts_and_metadatas
    vs = _build_vs_with_index_3doc(
        connection, resource_names, db_embedder_params, texts, metadatas
    )

    retriever = OracleTextSearchRetriever(
        vector_store=vs,
        column_name="text",
        operator_search=True,
        k=3,
        return_scores=True,
    )
    docs = retriever.invoke("tablespace AND database")
    assert len(docs) >= 1
    ids = [d.metadata["id"] for d in docs]
    assert "doc_tablespace" in ids


@pytest.mark.asyncio
async def test_operator_search_near_async(
    aconnection,
    connection,
    cleanup,
    resource_names,
    db_embedder_params,
    three_doc_texts_and_metadatas,
) -> None:
    """Async: operator_search=True with NEAR must return the expected document."""
    proxy = ""
    model = OracleEmbeddings(conn=connection, params=db_embedder_params, proxy=proxy)
    texts, metadatas = three_doc_texts_and_metadatas
    drop_table_purge(connection, resource_names["table_vs"])

    vs = await OracleVS.afrom_texts(
        texts,
        model,
        metadatas,
        client=aconnection,
        table_name=resource_names["table_vs"],
        distance_strategy=DistanceStrategy.COSINE,
    )
    await acreate_text_index(
        aconnection, idx_name=resource_names["index_vs_text"], vector_store=vs
    )

    retriever = OracleTextSearchRetriever(
        vector_store=vs,
        column_name="text",
        operator_search=True,
        k=1,
        return_scores=True,
    )
    docs = await retriever.ainvoke("NEAR((tablespace, database), 15, TRUE)")
    assert len(docs) == 1
    assert docs[0].metadata["id"] == "doc_tablespace"


# -------------------------
# Idempotency
# -------------------------


def test_create_text_index_repeatable(
    connection,
    cleanup,
    resource_names,
    db_embedder_params,
    three_doc_texts_and_metadatas,
) -> None:
    """Calling create_text_index twice on the same index must not raise
    and retrieval must still return correct results."""
    texts, metadatas = three_doc_texts_and_metadatas
    vs = _build_vs_with_index_3doc(
        connection, resource_names, db_embedder_params, texts, metadatas
    )

    # Second call — must not raise
    create_text_index(
        connection, idx_name=resource_names["index_vs_text"], vector_store=vs
    )

    retriever = OracleTextSearchRetriever(vector_store=vs, k=1)
    docs = retriever.invoke("tablespace")
    assert len(docs) >= 1


@pytest.mark.asyncio
async def test_acreate_text_index_repeatable(
    aconnection,
    connection,
    cleanup,
    resource_names,
    db_embedder_params,
    three_doc_texts_and_metadatas,
) -> None:
    """Async: calling acreate_text_index twice on the same index must not raise."""
    proxy = ""
    model = OracleEmbeddings(conn=connection, params=db_embedder_params, proxy=proxy)
    texts, metadatas = three_doc_texts_and_metadatas
    drop_table_purge(connection, resource_names["table_vs"])

    vs = await OracleVS.afrom_texts(
        texts,
        model,
        metadatas,
        client=aconnection,
        table_name=resource_names["table_vs"],
        distance_strategy=DistanceStrategy.COSINE,
    )
    await acreate_text_index(
        aconnection, idx_name=resource_names["index_vs_text"], vector_store=vs
    )
    # Second call — must not raise
    await acreate_text_index(
        aconnection, idx_name=resource_names["index_vs_text"], vector_store=vs
    )

    retriever = OracleTextSearchRetriever(vector_store=vs, k=1)
    docs = await retriever.ainvoke("tablespace")
    assert len(docs) >= 1


# -------------------------
# acreate_text_index raw table async + returned_columns
# -------------------------


@pytest.mark.asyncio
async def test_acreate_text_index_raw_table_with_returned_columns(
    aconnection, connection, cleanup, resource_names
) -> None:
    """acreate_text_index with table_name (async) and retriever with
    returned_columns must return the extra column in metadata."""
    with connection.cursor() as cur:
        cur.execute(
            f"BEGIN EXECUTE IMMEDIATE 'DROP TABLE {resource_names['table_raw']} PURGE';"
            "EXCEPTION WHEN OTHERS THEN NULL; END;"
        )
        cur.execute(
            f"CREATE TABLE {resource_names['table_raw']} "
            "(title VARCHAR2(200), body CLOB)"
        )
        cur.execute(
            f"INSERT INTO {resource_names['table_raw']}(title, body) VALUES (:1, :2)",
            ["Refund", "Our refund policy allows refunds within 30 days."],
        )
        cur.execute(
            f"INSERT INTO {resource_names['table_raw']}(title, body) VALUES (:1, :2)",
            ["SLA", "Please review the SLA describing uptime commitments."],
        )
        connection.commit()

    await acreate_text_index(
        aconnection,
        idx_name=resource_names["index_raw"],
        table_name=resource_names["table_raw"],
        column_name="body",
    )

    retriever = OracleTextSearchRetriever(
        client=aconnection,
        table_name=resource_names["table_raw"],
        column_name="body",
        k=1,
        return_scores=True,
        returned_columns=["title"],
    )
    docs = await retriever.ainvoke("refund policy")
    assert len(docs) == 1
    assert docs[0].metadata.get("title") == "Refund"
    assert docs[0].metadata.get("score", 0) > 0


# -------------------------
# k larger than document count
# -------------------------


def test_text_k_larger_than_doc_count(
    connection,
    cleanup,
    resource_names,
    db_embedder_params,
    three_doc_texts_and_metadatas,
) -> None:
    """Requesting more results than documents must return all available
    documents without raising."""
    texts, metadatas = three_doc_texts_and_metadatas
    vs = _build_vs_with_index_3doc(
        connection, resource_names, db_embedder_params, texts, metadatas
    )

    retriever = OracleTextSearchRetriever(vector_store=vs, k=100)
    docs = retriever.invoke("database tablespace cooking")
    assert 1 <= len(docs) <= len(texts)


@pytest.mark.asyncio
async def test_text_k_larger_than_doc_count_async(
    aconnection,
    connection,
    cleanup,
    resource_names,
    db_embedder_params,
    three_doc_texts_and_metadatas,
) -> None:
    """Async: k larger than doc count returns all available without raising."""
    proxy = ""
    model = OracleEmbeddings(conn=connection, params=db_embedder_params, proxy=proxy)
    texts, metadatas = three_doc_texts_and_metadatas
    drop_table_purge(connection, resource_names["table_vs"])

    vs = await OracleVS.afrom_texts(
        texts,
        model,
        metadatas,
        client=aconnection,
        table_name=resource_names["table_vs"],
        distance_strategy=DistanceStrategy.COSINE,
    )
    await acreate_text_index(
        aconnection, idx_name=resource_names["index_vs_text"], vector_store=vs
    )

    retriever = OracleTextSearchRetriever(vector_store=vs, k=100)
    docs = await retriever.ainvoke("database tablespace cooking")
    assert 1 <= len(docs) <= len(texts)


# -------------------------
# k override at call time
# -------------------------


def test_text_retriever_k_override_at_invoke(
    connection,
    cleanup,
    resource_names,
    db_embedder_params,
    three_doc_texts_and_metadatas,
) -> None:
    """k passed at retriever.invoke() call time must override the constructor k."""
    texts, metadatas = three_doc_texts_and_metadatas
    vs = _build_vs_with_index_3doc(
        connection, resource_names, db_embedder_params, texts, metadatas
    )

    retriever = OracleTextSearchRetriever(vector_store=vs, k=1)
    docs = retriever.invoke("database", k=3)
    assert len(docs) >= 2


# -------------------------
# ConnectionPool as client
# -------------------------


def test_create_text_index_with_pool(
    pool,
    connection,
    cleanup,
    resource_names,
    db_embedder_params,
    three_doc_texts_and_metadatas,
) -> None:
    """create_text_index must work when client is a ConnectionPool."""
    texts, metadatas = three_doc_texts_and_metadatas
    proxy = ""
    model = OracleEmbeddings(conn=connection, params=db_embedder_params, proxy=proxy)
    drop_table_purge(connection, resource_names["table_vs"])

    vs = OracleVS.from_texts(
        texts,
        model,
        metadatas,
        client=connection,
        table_name=resource_names["table_vs"],
        distance_strategy=DistanceStrategy.COSINE,
    )
    create_text_index(pool, idx_name=resource_names["index_vs_text"], vector_store=vs)

    retriever = OracleTextSearchRetriever(vector_store=vs, k=1)
    docs = retriever.invoke("tablespace")
    assert len(docs) >= 1


def test_text_retrieval_with_pool_in_vs(
    pool,
    connection,
    cleanup,
    resource_names,
    db_embedder_params,
    three_doc_texts_and_metadatas,
) -> None:
    """OracleTextSearchRetriever must work when OracleVS.client is a pool."""
    texts, metadatas = three_doc_texts_and_metadatas
    proxy = ""
    model = OracleEmbeddings(conn=connection, params=db_embedder_params, proxy=proxy)
    drop_table_purge(connection, resource_names["table_vs"])

    vs_pool = OracleVS.from_texts(
        texts,
        model,
        metadatas,
        client=pool,
        table_name=resource_names["table_vs"],
        distance_strategy=DistanceStrategy.COSINE,
    )
    create_text_index(
        pool, idx_name=resource_names["index_vs_text"], vector_store=vs_pool
    )

    retriever = OracleTextSearchRetriever(vector_store=vs_pool, k=2, return_scores=True)
    docs = retriever.invoke("database tablespace")
    assert len(docs) >= 1
    assert "score" in docs[0].metadata


def test_text_retrieval_raw_table_with_pool(
    pool, connection, cleanup, resource_names
) -> None:
    """OracleTextSearchRetriever with a raw table must work when client is a pool."""
    _create_raw_table_and_index_3doc(connection, resource_names)

    retriever = OracleTextSearchRetriever(
        client=pool,
        table_name=resource_names["table_raw"],
        column_name="body",
        k=1,
        return_scores=True,
        returned_columns=["title"],
    )
    docs = retriever.invoke("tablespace")
    assert len(docs) >= 1
    assert "score" in docs[0].metadata


# -------------------------
# Score descending ordering
# -------------------------


def test_text_scores_descending_vs(
    connection,
    cleanup,
    resource_names,
    db_embedder_params,
    three_doc_texts_and_metadatas,
) -> None:
    """Scores returned by OracleVS-backed text retrieval must be descending."""
    texts, metadatas = three_doc_texts_and_metadatas
    vs = _build_vs_with_index_3doc(
        connection, resource_names, db_embedder_params, texts, metadatas
    )

    retriever = OracleTextSearchRetriever(vector_store=vs, k=3, return_scores=True)
    docs = retriever.invoke("database tablespace questions")
    scores = [d.metadata["score"] for d in docs]
    assert scores == sorted(scores, reverse=True), (
        f"Text scores (VS) not in descending order: {scores}"
    )


def test_text_scores_descending_raw_table(connection, cleanup, resource_names) -> None:
    """Scores returned by raw-table text retrieval must be descending."""
    _create_raw_table_and_index_3doc(connection, resource_names)

    retriever = OracleTextSearchRetriever(
        client=connection,
        table_name=resource_names["table_raw"],
        column_name="body",
        k=3,
        return_scores=True,
        returned_columns=["title"],
    )
    docs = retriever.invoke("database tablespace questions")
    scores = [d.metadata["score"] for d in docs]
    assert scores == sorted(scores, reverse=True), (
        f"Text scores (raw table) not in descending order: {scores}"
    )


@pytest.mark.asyncio
async def test_text_scores_descending_async(
    aconnection,
    connection,
    cleanup,
    resource_names,
    db_embedder_params,
    three_doc_texts_and_metadatas,
) -> None:
    """Async: text scores must be in descending order."""
    proxy = ""
    model = OracleEmbeddings(conn=connection, params=db_embedder_params, proxy=proxy)
    texts, metadatas = three_doc_texts_and_metadatas
    drop_table_purge(connection, resource_names["table_vs"])

    vs = await OracleVS.afrom_texts(
        texts,
        model,
        metadatas,
        client=aconnection,
        table_name=resource_names["table_vs"],
        distance_strategy=DistanceStrategy.COSINE,
    )
    await acreate_text_index(
        aconnection, idx_name=resource_names["index_vs_text"], vector_store=vs
    )

    retriever = OracleTextSearchRetriever(vector_store=vs, k=3, return_scores=True)
    docs = await retriever.ainvoke("database tablespace questions")
    scores = [d.metadata["score"] for d in docs]
    assert scores == sorted(scores, reverse=True), (
        f"Async text scores not in descending order: {scores}"
    )
