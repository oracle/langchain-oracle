# Copyright (c) 2024, 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""
test_hybrid.py

Test Oracle AI Vector Search hybrid search
with OracleVS.
"""

# import required modules
import uuid
from typing import Any, Dict, Tuple

import oracledb
import pytest
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy

from langchain_oracledb.embeddings import OracleEmbeddings
from langchain_oracledb.retrievers.hybrid_search import (
    OracleVectorizerPreference,
    OracleVSHybridSearchRetriever,
    acreate_hybrid_index,
    create_hybrid_index,
    drop_preference,
)
from langchain_oracledb.vectorstores import OracleVS
from langchain_oracledb.vectorstores.utils import (
    drop_index,
    drop_table_purge,
)

# Connection details for tests
username = ""
password = ""
dsn = ""

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
    """Generate unique resource names per test to avoid collisions."""
    suffix = uuid.uuid4().hex[:8]
    return {
        "table": f"TB_HY_{suffix}",
        "index": f"IDX_HY_{suffix}",
        "pref": f"PREF_HY_{suffix}",
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
    Ensure tables, indexes and preferences are dropped before and after a test.
    Uses sync connection for cleanup for simplicity (works for async tests too).
    """
    # pre-clean
    for _ in range(2):
        try:
            drop_index(connection, resource_names["index"])
        except Exception:
            pass
        try:
            drop_preference(connection, resource_names["pref"])
        except Exception:
            pass
        try:
            drop_table_purge(connection, resource_names["table"])
        except Exception:
            pass
    yield
    # post-clean
    for _ in range(2):
        try:
            drop_index(connection, resource_names["index"])
        except Exception:
            pass
        try:
            drop_preference(connection, resource_names["pref"])
        except Exception:
            pass
        try:
            drop_table_purge(connection, resource_names["table"])
        except Exception:
            pass


@pytest.fixture(scope="function")
def db_embedder_params() -> Dict[str, Any]:
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
# Sync tests
# -------------------------
def test_preference(connection, cleanup, resource_names, db_embedder_params) -> None:
    # Set up OracleEmbeddings with DB vectorizer
    proxy = ""
    model = OracleEmbeddings(conn=connection, params=db_embedder_params, proxy=proxy)

    # Create a VS (no need to preload data for preference creation)
    vs_obj = OracleVS(
        connection, model, resource_names["table"], DistanceStrategy.EUCLIDEAN_DISTANCE
    )

    # Create a preference with explicit name
    preference = OracleVectorizerPreference.create_preference(
        vs_obj, resource_names["pref"]
    )
    assert preference.preference_name == resource_names["pref"]

    # Mismatch: wrong model value should raise
    with pytest.raises(ValueError, match="Mismatch"):
        OracleVectorizerPreference.create_preference(
            vs_obj,
            preference_name=f"{resource_names['pref']}_bad",
            params={"model": "non_existing_model"},
        )

    # Mismatch: wrong embedder_spec should raise
    with pytest.raises(ValueError, match="Mismatch"):
        params = {
            "embedder_spec": {
                "provider": "oracleai",
                "url": "http://myhost.us.example.com:9091/omlmodels/all_mini_l12/score",
                "host": "local",
                "model": "all_minilm_l12",
            }
        }
        OracleVectorizerPreference.create_preference(
            vs_obj, preference_name=f"{resource_names['pref']}_bad2", params=params
        )

    # Non-OracleEmbeddings should fail in preference creation
    model1 = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-mpnet-base-v2"
    )
    vs = OracleVS(
        connection, model1, resource_names["table"], DistanceStrategy.EUCLIDEAN_DISTANCE
    )
    with pytest.raises(ValueError, match="Only OracleEmbeddings"):
        OracleVectorizerPreference.create_preference(vs)


def test_create_hybrid_index(
    connection, cleanup, resource_names, db_embedder_params, sample_texts_and_metadatas
) -> None:
    proxy = ""
    model = OracleEmbeddings(conn=connection, params=db_embedder_params, proxy=proxy)

    texts, metadatas = sample_texts_and_metadatas

    # Ensure a fresh table
    drop_table_purge(connection, resource_names["table"])

    vs = OracleVS.from_texts(
        texts,
        model,
        metadatas,
        client=connection,
        table_name=resource_names["table"],
        distance_strategy=DistanceStrategy.COSINE,
    )

    # Create preference
    preference = OracleVectorizerPreference.create_preference(
        vs, resource_names["pref"]
    )

    # Passing vectorization params via 'parameters' should raise
    with pytest.raises(ValueError, match="Vectorization parameters must be given"):
        create_hybrid_index(
            connection,
            resource_names["index"],
            preference,
            params={"parameters": {"model": "allminilm"}},
        )

    # Create hybrid index successfully
    create_hybrid_index(connection, resource_names["index"], preference)

    # Drop preference and index via fixture cleanup


def test_hybrid_retrieval(
    connection, cleanup, resource_names, db_embedder_params, sample_texts_and_metadatas
) -> None:
    proxy = ""
    model = OracleEmbeddings(conn=connection, params=db_embedder_params, proxy=proxy)

    texts, metadatas = sample_texts_and_metadatas

    drop_table_purge(connection, resource_names["table"])

    vs = OracleVS.from_texts(
        texts,
        model,
        metadatas,
        client=connection,
        table_name=resource_names["table"],
        distance_strategy=DistanceStrategy.COSINE,
    )

    preference = OracleVectorizerPreference.create_preference(
        vs, resource_names["pref"]
    )
    create_hybrid_index(connection, resource_names["index"], preference)

    retriever = OracleVSHybridSearchRetriever(
        vector_store=vs, idx_name=resource_names["index"], k=2
    )

    assert retriever.search_mode == "hybrid"

    query = "database questions"
    documents = retriever.invoke(query)
    assert len(documents) == 2
    assert documents[0].metadata["id"] == "cncpt_15.5.3.2.2_P4"
    assert "score" not in documents[0].metadata

    retriever = OracleVSHybridSearchRetriever(
        vector_store=vs,
        idx_name=resource_names["index"],
        k=1,
        search_mode="semantic",
        return_scores=True,
    )
    query = "tablespace"
    documents = retriever.invoke(query)
    assert len(documents) == 1
    assert documents[0].metadata["id"] == "cncpt_15.5.5_P1"
    assert documents[0].metadata["vector_score"] > 0
    assert documents[0].metadata["text_score"] == 0

    retriever = OracleVSHybridSearchRetriever(
        vector_store=vs,
        idx_name=resource_names["index"],
        k=1,
        search_mode="keyword",
        return_scores=True,
    )
    query = "preceding questions"
    documents = retriever.invoke(query)
    assert len(documents) == 1
    assert documents[0].metadata["id"] == "cncpt_15.5.3.2.2_P4"
    assert documents[0].metadata["text_score"] > 0
    assert documents[0].metadata["vector_score"] == 0


# -------------------------
# Async tests
# -------------------------
@pytest.mark.asyncio
async def test_preference_async(
    aconnection, connection, cleanup, resource_names, db_embedder_params
) -> None:
    # OracleEmbeddings prefers a sync connection for DB vectorizer params
    proxy = ""
    model = OracleEmbeddings(conn=connection, params=db_embedder_params, proxy=proxy)

    vs_obj = await OracleVS.acreate(
        aconnection, model, resource_names["table"], DistanceStrategy.EUCLIDEAN_DISTANCE
    )

    # Create preference async
    preference = await OracleVectorizerPreference.acreate_preference(
        vs_obj, resource_names["pref"]
    )
    assert preference.preference_name == resource_names["pref"]

    # Mismatch model should raise
    with pytest.raises(ValueError, match="Mismatch"):
        await OracleVectorizerPreference.acreate_preference(
            vs_obj,
            preference_name=f"{resource_names['pref']}_bad",
            params={"model": "non_existing_model"},
        )

    # Non-OracleEmbeddings should fail
    model1 = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-mpnet-base-v2"
    )
    vs = await OracleVS.acreate(
        aconnection,
        model1,
        resource_names["table"],
        DistanceStrategy.EUCLIDEAN_DISTANCE,
    )
    with pytest.raises(ValueError, match="Only OracleEmbeddings"):
        await OracleVectorizerPreference.acreate_preference(vs)


@pytest.mark.asyncio
async def test_create_hybrid_index_async(
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

    drop_table_purge(connection, resource_names["table"])

    vs = await OracleVS.afrom_texts(
        texts,
        model,
        metadatas,
        client=aconnection,
        table_name=resource_names["table"],
        distance_strategy=DistanceStrategy.COSINE,
    )

    preference = await OracleVectorizerPreference.acreate_preference(
        vs, resource_names["pref"]
    )

    # Passing vectorization params via 'parameters' should raise
    with pytest.raises(ValueError, match="Vectorization parameters must be given"):
        await acreate_hybrid_index(
            aconnection,
            resource_names["index"],
            preference,
            params={"parameters": {"model": "allminilm"}},
        )

    # Create hybrid index successfully (async)
    await acreate_hybrid_index(aconnection, resource_names["index"], preference)


@pytest.mark.asyncio
async def test_hybrid_retrieval_async(
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

    drop_table_purge(connection, resource_names["table"])

    vs = await OracleVS.afrom_texts(
        texts,
        model,
        metadatas,
        client=aconnection,
        table_name=resource_names["table"],
        distance_strategy=DistanceStrategy.COSINE,
    )

    preference = await OracleVectorizerPreference.acreate_preference(
        vs, resource_names["pref"]
    )
    await acreate_hybrid_index(aconnection, resource_names["index"], preference)

    retriever = OracleVSHybridSearchRetriever(
        vector_store=vs, idx_name=resource_names["index"], k=2
    )

    assert retriever.search_mode == "hybrid"

    query = "database questions"
    documents = await retriever.ainvoke(query)
    assert len(documents) == 2
    assert documents[0].metadata["id"] == "cncpt_15.5.3.2.2_P4"
    assert "score" not in documents[0].metadata

    retriever = OracleVSHybridSearchRetriever(
        vector_store=vs,
        idx_name=resource_names["index"],
        k=1,
        search_mode="semantic",
        return_scores=True,
    )
    query = "tablespace"
    documents = await retriever.ainvoke(query)
    assert len(documents) == 1
    assert documents[0].metadata["id"] == "cncpt_15.5.5_P1"
    assert documents[0].metadata["vector_score"] > 0
    assert documents[0].metadata["text_score"] == 0

    retriever = OracleVSHybridSearchRetriever(
        vector_store=vs,
        idx_name=resource_names["index"],
        k=1,
        search_mode="keyword",
        return_scores=True,
    )
    query = "preceding questions"
    documents = await retriever.ainvoke(query)
    assert len(documents) == 1
    assert documents[0].metadata["id"] == "cncpt_15.5.3.2.2_P4"
    assert documents[0].metadata["text_score"] > 0
    assert documents[0].metadata["vector_score"] == 0


# -------------------------
# Additional tests derived from hybrid_search.py docstrings
# and validation/error and extended search params coverage
# -------------------------


def test_docstring_example_sync(
    connection,
    cleanup,
    resource_names,
    db_embedder_params,
    sample_docstring_texts_and_metadatas,
) -> None:
    """
    Mirrors the synchronous docstring example:
    - Create preference
    - Create hybrid index
    - Build retriever with params including 'return'
    - Run a query
    """
    proxy = ""
    model = OracleEmbeddings(conn=connection, params=db_embedder_params, proxy=proxy)

    texts, metadatas = sample_docstring_texts_and_metadatas
    drop_table_purge(connection, resource_names["table"])

    vs = OracleVS.from_texts(
        texts,
        model,
        metadatas,
        client=connection,
        table_name=resource_names["table"],
        distance_strategy=DistanceStrategy.COSINE,
    )

    pref = OracleVectorizerPreference.create_preference(vs, resource_names["pref"])

    # From docstring: provide additional index parameters (matching example)
    create_hybrid_index(
        connection,
        resource_names["index"],
        pref,
        params={
            "parallel": 4,
        },
    )

    # From docstring: include 'return' values in retriever params
    # (allowed; retriever sets defaults internally)
    retriever = OracleVSHybridSearchRetriever(
        vector_store=vs,
        idx_name=resource_names["index"],
        search_mode="hybrid",
        k=2,
        return_scores=True,
    )
    docs = retriever.invoke("refund policy for premium plan")
    assert len(docs) >= 1
    # Ensure the refund document ranks for this query
    assert docs[0].metadata["id"] == "doc_refund"
    # Ensure expected score fields exist
    assert (
        "score" in docs[0].metadata
        and "text_score" in docs[0].metadata
        and "vector_score" in docs[0].metadata
    )


def test_retriever_params_validation_errors(
    connection, cleanup, resource_names, db_embedder_params, sample_texts_and_metadatas
) -> None:
    """
    Validate error raises for invalid search params:
    - Disallow 'search_text' at top-level
    - Disallow 'search_text'/'search_vector'/'contains' inside 'vector'/'text'
    - Disallow bad params provided at call time as well
    """
    proxy = ""
    model = OracleEmbeddings(conn=connection, params=db_embedder_params, proxy=proxy)
    texts, metadatas = sample_texts_and_metadatas

    drop_table_purge(connection, resource_names["table"])
    vs = OracleVS.from_texts(
        texts,
        model,
        metadatas,
        client=connection,
        table_name=resource_names["table"],
        distance_strategy=DistanceStrategy.COSINE,
    )
    pref = OracleVectorizerPreference.create_preference(vs, resource_names["pref"])
    create_hybrid_index(connection, resource_names["index"], pref)

    # Top-level search_text should raise
    with pytest.raises(
        ValueError, match="Cannot provide search_text as a parameter at the top level"
    ):
        OracleVSHybridSearchRetriever(
            vector_store=vs,
            idx_name=resource_names["index"],
            params={"search_text": "bad"},
        )

    # Nested vector.search_text and text.search_text should raise
    with pytest.raises(
        ValueError,
        match=r"Cannot provide search_text as a parameter in params\['vector'\]",
    ):
        OracleVSHybridSearchRetriever(
            vector_store=vs,
            idx_name=resource_names["index"],
            params={"vector": {"search_text": "bad"}},
        )

    with pytest.raises(
        ValueError,
        match=r"Cannot provide search_text as a parameter in params\['text'\]",
    ):
        OracleVSHybridSearchRetriever(
            vector_store=vs,
            idx_name=resource_names["index"],
            params={"text": {"search_text": "bad"}},
        )

    # 'contains' under text should also raise (message mentions search_text)
    with pytest.raises(
        ValueError,
        match=r"Cannot provide search_text as a parameter in params\['text'\]",
    ):
        OracleVSHybridSearchRetriever(
            vector_store=vs,
            idx_name=resource_names["index"],
            params={"text": {"contains": "bad"}},
        )

    # Per-call invalid params should raise too
    retriever = OracleVSHybridSearchRetriever(
        vector_store=vs, idx_name=resource_names["index"]
    )
    with pytest.raises(
        ValueError,
        match=r"Cannot provide search_text as a parameter in params\['vector'\]",
    ):
        retriever.invoke("ok", params={"vector": {"search_text": "bad"}})
    with pytest.raises(
        ValueError, match="Cannot provide search_text as a parameter at the top level"
    ):
        retriever.invoke("ok", params={"search_text": "bad"})
    with pytest.raises(
        ValueError,
        match=r"Cannot provide search_text as a parameter in params\['text'\]",
    ):
        retriever.invoke("ok", params={"text": {"contains": "x"}})


def test_create_hybrid_index_invalid_parallel_and_reserved_params(
    connection, cleanup, resource_names, db_embedder_params, sample_texts_and_metadatas
) -> None:
    """
    Validate create_hybrid_index raises for:
    - Non-int 'parallel'
    - Reserved params for 'parameters' (vectorizer/embedder_spec/model/vector_idxtype)
    """
    proxy = ""
    model = OracleEmbeddings(conn=connection, params=db_embedder_params, proxy=proxy)
    texts, metadatas = sample_texts_and_metadatas

    drop_table_purge(connection, resource_names["table"])
    vs = OracleVS.from_texts(
        texts,
        model,
        metadatas,
        client=connection,
        table_name=resource_names["table"],
        distance_strategy=DistanceStrategy.COSINE,
    )
    pref = OracleVectorizerPreference.create_preference(vs, resource_names["pref"])

    with pytest.raises(ValueError, match="parallel must be int"):
        create_hybrid_index(
            connection,
            resource_names["index"],
            pref,
            params={"parallel": "4"},
        )

    with pytest.raises(ValueError, match="Vectorization parameters must be given"):
        create_hybrid_index(
            connection,
            resource_names["index"],
            pref,
            params={"parameters": {"vectorizer": "SOME_PREF"}},
        )

    with pytest.raises(ValueError, match="Vectorization parameters must be given"):
        create_hybrid_index(
            connection,
            resource_names["index"],
            pref,
            params={"parameters": {"embedder_spec": "{}"}},
        )


def test_hybrid_score_weight_effects(
    connection, cleanup, resource_names, db_embedder_params, sample_texts_and_metadatas
) -> None:
    """
    Validate 'score_weight' inside 'vector'/'text' influences overall score.
    """
    proxy = ""
    model = OracleEmbeddings(conn=connection, params=db_embedder_params, proxy=proxy)
    texts, metadatas = sample_texts_and_metadatas

    drop_table_purge(connection, resource_names["table"])
    vs = OracleVS.from_texts(
        texts,
        model,
        metadatas,
        client=connection,
        table_name=resource_names["table"],
        distance_strategy=DistanceStrategy.COSINE,
    )
    pref = OracleVectorizerPreference.create_preference(vs, resource_names["pref"])
    create_hybrid_index(connection, resource_names["index"], pref)

    retriever = OracleVSHybridSearchRetriever(
        vector_store=vs,
        idx_name=resource_names["index"],
        k=1,
        search_mode="hybrid",
        return_scores=True,
    )

    query = "preceding questions"
    docs_vec0 = retriever.invoke(query, params={"vector": {"score_weight": 1}})
    assert len(docs_vec0) == 1
    md = docs_vec0[0].metadata
    score1 = md["score"]

    docs_txt0 = retriever.invoke(query)
    assert len(docs_txt0) == 1
    md = docs_txt0[0].metadata
    score2 = md["score"]

    assert score1 != score2


@pytest.mark.asyncio
async def test_docstring_example_async(
    aconnection,
    connection,
    cleanup,
    resource_names,
    db_embedder_params,
    sample_docstring_texts_and_metadatas,
) -> None:
    """
    Mirrors the async docstring example:
    - Async preference creation
    - Async hybrid index creation
    - Async retrieval
    """
    proxy = ""
    model = OracleEmbeddings(conn=connection, params=db_embedder_params, proxy=proxy)

    texts, metadatas = sample_docstring_texts_and_metadatas
    drop_table_purge(connection, resource_names["table"])

    vs = await OracleVS.afrom_texts(
        texts,
        model,
        metadatas,
        client=aconnection,
        table_name=resource_names["table"],
        distance_strategy=DistanceStrategy.COSINE,
    )

    pref = await OracleVectorizerPreference.acreate_preference(
        vs, resource_names["pref"]
    )
    await acreate_hybrid_index(
        aconnection,
        resource_names["index"],
        pref,
        params={
            "parallel": 4,
        },
    )

    retriever = OracleVSHybridSearchRetriever(
        vector_store=vs, idx_name=resource_names["index"], k=2, return_scores=True
    )
    results = await retriever.ainvoke("latest SLA")
    assert len(results) >= 1
    # Ensure the SLA document ranks for this query
    assert results[0].metadata["id"] == "doc_sla"
    assert (
        "score" in results[0].metadata
        and "vector_score" in results[0].metadata
        and "text_score" in results[0].metadata
    )


@pytest.mark.asyncio
async def test_retriever_params_validation_errors_async(
    aconnection,
    connection,
    cleanup,
    resource_names,
    db_embedder_params,
    sample_texts_and_metadatas,
) -> None:
    """
    Async counterpart for params validation;
        instantiation/checks should raise similarly.
    """
    proxy = ""
    model = OracleEmbeddings(conn=connection, params=db_embedder_params, proxy=proxy)

    texts, metadatas = sample_texts_and_metadatas
    drop_table_purge(connection, resource_names["table"])

    vs = await OracleVS.afrom_texts(
        texts,
        model,
        metadatas,
        client=aconnection,
        table_name=resource_names["table"],
        distance_strategy=DistanceStrategy.COSINE,
    )
    pref = await OracleVectorizerPreference.acreate_preference(
        vs, resource_names["pref"]
    )
    await acreate_hybrid_index(aconnection, resource_names["index"], pref)

    with pytest.raises(
        ValueError, match="Cannot provide search_text as a parameter at the top level"
    ):
        OracleVSHybridSearchRetriever(
            vector_store=vs,
            idx_name=resource_names["index"],
            params={"search_text": "bad"},
        )
    with pytest.raises(
        ValueError,
        match=r"Cannot provide search_text as a parameter in params\['vector'\]",
    ):
        OracleVSHybridSearchRetriever(
            vector_store=vs,
            idx_name=resource_names["index"],
            params={"vector": {"search_text": "bad"}},
        )
    with pytest.raises(
        ValueError,
        match=r"Cannot provide search_text as a parameter in params\['text'\]",
    ):
        OracleVSHybridSearchRetriever(
            vector_store=vs,
            idx_name=resource_names["index"],
            params={"text": {"contains": "x"}},
        )

    retriever = OracleVSHybridSearchRetriever(
        vector_store=vs, idx_name=resource_names["index"]
    )
    with pytest.raises(
        ValueError,
        match=r"Cannot provide search_text as a parameter in params\['text'\]",
    ):
        await retriever.ainvoke("ok", params={"text": {"search_text": "bad"}})


@pytest.mark.asyncio
async def test_hybrid_score_weight_effects_async(
    aconnection,
    connection,
    cleanup,
    resource_names,
    db_embedder_params,
    sample_texts_and_metadatas,
) -> None:
    """
    Async validation for score_weight effects on overall score.
    """
    proxy = ""
    model = OracleEmbeddings(conn=connection, params=db_embedder_params, proxy=proxy)

    texts, metadatas = sample_texts_and_metadatas
    drop_table_purge(connection, resource_names["table"])

    vs = await OracleVS.afrom_texts(
        texts,
        model,
        metadatas,
        client=aconnection,
        table_name=resource_names["table"],
        distance_strategy=DistanceStrategy.COSINE,
    )

    pref = await OracleVectorizerPreference.acreate_preference(
        vs, resource_names["pref"]
    )
    await acreate_hybrid_index(aconnection, resource_names["index"], pref)

    retriever = OracleVSHybridSearchRetriever(
        vector_store=vs,
        idx_name=resource_names["index"],
        k=1,
        search_mode="hybrid",
        return_scores=True,
    )

    query = "preceding questions"
    docs_vec0 = await retriever.ainvoke(query, params={"vector": {"score_weight": 1}})
    assert len(docs_vec0) == 1
    md = docs_vec0[0].metadata
    score1 = md["score"]

    docs_txt0 = await retriever.ainvoke(query)
    assert len(docs_txt0) == 1
    md = docs_txt0[0].metadata
    score2 = md["score"]

    assert score1 != score2
