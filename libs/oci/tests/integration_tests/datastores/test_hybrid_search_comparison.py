# ruff: noqa: T201
"""Compare semantic-only vs keyword-only vs hybrid (RRF) on live OpenSearch.

Run from libs/oci/:
    .venv/bin/pytest tests/integration_tests/datastores/ -v -s -k hybrid
"""

import textwrap

import pytest

OPENSEARCH_ENDPOINT = "https://ai-dev.observ.us-ashburn-1.ocs.oraclecloud.com:9200"
OPENSEARCH_INDEX = "hf_pubmed_research_20260324164844"
OPENSEARCH_USERNAME = "ai_user"
OPENSEARCH_PASSWORD = "%36${082h_9}1574"

QUERIES = [
    "cancer treatment outcomes",
    "transanal pull-through procedure",
    "enteroscopy safety community",
]


def _build_store(embedding_model):
    from langchain_oci.datastores.vectorstores.opensearch import OpenSearch

    store = OpenSearch(
        endpoint=OPENSEARCH_ENDPOINT,
        index_name=OPENSEARCH_INDEX,
        username=OPENSEARCH_USERNAME,
        password=OPENSEARCH_PASSWORD,
        vector_field="vector_field",
        search_fields=["title", "content"],
        datastore_description="PubMed medical research papers",
    )
    store.connect(embedding_model)
    return store


def _fmt(label: str, results):
    lines = [f"\n{'=' * 60}", f"  {label}", f"{'=' * 60}"]
    for i, item in enumerate(results):
        doc, score = item if isinstance(item, tuple) else (item, None)
        title = (doc.metadata or {}).get("title", "untitled")
        doc_id = getattr(doc, "id", None) or (doc.metadata or {}).get("id", "?")
        score_str = f"  rrf={score:.5f}" if score is not None else ""
        lines.append(f"  [{i + 1}] {title[:70]}{score_str}")
        lines.append(f"       id={doc_id}")
    return "\n".join(lines)


def _doc_id(doc):
    return str(getattr(doc, "id", None) or (doc.metadata or {}).get("id"))


@pytest.fixture(scope="module")
def embedding_model():
    try:
        from langchain_oci import OCIGenAIEmbeddings
    except ImportError:
        pytest.skip("langchain_oci not available")

    return OCIGenAIEmbeddings(
        model_id="cohere.embed-v4.0",
        service_endpoint=(
            "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
        ),
        compartment_id=(
            "ocid1.tenancy.oc1..aaaaaaaah7ixt2oanvvualoahejm63r66c3pse5u4nd4gzviax7eeeqhrysq"
        ),
        auth_profile="BOAT-OC1",
        auth_type="SECURITY_TOKEN",
    )


@pytest.fixture(scope="module")
def store(embedding_model):
    return _build_store(embedding_model)


@pytest.mark.parametrize("query", QUERIES)
def test_compare_search_modes(store, query):
    top_k = 5

    semantic = store.search_documents_with_scores(query, top_k=top_k)
    keyword = store.keyword_search_documents(query, top_k=top_k)
    hybrid = store.hybrid_search_documents(query, top_k=top_k)

    sem_ids = [_doc_id(d) for d, _ in semantic]
    kw_ids = [_doc_id(d) for d in keyword]
    hyb_ids = [_doc_id(d) for d, _ in hybrid]

    print(f"\n\nQUERY: {query!r}")
    print(_fmt("SEMANTIC (vector kNN)", semantic))
    print(_fmt("KEYWORD  (BM25 multi_match)", keyword))
    print(_fmt("HYBRID   (RRF fusion)", hybrid))

    sem_set = set(sem_ids[:top_k])
    kw_set = set(kw_ids[:top_k])
    hyb_set = set(hyb_ids[:top_k])

    print(
        textwrap.dedent(f"""
        Overlap:
          semantic ∩ keyword : {len(sem_set & kw_set)} docs
          semantic only      : {len(sem_set - kw_set)} docs
          keyword only       : {len(kw_set - sem_set)} docs
          hybrid unique      : {len(hyb_set - sem_set - kw_set)} docs
        Hybrid top-1:   {hyb_ids[0] if hyb_ids else "n/a"}
        Semantic top-1: {sem_ids[0] if sem_ids else "n/a"}
        """)
    )

    assert len(hybrid) > 0, "hybrid returned nothing"
    assert len(hyb_set & sem_set) > 0, "hybrid shares no results with semantic"
