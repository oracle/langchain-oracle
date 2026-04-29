"""End-to-end OpenSearch Deep Research sample using two Hugging Face datasets."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Iterable, Optional, Sequence

from langchain_core.documents import Document

from langchain_oci import OCIGenAIEmbeddings, create_deep_research_agent
from langchain_oci.datastores import OpenSearch


def load_huggingface_dataset(
    dataset_name: str,
    *,
    split: str,
    subset: Optional[str] = None,
) -> list[dict]:
    """Load a Hugging Face dataset slice as plain Python rows."""
    from datasets import load_dataset

    dataset = load_dataset(
        dataset_name,
        subset,
        split=split,
        verification_mode="no_checks",
    )
    return list(dataset)


def pubmed_to_docs(rows: Iterable[dict]) -> list[Document]:
    """Convert PubMed QA rows into Documents."""
    docs: list[Document] = []
    for idx, row in enumerate(rows):
        context = " ".join((row.get("context") or {}).get("contexts", []))
        docs.append(
            Document(
                page_content=(
                    f"Question: {row.get('question', '')}\n"
                    f"Context: {context}\n"
                    f"Answer: {row.get('final_decision', '')}"
                ),
                metadata={
                    "id": f"pubmed-{idx}",
                    "title": row.get("question", "")[:120] or f"PubMed QA {idx}",
                    "source": "pubmed_qa",
                    "dataset": "pubmed_qa",
                    "domain": "medical",
                },
            )
        )
    return docs


def ag_news_to_docs(rows: Iterable[dict]) -> list[Document]:
    """Convert AG News rows into Documents."""
    docs: list[Document] = []
    for idx, row in enumerate(rows):
        docs.append(
            Document(
                page_content=f"News: {row.get('text', '')}",
                metadata={
                    "id": f"ag-news-{idx}",
                    "title": row.get("text", "")[:120] or f"AG News {idx}",
                    "source": "ag_news",
                    "dataset": "ag_news",
                    "domain": "news",
                    "label": row.get("label"),
                },
            )
        )
    return docs


def docs_to_records(documents: Sequence[Document]) -> list[dict]:
    """Convert Documents into datastore insert payloads."""
    records: list[dict] = []
    for index, document in enumerate(documents):
        metadata = dict(document.metadata or {})
        records.append(
            {
                "id": metadata.get("id", f"doc-{index}"),
                "title": metadata.get("title", f"Document {index}"),
                "content": document.page_content,
                "source": metadata.get("source", "sample"),
            }
        )
    return records


def require_env(name: str, default: Optional[str] = None) -> str:
    """Read a required environment variable."""
    value = os.environ.get(name, default)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def build_embedding_model() -> OCIGenAIEmbeddings:
    """Create the OCI embedding model used for ingest and query time."""
    region = os.environ.get("OCI_REGION", "us-ashburn-1")
    service_endpoint = os.environ.get(
        "OCI_SERVICE_ENDPOINT",
        f"https://inference.generativeai.{region}.oci.oraclecloud.com",
    )
    return OCIGenAIEmbeddings(
        model_id=os.environ.get("OCI_EMBEDDING_MODEL", "cohere.embed-v4.0"),
        compartment_id=require_env("OCI_COMPARTMENT_ID"),
        service_endpoint=service_endpoint,
        auth_type=os.environ.get("OCI_AUTH_TYPE", "API_KEY"),
        auth_profile=os.environ.get("OCI_AUTH_PROFILE", "DEFAULT"),
    )


def build_stores(run_id: str) -> dict[str, OpenSearch]:
    """Create the two OpenSearch datastores used in the sample."""
    common_kwargs = {
        "endpoint": require_env("OPENSEARCH_ENDPOINT"),
        "username": require_env("OPENSEARCH_USERNAME"),
        "password": require_env("OPENSEARCH_PASSWORD"),
        "use_ssl": os.environ.get("OPENSEARCH_USE_SSL", "true").lower() == "true",
        "verify_certs": (
            os.environ.get("OPENSEARCH_VERIFY_CERTS", "false").lower() == "true"
        ),
        "vector_field": os.environ.get("OPENSEARCH_VECTOR_FIELD", "vector_field"),
        "search_fields": os.environ.get(
            "OPENSEARCH_SEARCH_FIELDS",
            "title,content",
        ).split(","),
    }
    return {
        "medical_research": OpenSearch(
            index_name=f"hf_pubmed_research_{run_id}",
            datastore_description=(
                "Medical research, PubMed-style QA, treatment evidence, biomarkers"
            ),
            **common_kwargs,
        ),
        "news_research": OpenSearch(
            index_name=f"hf_ag_news_research_{run_id}",
            datastore_description=(
                "General news, public reporting, trend summaries, narrative framing"
            ),
            **common_kwargs,
        ),
    }


def ingest_records(
    store: OpenSearch,
    embedding_model: OCIGenAIEmbeddings,
    records: list[dict],
) -> int:
    """Connect to OpenSearch and ingest records."""
    store.connect(embedding_model)
    # For OpenSearch we compute embeddings client-side and write them into
    # the vector field that the sample index mapping expects.
    embeddings = embedding_model.embed_documents([record["content"] for record in records])
    ensure_opensearch_index(store, vector_dimension=len(embeddings[0]) if embeddings else 1536)
    return store.bulk_insert(records, embeddings)


def ensure_opensearch_index(store: OpenSearch, *, vector_dimension: int) -> None:
    """Create the target index with an Eolas-compatible vector mapping if needed."""
    if not hasattr(store, "_client"):
        return

    # Reuse the same knn_vector style Eolas uses so the sample behaves like
    # the real OpenSearch-backed Deep Research path.
    if store._client.indices.exists(index=store.index_name):
        return

    mapping = {
        "settings": {
            "index": {
                "knn": True,
                "knn.algo_param.ef_search": 512,
                "number_of_shards": 1,
                "number_of_replicas": 1,
            }
        },
        "mappings": {
            "properties": {
                store.vector_field: {
                    "type": "knn_vector",
                    "dimension": vector_dimension,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "lucene",
                        "parameters": {"ef_construction": 128, "m": 24},
                    },
                },
                "title": {"type": "text", "analyzer": "standard"},
                "content": {"type": "text", "analyzer": "standard"},
                "source": {"type": "keyword"},
            }
        },
    }
    store._client.indices.create(index=store.index_name, body=mapping)


def create_research_agent(
    datastores: dict[str, OpenSearch],
    embedding_model: OCIGenAIEmbeddings,
):
    """Create the Deep Research agent used in the sample."""
    region = os.environ.get("OCI_REGION", "us-ashburn-1")
    service_endpoint = os.environ.get(
        "OCI_SERVICE_ENDPOINT",
        f"https://inference.generativeai.{region}.oci.oraclecloud.com",
    )
    return create_deep_research_agent(
        datastores=datastores,
        embedding_model=embedding_model,
        model_id=os.environ.get("OCI_DEEP_RESEARCH_MODEL", "google.gemini-2.5-flash"),
        compartment_id=require_env("OCI_COMPARTMENT_ID"),
        service_endpoint=service_endpoint,
        auth_type=os.environ.get("OCI_AUTH_TYPE", "API_KEY"),
        auth_profile=os.environ.get("OCI_AUTH_PROFILE", "DEFAULT"),
        top_k=int(os.environ.get("OCI_DEEP_RESEARCH_TOP_K", "6")),
        max_tokens=int(os.environ.get("OCI_DEEP_RESEARCH_MAX_TOKENS", "65000")),
        middleware=[],
    )


def run_research(agent, prompt: str) -> str:
    """Run the Deep Research prompt and normalize the final content."""
    result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
    final_message = result["messages"][-1]
    return str(getattr(final_message, "content", "") or "")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Download two Hugging Face datasets, ingest them into OpenSearch, and "
            "run Deep Research across both indices."
        )
    )
    parser.add_argument("--limit", type=int, default=25, help="Rows per dataset.")
    parser.add_argument(
        "--run-id",
        default=time.strftime("%Y%m%d%H%M%S"),
        help="Unique suffix for the OpenSearch index names.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("opensearch_multi_index_huggingface_output.md"),
        help="Markdown file for the final Deep Research report.",
    )
    parser.add_argument(
        "--skip-research",
        action="store_true",
        help="Stop after ingestion and skip the Gemini Deep Research run.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the sample."""
    args = parse_args(argv)
    # Use two small public datasets so the sample stays fast and reproducible.
    pubmed_rows = load_huggingface_dataset(
        "pubmed_qa",
        subset="pqa_labeled",
        split=f"train[:{args.limit}]",
    )
    ag_news_rows = load_huggingface_dataset("ag_news", split=f"train[:{args.limit}]")

    medical_docs = pubmed_to_docs(pubmed_rows)
    news_docs = ag_news_to_docs(ag_news_rows)
    embedding_model = build_embedding_model()
    stores = build_stores(args.run_id)

    # Ingest each corpus into its own index so the agent can route between
    # them during the final research step.
    ingest_summary = {
        "medical_research": ingest_records(
            stores["medical_research"],
            embedding_model,
            docs_to_records(medical_docs),
        ),
        "news_research": ingest_records(
            stores["news_research"],
            embedding_model,
            docs_to_records(news_docs),
        ),
    }
    print(json.dumps({"ingested": ingest_summary, "run_id": args.run_id}, indent=2))

    if args.skip_research:
        return 0

    # Keep the prompt grounded in the tiny tutorial corpus. This makes the
    # sample reliable even when only a couple of documents are indexed.
    agent = create_research_agent(stores, embedding_model)
    report = run_research(
        agent,
        (
            "Using the medical_research and news_research datastores only, compare "
            "how each datastore presents information. Focus on structure, tone, "
            "audience, and the kind of evidence each document uses. Produce a "
            "structured research memo grounded only in the retrieved content."
        ),
    )
    args.output.write_text(report, encoding="utf-8")
    print(f"Wrote report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
