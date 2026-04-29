"""End-to-end ADB Deep Research sample using two Hugging Face datasets.

This sample downloads two small datasets from Hugging Face, ingests them into
two ADB vector tables, and runs `create_deep_research_agent(...)` across both
stores.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Iterable, Optional, Sequence

from langchain_core.documents import Document

from langchain_oci import OCIGenAIEmbeddings, create_deep_research_agent
from langchain_oci.datastores import ADB


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
        question = row.get("question", "")
        answer = row.get("final_decision", "")
        docs.append(
            Document(
                page_content=(
                    f"Question: {question}\n"
                    f"Context: {context}\n"
                    f"Answer: {answer}"
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


def build_stores(run_id: str) -> dict[str, ADB]:
    """Create the two ADB datastores used in the sample."""
    # Keep Oracle table names uppercase so later stats/read queries hit
    # the same physical tables that OracleVS creates.
    normalized_run_id = run_id.upper()
    wallet_location = os.path.expanduser(require_env("ADB_WALLET_LOCATION"))
    wallet_password = os.environ.get("ADB_WALLET_PASSWORD")
    common_kwargs = {
        "dsn": require_env("ADB_DSN"),
        "user": require_env("ADB_USER"),
        "password": require_env("ADB_PASSWORD"),
        "wallet_location": wallet_location,
        "wallet_password": wallet_password,
    }
    return {
        "medical_research": ADB(
            table_name=f"VECTOR_DOCUMENTS_MEDICAL_{normalized_run_id}",
            datastore_description=(
                "Medical research, PubMed-style QA, diagnosis, treatment, biomarkers"
            ),
            **common_kwargs,
        ),
        "news_research": ADB(
            table_name=f"VECTOR_DOCUMENTS_NEWS_{normalized_run_id}",
            datastore_description=(
                "General news, public reporting, trend summaries, narrative framing"
            ),
            **common_kwargs,
        ),
    }


def ingest_records(store: ADB, embedding_model: OCIGenAIEmbeddings, records: list[dict]) -> int:
    """Connect to ADB and ingest records."""
    store.connect(embedding_model)
    # ADB uses OracleVS under the hood, so embeddings are produced server-side
    # after the store is connected.
    return store.bulk_insert(records, embeddings=[[] for _ in records])


def create_research_agent(
    datastores: dict[str, ADB],
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
            "Download two Hugging Face datasets, ingest them into ADB, and run "
            "Deep Research across both tables."
        )
    )
    parser.add_argument("--limit", type=int, default=25, help="Rows per dataset.")
    parser.add_argument(
        "--run-id",
        default=time.strftime("%Y%m%d%H%M%S"),
        help="Unique suffix for the ADB table names.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("adb_multi_store_huggingface_output.md"),
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
    ag_news_rows = load_huggingface_dataset(
        "ag_news",
        split=f"train[:{args.limit}]",
    )

    medical_docs = pubmed_to_docs(pubmed_rows)
    news_docs = ag_news_to_docs(ag_news_rows)
    embedding_model = build_embedding_model()
    stores = build_stores(args.run_id)

    # Ingest each corpus into its own datastore so the agent can route
    # between them during the final research step.
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

    # Keep the prompt grounded in the tiny tutorial corpus. A broad medical
    # question would be a poor fit for a two-row demo dataset.
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
