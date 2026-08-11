# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Integration tests for OCIGenAIRerank against real OCI GenAI.

Requires a tenancy where the rerankText action is enabled (some tenancies
list the cohere.rerank-* models in the catalog but return 404 for on-demand
rerankText, which typically means a dedicated AI cluster or an additional
IAM policy is required).

Env vars:
    OCI_COMPARTMENT_ID   -- required
    OCI_RERANK_MODEL_ID  -- required (e.g. cohere.rerank-v3.5 or a dedicated
                            endpoint OCID); the test skips when unset so the
                            suite stays green on tenancies without rerank.
    OCI_REGION / OCI_AUTH_TYPE / OCI_CONFIG_PROFILE -- optional, as elsewhere.
"""

import os

import pytest
from langchain_core.documents import Document

from langchain_oci import OCIGenAIRerank

pytestmark = pytest.mark.skipif(
    not (
        os.environ.get("OCI_COMPARTMENT_ID") and os.environ.get("OCI_RERANK_MODEL_ID")
    ),
    reason="OCI_COMPARTMENT_ID and OCI_RERANK_MODEL_ID must be set",
)

DOCS = [
    Document(page_content="The capital of France is Paris."),
    Document(page_content="Oracle Database 23ai supports AI Vector Search."),
    Document(page_content="Bananas are rich in potassium."),
    Document(page_content="Vector databases store embeddings for similarity search."),
]
QUERY = "How does Oracle support vector similarity search?"


@pytest.fixture
def reranker() -> OCIGenAIRerank:
    region = os.environ.get("OCI_REGION", "us-chicago-1")
    return OCIGenAIRerank(
        model_id=os.environ["OCI_RERANK_MODEL_ID"],
        compartment_id=os.environ["OCI_COMPARTMENT_ID"],
        service_endpoint=(
            f"https://inference.generativeai.{region}.oci.oraclecloud.com"
        ),
        auth_type=os.environ.get("OCI_AUTH_TYPE", "API_KEY"),
        auth_profile=os.environ.get("OCI_CONFIG_PROFILE", "DEFAULT"),
        top_n=2,
    )


def test_compress_documents(reranker: OCIGenAIRerank) -> None:
    out = reranker.compress_documents(DOCS, QUERY)

    assert len(out) == 2
    assert all("relevance_score" in d.metadata for d in out)
    assert out[0].metadata["relevance_score"] >= out[1].metadata["relevance_score"]
    assert "Oracle" in out[0].page_content or "Vector" in out[0].page_content


async def test_acompress_documents(reranker: OCIGenAIRerank) -> None:
    out = await reranker.acompress_documents(DOCS, QUERY)

    assert len(out) == 2
    assert out[0].metadata["relevance_score"] >= out[1].metadata["relevance_score"]
    await reranker.aclose()
