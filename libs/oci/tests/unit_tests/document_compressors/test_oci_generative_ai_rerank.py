# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Unit tests for OCIGenAIRerank."""

from typing import Any, List
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.documents import Document

from langchain_oci import OCIGenAIRerank


class _Rank:
    def __init__(self, index: int, relevance_score: float):
        self.index = index
        self.relevance_score = relevance_score


@pytest.fixture
def mock_client() -> MagicMock:
    client = MagicMock()
    response = MagicMock()
    response.data.document_ranks = [_Rank(2, 0.98), _Rank(0, 0.42), _Rank(1, 0.05)]
    client.rerank_text.return_value = response
    client.base_client.signer = MagicMock()
    client.base_client.config = {}
    return client


@pytest.fixture
def reranker(mock_client: MagicMock) -> OCIGenAIRerank:
    return OCIGenAIRerank(
        model_id="cohere.rerank-v3.5",
        compartment_id="test-compartment",
        service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
        client=mock_client,
        top_n=2,
    )


DOCS = [
    Document(page_content="cats are mammals", metadata={"src": "a"}),
    Document(page_content="the sky is blue"),
    Document(page_content="oracle database supports vectors"),
]


@pytest.mark.requires("oci")
def test_rerank_returns_sorted_scores(reranker: OCIGenAIRerank) -> None:
    results = reranker.rerank(DOCS, "vector databases")

    assert [r["index"] for r in results] == [2, 0, 1]
    assert results[0]["relevance_score"] == 0.98


@pytest.mark.requires("oci")
def test_rerank_request_shape(reranker: OCIGenAIRerank, mock_client: MagicMock) -> None:
    reranker.rerank(DOCS, "vector databases")

    details = mock_client.rerank_text.call_args.args[0]
    assert details.input == "vector databases"
    assert details.documents == [d.page_content for d in DOCS]
    assert details.top_n == 2
    assert details.compartment_id == "test-compartment"
    assert details.serving_mode.model_id == "cohere.rerank-v3.5"


@pytest.mark.requires("oci")
def test_rerank_empty_documents(
    reranker: OCIGenAIRerank, mock_client: MagicMock
) -> None:
    assert reranker.rerank([], "q") == []
    mock_client.rerank_text.assert_not_called()


@pytest.mark.requires("oci")
def test_score_threshold(mock_client: MagicMock) -> None:
    reranker = OCIGenAIRerank(
        compartment_id="c",
        service_endpoint="https://example.com",
        client=mock_client,
        score_threshold=0.4,
    )
    results = reranker.rerank(DOCS, "q")
    assert [r["index"] for r in results] == [2, 0]


@pytest.mark.requires("oci")
def test_compress_documents(reranker: OCIGenAIRerank) -> None:
    compressed = reranker.compress_documents(DOCS, "vector databases")

    assert [d.page_content for d in compressed] == [
        "oracle database supports vectors",
        "cats are mammals",
        "the sky is blue",
    ]
    assert compressed[0].metadata["relevance_score"] == 0.98
    # original metadata preserved and source docs not mutated
    assert compressed[1].metadata["src"] == "a"
    assert "relevance_score" not in DOCS[0].metadata


@pytest.mark.requires("oci")
async def test_acompress_documents(reranker: OCIGenAIRerank) -> None:
    reranker.client.base_client.sanitize_for_serialization = lambda obj: {
        "input": obj.input,
        "documents": obj.documents,
    }
    async_client = MagicMock()
    async_client.rerank_text_async = AsyncMock(
        return_value={
            "documentRanks": [
                {"index": 1, "relevanceScore": 0.9},
                {"index": 0, "relevanceScore": 0.1},
            ]
        }
    )
    reranker.__dict__["_async_client"] = async_client

    compressed = await reranker.acompress_documents(DOCS, "sky")

    assert [d.page_content for d in compressed] == [
        "the sky is blue",
        "cats are mammals",
    ]
    assert compressed[0].metadata["relevance_score"] == 0.9
    reranker.client.rerank_text.assert_not_called()


@pytest.mark.requires("oci")
def test_rerank_accepts_strings_and_dicts(
    reranker: OCIGenAIRerank, mock_client: MagicMock
) -> None:
    docs: List[Any] = ["plain string", {"text": "dict text"}, DOCS[0]]
    reranker.rerank(docs, "q")
    details = mock_client.rerank_text.call_args.args[0]
    assert details.documents == ["plain string", "dict text", "cats are mammals"]
