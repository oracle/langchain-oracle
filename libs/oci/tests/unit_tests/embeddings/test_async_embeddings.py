# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Unit tests for native async support in OCI embeddings classes."""

from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest

from langchain_oci.embeddings.oci_generative_ai import OCIGenAIEmbeddings


@pytest.fixture
def mock_oci_client() -> MagicMock:
    client = MagicMock()
    # Identity serializer: EmbedTextDetails objects pass through as-is in
    # these tests; the mocked async client only records what it received.
    client.base_client.sanitize_for_serialization = lambda obj: {
        "inputs": obj.inputs,
        "inputType": getattr(obj, "input_type", None),
        "compartmentId": obj.compartment_id,
    }
    client.base_client.signer = MagicMock()
    client.base_client.config = {}
    return client


@pytest.fixture
def embeddings(mock_oci_client: MagicMock) -> OCIGenAIEmbeddings:
    return OCIGenAIEmbeddings(
        model_id="cohere.embed-v4.0",
        compartment_id="test-compartment",
        service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
        client=mock_oci_client,
    )


def _install_mock_async_client(
    embeddings: OCIGenAIEmbeddings, vectors_by_call: List[List[List[float]]]
) -> MagicMock:
    """Install a mock OCIAsyncClient returning canned embed responses."""
    async_client = MagicMock()
    responses = [{"embeddings": v} for v in vectors_by_call]
    async_client.embed_text_async = AsyncMock(side_effect=responses)
    async_client.close = AsyncMock()
    embeddings.__dict__["_async_client"] = async_client
    return async_client


@pytest.mark.requires("oci")
async def test_aembed_documents(embeddings: OCIGenAIEmbeddings) -> None:
    async_client = _install_mock_async_client(embeddings, [[[0.1, 0.2], [0.3, 0.4]]])

    result = await embeddings.aembed_documents(["hello", "world"])

    assert result == [[0.1, 0.2], [0.3, 0.4]]
    async_client.embed_text_async.assert_awaited_once()
    body = async_client.embed_text_async.await_args.args[0]
    assert body["inputs"] == ["hello", "world"]
    assert body["compartmentId"] == "test-compartment"


@pytest.mark.requires("oci")
async def test_aembed_documents_batches(embeddings: OCIGenAIEmbeddings) -> None:
    """Inputs beyond batch_size are split across multiple async requests."""
    embeddings.batch_size = 2
    async_client = _install_mock_async_client(embeddings, [[[1.0], [2.0]], [[3.0]]])

    result = await embeddings.aembed_documents(["a", "b", "c"])

    assert result == [[1.0], [2.0], [3.0]]
    assert async_client.embed_text_async.await_count == 2
    first, second = async_client.embed_text_async.await_args_list
    assert first.args[0]["inputs"] == ["a", "b"]
    assert second.args[0]["inputs"] == ["c"]


@pytest.mark.requires("oci")
async def test_aembed_query(embeddings: OCIGenAIEmbeddings) -> None:
    _install_mock_async_client(embeddings, [[[0.5, 0.6]]])

    result = await embeddings.aembed_query("hello")

    assert result == [0.5, 0.6]


@pytest.mark.requires("oci")
async def test_aembed_image_batch(embeddings: OCIGenAIEmbeddings) -> None:
    """Async image embedding routes data URIs with input_type=IMAGE."""
    async_client = _install_mock_async_client(embeddings, [[[0.7, 0.8]]])

    data_uri = "data:image/png;base64,aGVsbG8="
    result = await embeddings.aembed_image_batch([data_uri])

    assert result == [[0.7, 0.8]]
    body = async_client.embed_text_async.await_args.args[0]
    assert body["inputs"] == [data_uri]
    assert body["inputType"] == "IMAGE"


@pytest.mark.requires("oci")
async def test_aclose(embeddings: OCIGenAIEmbeddings) -> None:
    async_client = _install_mock_async_client(embeddings, [])

    await embeddings.aclose()

    async_client.close.assert_awaited_once()
    assert "_async_client" not in embeddings.__dict__


@pytest.mark.requires("oci")
async def test_async_uses_no_thread_pool(embeddings: OCIGenAIEmbeddings) -> None:
    """aembed_documents must not fall back to the sync client."""
    _install_mock_async_client(embeddings, [[[0.1]]])

    await embeddings.aembed_documents(["x"])

    embeddings.client.embed_text.assert_not_called()


@pytest.mark.requires("oci")
def test_sync_embed_documents_unchanged(embeddings: OCIGenAIEmbeddings) -> None:
    """Sync path still uses the SDK client."""
    response = MagicMock()
    response.data.embeddings = [[0.9]]
    embeddings.client.embed_text.return_value = response

    assert embeddings.embed_documents(["x"]) == [[0.9]]
    embeddings.client.embed_text.assert_called_once()


class TestModelDeploymentAsync:
    """Async tests for OCIModelDeploymentEndpointEmbeddings."""

    def _make(self) -> Any:
        pytest.importorskip("ads")
        from langchain_oci.embeddings.oci_data_science_model_deployment_endpoint import (  # noqa: E501
            OCIModelDeploymentEndpointEmbeddings,
        )

        return OCIModelDeploymentEndpointEmbeddings(
            endpoint="https://modeldeployment.example.com/predict",
            auth={"signer": None},
        )

    async def test_aembed_documents(self, monkeypatch: pytest.MonkeyPatch) -> None:
        embeddings = self._make()

        captured: Dict[str, Any] = {}

        async def fake_aembedding(texts: List[str]) -> List[List[float]]:
            captured["texts"] = texts
            return [[0.1] for _ in texts]

        monkeypatch.setattr(embeddings, "_aembedding", fake_aembedding)

        result = await embeddings.aembed_documents(["a", "b"])
        assert result == [[0.1], [0.1]]
        assert captured["texts"] == ["a", "b"]

    async def test_aembed_query(self, monkeypatch: pytest.MonkeyPatch) -> None:
        embeddings = self._make()

        async def fake_aembedding(texts: List[str]) -> List[List[float]]:
            return [[0.2]]

        monkeypatch.setattr(embeddings, "_aembedding", fake_aembedding)

        assert await embeddings.aembed_query("q") == [0.2]
