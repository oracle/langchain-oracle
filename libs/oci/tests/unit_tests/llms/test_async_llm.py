# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Unit tests for native async support in the OCIGenAI completion LLM."""

from typing import Any, AsyncIterator, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest

from langchain_oci.llms.oci_generative_ai import OCIGenAI


@pytest.fixture
def mock_oci_client() -> MagicMock:
    client = MagicMock()

    def serialize(obj: Any) -> Dict[str, Any]:
        return {
            "compartmentId": obj.compartment_id,
            "servingMode": {"modelId": obj.serving_mode.model_id},
            "inferenceRequest": {
                "prompt": obj.inference_request.prompt,
                "isStream": obj.inference_request.is_stream,
            },
        }

    client.base_client.sanitize_for_serialization = serialize
    client.base_client.signer = MagicMock()
    client.base_client.config = {}
    return client


def _make_llm(mock_oci_client: MagicMock, model_id: str) -> OCIGenAI:
    return OCIGenAI(
        model_id=model_id,
        compartment_id="test-compartment",
        service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
        client=mock_oci_client,
    )


def _install_async_gen(llm: OCIGenAI, payloads: List[Dict[str, Any]]) -> MagicMock:
    """Install a mock async client whose generate_text_async yields payloads."""
    async_client = MagicMock()
    calls: List[Dict[str, Any]] = []

    def generate_text_async(
        body: Dict[str, Any], stream: bool = False, timeout: int = 300
    ) -> AsyncIterator[Dict[str, Any]]:
        calls.append({"body": body, "stream": stream})

        async def gen() -> AsyncIterator[Dict[str, Any]]:
            for p in payloads:
                yield p

        return gen()

    async_client.generate_text_async = MagicMock(side_effect=generate_text_async)
    async_client.close = AsyncMock()
    async_client.calls = calls
    llm.__dict__["_async_client"] = async_client
    return async_client


@pytest.mark.requires("oci")
async def test_acall_cohere_response(mock_oci_client: MagicMock) -> None:
    llm = _make_llm(mock_oci_client, "cohere.command")
    async_client = _install_async_gen(
        llm,
        [{"inferenceResponse": {"generatedTexts": [{"text": "hello there"}]}}],
    )

    result = await llm.ainvoke("hi")

    assert result == "hello there"
    call = async_client.calls[0]
    assert call["stream"] is False
    assert call["body"]["inferenceRequest"]["prompt"] == "hi"
    assert call["body"]["inferenceRequest"]["isStream"] is False


@pytest.mark.requires("oci")
async def test_acall_meta_response(mock_oci_client: MagicMock) -> None:
    llm = _make_llm(mock_oci_client, "meta.llama-3.3-70b-instruct")
    _install_async_gen(
        llm,
        [{"inferenceResponse": {"choices": [{"text": "meta says hi"}]}}],
    )

    assert await llm.ainvoke("hi") == "meta says hi"


@pytest.mark.requires("oci")
async def test_acall_enforces_stop(mock_oci_client: MagicMock) -> None:
    llm = _make_llm(mock_oci_client, "cohere.command")
    _install_async_gen(
        llm,
        [{"inferenceResponse": {"generatedTexts": [{"text": "one two STOP three"}]}}],
    )

    result = await llm.ainvoke("hi", stop=["STOP"])
    assert result == "one two "


@pytest.mark.requires("oci")
async def test_astream(mock_oci_client: MagicMock) -> None:
    llm = _make_llm(mock_oci_client, "cohere.command")
    async_client = _install_async_gen(
        llm, [{"text": "chunk1"}, {"text": "chunk2"}, {"finishReason": "stop"}]
    )

    chunks = [c async for c in llm.astream("hi")]

    assert [c for c in chunks if c] and "".join(chunks) == "chunk1chunk2"
    assert async_client.calls[0]["stream"] is True
    assert async_client.calls[0]["body"]["inferenceRequest"]["isStream"] is True


@pytest.mark.requires("oci")
async def test_astream_does_not_mutate_is_stream(mock_oci_client: MagicMock) -> None:
    """Unlike the sync path, _astream must not flip shared instance state."""
    llm = _make_llm(mock_oci_client, "cohere.command")
    _install_async_gen(llm, [{"text": "x"}])

    assert llm.is_stream is False
    async for _ in llm.astream("hi"):
        pass
    assert llm.is_stream is False


@pytest.mark.requires("oci")
async def test_async_uses_no_sync_client(mock_oci_client: MagicMock) -> None:
    llm = _make_llm(mock_oci_client, "cohere.command")
    _install_async_gen(
        llm, [{"inferenceResponse": {"generatedTexts": [{"text": "y"}]}}]
    )

    await llm.ainvoke("hi")

    mock_oci_client.generate_text.assert_not_called()


@pytest.mark.requires("oci")
async def test_aclose(mock_oci_client: MagicMock) -> None:
    llm = _make_llm(mock_oci_client, "cohere.command")
    async_client = _install_async_gen(llm, [])

    await llm.aclose()

    async_client.close.assert_awaited_once()
    assert "_async_client" not in llm.__dict__
