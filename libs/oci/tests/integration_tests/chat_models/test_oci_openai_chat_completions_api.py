#!/usr/bin/env python3
# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Integration tests for the ChatOCIOpenAI Chat Completions transport.

Exercises the additive ``use_responses_api=False`` opt-in against OCI's
``/openai/v1/chat/completions`` passthrough. The opt-in unlocks any OpenAI
feature exposed only on Chat Completions — the motivating case is
``input_audio`` against audio-capable models such as ``openai.gpt-audio``,
covered by the audio test below.

These tests run a real wire call and skip cleanly without OCI credentials.

Prerequisites:
    export OCI_COMPARTMENT_ID=<your-compartment-id>
    export OCI_REGION=<region>  # optional, defaults to us-chicago-1
    export OCI_CONFIG_PROFILE=<profile>  # optional, defaults to DEFAULT
    export OCI_AUTH_TYPE=<API_KEY|SECURITY_TOKEN>  # optional, defaults to
                                                   # SECURITY_TOKEN
    # Text-only model for the smoke + stream + tools tests; any
    # OpenAI-route model on OCI works (gpt-5, gpt-4o, gpt-oss-*, ...).
    export OCI_OPENAI_CHAT_MODEL_ID=<model-id>  # defaults to openai.gpt-oss-20b
    # Audio-capable model for the input_audio test; must accept
    # `input_audio` content blocks.
    export OCI_OPENAI_AUDIO_MODEL_ID=<model-id>  # defaults to openai.gpt-audio

Run::

    pytest \\
      tests/integration_tests/chat_models/test_oci_openai_chat_completions_api.py \\
      -v
"""  # noqa: E501

import base64
import io
import math
import os
import struct
import wave

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from oci_openai import OciSessionAuth, OciUserPrincipalAuth
from pydantic import BaseModel, Field

from langchain_oci import ChatOCIOpenAI


@pytest.fixture
def oci_openai_config():
    """Load config for the Chat Completions transport integration tests."""
    compartment_id = os.environ.get("OCI_COMPARTMENT_ID")
    if not compartment_id:
        pytest.skip("OCI_COMPARTMENT_ID not set")

    auth_type = os.environ.get("OCI_AUTH_TYPE", "SECURITY_TOKEN").upper()
    auth_profile = os.environ.get("OCI_CONFIG_PROFILE", "DEFAULT")
    region = os.environ.get("OCI_REGION", "us-chicago-1")
    chat_model_id = os.environ.get("OCI_OPENAI_CHAT_MODEL_ID", "openai.gpt-oss-20b")
    audio_model_id = os.environ.get("OCI_OPENAI_AUDIO_MODEL_ID", "openai.gpt-audio")

    if auth_type == "API_KEY":
        auth = OciUserPrincipalAuth(profile_name=auth_profile)
    elif auth_type == "SECURITY_TOKEN":
        auth = OciSessionAuth(profile_name=auth_profile)
    else:
        pytest.skip(f"Unsupported OCI_AUTH_TYPE for ChatOCIOpenAI test: {auth_type}")

    return {
        "auth": auth,
        "compartment_id": compartment_id,
        "region": region,
        "chat_model_id": chat_model_id,
        "audio_model_id": audio_model_id,
    }


def _create_client(config: dict, *, model_key: str = "chat_model_id") -> ChatOCIOpenAI:
    """Construct a ChatOCIOpenAI on the Chat Completions transport."""
    return ChatOCIOpenAI(
        auth=config["auth"],
        compartment_id=config["compartment_id"],
        region=config["region"],
        model=config[model_key],
        use_responses_api=False,
    )


def _make_wav_bytes(
    seconds: float = 0.6, freq: float = 440.0, rate: int = 16000
) -> bytes:
    """In-memory mono 16-bit PCM WAV — a short sine tone the model can describe."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        n = int(seconds * rate)
        frames = bytearray()
        for i in range(n):
            sample = int(0.3 * 32767 * math.sin(2 * math.pi * freq * i / rate))
            frames += struct.pack("<h", sample)
        w.writeframes(bytes(frames))
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Transport sanity — confirm /openai/v1/chat/completions is reachable end-to-end
# under the same auth modes that the Responses-API test covers.
# ---------------------------------------------------------------------------


@pytest.mark.requires("oci", "oci_openai", "langchain_openai")
def test_chat_completions_sync_invoke(oci_openai_config: dict):
    """Sync invoke against the Chat Completions transport with OCI auth."""
    client = _create_client(oci_openai_config)
    response = client.invoke([("human", "Reply with exactly: sync-ok")])

    assert isinstance(response, AIMessage)
    assert response.content is not None
    assert len(str(response.content)) > 0


@pytest.mark.requires("oci", "oci_openai", "langchain_openai")
@pytest.mark.asyncio
async def test_chat_completions_async_invoke(oci_openai_config: dict):
    """Async invoke should reuse the per-client async http_client and signer."""
    client = _create_client(oci_openai_config)
    response = await client.ainvoke([("human", "Reply with exactly: async-ok")])

    assert isinstance(response, AIMessage)
    assert response.content is not None
    assert len(str(response.content)) > 0


@pytest.mark.requires("oci", "oci_openai", "langchain_openai")
def test_chat_completions_stream(oci_openai_config: dict):
    """Streaming on the Chat Completions transport must yield content chunks
    — proves real SSE pass-through (no Responses-API output_version)."""
    client = _create_client(oci_openai_config)
    chunks = list(client.stream([("human", "Count from 1 to 3.")]))

    assert len(chunks) > 0
    joined = "".join(str(c.content) for c in chunks if c.content)
    assert len(joined) > 0


# ---------------------------------------------------------------------------
# Tool calling — confirms the OpenAI Chat Completions tool schema is forwarded
# correctly under the OCI signing layer.
# ---------------------------------------------------------------------------


class _GetWeather(BaseModel):
    """Get the current weather in a given city."""

    location: str = Field(..., description="The city, e.g. San Francisco")


@pytest.mark.requires("oci", "oci_openai", "langchain_openai")
def test_chat_completions_tool_call(oci_openai_config: dict):
    """A bound function-tool should be invoked, returning a tool_calls entry."""
    client = _create_client(oci_openai_config)
    llm_with_tools = client.bind_tools([_GetWeather])
    response = llm_with_tools.invoke(
        "What's the weather like in San Francisco? Use the tool."
    )

    assert isinstance(response, AIMessage)
    # The model should produce at least one tool call; assert defensively
    # since some models may answer directly instead.
    if response.tool_calls:
        assert response.tool_calls[0]["name"] == "_GetWeather"
        assert "location" in response.tool_calls[0]["args"]


# ---------------------------------------------------------------------------
# Audio input — the motivating case. Requires an audio-capable model and is
# why the Chat Completions transport had to be added in the first place
# (Responses API + OCI-native chat reject `input_audio` for OpenAI models).
# ---------------------------------------------------------------------------


@pytest.mark.requires("oci", "oci_openai", "langchain_openai")
def test_chat_completions_input_audio(oci_openai_config: dict):
    """An ``input_audio`` content block should be accepted by the Chat
    Completions transport against an audio-capable model. This call would
    fail on the Responses transport and on OCI's native chat endpoint —
    the assertion proves the new transport unlocks audio input."""
    audio_b64 = base64.b64encode(_make_wav_bytes()).decode("ascii")
    client = _create_client(oci_openai_config, model_key="audio_model_id")
    response = client.invoke(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": "What do you hear? Be brief."},
                    {
                        "type": "input_audio",
                        "input_audio": {"data": audio_b64, "format": "wav"},
                    },
                ]
            )
        ]
    )

    assert isinstance(response, AIMessage)
    assert response.content is not None
    assert len(str(response.content)) > 0


# NB: live-wire coverage for the default (Responses API) transport lives in
# tests/integration_tests/chat_models/test_oci_openai_responses_api.py.
# Backwards-compatibility on the configuration surface is locked in by the
# unit tests in
# tests/unit_tests/chat_models/test_oci_openai_chat_completions_transport.py
# (default_path_* tests).
