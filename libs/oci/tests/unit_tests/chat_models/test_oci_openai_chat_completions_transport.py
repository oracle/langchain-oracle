# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Unit tests for the ChatOCIOpenAI Chat Completions transport.

Covers the additive ``use_responses_api=False`` opt-in on
:class:`ChatOCIOpenAI`, which targets OCI's
``/openai/v1/chat/completions`` passthrough. The opt-in unlocks any OpenAI
feature exposed only on Chat Completions — the motivating case is
``input_audio`` against audio-capable models such as ``openai.gpt-audio``,
but the same path applies to other Chat-Completions-only features. The
existing default (``True``) keeps using the Responses passthrough.

The tests prove two invariants — important because the change must be
strictly additive:

1. **Default (Responses) path is byte-equivalent** to the prior release —
   same headers (compartment + conversation-store), same
   ``use_responses_api=True``, same ``output_version="responses/v1"``.
2. **Chat Completions path is correctly stateless** — only the compartment
   header is sent; the Responses-only knobs (``conversation_store_id``
   header, ``output_version``) are absent.
"""

from unittest.mock import MagicMock

import pytest

from langchain_oci import ChatOCIOpenAI
from langchain_oci.chat_models.oci_generative_ai import (
    COMPARTMENT_ID_HEADER,
    CONVERSATION_STORE_ID_HEADER,
    OUTPUT_VERSION,
)

COMPARTMENT_ID = "ocid1.compartment.oc1..dummy"
CONVERSATION_STORE_ID = "ocid1.generativeaiconversationstore.oc1..dummy"
REGION = "us-chicago-1"
AUDIO_MODEL = "openai.gpt-audio"
TEXT_MODEL = "openai.gpt-5"


@pytest.fixture
def fake_auth():
    """Lightweight stand-in for an OciUserPrincipalAuth / SessionAuth instance."""
    return MagicMock()


# ---------------------------------------------------------------------------
# Backwards-compatibility: default path must be byte-equivalent to pre-change.
# ---------------------------------------------------------------------------


@pytest.mark.requires("langchain_openai")
def test_default_path_uses_responses_api(fake_auth):
    """No flag passed → Responses API (preserves prior behavior)."""
    client = ChatOCIOpenAI(
        auth=fake_auth,
        compartment_id=COMPARTMENT_ID,
        conversation_store_id=CONVERSATION_STORE_ID,
        region=REGION,
        model=TEXT_MODEL,
    )
    assert client.use_responses_api is True


@pytest.mark.requires("langchain_openai")
def test_default_path_sends_responses_output_version(fake_auth):
    """The Responses API requires output_version='responses/v1'; the default
    path must continue to set it."""
    client = ChatOCIOpenAI(
        auth=fake_auth,
        compartment_id=COMPARTMENT_ID,
        conversation_store_id=CONVERSATION_STORE_ID,
        region=REGION,
        model=TEXT_MODEL,
    )
    assert getattr(client, "output_version", None) == OUTPUT_VERSION


@pytest.mark.requires("langchain_openai")
def test_default_path_sends_compartment_and_conversation_store_headers(fake_auth):
    """The Responses path's two-header invariant (compartment +
    conversation-store) is the prior wire contract — locked in by this test
    so future refactors can't regress it."""
    client = ChatOCIOpenAI(
        auth=fake_auth,
        compartment_id=COMPARTMENT_ID,
        conversation_store_id=CONVERSATION_STORE_ID,
        region=REGION,
        model=TEXT_MODEL,
    )
    assert client.http_client is not None
    assert client.http_client.headers.get(COMPARTMENT_ID_HEADER) == COMPARTMENT_ID
    assert (
        client.http_client.headers.get(CONVERSATION_STORE_ID_HEADER)
        == CONVERSATION_STORE_ID
    )


@pytest.mark.requires("langchain_openai")
def test_default_path_still_requires_conversation_store_id_when_store_true(fake_auth):
    """Pre-existing validation must remain — store=True without a
    conversation-store OCID is still an error."""
    with pytest.raises(ValueError, match="Conversation Store Id"):
        ChatOCIOpenAI(
            auth=fake_auth,
            compartment_id=COMPARTMENT_ID,
            region=REGION,
            model=TEXT_MODEL,
        )


# ---------------------------------------------------------------------------
# Chat Completions opt-in (use_responses_api=False).
# ---------------------------------------------------------------------------


@pytest.mark.requires("langchain_openai")
def test_chat_completions_path_sets_use_responses_api_false(fake_auth):
    client = ChatOCIOpenAI(
        auth=fake_auth,
        compartment_id=COMPARTMENT_ID,
        region=REGION,
        model=AUDIO_MODEL,
        use_responses_api=False,
    )
    assert client.use_responses_api is False


@pytest.mark.requires("langchain_openai")
def test_chat_completions_path_omits_responses_output_version(fake_auth):
    """``output_version='responses/v1'`` is a Responses-API-only knob; the
    Chat Completions path must not send it."""
    client = ChatOCIOpenAI(
        auth=fake_auth,
        compartment_id=COMPARTMENT_ID,
        region=REGION,
        model=AUDIO_MODEL,
        use_responses_api=False,
    )
    assert getattr(client, "output_version", None) != OUTPUT_VERSION


@pytest.mark.requires("langchain_openai")
def test_chat_completions_path_sends_only_compartment_header(fake_auth):
    """Chat Completions is stateless on OCI's side, so only the compartment
    header is needed — the conversation-store header would be meaningless."""
    client = ChatOCIOpenAI(
        auth=fake_auth,
        compartment_id=COMPARTMENT_ID,
        region=REGION,
        model=AUDIO_MODEL,
        use_responses_api=False,
    )
    assert client.http_client is not None
    assert client.http_client.headers.get(COMPARTMENT_ID_HEADER) == COMPARTMENT_ID
    assert CONVERSATION_STORE_ID_HEADER not in client.http_client.headers


@pytest.mark.requires("langchain_openai")
def test_chat_completions_path_does_not_require_conversation_store_id(fake_auth):
    """Inverse of the default-path validation: the Chat Completions opt-in
    must construct cleanly without a conversation-store OCID."""
    ChatOCIOpenAI(
        auth=fake_auth,
        compartment_id=COMPARTMENT_ID,
        region=REGION,
        model=AUDIO_MODEL,
        use_responses_api=False,
    )


@pytest.mark.requires("langchain_openai")
def test_chat_completions_path_targets_openai_v1_base_url(fake_auth):
    """Both transports share the same ``/openai/v1`` base — ChatOpenAI then
    appends ``/responses`` or ``/chat/completions`` based on
    ``use_responses_api``. Locking the base URL down here so a future
    ``_resolve_base_url`` change doesn't silently break either transport."""
    client = ChatOCIOpenAI(
        auth=fake_auth,
        compartment_id=COMPARTMENT_ID,
        region=REGION,
        model=AUDIO_MODEL,
        use_responses_api=False,
    )
    base = str(client.openai_api_base)
    assert base.rstrip("/").endswith("/openai/v1")
