#!/usr/bin/env python3
# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Integration tests for ChatOCIOpenAI (OpenAI Responses API on OCI).

These tests validate both sync and async invocation paths with real OCI auth.

Prerequisites:
    export OCI_COMPARTMENT_ID=<your-compartment-id>
    export OCI_REGION=<region>  # optional, defaults to us-chicago-1
    export OCI_CONFIG_PROFILE=<profile>  # optional, defaults to DEFAULT
    export OCI_AUTH_TYPE=<API_KEY|SECURITY_TOKEN>  # optional, defaults to SECURITY_TOKEN
    export OCI_OPENAI_MODEL_ID=<model-id>  # optional, defaults to openai.gpt-oss-20b

Run:
    pytest tests/integration_tests/chat_models/test_oci_openai_responses_api.py -v
"""

import os

import pytest
from langchain_core.messages import AIMessage
from oci_openai import OciSessionAuth, OciUserPrincipalAuth

from langchain_oci import ChatOCIOpenAI


@pytest.fixture
def oci_openai_config():
    """Load config for ChatOCIOpenAI integration tests."""
    compartment_id = os.environ.get("OCI_COMPARTMENT_ID")
    if not compartment_id:
        pytest.skip("OCI_COMPARTMENT_ID not set")

    auth_type = os.environ.get("OCI_AUTH_TYPE", "SECURITY_TOKEN").upper()
    auth_profile = os.environ.get("OCI_CONFIG_PROFILE", "DEFAULT")
    region = os.environ.get("OCI_REGION", "us-chicago-1")
    model_id = os.environ.get("OCI_OPENAI_MODEL_ID", "openai.gpt-oss-20b")

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
        "model_id": model_id,
    }


def _create_client(config: dict) -> ChatOCIOpenAI:
    # Use store=False so conversation store setup is not required for this smoke test.
    return ChatOCIOpenAI(
        auth=config["auth"],
        compartment_id=config["compartment_id"],
        region=config["region"],
        model=config["model_id"],
        store=False,
    )


@pytest.mark.requires("oci", "oci_openai", "langchain_openai")
def test_oci_openai_sync_invoke(oci_openai_config: dict):
    """ChatOCIOpenAI sync invoke should succeed with OCI auth."""
    client = _create_client(oci_openai_config)
    response = client.invoke([("human", "Reply with exactly: sync-ok")])

    assert isinstance(response, AIMessage)
    assert response.content is not None
    assert len(str(response.content)) > 0


@pytest.mark.requires("oci", "oci_openai", "langchain_openai")
@pytest.mark.asyncio
async def test_oci_openai_async_invoke(oci_openai_config: dict):
    """ChatOCIOpenAI async invoke should use configured async client and succeed."""
    client = _create_client(oci_openai_config)
    response = await client.ainvoke([("human", "Reply with exactly: async-ok")])

    assert isinstance(response, AIMessage)
    assert response.content is not None
    assert len(str(response.content)) > 0
