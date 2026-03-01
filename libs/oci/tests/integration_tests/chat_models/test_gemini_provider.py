# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Integration tests for GeminiProvider.

Tests the GeminiProvider parameter normalization and basic functionality
with real OCI GenAI service calls.

Usage:
  export OCI_COMPARTMENT_ID="ocid1.compartment..."
  export OCI_CONFIG_PROFILE="NON-DEFAULT"  # Chicago region has Gemini
  export OCI_AUTH_TYPE="SECURITY_TOKEN"
  export OCI_REGION="us-chicago-1"

  python -m pytest tests/integration_tests/chat_models/test_gemini_provider.py -v

Output truncation diagnostic:
  export OCI_RUN_GEMINI_TRUNCATION_DIAGNOSTIC=1
  export OCI_MODEL_ID="google.gemini-2.5-pro"  # optional
  export OCI_MAX_TOKENS=8000                   # optional
  export OCI_TRUNCATION_TRIALS=10              # optional
  export OCI_TRUNCATION_LINES=800              # optional
"""

import os
import warnings
from typing import Any, Dict

import pytest

from langchain_oci.chat_models import ChatOCIGenAI


def _get_config() -> Dict[str, Any]:
    """Get OCI configuration from environment."""
    compartment_id = os.environ.get("OCI_COMPARTMENT_ID")
    if not compartment_id:
        pytest.skip("OCI_COMPARTMENT_ID not set")

    region = os.getenv("OCI_REGION", "us-chicago-1")
    endpoint = os.environ.get(
        "OCI_GENAI_ENDPOINT",
        f"https://inference.generativeai.{region}.oci.oraclecloud.com",
    )

    return {
        "compartment_id": compartment_id,
        "service_endpoint": endpoint,
        "auth_type": os.environ.get("OCI_AUTH_TYPE", "SECURITY_TOKEN"),
        "auth_profile": os.environ.get("OCI_CONFIG_PROFILE", "NON-DEFAULT"),
        "auth_file_location": os.path.expanduser("~/.oci/config"),
    }


@pytest.mark.requires("oci")
def test_gemini_basic_invoke() -> None:
    """Test basic invoke with Gemini model."""
    if os.environ.get("OCI_RUN_GEMINI_INTEGRATION") != "1":
        pytest.skip("Set OCI_RUN_GEMINI_INTEGRATION=1 to run this test")

    cfg = _get_config()
    model_id = os.getenv("OCI_MODEL_ID", "google.gemini-2.0-flash-001")

    llm = ChatOCIGenAI(
        model_id=model_id,
        service_endpoint=cfg["service_endpoint"],
        compartment_id=cfg["compartment_id"],
        auth_type=cfg["auth_type"],
        auth_profile=cfg["auth_profile"],
        auth_file_location=cfg["auth_file_location"],
        model_kwargs={"temperature": 0, "max_tokens": 50},
    )

    response = llm.invoke("Say 'Hello' and nothing else.")
    assert response.content is not None
    assert len(response.content) > 0


@pytest.mark.requires("oci")
def test_gemini_max_output_tokens_mapping() -> None:
    """Test that max_output_tokens is correctly mapped to max_tokens with warning."""
    if os.environ.get("OCI_RUN_GEMINI_INTEGRATION") != "1":
        pytest.skip("Set OCI_RUN_GEMINI_INTEGRATION=1 to run this test")

    cfg = _get_config()
    model_id = os.getenv("OCI_MODEL_ID", "google.gemini-2.0-flash-001")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        llm = ChatOCIGenAI(
            model_id=model_id,
            service_endpoint=cfg["service_endpoint"],
            compartment_id=cfg["compartment_id"],
            auth_type=cfg["auth_type"],
            auth_profile=cfg["auth_profile"],
            auth_file_location=cfg["auth_file_location"],
            model_kwargs={"temperature": 0, "max_output_tokens": 50},
        )

        response = llm.invoke("Say 'test' and nothing else.")

        # Verify warning was emitted
        mapping_warnings = [x for x in w if "max_output_tokens" in str(x.message)]
        assert len(mapping_warnings) >= 1, "Expected max_output_tokens warning"
        assert "Mapped" in str(mapping_warnings[0].message)

        # Verify response is valid
        assert response.content is not None


@pytest.mark.requires("oci")
def test_gemini_both_tokens_params() -> None:
    """Test behavior when both max_tokens and max_output_tokens are provided."""
    if os.environ.get("OCI_RUN_GEMINI_INTEGRATION") != "1":
        pytest.skip("Set OCI_RUN_GEMINI_INTEGRATION=1 to run this test")

    cfg = _get_config()
    model_id = os.getenv("OCI_MODEL_ID", "google.gemini-2.0-flash-001")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        llm = ChatOCIGenAI(
            model_id=model_id,
            service_endpoint=cfg["service_endpoint"],
            compartment_id=cfg["compartment_id"],
            auth_type=cfg["auth_type"],
            auth_profile=cfg["auth_profile"],
            auth_file_location=cfg["auth_file_location"],
            model_kwargs={
                "temperature": 0,
                "max_tokens": 50,
                "max_output_tokens": 100,  # Should be ignored
            },
        )

        response = llm.invoke("Say 'both' and nothing else.")

        # Verify warning about both being provided
        both_warnings = [x for x in w if "Both" in str(x.message)]
        assert len(both_warnings) >= 1, "Expected warning when both params provided"
        assert "ignoring" in str(both_warnings[0].message).lower()

        # Verify response is valid
        assert response.content is not None


@pytest.mark.requires("oci")
def test_gemini_streaming() -> None:
    """Test streaming with Gemini model."""
    if os.environ.get("OCI_RUN_GEMINI_INTEGRATION") != "1":
        pytest.skip("Set OCI_RUN_GEMINI_INTEGRATION=1 to run this test")

    cfg = _get_config()
    model_id = os.getenv("OCI_MODEL_ID", "google.gemini-2.0-flash-001")

    llm = ChatOCIGenAI(
        model_id=model_id,
        service_endpoint=cfg["service_endpoint"],
        compartment_id=cfg["compartment_id"],
        auth_type=cfg["auth_type"],
        auth_profile=cfg["auth_profile"],
        auth_file_location=cfg["auth_file_location"],
        model_kwargs={"temperature": 0, "max_tokens": 100},
    )

    chunks: list[str] = []
    for chunk in llm.stream("Count from 1 to 5."):
        if chunk.content:
            chunks.append(str(chunk.content))

    assert len(chunks) > 0, "Expected at least one chunk"
    full_response = "".join(chunks)
    assert len(full_response) > 0


@pytest.mark.requires("oci")
def test_gemini_streaming_with_max_output_tokens() -> None:
    """Test streaming with max_output_tokens parameter (should be normalized)."""
    if os.environ.get("OCI_RUN_GEMINI_INTEGRATION") != "1":
        pytest.skip("Set OCI_RUN_GEMINI_INTEGRATION=1 to run this test")

    cfg = _get_config()
    model_id = os.getenv("OCI_MODEL_ID", "google.gemini-2.0-flash-001")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        llm = ChatOCIGenAI(
            model_id=model_id,
            service_endpoint=cfg["service_endpoint"],
            compartment_id=cfg["compartment_id"],
            auth_type=cfg["auth_type"],
            auth_profile=cfg["auth_profile"],
            auth_file_location=cfg["auth_file_location"],
            model_kwargs={"temperature": 0, "max_output_tokens": 100},
        )

        chunks = []
        for chunk in llm.stream("Say 'streaming works' and nothing else."):
            if chunk.content:
                chunks.append(chunk.content)

        # Verify warning was emitted
        mapping_warnings = [x for x in w if "max_output_tokens" in str(x.message)]
        assert len(mapping_warnings) >= 1, "Expected max_output_tokens warning"

        assert len(chunks) > 0, "Expected at least one chunk"


# --- Output truncation diagnostic tests ---
# These tests verify that Gemini multipart content concatenation works correctly.
# They are opt-in diagnostics that send long-output requests to check for truncation.


def _make_truncation_prompt(sentinel: str, line_count: int) -> str:
    """Generate a prompt requesting structured long output with sentinel markers."""
    return (
        "You must output a plain-text payload with NO extra commentary.\n"
        f"First line: BEGIN:{sentinel}\n"
        f"Then output EXACTLY {line_count} lines numbered 1..{line_count}.\n"
        "Each numbered line must follow this pattern exactly:\n"
        f"  L1:{sentinel}\n"
        f"  L2:{sentinel}\n"
        "  ...\n"
        f"  L{line_count}:{sentinel}\n"
        f"Final line: END:{sentinel}\n"
        "Do not wrap in code fences. Do not omit any lines."
    )


def _summarize_response(resp: Any) -> Dict[str, Any]:
    """Extract summary info from a response for diagnostic output."""
    content = getattr(resp, "content", "") or ""
    usage = getattr(resp, "usage_metadata", None)
    finish_reason = None
    if hasattr(resp, "additional_kwargs"):
        finish_reason = resp.additional_kwargs.get("finish_reason")
    return {
        "chars": len(content),
        "finish_reason": finish_reason,
        "usage": usage.model_dump() if hasattr(usage, "model_dump") else usage,  # type: ignore[union-attr]
        "tail": content[-200:],
    }


@pytest.mark.requires("oci")
def test_gemini_output_truncation_diagnostic() -> None:
    """Diagnostic test for intermittent Gemini output truncation.

    This test validates that Gemini responses are not being cut off unexpectedly.
    It sends multiple requests asking for structured long output and checks
    that all content parts are properly concatenated.

    Enable with: OCI_RUN_GEMINI_TRUNCATION_DIAGNOSTIC=1

    Optional environment variables:
      - OCI_MODEL_ID: Model to test (default: google.gemini-2.5-pro)
      - OCI_MAX_TOKENS: Max output tokens (default: 8000)
      - OCI_TRUNCATION_TRIALS: Number of requests to run (default: 10)
      - OCI_TRUNCATION_LINES: Requested line count (default: 800)
      - OCI_TRUNCATION_TEMPERATURE: Temperature setting (default: 0)
    """
    if os.environ.get("OCI_RUN_GEMINI_TRUNCATION_DIAGNOSTIC") != "1":
        pytest.skip("Set OCI_RUN_GEMINI_TRUNCATION_DIAGNOSTIC=1 to run this test")

    cfg = _get_config()
    model_id = os.getenv("OCI_MODEL_ID", "google.gemini-2.5-pro")
    max_tokens = int(os.getenv("OCI_MAX_TOKENS", "8000"))
    temperature = float(os.getenv("OCI_TRUNCATION_TEMPERATURE", "0"))
    trials = int(os.getenv("OCI_TRUNCATION_TRIALS", "10"))
    line_count = int(os.getenv("OCI_TRUNCATION_LINES", "800"))

    llm = ChatOCIGenAI(
        model_id=model_id,
        service_endpoint=cfg["service_endpoint"],
        compartment_id=cfg["compartment_id"],
        auth_type=cfg["auth_type"],
        auth_profile=cfg["auth_profile"],
        auth_file_location=cfg["auth_file_location"],
        model_kwargs={
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
    )

    failures: list[tuple[int, Dict[str, Any]]] = []
    for i in range(trials):
        sentinel = f"TRUNC-{i}-X"
        prompt = _make_truncation_prompt(sentinel, line_count=line_count)
        resp = llm.invoke(prompt)
        content = resp.content or ""

        # Minimal, robust truncation signal: END marker missing.
        if f"END:{sentinel}" not in content:
            failures.append((i, _summarize_response(resp)))

    if failures:
        # Keep assertion output small but actionable.
        details = "\n".join(
            f"trial={idx} finish_reason={info.get('finish_reason')} "
            f"chars={info.get('chars')} usage={info.get('usage')} "
            f"tail={info.get('tail')!r}"
            for idx, info in failures[:5]
        )
        pytest.fail(
            f"Detected {len(failures)}/{trials} truncated responses for {model_id} "
            f"(max_tokens={max_tokens}). First failures:\n{details}"
        )
