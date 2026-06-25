# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Shared configuration for chat-model integration tests.

The model lists exercised by the chat-model suite (reasoning models,
standard models, multimodal models, etc.) are env-driven so a given
deployment can swap in whichever OCI Generative AI model ids it
actually has access to. All lists fall back to a sensible default
when the env var is unset, so the suite still runs unmodified on a
fresh checkout.

## Environment variables

- ``OCI_REASONING_MODELS`` — comma-separated model ids expected to
  populate the ``reasoning_content`` field on every response.
  Default: ``xai.grok-3-mini-fast``.
- ``OCI_STANDARD_MODELS`` — comma-separated model ids that should
  never populate ``reasoning_content``. Default:
  ``meta.llama-3.3-70b-instruct,cohere.command-r-08-2024``.

  ``openai.gpt-oss-120b`` is intentionally in neither default list: its
  ``reasoning_content`` is unreliable on the OCI deployment (sometimes
  present, sometimes null, on both invoke and stream), so it can't anchor
  either a "has reasoning" or "no reasoning" assertion. See #213 / #208.
- ``OCI_LLAMA_MODELS``, ``OCI_COHERE_MODELS``, ``OCI_GROK_MODELS``,
  ``OCI_OPENAI_MODELS`` — comma-separated model ids for the
  per-provider matrices exercised by ``test_multi_model.py``. Defaults
  match the previously-hardcoded lists.
- ``OCI_VISION_MODELS`` — comma-separated vision-capable model ids
  for ``test_vision_manual.py``. Default: a meta/google/xai mix.
"""

from __future__ import annotations

import os
from typing import List


def _env_list(name: str, default: List[str]) -> List[str]:
    """Read a comma-separated env var into a list, falling back to *default*."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return list(default)
    return [item.strip() for item in raw.split(",") if item.strip()]


_DEFAULT_REASONING_MODELS: List[str] = ["xai.grok-3-mini-fast"]
_DEFAULT_STANDARD_MODELS: List[str] = [
    "meta.llama-3.3-70b-instruct",
    "cohere.command-r-08-2024",
]


_DEFAULT_LLAMA_MODELS: List[str] = [
    "meta.llama-4-maverick-17b-128e-instruct-fp8",
    "meta.llama-4-scout-17b-16e-instruct",
    "meta.llama-3.3-70b-instruct",
    "meta.llama-3.1-70b-instruct",
]
_DEFAULT_COHERE_MODELS: List[str] = [
    "cohere.command-a-03-2025",
    "cohere.command-r-plus-08-2024",
    "cohere.command-r-08-2024",
]
_DEFAULT_GROK_MODELS: List[str] = [
    "xai.grok-4-fast-non-reasoning",
    "xai.grok-3-fast",
    "xai.grok-3-mini-fast",
]
_DEFAULT_OPENAI_MODELS: List[str] = [
    "openai.gpt-oss-20b",
    "openai.gpt-oss-120b",
]
_DEFAULT_VISION_MODELS: List[str] = [
    "meta.llama-3.2-90b-vision-instruct",
    "meta.llama-4-scout-17b-16e-instruct",
    "google.gemini-2.5-flash",
    "xai.grok-4",
]


def reasoning_models() -> List[str]:
    """Models the suite should treat as reasoning-capable."""
    return _env_list("OCI_REASONING_MODELS", _DEFAULT_REASONING_MODELS)


def standard_models() -> List[str]:
    """Models the suite should treat as non-reasoning."""
    return _env_list("OCI_STANDARD_MODELS", _DEFAULT_STANDARD_MODELS)


def llama_models() -> List[str]:
    """Meta Llama model ids exercised by the multi-model matrix."""
    return _env_list("OCI_LLAMA_MODELS", _DEFAULT_LLAMA_MODELS)


def cohere_models() -> List[str]:
    """Cohere model ids exercised by the multi-model matrix."""
    return _env_list("OCI_COHERE_MODELS", _DEFAULT_COHERE_MODELS)


def grok_models() -> List[str]:
    """xAI Grok model ids exercised by the multi-model matrix."""
    return _env_list("OCI_GROK_MODELS", _DEFAULT_GROK_MODELS)


def openai_models() -> List[str]:
    """OpenAI-on-OCI model ids exercised by the multi-model matrix."""
    return _env_list("OCI_OPENAI_MODELS", _DEFAULT_OPENAI_MODELS)


def vision_models() -> List[str]:
    """Vision-capable model ids exercised by manual vision smoke tests."""
    return _env_list("OCI_VISION_MODELS", _DEFAULT_VISION_MODELS)
