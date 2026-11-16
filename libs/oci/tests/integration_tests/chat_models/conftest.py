# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Shared configuration for chat-model integration tests.

The model lists exercised by the chat-model suite (reasoning models,
standard models, multimodal models, etc.) default to whatever the live
OCI Generative AI catalog currently serves (see ``../live_models.py``),
so retired models never turn into 404s. Each list can be pinned with an
env var for a given deployment. When the catalog can't be read and no
env var is set, the lists are empty and parametrized tests collect as
skipped.

## Environment variables

- ``OCI_REASONING_MODELS`` — comma-separated model ids expected to
  populate the ``reasoning_content`` field on every response.
  Default: the newest live xAI ``*-reasoning`` model.
- ``OCI_STANDARD_MODELS`` — comma-separated model ids that should
  never populate ``reasoning_content``. Default: the newest live Meta
  and Cohere chat models.

  ``openai.gpt-oss-*`` is intentionally in neither default list: its
  ``reasoning_content`` is unreliable on the OCI deployment (sometimes
  present, sometimes null, on both invoke and stream), so it can't anchor
  either a "has reasoning" or "no reasoning" assertion. See #213 / #208.
- ``OCI_LLAMA_MODELS``, ``OCI_COHERE_MODELS``, ``OCI_GROK_MODELS``,
  ``OCI_OPENAI_MODELS`` — comma-separated model ids for the
  per-provider matrices exercised by ``test_multi_model.py``. Defaults:
  the newest live Meta / Cohere / xAI chat models and the live
  ``openai.gpt-oss-*`` models.
- ``OCI_VISION_MODELS`` — comma-separated vision-capable model ids
  for ``test_vision_manual.py``. Default: one live vision model per vendor.
"""

from __future__ import annotations

import os
from typing import List

from .. import live_models as _live


def _env_list(name: str, default: List[str]) -> List[str]:
    """Read a comma-separated env var into a list, falling back to *default*."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return list(default)
    return [item.strip() for item in raw.split(",") if item.strip()]


def reasoning_models() -> List[str]:
    """Models the suite should treat as reasoning-capable."""
    return _live.reasoning_models("OCI_REASONING_MODELS")


def standard_models() -> List[str]:
    """Models the suite should treat as non-reasoning."""
    default = [m for v in ("meta", "cohere") for m in _live.chat_models(v, 1)]
    return _env_list("OCI_STANDARD_MODELS", default)


def llama_models() -> List[str]:
    """Meta Llama model ids exercised by the multi-model matrix."""
    return _live.chat_models("meta", 4, env_var="OCI_LLAMA_MODELS")


def cohere_models() -> List[str]:
    """Cohere model ids exercised by the multi-model matrix."""
    return _live.chat_models("cohere", 3, env_var="OCI_COHERE_MODELS")


def grok_models() -> List[str]:
    """xAI Grok model ids exercised by the multi-model matrix."""
    return _live.chat_models("xai", 3, env_var="OCI_GROK_MODELS")


def openai_models() -> List[str]:
    """OpenAI-on-OCI model ids exercised by the multi-model matrix."""
    return _live.family_models(r"^openai\.gpt-oss-", env_var="OCI_OPENAI_MODELS")


def vision_models() -> List[str]:
    """Vision-capable model ids exercised by manual vision smoke tests."""
    return _live.vision_models("OCI_VISION_MODELS")
