# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Pick integration-test models from the live OCI Generative AI catalog.

OCI retires on-demand models on a rolling schedule, so a hard-coded model id
turns into a 404 ("Entity with key ... not found" / "Model ... is retired")
months after it was written. Instead, tests ask this module for a model by
*role* (a vendor's chat model, the vision models, an embedding model, ...) and
get whatever the catalog currently serves.

A model counts as live when every catalog entry for it is ``ACTIVE`` and none
has a ``time_on_demand_retired`` in the past. Special-purpose variants that
the plain chat endpoint rejects (``-pro``, ``codex``, audio, multi-agent,
voice, ...) and dated snapshot aliases are left out of the chat roles.

Every role can be pinned with an environment variable (comma-separated ids),
which is also how to run the suite against a dedicated endpoint.

## Environment

Catalog access reuses the suite's usual variables: ``OCI_COMPARTMENT_ID``
(required), ``OCI_REGION`` (default ``us-chicago-1``), ``OCI_AUTH_TYPE``
(default ``API_KEY``, the same default the tests themselves use) and
``OCI_CONFIG_PROFILE`` (default ``DEFAULT``). When the catalog can't be read
every role comes back empty, so parametrized tests collect as skipped instead
of failing; the warning says why.
"""

from __future__ import annotations

import datetime as _dt
import functools
import os
import re
import warnings
from typing import Any, Dict, Iterable, List, Optional, Sequence

from langchain_oci.utils.vision import IMAGE_EMBEDDING_MODELS, VISION_MODELS

VENDORS = ("meta", "cohere", "google", "openai", "xai")

# Must match the default the tests pass to the chat/embedding classes, or an
# API-key setup with ``OCI_AUTH_TYPE`` unset would read an empty catalog and
# skip everything while the tests themselves could have run.
DEFAULT_AUTH_TYPE = "API_KEY"

# Name fragments of variants the plain chat endpoint can't serve, or that
# need inputs the generic chat tests don't send.
_NOT_GENERAL_CHAT = (
    "codex",
    "audio",
    "realtime",
    "transcribe",
    "tts",
    "search",
    "image",
    "multi-agent",
    "voice",
    "vision",
    "guard",
)
# OpenAI's ``gpt-*-pro`` tiers aren't chat models ("not supported in the
# v1/chat/completions endpoint"); Gemini's ``-pro`` is, so match OpenAI only.
_OPENAI_PRO = re.compile(r"^openai\..*-pro\b")
# Dated snapshot aliases, e.g. ``openai.gpt-4.1-2025-04-14`` or
# ``xai.grok-4.20-0309-reasoning``; the undated alias is always listed too.
# Cohere's ``command-a-03-2025`` is a canonical name, not an alias.
_DATED = re.compile(r"-\d{4}-\d{2}-\d{2}$|-\d{4}-")


def env_list(name: str) -> List[str]:
    """Read a comma-separated env var into a list (empty when unset)."""
    raw = os.environ.get(name, "").strip()
    return [item.strip() for item in raw.split(",") if item.strip()]


@functools.lru_cache(maxsize=1)
def _catalog() -> Dict[str, List[Any]]:
    """Map model name -> catalog entries, or ``{}`` if the catalog is unreadable."""
    compartment_id = os.environ.get("OCI_COMPARTMENT_ID")
    if not compartment_id:
        return {}
    try:
        import oci

        from langchain_oci.common.auth import create_oci_client_kwargs

        region = os.environ.get("OCI_REGION", "us-chicago-1")
        client = oci.generative_ai.GenerativeAiClient(
            **create_oci_client_kwargs(
                auth_type=os.environ.get("OCI_AUTH_TYPE", DEFAULT_AUTH_TYPE),
                service_endpoint=f"https://generativeai.{region}.oci.oraclecloud.com",
                auth_profile=os.environ.get("OCI_CONFIG_PROFILE", "DEFAULT"),
            )
        )
        items = oci.pagination.list_call_get_all_results(
            client.list_models, compartment_id=compartment_id
        ).data
    except Exception as e:  # noqa: BLE001 - any failure means "no catalog"
        warnings.warn(f"OCI model catalog unavailable, live-model roles are empty: {e}")
        return {}

    catalog: Dict[str, List[Any]] = {}
    for item in items:
        catalog.setdefault(item.display_name, []).append(item)
    return catalog


def _is_servable(entries: Sequence[Any]) -> bool:
    now = _dt.datetime.now(_dt.timezone.utc)
    return all(
        e.lifecycle_state == "ACTIVE"
        and (e.time_on_demand_retired is None or e.time_on_demand_retired > now)
        for e in entries
    )


def live_models(
    capability: str = "CHAT",
    vendor: Optional[str] = None,
) -> List[str]:
    """Servable model ids with *capability*, newest first."""
    rows = []
    for name, entries in _catalog().items():
        if vendor and not name.startswith(f"{vendor}."):
            continue
        if not any(capability in e.capabilities for e in entries):
            continue
        if not _is_servable(entries):
            continue
        rows.append((max(e.time_created for e in entries), name))
    return [name for _, name in sorted(rows, reverse=True)]


def is_live(model_id: str) -> bool:
    """Whether the catalog currently serves *model_id* on demand."""
    entries = _catalog().get(model_id, [])
    return bool(entries) and _is_servable(entries)


def _general_chat(names: Iterable[str]) -> List[str]:
    return [
        n
        for n in names
        if not any(part in n for part in _NOT_GENERAL_CHAT)
        and not _OPENAI_PRO.search(n)
        and not _DATED.search(n)
    ]


def _pinned_or(env_var: str, default: List[str]) -> List[str]:
    pinned = env_list(env_var)
    return pinned if pinned else default


def chat_models(vendor: str, n: int = 1, env_var: Optional[str] = None) -> List[str]:
    """Up to *n* of *vendor*'s newest live general chat models."""
    default = _general_chat(live_models("CHAT", vendor))[:n]
    return _pinned_or(env_var, default) if env_var else default


def chat_model(vendor: str, env_var: Optional[str] = None) -> Optional[str]:
    """*vendor*'s newest live general chat model, or ``None``."""
    models = chat_models(vendor, 1, env_var)
    return models[0] if models else None


def family_models(
    pattern: str, n: Optional[int] = None, env_var: Optional[str] = None
) -> List[str]:
    """Live general chat models whose id matches the regex *pattern*.

    For tests that exercise one model family's behavior (e.g.
    ``r"^openai\\.gpt-oss-"``) rather than "any chat model from this vendor".
    """
    regex = re.compile(pattern)
    default = [m for m in _general_chat(live_models("CHAT")) if regex.search(m)]
    if n is not None:
        default = default[:n]
    return _pinned_or(env_var, default) if env_var else default


def one_chat_model_per_vendor(
    vendors: Sequence[str] = VENDORS, env_var: Optional[str] = None
) -> List[str]:
    """The newest live general chat model of each vendor in *vendors*."""
    default = [m for v in vendors for m in chat_models(v, 1)]
    return _pinned_or(env_var, default) if env_var else default


def reasoning_models(env_var: str = "OCI_REASONING_MODELS") -> List[str]:
    """Live models that return ``reasoning_content`` (xAI ``*-reasoning``)."""
    default = [
        n
        for n in _general_chat(live_models("CHAT", "xai"))
        if n.endswith("-reasoning") and not n.endswith("non-reasoning")
    ][:1]
    return _pinned_or(env_var, default)


def vision_models(env_var: str = "OCI_VISION_MODELS") -> List[str]:
    """Live models in the library's vision registry, one per vendor."""
    live = live_models("CHAT")
    seen, default = set(), []
    for name in live:
        vendor = name.split(".", 1)[0]
        if name in VISION_MODELS and vendor not in seen:
            seen.add(vendor)
            default.append(name)
    return _pinned_or(env_var, default)


def embedding_models(env_var: str = "OCI_EMBEDDING_MODELS") -> List[str]:
    """Live text-embedding models, newest first."""
    return _pinned_or(env_var, live_models("TEXT_EMBEDDINGS"))


def image_embedding_models(env_var: str = "OCI_IMAGE_EMBEDDING_MODELS") -> List[str]:
    """Live embedding models in the library's image-embedding registry."""
    default = [m for m in live_models("TEXT_EMBEDDINGS") if m in IMAGE_EMBEDDING_MODELS]
    return _pinned_or(env_var, default)


def rerank_model(env_var: str = "OCI_RERANK_MODEL_ID") -> Optional[str]:
    """The newest live rerank model, or ``None``."""
    pinned = env_list(env_var)
    if pinned:
        return pinned[0]
    live = live_models("TEXT_RERANK")
    return live[0] if live else None
