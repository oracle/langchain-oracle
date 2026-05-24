# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Shared agent configuration and LLM builder."""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Optional, Union, cast

from pydantic import BaseModel, Field, create_model, model_validator

from langchain_oci.common.auth import OCIAuthType

logger = logging.getLogger(__name__)


class AgentConfig(BaseModel):
    """Base configuration shared by all OCI agents."""

    model_config = {"populate_by_name": True, "arbitrary_types_allowed": True}

    model_id: str = "meta.llama-4-scout-17b-16e-instruct"
    compartment_id: Optional[str] = None
    service_endpoint: Optional[str] = None
    auth_type: Union[str, OCIAuthType] = OCIAuthType.API_KEY
    auth_profile: str = "DEFAULT"
    auth_file_location: str = "~/.oci/config"
    system_prompt: Optional[str] = None
    checkpointer: Optional[Any] = None
    store: Optional[Any] = None
    interrupt_before: Optional[List[str]] = None
    interrupt_after: Optional[List[str]] = None
    debug: bool = False
    name: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _resolve_oci_env(self) -> "AgentConfig":
        """Resolve compartment and endpoint from environment if not set."""
        if not self.compartment_id:
            self.compartment_id = os.environ.get("OCI_COMPARTMENT_ID")
        if not self.compartment_id:
            raise ValueError(
                "compartment_id must be provided or set via "
                "OCI_COMPARTMENT_ID environment variable"
            )
        if not self.service_endpoint:
            self.service_endpoint = os.environ.get("OCI_SERVICE_ENDPOINT")
        if not self.service_endpoint:
            region = os.environ.get("OCI_REGION", "us-chicago-1")
            self.service_endpoint = (
                f"https://inference.generativeai.{region}.oci.oraclecloud.com"
            )
        return self


def _build_llm(config: AgentConfig, **extra_kwargs: Any) -> Any:
    """Build a ChatOCIGenAI instance from an AgentConfig.

    Args:
        config: Resolved agent configuration.
        **extra_kwargs: Additional kwargs passed to ChatOCIGenAI
            (e.g. max_sequential_tool_calls, tool_result_guidance).

    Returns:
        ChatOCIGenAI instance.
    """
    from langchain_oci.chat_models.oci_generative_ai import ChatOCIGenAI

    auth_type = (
        config.auth_type.name
        if isinstance(config.auth_type, OCIAuthType)
        else config.auth_type
    )

    merged_kwargs = {**config.model_kwargs}
    if config.temperature is not None:
        merged_kwargs["temperature"] = config.temperature
    if config.max_tokens is not None:
        # Provider-layer normalize_params remaps to max_completion_tokens for
        # OpenAI models, so agent-level config stays model-agnostic.
        merged_kwargs["max_tokens"] = config.max_tokens

    return ChatOCIGenAI(
        model_id=config.model_id,
        compartment_id=config.compartment_id or "",
        service_endpoint=config.service_endpoint or "",
        auth_type=auth_type,
        auth_profile=config.auth_profile,
        auth_file_location=config.auth_file_location,
        model_kwargs=merged_kwargs or None,
        **extra_kwargs,
    )


def _get_agent_factory() -> tuple[Callable[..., Any], bool]:
    """Get the appropriate agent factory function.

    Returns:
        Tuple of (factory_function, is_legacy_api).
        is_legacy_api is True when using langgraph.prebuilt.create_react_agent.
    """
    try:
        from langchain.agents import create_agent

        return create_agent, False
    except (ImportError, AttributeError):
        pass

    try:
        from langgraph.prebuilt import create_react_agent

        return create_react_agent, True
    except ImportError as ex:
        raise ImportError(
            "Could not import agent creation function. "
            "Please install langchain>=1.0.0 or langgraph."
        ) from ex


def _filter_none(**kwargs: Any) -> dict[str, Any]:
    """Filter out None values from keyword arguments."""
    return {k: v for k, v in kwargs.items() if v is not None}


# Exception types raised by pydantic for the OmitFromSchema + NotRequired combo
# that deepagents middleware attaches to injected runtime fields. Resolved at
# module load so the fallback catches real types rather than class-name strings.
_SCHEMA_FALLBACK_EXCEPTIONS: tuple = (TypeError,)
try:
    from pydantic import PydanticForbiddenQualifier

    _SCHEMA_FALLBACK_EXCEPTIONS = (PydanticForbiddenQualifier, TypeError)
except ImportError:
    pass


def _build_loose_model(
    model_name: str,
    field_definitions: Optional[Dict[str, Any]] = None,
    root: Optional[Any] = None,
) -> type[BaseModel]:
    """Build a permissive pydantic model when strict creation fails."""
    if root is not None:
        return create_model(model_name, root=(Any, None))
    fields = {name: (Any, None) for name in (field_definitions or {})} or {
        "output": (Any, None)
    }
    return cast(
        type[BaseModel],
        create_model(model_name, **fields),  # type: ignore[call-overload]
    )


@contextmanager
def _langgraph_schema_fallback() -> Iterator[None]:
    """Scope-limited workaround for a pydantic/langgraph schema bug.

    Deepagents' tool middleware attaches ``OmitFromSchema + NotRequired``
    annotations to runtime-injected fields. Pydantic refuses that combo, so
    langgraph's ``create_model`` blows up while compiling the agent graph.

    This context manager replaces ``langgraph._internal._pydantic.create_model``
    with a variant that falls back to a permissive ``Any`` schema when it hits
    the specific exceptions, and restores the original on exit. Scope is
    intentionally tight: the patch is only live for the wrapped call, not a
    global import-time side effect.

    Does nothing (and never raises) on langgraph versions where the internal
    module layout doesn't match; the call executes against the real upstream
    ``create_model`` in that case.
    """
    try:
        from langgraph._internal import _pydantic as lg_pydantic
    except ImportError:
        yield
        return

    original = getattr(lg_pydantic, "create_model", None)
    if original is None:
        yield
        return

    def patched(
        model_name: str,
        *,
        field_definitions: Optional[Dict[str, Any]] = None,
        root: Optional[Any] = None,
    ) -> type[BaseModel]:
        try:
            return original(
                model_name,
                field_definitions=field_definitions,
                root=root,
            )
        except _SCHEMA_FALLBACK_EXCEPTIONS as ex:
            logger.debug(
                "langgraph schema fallback for %s: %s",
                model_name,
                ex.__class__.__name__,
            )
            return _build_loose_model(
                model_name,
                field_definitions=field_definitions,
                root=root,
            )

    # Swap the source of truth plus any modules that did `from X import create_model`
    # (those capture the reference at import time).
    import sys

    rebound = []
    lg_pydantic.create_model = patched
    for mod_name in ("langgraph.pregel.main", "langgraph.pregel"):
        mod = sys.modules.get(mod_name)
        if mod is not None and getattr(mod, "create_model", None) is original:
            mod.create_model = patched  # type: ignore[attr-defined]
            rebound.append(mod)

    try:
        yield
    finally:
        lg_pydantic.create_model = original
        for mod in rebound:
            mod.create_model = original  # type: ignore[attr-defined]
