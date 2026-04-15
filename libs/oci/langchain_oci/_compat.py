"""Compatibility patches for upstream LangChain/LangGraph edge cases.

These patches are temporary workarounds for upstream schema generation bugs
in langgraph's pydantic integration. They only activate when the upstream
code raises specific known exceptions, and are no-ops when not needed.

The patches should be removed once the upstream fixes land. Track:
- langgraph: OmitFromSchema + NotRequired annotation handling
- deepagents: tool schema generation with injected runtime fields
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, cast

from pydantic import BaseModel, create_model

logger = logging.getLogger(__name__)


def _build_loose_schema(
    model_name: str,
    field_definitions: Optional[Dict[str, Any]] = None,
    root: Optional[Any] = None,
) -> type[BaseModel]:
    """Build a permissive model when upstream schema generation fails."""
    if root is not None:
        return create_model(model_name, root=(Any, None))

    fields = {name: (Any, None) for name in (field_definitions or {})} or {
        "output": (Any, None)
    }
    return cast(
        type[BaseModel],
        create_model(model_name, **fields),  # type: ignore[call-overload]
    )


def _needs_patch() -> bool:
    """Check if the current langgraph version needs the schema patch."""
    try:
        from langgraph._internal import _pydantic as langgraph_pydantic
    except ImportError:
        return False

    if not hasattr(langgraph_pydantic, "create_model"):
        return False

    # Already patched
    if getattr(langgraph_pydantic.create_model, "_langchain_oci_patched", False):
        return False

    return True


def apply_compat_patches() -> None:
    """Patch upstream schema generation bugs used by deepagents/langgraph.

    This keeps normal behavior intact and only falls back to permissive `Any`
    fields when langgraph cannot materialize a model because of unsupported
    schema annotations such as `OmitFromSchema` + `NotRequired`.
    """
    if not _needs_patch():
        return

    try:
        from langgraph._internal import _pydantic as langgraph_pydantic
    except ImportError:
        return

    original_create_model = langgraph_pydantic.create_model

    def patched_create_model(
        model_name: str,
        *,
        field_definitions: Optional[Dict[str, Any]] = None,
        root: Optional[Any] = None,
    ) -> type[BaseModel]:
        try:
            return original_create_model(
                model_name,
                field_definitions=field_definitions,
                root=root,
            )
        except Exception as ex:
            if ex.__class__.__name__ not in {
                "PydanticForbiddenQualifier",
                "ForbiddenQualifier",
                "TypeError",
            }:
                raise
            logger.debug(
                "langgraph schema fallback for %s: %s",
                model_name,
                ex.__class__.__name__,
            )
            return _build_loose_schema(
                model_name,
                field_definitions=field_definitions,
                root=root,
            )

    patched_create_model._langchain_oci_patched = True  # type: ignore[attr-defined]
    langgraph_pydantic.create_model = patched_create_model

    # Also patch modules that imported create_model directly before us,
    # since `from X import create_model` copies the reference.
    import sys

    for mod_name in ("langgraph.pregel.main", "langgraph.pregel"):
        mod = sys.modules.get(mod_name)
        if mod and getattr(mod, "create_model", None) is original_create_model:
            mod.create_model = patched_create_model  # type: ignore[attr-defined]
