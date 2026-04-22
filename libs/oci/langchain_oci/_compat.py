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
from typing import Any, Dict, Optional, Tuple, Type, cast

from pydantic import BaseModel, create_model

logger = logging.getLogger(__name__)

# Exception types that indicate the upstream pydantic-schema bug this module
# works around. Imported at module load so the patch catches the real types
# instead of matching on class name strings.
_SCHEMA_FALLBACK_EXCEPTIONS: Tuple[Type[BaseException], ...] = (TypeError,)
try:
    from pydantic import PydanticForbiddenQualifier

    _SCHEMA_FALLBACK_EXCEPTIONS = (PydanticForbiddenQualifier, TypeError)
except ImportError:
    # Older pydantic versions don't expose PydanticForbiddenQualifier at the
    # top level — fall back to TypeError, which is what those versions raise
    # for the same unsupported annotations.
    pass

# Upper bound on langgraph versions this patch targets. Kept deliberately
# coarse: bump when we verify the upstream fix has shipped so the patch
# self-expires rather than silently re-patching future internals.
_LANGGRAPH_MAX_PATCHED_VERSION = (1, 99, 99)

# Modules that reimport langgraph._internal._pydantic.create_model at import
# time (via `from X import create_model`), so patching the source is not
# enough — we must also rebind their copy.
_DOWNSTREAM_REFERENCE_MODULES = ("langgraph.pregel.main", "langgraph.pregel")


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


def _langgraph_version_in_range() -> bool:
    """Return True if the installed langgraph version is one we patch."""
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:
        return True  # very old python — be permissive
    try:
        raw = version("langgraph")
    except PackageNotFoundError:
        return False
    try:
        parts = tuple(int(p) for p in raw.split(".")[:3])
    except ValueError:
        return True  # non-standard version string — assume in range
    return parts <= _LANGGRAPH_MAX_PATCHED_VERSION


def _needs_patch() -> bool:
    """Check if the current langgraph install needs the schema patch."""
    try:
        from langgraph._internal import _pydantic as langgraph_pydantic
    except ImportError:
        return False

    if not hasattr(langgraph_pydantic, "create_model"):
        return False

    # Already patched
    if getattr(langgraph_pydantic.create_model, "_langchain_oci_patched", False):
        return False

    return _langgraph_version_in_range()


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
        except _SCHEMA_FALLBACK_EXCEPTIONS as ex:
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

    # Also rebind modules that imported create_model directly before us,
    # since `from X import create_model` copies the reference. If a target
    # module is present but its create_model attr has diverged from the one
    # we just replaced, langgraph's internal layout has likely moved and the
    # patch is no longer sufficient — surface that loudly so we can update
    # _LANGGRAPH_MAX_PATCHED_VERSION.
    import sys

    for mod_name in _DOWNSTREAM_REFERENCE_MODULES:
        mod = sys.modules.get(mod_name)
        if mod is None:
            continue
        current = getattr(mod, "create_model", None)
        if current is original_create_model:
            mod.create_model = patched_create_model  # type: ignore[attr-defined]
        elif current is not patched_create_model:
            logger.warning(
                "langchain-oci compat patch could not rebind create_model in "
                "%s (langgraph internals appear to have moved); deep-agent "
                "schema fallback may not apply here.",
                mod_name,
            )
