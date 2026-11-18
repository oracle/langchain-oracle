# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""Shared traversal of the OracleVS filter grammar for SQL and QBE renderers."""

from typing import Any, Callable, Optional, TypeVar

_Result = TypeVar("_Result")
LOGICAL_OPERATORS = frozenset({"$and", "$or", "$nor"})


def render_field(
    key: str,
    value: Any,
    field: Callable[[str, Any], _Result],
    combine: Callable[[Optional[str], list[_Result]], _Result],
) -> _Result:
    """Flatten nested metadata names while retaining each leaf's item context."""
    if isinstance(value, dict) and value:
        operators = [k.startswith("$") for k in value]
        if any(operators) and not all(operators):
            raise ValueError(
                "Invalid filter: operator keys ($...) cannot be mixed with "
                f"nested metadata keys in {value!r}."
            )
        if not any(operators):
            return combine(
                None,
                [
                    render_field(f"{key}.{child}", operand, field, combine)
                    for child, operand in value.items()
                ],
            )
    return field(key, value)


def render_filter(
    filter: dict,
    field: Callable[[str, Any], _Result],
    combine: Callable[[Optional[str], list[_Result]], _Result],
) -> _Result:
    """Render logical structure once, using backend-specific leaf operations."""
    if not isinstance(filter, dict):
        raise ValueError("Filter syntax is incorrect. Must be a dictionary.")
    clauses = []
    for key, operand in filter.items():
        if key.startswith("$"):
            if key not in LOGICAL_OPERATORS:
                raise ValueError(f"'{key}' is not a recognized logical operator.")
            if not isinstance(operand, list):
                raise ValueError("Logical operators require an array of values.")
            clauses.append(
                combine(key, [render_filter(v, field, combine) for v in operand])
            )
        else:
            clauses.append(render_field(key, operand, field, combine))
    return combine(None, clauses)
