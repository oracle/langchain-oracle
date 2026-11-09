# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""Preserve OracleVS filter semantics at the VecDB package boundary."""

from copy import deepcopy
from typing import Any


def translate_filter(filter_: Any) -> Any:
    """Validate with OracleVS, then translate the proven QBE-compatible subset.

    Keep this outside the copied SDK: its QBE grammar is a different public API.
    Never approximate an unsupported OracleVS predicate with a broader one.
    """
    from .oraclevs import _generate_where_clause

    if filter_ is None:
        return None
    _generate_where_clause(filter_, [])

    def field(key: str, value: Any) -> dict:
        if not isinstance(value, dict):
            return {key: deepcopy(value)}
        if value and not any(k.startswith("$") for k in value):
            return combine([field(f"{key}.{k}", v) for k, v in value.items()])
        mapped = {}
        for operator, operand in value.items():
            reason = None
            if operator in {"$eq", "$ne"} and isinstance(operand, (list, dict)):
                reason = "exact array/object equality requires package support"
            elif operator == "$ne":
                reason = "existential inequality differs for arrays and missing fields"
            elif operator == "$not":
                reason = "field negation and type coercion require package support"
            elif operator == "$regex":
                reason = "OracleVS like_regex differs from the package regex dialect"
            if reason:
                raise ValueError(f"Unsupported VecDB filter {key} {operator}: {reason}")
            mapped["$hasSubstring" if operator == "$instr" else operator] = deepcopy(
                operand
            )
        return {key: mapped}

    def combine(clauses: list[dict]) -> dict:
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}

    def expression(value: dict) -> dict:
        clauses = []
        for key, operand in value.items():
            if key.startswith("$"):
                clauses.append({key: [expression(item) for item in operand]})
            else:
                clauses.append(field(key, operand))
        return combine(clauses) if clauses else {}

    return expression(filter_)
