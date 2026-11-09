# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""Preserve OracleVS filter semantics at the VecDB package boundary."""

import math
import re
from copy import deepcopy
from typing import Any, Optional

from ._filters import render_field, render_filter


class SQLFilterRequired(ValueError):
    """A valid OracleVS predicate has no verified QBE translation."""


def _item_method(value: Any) -> str:
    if isinstance(value, bool):
        return "$boolean"
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise SQLFilterRequired("Unsupported VecDB filter nonfinite operand")
        return "$number"
    if value is None or isinstance(value, str):
        return "$string"
    raise SQLFilterRequired("Unsupported VecDB filter operand type")


def translate_filter(filter_: Any) -> Any:
    """Validate with OracleVS and translate without changing its JSON semantics.

    QBE defaults to strict typing and negated equality for $ne. Explicit item
    methods restore SQL's coercion and existential inequality. Negation belongs
    outside the complete positive predicate, including its existence test.
    """
    from .oraclevs import _generate_where_clause

    if filter_ is None:
        return None
    _generate_where_clause(filter_, [])

    def unsupported(key: str, operator: str) -> Any:
        raise SQLFilterRequired(f"Unsupported VecDB filter {key} {operator}")

    def combine(clauses: list[dict]) -> dict:
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}

    def array_equal(key: str, operand: list) -> dict:
        # Index guards must refer to one array, never different parents reached
        # by an implicitly unwrapped nested path or an explicit wildcard.
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) or any(
            isinstance(v, (list, dict)) for v in operand
        ):
            return unsupported(key, "$eq")
        clauses: list[dict] = [
            {key: {"$type": "array"}},
            {key: {"$size": len(operand)}},
        ]
        for index, value in enumerate(operand):
            method = _item_method(value)
            path = f"{key}[{index}]"
            kind = "null" if value is None else method[1:]
            clauses.append({path: {"$type": kind}})
            if value is not None:
                clauses.append({path: {"$eq": deepcopy(value)}})
        return combine(clauses)

    def field(key: str, value: Any) -> dict:
        if not isinstance(value, dict):
            value = {"$eq": value}
        clauses: list[dict] = []
        positive: dict = {}
        methods: set[str] = set()
        for operator, operand in value.items():
            if operator == "$not":
                clauses.append({"$nor": [render_field(key, operand, field, group)]})
            elif operator == "$exists":
                clauses.append({key: {operator: operand}})
            elif operator == "$nin":
                clauses.append({"$nor": [field(key, {"$in": operand})]})
            elif operator in {"$eq", "$ne"} and isinstance(operand, (list, dict)):
                if operator != "$eq" or not isinstance(operand, list):
                    return unsupported(key, operator)
                clauses.append(array_equal(key, operand))
            else:
                values = (
                    operand if operator in {"$in", "$all", "$between"} else [operand]
                )
                if not values or any(isinstance(v, (list, dict)) for v in values):
                    return unsupported(key, operator)
                if operator in {
                    "$regex",
                    "$instr",
                    "$hasSubstring",
                    "$startsWith",
                    "$like",
                } and not isinstance(operand, str):
                    return unsupported(key, operator)
                if operator in {"$gt", "$gte", "$lt", "$lte", "$between"} and any(
                    isinstance(v, bool) or v is None
                    for v in values
                    if operator != "$between" or v is not None
                ):
                    return unsupported(key, operator)
                # A compound predicate needs one common item context. Different
                # conversions could correlate different array elements.
                methods.update(
                    _item_method(v)
                    for v in values
                    if operator != "$between" or v is not None
                )
                if operator == "$regex" and operand == "":
                    clauses.append(
                        {"$and": [{key: {"$exists": True}}, {key: {"$exists": False}}]}
                    )
                    continue
                if (
                    operator in {"$instr", "$hasSubstring", "$startsWith", "$like"}
                    and operand == ""
                ):
                    return unsupported(key, operator)
                mapped = {"$instr": "$hasSubstring", "$regex": "$likeRegex"}.get(
                    operator, operator
                )
                # OracleVS binds None as VARCHAR2; its SQL/JSON comparison has
                # the same behavior as the empty-string operand on this path.
                normalized = (
                    ["" if v is None else v for v in operand]
                    if operator in {"$in", "$all"}
                    else ""
                    if operand is None
                    else operand
                )
                positive[mapped] = deepcopy(normalized)
        if positive:
            if len(methods) > 1:
                return unsupported(key, "mixed comparison types")
            clauses.insert(0, {key: {next(iter(methods), "$string"): positive}})
        return combine(clauses) if clauses else unsupported(key, "empty predicate")

    def group(operator: Optional[str], clauses: list[dict]) -> dict:
        if operator is not None:
            return {operator: clauses}
        return combine(clauses) if clauses else {}

    return render_filter(filter_, field, group)
