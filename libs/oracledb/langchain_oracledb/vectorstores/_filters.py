# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""OracleVS filter traversal and SQL predicates shared across backends."""

import json
import re
from typing import Any, Callable, List, Optional, TypeVar

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


LOGICAL_MAP = {
    "$and": (" AND ", "({0})"),
    "$or": (" OR ", "({0})"),
    "$nor": (" OR ", "( NOT ({0}) )"),
}


COMPARISON_MAP = {
    "$exists": "",
    "$eq": "@ == {0}",
    "$ne": "@ != {0}",
    "$gt": "@ > {0}",
    "$lt": "@ < {0}",
    "$gte": "@ >= {0}",
    "$lte": "@ <= {0}",
    "$between": "",
    "$startsWith": "@  starts with {0}",
    "$hasSubstring": "@  has substring {0}",
    "$instr": "@  has substring {0}",
    "$regex": "@  like_regex {0}",
    "$like": "@  like {0}",
    "$in": "",
    "$nin": "",
    "$all": "",
    "$not": "",
}


NOT_OPERS = ["$nin", "$not", "$exists"]


def _comparison_operand(value: Any, bind_variables: List[str]) -> tuple[str, str, str]:
    if isinstance(value, bool):
        # SQL BOOLEAN passing values can retain a preceding True in cached
        # statements. These two fixed JSON literals avoid that problem while
        # preserving the implicit boolean conversion of the SQL contract.
        return "@.boolean()", json.dumps(value), ""
    index = len(bind_variables)
    bind_variables.append(value)
    return "@", f"$val{index}", f':value{index} as "val{index}"'


def _passing_clause(passings: str) -> str:
    return " PASSING " + passings if passings else ""


def _get_comparison_string(
    oper: str, value: Any, bind_variables: List[str]
) -> tuple[str, str]:
    if oper not in COMPARISON_MAP:
        raise ValueError(f"Invalid operator: {oper}")

    # usual two sided operator case
    if COMPARISON_MAP[oper] != "":
        item, operand, passing = _comparison_operand(value, bind_variables)
        return COMPARISON_MAP[oper].replace("@", item).format(operand), passing

    # between - needs two bindings
    elif oper == "$between":
        if not isinstance(value, List) or len(value) != 2:
            raise ValueError(
                f"Invalid value for $between: {value}. "
                "It must be a list containing exactly 2 elements."
            )

        min_val, max_val = value
        if min_val is None and max_val is None:
            raise ValueError("At least one bound in $between must be non-null.")

        conditions = []
        passings = []
        if min_val is not None:
            item, operand, passing = _comparison_operand(min_val, bind_variables)
            conditions.append(f"{item} >= {operand}")
            if passing:
                passings.append(passing)

        if max_val is not None:
            item, operand, passing = _comparison_operand(max_val, bind_variables)
            conditions.append(f"{item} <= {operand}")
            if passing:
                passings.append(passing)

        passing_bind = ",".join(passings)

        return " && ".join(conditions), passing_bind

    # in/nin/all needs N bindings
    elif oper in ["$in", "$nin", "$all"]:
        if not isinstance(value, List):
            raise ValueError(
                f"Invalid value for $in: {value}. It must be a non-empty list."
            )

        value_binds = []
        passings = []
        comparisons = []
        for val in value:
            item, operand, passing = _comparison_operand(val, bind_variables)
            value_binds.append(operand)
            comparisons.append(f"{item} == {operand}")
            if passing:
                passings.append(passing)

        passing_bind = ",".join(passings)
        condition = ""

        if any(isinstance(val, bool) for val in value) and (
            oper == "$all" or all(isinstance(val, bool) for val in value)
        ):
            separator = " && " if oper == "$all" else " || "
            condition = "(" + separator.join(comparisons) + ")"
        elif oper == "$all":
            condition = "@ == " + " && @ == ".join(value_binds)

        else:
            value_bind = ",".join(value_binds)
            condition = f"@ in ({value_bind})"

        return condition, passing_bind

    else:
        raise ValueError(f"Invalid operator: {oper}. ")


def _validate_metadata_key(metadata_key: str) -> None:
    # Allow letters, digits, underscore, dot, brackets, comma, *, space (for 'to')
    pattern = re.compile(r"[a-zA-Z0-9_\.\[\],\s\*]*")

    if not pattern.fullmatch(metadata_key):
        raise ValueError(
            f"Invalid metadata key '{metadata_key}'. "
            "Only letters, numbers, underscores, nesting via '.', "
            "and array wildcards '[*]' are allowed."
        )


def _generate_condition(
    metadata_key: str, value: Any, bind_variables: List[str]
) -> str:
    # single check inside a JSON_EXISTS
    SINGLE_MASK = (
        "JSON_EXISTS(metadata, '$.{key}?(@ {oper} $val)' "
        'PASSING {value_bind} AS "val")'
    )
    # combined checks with multiple operators and passing values
    MULTIPLE_MASK = "JSON_EXISTS(metadata, '$.{key}?({filters})'{passes})"

    _validate_metadata_key(metadata_key)

    if not isinstance(value, (dict, list, tuple)):
        # scalar-equality Clause
        if isinstance(value, bool):
            condition, _ = _get_comparison_string("$eq", value, bind_variables)
            return MULTIPLE_MASK.format(key=metadata_key, filters=condition, passes="")
        bind = f":value{len(bind_variables)}"
        bind_variables.append(value)
        return SINGLE_MASK.format(key=metadata_key, oper="==", value_bind=bind)

    elif isinstance(value, dict):
        # all values are filters
        result: str
        passings: str

        # comparison operator keys
        if all(value_key.startswith("$") for value_key in value.keys()):
            not_dict = {}

            passing_values = []
            comparison_values = []

            for k, v in value.items():
                # if need to negate, cannot combine in single JSON_EXISTS
                if (
                    k in NOT_OPERS
                    or (k == "$eq" and isinstance(v, (list, dict)))
                    or (k == "$ne" and isinstance(v, (list, dict)))
                ):
                    not_dict[k] = v
                    continue

                result, passings = _get_comparison_string(k, v, bind_variables)

                comparison_values.append(result)
                passing_values.append(passings)

            # combine all operators in a single JSON_EXISTS
            all_conditions = []
            if len(comparison_values) != 0:
                all_conditions.append(
                    MULTIPLE_MASK.format(
                        key=metadata_key,
                        filters=" && ".join(comparison_values),
                        passes=_passing_clause(
                            " , ".join(p for p in passing_values if p)
                        ),
                    )
                )

            # handle negated filters one by one, one JSON_EXISTS for each
            for k, v in not_dict.items():
                if k == "$not":
                    condition = _generate_condition(metadata_key, v, bind_variables)
                    all_conditions.append(f"NOT ({condition})")

                elif k == "$exists":
                    if not isinstance(v, bool):
                        raise ValueError(
                            f"Invalid value for $exists: {value}. "
                            "It must be a boolean (true or false)."
                        )

                    if v:
                        all_conditions.append(
                            f"JSON_EXISTS(metadata, '$.{metadata_key}')"
                        )
                    else:
                        all_conditions.append(
                            f"NOT (JSON_EXISTS(metadata, '$.{metadata_key}'))"
                        )

                elif k == "$nin":  # for now only $nin
                    result, passings = _get_comparison_string(k, v, bind_variables)

                    all_conditions.append(
                        " NOT "
                        + MULTIPLE_MASK.format(
                            key=metadata_key,
                            filters=result,
                            passes=_passing_clause(passings),
                        )
                    )

                elif k == "$eq":
                    bind_l = len(bind_variables)
                    bind_variables.append(json.dumps(v))

                    all_conditions.append(
                        f"JSON_EQUAL("
                        f"    JSON_QUERY(metadata, '$.{metadata_key}' ),"
                        f"    JSON(:value{bind_l})"
                        ")"
                    )

                elif k == "$ne":
                    bind_l = len(bind_variables)
                    bind_variables.append(json.dumps(v))

                    all_conditions.append(
                        f"NOT (JSON_EQUAL("
                        f"    JSON_QUERY(metadata, '$.{metadata_key}' ),"
                        f"    JSON(:value{bind_l})"
                        "))"
                    )

            res = " AND ".join(all_conditions)

            if len(all_conditions) > 1:
                return "(" + res + ")"

            return res

        else:

            def combine(operator: Optional[str], clauses: list[str]) -> str:
                result = " AND ".join(clauses)
                return "(" + result + ")" if len(clauses) > 1 else result

            return render_field(
                metadata_key,
                value,
                lambda key, operand: _generate_condition(key, operand, bind_variables),
                combine,
            )

    else:
        raise ValueError("Filter format is invalid.")


def _generate_where_clause(filter: dict, bind_variables: List[str]) -> str:
    def combine(operator: Optional[str], clauses: list[str]) -> str:
        if operator is not None:
            separator, template = LOGICAL_MAP[operator]
            return template.format(separator.join(clauses))
        result = " AND ".join(clauses)
        return "(" + result + ")" if len(clauses) > 1 else result

    return render_filter(
        filter,
        lambda key, value: _generate_condition(key, value, bind_variables),
        combine,
    )
