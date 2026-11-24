# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""Shared validation and normalization of OracleVS vector-index options."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Optional

from .utils import _quote_indentifier

_ORGANIZATIONS = {
    "HNSW": "HNSW",
    "INMEMORY GRAPH": "HNSW",
    "INMEMORY NEIGHBOR GRAPH": "HNSW",
    "IVF": "IVF",
    "PARTITIONS": "IVF",
    "NEIGHBOR PARTITIONS": "IVF",
}


def normalize_index_params(
    params: Optional[dict[str, Any]], *, metric: Optional[str] = None
) -> Optional[dict[str, Any]]:
    """Return OracleVS-style options, rejecting conflicting equivalent settings.

    Algorithm defaults are left to the backend after alias normalization. A
    missing or empty request stays missing or empty, preserving ensure/rebuild
    semantics. Backend-specific options remain available for backend validation.
    """
    if params is None:
        return None
    if not isinstance(params, dict):
        raise TypeError("Index parameters must be a dictionary")
    result = deepcopy(params)

    def merge(key: str, value: Any, source: str) -> None:
        if key in result and (
            type(result[key]) is not type(value) or result[key] != value
        ):
            raise ValueError(f"Conflicting index parameters: {key} and {source}")
        result[key] = value

    if "idx_type" in result:
        kind = result["idx_type"]
        if not isinstance(kind, str) or kind.strip().upper() not in {"HNSW", "IVF"}:
            raise ValueError("idx_type must be HNSW or IVF")
        result["idx_type"] = kind.strip().upper()
    if "organization" in result:
        organization = result.pop("organization")
        if not isinstance(organization, str):
            raise ValueError("organization must select HNSW or IVF")
        kind = _ORGANIZATIONS.get(" ".join(organization.upper().split()))
        if kind is None:
            raise ValueError("organization must be INMEMORY GRAPH or PARTITIONS")
        merge("idx_type", kind, "organization")

    for alias, key in (
        ("parallel_creation", "parallel"),
        ("partitions", "neighbor_part"),
    ):
        if alias in result:
            merge(key, result.pop(alias), alias)

    advanced = result.pop("advanced_params", None)
    if advanced is not None:
        if not isinstance(advanced, dict):
            raise TypeError("advanced_params must be a dictionary")
        for nested, key in (
            ("neighbors", "neighbors"),
            ("efConstruction", "efConstruction"),
            ("partitions", "neighbor_part"),
        ):
            if nested in advanced:
                merge(key, advanced.pop(nested), f"advanced_params.{nested}")
        if advanced:
            result["advanced_params"] = advanced

    kind = result.get("idx_type", "HNSW")
    if kind == "HNSW" and "neighbor_part" in result:
        raise ValueError(
            "partitions/neighbor_part require idx_type=IVF or organization=PARTITIONS"
        )
    if kind == "IVF" and any(key in result for key in ("neighbors", "efConstruction")):
        raise ValueError("neighbors and efConstruction require an HNSW index")

    if "distance_metric" in result:
        distance = result["distance_metric"]
        if not isinstance(distance, str):
            raise ValueError("distance_metric must be a string")
        distance = distance.strip().upper()
        result["distance_metric"] = distance
        if metric is not None and distance != metric:
            raise ValueError("Index distance_metric must match distance_strategy")
    return result


def constructor_index_params(
    params: Optional[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    """Validate outer options and extract the optional nested index request."""
    if params is None:
        return None
    if not isinstance(params, dict):
        raise TypeError("OracleVS params must be a dictionary")
    unknown = set(params) - {"vector_index_params"}
    if unknown:
        raise ValueError(
            f"Unsupported OracleVS params: {sorted(unknown)}; "
            'use params={"vector_index_params": {...}} for vector index options'
        )
    return normalize_index_params(params.get("vector_index_params"))


def _validate_int_param(
    config: dict[str, Any],
    key: str,
    min_value: int,
    max_value: Optional[int] = None,
) -> None:
    if key not in config:
        return

    value = config[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer.")
    if value < min_value:
        raise ValueError(f"{key} must be at least {min_value}.")
    if max_value is not None and value > max_value:
        raise ValueError(f"{key} must be at most {max_value}.")


def _validate_index_type(config: dict[str, Any], expected_type: str) -> None:
    if "idx_type" not in config:
        return

    idx_type = config["idx_type"]
    if not isinstance(idx_type, str) or idx_type.upper() != expected_type:
        raise ValueError(f"idx_type must be {expected_type}.")
    config["idx_type"] = expected_type


def _validate_vector_index_common(config: dict[str, Any]) -> None:
    config["idx_name"] = _quote_indentifier(config["idx_name"])
    _validate_int_param(config, "accuracy", 1, 100)
    _validate_int_param(config, "parallel", 1)
