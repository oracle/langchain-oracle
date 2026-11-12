# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

# Adapted from oracle-vecdb: src/oracle_vecdb/services/ords/hybrid_search.py
# Source commit: cee35a4ee240dcf764d1258f15ac56a6d93c4129

"""JSON hybrid request validation and index planning copied from VecDB.

Transport execution is supplied by LangChain's adapter.
"""

from __future__ import annotations

import json
import math
import re
from copy import deepcopy
from typing import Any, Dict, List, Tuple


class HybridSearch:
    """Transport-independent subset of the SDK's JSON hybrid implementation."""

    def __init__(self) -> None:
        self._hybrid_index_configs: Dict[str, Dict[str, Any]] = {}

    def _describe_table_embed_params(self, table_name: str) -> Dict[str, Any]:
        raise NotImplementedError("The adapter must supply table embedding metadata")

    def _hybrid_index_rows_for_table(self, table_name: str) -> List[Dict[str, Any]]:
        raise NotImplementedError("The adapter must supply hybrid index discovery")

    @staticmethod
    def _require_hybrid_embed_params(
        embed_params: Any, operation: str
    ) -> Dict[str, Any]:
        if not isinstance(embed_params, dict):
            raise ValueError(
                f"Hybrid index {operation} requires embed_params with "
                "'model' and 'embed_metadata_jsonpath'."
            )
        model = embed_params.get("model")
        metadata_path = embed_params.get("embed_metadata_jsonpath")
        if not isinstance(model, str) or not model.strip():
            raise ValueError(
                f"Hybrid index {operation} requires embed_params['model']."
            )
        if not isinstance(metadata_path, str) or not metadata_path.strip():
            raise ValueError(
                f"Hybrid index {operation} requires "
                "embed_params['embed_metadata_jsonpath']."
            )
        HybridSearch._json_path_literal(metadata_path)
        return deepcopy(embed_params)

    @staticmethod
    def _hybrid_search_input(query_by: Dict[str, Any]) -> Tuple[str, Any]:
        if not isinstance(query_by, dict):
            raise TypeError("query_by must be a dictionary.")
        if set(query_by) == {"text"}:
            text = query_by.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    "Hybrid text query requires non-empty query_by['text']."
                )
            return "text", text.strip()
        if set(query_by) == {"vector"}:
            vector = query_by.get("vector")
            if not isinstance(vector, (list, tuple)) or not vector:
                raise ValueError(
                    "Hybrid vector query requires a non-empty query_by['vector'] list."
                )
            normalized = []
            for value in vector:
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                ):
                    raise ValueError(
                        "query_by['vector'] must contain only finite numeric values."
                    )
                normalized.append(value)
            return "vector", normalized
        raise ValueError(
            "Hybrid query requires exactly one of query_by={'text': '...'} "
            "or query_by={'vector': [...]}."
        )

    def _hybrid_search_request(
        self,
        table_name,
        query_kind,
        query_value,
        top_k,
        filters,
        advanced_options,
        hybrid_options,
    ):
        self._positive_integer(top_k, "top_k")
        if advanced_options:
            raise ValueError(
                "Use hybrid_options for DBMS_HYBRID_VECTOR.SEARCH parameters."
            )
        config = self._hybrid_index_configs.get(table_name.upper()) or {}
        embed = config.get("embed_params") or self._describe_table_embed_params(
            table_name
        )
        path = self._json_path_literal(
            self._require_hybrid_embed_params(embed, "query")["embed_metadata_jsonpath"]
        )
        request = deepcopy(hybrid_options)
        if not isinstance(request, dict):
            raise TypeError("hybrid_options must be a dictionary.")
        index_name = request.get("hybrid_index_name") or config.get("hybrid_index_name")
        if not index_name:
            rows = self._hybrid_index_rows_for_table(table_name)
            if len(rows) != 1:
                raise ValueError(
                    "Specify hybrid_index_name when the table does not have "
                    "exactly one HVI."
                )
            index_name = rows[0].get("IDX_NAME") or rows[0].get("idx_name")
        request["hybrid_index_name"] = self._identifier(index_name, "hybrid_index_name")
        returns = request.setdefault("return", {})
        if not isinstance(returns, dict):
            raise TypeError("return must be a dictionary.")
        returns.setdefault("topN", top_k)
        returns.setdefault("format", "JSON")
        returns.setdefault(
            "values",
            [
                "rowid",
                "score",
                "vector_score",
                "text_score",
                "chunk_text",
                "chunk_id",
            ],
        )
        if str(returns["format"]).upper() != "JSON":
            raise ValueError("The SDK result wrapper requires return.format=JSON.")
        self._positive_integer(returns["topN"], "return.topN")
        text_options = request.setdefault("text", {})
        vector_options = request.setdefault("vector", {})
        if not isinstance(text_options, dict) or not isinstance(vector_options, dict):
            raise TypeError("text and vector options must be dictionaries.")
        text_options.setdefault("inpath", [path])
        self._validate_json_path_list(text_options["inpath"], "text.inpath")
        if "contains" in request:
            if any(
                k in text_options
                for k in ("contains", "search_text", "json_textcontains")
            ):
                raise ValueError("Conflicting text search inputs.")
            text_options["contains"] = request.pop("contains")
        if query_kind == "text":
            if "search_text" not in request:
                if not any(
                    k in text_options
                    for k in ("contains", "search_text", "json_textcontains")
                ):
                    text_options["search_text"] = query_value
                if not any(
                    k in vector_options for k in ("search_text", "search_vector")
                ):
                    vector_options["search_text"] = query_value
        else:
            if "search_text" in request or any(
                k in vector_options for k in ("search_text", "search_vector")
            ):
                raise ValueError("Conflicting vector query inputs.")
            vector_options["search_vector"] = json.dumps(query_value)
        if filters:
            if "filter_by" in request:
                raise ValueError("Use filters or filter_by, not both.")
            request["filter_by"] = self._hybrid_filter(filters)
        # All other SEARCH parameters, including new nested parameters, pass through.
        return request

    @staticmethod
    def _hybrid_json_textcontains(value: Any) -> None:
        if not isinstance(value, list) or not value:
            raise ValueError("text.json_textcontains must be a non-empty list.")

    @staticmethod
    def _validate_json_path_list(value: Any, name: str) -> None:
        if not isinstance(value, list) or not value:
            raise ValueError(f"{name} must be a non-empty list of JSON paths.")
        for item in value:
            if not isinstance(item, str) or not item.strip().startswith("$"):
                raise ValueError(f"{name} values must be JSON path strings.")

    @staticmethod
    def _positive_integer(value: Any, name: str) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer greater than 0.")
        return value

    @classmethod
    def _hybrid_filter(cls, filters: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(filters, dict) or not filters:
            raise ValueError("filters must be a non-empty dictionary.")
        return cls._hybrid_filter_expr(filters)

    @classmethod
    def _hybrid_filter_expr(cls, expr: Any) -> Dict[str, Any]:
        if not isinstance(expr, dict) or not expr:
            raise ValueError("Hybrid filters must be non-empty dictionaries.")
        if len(expr) == 1:
            key, value = next(iter(expr.items()))
            logical_ops = {
                "$and": "AND",
                "$or": "OR",
                "$not": "NOT",
                "$notand": "NOTAND",
                "$notor": "NOTOR",
            }
            if key in logical_ops:
                return cls._hybrid_logical_filter(logical_ops[key], value)
        if any(str(key).startswith("$") for key in expr):
            raise ValueError(
                "Logical filters must use one of $and, $or, $not, $notand, $notor."
            )
        parts = [cls._hybrid_field_filter(key, value) for key, value in expr.items()]
        if len(parts) == 1:
            return parts[0]
        return {"op": "AND", "args": parts}

    @classmethod
    def _hybrid_logical_filter(cls, op: str, value: Any) -> Dict[str, Any]:
        if op == "NOT":
            return {"op": op, "args": [cls._hybrid_filter_expr(value)]}
        if not isinstance(value, list) or not value:
            raise ValueError(f"${op.lower()} requires a non-empty list of filters.")
        return {
            "op": op,
            "args": [cls._hybrid_filter_expr(item) for item in value],
        }

    @classmethod
    def _hybrid_field_filter(cls, key: Any, condition: Any) -> Dict[str, Any]:
        if not isinstance(key, str) or not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*", key
        ):
            raise ValueError(
                "Preview hybrid filter field must be a simple metadata key "
                "or dot-separated metadata path."
            )
        path = f"CONTENT_METADATA.{key}"
        if not isinstance(condition, dict):
            return cls._hybrid_comparison_filter(path, "=", condition)
        if not condition:
            raise ValueError(f"Filter condition for {key!r} cannot be empty.")
        if len(condition) != 1:
            nested = [
                cls._hybrid_field_filter(key, {op: value})
                for op, value in condition.items()
            ]
            return {"op": "AND", "args": nested}
        operator, value = next(iter(condition.items()))
        op_map = {
            "$eq": "=",
            "$ne": "!=",
            "$gt": ">",
            "$gte": ">=",
            "$lt": "<",
            "$lte": "<=",
            "$like": "LIKE",
            "$regexp_like": "REGEXP_LIKE",
            "$between": "BETWEEN",
            "$in": "IN",
            "$exists": "EXISTS",
        }
        if operator not in op_map:
            raise ValueError(
                "Unsupported hybrid filter operator. Supported operators are "
                f"{sorted(op_map)}."
            )
        return cls._hybrid_comparison_filter(path, op_map[operator], value)

    @classmethod
    def _hybrid_comparison_filter(
        cls, path: str, op: str, value: Any
    ) -> Dict[str, Any]:
        if op == "EXISTS":
            if not isinstance(value, bool):
                raise ValueError("$exists requires a boolean value.")
            exists_filter: Dict[str, Any] = {"op": "EXISTS", "path": path, "args": []}
            return exists_filter if value else {"op": "NOT", "args": [exists_filter]}
        if op == "BETWEEN":
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError("$between requires exactly two values.")
            args = list(value)
        elif op == "IN":
            if not isinstance(value, (list, tuple)) or not value:
                raise ValueError("$in requires a non-empty list of values.")
            args = list(value)
        else:
            args = [value]
        return {
            "op": op,
            "type": cls._hybrid_filter_type(args[0] if args else value),
            "path": path,
            "args": args,
        }

    @staticmethod
    def _hybrid_filter_type(value: Any) -> str:
        if isinstance(value, bool):
            return "string"
        if isinstance(value, (int, float)):
            return "number"
        return "string"

    @staticmethod
    def _identifier(value: Any, name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string.")
        identifier = value.strip().upper()
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", identifier):
            raise ValueError(f"{name} must be a simple Oracle identifier.")
        return identifier

    @classmethod
    def _json_path_literal(cls, raw_path: Any) -> str:
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError("Hybrid index creation requires embed_metadata_jsonpath.")
        path = raw_path.strip()
        if path.startswith("$."):
            path = path[2:]
        elif path.startswith("$"):
            raise ValueError(
                "Only simple JSON paths like text or $.text are supported."
            )
        parts = path.split(".")
        if any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", part) for part in parts):
            raise ValueError("JSON path must contain dot-separated identifier keys.")
        return "$" + "".join(f".{part}" for part in parts)

    @staticmethod
    def _create_hybrid_index_statement(
        index_name: str,
        table_identifier: str,
        text_column: str,
        parameters: str,
        parallel_clause: str = "",
    ) -> str:
        """Extracted from _hybrid_index_plan for caller-managed preferences."""
        escaped = parameters.replace("'", "''")
        return (
            f"CREATE HYBRID VECTOR INDEX {index_name} "
            f"ON {table_identifier}({text_column}) "
            f"PARAMETERS ('{escaped}'){parallel_clause}"
        )
