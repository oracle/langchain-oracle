# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

# Adapted from oracle-vecdb: src/oracle_vecdb/services/ords/hybrid_search.py
# Source commit: cee35a4ee240dcf764d1258f15ac56a6d93c4129
# Request helpers verified against cace12d498e98b2e3d2dfd9b14bf0662f46838da;
# unused filter translation, configuration cache and index discovery are omitted.
# The adapter supplies the resolved index name and table embedding metadata.

"""JSON hybrid request validation and index planning copied from VecDB.

Transport execution is supplied by LangChain's adapter.
"""

from __future__ import annotations

import json
import math
import re
from copy import deepcopy
from typing import Any, Dict, Tuple


class HybridSearch:
    """Transport-independent subset of the SDK's JSON hybrid implementation."""

    def _describe_table_embed_params(self, table_name: str) -> Dict[str, Any]:
        raise NotImplementedError("The adapter must supply table embedding metadata")

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
        hybrid_options,
    ):
        self._positive_integer(top_k, "top_k")
        embed = self._describe_table_embed_params(table_name)
        path = self._json_path_literal(
            self._require_hybrid_embed_params(embed, "query")["embed_metadata_jsonpath"]
        )
        request = deepcopy(hybrid_options)
        if not isinstance(request, dict):
            raise TypeError("hybrid_options must be a dictionary.")
        request["hybrid_index_name"] = self._identifier(
            request.get("hybrid_index_name"), "hybrid_index_name"
        )
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
