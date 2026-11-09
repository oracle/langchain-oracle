# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Adapt branch-derived VecDB requests and responses to LangChain.

The copied SDK modules own parameter schemas and SQL templates. This module
owns database-generation compatibility, output binds, and Document mapping.
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from decimal import Decimal
from typing import Any

from langchain_core.documents import Document

from .._vecdb_sdk.parameter_validation import validate_operation_arguments
from .._vecdb_sdk.services.ords.hybrid_search import HybridSearch
from .._vecdb_sdk.services.sqlnet.query_mapper import SQLNetQueryMapper
from ._vecdb_filters import translate_filter

# Internal document body field; user metadata remains flat.
TEXT_KEY = "lc_oracle_content"
CURRENT_VERSION = (23, 26, 3)


def as_dict(response: Any) -> dict[str, Any]:
    if hasattr(response, "read"):
        response = response.read()
    if isinstance(response, str):
        response = json.loads(response, parse_int=Decimal, parse_float=Decimal)
    if hasattr(response, "to_dict"):
        response = response.to_dict()
    elif hasattr(response, "model_dump"):
        response = response.model_dump()
    if not isinstance(response, dict):
        raise ValueError("Malformed VecDB response: expected a JSON object")
    return {key.lower(): value for key, value in response.items()}


def rows(response: Any, key: str) -> list[dict[str, Any]]:
    obj = as_dict(response)
    result = obj.get(key, obj.get("items"))
    if not isinstance(result, list):
        raise ValueError(f"Malformed VecDB response: missing {key} array")
    return [as_dict(item) for item in result]


def _metadata_numbers(value: Any) -> Any:
    """Match native SQL JSON NUMBER decoding without changing booleans."""
    if isinstance(value, dict):
        return {key: _metadata_numbers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_metadata_numbers(item) for item in value]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return Decimal(str(value))
    return value


def document(row: dict[str, Any]) -> Document:
    from .oraclevs import INTERNAL_ID_KEY

    metadata = _metadata_numbers(dict(row.get("metadata") or {}))
    text = metadata.pop(TEXT_KEY, None)
    if not isinstance(text, str) or not isinstance(row.get("id"), str):
        raise ValueError("VecDB document is missing its ID or metadata text")
    # The SQL backend's private ID field is not part of public metadata.
    metadata.pop(INTERNAL_ID_KEY, None)
    return Document(id=row["id"], page_content=text, metadata=metadata)


def _hybrid_documents(
    result: list, scores: bool, *, keyword_only: bool = False
) -> list[Document]:
    docs = []
    for id_, metadata, score, text_score, vector_score in result:
        doc = document({"id": str(id_), "metadata": metadata})
        if scores:
            doc.metadata.update(
                score=score,
                text_score=text_score,
                vector_score=0 if keyword_only else vector_score,
            )
        docs.append(doc)
    return docs


def _text_documents(result: list, scores: bool) -> list[Document]:
    docs = []
    for id_, metadata, score in result:
        doc = document({"id": str(id_), "metadata": metadata})
        if scores:
            doc.metadata["score"] = score
        docs.append(doc)
    return docs


def search_request(
    name: str,
    metric: str,
    embedding: list[float],
    k: int,
    filter: Any,
    include_vectors: bool,
    options: dict,
) -> dict:
    """Adapt the supported OracleVS search options to the SDK query contract."""
    options = dict(options)
    options.pop("db_filter", None)  # The public facade already resolved this alias.
    unknown = options.keys() - {"output_selector", "advanced_options"}
    if unknown:
        raise ValueError(
            f"Unsupported VecDB search options: {', '.join(sorted(unknown))}"
        )
    advanced = options.get("advanced_options")
    if advanced is None:
        advanced = {}
    if not isinstance(advanced, dict):
        raise ValueError("advanced_options must be a dictionary")
    if advanced.get("distance_metric", metric) != metric:
        raise ValueError(
            "advanced_options.distance_metric must match distance_strategy"
        )
    request = dict(
        table_name=name,
        query_by={"vector": embedding},
        top_k=k,
        filters=translate_filter(filter),
        include_vectors=include_vectors,
        advanced_options={**advanced, "distance_metric": metric},
    )
    selector = options.get("output_selector")
    if selector is not None:
        if not isinstance(selector, list) or not all(
            isinstance(key, str) for key in selector
        ):
            raise ValueError("output_selector must be a list of metadata key strings")
        # Selectors project metadata only. ID/distance/vector remain response fields.
        request["output_selector"] = list(dict.fromkeys([*selector, TEXT_KEY]))
    validate_operation_arguments("query", request)
    return request


class SQLNetAdapter:
    """Use SDK SQL templates with the store's output-bind execution contract."""

    def __init__(self, version: tuple[int, ...]):
        self.version = version

    def statement(self, operation: str, params: dict) -> tuple[str, dict]:
        method = "query" if operation == "search" else operation
        supported = {
            "create_vector_table",
            "describe_vector_table",
            "drop_vector_table",
            "upsert_vectors",
            "list_vectors",
            "query",
            "delete_vectors",
            "create_index",
            "rebuild_index",
        }
        if method not in supported:
            raise ValueError("Unsupported VecDB operation")
        params = dict(params)
        if method == "list_vectors":
            params.setdefault("limit", 100)
            params.setdefault("offset", 0)
        validate_operation_arguments(method, params)

        # The adapter emits JSON-returning functions for both generations.
        # Legacy create/drop names and create defaults differ; the copied
        # branch's upsert/drop OUT procedure templates are not used here.
        if method in {
            "create_vector_table",
            "drop_vector_table",
            "upsert_vectors",
            "rebuild_index",
        }:
            return self._function_statement(operation, params)

        mapped = dict(params)
        if "name" in mapped:
            mapped["table_name"] = mapped.pop("name")
        for key in (
            "ids",
            "query_by",
            "filters",
            "advanced_options",
            "output_selector",
            "index_params",
            "debug_flags",
        ):
            value = mapped.pop(key, None)
            mapped[key + "_json"] = (
                json.dumps(value, allow_nan=False) if value is not None else None
            )
        mapped.setdefault("include_vectors", False)
        query_name = (
            "query_with_output_selector"
            if method == "query" and params.get("output_selector") is not None
            else method
        )
        query = SQLNetQueryMapper.get_query(query_name)
        select = re.fullmatch(r"SELECT (.+) FROM dual", query, re.IGNORECASE)
        if select:
            query = f"BEGIN :result := {select[1]}; END;"
        else:
            # SDK templates discard their local CLOB; LangChain needs its value.
            query = re.sub(r"END;$", ":result := l_output; END;", query)
        required = set(re.findall(r":([a-z_]+)", query)) - {"result"}
        return query, {key: mapped.get(key) for key in required}

    def _function_statement(self, operation: str, params: dict) -> tuple[str, dict]:
        if operation in {"create_vector_table", "drop_vector_table"}:
            if self.version < CURRENT_VERSION:
                params["table_name"] = params.pop("name")
                if operation == "create_vector_table":
                    params.pop("table_params", None)
                    params.pop("embed_params", None)
                    params.update(
                        auto_generate_id=False,
                        vector_type="dense",
                        debug_flags=None,
                        request_id=None,
                    )
        arguments = []
        for key, value in params.items():
            if not re.fullmatch(r"[a-z_]+", key):
                raise ValueError("Invalid internal VecDB parameter")
            if key in {
                "vectors",
                "index_params",
                "table_params",
                "embed_params",
                "annotations",
                "debug_flags",
            }:
                params[key] = (
                    json.dumps(value, allow_nan=False) if value is not None else None
                )
                arguments.append(f"{key} => JSON(:{key})")
            else:
                arguments.append(f"{key} => :{key}")
        return (
            f"BEGIN :result := DBMS_VECTOR_DATABASE.{operation}("
            f"{', '.join(arguments)}); END;",
            params,
        )


def validate_hybrid_request(params: dict) -> None:
    """Validate query inputs with SDK helpers without narrowing future options.

    The branch deliberately forwards new SEARCH options (e.g. aggregator_params).
    Keep that behavior rather than applying its older option-name whitelist.
    """
    if not isinstance(params, dict):
        raise TypeError("Hybrid search parameters must be a dictionary")
    for section in ("vector", "text"):
        if section not in params:
            continue
        options = params[section]
        if not isinstance(options, dict):
            raise TypeError(f"{section} must be a dictionary")
        if section == "vector" and "search_vector" in options:
            vector = options["search_vector"]
            if isinstance(vector, str):
                vector = json.loads(vector)
            HybridSearch._hybrid_search_input({"vector": vector})
        if "inpath" in options:
            HybridSearch._validate_json_path_list(
                options["inpath"], f"{section}.inpath"
            )
        if "json_textcontains" in options:
            HybridSearch._hybrid_json_textcontains(options["json_textcontains"])
    if "return" in params:
        returns = params["return"]
        if not isinstance(returns, dict):
            raise TypeError("return must be a dictionary")
        if "topN" in returns:
            HybridSearch._positive_integer(returns["topN"], "return.topN")
        if str(returns.get("format", "JSON")).upper() != "JSON":
            raise ValueError("Hybrid document mapping requires return.format=JSON")


def hybrid_index_ddl(
    index: str, table: str, column: str, parameters: str, parallel: str = ""
) -> str:
    """Render the SDK's JSON-column HVI statement with validated SQL identifiers.

    Parameter preservation and recovery remain the native transport's concern.
    """
    return HybridSearch._create_hybrid_index_statement(
        index, table, column, parameters, parallel
    )


class _LangChainHybridSearch(HybridSearch):
    """Supply the metadata text path for caller-generated query embeddings."""

    @staticmethod
    def _identifier(value: Any, name: str) -> str:
        if name == "hybrid_index_name":
            if not isinstance(value, str) or not value.strip():
                raise ValueError("hybrid_index_name must be a non-empty string")
            # This is a catalog name in a bound JSON value, never SQL text.
            return value
        return HybridSearch._identifier(value, name)

    def _describe_table_embed_params(self, table_name: str) -> dict[str, Any]:
        # Both table layouts store page_content under the same internal key.
        return {"embed_metadata_jsonpath": TEXT_KEY}

    @staticmethod
    def _require_hybrid_embed_params(embed_params: Any, operation: str) -> dict:
        if operation != "query":
            return HybridSearch._require_hybrid_embed_params(embed_params, operation)
        # The SDK also requires a database model here. LangChain supplies query
        # vectors itself; request construction only needs the stored text path.
        HybridSearch._json_path_literal(embed_params["embed_metadata_jsonpath"])
        return deepcopy(embed_params)


def hybrid_search_request(name: str, params: dict) -> dict:
    """Translate retriever inputs to the branch's native SEARCH request shape."""
    validate_hybrid_request(params)
    options = deepcopy(params)
    vector = options.get("vector", {})
    text = options.get("text", {})
    if "search_vector" in vector:
        value = vector.pop("search_vector")
        if isinstance(value, str):
            value = json.loads(value)
        query_by = {"vector": value}
    else:
        value = (
            vector.get("search_text")
            or text.get("search_text")
            or text.get("contains")
            or options.get("search_text")
        )
        if value is None and "json_textcontains" in text:
            value = text["json_textcontains"][1]
        query_by = {"text": value}
    query_kind, query_value = HybridSearch._hybrid_search_input(query_by)
    top_k = options.get("return", {}).get("topN", 4)
    return _LangChainHybridSearch()._hybrid_search_request(
        name, query_kind, query_value, top_k, None, None, options
    )


def validate_retriever_options(params: dict) -> dict:
    """Keep caller options separate from query-derived retriever fields."""
    if not isinstance(params, dict):
        raise TypeError("Search parameters must be a dictionary")
    if "search_text" in params:
        raise ValueError(
            "Cannot provide search_text as a parameter at the top level; "
            "it is derived from the query."
        )
    for section, fields in (
        ("vector", {"search_text", "search_vector"}),
        ("text", {"search_text", "search_vector", "contains"}),
    ):
        options = params.get(section) or {}
        if not isinstance(options, dict):
            raise TypeError(f"{section} must be a dictionary")
        if fields & options.keys():
            raise ValueError(
                f"Cannot provide search_text as a parameter in params['{section}']; "
                "it is derived from the query."
            )
    if "return" in params:
        raise ValueError(
            "Cannot provide return as a parameter in params; "
            "it is handled internally. Use `return_scores` "
            "parameter to get the scores."
        )
    return params


def retriever_hybrid_options(
    query: str,
    mode: str,
    index: str,
    k: int,
    params: dict,
    overrides: dict,
) -> dict:
    """Map LangChain modes and literal text to SDK options, before embedding.

    The copied request builder owns the final SEARCH shape and return defaults.
    This mapping is shared by sync and async retrieval and never embeds text.
    """
    from ..retrievers.text_search import _generate_accum_query

    options = deepcopy(params)
    options.update(deepcopy(overrides))
    validate_retriever_options(options)
    if {"fetch_k", "rrf_k"} & options.keys():
        raise ValueError(
            "Use DBMS_HYBRID_VECTOR.SEARCH parameters for native hybrid search"
        )
    options["hybrid_index_name"] = index
    options["return"] = {"topN": HybridSearch._positive_integer(k, "k")}
    if mode in {"semantic", "hybrid"}:
        options["vector"] = dict(options.get("vector") or {})
    else:
        options.pop("vector", None)
    if mode in {"keyword", "hybrid"}:
        text = dict(options.get("text") or {})
        if "json_textcontains" in text:
            raise ValueError("The text path and query are derived internally")
        text["json_textcontains"] = [
            f"$.{TEXT_KEY}",
            _generate_accum_query(query, False),
        ]
        options["text"] = text
    else:
        options.pop("text", None)
    if mode in {"semantic", "keyword"}:
        options.setdefault(
            "search_fusion", "VECTOR_ONLY" if mode == "semantic" else "TEXT_ONLY"
        )
    return options
