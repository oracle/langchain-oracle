# Copyright (c) 2025, 2026, Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Cache integrations backed by Oracle Database."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Optional, Union

from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.caches import RETURN_VAL_TYPE, BaseCache
from langchain_core.embeddings import Embeddings
from langchain_core.load.dump import dumps
from langchain_core.load.load import loads
from langchain_core.outputs import Generation

from langchain_oracledb.vectorstores.oraclevs import OracleVS, create_index
from langchain_oracledb.vectorstores.utils import (
    _get_connection,
    _quote_indentifier,
    _validate_indentifier,
    drop_table_purge,
)

logger = logging.getLogger(__name__)
_DISTANCE_EPSILON = 1e-12
DEFAULT_TABLE_NAME = "langchain_semantic_cache"


def _hash_value(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _cache_entry_id(prompt: str, llm_string: str) -> str:
    return _hash_value(f"{prompt}\x00{llm_string}")


def _dumps_generations(generations: RETURN_VAL_TYPE) -> str:
    """Serialize a sequence of `Generation` objects."""
    return json.dumps([dumps(item) for item in generations])


def _reset_generation_ids(generations: RETURN_VAL_TYPE) -> None:
    """Clear the ``.id`` on cached chat messages before returning them.

    Cached ``AIMessage``s keep whatever ``id`` they had at write time.
    When LangChain caching is wired into a LangGraph agent, the graph's
    ``add_messages`` reducer dedupes by ``id``: a cached message with an
    id already present in state replaces the existing one instead of
    being appended, and state stops progressing. Dropping the id on load
    matches how a fresh LLM call would behave (the runtime mints a new
    ``lc_run--<uuid>`` id when the message is emitted).
    """
    for gen in generations:
        message = getattr(gen, "message", None)
        if message is not None and getattr(message, "id", None) is not None:
            try:
                message.id = None
            except (AttributeError, TypeError):
                pass


def _has_tool_calls(generations: RETURN_VAL_TYPE) -> bool:
    """Return True if any generation carries a tool-call message."""
    for gen in generations:
        message = getattr(gen, "message", None)
        if message is None:
            continue
        if getattr(message, "tool_calls", None):
            return True
        if getattr(message, "invalid_tool_calls", None):
            return True
    return False


def _loads_generations(generations_str: str) -> Union[RETURN_VAL_TYPE, None]:
    """Deserialize a sequence of `Generation` objects."""
    try:
        return [
            loads(item_str, allowed_objects="core")
            for item_str in json.loads(generations_str)
        ]
    except (json.JSONDecodeError, TypeError):
        pass

    try:
        gen_dicts = json.loads(generations_str)
        generations = [Generation(**generation_dict) for generation_dict in gen_dicts]
        logger.warning(
            "Legacy 'Generation' cached blob encountered: '%s'", generations_str
        )
        return generations
    except (json.JSONDecodeError, TypeError):
        logger.warning(
            "Malformed/unparsable cached blob encountered: '%s'", generations_str
        )
        return None


class OracleSemanticCache(BaseCache):
    """Semantic cache backed by Oracle AI Vector Search.

    The prompt text is stored as the searchable document text. Cache entries are
    separated by ``llm_string`` via metadata filtering, and updated entries reuse a
    stable hashed id so writes are idempotent.
    """

    PROMPT_HASH = "prompt_hash"
    LLM_HASH = "llm_string_hash"
    RETURN_VAL = "return_val"

    def __init__(
        self,
        client: Any,
        embedding: Embeddings,
        table_name: str = DEFAULT_TABLE_NAME,
        *,
        distance_strategy: DistanceStrategy = DistanceStrategy.COSINE,
        index_name: Optional[str] = None,
        create_index_if_missing: bool = False,
        index_params: Optional[dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
    ) -> None:
        """Initialize the Oracle semantic cache.

        Args:
            client: ``oracledb.Connection`` or ``oracledb.ConnectionPool``.
            embedding: Embedding model used to vectorize prompts.
            table_name: Table used to store semantic cache entries. Defaults to
                ``langchain_semantic_cache``.
            distance_strategy: Oracle vector distance function to use for lookup.
            index_name: Optional vector index name to create.
            create_index_if_missing: Whether to create a vector index on init.
            index_params: Optional Oracle vector index parameters.
            score_threshold: Optional maximum distance allowed for a cache hit.
                Oracle returns distance scores, so lower values are better.
        """
        if client is None:
            raise ValueError("client must be provided")
        if score_threshold is not None and score_threshold < 0:
            raise ValueError("score_threshold must be non-negative")

        _validate_indentifier(table_name)
        if index_name is not None:
            _validate_indentifier(index_name)

        self._client = client
        self._table_name = table_name
        self._score_threshold = score_threshold
        self._distance_strategy = distance_strategy
        self._vector_store = OracleVS(
            client=client,
            embedding_function=embedding,
            table_name=table_name,
            distance_strategy=distance_strategy,
            mutate_on_duplicate=True,
        )

        if create_index_if_missing:
            params = dict(index_params or {})
            if index_name is not None:
                params["idx_name"] = index_name
            if "idx_type" not in params:
                params["idx_type"] = "HNSW"
            create_index(client, self._vector_store, params=params)

    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        """Look up the nearest cached response for a prompt and llm configuration."""
        filter_ = {self.LLM_HASH: {"$eq": _hash_value(llm_string)}}
        search_response = self._vector_store.similarity_search_with_score(
            prompt, 1, filter=filter_
        )
        if not search_response:
            return None

        document, score = search_response[0]
        if (
            self._score_threshold is not None
            and score > self._score_threshold + _DISTANCE_EPSILON
        ):
            return None

        return_val = document.metadata.get(self.RETURN_VAL)
        if not isinstance(return_val, str):
            return None
        generations = _loads_generations(return_val)
        if generations is not None:
            _reset_generation_ids(generations)
        return generations

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        """Insert or update a semantic cache entry.

        Generations whose message carries ``tool_calls`` are intentionally
        **not** cached. Tool-call responses are procedural (one step inside
        an agent loop, not a final answer), and replaying them collapses
        agent state: a cached tool-call ``AIMessage`` keeps its original
        ``id``, LangGraph's ``add_messages`` reducer dedupes by ``id``, and
        the graph fails to advance past the tool node.
        """
        if _has_tool_calls(return_val):
            return

        metadata = {
            self.PROMPT_HASH: _hash_value(prompt),
            self.LLM_HASH: _hash_value(llm_string),
            self.RETURN_VAL: _dumps_generations(return_val),
        }
        self._vector_store.add_texts(
            [prompt],
            [metadata],
            ids=[_cache_entry_id(prompt, llm_string)],
        )

    def clear(self, **kwargs: Any) -> None:
        """Clear cache entries.

        Supported filters:
            - ``prompt``: delete only entries written for the exact prompt
            - ``llm_string``: delete only entries written for the exact llm string
        """
        prompt = kwargs.pop("prompt", None)
        llm_string = kwargs.pop("llm_string", None)
        if kwargs:
            unsupported = ", ".join(sorted(kwargs))
            raise ValueError(f"Unsupported clear filters: {unsupported}")

        query = f"DELETE FROM {self._quoted_table_name}"
        bind_vars: dict[str, Any] = {}
        conditions = []

        if prompt is not None:
            conditions.append(
                f"JSON_VALUE(metadata, '$.{self.PROMPT_HASH}') = :prompt_hash"
            )
            bind_vars["prompt_hash"] = _hash_value(prompt)

        if llm_string is not None:
            conditions.append(f"JSON_VALUE(metadata, '$.{self.LLM_HASH}') = :llm_hash")
            bind_vars["llm_hash"] = _hash_value(llm_string)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        with _get_connection(self._client) as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, bind_vars)
            connection.commit()

    @staticmethod
    def drop_table(client: Any, table_name: str = DEFAULT_TABLE_NAME) -> None:
        """Drop the semantic cache table.

        Args:
            client: ``oracledb.Connection`` or ``oracledb.ConnectionPool``.
            table_name: Table to drop. Defaults to ``langchain_semantic_cache``.
        """
        drop_table_purge(client, table_name)

    @property
    def _quoted_table_name(self) -> str:
        return _quote_indentifier(self._table_name)


__all__ = ["OracleSemanticCache"]
