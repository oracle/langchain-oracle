# Copyright (c) 2024, 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""Full-text search utilities for Oracle Database (Oracle Text).

This module provides:
- create_text_index/acreate_text_index: build an Oracle Text SEARCH INDEX
  over a table column (either using an OracleVS table or a user-provided table).
- OracleTextSearchRetriever: a LangChain retriever that executes Oracle Text
  CONTAINS queries and returns LangChain Documents.

Notes:
- When a vector_store (OracleVS) is provided, the supported searchable columns are
  limited to "text".
- You may also target an arbitrary table/column by supplying
  (client + table_name + column_name).

Query tips:
- For exact multi-token phrases, wrap the phrase in double quotes,
  for example: "refund policy".
  Oracle Text will enforce token order and adjacency for quoted phrases.
- With fuzzy=True, this retriever splits the query into tokens and
  applies fuzzy(...) to each token joined with AND, so phrase semantics
  from double quotes are not supported in fuzzy mode.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, List, Optional, Union, cast

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import model_validator

from langchain_oracledb.vectorstores.oraclevs import INTERNAL_ID_KEY, OracleVS
from langchain_oracledb.vectorstores.utils import (
    _aget_connection,
    _ahandle_exceptions,
    _aindex_exists,
    _get_connection,
    _handle_exceptions,
    _index_exists,
    _quote_indentifier,
    _validate_indentifier,
    output_type_string_handler,
)

if TYPE_CHECKING:
    from oracledb import (
        AsyncConnection,
        AsyncConnectionPool,
        Connection,
        ConnectionPool,
    )

logger = logging.getLogger(__name__)


def _get_text_index_ddl(
    idx_name: str,
    vector_store: Optional[OracleVS],
    table_name: Optional[str],
    column_name: Optional[str] = "text",
):
    """Build the CREATE SEARCH INDEX DDL statement and resolve the target table.

    Args:
        idx_name: Index name (will be quoted).
        vector_store: OracleVS instance. If provided, the table name is taken from it,
            and column_name must be "text".
        table_name: Explicit table name (quoted). Mutually exclusive with vector_store.
        column_name: Column to index. Defaults to "text". When vector_store is given,
            allowed value is only "text".

    Returns:
        tuple[str, str]: (ddl, resolved_table_name)

    Raises:
        ValueError: If both vector_store and table_name are provided, or if neither
            resolves to a valid target; also for invalid column choices.
    """

    if vector_store and table_name:
        raise ValueError("Only give one of vector_store or table_name.")
    if not vector_store and not table_name:
        raise ValueError("Provide either vector_store or table_name.")

    # Resolve table name and validate column
    if vector_store is not None:
        # For OracleVS we only allow the "text" column
        col = (column_name or "text").lower()
        if col != "text":
            raise ValueError(
                "When vector_store is provided, column_name must be 'text'."
            )
        resolved_table = vector_store.table_name  # already quoted by OracleVS
        resolved_column = col  # keep unquoted to avoid case-sensitivity issues
    else:
        if not table_name:
            raise ValueError(
                "table_name must be provided when vector_store is not used."
            )
        if not column_name:
            raise ValueError("column_name must be provided when table_name is used.")
        _validate_indentifier(table_name)
        resolved_table = table_name
        # Intentionally do not quote the column_name to avoid case-sensitive mismatches.
        # It must be a simple identifier known to the caller; no SQL is interpolated.
        _validate_indentifier(column_name)
        resolved_column = column_name

    ddl = f"CREATE SEARCH INDEX {idx_name} ON {resolved_table}({resolved_column})"
    return ddl, resolved_table


@_handle_exceptions
def create_text_index(
    client: Union["Connection", "ConnectionPool"],
    idx_name: str,
    vector_store: Optional[OracleVS] = None,
    table_name: Optional[str] = None,
    column_name: Optional[str] = "text",
) -> None:
    """Create an Oracle Text SEARCH INDEX if it does not already exist.

    Exactly one of vector_store or table_name must be provided.
    - If vector_store is given, column_name must be "text".
    - If table_name is given, column_name is required and used as-is (unquoted).

    Args:
        client: oracledb connection or connection pool.
        idx_name: Index name to create (quoted automatically).
        vector_store: OracleVS backing table to index ("text").
        table_name: Explicit table to index.
        column_name: Column to index. Defaults to "text".

    Raises:
        RuntimeError/ValueError: on DB/validation errors.
    """
    idx_name = _quote_indentifier(idx_name)
    ddl, resolved_table = _get_text_index_ddl(
        idx_name, vector_store, table_name, column_name
    )

    with _get_connection(client) as connection:
        if not _index_exists(connection, idx_name, resolved_table):
            with connection.cursor() as cur:
                cur.execute(ddl)
                logger.info(f"Index {idx_name} created successfully...")
        else:
            logger.info(f"Index {idx_name} already exists...")


@_ahandle_exceptions
async def acreate_text_index(
    client: Union["AsyncConnection", "AsyncConnectionPool"],
    idx_name: str,
    vector_store: Optional[OracleVS] = None,
    table_name: Optional[str] = None,
    column_name: Optional[str] = "text",
) -> None:
    """Async variant of create_text_index.

    Creates the Oracle Text SEARCH INDEX if it does not exist, using async APIs.

    Args:
        client: oracledb async connection or async connection pool.
        idx_name: Index name to create (quoted automatically).
        vector_store: OracleVS backing table to index ("text").
        table_name: Explicit table to index.
        column_name: Column to index. Defaults to "text".

    Raises:
        RuntimeError/ValueError: on DB/validation errors.
    """
    idx_name = _quote_indentifier(idx_name)
    ddl, resolved_table = _get_text_index_ddl(
        idx_name, vector_store, table_name, column_name
    )

    async with _aget_connection(client) as connection:
        if not await _aindex_exists(connection, idx_name, resolved_table):
            async with connection.cursor() as cur:
                await cur.execute(ddl)
                logger.info(f"Index {idx_name} created successfully...")
        else:
            logger.info(f"Index {idx_name} already exists...")


class OracleTextSearchRetriever(BaseRetriever):
    """LangChain retriever that executes Oracle Text CONTAINS searches.

    Usage modes:
    - Use an OracleVS instance to target the built-in "text" column.
    - Or supply a raw client + table_name + column_name to target any suitable table.

    Fields:
    - vector_store: OracleVS pointing to the table (optional if table_name provided).
    - client: oracledb connection or pool (required if vector_store is not provided).
    - table_name: Target table when not using OracleVS (quoted automatically).
    - column_name: Column to search. With OracleVS, allowed value: "text" only.
    - k: Number of results to return (default 4).
    - fuzzy: If True, wraps the query with fuzzy(...) expression.
    - return_scores: If True, includes Oracle Text SCORE(1) in metadata as "score".
    - returned_columns: Additional columns to return as metadata.

    Example:
        retriever = OracleTextSearchRetriever(
            vector_store=vs,  # or client=..., table_name="MYDOCS", column_name="TEXT"
            column_name="text",
            k=5,
            return_scores=True,
            returned_columns=["metadata"],
        )
        docs = retriever.invoke("refund policy for premium plan")
        for d in docs:
            print(d.page_content, d.metadata.get("score"))

    Query tips:
    - For exact multi-token phrases, wrap the phrase in double quotes
      (e.g., "refund policy"). Oracle Text enforces order and
      adjacency for quoted phrases.
    - Fuzzy mode (fuzzy=True) applies fuzzy(...) token-by-token and does not
      preserve phrase semantics.
    """

    vector_store: Optional[OracleVS] = None
    """OracleVS VectorStore."""
    client: Optional[Any] = None
    """oracledb Connection or ConnectionPool; used when vector_store is not provided."""
    table_name: Optional[str] = None
    """Target table name (quoted automatically when set)."""
    column_name: Optional[str] = "text"
    """Target column; with OracleVS must be 'text'."""
    k: Optional[int] = 4
    """Number of documents to return."""
    fuzzy: Optional[bool] = False
    """Whether to wrap the query with fuzzy(...)."""
    return_scores: Optional[bool] = False
    """If True, include Oracle Text SCORE(1) under metadata['score']."""  # noqa: E501

    returned_columns: Optional[list[str]] = None
    """Additional columns to fetch and include in metadata."""

    @model_validator(mode="after")
    def check_values(self):
        """Validate mutually exclusive inputs and normalize fields."""
        vs = self.vector_store
        tbl = self.table_name
        col = self.column_name or "text"

        # Validate mutual exclusivity and presence
        if vs and tbl:
            raise ValueError("Only give one of vector_store or table_name.")
        if not vs and not tbl:
            raise ValueError("Provide either vector_store or table_name.")
        if not vs and not self.client:
            raise ValueError("client must be provided when vector_store is not used.")

        # Resolve/validate column and table
        if vs:
            if col.lower() != "text":
                raise ValueError(
                    "When vector_store is provided, column_name must be 'text'."
                )
            resolved_table = vs.table_name
            resolved_column = "text"
        else:
            if not col:
                raise ValueError(
                    "column_name must be provided when table_name is used."
                )
            tbl = cast(str, tbl)
            _validate_indentifier(tbl)
            _validate_indentifier(col)
            resolved_table = tbl  # already quoted by validator
            resolved_column = col  # leave unquoted to avoid case-sensitive mismatches

        # Compute returned_columns
        rc = self.returned_columns or []
        if self.returned_columns is None and vs:
            rc = ["metadata"]

        # De-duplicate and ensure we don't include the main column twice
        rc = [c for c in rc if c and c.lower() != resolved_column.lower()]

        self.table_name = resolved_table
        self.column_name = resolved_column
        self.returned_columns = rc
        return self

    def _get_result_documents(self, rows: list[dict[str, Any]]) -> List[Document]:
        """Convert raw rows into LangChain Documents."""
        self.column_name = cast(str, self.column_name)
        docs: list[Document] = []
        for row in rows:
            score = row.get("score")

            if self.vector_store:
                result_dict: dict[str, Any] = {}
                if "metadata" in row and row["metadata"] is not None:
                    metadata = row["metadata"]
                    # Remove internal id if present
                    doc_id = metadata.pop(INTERNAL_ID_KEY, None)
                    result_dict["id"] = doc_id
                    result_dict["metadata"] = metadata
                if "text" in row:
                    result_dict["page_content"] = row["text"]

                if self.return_scores and score is not None:
                    mt = result_dict.get("metadata", {}) or {}
                    mt["score"] = score
                    result_dict["metadata"] = mt

                doc = Document(**result_dict)
            else:
                content = cast(str, row.get(self.column_name))
                metadata = {
                    ret_col: row.get(ret_col)
                    for ret_col in (self.returned_columns or [])
                }
                if self.return_scores and score is not None:
                    metadata["score"] = score
                doc = Document(page_content=content, metadata=metadata)

            docs.append(doc)

        return docs

    def _get_relevant_documents(self, query: str, **kwargs: Any) -> List[Document]:
        """Execute a synchronous Oracle Text CONTAINS search.

        The query string is passed directly to CONTAINS; when fuzzy=True, it is wrapped
        as fuzzy(query). The search uses SCORE(1) with label '1' in CONTAINS and orders
        results by SCORE descending, returning top-k rows.

        Args:
            query: Natural language query string or Oracle Text
              expression (e.g., fuzzy(cat)).
            **kwargs: Optional overrides including "k" (int).

        Returns:
            List[Document]: Top-k documents sorted by SCORE(1).

        Query notes:
        - To require an exact multi-token phrase, pass it in double quotes
          (e.g., "refund policy"). Oracle Text enforces token order and
          adjacency for quoted phrases.
        - With fuzzy=True, the query is split into tokens and each token
          is wrapped with fuzzy(...), joined by AND. Phrase semantics from
          quotes are not supported in fuzzy mode.
        """
        if self.fuzzy:
            query = " AND ".join([f"fuzzy({q})" for q in query.split()])

        # Build select column list: primary column + optional returned columns
        self.column_name = cast(str, self.column_name)
        select_cols = [self.column_name]
        if self.returned_columns:
            select_cols.extend(
                [
                    c
                    for c in self.returned_columns
                    if c.lower() != self.column_name.lower()
                ]
            )
        select_cols_str = ", ".join(select_cols)

        search_query = f"""
        SELECT SCORE(1) score, {select_cols_str} FROM {self.table_name}
        WHERE CONTAINS({self.column_name}, :query, 1) > 0
        ORDER BY score DESC FETCH FIRST {kwargs.get("k", None) or self.k or 4} ROWS ONLY
        """

        # Pick connection source
        conn_src = self.vector_store.client if self.vector_store else self.client
        with _get_connection(conn_src) as connection:
            with connection.cursor() as cursor:
                cursor.outputtypehandler = output_type_string_handler
                cursor.execute(
                    search_query,
                    query=query,
                )

                columns = [col[0].lower() for col in cursor.description]
                rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
                docs = self._get_result_documents(rows)

        return docs

    async def _aget_relevant_documents(
        self, query: str, **kwargs: Any
    ) -> List[Document]:
        """Async variant of _get_relevant_documents using async connection APIs.

        Args:
            query: Natural language query string or Oracle Text expression.
            **kwargs: Optional overrides including "k" (int).

        Returns:
            List[Document]: Top-k documents sorted by SCORE(1).

        Query notes:
        - To require an exact multi-token phrase, pass it in double quotes
          (e.g., "refund policy"). Oracle Text enforces token order and
          adjacency for quoted phrases.
        - With fuzzy=True, the query is split into tokens and each token is wrapped
          with fuzzy(...), joined by AND. Phrase semantics from quotes are not
          supported in fuzzy mode.

        """
        if self.fuzzy:
            query = " AND ".join([f"fuzzy({q})" for q in query.split()])

        self.column_name = cast(str, self.column_name)
        select_cols = [self.column_name]
        if self.returned_columns:
            select_cols.extend(
                [
                    c
                    for c in self.returned_columns
                    if c.lower() != self.column_name.lower()
                ]
            )
        select_cols_str = ", ".join(select_cols)

        search_query = f"""
        SELECT SCORE(1) score, {select_cols_str} FROM {self.table_name}
        WHERE CONTAINS({self.column_name}, :query, 1) > 0
        ORDER BY score DESC FETCH FIRST {kwargs.get("k", None) or self.k or 4} ROWS ONLY
        """

        conn_src = self.vector_store.client if self.vector_store else self.client
        async with _aget_connection(conn_src) as connection:
            async with connection.cursor() as cursor:
                cursor.outputtypehandler = output_type_string_handler
                await cursor.execute(
                    search_query,
                    query=query,
                )
                columns = [col[0].lower() for col in cursor.description]
                rows = [dict(zip(columns, row)) for row in await cursor.fetchall()]
                docs = self._get_result_documents(rows)

        return docs
