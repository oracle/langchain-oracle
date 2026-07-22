# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Key-value stores backed by Oracle Database.

Implements ``langchain_core.stores.BaseStore`` so Oracle can back the
patterns built on top of it:

- :class:`OracleByteStore` (``BaseStore[str, bytes]``) — e.g. for
  ``CacheBackedEmbeddings`` so re-ingestion doesn't re-embed unchanged text.
- :class:`OracleDocStore` (``BaseStore[str, Document]``) — e.g. as the
  parent-document store for ``ParentDocumentRetriever`` next to
  :class:`~langchain_oracledb.vectorstores.OracleVS`.

Note: this is the *langchain-core* key-value store abstraction. The
LangGraph long-term-memory ``BaseStore`` is a different interface, provided
by the ``langgraph-oracledb`` package.
"""

from __future__ import annotations

import json
from typing import (
    Any,
    Generic,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)

from langchain_core.documents import Document
from langchain_core.stores import BaseStore

from langchain_oracledb.vectorstores.utils import (
    _get_connection,
    _quote_indentifier,
    _validate_indentifier,
    drop_table_purge,
)

V = TypeVar("V")

DEFAULT_BYTE_STORE_TABLE_NAME = "langchain_byte_store"
DEFAULT_DOC_STORE_TABLE_NAME = "langchain_doc_store"

# Oracle limits IN-list literals to 1000 elements; stay under it.
_BATCH_SIZE = 500


class _OracleKVStore(BaseStore[str, V], Generic[V]):
    """Shared Oracle-table implementation behind the typed store classes.

    Subclasses define the value column type and the value (de)serialization.
    """

    _VALUE_COLUMN_TYPE = "BLOB"

    def __init__(self, client: Any, table_name: str) -> None:
        if client is None:
            raise ValueError("client must be provided")
        _validate_indentifier(table_name)
        self._client = client
        self._table_name = table_name
        self._ensure_table()

    @property
    def _quoted_table_name(self) -> str:
        return _quote_indentifier(self._table_name)

    def _ensure_table(self) -> None:
        ddl = f"""
            CREATE TABLE IF NOT EXISTS {self._quoted_table_name} (
                k VARCHAR2(1000) PRIMARY KEY,
                v {self._VALUE_COLUMN_TYPE} NOT NULL,
                updated_at TIMESTAMP DEFAULT SYSTIMESTAMP
            )
        """
        with _get_connection(self._client) as connection:
            with connection.cursor() as cursor:
                cursor.execute(ddl)
            connection.commit()

    # -- serialization hooks -------------------------------------------------

    def _serialize(self, value: V) -> Any:
        raise NotImplementedError

    def _deserialize(self, raw: Any) -> V:
        raise NotImplementedError

    @staticmethod
    def _read_raw(raw: Any) -> Any:
        return raw.read() if hasattr(raw, "read") else raw

    # -- BaseStore interface -------------------------------------------------

    def mget(self, keys: Sequence[str]) -> List[Optional[V]]:
        """Get the values associated with the given keys (order-preserving)."""
        if not keys:
            return []

        found: dict[str, V] = {}
        with _get_connection(self._client) as connection:
            with connection.cursor() as cursor:
                for start in range(0, len(keys), _BATCH_SIZE):
                    batch = list(keys[start : start + _BATCH_SIZE])
                    placeholders = ", ".join(f":k{i}" for i in range(len(batch)))
                    cursor.execute(
                        f"SELECT k, v FROM {self._quoted_table_name} "
                        f"WHERE k IN ({placeholders})",
                        {f"k{i}": key for i, key in enumerate(batch)},
                    )
                    for row_key, row_value in cursor.fetchall():
                        found[row_key] = self._deserialize(self._read_raw(row_value))

        return [found.get(key) for key in keys]

    def mset(self, key_value_pairs: Sequence[Tuple[str, V]]) -> None:
        """Set the values for the given keys (idempotent MERGE upsert)."""
        if not key_value_pairs:
            return

        merge = f"""
            MERGE INTO {self._quoted_table_name} t
            USING (SELECT :k AS k FROM dual) s
            ON (t.k = s.k)
            WHEN MATCHED THEN UPDATE SET
                t.v = :v,
                t.updated_at = SYSTIMESTAMP
            WHEN NOT MATCHED THEN INSERT (k, v) VALUES (:k, :v)
        """
        rows = [
            {"k": key, "v": self._serialize(value)} for key, value in key_value_pairs
        ]
        with _get_connection(self._client) as connection:
            with connection.cursor() as cursor:
                cursor.executemany(merge, rows)
            connection.commit()

    def mdelete(self, keys: Sequence[str]) -> None:
        """Delete the given keys (missing keys are ignored)."""
        if not keys:
            return

        with _get_connection(self._client) as connection:
            with connection.cursor() as cursor:
                cursor.executemany(
                    f"DELETE FROM {self._quoted_table_name} WHERE k = :k",
                    [{"k": key} for key in keys],
                )
            connection.commit()

    def yield_keys(self, *, prefix: Optional[str] = None) -> Iterator[str]:
        """Yield keys in the store, optionally filtered by prefix."""
        query = f"SELECT k FROM {self._quoted_table_name}"
        bind_vars: dict[str, Any] = {}
        if prefix is not None:
            # ESCAPE so % and _ in the caller's prefix match literally
            escaped = (
                prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            )
            query += " WHERE k LIKE :prefix ESCAPE '\\'"
            bind_vars["prefix"] = f"{escaped}%"

        with _get_connection(self._client) as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, bind_vars)
                for (key,) in cursor:
                    yield key

    @staticmethod
    def drop_table(client: Any, table_name: str) -> None:
        """Drop the store table."""
        drop_table_purge(client, table_name)


class OracleByteStore(_OracleKVStore[bytes]):
    """``BaseStore[str, bytes]`` backed by an Oracle table with a BLOB column.

    Use with ``CacheBackedEmbeddings`` to persist embedding vectors:

    .. code-block:: python

        import oracledb
        from langchain.embeddings import CacheBackedEmbeddings
        from langchain_oracledb import OracleByteStore

        connection = oracledb.connect(user=..., password=..., dsn=...)
        store = OracleByteStore(connection)
        cached_embedder = CacheBackedEmbeddings.from_bytes_store(
            underlying_embeddings, store, namespace=underlying_embeddings.model
        )
    """

    _VALUE_COLUMN_TYPE = "BLOB"

    def __init__(
        self,
        client: Any,
        table_name: str = DEFAULT_BYTE_STORE_TABLE_NAME,
    ) -> None:
        """Initialize the byte store.

        Args:
            client: ``oracledb.Connection`` or ``oracledb.ConnectionPool``.
            table_name: Table used to store entries. Defaults to
                ``langchain_byte_store``. Created on init if missing.
        """
        super().__init__(client, table_name)

    def _serialize(self, value: bytes) -> bytes:
        if not isinstance(value, bytes):
            raise TypeError(f"OracleByteStore values must be bytes, got {type(value)}")
        return value

    def _deserialize(self, raw: Any) -> bytes:
        return bytes(raw)


class OracleDocStore(_OracleKVStore[Document]):
    """``BaseStore[str, Document]`` backed by an Oracle table.

    Use as the parent-document store for ``ParentDocumentRetriever``:

    .. code-block:: python

        import oracledb
        from langchain.retrievers import ParentDocumentRetriever
        from langchain_oracledb import OracleDocStore, OracleVS

        connection = oracledb.connect(user=..., password=..., dsn=...)
        retriever = ParentDocumentRetriever(
            vectorstore=OracleVS(...),
            docstore=OracleDocStore(connection),
            child_splitter=child_splitter,
        )
    """

    _VALUE_COLUMN_TYPE = "CLOB"

    def __init__(
        self,
        client: Any,
        table_name: str = DEFAULT_DOC_STORE_TABLE_NAME,
    ) -> None:
        """Initialize the document store.

        Args:
            client: ``oracledb.Connection`` or ``oracledb.ConnectionPool``.
            table_name: Table used to store entries. Defaults to
                ``langchain_doc_store``. Created on init if missing.
        """
        super().__init__(client, table_name)

    def _serialize(self, value: Document) -> str:
        if not isinstance(value, Document):
            raise TypeError(
                f"OracleDocStore values must be Documents, got {type(value)}"
            )
        payload = {"page_content": value.page_content, "metadata": value.metadata}
        if value.id is not None:
            payload["id"] = value.id
        return json.dumps(payload)

    def _deserialize(self, raw: Any) -> Document:
        payload = json.loads(raw)
        return Document(
            page_content=payload["page_content"],
            metadata=payload.get("metadata") or {},
            id=payload.get("id"),
        )


__all__ = ["OracleByteStore", "OracleDocStore"]
