# Copyright (c) 2025, 2026, Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Chat message history backed by Oracle Database."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Sequence
from typing import Any, List, Optional

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict

from langchain_oracledb.vectorstores.utils import (
    _get_connection,
    _index_exists,
    _quote_indentifier,
    _table_exists,
    _validate_indentifier,
    drop_table_purge,
)

logger = logging.getLogger(__name__)

DEFAULT_TABLE_NAME = "langchain_message_store"
DEFAULT_SESSION_ID_KEY = "session_id"
DEFAULT_HISTORY_KEY = "message"
DEFAULT_ID_KEY = _quote_indentifier("id")


def _unqualified_identifier(name: str) -> str:
    return name.replace('"', "").split(".")[-1]


def _default_index_name(table_name: str, session_id_key: str) -> str:
    base_name = (
        f"idx_{_unqualified_identifier(table_name)}_"
        f"{_unqualified_identifier(session_id_key)}"
    )
    if len(base_name) <= 128:
        return base_name

    digest = hashlib.sha1(base_name.encode("utf-8")).hexdigest()[:8]
    return f"{base_name[:119]}_{digest}"


def _message_payload(value: Any) -> str:
    if hasattr(value, "read"):
        value = value.read()
    if not isinstance(value, str):
        raise TypeError(
            "Expected Oracle chat history payload to be a JSON string, "
            f"got {type(value).__name__}"
        )
    return value


class OracleChatMessageHistory(BaseChatMessageHistory):
    """Chat message history that stores messages in Oracle Database.

    The underlying table stores one row per message and preserves order by using an
    identity column. A single table can hold many sessions separated by
    ``session_id``.
    """

    def __init__(
        self,
        session_id: str,
        *,
        client: Any,
        table_name: str = DEFAULT_TABLE_NAME,
        session_id_key: str = DEFAULT_SESSION_ID_KEY,
        history_key: str = DEFAULT_HISTORY_KEY,
        create_table: bool = True,
        create_index: bool = True,
        history_size: Optional[int] = None,
    ) -> None:
        """Initialize the Oracle chat message history.

        Args:
            session_id: Application-defined session identifier.
            client: ``oracledb.Connection`` or ``oracledb.ConnectionPool``.
            table_name: Table used to store chat history rows. Defaults to
                ``langchain_message_store``.
            session_id_key: Column containing the session identifier.
            history_key: Column containing the serialized message payload.
            create_table: Whether to create the table if it does not exist.
            create_index: Whether to create an index on the session id column.
            history_size: Optional maximum number of most recent messages to return.
        """
        if not isinstance(session_id, str):
            raise ValueError("session_id must be a string")
        if client is None:
            raise ValueError("client must be provided")
        if history_size is not None and history_size < 1:
            raise ValueError("history_size must be greater than 0")

        _validate_indentifier(table_name)
        _validate_indentifier(session_id_key)
        _validate_indentifier(history_key)

        self._client = client
        self._session_id = session_id
        self._table_name = table_name
        self._session_id_key = session_id_key
        self._history_key = history_key
        self._history_size = history_size

        if create_table or create_index:
            self.create_tables(
                client,
                table_name,
                session_id_key=session_id_key,
                history_key=history_key,
                create_table=create_table,
                create_index=create_index,
            )

    @staticmethod
    def create_tables(
        client: Any,
        table_name: str = DEFAULT_TABLE_NAME,
        /,
        *,
        session_id_key: str = DEFAULT_SESSION_ID_KEY,
        history_key: str = DEFAULT_HISTORY_KEY,
        create_table: bool = True,
        create_index: bool = True,
    ) -> None:
        """Create the backing table and optional session index."""
        _validate_indentifier(table_name)
        _validate_indentifier(session_id_key)
        _validate_indentifier(history_key)

        quoted_table_name = _quote_indentifier(table_name)
        quoted_session_id_key = _quote_indentifier(session_id_key)
        quoted_history_key = _quote_indentifier(history_key)
        index_name = _default_index_name(table_name, session_id_key)
        quoted_index_name = _quote_indentifier(index_name)

        with _get_connection(client) as connection:
            table_exists = _table_exists(connection, quoted_table_name)
            with connection.cursor() as cursor:
                if create_table and not table_exists:
                    cursor.execute(
                        f"""
                        CREATE TABLE {quoted_table_name} (
                            {DEFAULT_ID_KEY} NUMBER GENERATED BY DEFAULT AS IDENTITY
                                PRIMARY KEY,
                            {quoted_session_id_key} VARCHAR2(255) NOT NULL,
                            {quoted_history_key} CLOB NOT NULL,
                            created_at TIMESTAMP WITH TIME ZONE DEFAULT
                                CURRENT_TIMESTAMP NOT NULL
                        )
                        """
                    )
                    table_exists = True

                if create_index:
                    if not table_exists:
                        raise ValueError(
                            f"Table {table_name} does not exist. "
                            "Set create_table=True or create the table first."
                        )
                    if not _index_exists(
                        connection,
                        quoted_index_name,
                        _unqualified_identifier(table_name),
                    ):
                        cursor.execute(
                            f"CREATE INDEX {quoted_index_name} "
                            f"ON {quoted_table_name} ({quoted_session_id_key})"
                        )

            connection.commit()

    @staticmethod
    def drop_table(client: Any, table_name: str = DEFAULT_TABLE_NAME, /) -> None:
        """Drop the backing table.

        Args:
            client: ``oracledb.Connection`` or ``oracledb.ConnectionPool``.
            table_name: Table to drop. Defaults to ``langchain_message_store``.
        """
        drop_table_purge(client, table_name)

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Add messages to the history in insertion order."""
        if not messages:
            return

        with _get_connection(self._client) as connection:
            with connection.cursor() as cursor:
                cursor.executemany(
                    self._insert_query,
                    self._serialize_messages(messages),
                )
            connection.commit()

    def get_messages(self) -> List[BaseMessage]:
        """Retrieve messages for the configured session."""
        if self._history_size is None:
            query = (
                f"SELECT {self._quoted_history_key}, {DEFAULT_ID_KEY} "
                f"FROM {self._quoted_table_name} "
                f"WHERE {self._quoted_session_id_key} = :session_id "
                f"ORDER BY {DEFAULT_ID_KEY}"
            )
            params: dict[str, Any] = {"session_id": self._session_id}
        else:
            query = (
                f"SELECT payload, {DEFAULT_ID_KEY} "
                "FROM ("
                f"    SELECT {self._quoted_history_key} AS payload, {DEFAULT_ID_KEY} "
                f"    FROM {self._quoted_table_name} "
                f"    WHERE {self._quoted_session_id_key} = :session_id "
                f"    ORDER BY {DEFAULT_ID_KEY} DESC"
                ") "
                "WHERE ROWNUM <= :history_size "
                f"ORDER BY {DEFAULT_ID_KEY}"
            )
            params = {
                "session_id": self._session_id,
                "history_size": self._history_size,
            }

        with _get_connection(self._client) as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, params)
                items = [json.loads(_message_payload(record[0])) for record in cursor]

        return messages_from_dict(items)

    @property
    def messages(self) -> List[BaseMessage]:
        """The messages stored for the configured session."""
        return self.get_messages()

    @messages.setter
    def messages(self, value: List[BaseMessage]) -> None:
        """Replace the stored messages for the configured session."""
        with _get_connection(self._client) as connection:
            try:
                with connection.cursor() as cursor:
                    cursor.execute(self._delete_query, {"session_id": self._session_id})
                    rows = self._serialize_messages(value)
                    if rows:
                        cursor.executemany(self._insert_query, rows)
                connection.commit()
            except Exception:
                connection.rollback()
                raise

    def clear(self) -> None:
        """Remove all messages for the configured session."""
        with _get_connection(self._client) as connection:
            with connection.cursor() as cursor:
                cursor.execute(self._delete_query, {"session_id": self._session_id})
            connection.commit()

    def _serialize_messages(
        self, messages: Sequence[BaseMessage]
    ) -> list[tuple[str, str]]:
        return [
            (self._session_id, json.dumps(message_to_dict(message)))
            for message in messages
        ]

    @property
    def _insert_query(self) -> str:
        return (
            f"INSERT INTO {self._quoted_table_name} "
            f"({self._quoted_session_id_key}, {self._quoted_history_key}) "
            "VALUES (:1, :2)"
        )

    @property
    def _delete_query(self) -> str:
        return (
            f"DELETE FROM {self._quoted_table_name} "
            f"WHERE {self._quoted_session_id_key} = :session_id"
        )

    @property
    def _quoted_table_name(self) -> str:
        return _quote_indentifier(self._table_name)

    @property
    def _quoted_session_id_key(self) -> str:
        return _quote_indentifier(self._session_id_key)

    @property
    def _quoted_history_key(self) -> str:
        return _quote_indentifier(self._history_key)


__all__ = ["OracleChatMessageHistory"]
