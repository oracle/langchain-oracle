# Copyright (c) 2025, 2026, Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from contextlib import contextmanager

import pytest
from langchain_core.messages import HumanMessage

from langchain_oracledb import chat_message_histories
from langchain_oracledb.chat_message_histories import (
    OracleChatMessageHistory,
    _default_index_name,
    _message_payload,
)


def test_chat_history_validates_constructor_arguments() -> None:
    with pytest.raises(ValueError, match="session_id must be a string"):
        OracleChatMessageHistory(
            session_id=123,  # type: ignore[arg-type]
            client=object(),
            create_table=False,
            create_index=False,
        )

    with pytest.raises(ValueError, match="client must be provided"):
        OracleChatMessageHistory(
            session_id="session-1",
            client=None,
            create_table=False,
            create_index=False,
        )

    with pytest.raises(ValueError, match="history_size must be greater than 0"):
        OracleChatMessageHistory(
            session_id="session-1",
            client=object(),
            history_size=0,
            create_table=False,
            create_index=False,
        )


def test_chat_history_default_index_name_is_truncated_when_needed() -> None:
    table_name = "table_" + ("x" * 80)
    session_column = "session_" + ("y" * 80)

    index_name = _default_index_name(table_name, session_column)

    assert len(index_name) <= 128
    assert index_name.startswith("idx_")


def test_chat_history_default_index_name_avoids_table_column_collisions() -> None:
    first = _default_index_name("customer_support_chat", "session_id")
    second = _default_index_name("customer_support", "chat_session_id")
    first_again = _default_index_name("customer_support_chat", "session_id")

    long_name = _default_index_name("table_" + ("x" * 80), "session_" + ("y" * 80))
    multibyte = _default_index_name("T_" + ("é" * 50), "S_" + ("x" * 70))
    boundary = _default_index_name(("T" * 114) + "éé", "x" * 20)

    assert first != second
    assert first == first_again
    for name in (first, second, long_name, multibyte, boundary):
        encoded = name.encode("utf-8")
        assert len(encoded) <= 128
        prefix, sep, digest = name.rpartition("_")
        assert sep == "_"
        assert len(digest) == 8
        assert len(prefix.encode("utf-8")) <= 119
        assert "\ufffd" not in name
        assert name.startswith("idx_")


def test_message_payload_rejects_non_string_values() -> None:
    with pytest.raises(TypeError, match="Expected Oracle chat history payload"):
        _message_payload({"not": "a string"})


def _create_index_statements(
    events: list[tuple[str, object, object] | str],
) -> list[str]:
    return [
        query
        for event in events
        if isinstance(event, tuple) and event[0] == "execute"
        for query in [event[1]]
        if isinstance(query, str) and "CREATE INDEX" in query.upper()
    ]


def _fake_history_connection(
    events: list[tuple[str, object, object] | str],
    *,
    existing_index_names: set[str] | None = None,
    indexed_columns: list[dict[str, object]] | None = None,
    current_schema: str = "TEST_SCHEMA",
):
    existing_index_names = existing_index_names or set()
    indexed_columns = indexed_columns or []

    class FakeCursor:
        def __init__(self) -> None:
            self._query = ""
            self._binds: dict[str, object] = {}

        def __enter__(self) -> "FakeCursor":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def execute(self, query: str, params: object = None, **binds: object) -> None:
            if not binds and isinstance(params, dict):
                binds = params
            self._query = query
            self._binds = binds
            events.append(("execute", query, binds or params))

        def fetchone(self) -> object:
            query = self._query.lower()
            if "all_indexes" in query:
                name = self._binds.get("idx_name")
                if name in existing_index_names:
                    return (name,)
                return None
            if "all_ind_columns" in query:
                table_name = self._binds.get("table_name")
                column_name = self._binds.get("column_name")
                owner = self._binds.get("owner")
                filters_owner = "table_owner" in query
                filters_position = "column_position" in query
                if owner is None and "sys_context" in query:
                    owner = current_schema
                for entry in indexed_columns:
                    if entry["table_name"] != table_name:
                        continue
                    if entry["column_name"] != column_name:
                        continue
                    if filters_owner and entry["owner"] != owner:
                        continue
                    if filters_position and entry["column_position"] != 1:
                        continue
                    return (1,)
                return None
            return None

    class FakeConnection:
        def cursor(self) -> FakeCursor:
            return FakeCursor()

        def commit(self) -> None:
            events.append("commit")

        def rollback(self) -> None:
            events.append("rollback")

    return FakeConnection()


def test_create_tables_skips_index_when_legacy_session_index_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, object, object] | str] = []
    table_name = "customer_support_chat"
    session_column = "session_id"
    legacy_index_name = f"idx_{table_name}_{session_column}"

    @contextmanager
    def fake_get_connection(client: object):
        assert client is fake_client
        yield _fake_history_connection(
            events,
            existing_index_names={legacy_index_name},
        )

    fake_client = object()
    monkeypatch.setattr(chat_message_histories, "_get_connection", fake_get_connection)

    OracleChatMessageHistory(
        session_id="session-1",
        client=fake_client,
        table_name=table_name,
        session_id_key=session_column,
        create_table=False,
        create_index=True,
    )

    assert _create_index_statements(events) == []


def test_create_tables_creates_hashed_index_when_session_column_unindexed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, object, object] | str] = []
    table_name = "customer_support_chat"
    session_column = "session_id"
    hashed_index_name = _default_index_name(table_name, session_column)

    @contextmanager
    def fake_get_connection(client: object):
        assert client is fake_client
        yield _fake_history_connection(events)

    fake_client = object()
    monkeypatch.setattr(chat_message_histories, "_get_connection", fake_get_connection)

    OracleChatMessageHistory(
        session_id="session-1",
        client=fake_client,
        table_name=table_name,
        session_id_key=session_column,
        create_table=False,
        create_index=True,
    )

    create_index_statements = _create_index_statements(events)
    assert len(create_index_statements) == 1
    assert hashed_index_name in create_index_statements[0]


def test_create_tables_indexes_when_session_column_indexed_in_other_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, object, object] | str] = []
    table_name = "customer_support_chat"
    session_column = "session_id"
    hashed_index_name = _default_index_name(table_name, session_column)

    @contextmanager
    def fake_get_connection(client: object):
        assert client is fake_client
        yield _fake_history_connection(
            events,
            indexed_columns=[
                {
                    "owner": "OTHER_SCHEMA",
                    "table_name": table_name,
                    "column_name": session_column,
                    "column_position": 1,
                }
            ],
        )

    fake_client = object()
    monkeypatch.setattr(chat_message_histories, "_get_connection", fake_get_connection)

    OracleChatMessageHistory(
        session_id="session-1",
        client=fake_client,
        table_name=table_name,
        session_id_key=session_column,
        create_table=False,
        create_index=True,
    )

    create_index_statements = _create_index_statements(events)
    assert len(create_index_statements) == 1
    assert hashed_index_name in create_index_statements[0]


def test_create_tables_indexes_when_session_column_is_not_leading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, object, object] | str] = []
    table_name = "customer_support_chat"
    session_column = "session_id"
    hashed_index_name = _default_index_name(table_name, session_column)

    @contextmanager
    def fake_get_connection(client: object):
        assert client is fake_client
        yield _fake_history_connection(
            events,
            indexed_columns=[
                {
                    "owner": "TEST_SCHEMA",
                    "table_name": table_name,
                    "column_name": session_column,
                    "column_position": 2,
                }
            ],
        )

    fake_client = object()
    monkeypatch.setattr(chat_message_histories, "_get_connection", fake_get_connection)

    OracleChatMessageHistory(
        session_id="session-1",
        client=fake_client,
        table_name=table_name,
        session_id_key=session_column,
        create_table=False,
        create_index=True,
    )

    create_index_statements = _create_index_statements(events)
    assert len(create_index_statements) == 1
    assert hashed_index_name in create_index_statements[0]


def test_messages_setter_rolls_back_when_insert_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, object, object] | str] = []

    class FakeCursor:
        def __enter__(self) -> "FakeCursor":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def execute(self, query: str, params: object = None) -> None:
            events.append(("execute", query, params))

        def executemany(self, query: str, rows: object) -> None:
            events.append(("executemany", query, rows))
            raise RuntimeError("insert failed")

    class FakeConnection:
        def cursor(self) -> FakeCursor:
            return FakeCursor()

        def commit(self) -> None:
            events.append("commit")

        def rollback(self) -> None:
            events.append("rollback")

    @contextmanager
    def fake_get_connection(client: object):
        assert client is fake_client
        yield FakeConnection()

    fake_client = object()
    monkeypatch.setattr(chat_message_histories, "_get_connection", fake_get_connection)

    history = OracleChatMessageHistory(
        session_id="session-1",
        client=fake_client,
        create_table=False,
        create_index=False,
    )

    with pytest.raises(RuntimeError, match="insert failed"):
        history.messages = [HumanMessage(content="replacement")]

    assert isinstance(events[0], tuple)
    assert isinstance(events[1], tuple)
    assert events[0][0] == "execute"
    assert events[1][0] == "executemany"
    assert events[-1] == "rollback"
    assert "commit" not in events
