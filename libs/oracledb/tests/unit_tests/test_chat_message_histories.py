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


def test_message_payload_rejects_non_string_values() -> None:
    with pytest.raises(TypeError, match="Expected Oracle chat history payload"):
        _message_payload({"not": "a string"})


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
