# Copyright (c) 2025, 2026, Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import hashlib
import json
import os
import uuid
from collections.abc import Generator

import oracledb
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langchain_oracledb.chat_message_histories import OracleChatMessageHistory
from langchain_oracledb.vectorstores.utils import _index_exists, _quote_indentifier

username = os.environ.get("VECDB_USER")
password = os.environ.get("VECDB_PASS")
dsn = os.environ.get("VECDB_HOST")

try:
    oracledb.connect(user=username, password=password, dsn=dsn)
except Exception as e:
    pytest.skip(
        allow_module_level=True,
        reason=f"Database connection failed: {e}, skipping tests.",
    )


@pytest.fixture(scope="function")
def connection() -> Generator[oracledb.Connection, None, None]:
    conn = oracledb.connect(user=username, password=password, dsn=dsn)
    try:
        yield conn
    finally:
        try:
            conn.close()
        except Exception:
            pass


@pytest.fixture(scope="function")
def history_resources(
    connection: oracledb.Connection,
) -> Generator[dict[str, str], None, None]:
    suffix = uuid.uuid4().hex[:8].upper()
    resources = {
        "table_name": f"CHAT_HISTORY_{suffix}",
        "session_id": f"session-{suffix}",
        "other_session_id": f"other-session-{suffix}",
    }
    try:
        yield resources
    finally:
        try:
            OracleChatMessageHistory.drop_table(connection, resources["table_name"])
        except Exception:
            pass


def test_chat_history_create_add_get_and_clear(
    connection: oracledb.Connection, history_resources: dict[str, str]
) -> None:
    history = OracleChatMessageHistory(
        session_id=history_resources["session_id"],
        client=connection,
        table_name=history_resources["table_name"],
    )

    assert history.messages == []

    messages = [
        HumanMessage(content="hello"),
        AIMessage(content="world"),
        HumanMessage(content="goodbye"),
    ]
    history.add_messages(messages)

    assert history.messages == messages

    history.clear()

    assert history.messages == []


def test_chat_history_limits_to_session_and_history_size(
    connection: oracledb.Connection, history_resources: dict[str, str]
) -> None:
    table_name = history_resources["table_name"]

    primary_history = OracleChatMessageHistory(
        session_id=history_resources["session_id"],
        client=connection,
        table_name=table_name,
        history_size=2,
    )
    secondary_history = OracleChatMessageHistory(
        session_id=history_resources["other_session_id"],
        client=connection,
        table_name=table_name,
        create_table=False,
        create_index=False,
    )

    primary_messages = [
        HumanMessage(content="one"),
        AIMessage(content="two"),
        HumanMessage(content="three"),
    ]
    secondary_messages = [HumanMessage(content="isolated")]

    primary_history.add_messages(primary_messages)
    secondary_history.add_messages(secondary_messages)

    assert primary_history.messages == primary_messages[-2:]

    unbounded_primary = OracleChatMessageHistory(
        session_id=history_resources["session_id"],
        client=connection,
        table_name=table_name,
        create_table=False,
        create_index=False,
    )
    assert unbounded_primary.messages == primary_messages
    assert secondary_history.messages == secondary_messages


def test_chat_history_messages_setter_replaces_session_history(
    connection: oracledb.Connection, history_resources: dict[str, str]
) -> None:
    history = OracleChatMessageHistory(
        session_id=history_resources["session_id"],
        client=connection,
        table_name=history_resources["table_name"],
    )

    history.add_messages(
        [HumanMessage(content="before"), AIMessage(content="before-response")]
    )
    history.messages = [SystemMessage(content="replacement")]

    assert history.messages == [SystemMessage(content="replacement")]


def test_chat_history_empty_batch_is_noop(
    connection: oracledb.Connection, history_resources: dict[str, str]
) -> None:
    history = OracleChatMessageHistory(
        session_id=history_resources["session_id"],
        client=connection,
        table_name=history_resources["table_name"],
    )

    history.add_messages([])

    assert history.messages == []


def test_chat_history_create_tables_requires_existing_table_when_only_index_requested(
    connection: oracledb.Connection, history_resources: dict[str, str]
) -> None:
    with pytest.raises(
        ValueError,
        match="Set create_table=True or create the table first",
    ):
        OracleChatMessageHistory.create_tables(
            connection,
            history_resources["table_name"],
            create_table=False,
            create_index=True,
        )


def test_chat_history_create_tables_supports_custom_columns_and_index(
    connection: oracledb.Connection, history_resources: dict[str, str]
) -> None:
    table_name = history_resources["table_name"]
    session_column = "conversation_id"
    history_column = "payload"

    OracleChatMessageHistory.create_tables(
        connection,
        table_name,
        session_id_key=session_column,
        history_key=history_column,
    )

    expected_index_name = (
        f"idx_{table_name}_{session_column}"
        if len(f"idx_{table_name}_{session_column}") <= 128
        else (
            f"{f'idx_{table_name}_{session_column}'[:119]}_"
            f"{hashlib.sha1(f'idx_{table_name}_{session_column}'.encode('utf-8')).hexdigest()[:8]}"
        )
    )

    with connection.cursor() as cursor:
        quoted_table_name = _quote_indentifier(table_name)
        quoted_session_column = _quote_indentifier(session_column)
        quoted_history_column = _quote_indentifier(history_column)
        cursor.execute(
            f"INSERT INTO {quoted_table_name} "
            f"({quoted_session_column}, {quoted_history_column}) "
            "VALUES (:1, :2)",
            [
                history_resources["session_id"],
                json.dumps(
                    {
                        "type": "human",
                        "data": {
                            "content": "custom-columns",
                            "additional_kwargs": {},
                            "response_metadata": {},
                        },
                    }
                ),
            ],
        )
        connection.commit()

    history = OracleChatMessageHistory(
        session_id=history_resources["session_id"],
        client=connection,
        table_name=table_name,
        session_id_key=session_column,
        history_key=history_column,
        create_table=False,
        create_index=False,
    )

    assert history.messages == [HumanMessage(content="custom-columns")]
    assert _index_exists(
        connection,
        _quote_indentifier(expected_index_name),
        table_name,
    )
