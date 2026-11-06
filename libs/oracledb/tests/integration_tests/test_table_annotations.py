# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Table annotations on creation and reopen."""

import os
import uuid
from unittest.mock import MagicMock, patch

import oracledb
import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from langchain_oracledb import (
    OracleByteStore,
    OracleCache,
    OracleChatMessageHistory,
    OracleDocStore,
    OracleSemanticCache,
    OracleVS,
)
from langchain_oracledb._annotations import (
    create_table_with_annotations,
    table_annotations,
)


def test_sql_classes_create_annotations_and_preserve_them_on_reopen(sql_connection):
    conn = sql_connection
    prefix = "LCCA_" + uuid.uuid4().hex[:10].upper()
    names = [f"{prefix}_{i}" for i in range(6)]
    embedding = DeterministicFakeEmbedding(size=3)
    factories = [
        lambda: OracleVS(conn, embedding, names[0]),
        lambda: OracleSemanticCache(conn, embedding, names[1]),
        lambda: OracleCache(conn, names[2]),
        lambda: OracleChatMessageHistory.create_tables(
            conn, names[3], create_index=False
        ),
        lambda: OracleByteStore(conn, names[4]),
        lambda: OracleDocStore(conn, names[5]),
    ]
    kinds = [
        "vector_store",
        "semantic_cache",
        "exact_cache",
        "chat_history",
        "kv_store",
        "kv_store",
    ]
    expected = {
        name: table_annotations(kind)["LANGCHAIN_ORACLE"]
        for name, kind in zip(names, kinds)
    }
    with conn.cursor() as cursor:
        try:
            for factory in factories:
                factory()
            cursor.execute(
                "SELECT object_name, annotation_value FROM user_annotations_usage "
                "WHERE object_type='TABLE' AND column_name IS NULL "
                "AND annotation_name='LANGCHAIN_ORACLE' AND object_name LIKE :p",
                p=prefix + "%",
            )
            assert dict(cursor.fetchall()) == expected
            with patch("langchain_oracledb._annotations.version", return_value="999.0"):
                for factory in factories:
                    factory()
            cursor.execute(
                "SELECT object_name, annotation_value FROM user_annotations_usage "
                "WHERE object_type='TABLE' AND column_name IS NULL "
                "AND annotation_name='LANGCHAIN_ORACLE' AND object_name LIKE :p",
                p=prefix + "%",
            )
            assert dict(cursor.fetchall()) == expected
        finally:
            for name in names:
                cursor.execute(
                    "SELECT COUNT(*) FROM user_tables WHERE table_name=:n", n=name
                )
                if cursor.fetchone()[0]:
                    cursor.execute(f'DROP TABLE "{name}" PURGE')


@pytest.fixture
def sql_connection():
    if not os.environ.get("VECDB_HOST"):
        pytest.skip("Set VECDB_* for live Oracle checks")
    with oracledb.connect(
        user=os.environ["VECDB_USER"],
        password=os.environ["VECDB_PASS"],
        dsn=os.environ["VECDB_HOST"],
        tcp_connect_timeout=10,
        retry_count=0,
    ) as connection:
        yield connection


def test_plain_creation_after_a_real_server_clause_syntax_error(sql_connection):
    # The test database supports annotations. Replace only the appended keyword
    # to exercise a real server parse error, not to claim an older DB version.
    name = "LCFALL_" + uuid.uuid4().hex[:16].upper()
    errors = []
    with sql_connection.cursor() as cursor:
        proxy = MagicMock(wraps=cursor)

        def execute(statement):
            sent = statement.replace(" ANNOTATIONS (", " ANNOTATIONS_UNSUPPORTED (", 1)
            try:
                return cursor.execute(sent)
            except oracledb.DatabaseError as error:
                errors.append(error.args[0])
                raise

        proxy.execute.side_effect = execute
        try:
            create_table_with_annotations(
                proxy, f'CREATE TABLE "{name}" (id NUMBER)', "kv_store"
            )
            assert proxy.execute.call_count == 2
            assert len(errors) == 1
            assert errors[0].code in {922, 933, 3049}
            cursor.execute(
                "SELECT COUNT(*) FROM user_tables WHERE table_name=:n", n=name
            )
            assert cursor.fetchone()[0] == 1
            cursor.execute(
                "SELECT COUNT(*) FROM user_annotations_usage "
                "WHERE object_name=:n AND annotation_name='LANGCHAIN_ORACLE'",
                n=name,
            )
            assert cursor.fetchone()[0] == 0
        finally:
            cursor.execute(
                "SELECT COUNT(*) FROM user_tables WHERE table_name=:n", n=name
            )
            if cursor.fetchone()[0]:
                cursor.execute(f'DROP TABLE "{name}" PURGE')
