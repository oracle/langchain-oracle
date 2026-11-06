# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Table annotations on creation and reopen."""

import os
import uuid
from unittest.mock import patch

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
from langchain_oracledb._annotations import table_annotations


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
