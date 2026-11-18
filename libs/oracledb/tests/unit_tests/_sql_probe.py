# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""Model Oracle parsing the explicit columns used by the SQL backend."""

from types import SimpleNamespace

import oracledb


def sql_probe(statement, columns, error=None):
    sql = statement.startswith("SELECT id, text, metadata, embedding FROM ")
    vecdb = statement.startswith("SELECT id, dense_vector, ")
    if not sql and not vecdb:
        return False
    if error is not None:
        raise error
    if not columns:
        raise oracledb.DatabaseError(SimpleNamespace(code=942))
    expected = (
        {"ID", "TEXT", "METADATA", "EMBEDDING"}
        if sql
        else {
            "ID",
            "DENSE_VECTOR",
            "CONTENT_METADATA" if "CONTENT_METADATA" in statement else "METADATA",
        }
    )
    if not expected <= set(columns):
        raise oracledb.DatabaseError(SimpleNamespace(code=904))
    return True
