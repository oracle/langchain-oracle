# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Stable SQL statement markers; no request/user data in tags."""

import re

import oracledb

DEVELOPERHUB_PROGRAM = "devrel-developerhub-langchain"

SQL_COMMENTS = {
    "vector": "/* lc_oracledb mode=vector */",
    "text": "/* lc_oracledb mode=text */",
    "hybrid": "/* lc_oracledb mode=hybrid */",
    "keyword": "/* lc_oracledb mode=keyword */",
    "semantic": "/* lc_oracledb mode=semantic */",
    "embedding": "/* lc_oracledb operation=embedding */",
    "rerank": "/* lc_oracledb operation=rerank */",
}

_STATEMENT_START = re.compile(
    r"\A(\s*(?:SELECT|BEGIN|DECLARE|WITH)\b(?:\s*/\*\+.*?\*/)?)",
    re.IGNORECASE | re.DOTALL,
)


def tag_sql(statement: str, operation: str, *, backend: str = "sql") -> str:
    """Tag generated executable SQL while leaving an existing optimizer hint first."""
    if operation not in SQL_COMMENTS:
        raise ValueError("Unsupported LangChain telemetry operation")
    if backend not in {"sql", "vecdb"}:
        raise ValueError("Unsupported LangChain telemetry backend")
    fields = f" backend={backend}"
    if getattr(oracledb.defaults, "program", None) == DEVELOPERHUB_PROGRAM:
        fields += f" source={DEVELOPERHUB_PROGRAM}"
    match = _STATEMENT_START.match(statement)
    if match is None:
        raise ValueError("Unsupported LangChain telemetry statement")
    return (
        statement[: match.end()]
        + " "
        + SQL_COMMENTS[operation].replace(" */", fields + " */")
        + statement[match.end() :]
    )
