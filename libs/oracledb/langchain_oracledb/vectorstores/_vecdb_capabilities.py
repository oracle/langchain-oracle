# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""Non-mutating checks for the native VecDB operation about to execute."""

from typing import Any

import oracledb

INDEX_FEATURES_SQL = (
    "SELECT object_name, procedure_name FROM all_procedures "
    "WHERE (owner='SYS' AND object_name='DBMS_VECTOR_DATABASE' "
    "AND procedure_name='HANDLE_MVI_INDEX') "
    "OR (owner='CTXSYS' AND object_name='DBMS_HYBRID_VECTOR' "
    "AND procedure_name='SEARCH')"
)


def function_preflight(statement: str) -> str:
    """Compile the exact block and bind types without invoking its operations.

    Only adapter-generated PL/SQL is accepted. Never parse/execute DDL to discover
    whether it is supported. The unreachable branch still resolves signatures.
    """
    prefix = statement.lstrip().split(maxsplit=1)
    if not prefix or prefix[0].upper() not in {"BEGIN", "DECLARE"}:
        raise ValueError("VecDB signature preflight requires an adapter PL/SQL block")
    return f"BEGIN IF FALSE THEN {statement} END IF; END;"


def signature_error(operation: str, version: str, error: Exception) -> None:
    # Only compilation failures establish an unavailable signature. Permission,
    # network, and unknown execution errors retain their original driver types.
    if (
        isinstance(error, oracledb.DatabaseError)
        and getattr(error.args[0], "code", None) == 6550
    ):
        raise ValueError(
            f"Native VecDB operation {operation!r} cannot compile its expected "
            f"signature on Oracle Database {version}; verify the package "
            "signature and execute privileges. No operation was executed."
        ) from error
    raise error


def require_index_features(found: list, hybrid: bool) -> None:
    required = {("DBMS_VECTOR_DATABASE", "HANDLE_MVI_INDEX")}
    if hybrid:
        required.add(("DBMS_HYBRID_VECTOR", "SEARCH"))
    missing = required - {tuple(row) for row in found}
    if missing:
        names = ", ".join(f"{package}.{method}" for package, method in sorted(missing))
        raise ValueError(
            f"Native VecDB {'hybrid' if hybrid else 'metadata'} indexing requires "
            f"visible package operations: {names}. No index DDL was executed."
        )


def index_status_query(owner: str, kind: str) -> str:
    if kind not in {"vector", "metadata"}:
        raise ValueError("Index kind must be 'vector' or 'metadata'")
    predicate = (
        "index_type='VECTOR'"
        if kind == "vector"
        else (
            "ityp_owner='CTXSYS' AND ityp_name IN ('CONTEXT','CONTEXT_V2') "
            "AND EXISTS (SELECT 1 FROM all_ind_columns c "
            "WHERE c.index_owner=i.owner AND c.index_name=i.index_name "
            "AND c.table_owner=i.table_owner AND c.table_name=i.table_name "
            "AND c.column_name IN ('METADATA','CONTENT_METADATA'))"
        )
    )
    return (
        "SELECT owner, index_name, status, domidx_status, domidx_opstatus "
        f"FROM all_indexes i WHERE table_owner={owner} AND table_name=:name "
        f"AND ({predicate}) ORDER BY owner,index_name"
    )


def native_index_status(found: list) -> dict[str, Any]:
    indexes = [
        dict(
            zip(
                ("owner", "index_name", "status", "domidx_status", "domidx_opstatus"),
                row,
            )
        )
        for row in found
    ]
    valid = all(
        row["status"] == "VALID"
        and row["domidx_status"] in {None, "VALID"}
        and row["domidx_opstatus"] in {None, "VALID"}
        for row in indexes
    )
    failed_or_pending = any(
        row["status"] in {"UNUSABLE", "INVALID", "FAILED", "IN_PROGRESS"}
        or row["domidx_status"] in {"INVALID", "FAILED", "IN_PROGRESS"}
        or row["domidx_opstatus"] in {"INVALID", "FAILED", "IN_PROGRESS"}
        for row in indexes
    )
    # Partition-level readiness is not inferred from a parent status of N/A.
    ready = False if not indexes or failed_or_pending else True if valid else None
    return {"ready": ready, "details": {"indexes": indexes}}


def sdk_index_status(response: dict, job: bool) -> dict[str, Any]:
    state = str(response.get("index_status") or "").upper()
    ready = None
    if not job:
        if state in {"READY", "VALID", "INDEX CREATION COMPLETED"}:
            ready = True
        elif state in {"BUILDING", "PENDING", "FAILED", "UNUSABLE"}:
            ready = False
    # A completed job is not evidence that an index is usable. Unknown values
    # remain unknown; preserve the complete public SDK response for callers.
    return {"ready": ready, "details": response}
