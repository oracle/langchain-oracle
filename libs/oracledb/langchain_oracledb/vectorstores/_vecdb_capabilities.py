# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""Non-mutating checks for the native VecDB operation about to execute."""

import re
from typing import Any


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


def sdk_index_state(response: dict) -> str:
    state = str(response.get("index_status") or "").upper()
    # ORDS can return a sentence containing the quoted owner/index and type,
    # rather than the bare status. Only normalize the observed success grammar.
    found = re.fullmatch(
        r'"(?:[^"\r\n]|"")+"\."(?:[^"\r\n]|"")+" (?:HNSW|IVF) '
        r"(INDEX CREATION COMPLETED)",
        state,
    )
    return found.group(1) if found else state


def sdk_index_status(response: dict, job: bool) -> dict[str, Any]:
    state = sdk_index_state(response)
    ready = None
    if not job:
        if state in {"READY", "VALID", "INDEX CREATION COMPLETED"}:
            ready = True
        elif state in {"BUILDING", "PENDING", "FAILED", "UNUSABLE"}:
            ready = False
    # A completed job is not evidence that an index is usable. Unknown values
    # remain unknown; preserve the complete public SDK response for callers.
    return {"ready": ready, "details": response}
