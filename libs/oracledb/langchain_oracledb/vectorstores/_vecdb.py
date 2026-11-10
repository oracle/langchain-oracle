# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""Private VecDB transport, layout compatibility, and document mapping.

REST uses the optional public SDK; native calls use DBMS_VECTOR_DATABASE.
There are no SDK imports on the SQL-only import path.
"""

from __future__ import annotations

import array
import asyncio
import json
import logging
import math
import re
from contextlib import asynccontextmanager, contextmanager
from copy import copy
from dataclasses import dataclass
from decimal import Decimal
from functools import wraps
from typing import Any, AsyncIterator, Callable, Iterator, Optional, TypeVar, cast

import numpy as np
import oracledb
from langchain_core.documents import Document

from .._annotations import TableType, table_annotations
from .._telemetry import tag_sql
from .._vecdb_sdk.parameter_validation import validate_operation_arguments
from ._vecdb_adapter import (
    CURRENT_VERSION,
    LOGICAL_ID_KEY,
    TEXT_KEY,
    SQLNetAdapter,
    _hybrid_documents,
    _text_documents,
    as_dict,
    document,
    hybrid_search_request,
    rows,
    search_request,
    storage_id,
    validate_hybrid_request,
)
from ._vecdb_capabilities import (
    INDEX_FEATURES_SQL,
    function_preflight,
    index_status_query,
    native_index_status,
    require_index_features,
    sdk_index_state,
    sdk_index_status,
    signature_error,
)
from ._vecdb_filters import SQLFilterRequired
from .utils import (
    DistanceStrategy,
    _aget_connection,
    _get_connection,
    _quote_indentifier,
)

logger = logging.getLogger(__name__)


def _filter_query(
    name: str, metadata_column: str, request: dict, filter: dict
) -> tuple:
    from .oraclevs import _generate_where_clause

    quoted, _ = table_identifier(name)
    values: list = []
    predicate = _generate_where_clause(filter, values)
    metric = request["advanced_options"]["distance_metric"]
    if metric not in {"COSINE", "EUCLIDEAN", "DOT"}:
        raise ValueError("Unsupported vector distance metric")
    # Alias the physical metadata column instead of replacing SQL text: a
    # caller's JSON path or value can itself contain the word 'metadata'.
    sql = f"""SELECT id, metadata,
        VECTOR_DISTANCE(dense_vector, :embedding, {metric}) AS distance
        {", dense_vector" if request["include_vectors"] else ""}
        FROM (SELECT id, dense_vector, {metadata_column} AS metadata FROM {quoted})
        WHERE {predicate}
        ORDER BY distance FETCH FIRST :k ROWS ONLY"""
    binds = {f"value{i}": value for i, value in enumerate(values)}
    binds.update(
        embedding=array.array("f", request["query_by"]["vector"]), k=request["top_k"]
    )
    return tag_sql(sql, "vector", backend="vecdb"), binds


def _filter_response(result: list, request: dict) -> dict:
    hits = []
    selector = request.get("output_selector")
    for id_, metadata, distance, *vector in result:
        if selector is not None:
            metadata = {
                key: value for key, value in metadata.items() if key in selector
            }
        hit = {"id": id_, "metadata": metadata, "distance": distance}
        if request["include_vectors"]:
            hit["vector"] = list(vector[0])
        hits.append(hit)
    return {"results": hits}


LEGACY_VERSION = (23, 26, 1)
DENSE_COMPATIBLE = (23, 4, 0)
SPARSE_COMPATIBLE = (23, 6, 0)
COMPATIBILITY_SQL = "BEGIN DBMS_UTILITY.DB_VERSION(:db_version, :compatibility); END;"
# Initialization needs SEARCH capability and COMPATIBLE in the same round trip.
VECDB_CAPABILITY_SQL = """
WITH FUNCTION lc_compatible RETURN VARCHAR2 IS
    db_version VARCHAR2(128);
    compatibility VARCHAR2(128);
BEGIN
    SYS.DBMS_UTILITY.DB_VERSION(db_version, compatibility);
    RETURN compatibility;
END;
SELECT COUNT(*) AS search_supported,
       CASE WHEN COUNT(*) > 0 THEN lc_compatible() END AS compatibility
FROM all_procedures
WHERE owner = 'SYS' AND object_name = 'DBMS_VECTOR_DATABASE'
  AND procedure_name = 'SEARCH'
"""
BASE_COLUMNS = {"ID", "DENSE_VECTOR"}
CURRENT_COLUMNS = BASE_COLUMNS | {
    "CONTENT_METADATA",
    "SPARSE_VECTOR",
    "CONTENT",
    "CONTENT_TYPE",
}


def parse_version(version: str) -> tuple[int, ...]:
    matches = re.findall(r"(?<!\d)(\d+)\.(\d+)\.(\d+)", version)
    if not matches:
        raise ValueError("Cannot determine the Oracle Database version")
    return tuple(map(int, matches[-1]))


def table_identifier(name: str) -> tuple[str, str]:
    """Return quoted SQL and REST API forms; never rebuild SQL from the latter."""
    quoted = _quote_indentifier(name)
    return quoted, quoted.replace('"', "")


def _identifier_parts(name: str) -> list[str]:
    """Split only at unquoted dots, using the shared SQL identifier rules."""
    quoted, _ = table_identifier(name)
    parts = re.findall(r'"([^"]+)"', quoted)
    if len(parts) not in {1, 2}:
        raise ValueError("Object names must contain a name and optional schema")
    return parts


def _catalog_target(name: str) -> tuple[str, dict]:
    """Match Oracle SQL name resolution without changing session state."""
    parts = _identifier_parts(name)
    binds = {"name": parts[-1]}
    if len(parts) == 2:
        binds["owner"] = parts[0]
        return ":owner", binds
    return "SYS_CONTEXT('USERENV', 'CURRENT_SCHEMA')", binds


def _layout_query(name: str) -> tuple[str, dict]:
    """Inspect the exact SQL object in its explicit or current schema."""
    owner, binds = _catalog_target(name)
    return (
        "SELECT column_name FROM all_tab_columns "
        f"WHERE owner = {owner} AND table_name = :name",
        binds,
    )


def _synonym_query(name: str) -> tuple[str, dict]:
    owner, binds = _catalog_target(name)
    owners = f"owner = {owner}" if "owner" in binds else f"owner IN ({owner}, 'PUBLIC')"
    return (
        "SELECT 1 FROM all_synonyms WHERE synonym_name = :name AND " + owners,
        binds,
    )


def _inspect_columns(cursor: Any, name: str) -> set[str]:
    statement, binds = _layout_query(name)
    cursor.execute(statement, **binds)
    columns = {row[0] for row in cursor.fetchall()}
    if columns:
        return columns
    # Synonyms have no catalog columns. Let Oracle resolve the actual SQL
    # object, including public/chained synonyms, without reading its rows.
    try:
        cursor.execute(f"SELECT * FROM {_quote_indentifier(name)} WHERE 1 = 0")
    except oracledb.DatabaseError as error:
        if getattr(error.args[0], "code", None) != 942:
            raise
        statement, binds = _synonym_query(name)
        cursor.execute(statement, **binds)
        if cursor.fetchone():
            # A synonym that cannot be read is not a missing table.
            raise
        return set()
    return {column[0] for column in cursor.description or ()}


async def _ainspect_columns(cursor: Any, name: str) -> set[str]:
    statement, binds = _layout_query(name)
    await cursor.execute(statement, **binds)
    columns = {row[0] for row in await cursor.fetchall()}
    if columns:
        return columns
    try:
        await cursor.execute(f"SELECT * FROM {_quote_indentifier(name)} WHERE 1 = 0")
    except oracledb.DatabaseError as error:
        if getattr(error.args[0], "code", None) != 942:
            raise
        statement, binds = _synonym_query(name)
        await cursor.execute(statement, **binds)
        if await cursor.fetchone():
            raise
        return set()
    return {column[0] for column in cursor.description or ()}


def _package_name_error(name: str) -> Optional[str]:
    parts = _identifier_parts(name)
    if len(parts) != 1:
        return (
            "Native VecDB package operations do not support "
            f"schema-qualified table names and cannot preserve {name!r}. "
            "Use an unqualified name with CURRENT_SCHEMA set to the owning "
            "schema. Full support for this identifier requires support in "
            "the underlying VecDB package."
        )
    return None


def _metadata_index_query(name: str, column: str) -> tuple[str, dict]:
    owner, binds = _catalog_target(name)
    binds["metadata_column"] = column
    return (
        "SELECT i.index_name, i.parameters, i.owner FROM all_indexes i "
        "JOIN all_ind_columns c ON c.index_owner = i.owner "
        "AND c.index_name = i.index_name AND c.table_owner = i.table_owner "
        "AND c.table_name = i.table_name "
        f"WHERE i.table_owner = {owner} AND i.table_name = :name "
        "AND c.column_name = :metadata_column AND i.ityp_owner = 'CTXSYS' "
        "AND i.ityp_name IN ('CONTEXT', 'CONTEXT_V2')",
        binds,
    )


def _index_query(index: str) -> tuple[str, dict]:
    owner, binds = _catalog_target(index)
    return (
        f"SELECT index_name FROM all_indexes WHERE owner = {owner} "
        "AND index_name = :name",
        binds,
    )


def _index_identifier(table: str, index: str) -> str:
    """An unqualified new index belongs to the explicitly named table schema."""
    table_parts = _identifier_parts(table)
    parts = _identifier_parts(index)
    if len(parts) == 1 and len(table_parts) == 2:
        return _quote_indentifier(f'"{table_parts[0]}"."{parts[0]}"')
    return _quote_indentifier(index)


def _catalog_index_identifier(table: str, existing: tuple) -> str:
    """Catalog names are single components, even when they contain dots."""
    owner = [existing[2]] if len(existing) > 2 else _identifier_parts(table)[:-1]
    return _quote_indentifier(".".join(f'"{part}"' for part in [*owner, existing[0]]))


def _text_enabled(parameters: Optional[str]) -> bool:
    if "SIMPLIFIED_JSON_HYBRID" in (parameters or "").upper():
        return True
    return bool(
        re.search(
            r"\bsearch_on\s+text(?:_value(?:_string)?)?\b",
            parameters or "",
            re.IGNORECASE,
        )
    )


DATAGUIDE_CALLBACK = "ON CHANGE SYS.DBMS_VECTOR_DATABASE.HANDLE_MVI_INDEX"
DEFAULT_METADATA_PARAMETERS = (
    f'DATAGUIDE {DATAGUIDE_CALLBACK} SYNC (EVERY "freq=secondly; interval=30")'
)


def _metadata_parameters(parameters: str, hybrid: bool) -> str:
    """Retain catalog settings, excluding the index-kind marker/search mode."""
    if not re.match(r"^SIMPLIFIED_JSON\b", parameters.strip(), re.IGNORECASE):
        raise ValueError("Only a recognized VecDB JSON metadata index can be replaced")
    parameters = re.sub(r"^\s*SIMPLIFIED_JSON\s*", "", parameters, flags=re.IGNORECASE)
    callback = re.search(
        r"\bDATAGUIDE\s+ON\s+CHANGE\s+([\w.$#]+)", parameters, re.IGNORECASE
    )
    if callback and callback[1].upper() != "SYS.DBMS_VECTOR_DATABASE.HANDLE_MVI_INDEX":
        raise ValueError("Cannot replace a metadata index with a different callback")
    if not callback:
        if re.search(r"\bDATAGUIDE\b", parameters, re.IGNORECASE):
            raise ValueError("Cannot replace an unrecognized data-guide configuration")
        parameters += f" DATAGUIDE {DATAGUIDE_CALLBACK}"
    parameters = re.sub(
        r"\bSEARCH_ON\s+(?:TEXT_VALUE_STRING|TEXT_VALUE|TEXT|VALUE|NONE)\b",
        "",
        parameters,
        flags=re.IGNORECASE,
    ).strip()
    if not hybrid:
        parameters += " SEARCH_ON TEXT"
    return parameters


def _search_index_ddl(index: str, table: str, meta: str, parameters: str) -> str:
    escaped = parameters.replace("'", "''")
    return (
        f"CREATE SEARCH INDEX {index} ON {table} ({meta}) FOR JSON "
        f"PARAMETERS ('{escaped}')"
    )


def _metadata_index_plan(
    name: str,
    index: str,
    columns: set[str],
    existing: Optional[tuple],
    hybrid: bool,
    build: Optional[Callable[[str, str, str], str]],
) -> tuple[str, Optional[str], Optional[str]]:
    """Return physical name, replacement DDL, and original-index recovery DDL."""
    quoted, _ = table_identifier(name)
    if _layout(columns) != "vecdb":
        raise ValueError("Metadata indexing requires a recognized VecDB table")
    meta = "CONTENT_METADATA" if "CONTENT_METADATA" in columns else "METADATA"
    requested = _index_identifier(name, index)
    physical = _catalog_index_identifier(name, existing) if existing else requested
    if existing and (
        "SIMPLIFIED_JSON_HYBRID" in (existing[1] or "").upper()
        or (not hybrid and _text_enabled(existing[1]))
    ):
        return physical, None, None
    parameters = (
        _metadata_parameters(existing[1] or "", hybrid)
        if existing
        else DEFAULT_METADATA_PARAMETERS + ("" if hybrid else " SEARCH_ON TEXT")
    )
    restore = None
    if existing:
        original = re.sub(
            r"^\s*SIMPLIFIED_JSON\s*", "", existing[1], flags=re.IGNORECASE
        )
        restore = _search_index_ddl(physical, quoted, meta, original)
    if hybrid:
        if build is None:
            raise ValueError("Hybrid indexing requires a vectorizer preference")
        ddl = build(physical, parameters, meta)
    else:
        ddl = _search_index_ddl(physical, quoted, meta, parameters)
    return physical, ddl, restore


def _layout(columns: set[str]) -> str:
    """Validate column compatibility, not VecDB package ownership."""
    if not columns:
        return "missing"
    if {"ID", "EMBEDDING", "METADATA", "TEXT"} <= columns:
        return "sql"
    if not BASE_COLUMNS <= columns or {"METADATA", "CONTENT_METADATA"} <= columns:
        raise ValueError("Incompatible VecDB table layout")
    if "METADATA" not in columns and "CONTENT_METADATA" not in columns:
        raise ValueError("VecDB table has no recognized metadata column")
    return "vecdb"


VECDB_RECOGNITION_SQL = (
    "SELECT COUNT(*) FROM all_procedures WHERE owner = 'SYS' "
    "AND object_name = 'DBMS_VECTOR_DATABASE' "
    "AND procedure_name = 'DESCRIBE_VECTOR_TABLE'"
)
VECDB_TABLE_SQL = (
    "SELECT SYS.DBMS_VECTOR_DATABASE.DESCRIBE_VECTOR_TABLE(:name), "
    "SYS_CONTEXT('USERENV', 'CURRENT_SCHEMA') FROM SYS.DUAL"
)

VECDB_QUALIFIED_TABLE_SQL = (
    "SELECT SYS.DBMS_VECTOR_DATABASE.DESCRIBE_VECTOR_TABLE("
    "CASE WHEN :owner = SYS_CONTEXT('USERENV', 'CURRENT_SCHEMA') "
    "THEN :unqualified_name ELSE :name END), "
    "SYS_CONTEXT('USERENV', 'CURRENT_SCHEMA') FROM SYS.DUAL"
)
# Older packages reject qualified/case-sensitive table descriptions. Keep this
# compatibility lookup server-side: one registry generation and one scalar bind.
VECDB_QUALIFIED_MEMBERSHIP_SQL = """
DECLARE
    payload CLOB;
BEGIN
    payload := SYS.DBMS_VECTOR_DATABASE.LIST_VECTOR_TABLES();
    SELECT CASE
        WHEN payload IS NULL THEN NULL
        WHEN JSON_VALUE(payload, '$.vector_tables.type()' ERROR ON ERROR) <> 'array'
            THEN NULL
        WHEN JSON_EXISTS(payload,
            '$.vector_tables[*]?(!exists(@.owner) || !exists(@.table_name) ||
                @.owner.type() != "string" || @.table_name.type() != "string")'
            ERROR ON ERROR) THEN NULL
        WHEN JSON_EXISTS(payload,
            '$.vector_tables[*]?(@.owner == $owner && @.table_name == $name)'
            PASSING COALESCE(:owner, SYS_CONTEXT('USERENV', 'CURRENT_SCHEMA'))
                AS "owner", :name AS "name" ERROR ON ERROR)
            THEN 1
        ELSE 0
    END INTO :membership FROM SYS.DUAL;
END;
"""


def _description_statement(name: str) -> tuple[str, dict]:
    quoted, _ = table_identifier(name)
    parts = _identifier_parts(name)
    if len(parts) == 1:
        return VECDB_TABLE_SQL, {"name": quoted}
    return VECDB_QUALIFIED_TABLE_SQL, {
        "name": quoted,
        "owner": parts[0],
        "unqualified_name": '"' + parts[-1].replace('"', '""') + '"',
    }


def _qualified_membership_binds(name: str, error: oracledb.DatabaseError) -> dict:
    parts = _identifier_parts(name)
    if getattr(error.args[0], "code", None) != 44003 or len(parts) != 2:
        raise error
    return {"owner": parts[0], "name": parts[-1]}


def _case_sensitive_membership_binds(
    name: str, error: oracledb.DatabaseError
) -> Optional[dict]:
    parts = _identifier_parts(name)
    if getattr(error.args[0], "code", None) != 942 or all(
        part == part.upper() for part in parts
    ):
        return None
    return {"owner": parts[0] if len(parts) == 2 else None, "name": parts[-1]}


def _membership_flag(value: Any) -> bool:
    if value not in (0, 1):
        raise ValueError("Malformed VecDB table registry identity")
    return bool(value)


def _described_table(name: str, result: tuple) -> bool:
    payload, schema = result
    # The package returns SQL NULL for an existing, unregistered SQL table.
    if payload is None:
        return False
    response = as_dict(payload)
    if not all(isinstance(response.get(key), str) for key in ("owner", "table_name")):
        raise ValueError("Malformed VecDB table identity")
    parts = _identifier_parts(name)
    owner = parts[0] if len(parts) == 2 else schema
    return response["owner"] == owner and response["table_name"] == parts[-1]


def _is_sql_synonym(cursor: Any, name: str) -> bool:
    """Prove SQL synonym resolution before treating package ORA-00942 as absent.

    DESCRIBE_VECTOR_TABLE does not resolve synonyms. A real table, including one
    shadowing a public synonym, must still propagate its description errors.
    """
    statement, binds = _layout_query(name)
    cursor.execute(statement, **binds)
    if cursor.fetchall():
        return False
    statement, binds = _synonym_query(name)
    cursor.execute(statement, **binds)
    if not cursor.fetchone():
        return False
    cursor.execute(f"SELECT * FROM {_quote_indentifier(name)} WHERE 1 = 0")
    columns = {column[0] for column in cursor.description or ()}
    return {"ID", "TEXT", "METADATA", "EMBEDDING"} <= columns


async def _ais_sql_synonym(cursor: Any, name: str) -> bool:
    statement, binds = _layout_query(name)
    await cursor.execute(statement, **binds)
    if await cursor.fetchall():
        return False
    statement, binds = _synonym_query(name)
    await cursor.execute(statement, **binds)
    if not await cursor.fetchone():
        return False
    await cursor.execute(f"SELECT * FROM {_quote_indentifier(name)} WHERE 1 = 0")
    columns = {column[0] for column in cursor.description or ()}
    return {"ID", "TEXT", "METADATA", "EMBEDDING"} <= columns


def _is_vecdb_table(cursor: Any, name: str) -> bool:
    """Describe one exact table without transferring the package registry."""
    cursor.execute(VECDB_RECOGNITION_SQL)
    if not cursor.fetchone()[0]:
        return False
    statement, binds = _description_statement(name)
    try:
        cursor.execute(statement, **binds)
    except oracledb.DatabaseError as error:
        if getattr(error.args[0], "code", None) == 942 and _is_sql_synonym(
            cursor, name
        ):
            return False
        membership_binds = _case_sensitive_membership_binds(name, error)
        if membership_binds is not None:
            # Some older packages fold quoted names. Prove this exact table
            # exists, then check registration without changing its case.
            statement, layout_binds = _layout_query(name)
            cursor.execute(statement, **layout_binds)
            if not cursor.fetchall():
                raise error
        else:
            membership_binds = _qualified_membership_binds(name, error)
        membership = cursor.var(int)
        cursor.execute(
            VECDB_QUALIFIED_MEMBERSHIP_SQL, membership=membership, **membership_binds
        )
        return _membership_flag(membership.getvalue())
    result = cursor.fetchone()
    if result is None:
        raise ValueError("Missing VecDB recognition response")
    return _described_table(name, result)


async def _ais_vecdb_table(cursor: Any, name: str) -> bool:
    await cursor.execute(VECDB_RECOGNITION_SQL)
    if not (await cursor.fetchone())[0]:
        return False
    statement, binds = _description_statement(name)
    try:
        await cursor.execute(statement, **binds)
    except oracledb.DatabaseError as error:
        if getattr(error.args[0], "code", None) == 942 and await _ais_sql_synonym(
            cursor, name
        ):
            return False
        membership_binds = _case_sensitive_membership_binds(name, error)
        if membership_binds is not None:
            statement, layout_binds = _layout_query(name)
            await cursor.execute(statement, **layout_binds)
            if not await cursor.fetchall():
                raise error
        else:
            membership_binds = _qualified_membership_binds(name, error)
        membership = cursor.var(int)
        await cursor.execute(
            VECDB_QUALIFIED_MEMBERSHIP_SQL, membership=membership, **membership_binds
        )
        return _membership_flag(membership.getvalue())
    result = await cursor.fetchone()
    if result is None:
        raise ValueError("Missing VecDB recognition response")
    payload, schema = result
    if hasattr(payload, "read"):
        payload = await payload.read()
    return _described_table(name, (payload, schema))


def _table_kind(columns: set[str], registered: bool) -> str:
    if not columns:
        return "missing"
    if registered:
        if _layout(columns) != "vecdb":
            raise ValueError("Incompatible VecDB table layout")
        return "vecdb"
    if {"ID", "EMBEDDING", "METADATA", "TEXT"} <= columns:
        return "sql"
    raise ValueError(
        "Incompatible OracleVS table layout; table is not registered in VecDB"
    )


def _migration_actions(columns: set[str], quoted: str) -> list[str]:
    if _layout(columns) != "vecdb":
        raise ValueError("Only recognized VecDB layouts can be migrated")
    actions = []
    if "METADATA" in columns:
        actions.append(
            f"ALTER TABLE {quoted} RENAME COLUMN METADATA TO CONTENT_METADATA"
        )
    for name, sql_type in (
        ("SPARSE_VECTOR", "VECTOR(*, *, SPARSE)"),
        ("CONTENT", "BLOB"),
        ("CONTENT_TYPE", "VARCHAR2(256)"),
    ):
        if name not in columns:
            actions.append(f"ALTER TABLE {quoted} ADD {name} {sql_type}")
    return actions


def layout_error(error: Exception) -> bool:
    """Only a missing current-layout column is a candidate for repair.

    Physical inspection and database generation are checked before any DDL.
    ORA-00904 alone (e.g. a bad user identifier) never triggers migration.
    """
    message = str(error).upper().replace('\\"', '"')
    return bool(
        re.search(
            r'ORA-00904:\s*(?:(?:"[^"]+"|\w+)\.)*"?'
            r'(?:CONTENT_METADATA|SPARSE_VECTOR|CONTENT_TYPE|CONTENT)"?'
            r"(?=\s*(?::|$))",
            message,
        )
    )


def _hybrid_search_mode(params: dict) -> str:
    """Match the retriever mode from its vector/text request sections."""
    if "vector" in params and "text" not in params:
        return "semantic"
    if "text" in params and "vector" not in params:
        return "keyword"
    return "hybrid"


def _hybrid_sql(name: str, columns: set[str]) -> str:
    quoted, _ = table_identifier(name)
    _layout(columns)
    meta = "CONTENT_METADATA" if "CONTENT_METADATA" in columns else "METADATA"
    return (
        f"SELECT t.ID, t.{meta}, h.score, h.text_score, h.vector_score "
        "FROM JSON_TABLE(DBMS_HYBRID_VECTOR.SEARCH(JSON(:search_params)), "
        "'$[*]' COLUMNS (position FOR ORDINALITY, "
        "rid VARCHAR2(128) PATH '$.rowid', score NUMBER PATH '$.score', "
        "text_score NUMBER PATH '$.text_score', "
        "vector_score NUMBER PATH '$.vector_score')) h "
        f"JOIN {quoted} t ON t.ROWID = CHARTOROWID(h.rid) ORDER BY h.position"
    )


async def _complete_cleanup(awaitable: Any) -> Any:
    """Finish session-owned cleanup before propagating caller cancellation."""
    task = asyncio.ensure_future(awaitable)
    cancelled = None
    while True:
        try:
            result = await asyncio.shield(task)
            break
        except asyncio.CancelledError as error:
            if task.cancelled():
                raise
            cancelled = error
    if cancelled is not None:
        raise cancelled
    return result


_NativeMethod = TypeVar("_NativeMethod", bound=Callable[..., Any])


@dataclass(frozen=True)
class _TableInfo:
    columns: frozenset[str]
    hybrid_index: Optional[str]

    @property
    def metadata_column(self) -> str:
        return "CONTENT_METADATA" if "CONTENT_METADATA" in self.columns else "METADATA"


def _table_info(columns: set[str], existing: Optional[tuple]) -> _TableInfo:
    if _layout(columns) != "vecdb":
        raise ValueError("Metadata searches require a recognized VecDB table")
    index = None
    if existing and "SIMPLIFIED_JSON_HYBRID" in (existing[1] or "").upper():
        index = _quote_indentifier(f'"{existing[0]}"')
    return _TableInfo(frozenset(columns), index)


def _metadata_search_error(error: Exception) -> bool:
    """Refresh only missing layout columns or a missing cached index."""
    return layout_error(error) or bool(
        re.search(
            r'ORA-00904:\s*(?:(?:"[^"]+"|\w+)\.)*"?METADATA"?(?=\s*(?::|$))',
            str(error),
            re.IGNORECASE,
        )
        or re.search(r"\b(?:ORA-01418|DRG-10502):", str(error))
    )


def _is_native_pool(client: Any, *, asynchronous: bool = False) -> bool:
    pool_type = getattr(
        oracledb, "AsyncConnectionPool" if asynchronous else "ConnectionPool", None
    )
    return pool_type is not None and isinstance(client, pool_type)


def _with_native_session(method: _NativeMethod) -> _NativeMethod:
    """Bind a pooled native operation without changing shared transport state."""

    @wraps(method)
    def wrapped(self: NativeTransport, *args: Any, **kwargs: Any) -> Any:
        if not _is_native_pool(self.client):
            return method(self, *args, **kwargs)
        with self.session() as bound:
            return method(bound, *args, **kwargs)

    return cast(_NativeMethod, wrapped)


def _with_async_native_session(method: _NativeMethod) -> _NativeMethod:
    @wraps(method)
    async def wrapped(self: AsyncNativeTransport, *args: Any, **kwargs: Any) -> Any:
        if not _is_native_pool(self.client, asynchronous=True):
            return await method(self, *args, **kwargs)
        async with self.asession() as bound:
            return await method(bound, *args, **kwargs)

    return cast(_NativeMethod, wrapped)


def _with_native_backend_session(method: _NativeMethod) -> _NativeMethod:
    """Keep composed operations (prepare, batches and migration retries) together."""

    @wraps(method)
    def wrapped(self: VecDBBackend, *args: Any, **kwargs: Any) -> Any:
        transport = self.transport
        if not isinstance(transport, NativeTransport) or not _is_native_pool(
            transport.client
        ):
            return method(self, *args, **kwargs)
        with transport.session() as bound:
            backend = copy(self)
            backend.transport = bound
            return method(backend, *args, **kwargs)

    return cast(_NativeMethod, wrapped)


def _with_async_native_backend_session(method: _NativeMethod) -> _NativeMethod:
    @wraps(method)
    async def wrapped(self: VecDBBackend, *args: Any, **kwargs: Any) -> Any:
        transport = self.transport
        if not isinstance(transport, AsyncNativeTransport) or not _is_native_pool(
            transport.client, asynchronous=True
        ):
            return await method(self, *args, **kwargs)
        async with transport.asession() as bound:
            backend = copy(self)
            backend.transport = bound
            return await method(backend, *args, **kwargs)

    return cast(_NativeMethod, wrapped)


class NativeTransport:
    """Execute a bounded set of package calls using caller-owned connections."""

    def __init__(self, client: Any, version: tuple[int, ...]):
        self.client = client
        self.version = version
        # Bound pool transports share snapshots, never mutable session state.
        self._table_info_cache: dict[tuple, _TableInfo] = {}

    def _table_key(self, name: str) -> tuple:
        return tuple(_identifier_parts(name)), self.version

    def _remember_table(self, name: str, info: _TableInfo) -> None:
        if info.hybrid_index:
            # Unqualified table/index names resolve in the executing session,
            # just as they do in the regular LangChain SQL path.
            info = _TableInfo(
                info.columns,
                _quote_indentifier(f'"{_identifier_parts(info.hybrid_index)[-1]}"'),
            )
        self._table_info_cache[self._table_key(name)] = info

    def _forget_table(self, name: str) -> None:
        self._table_info_cache.pop(self._table_key(name), None)

    @_with_native_session
    def discover(self, name: str, columns: Optional[set[str]] = None) -> _TableInfo:
        with _get_connection(self.client) as connection:
            with connection.cursor() as cursor:
                if columns is None:
                    columns = self.columns(name)
                meta = (
                    "CONTENT_METADATA" if "CONTENT_METADATA" in columns else "METADATA"
                )
                statement, binds = _metadata_index_query(name, meta)
                cursor.execute(statement, **binds)
                info = _table_info(columns, cursor.fetchone())
                self._remember_table(name, info)
                return self._table_info_cache[self._table_key(name)]

    def _search_info(self, name: str) -> _TableInfo:
        info = self._table_info_cache.get(self._table_key(name))
        return info if info is not None else self.discover(name)

    @contextmanager
    def session(self) -> Iterator[NativeTransport]:
        with _get_connection(self.client) as connection:
            bound = copy(self)
            bound.client = connection
            # A pool checkout, not the store's initial checkout, owns this snapshot.
            bound.version = parse_version(connection.version)
            yield bound

    def columns(self, name: str) -> set[str]:
        with _get_connection(self.client) as connection:
            with connection.cursor() as cursor:
                return _inspect_columns(cursor, name)

    def _statement(self, operation: str, params: dict) -> tuple[str, dict]:
        # Quoted package arguments preserve case and literal dots, just as SQL
        # identifiers do. Schema-qualified package operations remain unsupported.
        params = dict(params)
        for key in ("name", "table_name"):
            if key in params:
                name_error = _package_name_error(params[key])
                if name_error:
                    raise ValueError(name_error)
                params[key] = _quote_indentifier(params[key])
        statement, binds = SQLNetAdapter(self.version).statement(operation, params)
        if operation in {"search", "query"}:
            statement = tag_sql(statement, "vector", backend="vecdb")
        return statement, binds

    @_with_native_session
    def call(self, operation: str, **params: Any) -> dict:
        statement, binds = self._statement(operation, params)
        with _get_connection(self.client) as connection:
            with connection.cursor() as cursor:
                if (
                    operation == "drop_vector_table"
                    and connection.transaction_in_progress is True
                ):
                    raise RuntimeError("VecDB table drop requires no pending work")
                if (
                    operation in {"create_index", "rebuild_index"}
                    and connection.transaction_in_progress is True
                ):
                    raise RuntimeError(
                        "Index submission requires a connection without pending work"
                    )
                if operation in {"create_vector_table", "drop_vector_table"}:
                    self._forget_table(params["name"])
                out = cursor.var(oracledb.DB_TYPE_CLOB)
                cursor.setinputsizes(
                    **{
                        key: oracledb.DB_TYPE_CLOB
                        for key in binds
                        if f"JSON(:{key})" in statement
                    }
                )
                if operation not in {"search", "query", "list_vectors"}:
                    try:
                        cursor.execute(
                            function_preflight(statement), result=out, **binds
                        )
                    except Exception as error:
                        signature_error(operation, connection.version, error)
                cursor.execute(statement, result=out, **binds)
                return as_dict(out.getvalue())

    @_with_native_session
    def preflight_index(self, hybrid: bool = False) -> None:
        with _get_connection(self.client) as connection:
            with connection.cursor() as cursor:
                if connection.transaction_in_progress:
                    raise RuntimeError(
                        "Index DDL requires a connection without pending work"
                    )
                cursor.execute(INDEX_FEATURES_SQL)
                require_index_features(cursor.fetchall(), hybrid)

    @_with_native_session
    def describe_index(self, name: str, kind: str = "vector") -> dict:
        owner, binds = _catalog_target(name)
        statement = index_status_query(owner, kind)
        with _get_connection(self.client) as connection:
            with connection.cursor() as cursor:
                cursor.execute(statement, **binds)
                return native_index_status(cursor.fetchall())

    @_with_native_session
    def migrate(self, name: str) -> None:
        quoted, _ = table_identifier(name)
        with _get_connection(self.client) as connection:
            self.version = parse_version(connection.version)
            if self.version < CURRENT_VERSION:
                raise ValueError("Layout migration requires Oracle Database 23.26.3+")
            if connection.transaction_in_progress:
                raise RuntimeError(
                    "VecDB migration requires a connection without pending work"
                )
            with connection.cursor() as cursor:
                db_version = cursor.var(str, size=128)
                compatibility = cursor.var(str, size=128)
                cursor.execute(
                    COMPATIBILITY_SQL,
                    db_version=db_version,
                    compatibility=compatibility,
                )
                _compatible_backend(
                    self.version, compatibility.getvalue(), "vecdb", "vecdb"
                )

                def inspect() -> set[str]:
                    return _inspect_columns(cursor, name)

                self._forget_table(name)
                for action in _migration_actions(inspect(), quoted):
                    try:
                        cursor.execute(action)
                    except oracledb.DatabaseError as exc:
                        # Another process may have completed this exact DDL.
                        code = exc.args[0].code
                        if code not in {904, 1430} or action in _migration_actions(
                            inspect(), quoted
                        ):
                            raise
                if not CURRENT_COLUMNS <= inspect():
                    raise RuntimeError(
                        "VecDB migration did not produce the required layout"
                    )

    @_with_native_session
    def create_metadata_index(
        self,
        name: str,
        index: str,
        *,
        hybrid: bool = False,
        build: Optional[Callable[[str, str, str], str]] = None,
    ) -> None:
        columns = self.columns(name)
        meta = "CONTENT_METADATA" if "CONTENT_METADATA" in columns else "METADATA"
        with _get_connection(self.client) as connection:
            with connection.cursor() as cursor:
                statement, binds = _metadata_index_query(name, meta)
                cursor.execute(statement, **binds)
                found = cursor.fetchall()
                if len(found) > 1:
                    raise ValueError("Multiple metadata domain indexes found")
                existing = found[0] if found else None
                physical, ddl, restore = _metadata_index_plan(
                    name, index, columns, existing, hybrid, build
                )
                if ddl is None:
                    self._remember_table(name, _table_info(columns, existing))
                    return
                if connection.transaction_in_progress:
                    raise RuntimeError(
                        "Index DDL requires a connection without pending work"
                    )
                if not existing:
                    statement, binds = _index_query(physical)
                    cursor.execute(statement, **binds)
                    if cursor.fetchone():
                        raise ValueError(
                            "Index name is already used by an incompatible index"
                        )
                self.preflight_index(hybrid)
                self._forget_table(name)
                build_started = False
                try:
                    if restore:
                        cursor.execute(f"DROP INDEX {physical}")
                    build_started = True
                    cursor.execute(ddl)
                except BaseException:
                    if restore:
                        try:
                            statement, binds = _index_query(physical)
                            cursor.execute(statement, **binds)
                            present = cursor.fetchone()
                            if present and build_started:
                                cursor.execute(f"DROP INDEX {physical} FORCE")
                            if not present or build_started:
                                cursor.execute(restore)
                            logger.warning(
                                "Retained or restored metadata index %s "
                                "after replacement failed",
                                physical,
                            )
                        except BaseException as recovery:
                            raise RuntimeError(
                                f"Could not restore metadata index {physical} after "
                                "replacement failed; manual recovery is required"
                            ) from recovery
                    raise
                self._remember_table(
                    name,
                    _TableInfo(frozenset(columns), physical if hybrid else None),
                )
                logger.info(
                    "Created %s metadata index %s",
                    "hybrid" if hybrid else "text",
                    physical,
                )

    def create_text_index(self, name: str, index: str) -> None:
        self.create_metadata_index(name, index)

    @_with_native_session
    def insert(self, name: str, records: list[dict]) -> list[str]:
        import array

        quoted, _ = table_identifier(name)
        columns = self.columns(name)
        _layout(columns)
        meta = "CONTENT_METADATA" if "CONTENT_METADATA" in columns else "METADATA"
        values = [
            (r["id"], array.array("f", r["dense_vector"]), r["metadata"])
            for r in records
        ]
        with _get_connection(self.client) as connection:
            # Match regular OracleVS SQL inserts: report row errors and commit
            # successful rows, including any caller work on this connection.
            with connection.cursor() as cursor:
                cursor.setinputsizes(
                    None, oracledb.DB_TYPE_VECTOR, oracledb.DB_TYPE_JSON
                )
                cursor.executemany(
                    f"INSERT INTO {quoted} (ID, DENSE_VECTOR, {meta}) "
                    "VALUES (:1, :2, :3)",
                    values,
                    batcherrors=True,
                )
                errors = cursor.getbatcherrors() or []
                for error in errors:
                    logger.warning(
                        "Could not insert row at offset %s due to error: %s",
                        error.offset,
                        error.message,
                    )
                connection.commit()
                failed = {error.offset for error in errors}
                return [
                    record["id"] for i, record in enumerate(records) if i not in failed
                ]

    @_with_native_session
    def hybrid_search(self, name: str, params: dict, scores: bool) -> list:
        validate_hybrid_request(params)
        with _get_connection(self.client) as connection:
            info = self._search_info(name)
            # An index may have been created externally since store initialization.
            if info.hybrid_index is None:
                info = self.discover(name)
            for attempt in range(2):
                if info.hybrid_index is None:
                    raise ValueError(
                        "Create a hybrid vector index on the VecDB "
                        "metadata column first"
                    )
                requested = params.get("hybrid_index_name")
                index = (
                    requested
                    if requested and len(_identifier_parts(requested)) == 2
                    else info.hybrid_index
                )
                request = hybrid_search_request(
                    name, {**params, "hybrid_index_name": index}
                )
                try:
                    with connection.cursor() as cursor:
                        cursor.setinputsizes(search_params=oracledb.DB_TYPE_CLOB)
                        cursor.execute(
                            tag_sql(
                                _hybrid_sql(name, set(info.columns)),
                                _hybrid_search_mode(params),
                                backend="vecdb",
                            ),
                            search_params=json.dumps(request, allow_nan=False),
                        )
                        return _hybrid_documents(
                            cursor.fetchall(),
                            scores,
                            keyword_only=_hybrid_search_mode(params) == "keyword",
                        )
                except Exception as error:
                    if attempt or not _metadata_search_error(error):
                        raise
                    self._forget_table(name)
                    info = self.discover(name)
            raise AssertionError("Unreachable metadata search retry")

    @_with_native_session
    def filter_search(self, name: str, filter: dict, request: dict) -> dict:
        with _get_connection(self.client) as connection:
            info = self._search_info(name)
            for attempt in range(2):
                sql, binds = _filter_query(name, info.metadata_column, request, filter)
                try:
                    with connection.cursor() as cursor:
                        cursor.execute(sql, **binds)
                        return _filter_response(cursor.fetchall(), request)
                except Exception as error:
                    if attempt or not _metadata_search_error(error):
                        raise
                    self._forget_table(name)
                    info = self.discover(name)
            raise AssertionError("Unreachable filter search retry")

    @_with_native_session
    def text_search(self, name: str, query: str, k: int, scores: bool = False) -> list:
        quoted, _ = table_identifier(name)
        with _get_connection(self.client) as connection:
            info = self._search_info(name)
            for attempt in range(2):
                meta = info.metadata_column
                sql = (
                    f"SELECT id, {meta}, SCORE(1) FROM {quoted} "
                    f"WHERE JSON_TEXTCONTAINS({meta}, '$.{TEXT_KEY}', :query, 1) "
                    "ORDER BY SCORE(1) DESC, id FETCH FIRST :k ROWS ONLY"
                )
                try:
                    with connection.cursor() as cursor:
                        cursor.execute(
                            tag_sql(sql, "text", backend="vecdb"), query=query, k=k
                        )
                        return _text_documents(cursor.fetchall(), scores)
                except Exception as error:
                    if attempt or not _metadata_search_error(error):
                        raise
                    self._forget_table(name)
                    info = self.discover(name)
            raise AssertionError("Unreachable metadata search retry")


class RestTransport:
    """Only public SDK methods; no direct HTTP or private SDK attributes."""

    def __init__(self, client: Any):
        config = client.config
        # Public configuration flags describe transport; SDK class identity does not.
        if getattr(config, "has_sqlnet_connection", False) is True:
            raise ValueError(
                "OracleVS does not support SQLNet-configured oracle-vecdb SDK "
                "clients. Pass an oracledb connection or pool for native VecDB."
            )
        if not (
            getattr(config, "has_rest_connection", False) is True
            or getattr(config, "has_rest_url", False) is True
        ):
            raise ValueError(
                "OracleVS requires an ORDS-configured oracle-vecdb SDK client"
            )
        self.client = client

    def call(self, operation: str, **params: Any) -> dict:
        method = "query" if operation == "search" else operation
        return as_dict(getattr(self.client, method)(**params))

    def migrate(self, name: str) -> None:
        raise RuntimeError("oracle-vecdb REST SDK does not support schema migration")

    def text_search(self, name: str, query: str, k: int, scores: bool = False) -> list:
        raise ValueError("oracle-vecdb REST SDK does not support metadata text search")


def is_sdk_client(client: Any) -> bool:
    # Avoid importing the optional SDK for a native/SQL store.
    return any(
        cls.__module__.startswith("oracle_vecdb.") and cls.__name__ == "OracleVecDB"
        for cls in type(client).__mro__
    )


def _compatible_backend(
    version: tuple[int, ...], compatibility: str, kind: str, backend: str
) -> str:
    minimum = SPARSE_COMPATIBLE if version >= CURRENT_VERSION else DENSE_COMPATIBLE
    if parse_version(compatibility) < minimum:
        required = ".".join(map(str, minimum))
        if backend == "vecdb" or kind == "vecdb":
            message = (
                f"VecDB requires COMPATIBLE >= {required}; "
                f"configured value is {compatibility}"
            )
            if kind == "vecdb":
                message += ". An existing VecDB table cannot use the SQL backend."
            raise ValueError(message)
        logger.info(
            "Using SQL backend: COMPATIBLE %s is below VecDB requirement %s",
            compatibility,
            required,
        )
        return "sql"
    return "vecdb"


def native_selection(
    connection: Any, name: str, backend: str
) -> tuple[str, tuple[int, ...]]:
    version = parse_version(connection.version)
    with connection.cursor() as cursor:
        columns = _inspect_columns(cursor, name)
        kind = _table_kind(columns, _is_vecdb_table(cursor, name) if columns else False)
        if backend == "sql":
            if kind == "vecdb":
                raise ValueError(
                    "The SQL backend cannot use a VecDB table layout; "
                    "use backend='auto' or backend='vecdb'"
                )
            return "sql", version
        if kind == "sql":
            if backend == "vecdb":
                raise ValueError(
                    "The VecDB backend cannot use an ordinary OracleVS table layout; "
                    "use backend='auto' or backend='sql'"
                )
            return "sql", version
        if kind == "missing":
            name_error = _package_name_error(name)
            if name_error:
                if backend == "vecdb":
                    raise ValueError(name_error)
                return "sql", version
        if version < LEGACY_VERSION:
            if backend == "vecdb" or kind == "vecdb":
                raise ValueError("VecDB requires Oracle Database 23.26.1+")
            return "sql", version
        cursor.execute(VECDB_CAPABILITY_SQL)
        supported, compatibility = cursor.fetchone()
        if not supported:
            if backend == "vecdb" or kind == "vecdb":
                raise ValueError("DBMS_VECTOR_DATABASE.SEARCH is unavailable")
            return "sql", version
        if (
            kind == "vecdb"
            and version < CURRENT_VERSION
            and "CONTENT_METADATA" in columns
        ):
            raise ValueError(
                "Current VecDB table layout cannot be used on a legacy database"
            )
        return _compatible_backend(version, compatibility, kind, backend), version


class VecDBBackend:
    def __init__(
        self,
        transport: Any,
        name: str,
        metric: DistanceStrategy,
        *,
        table_type: TableType = "vector_store",
        vector_index_params: Optional[dict] = None,
    ):
        self.transport = transport
        self.table_type = table_type
        self.quoted, api_name = table_identifier(name)
        # Native calls must retain component boundaries, including literal dots.
        self.name = self.quoted if isinstance(transport, NativeTransport) else api_name
        from .oraclevs import _get_distance_function

        self.metric = _get_distance_function(
            DistanceStrategy.DOT_PRODUCT
            if metric == DistanceStrategy.MAX_INNER_PRODUCT
            else metric
        )
        self._creation_index_params = self.index_params(vector_index_params)
        validate_operation_arguments(
            "create_vector_table", {"index_params": self._creation_index_params}
        )

    def _table_creation_params(self) -> dict:
        params = {
            "name": self.name,
            "table_params": {"auto_generate_id": False},
            "annotations": table_annotations(self.table_type, "vecdb"),
        }
        if self._creation_index_params is not None:
            params["index_params"] = self._creation_index_params
        return params

    @_with_native_backend_session
    def prepare(self) -> None:
        if isinstance(self.transport, NativeTransport):
            columns = self.transport.columns(self.name)
            created = not columns
            if created:
                self.transport.call(
                    "create_vector_table",
                    **self._table_creation_params(),
                )
                columns = self.transport.columns(self.name)
            self.transport.discover(self.name, columns)
            logger.info(
                "Table %s %s...",
                self.quoted,
                "created successfully" if created else "already exists",
            )
            return
        try:
            self.transport.call("describe_vector_table", name=self.name)
        except Exception as exc:
            if getattr(exc, "status", None) != 404:
                raise
            self.transport.call(
                "create_vector_table",
                **self._table_creation_params(),
            )
            logger.info("Table %s created successfully...", self.quoted)
        else:
            logger.info("Table %s already exists...", self.quoted)

    @_with_native_backend_session
    def query(self, operation: str, **params: Any) -> dict:
        try:
            return self.transport.call(operation, **params)
        except Exception as exc:
            if not layout_error(exc):
                raise
            self.transport.migrate(self.name)
            return self.transport.call(operation, **params)

    def records(
        self,
        texts: list[str],
        vectors: list[list[float]],
        metadatas: list[dict],
        ids: list[str],
    ) -> list[dict]:
        if not texts:
            return []
        if len(texts) != len(vectors):
            raise ValueError("Text and embedding batch sizes must match")
        records = []
        for text, vector, metadata, id_ in zip(texts, vectors, metadatas, ids):
            if LOGICAL_ID_KEY in metadata:
                raise ValueError(
                    f"Document metadata key {LOGICAL_ID_KEY!r} is reserved"
                )
            if TEXT_KEY in metadata:
                raise ValueError(
                    f"Document metadata key {TEXT_KEY!r} is reserved for content"
                )
            records.append(
                {
                    "id": storage_id(id_),
                    "dense_vector": vector,
                    "metadata": {**metadata, TEXT_KEY: text, LOGICAL_ID_KEY: id_},
                }
            )
        return records

    def add(
        self,
        texts: list[str],
        vectors: list[list[float]],
        metadatas: list[dict],
        ids: list[str],
        mutate: bool,
    ) -> list[str]:
        records = self.records(texts, vectors, metadatas, ids)
        if not records:
            return []
        if not mutate:
            if isinstance(self.transport, RestTransport):
                raise ValueError(
                    "oracle-vecdb REST SDK does not support insert-only writes; "
                    "set mutate_on_duplicate=True to enable upserts "
                    "(existing IDs will be overwritten)"
                )
            inserted = self.transport.insert(self.name, records)
            logical_ids = {record["id"]: id_ for record, id_ in zip(records, ids)}
            return [logical_ids[id_] for id_ in inserted]
        self.transport.call("upsert_vectors", table_name=self.name, vectors=records)
        return ids

    def search(
        self,
        embedding: list[float],
        k: int,
        filter: Optional[dict],
        include_vectors: bool = False,
        **kwargs: Any,
    ) -> list:
        request, fallback = self._search_request(
            embedding, k, filter, include_vectors, kwargs
        )
        response = (
            self.transport.filter_search(self.name, filter, request)
            if fallback
            else self.query("search", **request)
        )
        return self.search_results(response, include_vectors)

    def _search_request(
        self, embedding: list, k: int, filter: Any, include: bool, options: dict
    ) -> tuple:
        try:
            return search_request(
                self.name, self.metric, embedding, k, filter, include, options
            ), False
        except SQLFilterRequired:
            if not isinstance(self.transport, NativeTransport):
                raise
            # Revalidate all search options even when filter translation cannot
            # use the package. Only the typed translation signal selects SQL.
            request = search_request(
                self.name, self.metric, embedding, k, None, include, options
            )
            logger.debug(
                "Using native SQL filter fallback for VecDB table %s", self.name
            )
            return request, True

    def search_results(self, response: dict, include_vectors: bool) -> list:
        result = []
        for row in rows(response, "results"):
            distance = row.get("distance")
            if not isinstance(distance, (int, float, Decimal)) or not math.isfinite(
                distance
            ):
                raise ValueError("VecDB returned an invalid distance")
            hit: tuple[Any, ...] = (document(row), float(distance))
            if include_vectors:
                if not isinstance(row.get("vector"), list):
                    raise ValueError("VecDB did not return requested embeddings")
                hit += (np.asarray(row["vector"], dtype=np.float32),)
            result.append(hit)
        return result

    @_with_native_backend_session
    def get(self, ids: list[str]) -> list[Document]:
        if not ids:
            return []
        result: list[Document] = []
        for offset in range(0, len(ids), 100):
            batch = [storage_id(id_) for id_ in ids[offset : offset + 100]]
            response = self.query(
                "list_vectors",
                table_name=self.name,
                ids=batch,
                limit=len(batch),
                offset=0,
            )
            result.extend(document(row) for row in rows(response, "vectors"))
        return result

    def delete(self, ids: Optional[list[str]]) -> None:
        self.transport.call(
            "delete_vectors",
            table_name=self.name,
            ids=None if ids is None else [storage_id(id_) for id_ in ids],
        )

    def text_search(self, query: str, k: int, scores: bool = False) -> list[Document]:
        return self.transport.text_search(self.name, query, k, scores)

    def native_transport(
        self, operation: str = "SQL-based metadata index operations"
    ) -> NativeTransport:
        if isinstance(self.transport, RestTransport):
            raise ValueError(f"oracle-vecdb REST SDK does not support {operation}")
        return self.transport

    def hybrid_search(self, params: dict, scores: bool) -> list[Document]:
        return self.native_transport("hybrid search").hybrid_search(
            self.name, params, scores
        )

    def create_text_index(self, index: str) -> None:
        self.native_transport("text index creation").create_text_index(self.name, index)

    def describe_index(
        self, kind: str = "vector", job_name: Optional[str] = None
    ) -> dict:
        if kind not in {"vector", "metadata"}:
            raise ValueError("Index kind must be 'vector' or 'metadata'")
        if isinstance(self.transport, RestTransport):
            if kind != "vector":
                raise ValueError("ORDS SDK metadata index status is unsupported")
            response = self.transport.call(
                "describe_index_job" if job_name is not None else "describe_index",
                **(
                    {"index_job_name": job_name}
                    if job_name is not None
                    else {"table_name": self.name}
                ),
            )
            return sdk_index_status(response, job_name is not None)
        if job_name is not None:
            raise ValueError(
                "Native VecDB job status is unsupported; "
                "describe the physical index instead"
            )
        return self.transport.describe_index(self.name, kind)

    def index_params(self, params: Optional[dict]) -> Optional[dict]:
        if params is not None and not isinstance(params, dict):
            raise TypeError("Index parameters must be a dictionary")
        if not params:
            return None
        from .oraclevs import (
            _validate_index_type,
            _validate_int_param,
            _validate_vector_index_common,
        )

        params = dict(params or {})
        if params.get("distance_metric", self.metric) != self.metric:
            raise ValueError("Index distance_metric must match distance_strategy")
        kind = params.get("idx_type", "HNSW")
        kind = "IVF" if isinstance(kind, str) and kind.upper() == "IVF" else "HNSW"
        _validate_index_type(params, kind)
        params.setdefault("idx_name", kind)
        _validate_vector_index_common(params)
        params.pop("idx_type", None)
        params.pop("idx_name", None)  # VecDB owns the physical index name.
        accuracy = params.pop("accuracy", 90)
        parallel = params.pop("parallel", None)
        if kind == "IVF" and "partitions" in params:
            params.setdefault("neighbor_part", params.pop("partitions"))
        mapped: dict[str, Any] = {
            "organization": "INMEMORY GRAPH" if kind == "HNSW" else "PARTITIONS",
            "distance_metric": self.metric,
            "accuracy": accuracy,
        }
        # Materialize OracleVS construction defaults for explicit requests:
        # supported packages can emit invalid DDL when these are omitted.
        # A completely omitted request still leaves defaults to the server.
        advanced = (
            {"neighbors": 32, "efConstruction": 200}
            if kind == "HNSW"
            else {"partitions": 32}
        )
        ranges = (
            {"neighbors": (2, 2048), "efConstruction": (1, 65535)}
            if kind == "HNSW"
            else {"neighbor_part": (1, 10000000)}
        )
        for key, (low, high) in ranges.items():
            _validate_int_param(params, key, low, high)
            if key in params:
                advanced["partitions" if key == "neighbor_part" else key] = params.pop(
                    key
                )
        mapped.update(params)  # The SDK/database validates additional options.
        if mapped.get("distribute_params") is None:
            mapped.pop("distribute_params", None)
        if advanced:
            supplied = mapped.get("advanced_params")
            if supplied is None:
                mapped["advanced_params"] = advanced
            elif isinstance(supplied, dict):
                mapped["advanced_params"] = {**advanced, **supplied}
            # Leave malformed SDK options for its validation to reject.
        if getattr(self.transport, "version", CURRENT_VERSION) < CURRENT_VERSION:
            mapped["indexing"] = "auto"
            if parallel is not None:
                mapped["parallel_creation"] = parallel
            return mapped
        mapped["auto_index"] = True
        mapped["organization"] = "INMEMORY GRAPH" if kind == "HNSW" else "PARTITIONS"
        result: dict[str, Any] = {"vector_index_params": mapped}
        if parallel is not None:
            result["parallel_creation"] = parallel
        return result

    def _vector_index_exists(self) -> bool:
        if isinstance(self.transport, NativeTransport):
            return bool(self.describe_index()["details"]["indexes"])
        try:
            status = self.describe_index()
        except Exception as error:
            if getattr(error, "status", None) != 404:
                raise
            # A missing table must remain an error, rather than be interpreted
            # as permission to create its index.
            self.transport.call("describe_vector_table", name=self.name)
            return False
        state = sdk_index_state(status["details"])
        if state in {
            "READY",
            "VALID",
            "INDEX CREATION COMPLETED",
            "BUILDING",
            "PENDING",
            "FAILED",
            "UNUSABLE",
        }:
            return True
        raise ValueError(
            f"Cannot determine VecDB vector-index existence from SDK status {state!r}"
        )

    def _index_request(self, mapped: Optional[dict], exists: bool) -> tuple[str, dict]:
        params: dict[str, Any] = {"table_name": self.name}
        operation = "rebuild_index" if exists else "create_index"
        if mapped is not None:
            params["index_params"] = (
                {**mapped, "index_type": "vector"} if exists else mapped
            )
        return operation, params

    @_with_native_backend_session
    def create_index(self, params: Optional[dict]) -> Optional[dict]:
        mapped = self.index_params(params)
        validate_operation_arguments("create_index", {"index_params": mapped})
        exists = self._vector_index_exists()
        if exists and mapped is None:
            return None
        operation, request = self._index_request(mapped, exists)
        return self.transport.call(operation, **request)

    @_with_async_native_backend_session
    async def aquery(self, operation: str, **params: Any) -> dict:
        try:
            return await self.transport.acall(operation, **params)
        except Exception as exc:
            if not layout_error(exc):
                raise
            await self.transport.amigrate(self.name)
            return await self.transport.acall(operation, **params)

    @_with_async_native_backend_session
    async def run_async(self, method: str, *args: Any, **kwargs: Any) -> Any:
        if not isinstance(self.transport, AsyncNativeTransport):
            return await asyncio.to_thread(getattr(self, method), *args, **kwargs)
        if method == "describe_index":
            kind, job_name = args
            if job_name is not None:
                raise ValueError(
                    "Native VecDB job status is unsupported; "
                    "describe the physical index instead"
                )
            return await self.transport.adescribe_index(self.name, kind)
        if method == "create_index":
            mapped = self.index_params(args[0])
            validate_operation_arguments("create_index", {"index_params": mapped})
            status = await self.transport.adescribe_index(self.name)
            exists = bool(status["details"]["indexes"])
            if exists and mapped is None:
                return None
            operation, request = self._index_request(mapped, exists)
            return await self.transport.acall(operation, **request)
        if method == "create_text_index":
            await self.transport.acreate_text_index(self.name, args[0])
            return
        if method == "prepare":
            columns = await self.transport.acolumns(self.name)
            created = not columns
            if created:
                await self.transport.acall(
                    "create_vector_table",
                    **self._table_creation_params(),
                )
                columns = await self.transport.acolumns(self.name)
            await self.transport.adiscover(self.name, columns)
            logger.info(
                "Table %s %s...",
                self.quoted,
                "created successfully" if created else "already exists",
            )
            return
        if method == "add":
            texts, vectors, metadatas, ids, mutate = args
            records = self.records(texts, vectors, metadatas, ids)
            if not records:
                return []
            if not mutate:
                inserted = await self.transport.ainsert(self.name, records)
                logical_ids = {record["id"]: id_ for record, id_ in zip(records, ids)}
                return [logical_ids[id_] for id_ in inserted]
            await self.transport.acall(
                "upsert_vectors", table_name=self.name, vectors=records
            )
            return ids
        if method == "delete":
            await self.transport.acall(
                "delete_vectors",
                table_name=self.name,
                ids=None if args[0] is None else [storage_id(id_) for id_ in args[0]],
            )
            return
        if method == "get":
            ids = args[0]
            result: list[Document] = []
            for offset in range(0, len(ids), 100):
                batch = [storage_id(id_) for id_ in ids[offset : offset + 100]]
                response = await self.aquery(
                    "list_vectors",
                    table_name=self.name,
                    ids=batch,
                    limit=len(batch),
                    offset=0,
                )
                result.extend(document(row) for row in rows(response, "vectors"))
            return result
        if method == "search":
            embedding, k, filter = args[:3]
            include = args[3] if len(args) > 3 else False
            request, fallback = self._search_request(
                embedding, k, filter, include, kwargs
            )
            response = (
                await self.transport.afilter_search(self.name, filter, request)
                if fallback
                else await self.aquery("search", **request)
            )
            return self.search_results(response, include)
        if method == "hybrid_search":
            return await self.transport.ahybrid_search(self.name, *args)
        if method == "text_search":
            return await self.transport.atext_search(self.name, *args)
        raise ValueError("Unsupported async VecDB operation")


class AsyncNativeTransport(NativeTransport):
    """Same package contract using python-oracledb's async connections."""

    @asynccontextmanager
    async def asession(self) -> AsyncIterator[AsyncNativeTransport]:
        async with _aget_connection(self.client) as connection:
            bound = copy(self)
            bound.client = connection
            bound.version = parse_version(connection.version)
            yield bound

    @_with_async_native_session
    async def adiscover(
        self, name: str, columns: Optional[set[str]] = None
    ) -> _TableInfo:
        async with _aget_connection(self.client) as connection:
            with connection.cursor() as cursor:
                if columns is None:
                    columns = await self.acolumns(name)
                meta = (
                    "CONTENT_METADATA" if "CONTENT_METADATA" in columns else "METADATA"
                )
                statement, binds = _metadata_index_query(name, meta)
                await cursor.execute(statement, **binds)
                info = _table_info(columns, await cursor.fetchone())
                self._remember_table(name, info)
                return self._table_info_cache[self._table_key(name)]

    async def _asearch_info(self, name: str) -> _TableInfo:
        info = self._table_info_cache.get(self._table_key(name))
        return info if info is not None else await self.adiscover(name)

    async def acolumns(self, name: str) -> set[str]:
        async with _aget_connection(self.client) as connection:
            with connection.cursor() as cursor:
                return await _ainspect_columns(cursor, name)

    @_with_async_native_session
    async def acall(self, operation: str, **params: Any) -> dict:
        statement, binds = self._statement(operation, params)
        async with _aget_connection(self.client) as connection:
            with connection.cursor() as cursor:
                if (
                    operation == "drop_vector_table"
                    and connection.transaction_in_progress is True
                ):
                    raise RuntimeError("VecDB table drop requires no pending work")
                if (
                    operation in {"create_index", "rebuild_index"}
                    and connection.transaction_in_progress is True
                ):
                    raise RuntimeError(
                        "Index submission requires a connection without pending work"
                    )
                if operation in {"create_vector_table", "drop_vector_table"}:
                    self._forget_table(params["name"])
                out = cursor.var(oracledb.DB_TYPE_CLOB)
                cursor.setinputsizes(
                    **{
                        key: oracledb.DB_TYPE_CLOB
                        for key in binds
                        if f"JSON(:{key})" in statement
                    }
                )
                if operation not in {"search", "query", "list_vectors"}:
                    try:
                        await cursor.execute(
                            function_preflight(statement), result=out, **binds
                        )
                    except Exception as error:
                        signature_error(operation, connection.version, error)
                await cursor.execute(statement, result=out, **binds)
                value = out.getvalue()
                if hasattr(value, "read"):
                    value = await value.read()
                return as_dict(value)

    @_with_async_native_session
    async def apreflight_index(self, hybrid: bool = False) -> None:
        async with _aget_connection(self.client) as connection:
            with connection.cursor() as cursor:
                if connection.transaction_in_progress:
                    raise RuntimeError(
                        "Index DDL requires a connection without pending work"
                    )
                await cursor.execute(INDEX_FEATURES_SQL)
                require_index_features(await cursor.fetchall(), hybrid)

    @_with_async_native_session
    async def adescribe_index(self, name: str, kind: str = "vector") -> dict:
        owner, binds = _catalog_target(name)
        statement = index_status_query(owner, kind)
        async with _aget_connection(self.client) as connection:
            with connection.cursor() as cursor:
                await cursor.execute(statement, **binds)
                return native_index_status(await cursor.fetchall())

    @_with_async_native_session
    async def amigrate(self, name: str) -> None:
        quoted, _ = table_identifier(name)
        async with _aget_connection(self.client) as connection:
            self.version = parse_version(connection.version)
            if self.version < CURRENT_VERSION:
                raise ValueError("Layout migration requires Oracle Database 23.26.3+")
            if connection.transaction_in_progress:
                raise RuntimeError(
                    "VecDB migration requires a connection without pending work"
                )
            with connection.cursor() as cursor:
                db_version = cursor.var(str, size=128)
                compatibility = cursor.var(str, size=128)
                await cursor.execute(
                    COMPATIBILITY_SQL,
                    db_version=db_version,
                    compatibility=compatibility,
                )
                _compatible_backend(
                    self.version, compatibility.getvalue(), "vecdb", "vecdb"
                )

                async def inspect() -> set[str]:
                    return await _ainspect_columns(cursor, name)

                self._forget_table(name)
                for action in _migration_actions(await inspect(), quoted):
                    try:
                        await cursor.execute(action)
                    except oracledb.DatabaseError as exc:
                        if exc.args[0].code not in {
                            904,
                            1430,
                        } or action in _migration_actions(await inspect(), quoted):
                            raise
                if not CURRENT_COLUMNS <= await inspect():
                    raise RuntimeError(
                        "VecDB migration did not produce the required layout"
                    )

    @_with_async_native_session
    async def acreate_metadata_index(
        self,
        name: str,
        index: str,
        *,
        hybrid: bool = False,
        build: Optional[Callable[[str, str, str], str]] = None,
    ) -> None:
        columns = await self.acolumns(name)
        meta = "CONTENT_METADATA" if "CONTENT_METADATA" in columns else "METADATA"
        async with _aget_connection(self.client) as connection:
            with connection.cursor() as cursor:
                statement, binds = _metadata_index_query(name, meta)
                await cursor.execute(statement, **binds)
                found = await cursor.fetchall()
                if len(found) > 1:
                    raise ValueError("Multiple metadata domain indexes found")
                existing = found[0] if found else None
                physical, ddl, restore = _metadata_index_plan(
                    name, index, columns, existing, hybrid, build
                )
                if ddl is None:
                    self._remember_table(name, _table_info(columns, existing))
                    return
                if connection.transaction_in_progress:
                    raise RuntimeError(
                        "Index DDL requires a connection without pending work"
                    )
                if not existing:
                    statement, binds = _index_query(physical)
                    await cursor.execute(statement, **binds)
                    if await cursor.fetchone():
                        raise ValueError(
                            "Index name is already used by an incompatible index"
                        )
                await self.apreflight_index(hybrid)
                self._forget_table(name)
                build_started = False
                try:
                    if restore:
                        await cursor.execute(f"DROP INDEX {physical}")
                    build_started = True
                    await cursor.execute(ddl)
                except BaseException:
                    if restore:
                        await _complete_cleanup(
                            self._arestore_index(
                                cursor, physical, restore, build_started
                            )
                        )
                    raise
                self._remember_table(
                    name,
                    _TableInfo(frozenset(columns), physical if hybrid else None),
                )
                logger.info(
                    "Created %s metadata index %s",
                    "hybrid" if hybrid else "text",
                    physical,
                )

    async def _arestore_index(
        self, cursor: Any, physical: str, restore: str, build_started: bool
    ) -> None:
        try:
            statement, binds = _index_query(physical)
            await cursor.execute(statement, **binds)
            present = await cursor.fetchone()
            if present and build_started:
                await cursor.execute(f"DROP INDEX {physical} FORCE")
            if not present or build_started:
                await cursor.execute(restore)
            logger.warning(
                "Retained or restored metadata index %s after replacement failed",
                physical,
            )
        except BaseException as recovery:
            raise RuntimeError(
                f"Could not restore metadata index {physical} after "
                "replacement failed; manual recovery is required"
            ) from recovery

    async def acreate_text_index(self, name: str, index: str) -> None:
        await self.acreate_metadata_index(name, index)

    @_with_async_native_session
    async def ainsert(self, name: str, records: list[dict]) -> list[str]:
        import array

        quoted, _ = table_identifier(name)
        columns = await self.acolumns(name)
        _layout(columns)
        meta = "CONTENT_METADATA" if "CONTENT_METADATA" in columns else "METADATA"
        values = [
            (r["id"], array.array("f", r["dense_vector"]), r["metadata"])
            for r in records
        ]
        async with _aget_connection(self.client) as connection:
            # Match regular OracleVS SQL inserts: report row errors and commit
            # successful rows, including any caller work on this connection.
            with connection.cursor() as cursor:
                cursor.setinputsizes(
                    None, oracledb.DB_TYPE_VECTOR, oracledb.DB_TYPE_JSON
                )
                await cursor.executemany(
                    f"INSERT INTO {quoted} (ID, DENSE_VECTOR, {meta}) "
                    "VALUES (:1, :2, :3)",
                    values,
                    batcherrors=True,
                )
                errors = cursor.getbatcherrors() or []
                for error in errors:
                    logger.warning(
                        "Could not insert row at offset %s due to error: %s",
                        error.offset,
                        error.message,
                    )
                await connection.commit()
                failed = {error.offset for error in errors}
                return [
                    record["id"] for i, record in enumerate(records) if i not in failed
                ]

    @_with_async_native_session
    async def ahybrid_search(self, name: str, params: dict, scores: bool) -> list:
        validate_hybrid_request(params)
        async with _aget_connection(self.client) as connection:
            info = await self._asearch_info(name)
            # An index may have been created externally since store initialization.
            if info.hybrid_index is None:
                info = await self.adiscover(name)
            for attempt in range(2):
                if info.hybrid_index is None:
                    raise ValueError(
                        "Create a hybrid vector index on the VecDB "
                        "metadata column first"
                    )
                requested = params.get("hybrid_index_name")
                index = (
                    requested
                    if requested and len(_identifier_parts(requested)) == 2
                    else info.hybrid_index
                )
                request = hybrid_search_request(
                    name, {**params, "hybrid_index_name": index}
                )
                try:
                    with connection.cursor() as cursor:
                        cursor.setinputsizes(search_params=oracledb.DB_TYPE_CLOB)
                        await cursor.execute(
                            tag_sql(
                                _hybrid_sql(name, set(info.columns)),
                                _hybrid_search_mode(params),
                                backend="vecdb",
                            ),
                            search_params=json.dumps(request, allow_nan=False),
                        )
                        return _hybrid_documents(
                            await cursor.fetchall(),
                            scores,
                            keyword_only=_hybrid_search_mode(params) == "keyword",
                        )
                except Exception as error:
                    if attempt or not _metadata_search_error(error):
                        raise
                    self._forget_table(name)
                    info = await self.adiscover(name)
            raise AssertionError("Unreachable metadata search retry")

    @_with_async_native_session
    async def afilter_search(self, name: str, filter: dict, request: dict) -> dict:
        async with _aget_connection(self.client) as connection:
            info = await self._asearch_info(name)
            for attempt in range(2):
                sql, binds = _filter_query(name, info.metadata_column, request, filter)
                try:
                    with connection.cursor() as cursor:
                        await cursor.execute(sql, **binds)
                        return _filter_response(await cursor.fetchall(), request)
                except Exception as error:
                    if attempt or not _metadata_search_error(error):
                        raise
                    self._forget_table(name)
                    info = await self.adiscover(name)
            raise AssertionError("Unreachable filter search retry")

    @_with_async_native_session
    async def atext_search(
        self, name: str, query: str, k: int, scores: bool = False
    ) -> list:
        quoted, _ = table_identifier(name)
        async with _aget_connection(self.client) as connection:
            info = await self._asearch_info(name)
            for attempt in range(2):
                meta = info.metadata_column
                sql = (
                    f"SELECT id, {meta}, SCORE(1) FROM {quoted} "
                    f"WHERE JSON_TEXTCONTAINS({meta}, '$.{TEXT_KEY}', :query, 1) "
                    "ORDER BY SCORE(1) DESC, id FETCH FIRST :k ROWS ONLY"
                )
                try:
                    with connection.cursor() as cursor:
                        await cursor.execute(
                            tag_sql(sql, "text", backend="vecdb"), query=query, k=k
                        )
                        return _text_documents(await cursor.fetchall(), scores)
                except Exception as error:
                    if attempt or not _metadata_search_error(error):
                        raise
                    self._forget_table(name)
                    info = await self.adiscover(name)
            raise AssertionError("Unreachable metadata search retry")


async def anative_selection(
    connection: Any, name: str, backend: str
) -> tuple[str, tuple[int, ...]]:
    version = parse_version(connection.version)
    with connection.cursor() as cursor:
        columns = await _ainspect_columns(cursor, name)
        kind = _table_kind(
            columns, await _ais_vecdb_table(cursor, name) if columns else False
        )
        if backend == "sql":
            if kind == "vecdb":
                raise ValueError(
                    "The SQL backend cannot use a VecDB table layout; "
                    "use backend='auto' or backend='vecdb'"
                )
            return "sql", version
        if kind == "sql":
            if backend == "vecdb":
                raise ValueError(
                    "The VecDB backend cannot use an ordinary OracleVS table layout; "
                    "use backend='auto' or backend='sql'"
                )
            return "sql", version
        if kind == "missing":
            name_error = _package_name_error(name)
            if name_error:
                if backend == "vecdb":
                    raise ValueError(name_error)
                return "sql", version
        if version < LEGACY_VERSION:
            if backend == "vecdb" or kind == "vecdb":
                raise ValueError("VecDB requires Oracle Database 23.26.1+")
            return "sql", version
        await cursor.execute(VECDB_CAPABILITY_SQL)
        supported, compatibility = await cursor.fetchone()
        if not supported:
            if backend == "vecdb" or kind == "vecdb":
                raise ValueError("DBMS_VECTOR_DATABASE.SEARCH is unavailable")
            return "sql", version
        if (
            kind == "vecdb"
            and version < CURRENT_VERSION
            and "CONTENT_METADATA" in columns
        ):
            raise ValueError(
                "Current VecDB table layout cannot be used on a legacy database"
            )
        return _compatible_backend(version, compatibility, kind, backend), version
