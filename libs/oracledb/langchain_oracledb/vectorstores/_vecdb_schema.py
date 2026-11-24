# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""VecDB table identity, layout compatibility, and index lifecycle SQL."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Optional

import oracledb

from ._vecdb_adapter import CURRENT_VERSION, as_dict
from .utils import _quote_indentifier

LEGACY_VERSION = (23, 26, 1)


QUOTED_NAMES_VERSION = (23, 27, 2)


VECDB_CAPABILITY_SQL = """
SELECT COUNT(*) AS search_supported
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
    """Build an exact catalog predicate for migration or index operations."""
    parts = _identifier_parts(name)
    binds = {"name": parts[-1]}
    if len(parts) == 2:
        binds["owner"] = parts[0]
        return ":owner", binds
    return "SYS_CONTEXT('USERENV', 'CURRENT_SCHEMA')", binds


def _layout_query(name: str) -> tuple[str, dict]:
    owner, binds = _catalog_target(name)
    return (
        "SELECT column_name FROM all_tab_columns "
        f"WHERE owner = {owner} AND table_name = :name",
        binds,
    )


def _inspect_columns(cursor: Any, name: str) -> set[str]:
    statement, binds = _layout_query(name)
    cursor.execute(statement, **binds)
    return {row[0] for row in cursor.fetchall()}


async def _ainspect_columns(cursor: Any, name: str) -> set[str]:
    statement, binds = _layout_query(name)
    await cursor.execute(statement, **binds)
    return {row[0] for row in await cursor.fetchall()}


SQL_COLUMNS = {"ID", "TEXT", "METADATA", "EMBEDDING"}


def _default_columns(version: tuple[int, ...]) -> set[str]:
    return set(
        CURRENT_COLUMNS if version >= CURRENT_VERSION else BASE_COLUMNS | {"METADATA"}
    )


def _sql_layout(cursor: Any, name: str) -> str:
    try:
        cursor.execute(
            "SELECT id, text, metadata, embedding "
            f"FROM {_quote_indentifier(name)} WHERE ROWNUM < 1"
        )
    except oracledb.DatabaseError as error:
        code = getattr(error.args[0], "code", None)
        if code == 942:
            return "missing"
        if code == 904:
            return "other"
        raise
    return "sql"


async def _asql_layout(cursor: Any, name: str) -> str:
    try:
        await cursor.execute(
            "SELECT id, text, metadata, embedding "
            f"FROM {_quote_indentifier(name)} WHERE ROWNUM < 1"
        )
    except oracledb.DatabaseError as error:
        code = getattr(error.args[0], "code", None)
        if code == 942:
            return "missing"
        if code == 904:
            return "other"
        raise
    return "sql"


def _sparse_compatibility_error(error: oracledb.DatabaseError) -> bool:
    """Recognize the database rejecting sparse storage, not arbitrary DDL errors."""
    detail = error.args[0] if error.args else None
    message = str(getattr(detail, "message", error)).upper()
    return (
        # Autonomous also gates sparse storage on rolling-patch enablement.
        getattr(detail, "code", None) in {406, 45493}
        and "ORA-00722" in message
        and bool(re.search(r"SQL VECTOR SPARSE (?:TYPE|FORMAT)\b", message))
    )


def creation_can_fallback(
    cursor: Any, name: str, error: oracledb.DatabaseError
) -> bool:
    """Permit SQL creation only after a sparse rejection left no table behind."""
    if not _sparse_compatibility_error(error):
        return False
    try:
        return _sql_layout(cursor, name) == "missing"
    except Exception:
        # Preserve the creation error if object absence cannot be established.
        return False


async def acreation_can_fallback(
    cursor: Any, name: str, error: oracledb.DatabaseError
) -> bool:
    if not _sparse_compatibility_error(error):
        return False
    try:
        return await _asql_layout(cursor, name) == "missing"
    except Exception:
        return False


def _vecdb_layout(cursor: Any, name: str) -> set[str]:
    # Parse only the columns used by direct SQL, without reading table rows.
    for metadata in ("CONTENT_METADATA", "METADATA"):
        try:
            cursor.execute(
                f"SELECT id, dense_vector, {metadata} "
                f"FROM {_quote_indentifier(name)} WHERE ROWNUM < 1"
            )
            return BASE_COLUMNS | {metadata}
        except oracledb.DatabaseError as error:
            if (
                metadata != "CONTENT_METADATA"
                or getattr(error.args[0], "code", None) != 904
            ):
                raise
    raise AssertionError("Unreachable VecDB column resolution")


async def _avecdb_layout(cursor: Any, name: str) -> set[str]:
    for metadata in ("CONTENT_METADATA", "METADATA"):
        try:
            await cursor.execute(
                f"SELECT id, dense_vector, {metadata} "
                f"FROM {_quote_indentifier(name)} WHERE ROWNUM < 1"
            )
            return BASE_COLUMNS | {metadata}
        except oracledb.DatabaseError as error:
            if (
                metadata != "CONTENT_METADATA"
                or getattr(error.args[0], "code", None) != 904
            ):
                raise
    raise AssertionError("Unreachable VecDB column resolution")


def _package_name_error(
    name: str, version: Optional[tuple[int, ...]] = None
) -> Optional[str]:
    parts = _identifier_parts(name)
    if len(parts) != 1:
        return (
            "Native VecDB package operations do not support "
            f"schema-qualified table names and cannot preserve {name!r}. "
            "Connect as the owning user and use an unqualified name. "
            "Full support for this identifier requires support in "
            "the underlying VecDB package."
        )
    if (
        version is not None
        and version < QUOTED_NAMES_VERSION
        and re.fullmatch(r"[A-Z][A-Z0-9_$#]*", name.strip()) is None
    ):
        return (
            "Native VecDB requires unquoted uppercase table names before "
            f"Oracle Database 23.27.2 and cannot preserve {name!r}. "
            "Use backend='sql' or an unquoted uppercase table name."
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
    return VECDB_TABLE_SQL, {"name": table_identifier(name)[0]}


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


def _described_table(
    name: str, result: tuple, description: Optional[dict] = None
) -> bool:
    payload, schema = result
    # The package returns SQL NULL for an existing, unregistered SQL table.
    if payload is None:
        return False
    response = as_dict(payload)
    if not all(isinstance(response.get(key), str) for key in ("owner", "table_name")):
        raise ValueError("Malformed VecDB table identity")
    parts = _identifier_parts(name)
    owner = parts[0] if len(parts) == 2 else schema
    matched = response["owner"] == owner and response["table_name"] == parts[-1]
    if matched and description is not None:
        description.update(response)
    return matched


def _is_vecdb_table(
    cursor: Any, name: str, *, description: Optional[dict] = None
) -> bool:
    """Describe one exact table without transferring the package registry."""
    statement, binds = _description_statement(name)
    try:
        cursor.execute(statement, **binds)
    except oracledb.DatabaseError as error:
        # A direct call avoids a capability query on every successful reopen.
        # Only an unavailable operation can turn a compile error into nonmembership.
        if getattr(error.args[0], "code", None) in {904, 6550, 6576}:
            cursor.execute(VECDB_RECOGNITION_SQL)
            if not cursor.fetchone()[0]:
                return False
            raise
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
    return _described_table(name, result, description)


async def _ais_vecdb_table(
    cursor: Any, name: str, *, description: Optional[dict] = None
) -> bool:
    statement, binds = _description_statement(name)
    try:
        await cursor.execute(statement, **binds)
    except oracledb.DatabaseError as error:
        # A direct call avoids a capability query on every successful reopen.
        # Only an unavailable operation can turn a compile error into nonmembership.
        if getattr(error.args[0], "code", None) in {904, 6550, 6576}:
            await cursor.execute(VECDB_RECOGNITION_SQL)
            if not (await cursor.fetchone())[0]:
                return False
            raise
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
    return _described_table(name, (payload, schema), description)


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
    if "SPARSE_VECTOR" not in columns:
        # Let Oracle validate sparse support before changing the legacy layout.
        actions.append(f"ALTER TABLE {quoted} ADD SPARSE_VECTOR VECTOR(*, *, SPARSE)")
    if "METADATA" in columns:
        actions.append(
            f"ALTER TABLE {quoted} RENAME COLUMN METADATA TO CONTENT_METADATA"
        )
    for name, sql_type in (
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


def _insert_layout_error(error: Exception, metadata_column: str) -> bool:
    """Recognize rejection of the exact cached column in our INSERT statement."""
    return (
        isinstance(error, oracledb.DatabaseError)
        and getattr(error.args[0], "code", None) == 904
        and bool(
            re.search(
                r'ORA-00904:\s*(?:(?:"[^"]+"|\w+)\.)*"?'
                + re.escape(metadata_column)
                + r'"?(?=\s*(?::|$))',
                str(error).upper().replace('\\"', '"'),
            )
        )
    )


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


def _metadata_search_error(error: Exception, metadata_column: str) -> bool:
    """Refresh a missing index or the exact cached metadata column once."""
    message = str(error).upper().replace('\\"', '"')
    return bool(
        re.search(r"\b(?:ORA-01418|DRG-10502):", message)
        or re.search(
            r'ORA-00904:\s*(?:(?:"[^"]+"|\w+)\.)*"?'
            + re.escape(metadata_column)
            + r'"?(?=\s*(?::|$))',
            message,
        )
    )


def _sql_selection(columns: set[str], backend: str) -> Optional[str]:
    """Resolve ordinary or explicitly requested SQL before VecDB recognition."""
    if {
        "ID",
        "EMBEDDING",
        "METADATA",
        "TEXT",
    } <= columns and "DENSE_VECTOR" not in columns:
        if backend == "vecdb":
            raise ValueError(
                "The VecDB backend cannot use an ordinary OracleVS table layout; "
                "use backend='auto' or backend='sql'"
            )
        return "sql"
    if backend == "sql":
        if columns:
            _layout(columns)
            raise ValueError(
                "Incompatible OracleVS table layout for backend='sql'; "
                "expected ID, TEXT, METADATA, and EMBEDDING columns"
            )
        return "sql"
    return None


def _vecdb_selection(
    name: str,
    backend: str,
    version: tuple[int, ...],
    kind: str,
    columns: set[str],
) -> Optional[str]:
    """Apply VecDB policy; None means a new table needs a capability probe."""
    if kind == "sql":
        return _sql_selection(SQL_COLUMNS, backend)
    if kind == "missing" or (kind == "vecdb" and version < QUOTED_NAMES_VERSION):
        name_error = _package_name_error(name, version)
        if name_error:
            if backend == "vecdb" or kind == "vecdb":
                raise ValueError(name_error)
            return "sql"
    if version < LEGACY_VERSION:
        if backend == "vecdb" or kind == "vecdb":
            raise ValueError("VecDB requires Oracle Database 23.26.1+")
        return "sql"
    if kind == "missing":
        return None
    if kind == "vecdb" and version < CURRENT_VERSION and "CONTENT_METADATA" in columns:
        raise ValueError(
            "Current VecDB table layout cannot be used on a legacy database"
        )
    return "vecdb"


def _creation_selection(backend: str, supported: Any) -> str:
    """Resolve a new table's backend from the SEARCH capability result."""
    if not supported:
        if backend == "vecdb":
            raise ValueError("DBMS_VECTOR_DATABASE.SEARCH is unavailable")
        return "sql"
    return "vecdb"


def native_selection(
    connection: Any,
    name: str,
    backend: str,
    *,
    columns: Optional[set[str]] = None,
    description: Optional[dict] = None,
    layout: Optional[set[str]] = None,
) -> tuple[str, tuple[int, ...]]:
    """Select a backend using columns already inspected on this connection, if given."""
    version = parse_version(connection.version)
    with connection.cursor() as cursor:
        infer_layout = columns is None
        if columns is None:
            sql_kind = _sql_layout(cursor, name)
            columns = (
                set(SQL_COLUMNS)
                if sql_kind == "sql"
                else _default_columns(version)
                if sql_kind == "other"
                else set()
            )
        selected = _sql_selection(columns, backend)
        if selected is not None:
            return selected, version
        kind = _table_kind(
            columns,
            _is_vecdb_table(cursor, name, description=description)
            if columns
            else False,
        )
        if kind == "vecdb" and infer_layout:
            columns = _vecdb_layout(cursor, name)
        if layout is not None:
            layout.update(columns)
        selected = _vecdb_selection(name, backend, version, kind, columns)
        if selected is None:
            # Only eligible new tables need the SEARCH capability probe.
            cursor.execute(VECDB_CAPABILITY_SQL)
            supported = cursor.fetchone()[0]
            selected = _creation_selection(backend, supported)
        return selected, version


async def anative_selection(
    connection: Any,
    name: str,
    backend: str,
    *,
    columns: Optional[set[str]] = None,
    description: Optional[dict] = None,
    layout: Optional[set[str]] = None,
) -> tuple[str, tuple[int, ...]]:
    """Select a backend using columns already inspected on this connection, if given."""
    version = parse_version(connection.version)
    with connection.cursor() as cursor:
        infer_layout = columns is None
        if columns is None:
            sql_kind = await _asql_layout(cursor, name)
            columns = (
                set(SQL_COLUMNS)
                if sql_kind == "sql"
                else _default_columns(version)
                if sql_kind == "other"
                else set()
            )
        selected = _sql_selection(columns, backend)
        if selected is not None:
            return selected, version
        kind = _table_kind(
            columns,
            await _ais_vecdb_table(cursor, name, description=description)
            if columns
            else False,
        )
        if kind == "vecdb" and infer_layout:
            columns = await _avecdb_layout(cursor, name)
        if layout is not None:
            layout.update(columns)
        selected = _vecdb_selection(name, backend, version, kind, columns)
        if selected is None:
            # Only eligible new tables need the SEARCH capability probe.
            await cursor.execute(VECDB_CAPABILITY_SQL)
            supported = (await cursor.fetchone())[0]
            selected = _creation_selection(backend, supported)
        return selected, version
