# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Annotations for package-managed tables.

The single annotation uses ordered, semicolon-separated key=value fields.
Version describes the creating package, not the table schema or later clients.
SQL backfill preserves existing markers and requires an idle connection.
"""

import logging
import re
from importlib.metadata import PackageNotFoundError, version
from typing import Literal, Optional

from oracledb import AsyncConnection, Connection, DatabaseError

logger = logging.getLogger(__name__)

TableType = Literal[
    "vector_store", "semantic_cache", "exact_cache", "chat_history", "kv_store"
]
Backend = Literal["sql", "vecdb"]

ANNOTATION_NAME = "LANGCHAIN_ORACLE"
_PACKAGE = "langchain-oracledb"
_TABLE_TYPES = {
    "vector_store",
    "semantic_cache",
    "exact_cache",
    "chat_history",
    "kv_store",
}


def supports_table_annotations(connection: Connection) -> bool:
    """Check annotation availability for chat history's older databases.

    The dictionary view shipped with annotations, including the 19.28 backport.
    A zero-row query reads no application data. Only a missing/inaccessible view
    disables telemetry; other database errors retain their normal behavior.
    """
    with connection.cursor() as cursor:
        try:
            cursor.execute("SELECT 1 FROM user_annotations_usage WHERE 1 = 0")
        except DatabaseError as exc:
            if exc.args[0].code == 942:
                return False
            raise
    return True


def table_annotations(
    table_type: TableType,
    backend: Backend = "sql",
    *,
    creator_version: Optional[str] = None,
) -> dict[str, str]:
    """Return annotations using installed package metadata and a backend field."""
    if table_type not in _TABLE_TYPES:
        raise ValueError("Unsupported LangChain table type")
    if backend not in {"sql", "vecdb"}:
        raise ValueError("Unsupported LangChain table backend")
    package_version = creator_version
    if package_version is None:
        try:
            package_version = version(_PACKAGE)
        except PackageNotFoundError:
            package_version = "unknown"
    if not package_version or any(c in package_version for c in ";=\r\n"):
        raise ValueError("Package version contains an annotation delimiter")
    return {
        ANNOTATION_NAME: (
            f"format=1;package={_PACKAGE};version={package_version};"
            f"type={table_type};backend={backend}"
        )
    }


def table_annotation_clause(
    table_type: TableType,
    backend: Backend = "sql",
    *,
    creator_version: Optional[str] = None,
    if_not_exists: bool = False,
) -> str:
    """Render the annotation for CREATE/ALTER TABLE without accepting object names."""
    value = table_annotations(table_type, backend, creator_version=creator_version)[
        ANNOTATION_NAME
    ].replace("'", "''")
    operation = "ADD IF NOT EXISTS " if if_not_exists else ""
    return f"ANNOTATIONS ({operation}{ANNOTATION_NAME} '{value}')"


_BACKFILL_QUERY = """
    SELECT t.owner, t.table_name
    FROM all_tables t
    WHERE t.owner = NVL(:owner, SYS_CONTEXT('USERENV', 'CURRENT_SCHEMA'))
      AND t.table_name = :name
      AND NOT EXISTS (
          SELECT 1 FROM all_annotations_usage a
          WHERE a.annotation_owner = t.owner AND a.object_name = t.table_name
            AND a.object_type = 'TABLE' AND a.column_name IS NULL
            AND a.domain_name IS NULL AND a.annotation_name = :annotation_name
      )
"""


def _backfill_binds(table_name: str) -> dict[str, Optional[str]]:
    # Use the existing quoting contract, preserving dots within quoted components.
    from .vectorstores.utils import _quote_indentifier

    parts = re.findall(r'"([^"]+)"', _quote_indentifier(table_name))
    if len(parts) not in {1, 2}:
        raise ValueError("A table name must have a name and optional schema")
    return {
        "owner": parts[0] if len(parts) == 2 else None,
        "name": parts[-1],
        "annotation_name": ANNOTATION_NAME,
    }


def _backfill_ddl(target: tuple[str, str], table_type: TableType) -> str:
    owner, name = target
    quoted = '"' + owner.replace('"', '""') + '"."' + name.replace('"', '""') + '"'
    return f"ALTER TABLE {quoted} " + table_annotation_clause(
        table_type, creator_version="unknown", if_not_exists=True
    )


def _skip_backfill(error: DatabaseError) -> bool:
    # Missing dictionary/table, insufficient ALTER privileges, or a busy object:
    # optional telemetry must not stop use of an otherwise accessible SQL table.
    if error.args[0].code in {942, 1031, 54}:
        logger.warning("SQL annotation backfill skipped (ORA-%05d)", error.args[0].code)
        return True
    return False


def backfill_table_annotation(
    connection: Connection, table_name: str, table_type: TableType
) -> None:
    """Add a missing SQL table marker without committing caller work.

    Only direct tables are targets; synonyms/views are left unchanged. No commit
    or rollback is issued. The ALTER itself commits, so require a known idle
    connection both before inspection and immediately before DDL.
    """
    if connection.transaction_in_progress is not False:
        logger.debug("SQL annotation backfill deferred: connection has pending work")
        return
    try:
        with connection.cursor() as cursor:
            cursor.execute(_BACKFILL_QUERY, **_backfill_binds(table_name))
            target = cursor.fetchone()
            if target and connection.transaction_in_progress is False:
                cursor.execute(_backfill_ddl(target, table_type))
    except DatabaseError as error:
        if not _skip_backfill(error):
            raise


async def abackfill_table_annotation(
    connection: AsyncConnection, table_name: str, table_type: TableType
) -> None:
    """Async equivalent of backfill_table_annotation on the same connection."""
    if connection.transaction_in_progress is not False:
        logger.debug("SQL annotation backfill deferred: connection has pending work")
        return
    try:
        with connection.cursor() as cursor:
            await cursor.execute(_BACKFILL_QUERY, **_backfill_binds(table_name))
            target = await cursor.fetchone()
            if target and connection.transaction_in_progress is False:
                await cursor.execute(_backfill_ddl(target, table_type))
    except DatabaseError as error:
        if not _skip_backfill(error):
            raise
