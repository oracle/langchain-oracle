# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""Native VecDB execution with operation-owned sessions and cached metadata."""

from __future__ import annotations

import array
import asyncio
import json
import logging
from contextlib import asynccontextmanager, contextmanager
from copy import copy
from functools import wraps
from typing import Any, AsyncIterator, Callable, Iterator, Optional, TypeVar, cast

import oracledb

from .._telemetry import tag_sql
from . import _vecdb_schema as _schema
from . import utils as _db
from ._filters import _generate_where_clause
from ._vecdb_adapter import (
    CURRENT_VERSION,
    TEXT_KEY,
    SQLNetAdapter,
    _hybrid_documents,
    _text_documents,
    as_dict,
    hybrid_search_request,
    validate_hybrid_request,
)
from ._vecdb_capabilities import index_status_query, native_index_status
from .utils import _quote_indentifier

# Preserve the logger category used before the transport module split.
logger = logging.getLogger(__package__ + "._vecdb")


def _filter_query(
    name: str, metadata_column: str, request: dict, filter: dict
) -> tuple:
    quoted, _ = _schema.table_identifier(name)
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


def _hybrid_search_mode(params: dict) -> str:
    """Match the retriever mode from its vector/text request sections."""
    if "vector" in params and "text" not in params:
        return "semantic"
    if "text" in params and "vector" not in params:
        return "keyword"
    return "hybrid"


def _hybrid_sql(name: str, columns: set[str]) -> str:
    quoted, _ = _schema.table_identifier(name)
    _schema._layout(columns)
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


def _index_status_needs_catalog(error: oracledb.DatabaseError) -> bool:
    code = getattr(error.args[0], "code", None)
    return code == 1418 or (code == 904 and "INDEX_BUILD_STATUS" in str(error).upper())


class NativeTransport:
    """Execute a bounded set of package calls using caller-owned connections."""

    def __init__(self, client: Any, version: tuple[int, ...]):
        self.client = client
        self.version = version
        # Bound pool transports share snapshots, never mutable session state.
        self._table_info_cache: dict[tuple, _schema._TableInfo] = {}

    def _table_key(self, name: str) -> tuple:
        return tuple(_schema._identifier_parts(name)), self.version

    def _remember_table(self, name: str, info: _schema._TableInfo) -> None:
        if info.hybrid_index:
            # Unqualified table/index names resolve in the executing session,
            # just as they do in the regular LangChain SQL path.
            info = _schema._TableInfo(
                info.columns,
                _quote_indentifier(
                    f'"{_schema._identifier_parts(info.hybrid_index)[-1]}"'
                ),
            )
        self._table_info_cache[self._table_key(name)] = info

    def _forget_table(self, name: str) -> None:
        self._table_info_cache.pop(self._table_key(name), None)

    @_with_native_session
    def discover(
        self, name: str, columns: Optional[set[str]] = None
    ) -> _schema._TableInfo:
        with _db._get_connection(self.client) as connection:
            with connection.cursor() as cursor:
                if columns is None:
                    columns = self.columns(name)
                meta = (
                    "CONTENT_METADATA" if "CONTENT_METADATA" in columns else "METADATA"
                )
                statement, binds = _schema._metadata_index_query(name, meta)
                cursor.execute(statement, **binds)
                info = _schema._table_info(columns, cursor.fetchone())
                self._remember_table(name, info)
                return self._table_info_cache[self._table_key(name)]

    def _search_info(self, name: str) -> _schema._TableInfo:
        info = self._table_info_cache.get(self._table_key(name))
        if info is None:
            with _db._get_connection(self.client) as connection:
                with connection.cursor() as cursor:
                    info = _schema._table_info(
                        _schema._vecdb_layout(cursor, name), None
                    )
            self._remember_table(name, info)
        return info

    def _refresh_layout(self, name: str) -> _schema._TableInfo:
        previous = self._table_info_cache.get(self._table_key(name))
        with _db._get_connection(self.client) as connection:
            with connection.cursor() as cursor:
                columns = _schema._vecdb_layout(cursor, name)
        _schema._table_info(columns, None)
        self._remember_table(
            name,
            _schema._TableInfo(
                frozenset(columns), previous.hybrid_index if previous else None
            ),
        )
        return self._table_info_cache[self._table_key(name)]

    @contextmanager
    def session(self) -> Iterator[NativeTransport]:
        with _db._get_connection(self.client) as connection:
            bound = copy(self)
            bound.client = connection
            # A pool checkout, not the store's initial checkout, owns this snapshot.
            bound.version = _schema.parse_version(connection.version)
            yield bound

    def columns(self, name: str) -> set[str]:
        with _db._get_connection(self.client) as connection:
            with connection.cursor() as cursor:
                return _schema._inspect_columns(cursor, name)

    def _statement(self, operation: str, params: dict) -> tuple[str, dict]:
        # Quoted package arguments preserve case and literal dots, just as SQL
        # identifiers do. Schema-qualified package operations remain unsupported.
        params = dict(params)
        for key in ("name", "table_name"):
            if key in params:
                name_error = _schema._package_name_error(params[key])
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
        with _db._get_connection(self.client) as connection:
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
                cursor.execute(statement, result=out, **binds)
                return as_dict(out.getvalue())

    @_with_native_session
    def vector_index_status(self, name: str) -> dict:
        with _db._get_connection(self.client) as connection:
            with connection.cursor() as cursor:
                try:
                    cursor.execute(
                        "SELECT SYS.DBMS_VECTOR_DATABASE.INDEX_BUILD_STATUS(:name) "
                        "FROM SYS.DUAL",
                        name=_schema.table_identifier(name)[0],
                    )
                except oracledb.DatabaseError as error:
                    if not _index_status_needs_catalog(error):
                        raise
                    found = self.describe_index(name)["details"]["indexes"]
                    return {
                        "index_status": "EXISTS"
                        if found
                        else "VECTOR INDEX IS NOT CREATED"
                    }
                return as_dict(cursor.fetchone()[0])

    @_with_native_session
    def describe_index(self, name: str, kind: str = "vector") -> dict:
        with _db._get_connection(self.client) as connection:
            with connection.cursor() as cursor:
                owner, binds = _schema._catalog_target(name)
                statement = index_status_query(owner, kind)
                cursor.execute(statement, **binds)
                return native_index_status(cursor.fetchall())

    @_with_native_session
    def migrate(self, name: str) -> None:
        quoted, _ = _schema.table_identifier(name)
        with _db._get_connection(self.client) as connection:
            self.version = _schema.parse_version(connection.version)
            if self.version < CURRENT_VERSION:
                raise ValueError("Layout migration requires Oracle Database 23.26.3+")
            if connection.transaction_in_progress:
                raise RuntimeError(
                    "VecDB migration requires a connection without pending work"
                )
            with connection.cursor() as cursor:

                def inspect() -> set[str]:
                    return _schema._inspect_columns(cursor, name)

                cached = self._table_info_cache.get(self._table_key(name))
                self._forget_table(name)
                for action in _schema._migration_actions(inspect(), quoted):
                    try:
                        cursor.execute(action)
                    except oracledb.DatabaseError as exc:
                        # Another process may have completed this exact DDL.
                        code = exc.args[0].code
                        if code not in {
                            904,
                            1430,
                        } or action in _schema._migration_actions(inspect(), quoted):
                            raise
                columns = inspect()
                if not _schema.CURRENT_COLUMNS <= columns:
                    raise RuntimeError(
                        "VecDB migration did not produce the required layout"
                    )
                self._remember_table(
                    name,
                    _schema._TableInfo(
                        frozenset(columns), cached.hybrid_index if cached else None
                    ),
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
        with _db._get_connection(self.client) as connection:
            with connection.cursor() as cursor:
                statement, binds = _schema._metadata_index_query(name, meta)
                cursor.execute(statement, **binds)
                found = cursor.fetchall()
                if len(found) > 1:
                    raise ValueError("Multiple metadata domain indexes found")
                existing = found[0] if found else None
                physical, ddl, restore = _schema._metadata_index_plan(
                    name, index, columns, existing, hybrid, build
                )
                if ddl is None:
                    self._remember_table(name, _schema._table_info(columns, existing))
                    return
                if connection.transaction_in_progress:
                    raise RuntimeError(
                        "Index DDL requires a connection without pending work"
                    )
                if not existing:
                    statement, binds = _schema._index_query(physical)
                    cursor.execute(statement, **binds)
                    if cursor.fetchone():
                        raise ValueError(
                            "Index name is already used by an incompatible index"
                        )
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
                            statement, binds = _schema._index_query(physical)
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
                    _schema._TableInfo(
                        frozenset(columns), physical if hybrid else None
                    ),
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

        quoted, _ = _schema.table_identifier(name)
        info = self._table_info_cache.get(self._table_key(name))
        if info is None:
            info = self._search_info(name)
        meta = info.metadata_column
        values = [
            (r["id"], array.array("f", r["dense_vector"]), r["metadata"])
            for r in records
        ]
        with _db._get_connection(self.client) as connection:
            # Match regular OracleVS SQL inserts: report row errors and commit
            # successful rows, including any caller work on this connection.
            with connection.cursor() as cursor:
                cursor.setinputsizes(
                    None, oracledb.DB_TYPE_VECTOR, oracledb.DB_TYPE_JSON
                )
                for attempt in range(2):
                    try:
                        cursor.executemany(
                            f"INSERT INTO {quoted} (ID, DENSE_VECTOR, {meta}) "
                            "VALUES (:1, :2, :3)",
                            values,
                            batcherrors=True,
                        )
                        break
                    except oracledb.DatabaseError as failure:
                        # Never replay a partially executed batch or row-level errors.
                        if (
                            attempt
                            or not _schema._insert_layout_error(failure, meta)
                            or cursor.rowcount != 0
                            or cursor.getbatcherrors()
                        ):
                            raise
                        self._forget_table(name)
                        refreshed = self._refresh_layout(name)
                        info = _schema._TableInfo(refreshed.columns, info.hybrid_index)
                        self._remember_table(name, info)
                        if info.metadata_column == meta:
                            raise
                        meta = info.metadata_column
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
        with _db._get_connection(self.client) as connection:
            info = self._table_info_cache.get(self._table_key(name))
            # VecDB conversion retains the table's metadata index name. The
            # caller's creation/retriever name need not be that physical name.
            if info is None:
                info = self.discover(name)
            elif info.hybrid_index is None:
                info = self.discover(name, set(info.columns))
                if info.hybrid_index is None:
                    # A different pool session or external DDL may have moved
                    # the index to the other metadata column since preparation.
                    refreshed = self._refresh_layout(name)
                    if refreshed.metadata_column != info.metadata_column:
                        info = self.discover(name, set(refreshed.columns))
            for attempt in range(2):
                index = info.hybrid_index
                if index is None:
                    raise ValueError(
                        "Create a hybrid vector index on the VecDB "
                        "metadata column first"
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
                    if attempt or not _schema._metadata_search_error(
                        error, info.metadata_column
                    ):
                        raise
                    self._forget_table(name)
                    info = self.discover(name)
            raise AssertionError("Unreachable metadata search retry")

    @_with_native_session
    def filter_search(self, name: str, filter: dict, request: dict) -> dict:
        with _db._get_connection(self.client) as connection:
            info = self._search_info(name)
            for attempt in range(2):
                sql, binds = _filter_query(name, info.metadata_column, request, filter)
                try:
                    with connection.cursor() as cursor:
                        cursor.execute(sql, **binds)
                        return _filter_response(cursor.fetchall(), request)
                except Exception as error:
                    if attempt or not _schema._metadata_search_error(
                        error, info.metadata_column
                    ):
                        raise
                    info = self._refresh_layout(name)
            raise AssertionError("Unreachable filter search retry")

    @_with_native_session
    def text_search(self, name: str, query: str, k: int, scores: bool = False) -> list:
        quoted, _ = _schema.table_identifier(name)
        with _db._get_connection(self.client) as connection:
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
                    if attempt or not _schema._metadata_search_error(
                        error, info.metadata_column
                    ):
                        raise
                    info = self._refresh_layout(name)
            raise AssertionError("Unreachable metadata search retry")


class AsyncNativeTransport(NativeTransport):
    """Same package contract using python-oracledb's async connections."""

    @asynccontextmanager
    async def asession(self) -> AsyncIterator[AsyncNativeTransport]:
        async with _db._aget_connection(self.client) as connection:
            bound = copy(self)
            bound.client = connection
            bound.version = _schema.parse_version(connection.version)
            yield bound

    @_with_async_native_session
    async def adiscover(
        self, name: str, columns: Optional[set[str]] = None
    ) -> _schema._TableInfo:
        async with _db._aget_connection(self.client) as connection:
            with connection.cursor() as cursor:
                if columns is None:
                    columns = await self.acolumns(name)
                meta = (
                    "CONTENT_METADATA" if "CONTENT_METADATA" in columns else "METADATA"
                )
                statement, binds = _schema._metadata_index_query(name, meta)
                await cursor.execute(statement, **binds)
                info = _schema._table_info(columns, await cursor.fetchone())
                self._remember_table(name, info)
                return self._table_info_cache[self._table_key(name)]

    async def _asearch_info(self, name: str) -> _schema._TableInfo:
        info = self._table_info_cache.get(self._table_key(name))
        if info is None:
            async with _db._aget_connection(self.client) as connection:
                with connection.cursor() as cursor:
                    info = _schema._table_info(
                        await _schema._avecdb_layout(cursor, name), None
                    )
            self._remember_table(name, info)
        return info

    async def _arefresh_layout(self, name: str) -> _schema._TableInfo:
        previous = self._table_info_cache.get(self._table_key(name))
        async with _db._aget_connection(self.client) as connection:
            with connection.cursor() as cursor:
                columns = await _schema._avecdb_layout(cursor, name)
        _schema._table_info(columns, None)
        self._remember_table(
            name,
            _schema._TableInfo(
                frozenset(columns), previous.hybrid_index if previous else None
            ),
        )
        return self._table_info_cache[self._table_key(name)]

    async def acolumns(self, name: str) -> set[str]:
        async with _db._aget_connection(self.client) as connection:
            with connection.cursor() as cursor:
                return await _schema._ainspect_columns(cursor, name)

    @_with_async_native_session
    async def acall(self, operation: str, **params: Any) -> dict:
        statement, binds = self._statement(operation, params)
        async with _db._aget_connection(self.client) as connection:
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
                await cursor.execute(statement, result=out, **binds)
                value = out.getvalue()
                if hasattr(value, "read"):
                    value = await value.read()
                return as_dict(value)

    @_with_async_native_session
    async def avector_index_status(self, name: str) -> dict:
        async with _db._aget_connection(self.client) as connection:
            with connection.cursor() as cursor:
                try:
                    await cursor.execute(
                        "SELECT SYS.DBMS_VECTOR_DATABASE.INDEX_BUILD_STATUS(:name) "
                        "FROM SYS.DUAL",
                        name=_schema.table_identifier(name)[0],
                    )
                except oracledb.DatabaseError as error:
                    if not _index_status_needs_catalog(error):
                        raise
                    found = (await self.adescribe_index(name))["details"]["indexes"]
                    return {
                        "index_status": "EXISTS"
                        if found
                        else "VECTOR INDEX IS NOT CREATED"
                    }
                value = (await cursor.fetchone())[0]
                if hasattr(value, "read"):
                    value = await value.read()
                return as_dict(value)

    @_with_async_native_session
    async def adescribe_index(self, name: str, kind: str = "vector") -> dict:
        async with _db._aget_connection(self.client) as connection:
            with connection.cursor() as cursor:
                owner, binds = _schema._catalog_target(name)
                statement = index_status_query(owner, kind)
                await cursor.execute(statement, **binds)
                return native_index_status(await cursor.fetchall())

    @_with_async_native_session
    async def amigrate(self, name: str) -> None:
        quoted, _ = _schema.table_identifier(name)
        async with _db._aget_connection(self.client) as connection:
            self.version = _schema.parse_version(connection.version)
            if self.version < CURRENT_VERSION:
                raise ValueError("Layout migration requires Oracle Database 23.26.3+")
            if connection.transaction_in_progress:
                raise RuntimeError(
                    "VecDB migration requires a connection without pending work"
                )
            with connection.cursor() as cursor:

                async def inspect() -> set[str]:
                    return await _schema._ainspect_columns(cursor, name)

                cached = self._table_info_cache.get(self._table_key(name))
                self._forget_table(name)
                for action in _schema._migration_actions(await inspect(), quoted):
                    try:
                        await cursor.execute(action)
                    except oracledb.DatabaseError as exc:
                        if exc.args[0].code not in {
                            904,
                            1430,
                        } or action in _schema._migration_actions(
                            await inspect(), quoted
                        ):
                            raise
                columns = await inspect()
                if not _schema.CURRENT_COLUMNS <= columns:
                    raise RuntimeError(
                        "VecDB migration did not produce the required layout"
                    )
                self._remember_table(
                    name,
                    _schema._TableInfo(
                        frozenset(columns), cached.hybrid_index if cached else None
                    ),
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
        async with _db._aget_connection(self.client) as connection:
            with connection.cursor() as cursor:
                statement, binds = _schema._metadata_index_query(name, meta)
                await cursor.execute(statement, **binds)
                found = await cursor.fetchall()
                if len(found) > 1:
                    raise ValueError("Multiple metadata domain indexes found")
                existing = found[0] if found else None
                physical, ddl, restore = _schema._metadata_index_plan(
                    name, index, columns, existing, hybrid, build
                )
                if ddl is None:
                    self._remember_table(name, _schema._table_info(columns, existing))
                    return
                if connection.transaction_in_progress:
                    raise RuntimeError(
                        "Index DDL requires a connection without pending work"
                    )
                if not existing:
                    statement, binds = _schema._index_query(physical)
                    await cursor.execute(statement, **binds)
                    if await cursor.fetchone():
                        raise ValueError(
                            "Index name is already used by an incompatible index"
                        )
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
                    _schema._TableInfo(
                        frozenset(columns), physical if hybrid else None
                    ),
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
            statement, binds = _schema._index_query(physical)
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

        quoted, _ = _schema.table_identifier(name)
        info = self._table_info_cache.get(self._table_key(name))
        if info is None:
            info = await self._asearch_info(name)
        meta = info.metadata_column
        values = [
            (r["id"], array.array("f", r["dense_vector"]), r["metadata"])
            for r in records
        ]
        async with _db._aget_connection(self.client) as connection:
            # Match regular OracleVS SQL inserts: report row errors and commit
            # successful rows, including any caller work on this connection.
            with connection.cursor() as cursor:
                cursor.setinputsizes(
                    None, oracledb.DB_TYPE_VECTOR, oracledb.DB_TYPE_JSON
                )
                for attempt in range(2):
                    try:
                        await cursor.executemany(
                            f"INSERT INTO {quoted} (ID, DENSE_VECTOR, {meta}) "
                            "VALUES (:1, :2, :3)",
                            values,
                            batcherrors=True,
                        )
                        break
                    except oracledb.DatabaseError as failure:
                        # Never replay a partially executed batch or row-level errors.
                        if (
                            attempt
                            or not _schema._insert_layout_error(failure, meta)
                            or cursor.rowcount != 0
                            or cursor.getbatcherrors()
                        ):
                            raise
                        self._forget_table(name)
                        refreshed = await self._arefresh_layout(name)
                        info = _schema._TableInfo(refreshed.columns, info.hybrid_index)
                        self._remember_table(name, info)
                        if info.metadata_column == meta:
                            raise
                        meta = info.metadata_column
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
        async with _db._aget_connection(self.client) as connection:
            info = self._table_info_cache.get(self._table_key(name))
            # VecDB conversion retains the table's metadata index name. The
            # caller's creation/retriever name need not be that physical name.
            if info is None:
                info = await self.adiscover(name)
            elif info.hybrid_index is None:
                info = await self.adiscover(name, set(info.columns))
                if info.hybrid_index is None:
                    refreshed = await self._arefresh_layout(name)
                    if refreshed.metadata_column != info.metadata_column:
                        info = await self.adiscover(name, set(refreshed.columns))
            for attempt in range(2):
                index = info.hybrid_index
                if index is None:
                    raise ValueError(
                        "Create a hybrid vector index on the VecDB "
                        "metadata column first"
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
                    if attempt or not _schema._metadata_search_error(
                        error, info.metadata_column
                    ):
                        raise
                    self._forget_table(name)
                    info = await self.adiscover(name)
            raise AssertionError("Unreachable metadata search retry")

    @_with_async_native_session
    async def afilter_search(self, name: str, filter: dict, request: dict) -> dict:
        async with _db._aget_connection(self.client) as connection:
            info = await self._asearch_info(name)
            for attempt in range(2):
                sql, binds = _filter_query(name, info.metadata_column, request, filter)
                try:
                    with connection.cursor() as cursor:
                        await cursor.execute(sql, **binds)
                        return _filter_response(await cursor.fetchall(), request)
                except Exception as error:
                    if attempt or not _schema._metadata_search_error(
                        error, info.metadata_column
                    ):
                        raise
                    info = await self._arefresh_layout(name)
            raise AssertionError("Unreachable filter search retry")

    @_with_async_native_session
    async def atext_search(
        self, name: str, query: str, k: int, scores: bool = False
    ) -> list:
        quoted, _ = _schema.table_identifier(name)
        async with _db._aget_connection(self.client) as connection:
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
                    if attempt or not _schema._metadata_search_error(
                        error, info.metadata_column
                    ):
                        raise
                    info = await self._arefresh_layout(name)
            raise AssertionError("Unreachable metadata search retry")
