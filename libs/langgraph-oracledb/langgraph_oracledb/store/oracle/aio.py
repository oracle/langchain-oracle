from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Iterable, Sequence
from contextlib import asynccontextmanager
from types import TracebackType
from typing import Any, Union, cast

import oracledb
from langgraph.store.base import (
    GetOp,
    ListNamespacesOp,
    Op,
    PutOp,
    Result,
    SearchOp,
    TTLConfig,
)
from langgraph.store.base.batch import AsyncBatchedBaseStore

from langgraph_oracledb.checkpoint.oracle import _ainternal as _ainternal
from langgraph_oracledb.store.oracle.base import (
    BaseOracleStore,
    OracleIndexConfig,
    PoolConfig,
    _decode_ns_bytes,
    _detect_dimensions,
    _ensure_index_config,
    _generate_suffix,
    _group_ops,
    _namespace_to_text,
    _normalize_existing_index_params,
    _row_to_item,
    _row_to_search_item,
    _should_ignore_ttl_refresh_error,
    _validate_table_suffix,
)

from ...checkpoint.oracle.sync import _validate_conn_string

logger = logging.getLogger(__name__)

# Type alias for Oracle connections (single connection or pool)
OracleConn = Union[oracledb.AsyncConnection, oracledb.AsyncConnectionPool]


class AsyncOracleStore(
    AsyncBatchedBaseStore, BaseOracleStore[oracledb.AsyncConnection]
):
    """Asynchronous Oracle-backed store with optional vector search using Oracle AI Vector Search.

    !!! example \"Examples\"
        Basic setup and usage:
        ```python
        from langgraph_oracledb.store.oracle import AsyncOracleStore

        # For local Oracle database:
        conn_string = ...

        # For Oracle Cloud with proper DSN:
        # conn_string = ...

        async with AsyncOracleStore.from_conn_string(conn_string) as store:
            await store.setup()  # Run migrations. Done once

            # Store and retrieve data
            await store.aput((\"users\", \"123\"), \"prefs\", {\"theme\": \"dark\"})
            item = await store.aget((\"users\", \"123\"), \"prefs\")
        ```

        Vector search using LangChain embeddings:
        ```python
        from langchain.embeddings import init_embeddings
        from langgraph_oracledb.store.oracle import AsyncOracleStore

        # Connection string format - use appropriate format for your setup
        conn_string = ...

        async with AsyncOracleStore.from_conn_string(
            conn_string,
            index={
                \"dims\": 1536,
                \"embed\": init_embeddings(\"openai:text-embedding-3-small\"),
                \"fields\": [\"text\"],  # specify which fields to embed. Default is the whole serialized value
                \"index_type\": {
                    \"type\": \"hnsw\",
                    \"neighbors\": 16,
                    \"efconstruction\": 200,
                    \"distance_metric\": \"COSINE\"
                }
            }
        ) as store:
            await store.setup()  # Run migrations. Done once

            # Store documents
            await store.aput((\"docs\",), \"doc1\", {\"text\": \"Python tutorial\"})
            await store.aput((\"docs\",), \"doc2\", {\"text\": \"TypeScript guide\"})
            await store.aput((\"docs\",), \"doc3\", {\"text\": \"Other guide\"}, index=False)  # don't index

            # Search by similarity
            results = await store.asearch((\"docs\",), query=\"programming guides\", limit=2)
        ```

    Warning:
        Make sure to:
        1. Call `setup()` before first use to create necessary tables and indexes
    """

    def __init__(
        self,
        conn: OracleConn,
        *,
        index: OracleIndexConfig | None = None,
        ttl: TTLConfig | None = None,
        table_suffix: str | None = None,
    ) -> None:
        super().__init__()
        self.conn = conn
        self.index_config = index

        if self.index_config:
            self.embeddings, self.index_config = _ensure_index_config(self.index_config)
        else:
            self.embeddings = None

        if table_suffix is not None:
            _validate_table_suffix(table_suffix)
        self.table_suffix = table_suffix  # Will be None for auto-generated suffixes
        self._needs_validation = bool(
            table_suffix
        )  # Only validate user-provided suffixes
        self.detected_dims = None

        # Defer table names setup to setup() method for consistent initialization
        self.table_names = {}

        self.ttl_config = ttl
        self._ttl_sweeper_task: asyncio.Task[None] | None = None
        self._ttl_stop_event = asyncio.Event()

    @asynccontextmanager
    async def _cursor(self) -> AsyncIterator[oracledb.AsyncCursor]:
        """Create a database cursor as a context manager."""
        async with _ainternal.get_connection(self.conn) as conn:
            async with conn.cursor() as cur:
                yield cur

    async def _validate_configuration(
        self,
        suffix: str,
        new_config: OracleIndexConfig,
    ) -> None:
        """Validate that new configuration matches existing tables."""
        async with self._cursor() as cur:
            # Check if configuration exists
            try:
                await cur.execute(
                    """
                    SELECT detected_dims, distance_type, index_params 
                    FROM store_configs 
                    WHERE table_suffix = :1
                """,
                    (suffix,),
                )

                row = await cur.fetchone()
            except oracledb.DatabaseError as e:
                if e.args[0].code == 942:
                    logger.info("store_configs table does not exists")
                    return
                else:
                    raise

        detected_dims = _detect_dimensions(new_config)
        if row:
            existing_dims, existing_distance, existing_params = row

            # Validate dimensions
            if detected_dims != existing_dims:
                raise ValueError(
                    f"Dimension mismatch for suffix '{suffix}':\n"
                    f"  Existing: {existing_dims} dimensions\n"
                    f"  Detected: {detected_dims} dimensions\n"
                    "Different embedding dimensions require different table sets. "
                    "Use a different table_suffix or drop existing tables."
                )

            # Validate distance type
            index_type_config = new_config.get("index_type", {})
            new_distance = index_type_config.get("distance_metric", "cosine").upper()
            if new_distance != existing_distance:
                raise ValueError(
                    f"Distance type mismatch for suffix '{suffix}':\n"
                    f"  Existing: {existing_distance}\n"
                    f"  Provided: {new_distance}\n"
                    "Changing distance metrics requires new tables."
                )

            # Validate index parameters
            new_params = json.dumps(
                new_config.get("index_type", {"type": "hnsw"}), sort_keys=True
            )
            existing_params_dict = _normalize_existing_index_params(existing_params)

            accuracy = existing_params_dict.pop("accuracy", None)
            if accuracy != new_config.get("accuracy", None):
                raise ValueError(
                    f"Index accuracy type mismatch for suffix '{suffix}':\n"
                    f"  Existing: {accuracy}\n"
                    f"  Provided: {new_config.get('accuracy', None)}\n"
                    "Changing accuracy requires new tables."
                )

            existing_params_normalized = json.dumps(
                existing_params_dict, sort_keys=True
            )

            if new_params != existing_params_normalized:
                raise ValueError(
                    f"Index parameter mismatch for suffix '{suffix}':\n"
                    f"  Existing: {existing_params_dict}\n"
                    f"  Provided: {new_params}\n"
                    "Changing index parameters requires new tables."
                )

    async def _register_configuration(
        self, suffix: str, config: OracleIndexConfig
    ) -> None:
        """Register a new configuration in the store_configs table."""
        async with self._cursor() as cur:
            # First ensure store_configs table exists by running setup
            try:
                await cur.execute("SELECT 1 FROM store_configs WHERE ROWNUM = 1")
            except oracledb.DatabaseError:
                raise

            detected_dims = _detect_dimensions(config)

            # Insert new configuration
            index_type_config = config.get("index_type", {})
            distance_type = index_type_config.get("distance_metric", "cosine").upper()
            dict_params = config.get("index_type", {"type": "hnsw"})
            dict_params["accuracy"] = config.get("accuracy", None)
            index_params = json.dumps(dict_params, sort_keys=True)
            dict_params.pop("accuracy")
            fields = config.get("fields", ["$"])
            embed_fields = ",".join(fields) if fields else "$"

            await cur.execute(
                """
                INSERT /*+ IGNORE_ROW_ON_DUPKEY_INDEX(store_configs (table_suffix)) */ INTO store_configs 
                (table_suffix, detected_dims, distance_type, index_params, embed_fields, created_at, last_used)
                VALUES (:1, :2, :3, :4, :5, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """,
                (suffix, detected_dims, distance_type, index_params, embed_fields),
            )

            await cur.connection.commit()
        logger.info(
            f"Registered new configuration for table suffix '{suffix}' with {detected_dims} dimensions"
        )

    @classmethod
    @asynccontextmanager
    async def from_conn_string(
        cls,
        conn_string: str,
        *,
        pool_config: PoolConfig | None = None,
        index: OracleIndexConfig | None = None,
        ttl: TTLConfig | None = None,
        table_suffix: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[AsyncOracleStore]:
        """Create a new AsyncOracleStore instance from a connection string.

        Args:
            conn_string: Oracle connection string DSN
            pool_config: Configuration for the connection pool.
                If provided, will create a connection pool and use it instead of a single connection.
            index: The embedding config for vector search
            ttl: TTL configuration for automatic expiration
            **kwargs: Additional connection parameters

        Returns:
            AsyncOracleStore: A new AsyncOracleStore instance.
        """
        # Parse the connection string
        user, password, dsn = _validate_conn_string(conn_string)

        if pool_config is not None:
            # Create connection pool
            pc = pool_config.copy()
            pool = oracledb.create_pool_async(
                user=user,
                password=password,
                dsn=dsn,
                min=pc.get("min_size", 1),
                max=pc.get("max_size", 10),
            )

            try:
                # Set session parameters for all connections in the pool
                # Yield the saver with the pool itself, not a connection from the pool
                store = cls(conn=pool, index=index, ttl=ttl, table_suffix=table_suffix)
                yield store
            finally:
                await pool.close()
        else:
            conn = await oracledb.connect_async(user=user, password=password, dsn=dsn)
            try:
                store = cls(conn=conn, index=index, ttl=ttl, table_suffix=table_suffix)
                yield store
            finally:
                await conn.close()

    async def setup(self) -> None:
        """Set up the store database asynchronously.

        This method creates the necessary tables in the Oracle database if they don't
        already exist and runs database migrations. It MUST be called directly by the user
        the first time the store is used.
        """

        async def _get_version(cur: oracledb.AsyncCursor, table: str) -> int:
            await cur.execute(
                f"CREATE TABLE IF NOT EXISTS {table} (v INTEGER PRIMARY KEY)"
            )
            await cur.connection.commit()

            await cur.execute(
                f"SELECT v FROM {table} ORDER BY v DESC FETCH FIRST 1 ROWS ONLY"
            )
            row = await cur.fetchone()
            if row is None:
                version = -1
            else:
                version = row[0]

            return version

        # Generate table suffix if not already set - must happen before any table operations
        if not self.table_suffix:
            if self.index_config and "embed" in self.index_config:
                self.table_suffix = _generate_suffix(self.index_config)
            else:
                # Non-vector store
                self.table_suffix = "novec"

            # Update validation flag for auto-generated suffixes
            self._needs_validation = False

        # Set table names now that we have the suffix
        self.table_names = {
            "store": f"store_{self.table_suffix}",
            "store_vectors": f"store_vectors_{self.table_suffix}",
            "store_migrations": f"store_migrations_{self.table_suffix}",
            "vector_migrations": f"vector_migrations_{self.table_suffix}",
        }
        if self.index_config:
            if self._needs_validation:
                await self._validate_configuration(self.table_suffix, self.index_config)

        async with self._cursor() as cur:
            version = await _get_version(
                cur, table=self.table_names["store_migrations"]
            )

            for v, sql in enumerate(self.MIGRATIONS[version + 1 :], start=version + 1):
                try:
                    sql_formatted = sql.format(**self.table_names)
                    await cur.execute(sql_formatted)
                    await cur.execute(
                        f"INSERT /*+ IGNORE_ROW_ON_DUPKEY_INDEX({self.table_names['store_migrations']} (v)) */ INTO {self.table_names['store_migrations']} (v) VALUES (:1)",
                        (v,),
                    )
                    await cur.connection.commit()
                except Exception as e:
                    logger.error(
                        f"Failed to apply migration {v}.\nSql={sql}\nError={e}"
                    )
                    raise
            # Only run vector migrations if index_config is provided
            if self.index_config:
                vector_migration_table_name = self.table_names["vector_migrations"]
                version = await _get_version(cur, table=vector_migration_table_name)

                for v, migration in enumerate(
                    self.VECTOR_MIGRATIONS[version + 1 :], start=version + 1
                ):
                    if migration.condition and not migration.condition(self):
                        continue
                    sql = migration.sql
                    if migration.params:
                        params = {
                            k: v(self) if v is not None and callable(v) else v
                            for k, v in migration.params.items()
                        }
                        sql = sql % params

                    await cur.execute(sql)
                    await cur.execute(
                        f"INSERT /*+ IGNORE_ROW_ON_DUPKEY_INDEX({vector_migration_table_name} (v)) */ INTO {vector_migration_table_name} (v) VALUES (:1)",
                        (v,),
                    )
                    await cur.connection.commit()

                try:
                    await self._register_configuration(
                        self.table_suffix, self.index_config
                    )
                except oracledb.IntegrityError as e:
                    # Configuration already exists (unique constraint violation), which is fine for deterministic suffixes
                    logger.debug(
                        f"Configuration for suffix '{self.table_suffix}' already exists: {e}"
                    )
                except oracledb.DatabaseError as e:
                    # Unexpected database error during configuration registration
                    logger.error(
                        f"Database error during configuration registration for suffix '{self.table_suffix}': {e}"
                    )
                    raise RuntimeError(f"Failed to register configuration: {e}") from e
                except Exception as e:
                    # Unexpected non-database error
                    logger.error(
                        f"Unexpected error during configuration registration: {e}"
                    )
                    raise

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        """Execute a batch of operations."""
        grouped_ops, num_ops = _group_ops(ops)
        results: list[Result] = [None] * num_ops

        prepared_statements = self._prepare_statements(grouped_ops)

        async with self._cursor() as cur:
            if GetOp in grouped_ops:
                await self._batch_get_ops(prepared_statements["get"], results, cur)

            if SearchOp in grouped_ops:
                await self._batch_search_ops(
                    prepared_statements["search"],
                    cast(Sequence[tuple[int, SearchOp]], grouped_ops[SearchOp]),
                    results,
                    cur,
                )

            if ListNamespacesOp in grouped_ops:
                await self._batch_list_namespaces_ops(
                    prepared_statements["list"],
                    cast(
                        Sequence[tuple[int, ListNamespacesOp]],
                        grouped_ops[ListNamespacesOp],
                    ),
                    results,
                    cur,
                )
            if PutOp in grouped_ops:
                await self._batch_put_ops(prepared_statements["put"], cur)

            await cur.connection.commit()

        return results

    async def _batch_get_ops(
        self,
        prepared_statements,
        results: list[Result],
        cur: oracledb.AsyncCursor,
    ) -> None:
        for query, params, namespace, items in prepared_statements:
            await cur.execute(query, params)

            if "UPDATE" in query:
                await cur.connection.commit()
                continue

            rows = await cur.fetchall()
            columns = [col.name for col in cur.description]
            rows = [
                {columns[i].lower(): row[i] for i in range(len(row))} for row in rows
            ]

            key_to_row = {row["key"]: row for row in rows}
            for idx, key in items:
                row = key_to_row.get(key)
                if row:
                    results[idx] = _row_to_item(namespace, row)
                else:
                    results[idx] = None

    async def _batch_put_ops(
        self,
        prepared_statements,
        cur: oracledb.AsyncCursor,
    ) -> None:
        """Execute batch put operations."""
        queries, _ = prepared_statements

        for query, params in queries:
            await cur.execute(query, params)

        await cur.connection.commit()

    async def _batch_search_ops(
        self,
        prepared_statements,
        search_ops: Sequence[tuple[int, SearchOp]],
        results: list[Result],
        cur: oracledb.AsyncCursor,
    ) -> bool:
        """Execute batch search operations."""
        queries, _ = prepared_statements

        for (idx, op), (query, params) in zip(search_ops, queries):
            await cur.execute(query, params)
            rows = await cur.fetchall()

            col_names = [desc[0].lower() for desc in cur.description]
            dict_rows = [dict(zip(col_names, row)) for row in rows]

            items = []
            for row in dict_rows:
                namespace = _decode_ns_bytes(row["prefix"])
                items.append(_row_to_search_item(namespace, row))

            if op.refresh_ttl and items:
                for item in items:
                    ttl_update_query = f"""
                        UPDATE {self.table_names["store"]} 
                        SET expires_at = CURRENT_TIMESTAMP + NUMTODSINTERVAL(ttl_minutes, 'MINUTE')
                        WHERE prefix = :prefix AND key = :key AND ttl_minutes IS NOT NULL
                    """
                    try:
                        await cur.execute(
                            ttl_update_query,
                            {
                                "prefix": _namespace_to_text(item.namespace),
                                "key": item.key,
                            },
                        )
                    except oracledb.DatabaseError as e:
                        if _should_ignore_ttl_refresh_error(
                            e, item.namespace, item.key
                        ):
                            continue
                        raise

            results[idx] = items

    async def _batch_list_namespaces_ops(
        self,
        prepared_statements,
        list_ops: Sequence[tuple[int, ListNamespacesOp]],
        results: list[Result],
        cur: oracledb.AsyncCursor,
    ) -> None:
        for (query, params), (idx, _) in zip(
            prepared_statements, list_ops, strict=False
        ):
            await cur.execute(query, params)
            rows = await cur.fetchall()

            col_names = [desc[0].lower() for desc in cur.description]
            dict_rows = [dict(zip(col_names, row)) for row in rows]

            results[idx] = [
                _decode_ns_bytes(row["truncated_prefix"]) for row in dict_rows
            ]

    async def sweep_ttl(self) -> int:
        """Delete expired store items based on TTL."""
        async with self._cursor() as cur:
            await cur.execute(
                f"""
                DELETE FROM {self.table_names["store"]}
                WHERE expires_at IS NOT NULL AND expires_at < CURRENT_TIMESTAMP
                """
            )
            deleted_count = cur.rowcount
            await cur.connection.commit()
            return deleted_count

    async def start_ttl_sweeper(
        self, sweep_interval_minutes: int | None = None
    ) -> asyncio.Task[None]:
        """Periodically delete expired store items based on TTL."""
        if not self.ttl_config:
            return asyncio.create_task(asyncio.sleep(0))

        if self._ttl_sweeper_task is not None and not self._ttl_sweeper_task.done():
            return self._ttl_sweeper_task

        self._ttl_stop_event.clear()

        interval = float(
            sweep_interval_minutes or self.ttl_config.get("sweep_interval_minutes") or 5
        )
        logger.info(f"Starting store TTL sweeper with interval {interval} minutes")

        async def _sweep_loop() -> None:
            while not self._ttl_stop_event.is_set():
                try:
                    try:
                        await asyncio.wait_for(
                            self._ttl_stop_event.wait(),
                            timeout=interval * 60,
                        )
                        break
                    except asyncio.TimeoutError:
                        pass

                    expired_items = await self.sweep_ttl()
                    if expired_items > 0:
                        logger.info(f"Store swept {expired_items} expired items")
                except asyncio.CancelledError:
                    break
                except Exception as exc:
                    logger.exception("Store TTL sweep iteration failed", exc_info=exc)

        task = asyncio.create_task(_sweep_loop())
        task.set_name("ttl_sweeper")
        self._ttl_sweeper_task = task
        return task

    async def stop_ttl_sweeper(self, timeout: float | None = None) -> bool:
        """Stop the TTL sweeper task if it's running."""
        if self._ttl_sweeper_task is None or self._ttl_sweeper_task.done():
            return True

        logger.info("Stopping TTL sweeper task")
        self._ttl_stop_event.set()

        if timeout is not None:
            try:
                await asyncio.wait_for(self._ttl_sweeper_task, timeout=timeout)
                success = True
            except asyncio.TimeoutError:
                success = False
        else:
            await self._ttl_sweeper_task
            success = True

        if success:
            self._ttl_sweeper_task = None
            logger.info("TTL sweeper task stopped")
        else:
            logger.warning("Timed out waiting for TTL sweeper task to stop")

        return success

    async def ateardown(self) -> None:
        """Clean up all tables and configurations associated with this store."""
        # Stop TTL sweeper if it's running
        if hasattr(self, "_ttl_sweeper_task") and self._ttl_sweeper_task:
            await self.stop_ttl_sweeper(timeout=1.0)

        async with self._cursor() as cur:
            try:
                # Delete from store_configs table
                if hasattr(self, "table_suffix") and self.table_suffix:
                    await cur.execute(
                        """
                        DELETE FROM store_configs 
                        WHERE table_suffix = :1
                    """,
                        (self.table_suffix,),
                    )
                    await cur.connection.commit()
                    logger.info(
                        f"Removed configuration for table suffix '{self.table_suffix}'"
                    )
            except oracledb.DatabaseError as e:
                # store_configs table might not exist
                logger.debug(f"Could not delete from store_configs: {e}")

            # Drop indexes first, then tables in reverse order of dependencies
            # This ensures vector indexes are properly cleaned up before table drops
            if hasattr(self, "table_names"):
                # Drop vector indexes explicitly
                if "store_vectors" in self.table_names:
                    try:
                        # Query for vector indexes on the table
                        await cur.execute(
                            """
                            SELECT index_name FROM user_indexes 
                            WHERE table_name = :1 AND index_type = 'VECTOR'
                            """,
                            (self.table_names["store_vectors"].upper(),),
                        )
                        vector_indexes = await cur.fetchall()
                        for (index_name,) in vector_indexes:
                            try:
                                await cur.execute(f"DROP INDEX {index_name}")
                                await cur.connection.commit()
                                logger.info(f"Dropped vector index {index_name}")
                            except oracledb.DatabaseError as e:
                                logger.debug(f"Could not drop index {index_name}: {e}")
                    except oracledb.DatabaseError as e:
                        logger.debug(f"Could not query for vector indexes: {e}")

                # Drop tables in proper order
                tables_to_drop = []
                # Order matters: drop dependent tables first
                if "store_vectors" in self.table_names:
                    tables_to_drop.append(self.table_names["store_vectors"])
                if "vector_migrations" in self.table_names:
                    tables_to_drop.append(self.table_names["vector_migrations"])
                if "store" in self.table_names:
                    tables_to_drop.append(self.table_names["store"])
                if "store_migrations" in self.table_names:
                    tables_to_drop.append(self.table_names["store_migrations"])

                for table in tables_to_drop:
                    try:
                        await cur.execute(
                            f"DROP TABLE {table} CASCADE CONSTRAINTS PURGE"
                        )
                        await cur.connection.commit()
                        logger.info(f"Dropped table {table}")
                    except oracledb.DatabaseError as e:
                        # Table might not exist, which is fine
                        logger.debug(f"Could not drop table {table}: {e}")

    async def __aenter__(self) -> AsyncOracleStore:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        # Ensure the TTL sweeper task is stopped when exiting the context
        if hasattr(self, "_ttl_sweeper_task") and self._ttl_sweeper_task is not None:
            # Set the event to signal the task to stop
            self._ttl_stop_event.set()
