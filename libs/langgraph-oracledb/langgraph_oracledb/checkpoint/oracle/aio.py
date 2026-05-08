from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import asynccontextmanager
from typing import Any

import oracledb
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
    get_checkpoint_metadata,
)
from langgraph.checkpoint.serde.base import SerializerProtocol

from langgraph_oracledb.checkpoint.oracle import _ainternal
from langgraph_oracledb.checkpoint.oracle.base import BaseOracleSaver

from .sync import _validate_conn_string

Conn = _ainternal.Conn  # For backward compatibility


class AsyncOracleSaver(BaseOracleSaver):
    """Asynchronous checkpointer that stores checkpoints in a Oracle database."""

    lock: asyncio.Lock

    def __init__(
        self,
        conn: _ainternal.Conn,
        serde: SerializerProtocol | None = None,
        json_size_threshold_mb: float = 1.0,
    ) -> None:
        super().__init__(serde=serde)
        self.conn = conn
        self.json_size_threshold_mb = json_size_threshold_mb
        self.loop = asyncio.get_running_loop()

    @classmethod
    @asynccontextmanager
    async def from_conn_string(
        cls,
        conn_string: str,
        *,
        pool_config: dict[str, Any] | None = None,
        serde: SerializerProtocol | None = None,
        json_size_threshold_mb: float = 1.0,
    ) -> AsyncIterator[AsyncOracleSaver]:
        """Create a new AsyncOracleSaver instance from a connection string."""
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
                yield cls(
                    conn=pool,
                    serde=serde,
                    json_size_threshold_mb=json_size_threshold_mb,
                )
            finally:
                await pool.close()
        else:
            conn = await oracledb.connect_async(user=user, password=password, dsn=dsn)
            try:
                yield cls(
                    conn=conn,
                    serde=serde,
                    json_size_threshold_mb=json_size_threshold_mb,
                )
            finally:
                await conn.close()

    async def setup(self) -> None:
        """Set up the checkpoint database asynchronously."""
        async with self._cursor() as cur:
            await cur.execute(self.MIGRATIONS[0])
            await cur.execute(
                "SELECT v FROM checkpoint_migrations ORDER BY v DESC FETCH FIRST 1 ROWS ONLY"
            )
            row = await cur.fetchone()
            if row is None:
                version = -1
            else:
                version = row[0]

            max_retries = 3
            base_delay = 1.0

            for v, migration in zip(
                range(version + 1, len(self.MIGRATIONS)),
                self.MIGRATIONS[version + 1 :],
            ):
                # Retry only the DDL step (e.g., CREATE INDEX ONLINE) on ORA-00054
                for attempt in range(max_retries):
                    try:
                        await cur.execute(migration)
                        break
                    except oracledb.Error as e:
                        err = e.args[0]
                        code = getattr(err, "code", None)
                        text = migration.upper()

                        # ORA-00054: resource busy/acquire lock failed -> retry with backoff
                        if code == 54:
                            if attempt == max_retries - 1:
                                raise
                            await asyncio.sleep(base_delay * (2**attempt))
                            continue

                        # If creating an index and it already exists, treat as success
                        if (
                            "CREATE INDEX" in text or "CREATE UNIQUE INDEX" in text
                        ) and code in (955, 1408):
                            # ORA-00955: name is already used by an existing object
                            # ORA-01408: such column list already indexed
                            break

                        # Otherwise, re-raise unexpected errors
                        raise

                await cur.execute(
                    "INSERT /*+ IGNORE_ROW_ON_DUPKEY_INDEX(checkpoint_migrations (v)) */ INTO checkpoint_migrations (v) VALUES (:1)",
                    [v],
                )

                await cur.connection.commit()

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """List checkpoints from the database asynchronously."""
        where, args = self._search_where(config, filter, before)
        query = self.SELECT_SQL + where + " ORDER BY checkpoint_id DESC"
        if limit:
            query += f" FETCH FIRST {limit} ROWS ONLY"

        async with self._cursor() as cur:
            cur.outputtypehandler = self.output_type_handler
            await cur.execute(query, args)
            values_fetch = await cur.fetchall()
            if not values_fetch:
                return

            # Convert Oracle rows to dictionary-like structure
            columns = [desc[0].lower() for desc in cur.description]
            values = [dict(zip(columns, row)) for row in values_fetch]
            # Oracle JSON columns return numbers as Decimal; coerce so downstream
            # LangGraph code (e.g. metadata["step"] in uuid6) sees int/float.
            for v in values:
                v["checkpoint"] = self._coerce_decimals(v["checkpoint"])
                v["metadata"] = self._coerce_decimals(v["metadata"])

            where_channels, args_channels = self._search_where_channels_pending_writes(
                config, use_checkpoint_id=False
            )
            await cur.execute(
                self.SELECT_CHANNEL_VALUES_SQL + where_channels, args_channels
            )
            res = await cur.fetchall()
            channel_values = self._get_channel_values(res, values)

            where_writes, args_writes = self._search_where_channels_pending_writes(
                config, use_checkpoint_id=True
            )
            await cur.execute(
                self.SELECT_PENDING_WRITES_SQL.format(where_clause=where_writes),
                args_writes,
            )
            res = await cur.fetchall()
            pending_writes = self._get_pending_writes(res, values)
            for i, v in enumerate(values):
                v["channel_values"] = channel_values[i]
                v["pending_writes"] = pending_writes[i]

            # migrate pending sends if necessary
            # pending_sends were removed on checkpoint version 4
            if to_migrate := [
                v
                for v in values
                if v["checkpoint"]
                and v["checkpoint"].get("v", 0) < 4
                and v["parent_checkpoint_id"]
            ]:
                checkpoint_id_binds_values = [
                    v["parent_checkpoint_id"] for v in to_migrate
                ]
                checkpoint_id_binds = [
                    f":cp_{i}" for i in range(len(checkpoint_id_binds_values))
                ]
                await cur.execute(
                    self.SELECT_PENDING_SENDS_SQL.format(
                        checkpoint_bind=",".join(checkpoint_id_binds)
                    ),
                    {
                        "thread_id": values[0]["thread_id"],
                        **{
                            _k[1:]: _v
                            for _k, _v in zip(
                                checkpoint_id_binds, checkpoint_id_binds_values
                            )
                        },
                    },
                )

                # group by the checkpoint_id
                pending_sends_by_checkpoint = defaultdict(list)
                for sends in await cur.fetchall():
                    checkpoint_id, type_str, blob = sends
                    blob_data = blob  # .read() if hasattr(blob, "read") else blob
                    pending_sends_by_checkpoint[checkpoint_id].append(
                        (type_str, blob_data)
                    )

                grouped_by_parent = defaultdict(list)
                for value in to_migrate:
                    grouped_by_parent[value["parent_checkpoint_id"]].append(value)

                for (
                    checkpoint_id,
                    pending_sends_data,
                ) in pending_sends_by_checkpoint.items():
                    for value in grouped_by_parent[checkpoint_id]:
                        if value.get("channel_values") is None:
                            value["channel_values"] = []
                        self._migrate_pending_sends(
                            pending_sends_data,
                            value["checkpoint"],
                            value["channel_values"],
                        )
            for value in values:
                yield await self._load_checkpoint_tuple(value)

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Get a checkpoint tuple from the database asynchronously."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = get_checkpoint_id(config)
        checkpoint_ns = self._encode_checkpoint_ns(
            config["configurable"].get("checkpoint_ns", "")
        )

        if checkpoint_id:
            args = {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
            where = "WHERE thread_id = :thread_id AND checkpoint_ns = :checkpoint_ns AND checkpoint_id = :checkpoint_id"
        else:
            args = {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns}
            where = "WHERE thread_id = :thread_id AND checkpoint_ns = :checkpoint_ns ORDER BY checkpoint_id DESC FETCH FIRST 1 ROWS ONLY"

        async with self._cursor() as cur:
            cur.outputtypehandler = self.output_type_handler
            await cur.execute(
                self.SELECT_SQL + where,
                args,
            )
            values_fetch = await cur.fetchone()
            if values_fetch is None:
                return None

            columns = [desc[0].lower() for desc in cur.description]
            value = dict(zip(columns, values_fetch))
            # See alist(): decimals from Oracle JSON columns break downstream.
            value["checkpoint"] = self._coerce_decimals(value["checkpoint"])
            value["metadata"] = self._coerce_decimals(value["metadata"])

            where_channel_values = (
                "WHERE thread_id = :thread_id AND checkpoint_ns = :checkpoint_ns"
            )
            await cur.execute(
                self.SELECT_CHANNEL_VALUES_SQL + where_channel_values,
                {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns},
            )
            res = await cur.fetchall()
            channel_values = self._get_channel_values(res, [value])
            assert len(channel_values) == 1

            where_pending_values = "WHERE thread_id = :thread_id AND checkpoint_ns = :checkpoint_ns AND checkpoint_id = :checkpoint_id"
            await cur.execute(
                self.SELECT_PENDING_WRITES_SQL.format(
                    where_clause=where_pending_values
                ),
                {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": value["checkpoint_id"],
                },
            )

            res = await cur.fetchall()
            pending_writes = self._get_pending_writes(res, [value])

            value["channel_values"] = channel_values[0]
            value["pending_writes"] = pending_writes[0]

            # migrate pending sends if necessary
            if value["checkpoint"]["v"] < 4 and value["parent_checkpoint_id"]:
                await cur.execute(
                    self.SELECT_PENDING_SENDS_SQL.format(checkpoint_bind=":cp_0"),
                    {"thread_id": thread_id, "cp_0": value["parent_checkpoint_id"]},
                )

                # group by the checkpoint_id
                pending_sends_by_checkpoint = []
                for sends in await cur.fetchall():
                    checkpoint_id, type_str, blob = sends
                    blob_data = blob  # .read() if hasattr(blob, "read") else blob
                    pending_sends_by_checkpoint.append((type_str, blob_data))

                if len(pending_sends_by_checkpoint) > 0:
                    if value["channel_values"] is None:
                        value["channel_values"] = []
                    self._migrate_pending_sends(
                        pending_sends_by_checkpoint,
                        value["checkpoint"],
                        value["channel_values"],
                    )

            return await self._load_checkpoint_tuple(value)

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint to the database asynchronously."""
        configurable = config["configurable"].copy()
        thread_id = configurable.pop("thread_id")
        checkpoint_ns = self._encode_checkpoint_ns(
            configurable.pop("checkpoint_ns", "")
        )
        checkpoint_id = configurable.pop("checkpoint_id", None)

        copy = checkpoint.copy()
        copy["channel_values"] = copy["channel_values"].copy()
        next_config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": self._decode_checkpoint_ns(checkpoint_ns),
                "checkpoint_id": checkpoint["id"],
            }
        }

        # inline primitive values in checkpoint table
        # others are stored in blobs table
        blob_values = {}
        for k, v in checkpoint["channel_values"].items():
            if self._should_use_blob(v, self.json_size_threshold_mb):
                blob_values[k] = copy["channel_values"].pop(k)

        async with self._cursor() as cur:
            if blob_versions := {
                k: v for k, v in new_versions.items() if k in blob_values
            }:
                blob_data = await asyncio.to_thread(
                    self._dump_blobs,
                    thread_id,
                    checkpoint_ns,
                    blob_values,
                    blob_versions,
                )

                cur.setinputsizes(blob=oracledb.DB_TYPE_BLOB)
                await cur.executemany(self.UPSERT_CHECKPOINT_BLOBS_SQL, blob_data)

            if "channel_versions" not in copy:
                copy["channel_versions"] = {}
            copy["channel_versions"].update(new_versions)

            cur.setinputsizes(
                checkpoint=oracledb.DB_TYPE_JSON, metadata=oracledb.DB_TYPE_JSON
            )
            await cur.execute(
                self.UPSERT_CHECKPOINTS_SQL,
                {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint["id"],
                    "parent_checkpoint_id": checkpoint_id,
                    "checkpoint": copy,
                    "metadata": get_checkpoint_metadata(config, metadata),
                },
            )
            await cur.connection.commit()
        return next_config

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Store intermediate writes linked to a checkpoint asynchronously."""
        query = (
            self.UPSERT_CHECKPOINT_WRITES_SQL
            if all(w[0] in WRITES_IDX_MAP for w in writes)
            else self.INSERT_CHECKPOINT_WRITES_SQL
        )
        params = await asyncio.to_thread(
            self._dump_writes,
            config["configurable"]["thread_id"],
            self._encode_checkpoint_ns(config["configurable"].get("checkpoint_ns", "")),
            config["configurable"]["checkpoint_id"],
            task_id,
            self._encode_task_path(task_path),
            writes,
        )
        async with self._cursor() as cur:
            # See sync.put_writes: without this hint oracledb maps b"" -> NULL
            # and the NOT NULL `blob` column rejects the row with ORA-01400.
            cur.setinputsizes(blob=oracledb.DB_TYPE_BLOB)
            await cur.executemany(query, params)
            await cur.connection.commit()

    async def adelete_thread(self, thread_id: str) -> None:
        """Delete all checkpoints and writes associated with a thread ID."""
        async with self._cursor() as cur:
            await cur.execute(
                "DELETE FROM checkpoints WHERE thread_id = :1",
                [str(thread_id)],
            )
            await cur.execute(
                "DELETE FROM checkpoint_blobs WHERE thread_id = :1",
                [str(thread_id)],
            )
            await cur.execute(
                "DELETE FROM checkpoint_writes WHERE thread_id = :1",
                [str(thread_id)],
            )
            await cur.connection.commit()

    @asynccontextmanager
    async def _cursor(self) -> AsyncIterator[oracledb.AsyncCursor]:
        """Create a database cursor as a context manager."""
        async with _ainternal.get_connection(self.conn) as conn:
            async with conn.cursor() as cur:
                yield cur

    async def _load_checkpoint_tuple(self, value: dict) -> CheckpointTuple:
        """
        Convert a database row into a CheckpointTuple object.
        """
        return CheckpointTuple(
            {
                "configurable": {
                    "thread_id": value["thread_id"],
                    "checkpoint_ns": self._decode_checkpoint_ns(value["checkpoint_ns"]),
                    "checkpoint_id": value["checkpoint_id"],
                }
            },
            {
                **value["checkpoint"],
                "channel_values": {
                    **(value["checkpoint"].get("channel_values") or {}),
                    **self._load_blobs(value["channel_values"]),
                },
            },
            value["metadata"],
            (
                {
                    "configurable": {
                        "thread_id": value["thread_id"],
                        "checkpoint_ns": self._decode_checkpoint_ns(
                            value["checkpoint_ns"]
                        ),
                        "checkpoint_id": value["parent_checkpoint_id"],
                    }
                }
                if value["parent_checkpoint_id"]
                else None
            ),
            await asyncio.to_thread(self._load_writes, value["pending_writes"]),
        )

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints from the database."""
        try:
            # check if we are in the main thread, only bg threads can block
            # we don't check in other methods to avoid the overhead
            if asyncio.get_running_loop() is self.loop:
                raise asyncio.InvalidStateError(
                    "Synchronous calls to AsyncOracleSaver are only allowed from a "
                    "different thread. From the main thread, use the async interface. "
                    "For example, use `checkpointer.alist(...)` or `await "
                    "graph.ainvoke(...)`."
                )
        except RuntimeError:
            pass
        aiter_ = self.alist(config, filter=filter, before=before, limit=limit)
        while True:
            try:
                yield asyncio.run_coroutine_threadsafe(
                    anext(aiter_),  # type: ignore[arg-type]  # noqa: F821
                    self.loop,
                ).result()
            except StopAsyncIteration:
                break

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Get a checkpoint tuple from the database."""
        try:
            # check if we are in the main thread, only bg threads can block
            # we don't check in other methods to avoid the overhead
            if asyncio.get_running_loop() is self.loop:
                raise asyncio.InvalidStateError(
                    "Synchronous calls to AsyncOracleSaver are only allowed from a "
                    "different thread. From the main thread, use the async interface. "
                    "For example, use `await checkpointer.aget_tuple(...)` or `await "
                    "graph.ainvoke(...)`."
                )
        except RuntimeError:
            pass
        return asyncio.run_coroutine_threadsafe(
            self.aget_tuple(config), self.loop
        ).result()

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint to the database."""
        return asyncio.run_coroutine_threadsafe(
            self.aput(config, checkpoint, metadata, new_versions), self.loop
        ).result()

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Store intermediate writes linked to a checkpoint."""
        return asyncio.run_coroutine_threadsafe(
            self.aput_writes(config, writes, task_id, task_path), self.loop
        ).result()

    def delete_thread(self, thread_id: str) -> None:
        """Delete all checkpoints and writes associated with a thread ID."""
        try:
            # check if we are in the main thread, only bg threads can block
            # we don't check in other methods to avoid the overhead
            if asyncio.get_running_loop() is self.loop:
                raise asyncio.InvalidStateError(
                    "Synchronous calls to AsyncOracleSaver are only allowed from a "
                    "different thread. From the main thread, use the async interface. "
                    "For example, use `await checkpointer.aget_tuple(...)` or `await "
                    "graph.ainvoke(...)`."
                )
        except RuntimeError:
            pass
        return asyncio.run_coroutine_threadsafe(
            self.adelete_thread(thread_id), self.loop
        ).result()


__all__ = ["AsyncOracleSaver", "Conn"]
