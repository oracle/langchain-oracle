from __future__ import annotations

import threading
import time
from collections import defaultdict
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
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

from langgraph_oracledb.checkpoint.oracle import _internal
from langgraph_oracledb.checkpoint.oracle._lob import with_blob_lobs
from langgraph_oracledb.checkpoint.oracle.base import BaseOracleSaver

Conn = _internal.Conn  # For backward compatibility


def _validate_conn_string(conn_string: str):
    parts = conn_string.split("@")
    if len(parts) != 2:
        raise ValueError(
            "Invalid Oracle connection string format. Expected 'user/password@host:port/service_name'"
        )

    user_pass, dsn = parts
    user_parts = user_pass.split("/")
    if len(user_parts) != 2:
        raise ValueError(
            "Invalid Oracle connection string format. Expected 'user/password@host:port/service_name'"
        )

    user, password = user_parts

    return user, password, dsn


class OracleSaver(BaseOracleSaver):
    """Checkpointer that stores checkpoints in a Oracle database."""

    lock: threading.Lock

    def __init__(
        self,
        conn: _internal.Conn,
        serde: SerializerProtocol | None = None,
        json_size_threshold_mb: float = 1.0,
    ) -> None:
        super().__init__(serde=serde)

        self.conn = conn
        self.json_size_threshold_mb = json_size_threshold_mb

    @classmethod
    @contextmanager
    def from_conn_string(
        cls,
        conn_string: str,
        *,
        pool_config: dict[str, Any] | None = None,
        serde: SerializerProtocol | None = None,
        json_size_threshold_mb: float = 1.0,
    ) -> Iterator[OracleSaver]:
        """Create a new OracleSaver instance from a connection string.

        Args:
            conn_string: The Oracle connection string.
            pool_config: Optional pool configuration with min_size and max_size.
            serde: Optional serializer for checkpoint data.
            json_size_threshold_mb: Size threshold in MB for storing data in JSON vs BLOB (default 5MB).

        Returns:
            OracleSaver: A new OracleSaver instance.
        """
        # Parse the connection string
        user, password, dsn = _validate_conn_string(conn_string)

        if pool_config is not None:
            # Create connection pool
            pc = pool_config.copy()
            pool = oracledb.create_pool(
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
                pool.close()
        else:
            with oracledb.connect(user=user, password=password, dsn=dsn) as conn:
                yield cls(
                    conn=conn,
                    serde=serde,
                    json_size_threshold_mb=json_size_threshold_mb,
                )

    def setup(self) -> None:
        """Set up the checkpoint database."""
        with self._cursor() as cur:
            cur.execute(self.MIGRATIONS[0])
            cur.execute(
                "SELECT v FROM checkpoint_migrations ORDER BY v DESC FETCH FIRST 1 ROWS ONLY"
            )
            row = cur.fetchone()
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
                for attempt in range(max_retries):
                    try:
                        cur.execute(migration)
                        break
                    except oracledb.Error as e:
                        err = e.args[0]
                        code = getattr(err, "code", None)
                        text = migration.upper()

                        # ORA-00054: resource busy/acquire lock failed -> retry with backoff
                        if code == 54:
                            if attempt == max_retries - 1:
                                raise
                            time.sleep(base_delay * (2**attempt))
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

                cur.execute(
                    "INSERT /*+ IGNORE_ROW_ON_DUPKEY_INDEX(checkpoint_migrations (v)) */ INTO checkpoint_migrations (v) VALUES (:1)",
                    [v],
                )

                cur.connection.commit()

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints from the database."""
        where, args = self._search_where(config, filter, before)
        query = self.SELECT_SQL + where + " ORDER BY checkpoint_id DESC"
        if limit:
            query += f" FETCH FIRST {limit} ROWS ONLY"

        with self._cursor() as cur:
            cur.outputtypehandler = self.output_type_handler
            cur.execute(query, args)
            values_fetch = cur.fetchall()
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
            cur.execute(self.SELECT_CHANNEL_VALUES_SQL + where_channels, args_channels)
            res = cur.fetchall()
            channel_values = self._get_channel_values(res, values)

            where_writes, args_writes = self._search_where_channels_pending_writes(
                config, use_checkpoint_id=True
            )
            cur.execute(
                self.SELECT_PENDING_WRITES_SQL.format(where_clause=where_writes),
                args_writes,
            )
            res = cur.fetchall()
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
                cur.execute(
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
                for sends in cur:
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
                yield self._load_checkpoint_tuple(value)

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Get a checkpoint tuple from the database."""  # noqa
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

        with self._cursor() as cur:
            cur.outputtypehandler = self.output_type_handler
            cur.execute(
                self.SELECT_SQL + where,
                args,
            )
            values_fetch = cur.fetchone()
            if values_fetch is None:
                return None

            columns = [desc[0].lower() for desc in cur.description]
            value = dict(zip(columns, values_fetch))
            # See list(): decimals from Oracle JSON columns break downstream.
            value["checkpoint"] = self._coerce_decimals(value["checkpoint"])
            value["metadata"] = self._coerce_decimals(value["metadata"])

            where_channel_values = (
                "WHERE thread_id = :thread_id AND checkpoint_ns = :checkpoint_ns"
            )
            cur.execute(
                self.SELECT_CHANNEL_VALUES_SQL + where_channel_values,
                {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns},
            )
            res = cur.fetchall()
            channel_values = self._get_channel_values(res, [value])

            where_pending_values = "WHERE thread_id = :thread_id AND checkpoint_ns = :checkpoint_ns AND checkpoint_id = :checkpoint_id"
            cur.execute(
                self.SELECT_PENDING_WRITES_SQL.format(
                    where_clause=where_pending_values
                ),
                {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": value["checkpoint_id"],
                },
            )
            res = cur.fetchall()
            pending_writes = self._get_pending_writes(res, [value])

            value["channel_values"] = channel_values[0]
            value["pending_writes"] = pending_writes[0]

            # migrate pending sends if necessary
            if value["checkpoint"]["v"] < 4 and value["parent_checkpoint_id"]:
                cur.execute(
                    self.SELECT_PENDING_SENDS_SQL.format(checkpoint_bind=":cp_0"),
                    {"thread_id": thread_id, "cp_0": value["parent_checkpoint_id"]},
                )

                # group by the checkpoint_id
                pending_sends_by_checkpoint = []
                for sends in cur:
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

            return self._load_checkpoint_tuple(value)

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint to the database."""
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

        with self._cursor() as cur:
            if blob_versions := {
                k: v for k, v in new_versions.items() if k in blob_values
            }:
                blob_data = self._dump_blobs(
                    thread_id,
                    checkpoint_ns,
                    blob_values,
                    blob_versions,
                )
                cur.executemany(
                    self.UPSERT_CHECKPOINT_BLOBS_SQL,
                    with_blob_lobs(cur.connection, blob_data),
                )

            if "channel_versions" not in copy:
                copy["channel_versions"] = {}
            copy["channel_versions"].update(new_versions)

            cur.setinputsizes(
                checkpoint=oracledb.DB_TYPE_JSON, metadata=oracledb.DB_TYPE_JSON
            )
            cur.execute(
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
            cur.connection.commit()
        return next_config

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Store intermediate writes linked to a checkpoint."""
        query = (
            self.UPSERT_CHECKPOINT_WRITES_SQL
            if all(w[0] in WRITES_IDX_MAP for w in writes)
            else self.INSERT_CHECKPOINT_WRITES_SQL
        )
        params = self._dump_writes(
            config["configurable"]["thread_id"],
            self._encode_checkpoint_ns(config["configurable"].get("checkpoint_ns", "")),
            config["configurable"]["checkpoint_id"],
            task_id,
            self._encode_task_path(task_path),
            writes,
        )
        with self._cursor() as cur:
            cur.executemany(query, with_blob_lobs(cur.connection, params))
            cur.connection.commit()

    def delete_thread(self, thread_id: str) -> None:
        """Delete all checkpoints and writes associated with a thread ID."""
        with self._cursor() as cur:
            cur.execute(
                "DELETE FROM checkpoints WHERE thread_id = :1",
                [str(thread_id)],
            )
            cur.execute(
                "DELETE FROM checkpoint_blobs WHERE thread_id = :1",
                [str(thread_id)],
            )
            cur.execute(
                "DELETE FROM checkpoint_writes WHERE thread_id = :1",
                [str(thread_id)],
            )
            cur.connection.commit()

    @contextmanager
    def _cursor(self) -> Iterator[oracledb.Cursor]:
        """Create a database cursor as a context manager."""
        with _internal.get_connection(self.conn) as conn:
            with conn.cursor() as cur:
                yield cur

    def _load_checkpoint_tuple(self, value: dict) -> CheckpointTuple:
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
            self._load_writes(value["pending_writes"]),
        )


__all__ = ["OracleSaver"]
