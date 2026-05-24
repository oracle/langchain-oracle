from __future__ import annotations

import json
import random
from collections.abc import Sequence
from decimal import Decimal
from typing import Any, Optional, cast

import oracledb
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    get_checkpoint_id,
)
from langgraph.checkpoint.serde.types import TASKS

MetadataInput = Optional[dict[str, Any]]
EMPTY_STRING_SENTINEL = " "

"""
To add a new migration, add a new string to the MIGRATIONS list.
The position of the migration in the list is the version number.
"""
MIGRATIONS = [
    """CREATE TABLE IF NOT EXISTS checkpoint_migrations (
    v NUMBER(10) PRIMARY KEY)
""",
    """
    CREATE TABLE IF NOT EXISTS checkpoints (
    thread_id VARCHAR2(2000) NOT NULL,
    checkpoint_ns VARCHAR2(2000) NOT NULL,
    checkpoint_id VARCHAR2(2000) NOT NULL,
    parent_checkpoint_id VARCHAR2(2000),
    type VARCHAR2(2000),
    checkpoint JSON NOT NULL,
    metadata JSON DEFAULT '{}' NOT NULL,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id))
""",
    """CREATE TABLE IF NOT EXISTS checkpoint_blobs (
    thread_id VARCHAR2(2000) NOT NULL,
    checkpoint_ns VARCHAR2(2000) NOT NULL,
    channel VARCHAR2(2000) NOT NULL,
    version VARCHAR2(2000) NOT NULL,
    type VARCHAR2(2000) NOT NULL,
    blob BLOB,
    PRIMARY KEY (thread_id, checkpoint_ns, channel, version))
""",
    """
    CREATE TABLE IF NOT EXISTS checkpoint_writes (
    thread_id VARCHAR2(2000) NOT NULL,
    checkpoint_ns VARCHAR2(2000) NOT NULL,
    checkpoint_id VARCHAR2(2000) NOT NULL,
    task_id VARCHAR2(2000) NOT NULL,
    idx NUMBER(10) NOT NULL,
    channel VARCHAR2(2000) NOT NULL,
    type VARCHAR2(2000),
    blob BLOB NOT NULL,
    task_path VARCHAR2(2000) NOT NULL,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx))
""",
    """
    CREATE INDEX IF NOT EXISTS checkpoints_thread_id_idx ON checkpoints(thread_id) ONLINE
    """,
    """
    CREATE INDEX IF NOT EXISTS checkpoint_blobs_thread_id_idx ON checkpoint_blobs(thread_id) ONLINE
    """,
    """
    CREATE INDEX IF NOT EXISTS checkpoint_writes_thread_id_idx ON checkpoint_writes(thread_id) ONLINE
    """,
]

SELECT_SQL = """
    SELECT
        thread_id,
        checkpoint,
        checkpoint_ns,
        checkpoint_id,
        parent_checkpoint_id,
        metadata
    FROM checkpoints 
"""

SELECT_CHANNEL_VALUES_SQL = """
    SELECT thread_id, checkpoint_ns, channel, version, type, blob 
    FROM checkpoint_blobs
"""

SELECT_PENDING_WRITES_SQL = """
    SELECT thread_id, checkpoint_ns, checkpoint_id, task_id, channel, type, blob 
    FROM checkpoint_writes 
    {where_clause}
    ORDER BY task_id, idx
"""

SELECT_PENDING_SENDS_SQL = (
    """
    SELECT
        checkpoint_id,
        type,
        blob
    FROM checkpoint_writes
    WHERE thread_id = :thread_id
        AND checkpoint_id IN ({checkpoint_bind})
        AND channel = '"""
    + TASKS
    + """'
    ORDER BY task_path, task_id, idx
"""
)

UPSERT_CHECKPOINT_BLOBS_SQL = """
    MERGE INTO checkpoint_blobs dest
    USING (
        SELECT
        :thread_id AS thread_id,
        :checkpoint_ns AS checkpoint_ns,
        :channel AS channel,
        :version AS version
        FROM dual
    ) src 
    ON (
        dest.thread_id = src.thread_id AND
        dest.checkpoint_ns = src.checkpoint_ns AND
        dest.channel = src.channel AND
        dest.version = src.version
    )
    WHEN NOT MATCHED THEN
        INSERT (thread_id, checkpoint_ns, channel, version, type, blob)
        VALUES (src.thread_id, src.checkpoint_ns, src.channel, src.version, :type, :blob)
"""

UPSERT_CHECKPOINTS_SQL = """
MERGE INTO checkpoints dest
USING (
    SELECT
        :thread_id            AS thread_id,
        :checkpoint_ns        AS checkpoint_ns,
        :checkpoint_id        AS checkpoint_id,
        :parent_checkpoint_id AS parent_checkpoint_id,
        :checkpoint           AS checkpoint,
        :metadata             AS metadata
    FROM dual
) src 
ON (
    dest.thread_id = src.thread_id AND
    dest.checkpoint_ns = src.checkpoint_ns AND
    dest.checkpoint_id = src.checkpoint_id
)
WHEN MATCHED THEN
  UPDATE SET
    dest.checkpoint = src.checkpoint,
    dest.metadata = src.metadata
WHEN NOT MATCHED THEN
  INSERT (
    thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, checkpoint, metadata
  ) VALUES (
    src.thread_id, src.checkpoint_ns, src.checkpoint_id,
    src.parent_checkpoint_id, src.checkpoint, src.metadata
  )
"""

UPSERT_CHECKPOINT_WRITES_SQL = """
MERGE INTO checkpoint_writes dest
USING (
    SELECT
        :thread_id      AS thread_id,
        :checkpoint_ns  AS checkpoint_ns,
        :checkpoint_id  AS checkpoint_id,
        :task_id        AS task_id,
        :task_path      AS task_path,
        :idx            AS idx,
        :channel        AS channel,
        :type           AS type,
        :blob           AS blob
    FROM dual
) src 
ON (
    dest.thread_id = src.thread_id AND
    dest.checkpoint_ns = src.checkpoint_ns AND
    dest.checkpoint_id = src.checkpoint_id AND
    dest.task_id = src.task_id AND
    dest.idx = src.idx
)
WHEN MATCHED THEN
  UPDATE SET
    dest.channel = src.channel,
    dest.type = src.type,
    dest.blob = src.blob
WHEN NOT MATCHED THEN
  INSERT (
    thread_id, checkpoint_ns, checkpoint_id,
    task_id, task_path, idx, channel, type, blob
  ) VALUES (
    src.thread_id, src.checkpoint_ns, src.checkpoint_id,
    src.task_id, src.task_path, src.idx,
    src.channel, src.type, src.blob
  )
"""

INSERT_CHECKPOINT_WRITES_SQL = """
MERGE INTO checkpoint_writes dest
USING (
    SELECT
 :thread_id AS thread_id,
 :checkpoint_ns AS checkpoint_ns,
 :checkpoint_id AS checkpoint_id,
 :task_id AS task_id,
 :task_path AS task_path,
 :idx AS idx,
 :channel AS channel,
 :type AS type,
 :blob AS blob
 FROM dual
) src 
ON (
    dest.thread_id = src.thread_id AND
    dest.checkpoint_ns = src.checkpoint_ns AND
    dest.checkpoint_id = src.checkpoint_id AND
    dest.task_id = src.task_id AND
    dest.idx = src.idx
)
WHEN NOT MATCHED THEN
  INSERT (
    thread_id, checkpoint_ns, checkpoint_id,
    task_id, task_path, idx, channel, type, blob
  ) VALUES (
    src.thread_id, src.checkpoint_ns, src.checkpoint_id,
    src.task_id, src.task_path, src.idx,
    src.channel, src.type, src.blob
  )
"""


class BaseOracleSaver(BaseCheckpointSaver[str]):
    SELECT_SQL = SELECT_SQL
    SELECT_PENDING_SENDS_SQL = SELECT_PENDING_SENDS_SQL
    SELECT_CHANNEL_VALUES_SQL = SELECT_CHANNEL_VALUES_SQL
    SELECT_PENDING_WRITES_SQL = SELECT_PENDING_WRITES_SQL
    MIGRATIONS = MIGRATIONS
    UPSERT_CHECKPOINT_BLOBS_SQL = UPSERT_CHECKPOINT_BLOBS_SQL
    UPSERT_CHECKPOINTS_SQL = UPSERT_CHECKPOINTS_SQL
    UPSERT_CHECKPOINT_WRITES_SQL = UPSERT_CHECKPOINT_WRITES_SQL
    INSERT_CHECKPOINT_WRITES_SQL = INSERT_CHECKPOINT_WRITES_SQL

    supports_pipeline: bool

    @staticmethod
    def _encode_not_null_text(value: str | None, *, field_name: str) -> str:
        """Encode empty public API strings for NOT NULL Oracle columns."""
        if value in (None, ""):
            return EMPTY_STRING_SENTINEL
        if value == EMPTY_STRING_SENTINEL:
            raise ValueError(
                f"{field_name} cannot be a single space because that value is "
                "reserved for Oracle empty-string storage."
            )
        return value

    @staticmethod
    def _decode_not_null_text(value: str) -> str:
        """Decode Oracle NOT NULL sentinel values back to the public API form."""
        return "" if value == EMPTY_STRING_SENTINEL else value

    def _encode_checkpoint_ns(self, value: str | None) -> str:
        return self._encode_not_null_text(value, field_name="checkpoint_ns")

    def _decode_checkpoint_ns(self, value: str) -> str:
        return self._decode_not_null_text(value)

    def _encode_task_path(self, value: str | None) -> str:
        return self._encode_not_null_text(value, field_name="task_path")

    @staticmethod
    def _validate_json_path_key(key: str) -> None:
        """Validate a key used in Oracle JSON path expressions."""
        import re

        if not re.match(r"^[a-zA-Z0-9_\.\[\],\s\*]+$", key):
            raise ValueError(
                f"Illegal metadata key: {key}. "
                f"Keys must contain only alphanumeric characters, underscores, and dots "
                f"to prevent JSON path injection vulnerabilities."
            )

    def _migrate_pending_sends(
        self,
        pending_sends: list[tuple[str, bytes]],
        checkpoint: dict[str, Any],
        channel_values: list[tuple[str, str, bytes]],
    ) -> None:
        if not pending_sends:
            return
        # add to values
        enc, blob = self.serde.dumps_typed(
            [self.serde.loads_typed((c, b)) for c, b in pending_sends],
        )
        channel_values.append((TASKS, enc, blob))
        # add to versions
        checkpoint["channel_versions"][TASKS] = (
            max(checkpoint["channel_versions"].values())
            if checkpoint["channel_versions"]
            else self.get_next_version(None, None)
        )

    def _load_blobs(self, blob_values: list[tuple[str, str, bytes]]) -> dict[str, Any]:
        if not blob_values:
            return {}
        return {
            k: self._load_typed_blob(t, v) for k, t, v in blob_values if t != "empty"
        }

    def _load_typed_blob(self, type_: str, blob: bytes | None) -> Any:
        if blob is None:
            blob = b""
        return self.serde.loads_typed((type_, blob))

    def _dump_blobs(
        self,
        thread_id: str,
        checkpoint_ns: str,
        values: dict[str, Any],
        versions: ChannelVersions,
    ) -> list[tuple[str, str, str, str, str, bytes | None]]:
        """Prepare blob data for database storage."""

        if not versions:
            return []

        return [
            {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "channel": k,
                "version": cast(str, ver),
                "type": (
                    self.serde.dumps_typed(values[k])[0] if k in values else "empty"
                ),
                "blob": (self.serde.dumps_typed(values[k])[1] if k in values else None),
            }
            for k, ver in versions.items()
        ]

    def _load_writes(
        self, writes: list[tuple[str, str, str, bytes]]
    ) -> list[tuple[str, str, Any]]:
        return (
            [
                (
                    tid,
                    channel,
                    self._load_typed_blob(t, v),
                )
                for tid, channel, t, v in writes
            ]
            if writes
            else []
        )

    def _dump_writes(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
        task_id: str,
        task_path: str,
        writes: Sequence[tuple[str, Any]],
    ) -> list[tuple[str, str, str, str, str, int, str, str, bytes]]:
        result = []
        for idx, (channel, value) in enumerate(writes):
            # Always serialize the value, even if it's None
            type_str, blob_data = self.serde.dumps_typed(value)

            # Ensure blob is never None
            if blob_data is None:
                blob_data = b""

            result.append(
                {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                    "task_id": task_id,
                    "task_path": task_path,
                    "idx": WRITES_IDX_MAP.get(channel, int(idx)),
                    "channel": channel,
                    "type": type_str,
                    "blob": blob_data,
                }
            )
        return result

    def get_next_version(self, current: str | None, channel: None) -> str:
        if current is None:
            current_v = 0
        elif isinstance(current, int):
            current_v = current
        else:
            current_v = int(current.split(".")[0])
        next_v = current_v + 1
        next_h = random.random()
        return f"{next_v:032}.{next_h:016}"

    def _search_where_channels_pending_writes(
        self, config: RunnableConfig | None, use_checkpoint_id: bool = False
    ) -> tuple[str, list[Any]]:
        """Return WHERE clause predicates for alist() given config, filter, before."""
        wheres = []
        param_values = {}

        # construct predicate for config filter
        if config:
            wheres.append("thread_id = :thread_id ")
            param_values["thread_id"] = config["configurable"]["thread_id"]
            if "checkpoint_ns" in config["configurable"]:
                checkpoint_ns = self._encode_checkpoint_ns(
                    config["configurable"]["checkpoint_ns"]
                )
                wheres.append("checkpoint_ns = :checkpoint_ns")
                param_values["checkpoint_ns"] = checkpoint_ns

            if use_checkpoint_id:
                if checkpoint_id := get_checkpoint_id(config):
                    wheres.append("checkpoint_id = :checkpoint_id")
                    param_values["checkpoint_id"] = checkpoint_id

        return (
            "WHERE " + " AND ".join(wheres) if wheres else "",
            param_values,
        )

    def _search_where(
        self,
        config: RunnableConfig | None,
        filter: MetadataInput,
        before: RunnableConfig | None = None,
    ) -> tuple[str, list[Any]]:
        """Return WHERE clause predicates for alist() given config, filter, before."""
        wheres = []
        param_values = {}

        # construct predicate for config filter
        if config:
            wheres.append("thread_id = :thread_id ")
            param_values["thread_id"] = config["configurable"]["thread_id"]
            if "checkpoint_ns" in config["configurable"]:
                checkpoint_ns = self._encode_checkpoint_ns(
                    config["configurable"]["checkpoint_ns"]
                )
                wheres.append("checkpoint_ns = :checkpoint_ns")
                param_values["checkpoint_ns"] = checkpoint_ns

            if checkpoint_id := get_checkpoint_id(config):
                wheres.append("checkpoint_id = :checkpoint_id")
                param_values["checkpoint_id"] = checkpoint_id

        # construct predicate for metadata filter
        if filter:
            # Oracle JSON filtering - build individual conditions for each key-value pair
            filter_conditions = []
            for i, (key, value) in enumerate(filter.items()):
                # SECURITY: Validate key to prevent JSON path injection attacks
                self._validate_json_path_key(key)
                if value is None:
                    # Check for null values
                    filter_conditions.append(f"JSON_VALUE(metadata, '$.{key}') IS NULL")
                elif isinstance(value, (dict, list)):
                    # For complex objects, use JSON_EQUAL for exact match
                    param_name = f"filter_json_{i}"
                    filter_conditions.append(
                        f"JSON_EQUAL(JSON_QUERY(metadata, '$.{key}'), :{param_name})"
                    )
                    param_values[param_name] = json.dumps(value)
                else:
                    # Check for simple values - preserve data types for proper comparison
                    param_name = f"filter_key_{i}"
                    if isinstance(value, bool):
                        # For boolean values, use JSON_EXISTS to check for boolean values safely
                        # Oracle has issues with boolean parameter binding, so use literal comparison
                        # NOTE: Must check bool BEFORE int/float since bool is subclass of int
                        bool_str = "true" if value else "false"
                        filter_conditions.append(
                            f"JSON_VALUE(metadata, '$.{key}') = '{bool_str}'"
                        )
                    elif isinstance(value, (int, float)):
                        # For numeric values, use JSON_VALUE with RETURNING NUMBER for proper type comparison
                        filter_conditions.append(
                            f"JSON_VALUE(metadata, '$.{key}' RETURNING NUMBER) = :{param_name}"
                        )
                        param_values[param_name] = value
                    else:
                        # For string values, use direct comparison
                        filter_conditions.append(
                            f"JSON_VALUE(metadata, '$.{key}') = :{param_name}"
                        )
                        param_values[param_name] = value

            if len(filter_conditions) > 0:
                wheres.append("(" + " AND ".join(filter_conditions) + ")")

        # construct predicate for `before`
        if before is not None:
            wheres.append("checkpoint_id < :before_checkpoint_id")
            param_values["before_checkpoint_id"] = get_checkpoint_id(before)

        return (
            "WHERE " + " AND ".join(wheres) if wheres else "",
            param_values,
        )

    def _get_channel_values(self, all_channel_values, all_values):
        channel_values_dict = {}

        for (
            thread_id,
            checkpoint_ns,
            channel,
            version,
            type,
            blob,
        ) in all_channel_values:
            if (thread_id, checkpoint_ns) not in channel_values_dict:
                channel_values_dict[(thread_id, checkpoint_ns)] = {}
            channel_values_dict[(thread_id, checkpoint_ns)][(channel, version)] = (
                channel,
                type,
                blob,
            )

        channel_values = []

        # For all checkpoints filtered
        for v in all_values:
            thread_id = v["thread_id"]
            checkpoint_ns = v["checkpoint_ns"]
            checkpoint = v.get("checkpoint", {})
            channel_versions = checkpoint.get("channel_versions", {})

            checkpoint_channel_values = []

            # Channel values from the blob table
            _values_dict = channel_values_dict.get((thread_id, checkpoint_ns), {})

            for channel, version in _values_dict:
                if (channel in channel_versions) and (
                    channel_versions[channel] == version
                ):
                    checkpoint_channel_values.append(_values_dict[(channel, version)])

            if len(checkpoint_channel_values) == 0:
                channel_values.append(None)
            else:
                channel_values.append(checkpoint_channel_values)

        return channel_values

    def _get_pending_writes(self, res, values):
        pending_writes_dict = {}

        for (
            thread_id,
            checkpoint_ns,
            checkpoint_id,
            task_id,
            channel,
            type,
            blob,
        ) in res:
            if (thread_id, checkpoint_ns, checkpoint_id) not in pending_writes_dict:
                pending_writes_dict[(thread_id, checkpoint_ns, checkpoint_id)] = []

            pending_writes_dict[(thread_id, checkpoint_ns, checkpoint_id)].append(
                (task_id, channel, type, blob)
            )

        pending_writes = []

        for v in values:
            thread_id = v["thread_id"]
            checkpoint_ns = v["checkpoint_ns"]
            checkpoint_id = v["checkpoint_id"]

            checkpoint_pending_writes = pending_writes_dict.get(
                (thread_id, checkpoint_ns, checkpoint_id), None
            )

            pending_writes.append(checkpoint_pending_writes)

        return pending_writes

    def output_type_handler(self, cursor, metadata):
        if metadata.type_code is oracledb.DB_TYPE_CLOB:
            return cursor.var(oracledb.DB_TYPE_LONG, arraysize=cursor.arraysize)
        if metadata.type_code is oracledb.DB_TYPE_BLOB:
            return cursor.var(oracledb.DB_TYPE_LONG_RAW, arraysize=cursor.arraysize)
        if metadata.type_code is oracledb.DB_TYPE_NCLOB:
            return cursor.var(
                oracledb.DB_TYPE_LONG_NVARCHAR, arraysize=cursor.arraysize
            )

    @staticmethod
    def _coerce_decimals(obj: Any) -> Any:
        """Recursively convert ``decimal.Decimal`` values to ``int``/``float``.

        Oracle's native JSON type deserializes every number as ``Decimal`` to
        preserve precision, but LangGraph expects plain ``int`` in several
        places (notably ``metadata["step"]``, which flows into
        ``uuid6(clock_seq=step)`` — that function does a bitwise AND and
        blows up on a ``Decimal``). We apply this on every dict loaded from a
        JSON column so callers see normal Python numerics.
        """
        if isinstance(obj, Decimal):
            return int(obj) if obj == obj.to_integral_value() else float(obj)
        if isinstance(obj, dict):
            return {k: BaseOracleSaver._coerce_decimals(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [BaseOracleSaver._coerce_decimals(v) for v in obj]
        return obj

    def _should_use_blob(self, value: Any, size_threshold_mb: float = 5.0) -> bool:
        """
        Decide if a channel value should go to blob storage.
        """
        # Always keep primitives in JSON (current behavior)
        if value is None or isinstance(value, (str, int, float, bool)):
            return False

        # For complex objects, check serialized size
        try:
            # Serialize to JSON to estimate size
            serialized = json.dumps(value)
            size_bytes = len(serialized)
            size_mb = size_bytes / (1024 * 1024)

            # Use blob if larger than threshold
            return size_mb > size_threshold_mb

        except TypeError:
            # If can't serialize to JSON (custom objects, etc.), use blob
            return True
