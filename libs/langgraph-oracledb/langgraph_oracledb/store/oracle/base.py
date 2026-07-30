from __future__ import annotations

import array
import asyncio
import concurrent.futures
import hashlib
import json
import logging
import re
import threading
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Literal,
    NamedTuple,
    TypeVar,
    Union,
    cast,
)

import oracledb
import orjson
from langgraph.store.base import (
    BaseStore,
    GetOp,
    IndexConfig,
    Item,
    ListNamespacesOp,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
    TTLConfig,
    ensure_embeddings,
    get_text_at_path,
    tokenize_path,
)
from typing_extensions import TypedDict

from langgraph_oracledb.checkpoint.oracle import _internal as _or_internal

from ...checkpoint.oracle.sync import _validate_conn_string

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

_TABLE_SUFFIX_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
_VALID_DISTANCE_METRICS = {"COSINE", "EUCLIDEAN", "DOT"}
_HNSW_INDEX_TYPE_KEYS = {"type", "distance_metric", "neighbors", "efconstruction"}
_IVF_INDEX_TYPE_KEYS = {
    "type",
    "distance_metric",
    "neighbor_partitions",
    "samples_per_partition",
    "min_vectors_per_partition",
}


class Migration(NamedTuple):
    """A database migration with optional conditions and parameters."""

    sql: str
    params: dict[str, Any] | None = None
    condition: Callable[[BaseOracleStore], bool] | None = None


class HNSWConfig(TypedDict, total=False):
    """HNSW (Hierarchical Navigable Small World) index configuration."""

    type: Literal["hnsw"]
    neighbors: int  # Maximum number of connections per node (range: 2-2048)
    efconstruction: (
        int  # Size of dynamic candidate list during construction (range: 1-65535)
    )
    distance_metric: Literal["COSINE", "EUCLIDEAN", "DOT"]


class IVFConfig(TypedDict, total=False):
    """IVF (Inverted File) index configuration."""

    type: Literal["ivf"]
    neighbor_partitions: int  # Number of partitions/clusters (range: 1-10000000)
    distance_metric: Literal["COSINE", "EUCLIDEAN", "DOT"]
    samples_per_partition: int  # SQL parameter name
    min_vectors_per_partition: int  # Range lower bound is 0


class PoolConfig(TypedDict, total=False):
    """Connection pool settings for Oracle connections."""

    min_size: int
    """Minimum number of connections maintained in the pool. Defaults to 1."""

    max_size: int | None
    """Maximum number of connections allowed in the pool. None means unlimited."""

    kwargs: dict
    """Additional connection arguments passed to each connection in the pool."""


class OracleIndexConfig(IndexConfig, total=False):
    """Configuration for vector embeddings in Oracle store with Oracle AI Vector Search specific options.

    Extends IndexConfig with additional configuration for Oracle vector index types.
    The vector index configuration determines the underlying algorithm and parameters
    used for similarity search operations.

    Important:
        The vector index configuration directly affects table creation and isolation.
        When you configure the store with different index parameters (including
        embedding dimensions, distance metrics, or index algorithm parameters),
        a new set of vector tables is automatically created with a unique suffix.
        This ensures that incompatible vector configurations don't interfere with
        each other, but also means you cannot share data between stores with
        different vector configurations.

        Example table isolation scenarios:
        - Store A: 1536 dimensions, COSINE distance, HNSW → tables with suffix "a1b2c3"
        - Store B: 768 dimensions, EUCLIDEAN distance, HNSW → tables with suffix "d4e5f6"
        - Store C: 1536 dimensions, COSINE distance, IVF → tables with suffix "g7h8i9"

        Each configuration gets its own isolated set of tables, allowing multiple
        vector search configurations to coexist in the same database.

    Warning:
        Oracle AI Vector Search requires Oracle Database 23c or higher with the vector
        capabilities enabled. The vector index creation will fail if the database
        doesn't support vector operations.

    Note:
        If no index_type is specified, defaults to HNSW with COSINE distance metric.
        The embedding dimensions are automatically detected from your embedding model.
    """

    index_type: HNSWConfig | IVFConfig
    accuracy: int


MIGRATIONS = [
    """
    CREATE TABLE IF NOT EXISTS {store} (
    -- 'prefix' represents the doc's 'namespace'
    prefix VARCHAR2(4000) NOT NULL,
    key VARCHAR2(4000) NOT NULL,
    value JSON NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE DEFAULT NULL,
    ttl_minutes NUMBER DEFAULT NULL,
    PRIMARY KEY (prefix, key))
    """,
    """
    -- For faster lookups by prefix
    CREATE INDEX IF NOT EXISTS {store}_prefix_idx ON {store} (prefix) ONLINE
    """,
    """
    -- Add indexes for efficient TTL sweeping
    CREATE INDEX IF NOT EXISTS idx_{store}_expires_at ON {store} (expires_at) ONLINE
    """,
    """
    CREATE TABLE IF NOT EXISTS store_configs (
        table_suffix VARCHAR2(4000) PRIMARY KEY,
        detected_dims NUMBER NOT NULL,
        distance_type VARCHAR2(4000) DEFAULT 'COSINE',
        index_params JSON,
        embed_fields VARCHAR2(4000),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_store_configs_table_suffix ON store_configs(table_suffix) ONLINE
    """,
]

VECTOR_MIGRATIONS: Sequence[Migration] = [
    Migration(
        """
        CREATE TABLE IF NOT EXISTS %(table_name)s (
            prefix VARCHAR2(2000) NOT NULL,
            key VARCHAR2(2000) NOT NULL,
            field_name VARCHAR2(2000) NOT NULL,

                        embedding VECTOR(%(dims)s),
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT %(table_name)s_pk PRIMARY KEY (prefix, key, field_name),
            CONSTRAINT %(table_name)s_fk FOREIGN KEY (prefix, key) 
                REFERENCES %(store_table_name)s(prefix, key) ON DELETE CASCADE
        )
        """,
        params={
            "table_name": lambda store: store.table_names["store_vectors"],
            "store_table_name": lambda store: store.table_names["store"],
            "dims": lambda store: store.index_config["dims"],
        },
    ),
    Migration(
        """
        CREATE VECTOR INDEX %(index_name)s 
        ON %(table_name)s(embedding) 
        ORGANIZATION %(organization_clause)s
        DISTANCE %(distance_metric)s
        %(target_accuracy_clause)s
        %(parameters_clause)s
        """,
        condition=lambda store: (
            store.index_config is not None
        ),  # Enable vector index creation when index_config is provided
        params={
            "table_name": lambda store: store.table_names["store_vectors"],
            "index_name": lambda store: (
                f"{store.table_names['store_vectors']}_idx_{hash(str(store.index_config)) % 1000000}"
            ),
            "organization_clause": lambda store: _get_organization_clause(store),
            "distance_metric": lambda store: _get_distance_metric(store),
            "target_accuracy_clause": lambda store: _get_target_accuracy_clause(store),
            "parameters_clause": lambda store: _get_parameters_clause(store),
        },
    ),
]


def _namespace_to_text(namespace: tuple[str, ...]) -> str:
    """Convert namespace tuple to text string."""
    return ".".join(namespace)


def _decode_ns_bytes(namespace: str | bytes | list) -> tuple[str, ...]:
    if isinstance(namespace, list):
        return tuple(namespace)
    if isinstance(namespace, bytes):
        namespace = namespace.decode()[1:]
    return tuple(namespace.split("."))


def _row_to_item(namespace: tuple[str, ...], row: dict) -> Item:
    """Convert a row from the database into an Item."""
    val = row["value"]
    # Oracle JSON columns return native Python objects (dict/list) directly
    # No parsing needed

    return Item(
        key=row["key"],
        namespace=namespace,
        value=val,
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _row_to_search_item(namespace: tuple[str, ...], row: dict) -> SearchItem:
    """Convert a row from the database into a SearchItem."""
    val = row["value"]
    # Oracle JSON columns return native Python objects (dict/list) directly
    # No parsing needed

    score = row.get("score")
    if score is not None:
        try:
            score = float(score)
        except (ValueError, TypeError):
            logger.warning("Invalid score: %s", score)
            score = None

    return SearchItem(
        value=val,
        key=row["key"],
        namespace=namespace,
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        score=score,
    )


def _group_ops(ops: Iterable[Op]) -> tuple[dict[type, list[tuple[int, Op]]], int]:
    grouped_ops: dict[type, list[tuple[int, Op]]] = defaultdict(list)
    tot = 0
    for idx, op in enumerate(ops):
        grouped_ops[type(op)].append((idx, op))
        tot += 1
    return grouped_ops, tot


C = TypeVar("C", bound=Union[oracledb.Connection, oracledb.AsyncConnection])


def _should_ignore_ttl_refresh_error(
    exc: Exception, namespace: tuple[str, ...], key: str
) -> bool:
    """Return whether a TTL refresh failure should be treated as best effort."""
    err = exc.args[0] if exc.args else exc
    code = getattr(err, "code", None)
    if code is not None:
        code = abs(code)

    if code == 54:
        logger.debug(
            "Skipping TTL refresh for %s/%s due to lock contention (ORA-00054)",
            namespace,
            key,
        )
        return True

    if code == 12801:
        logger.debug(
            "Skipping TTL refresh for %s/%s due to parallel execution error (ORA-12801)",
            namespace,
            key,
        )
        return True

    if code == 942:
        logger.debug(
            "Skipping TTL refresh for %s/%s because the backing table does not exist (ORA-00942)",
            namespace,
            key,
        )
        return True

    if code in (1403, 1001):
        logger.debug(
            "Skipping TTL refresh for %s/%s because the item was not found or the cursor became invalid (ORA-%05d)",
            namespace,
            key,
            code,
        )
        return True

    logger.warning(
        "TTL refresh failed for %s/%s with database error (ORA-%s): %s",
        namespace,
        key,
        f"{code:05d}" if code is not None else "unknown",
        exc,
    )
    return False


def _detect_dimensions(config) -> int:
    return config["dims"]


def _normalize_existing_index_params(existing_params: Any) -> dict[str, Any]:
    """Normalize persisted store config params into a plain dict."""
    if hasattr(existing_params, "read"):
        existing_params = existing_params.read()

    if isinstance(existing_params, bytes):
        existing_params = existing_params.decode()

    if isinstance(existing_params, Mapping):
        return dict(existing_params)

    if isinstance(existing_params, str):
        try:
            parsed = json.loads(existing_params)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Stored index configuration is not valid JSON. "
                "Use a different table_suffix or drop existing tables."
            ) from exc
        if not isinstance(parsed, dict):
            raise ValueError(
                "Stored index configuration must decode to a JSON object. "
                "Use a different table_suffix or drop existing tables."
            )
        return dict(parsed)

    raise ValueError(
        "Stored index configuration has an unsupported format. "
        "Use a different table_suffix or drop existing tables."
    )


def _validate_table_suffix(suffix: str) -> None:
    if not isinstance(suffix, str) or not _TABLE_SUFFIX_RE.fullmatch(suffix):
        raise ValueError(
            "table_suffix must start with a letter and contain only letters, "
            "digits, or underscores, with a maximum length of 64 characters."
        )


def _schema_inconsistency_error(
    *,
    table_suffix: str,
    migration_table: str,
    migration_version: int,
    missing_table: str,
) -> RuntimeError:
    return RuntimeError(
        "OracleStore schema is inconsistent for "
        f"table_suffix='{table_suffix}': {migration_table.upper()} records "
        f"migration {migration_version}, but {missing_table.upper()} is missing. "
        "Call teardown()/ateardown() on this store instance, then call setup() again, "
        "or use a new table_suffix."
    )


def _validate_int_range(
    name: str,
    value: Any,
    min_value: int,
    max_value: int | None = None,
) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if value < min_value or (max_value is not None and value > max_value):
        if max_value is None:
            raise ValueError(f"{name} must be at least {min_value}")
        raise ValueError(f"{name} must be between {min_value} and {max_value}")


def _validate_index_config(config: OracleIndexConfig) -> None:
    """Validate values interpolated into Oracle vector DDL."""
    _validate_int_range("dims", config.get("dims"), 1)

    accuracy = config.get("accuracy")
    if accuracy is not None:
        _validate_int_range("accuracy", accuracy, 1, 100)

    index_type = config.get("index_type", _DEFAULT_INDEX_CONFIG)
    if not isinstance(index_type, dict):
        raise ValueError("index_type must be a dictionary")

    index_kind = index_type.get("type", "hnsw")
    if index_kind not in {"hnsw", "ivf"}:
        raise ValueError("index_type.type must be 'hnsw' or 'ivf'")

    allowed_keys = (
        _HNSW_INDEX_TYPE_KEYS if index_kind == "hnsw" else _IVF_INDEX_TYPE_KEYS
    )
    unknown_keys = set(index_type) - allowed_keys
    if unknown_keys:
        unknown_keys_str = ", ".join(sorted(unknown_keys))
        raise ValueError(f"index_type contains unsupported keys: {unknown_keys_str}")

    distance_metric = index_type.get("distance_metric", "COSINE")
    if not isinstance(distance_metric, str):
        raise ValueError("distance_metric must be a string")
    if distance_metric.upper() not in _VALID_DISTANCE_METRICS:
        raise ValueError("distance_metric must be one of COSINE, EUCLIDEAN, or DOT")

    if index_kind == "hnsw":
        if "neighbors" in index_type:
            _validate_int_range(
                "index_type.neighbors", index_type["neighbors"], 2, 2048
            )
        if "efconstruction" in index_type:
            _validate_int_range(
                "index_type.efconstruction", index_type["efconstruction"], 1, 65535
            )
    else:
        if "neighbor_partitions" in index_type:
            _validate_int_range(
                "index_type.neighbor_partitions",
                index_type["neighbor_partitions"],
                1,
                10000000,
            )
        if "samples_per_partition" in index_type:
            _validate_int_range(
                "index_type.samples_per_partition",
                index_type["samples_per_partition"],
                1,
            )
        if "min_vectors_per_partition" in index_type:
            _validate_int_range(
                "index_type.min_vectors_per_partition",
                index_type["min_vectors_per_partition"],
                0,
            )


def _generate_suffix(config: OracleIndexConfig) -> str:
    """Generate deterministic suffix from configuration."""
    if not config:
        config_str = json.dumps("None", sort_keys=True)
        hash_value = hashlib.sha256(config_str.encode()).hexdigest()[:6]
        return hash_value
    # Create stable hash components
    hash_input = {
        "dims": _detect_dimensions(config),  # Auto-detected dimensions
        "index_params": config.get("index_type", {"type": "hnsw"}),  # Full params
        "fields": config.get("fields", ["$"]),
    }

    # Generate hash
    config_str = json.dumps(hash_input, sort_keys=True)
    hash_value = hashlib.sha256(config_str.encode()).hexdigest()[:6]

    # Create readable suffix: just the hash
    return hash_value  # Example: a1b2c3


class BaseOracleStore(Generic[C]):
    MIGRATIONS = MIGRATIONS
    VECTOR_MIGRATIONS = VECTOR_MIGRATIONS
    conn: C
    _deserializer: Callable[[bytes | orjson.Fragment], dict[str, Any]] | None
    index_config: OracleIndexConfig | None
    table_suffix: str | None
    table_names: dict[str, str]
    detected_dims: int | None
    db_model_name: str | None

    @staticmethod
    def _validate_json_path_key(key: str) -> None:
        """Validate a key used in Oracle JSON path expressions."""
        import re

        if not re.match(r"^[A-Za-z0-9_.]+$", key):
            raise ValueError(
                f"Illegal metadata key: {key}. "
                f"Keys must contain only alphanumeric characters, underscores, and dots "
                f"to prevent JSON path injection vulnerabilities."
            )

    def _get_batch_GET_ops_queries(
        self,
        get_ops: Sequence[tuple[int, GetOp]],
    ) -> list[tuple[str, tuple, tuple[str, ...], list]]:
        """Prepare GET queries for batch operations."""
        namespace_groups = defaultdict(list)
        refresh_ttls = defaultdict(list)
        for idx, op in get_ops:
            namespace_groups[op.namespace].append((idx, op.key))
            refresh_ttls[op.namespace].append(op.refresh_ttl)

        results = []
        for namespace, items in namespace_groups.items():
            _, keys = zip(*items, strict=False)
            keys = list(keys)
            this_refresh_ttls = refresh_ttls[namespace]

            filtered_keys = [x for i, x in enumerate(keys) if this_refresh_ttls[i]]

            if len(filtered_keys) > 0:
                placeholders = [f":key{i}" for i in range(len(filtered_keys))]
                params = {
                    "namespace": _namespace_to_text(namespace),
                    **{a[1:]: b for a, b in zip(placeholders, filtered_keys)},
                }
                placeholders = ",".join(placeholders)

                query_update = f"""
                    UPDATE {self.table_names["store"]} 
                    SET expires_at = CURRENT_TIMESTAMP + NUMTODSINTERVAL(ttl_minutes, 'MINUTE')
                    WHERE prefix = :namespace {f" AND key IN ({placeholders})" if len(placeholders) > 0 else ""} AND ttl_minutes IS NOT NULL
                """

                results.append((query_update, params, namespace, items))

            placeholders = [f":key{i}" for i in range(len(keys))]
            params = {
                "namespace": _namespace_to_text(namespace),
                **{a[1:]: b for a, b in zip(placeholders, keys)},
            }
            placeholders = ",".join(placeholders)

            query_select = f"""
                SELECT key, value, created_at, updated_at
                FROM {self.table_names["store"]} 
                WHERE prefix = :namespace {f"AND key IN ({placeholders})" if len(placeholders) > 0 else ""}
            """

            results.append((query_select, params, namespace, items))

        return results

    def _prepare_batch_PUT_queries(
        self,
        put_ops: Sequence[tuple[int, PutOp]],
    ) -> tuple[list[tuple[str, Sequence]], tuple[str, Sequence] | None]:
        """Prepare PUT queries for batch operations."""
        dedupped_ops: dict[tuple[tuple[str, ...], str], PutOp] = {}
        for _, op in put_ops:
            dedupped_ops[(op.namespace, op.key)] = op

        inserts: list[PutOp] = []
        deletes: list[PutOp] = []
        for op in dedupped_ops.values():
            if op.value is None:
                deletes.append(op)
            else:
                inserts.append(op)

        queries: list[tuple[str, Sequence]] = []

        if deletes:
            namespace_groups: dict[tuple[str, ...], list[str]] = defaultdict(list)
            for op in deletes:
                namespace_groups[op.namespace].append(op.key)
            for namespace, keys in namespace_groups.items():
                placeholders = ",".join([f":key{i}" for i in range(len(keys))])
                query = f"DELETE FROM {self.table_names['store']} WHERE prefix = :namespace AND key IN ({placeholders})"
                params = {"namespace": _namespace_to_text(namespace)}
                for i, key in enumerate(keys):
                    params[f"key{i}"] = key
                queries.append((query, params))

        embedding_request: (
            tuple[tuple[str, str], Sequence[tuple[str, str, str, str]]] | None
        ) = None
        if inserts:
            embedding_request_params = []

            # First handle main store insertions
            for op in inserts:
                if op.ttl is not None:
                    expires_at_expr = (
                        f"CURRENT_TIMESTAMP + NUMTODSINTERVAL({op.ttl}, 'MINUTE')"
                    )
                    ttl_minutes = op.ttl
                else:
                    expires_at_expr = "NULL"
                    ttl_minutes = None

                merge_sql = f"""
                    MERGE INTO {self.table_names["store"]} s
                    USING (SELECT :prefix as prefix, :key as key, 
                                  :value as value, 
                                  {expires_at_expr} as expires_at, 
                                  :ttl_minutes as ttl_minutes 
                           FROM DUAL) src
                    ON (s.prefix = src.prefix AND s.key = src.key)
                    WHEN MATCHED THEN UPDATE SET
                        value = src.value,
                        updated_at = CURRENT_TIMESTAMP,
                        expires_at = src.expires_at,
                        ttl_minutes = src.ttl_minutes
                    WHEN NOT MATCHED THEN INSERT (
                        prefix, key, value, created_at, updated_at, expires_at, ttl_minutes
                    ) VALUES (
                        src.prefix, src.key, src.value, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP,
                        src.expires_at, src.ttl_minutes
                    )
                """

                params = {
                    "prefix": _namespace_to_text(op.namespace),
                    "key": op.key,
                    "value": json.dumps(op.value),
                    "ttl_minutes": ttl_minutes,
                }

                """if op.ttl is not None:
                    params["ttl"] = op.ttl"""

                queries.append((merge_sql, params))

            # Then handle embeddings if configured
            if self.index_config:
                for op in inserts:
                    if op.index is False:
                        continue
                    value = op.value
                    ns = _namespace_to_text(op.namespace)
                    k = op.key

                    if op.index is None:
                        paths = cast(dict, self.index_config)["__tokenized_fields"]
                    else:
                        paths = [(ix, tokenize_path(ix)) for ix in op.index]

                    for path, tokenized_path in paths:
                        texts = get_text_at_path(value, tokenized_path)
                        for i, text in enumerate(texts):
                            pathname = f"{path}.{i}" if len(texts) > 1 else path
                            embedding_request_params.append(
                                {
                                    "prefix_vec": ns,
                                    "key_vec": k,
                                    "field_name_vec": pathname,
                                    "embedding_vector": text,
                                }
                            )

            if embedding_request_params:
                merge_q = f"""
                    MERGE INTO {self.table_names["store_vectors"]} dst
                    USING (
                        SELECT :prefix_vec AS prefix,
                            :key_vec AS key,
                            :field_name_vec AS field_name,
                            :embedding_vector AS embedding,
                            CURRENT_TIMESTAMP AS created_at,
                            CURRENT_TIMESTAMP AS updated_at
                        FROM dual
                    ) src
                    ON (
                        dst.prefix = src.prefix
                        AND dst.key = src.key
                        AND dst.field_name = src.field_name
                    )
                    WHEN MATCHED THEN
                        UPDATE SET
                            dst.embedding = src.embedding,
                            dst.updated_at = CURRENT_TIMESTAMP
                    WHEN NOT MATCHED THEN
                        INSERT (prefix, key, field_name, embedding, created_at, updated_at)
                        VALUES (src.prefix, src.key, src.field_name, src.embedding, src.created_at, src.updated_at);
                """

                embedding_request = (merge_q, embedding_request_params)

        return queries, embedding_request

    def _prepare_batch_search_queries(
        self,
        search_ops: Sequence[tuple[int, SearchOp]],
    ) -> tuple[list[tuple[str, list]], list[tuple[int, str]]]:
        """Prepare search queries for batch operations."""
        queries = []
        embedding_requests = []

        store_table = self.table_names["store"]

        for idx, (_, op) in enumerate(search_ops):
            filter_clauses = []
            filter_param_dict = {}
            if op.filter:
                filter_counter = 0
                for key, value in op.filter.items():
                    if isinstance(value, dict):
                        for op_name, val in value.items():
                            param_name = f"filter_{filter_counter}"
                            condition, param_ = self._get_filter_condition(
                                key, op_name, val, param_name
                            )
                            filter_clauses.append(condition)
                            filter_param_dict[param_name] = param_
                            filter_counter += 1
                    else:
                        # SECURITY: Validate key to prevent JSON path injection attacks
                        param_name = f"filter_{filter_counter}"
                        condition, param_ = self._get_filter_condition(
                            key, "$eq", value, param_name
                        )
                        filter_clauses.append(condition)
                        filter_param_dict[param_name] = param_
                        filter_counter += 1

            ns_condition = "1=1"
            ns_param: Sequence[str] | None = None
            if op.namespace_prefix:
                ns_condition = "st.prefix LIKE :ns_prefix"
                ns_param = (f"{_namespace_to_text(op.namespace_prefix)}%",)

            extra_filters = (
                " AND " + " AND ".join(filter_clauses) if filter_clauses else ""
            )

            if op.query and self.index_config:
                # We'll embed the text later, so record the request.
                embedding_requests.append((idx, op.query))

                score_operator, post_operator = get_distance_operator(self)
                post_operator = post_operator.replace("scored", "uniq")

                vectors_per_doc_estimate = cast(dict, self.index_config)[
                    "__estimated_num_vectors"
                ]
                expanded_limit = (op.limit * vectors_per_doc_estimate * 2) + 1

                # Oracle vector search CTE

                vectors_table = self.table_names["store_vectors"]
                vector_search_cte = f"""
                        SELECT st.prefix, st.key, st.value, st.created_at, st.updated_at,
                            {score_operator} AS neg_score
                        FROM {store_table} st
                        JOIN {vectors_table} sv ON st.prefix = sv.prefix AND st.key = sv.key
                        WHERE {ns_condition} {extra_filters}
                        ORDER BY {score_operator} ASC
                        FETCH FIRST :expanded_limit ROWS ONLY
                    """

                search_results_sql = f"""
                        WITH scored AS (
                            {vector_search_cte}
                        )
                        SELECT uniq.prefix, uniq.key, uniq.value, uniq.created_at, uniq.updated_at,
                            {post_operator} AS score
                        FROM (
                            SELECT prefix, key, value, created_at, updated_at, neg_score
                            FROM (
                                SELECT 
                                    prefix, key, value, created_at, updated_at, neg_score, ROW_NUMBER() OVER (PARTITION BY prefix, key ORDER BY neg_score ASC) AS rn
                                FROM scored
                            )
                            WHERE rn = 1
                        ) uniq
                        ORDER BY score DESC
                        OFFSET :offset ROWS FETCH NEXT :limit ROWS ONLY
                    """

                # Build parameter dictionary directly for Oracle
                search_results_params = {
                    "expanded_limit": expanded_limit,
                    "offset": op.offset,
                    "limit": op.limit,
                }
                if ns_param:
                    search_results_params["ns_prefix"] = ns_param[0]
                # Add filter params with their exact names
                search_results_params.update(filter_param_dict)

            else:
                base_query = f"""
                        SELECT st.prefix, st.key, st.value, st.created_at, st.updated_at, NULL AS score
                        FROM {store_table} st
                        WHERE {ns_condition} {extra_filters}
                        ORDER BY st.updated_at DESC
                        OFFSET :offset ROWS FETCH NEXT :limit ROWS ONLY
                    """
                search_results_sql = base_query
                # Build parameter dictionary directly for Oracle
                search_results_params = {
                    "offset": op.offset,
                    "limit": op.limit,
                }
                if ns_param:
                    search_results_params["ns_prefix"] = ns_param[0]
                # Add filter params with their exact names
                search_results_params.update(filter_param_dict)

            final_sql = search_results_sql
            final_params = search_results_params

            queries.append((final_sql, final_params))

        return queries, embedding_requests

    def _get_filter_condition(
        self, key: str, op: str, value: Any, param_name: str = "value"
    ) -> tuple[str, list]:
        """Helper to generate filter conditions for Oracle JSON."""
        self._validate_json_path_key(key)

        if op == "$eq":
            return (
                f"JSON_EQUAL(JSON_QUERY(value, '$.{key}'), :{param_name})",
                json.dumps(value),
            )
        elif op == "$gt":
            return f"TO_NUMBER(JSON_VALUE(value, '$.{key}')) > :{param_name}", str(
                value
            )
        elif op == "$gte":
            return f"TO_NUMBER(JSON_VALUE(value, '$.{key}')) >= :{param_name}", str(
                value
            )
        elif op == "$lt":
            return f"TO_NUMBER(JSON_VALUE(value, '$.{key}')) < :{param_name}", str(
                value
            )
        elif op == "$lte":
            return f"TO_NUMBER(JSON_VALUE(value, '$.{key}')) <= :{param_name}", str(
                value
            )
        elif op == "$ne":
            return (
                f"NOT JSON_EQUAL(JSON_QUERY(value, '$.{key}'), :{param_name})",
                json.dumps(value),
            )
        else:
            raise ValueError(f"Unsupported operator: {op}")

    def _get_batch_list_namespaces_queries(
        self,
        list_ops: Sequence[tuple[int, ListNamespacesOp]],
    ) -> list[tuple[str, Sequence]]:
        """Prepare list namespaces queries for batch operations."""
        queries: list[tuple[str, Sequence]] = []

        for _, op in list_ops:
            where_clauses = []
            params = {}
            param_count = 0

            # Handle match conditions
            if op.match_conditions:
                for condition in op.match_conditions:
                    param_count += 1
                    if condition.match_type == "prefix":
                        where_clauses.append(f"prefix LIKE :match_param{param_count}")
                        prefix_text = _namespace_to_text(condition.path)
                        # Handle wildcards by replacing * with %
                        prefix_text = prefix_text.replace("*", "%")
                        params[f"match_param{param_count}"] = f"{prefix_text}%"
                    elif condition.match_type == "suffix":
                        where_clauses.append(f"prefix LIKE :match_param{param_count}")
                        suffix_text = _namespace_to_text(condition.path)
                        suffix_text = suffix_text.replace("*", "%")
                        params[f"match_param{param_count}"] = f"%{suffix_text}"
                    else:
                        logger.warning(
                            f"Unknown match_type in list_namespaces: {condition.match_type}"
                        )

            where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

            query = f"""
            WITH per_row AS (
                SELECT
                    s.prefix,
                    CASE
                        WHEN :num_parts IS NULL THEN 
                            s.prefix
                        ELSE (
                            SELECT LISTAGG(REGEXP_SUBSTR(s.prefix, '[^.]+', 1, LEVEL), '.')
                                WITHIN GROUP (ORDER BY LEVEL)
                            FROM dual
                            CONNECT BY LEVEL <= LEAST(
                                :num_parts,
                                REGEXP_COUNT(s.prefix, '.') + 1
                            )
                        )
                    END AS truncated_prefix
                FROM {self.table_names["store"]} s
                {where_sql}
            ),
            ranked AS (
                SELECT
                    truncated_prefix,
                    prefix,
                    ROW_NUMBER() OVER (
                        PARTITION BY truncated_prefix 
                        ORDER BY prefix
                    ) AS rn
                FROM per_row
            )
            SELECT
                truncated_prefix,
                prefix
            FROM ranked
            WHERE rn = 1
            ORDER BY prefix
            OFFSET :offset ROWS FETCH NEXT :limit ROWS ONLY
            """

            params.update(
                {
                    "num_parts": op.max_depth,
                    "offset": op.offset,
                    "limit": op.limit,
                }
            )

            queries.append((query, params))

        return queries

    def _prepare_statements(self, grouped_ops):
        prepared_statements = {}

        if GetOp in grouped_ops:
            get_ops = cast(Sequence[tuple[int, GetOp]], grouped_ops[GetOp])
            prepared_statements["get"] = self._get_batch_GET_ops_queries(get_ops)

        if SearchOp in grouped_ops:
            search_ops = cast(Sequence[tuple[int, SearchOp]], grouped_ops[SearchOp])
            queries, embedding_requests = self._prepare_batch_search_queries(search_ops)
            self._embed_requests(queries, embedding_requests, op_type="search")
            prepared_statements["search"] = (queries, embedding_requests)

        if ListNamespacesOp in grouped_ops:
            list_ops = cast(
                Sequence[tuple[int, ListNamespacesOp]], grouped_ops[ListNamespacesOp]
            )
            prepared_statements["list"] = self._get_batch_list_namespaces_queries(
                list_ops
            )

        if PutOp in grouped_ops:
            put_ops = cast(Sequence[tuple[int, PutOp]], grouped_ops[PutOp])
            queries, embedding_requests = self._prepare_batch_PUT_queries(put_ops)
            self._embed_requests(queries, embedding_requests, op_type="put")
            prepared_statements["put"] = (queries, embedding_requests)

        return prepared_statements

    def _embed_requests(self, queries, embedding_requests, op_type="search"):
        if not self.embeddings or not embedding_requests:
            return

        if embedding_requests:
            if self.embeddings is None:
                # Should not get here since the embedding config is required
                # to return an embedding_request above
                raise ValueError(
                    "Embedding configuration is required for vector operations "
                    f"(for semantic search). "
                    f"Please provide an Embeddings when initializing the {self.__class__.__name__}."
                )

        if op_type == "search":
            embeddings = self.embeddings.embed_documents(
                [query for _, query in embedding_requests]
            )

            for (idx, _), embedding in zip(
                embedding_requests, embeddings, strict=False
            ):
                _params_dict = queries[idx][1]
                _params_dict["embedding_vector"] = array.array("f", embedding)
        elif op_type == "put":
            query, txt_params = embedding_requests

            vectors = self.embeddings.embed_documents(
                [param["embedding_vector"] for param in txt_params]
            )

            for p_, vector in zip(txt_params, vectors, strict=False):
                p_["embedding_vector"] = array.array("f", vector)
                queries.append((query, p_))


class OracleStore(BaseStore, BaseOracleStore[oracledb.Connection]):
    """Oracle-backed store with optional vector search using Oracle AI Vector Search.

    !!! example \"Examples\"
        Basic setup and usage:
        ```python
        from langgraph_oracledb.store.oracle import OracleStore
        import oracledb

        conn_string = ...

        # Using direct connection
        with oracledb.connect(conn_string) as conn:
            store = OracleStore(conn)
            store.setup()  # Run migrations. Done once

            # Store and retrieve data
            store.put((\"users\", \"123\"), \"prefs\", {\"theme\": \"dark\"})
            item = store.get((\"users\", \"123\"), \"prefs\")
        ```

        Or using the convenient from_conn_string helper:
        ```python
        from langgraph_oracledb.store.oracle import OracleStore

        conn_string = ...

        with OracleStore.from_conn_string(conn_string) as store:
            store.setup()

            # Store and retrieve data
            store.put((\"users\", \"123\"), \"prefs\", {\"theme\": \"dark\"})
            item = store.get((\"users\", \"123\"), \"prefs\")
        ```

        Vector search using LangChain embeddings:
        ```python
        from langchain.embeddings import init_embeddings
        from langgraph_oracledb.store.oracle import OracleStore

        conn_string = ...

        with OracleStore.from_conn_string(
            conn_string,
            index={
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
            store.setup()  # Do this once to run migrations

            # Store documents
            store.put((\"docs\",), \"doc1\", {\"text\": \"Python tutorial\"})
            store.put((\"docs\",), \"doc2\", {\"text\": \"TypeScript guide\"})
            store.put((\"docs\",), \"doc3\", {\"text\": \"Other guide\"}, index=False)  # don't index

            # Search by similarity
            results = store.search((\"docs\",), query=\"programming guides\", limit=2)
        ```
    """

    supports_ttl: bool = True

    def __init__(
        self,
        conn: oracledb.Connection,
        *,
        index: OracleIndexConfig | None = None,
        ttl: TTLConfig | None = None,
        table_suffix: str | None | None = None,
    ) -> None:
        super().__init__()
        self.conn = conn
        self.index_config = index

        if self.index_config:
            self.embeddings, self.index_config = _ensure_index_config(self.index_config)
            _validate_index_config(self.index_config)
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
        self._ttl_sweeper_thread: threading.Thread | None = None
        self._ttl_stop_event = threading.Event()

    def _validate_configuration(
        self, suffix: str, new_config: OracleIndexConfig
    ) -> None:
        """Validate that new configuration matches existing tables."""
        with self._cursor() as cur:
            try:
                # Check if configuration exists
                cur.execute(
                    """
                    SELECT detected_dims, distance_type, index_params 
                    FROM store_configs 
                    WHERE table_suffix = :1
                """,
                    (suffix,),
                )
                row = cur.fetchone()
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

    def _register_configuration(self, suffix: str, config: OracleIndexConfig) -> None:
        """Register a new configuration in the store_configs table."""
        with self._cursor() as cur:
            # First ensure store_configs table exists by running setup
            try:
                cur.execute("SELECT 1 FROM store_configs WHERE ROWNUM = 1")
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

            cur.execute(
                """
                INSERT /*+ IGNORE_ROW_ON_DUPKEY_INDEX(store_configs (table_suffix)) */ INTO store_configs 
                (table_suffix, detected_dims, distance_type, index_params, embed_fields, created_at, last_used)
                VALUES (:1, :2, :3, :4, :5, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """,
                (suffix, detected_dims, distance_type, index_params, embed_fields),
            )

            cur.connection.commit()
        logger.info(
            f"Registered new configuration for table suffix '{suffix}' with {detected_dims} dimensions"
        )

    @classmethod
    @contextmanager
    def from_conn_string(
        cls,
        conn_string: str,
        *,
        pool_config: dict[str, Any] | None = None,
        index: OracleIndexConfig | None = None,
        ttl: TTLConfig | None = None,
        table_suffix: str | None | None = None,
    ) -> Iterator[OracleStore]:
        """Create a new OracleStore instance from a connection string.

        Args:
            conn_string: Oracle connection string
            pool_config: Optional pool configuration with min_size and max_size
            index: Optional index configuration for vector search
            ttl: Optional TTL configuration for expiration
            table_suffix: Optional table suffix for testing (replaces system generated hash suffix)
        """

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
                store = cls(conn=pool, index=index, ttl=ttl, table_suffix=table_suffix)
                yield store
            finally:
                pool.close()
        else:
            with oracledb.connect(user=user, password=password, dsn=dsn) as conn:
                store = cls(conn=conn, index=index, ttl=ttl, table_suffix=table_suffix)
                yield store

    @contextmanager
    def _cursor(self) -> Iterator[oracledb.Cursor]:
        """Create a database cursor as a context manager.

        Args:
            pipeline: whether to use pipeline for the DB operations inside the context manager.
                Will be applied regardless of whether the OracleSaver instance was initialized with a pipeline.
                If pipeline mode is not supported, will fall back to using transaction context manager.
        """
        with _or_internal.get_connection(self.conn) as conn:
            with conn.cursor() as cur:
                yield cur

    def _table_exists(self, cur: oracledb.Cursor, table_name: str) -> bool:
        cur.execute(
            """
            SELECT 1
            FROM user_tables
            WHERE table_name = :1
            """,
            (table_name.upper(),),
        )
        return cur.fetchone() is not None

    def _raise_if_schema_inconsistent(
        self,
        cur: oracledb.Cursor,
        *,
        migration_table: str,
        migration_version: int,
        required_table: str,
    ) -> None:
        if migration_version >= 0 and not self._table_exists(cur, required_table):
            raise _schema_inconsistency_error(
                table_suffix=self.table_suffix or "",
                migration_table=migration_table,
                migration_version=migration_version,
                missing_table=required_table,
            )

    def setup(self) -> None:
        """Set up the store database.

        This method creates the necessary tables in the Oracle database if they don't
        already exist and runs database migrations. It MUST be called directly by the user
        the first time the store is used.
        """

        def _get_version(cur: oracledb.Cursor, table: str) -> int:
            cur.execute(f"CREATE TABLE IF NOT EXISTS {table} (v INTEGER PRIMARY KEY)")
            cur.connection.commit()
            cur.execute(
                f"SELECT v FROM {table} ORDER BY v DESC FETCH FIRST 1 ROWS ONLY"
            )
            row = cur.fetchone()
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

        # Register or validate configuration if we have an index config
        if self.index_config:
            if self._needs_validation:
                self._validate_configuration(self.table_suffix, self.index_config)

        with self._cursor() as cur:
            # First ensure global migrations are run (including store_configs table)
            # These use hardcoded table names and should only run once per database
            version = _get_version(cur, table=self.table_names["store_migrations"])
            self._raise_if_schema_inconsistent(
                cur,
                migration_table=self.table_names["store_migrations"],
                migration_version=version,
                required_table=self.table_names["store"],
            )
            if version >= 3:
                self._raise_if_schema_inconsistent(
                    cur,
                    migration_table=self.table_names["store_migrations"],
                    migration_version=version,
                    required_table="store_configs",
                )
            for v, sql in enumerate(self.MIGRATIONS[version + 1 :], start=version + 1):
                try:
                    sql_formatted = sql.format(**self.table_names)
                    cur.execute(sql_formatted)
                    cur.execute(
                        f"INSERT /*+ IGNORE_ROW_ON_DUPKEY_INDEX({self.table_names['store_migrations']} (v)) */ INTO {self.table_names['store_migrations']} (v) VALUES (:1)",
                        (v,),
                    )
                    cur.connection.commit()
                except Exception as e:
                    logger.error(
                        f"Failed to apply migration {v}.\nSql={sql}\nError={e}"
                    )
                    raise

            # Only run vector migrations if index_config is provided
            if self.index_config:
                vector_migration_table_name = self.table_names["vector_migrations"]
                version = _get_version(cur, table=vector_migration_table_name)
                self._raise_if_schema_inconsistent(
                    cur,
                    migration_table=vector_migration_table_name,
                    migration_version=version,
                    required_table=self.table_names["store_vectors"],
                )

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

                    cur.execute(sql)
                    cur.execute(
                        f"INSERT /*+ IGNORE_ROW_ON_DUPKEY_INDEX({vector_migration_table_name} (v)) */ INTO {vector_migration_table_name} (v) VALUES (:1)",
                        (v,),
                    )
                    cur.connection.commit()

                try:
                    self._register_configuration(self.table_suffix, self.index_config)
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

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        """Execute a batch of operations."""
        grouped_ops, num_ops = _group_ops(ops)
        results: list[Result] = [None] * num_ops

        prepared_statements = self._prepare_statements(grouped_ops)

        with self._cursor() as cur:
            if GetOp in grouped_ops:
                self._batch_get_ops(prepared_statements["get"], results, cur)

            if SearchOp in grouped_ops:
                self._batch_search_ops(
                    prepared_statements["search"],
                    cast(Sequence[tuple[int, SearchOp]], grouped_ops[SearchOp]),
                    results,
                    cur,
                )

            if ListNamespacesOp in grouped_ops:
                self._batch_list_namespaces_ops(
                    prepared_statements["list"],
                    cast(
                        Sequence[tuple[int, ListNamespacesOp]],
                        grouped_ops[ListNamespacesOp],
                    ),
                    results,
                    cur,
                )

            if PutOp in grouped_ops:
                self._batch_put_ops(prepared_statements["put"], cur)

        return results

    def _batch_get_ops(
        self,
        prepared_statements,
        results: list[Result],
        cur: oracledb.Cursor,
    ) -> None:
        for query, params, namespace, items in prepared_statements:
            cur.execute(query, params)

            if "UPDATE" in query:
                cur.connection.commit()
                continue

            rows = cur.fetchall()
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

    def _batch_put_ops(
        self,
        prepared_statements,
        cur: oracledb.Cursor,
    ) -> None:
        """Execute batch put operations."""
        queries, _ = prepared_statements

        for query, params in queries:
            cur.execute(query, params)

        cur.connection.commit()

    def _batch_search_ops(
        self,
        prepared_statements,
        search_ops: Sequence[tuple[int, SearchOp]],
        results: list[Result],
        cur: oracledb.Cursor,
    ) -> None:
        """Execute batch search operations."""
        queries, _ = prepared_statements

        for (idx, op), (query, params) in zip(search_ops, queries):
            cur.execute(query, params)
            rows = cur.fetchall()

            col_names = [desc[0].lower() for desc in cur.description]
            dict_rows = [dict(zip(col_names, row)) for row in rows]

            items = []
            for row in dict_rows:
                namespace = _decode_ns_bytes(row["prefix"])
                items.append(_row_to_search_item(namespace, row))

            # Handle TTL refresh if requested - use NOWAIT to avoid deadlocks
            if op.refresh_ttl and items:
                # Use individual updates with NOWAIT to avoid deadlocks
                # If we can't get a lock immediately, skip the TTL refresh for that item
                for item in items:
                    ttl_update_query = f"""
                        UPDATE /*+ NOWAIT */ {self.table_names["store"]} 
                        SET expires_at = CURRENT_TIMESTAMP + NUMTODSINTERVAL(ttl_minutes, 'MINUTE')
                        WHERE prefix = :prefix AND key = :key AND ttl_minutes IS NOT NULL
                    """
                    try:
                        cur.execute(
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

    def _batch_list_namespaces_ops(
        self,
        prepared_statements,
        list_ops: Sequence[tuple[int, ListNamespacesOp]],
        results: list[Result],
        cur: oracledb.Cursor,
    ) -> None:
        for (query, params), (idx, _) in zip(
            prepared_statements, list_ops, strict=False
        ):
            cur.execute(query, params)
            rows = cur.fetchall()

            col_names = [desc[0].lower() for desc in cur.description]
            dict_rows = [dict(zip(col_names, row)) for row in rows]

            results[idx] = [
                _decode_ns_bytes(row["truncated_prefix"]) for row in dict_rows
            ]

    def sweep_ttl(self) -> int:
        """Delete expired store items based on TTL."""
        with self._cursor() as cur:
            cur.execute(
                f"""
                DELETE FROM {self.table_names["store"]}
                WHERE expires_at IS NOT NULL AND expires_at < CURRENT_TIMESTAMP
                """
            )
            deleted_count = cur.rowcount
            cur.connection.commit()
            return deleted_count

    def start_ttl_sweeper(
        self, sweep_interval_minutes: int | None = None
    ) -> concurrent.futures.Future[None]:
        """Periodically delete expired store items based on TTL."""
        if not self.ttl_config:
            future: concurrent.futures.Future[None] = concurrent.futures.Future()
            future.set_result(None)
            return future

        if self._ttl_sweeper_thread and self._ttl_sweeper_thread.is_alive():
            logger.info("TTL sweeper thread is already running")
            # Return a future that can be used to cancel the existing thread
            future = concurrent.futures.Future()
            future.add_done_callback(
                lambda f: self._ttl_stop_event.set() if f.cancelled() else None
            )
            return future

        self._ttl_stop_event.clear()

        interval = float(
            sweep_interval_minutes or self.ttl_config.get("sweep_interval_minutes") or 5
        )

        future = concurrent.futures.Future()

        def _sweep_loop() -> None:
            try:
                while not self._ttl_stop_event.is_set():
                    if self._ttl_stop_event.wait(interval * 60):
                        break

                    try:
                        expired_items = self.sweep_ttl()
                        if expired_items > 0:
                            logger.info(f"Store swept {expired_items} expired items")
                    except Exception as exc:
                        logger.exception(
                            "Store TTL sweep iteration failed", exc_info=exc
                        )
                future.set_result(None)
            except Exception as exc:
                future.set_exception(exc)

        thread = threading.Thread(target=_sweep_loop, daemon=True, name="ttl-sweeper")
        self._ttl_sweeper_thread = thread
        thread.start()

        future.add_done_callback(
            lambda f: self._ttl_stop_event.set() if f.cancelled() else None
        )
        return future

    def stop_ttl_sweeper(self, timeout: float | None = None) -> bool:
        """Stop the TTL sweeper thread if it's running."""
        if not self._ttl_sweeper_thread or not self._ttl_sweeper_thread.is_alive():
            return True

        logger.info("Stopping TTL sweeper thread")
        self._ttl_stop_event.set()
        self._ttl_sweeper_thread.join(timeout)
        success = not self._ttl_sweeper_thread.is_alive()

        if success:
            self._ttl_sweeper_thread = None
            logger.info("TTL sweeper thread stopped")
        else:
            logger.warning("Timed out waiting for TTL sweeper thread to stop")

        return success

    def teardown(self) -> None:
        """Clean up all tables and configurations associated with this store.

        This method will:
        1. Stop the TTL sweeper if running
        2. Delete the store's entry from store_configs table
        3. Drop all tables created by this store instance

        Warning:
            This is a destructive operation that will permanently delete all data
            stored in this instance's tables.
        """
        # Stop TTL sweeper if it's running
        if hasattr(self, "_ttl_sweeper_thread") and self._ttl_sweeper_thread:
            self.stop_ttl_sweeper(timeout=1.0)

        with self._cursor() as cur:
            try:
                # Delete from store_configs table
                if hasattr(self, "table_suffix") and self.table_suffix:
                    cur.execute(
                        """
                        DELETE FROM store_configs 
                        WHERE table_suffix = :1
                    """,
                        (self.table_suffix,),
                    )
                    cur.connection.commit()
                    logger.info(
                        f"Removed configuration for table suffix '{self.table_suffix}'"
                    )
            except oracledb.DatabaseError as e:
                # store_configs table might not exist
                logger.debug(f"Could not delete from store_configs: {e}")

            # Drop tables in reverse order of dependencies
            # First drop vector tables (they have foreign keys to store table)
            tables_to_drop = []
            if hasattr(self, "table_names"):
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
                    cur.execute(f"DROP TABLE {table} CASCADE CONSTRAINTS PURGE")
                    cur.connection.commit()
                    logger.info(f"Dropped table {table}")
                except oracledb.DatabaseError as e:
                    # Table might not exist, which is fine
                    logger.debug(f"Could not drop table {table}: {e}")

    def __del__(self) -> None:
        """Ensure the TTL sweeper thread is stopped when the object is garbage collected."""
        if hasattr(self, "_ttl_stop_event") and hasattr(self, "_ttl_sweeper_thread"):
            self.stop_ttl_sweeper(timeout=0.1)

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        """Async version of batch operations."""
        return await asyncio.get_running_loop().run_in_executor(None, self.batch, ops)


# Private utilities

_DEFAULT_INDEX_CONFIG = HNSWConfig(
    type="hnsw",
    distance_metric="COSINE",
)


def _get_distance_metric(store: BaseOracleStore) -> str:
    """Get Oracle distance metric from config."""
    if not store.index_config:
        return "COSINE"

    index_config = store.index_config.get("index_type", _DEFAULT_INDEX_CONFIG)
    metric = index_config.get("distance_metric", "COSINE")
    return metric.upper()


def get_distance_operator(store: BaseOracleStore) -> tuple[str, str]:
    """Get the distance operator and score expression based on config."""
    if not store.index_config:
        raise ValueError(
            "Embedding configuration is required for vector operations "
            f"(for semantic search). "
            f"Please provide an Embeddings when initializing the {store.__class__.__name__}."
        )

    index_config = store.index_config.get("index_type", _DEFAULT_INDEX_CONFIG)
    distance_type = index_config.get("distance_metric", "COSINE").upper()

    if distance_type == "COSINE":
        # For cosine: convert distance to similarity (1 - distance)
        return (
            "VECTOR_DISTANCE(sv.embedding, :embedding_vector, COSINE)",
            "1 - uniq.neg_score",
        )
    if distance_type in _VALID_DISTANCE_METRICS:
        return (
            f"VECTOR_DISTANCE(sv.embedding, :embedding_vector, {distance_type})",
            "-uniq.neg_score",
        )
    raise ValueError(f"Unsupported distance metric: {distance_type}")


# No need for db embeddings to be different, still langchain embeddings
def _ensure_index_config(
    index_config: OracleIndexConfig,
) -> tuple[Embeddings | str | None, OracleIndexConfig]:
    """Ensure index configuration is properly set up."""
    index_config = index_config.copy()
    tokenized: list[tuple[str, Literal["$"] | list[str]]] = []
    tot = 0
    fields = index_config.get("fields") or ["$"]
    if isinstance(fields, str):
        fields = [fields]
    if not isinstance(fields, list):
        raise ValueError(f"Text fields must be a list or a string. Got {fields}")
    for p in fields:
        if p == "$":
            tokenized.append((p, "$"))
            tot += 1
        else:
            toks = tokenize_path(p)
            tokenized.append((p, toks))
            tot += len(toks)
    index_config["__tokenized_fields"] = tokenized
    index_config["__estimated_num_vectors"] = tot
    embeddings = ensure_embeddings(
        index_config.get("embed"),
    )
    _validate_index_config(index_config)
    return embeddings, index_config


# Helper functions for Oracle vector index creation


def _get_organization_clause(store: BaseOracleStore) -> str:
    """Generate the ORGANIZATION clause for vector index creation."""
    if not store.index_config or "index_type" not in store.index_config:
        return "INMEMORY NEIGHBOR GRAPH"

    index_type = store.index_config["index_type"]
    if isinstance(index_type, dict) and index_type.get("type") == "hnsw":
        return "INMEMORY NEIGHBOR GRAPH"
    elif isinstance(index_type, dict) and index_type.get("type") == "ivf":
        return "NEIGHBOR PARTITIONS"
    else:
        return "INMEMORY NEIGHBOR GRAPH"


def _get_target_accuracy_clause(store: BaseOracleStore) -> str:
    """Generate the TARGET ACCURACY clause for vector index creation."""
    if not store.index_config:
        return ""

    accuracy = store.index_config.get("accuracy", None)
    if isinstance(accuracy, int):
        return f"WITH TARGET ACCURACY {accuracy}"
    elif not accuracy:
        return ""

    raise ValueError("Accuracy must be int")


def _get_parameters_clause(store: BaseOracleStore) -> str:
    """Generate the PARAMETERS clause for vector index creation."""
    if not store.index_config or "index_type" not in store.index_config:
        return ""

    index_config = store.index_config["index_type"]
    if isinstance(index_config, dict):
        if index_config.get("type") == "hnsw":
            params = "PARAMETERS (type HNSW"
            neighbors = index_config.get("neighbors", None)
            if neighbors is not None:
                params += f", neighbors {neighbors}"
            efconstruction = index_config.get("efconstruction", None)
            if efconstruction is not None:
                params += f", efconstruction {efconstruction}"
            return params + ")"

        elif index_config.get("type") == "ivf":
            params = "PARAMETERS (type IVF"

            neighbor_partitions = index_config.get("neighbor_partitions", None)
            if neighbor_partitions is not None:
                params += f", neighbor partitions {neighbor_partitions}"

            samples_per_partition = index_config.get("samples_per_partition", None)
            if samples_per_partition is not None:
                params += f", samples_per_partition {samples_per_partition}"

            min_vectors_per_partition = index_config.get(
                "min_vectors_per_partition", None
            )
            if min_vectors_per_partition is not None:
                params += f", min_vectors_per_partition {min_vectors_per_partition}"

            return params + ")"

    raise ValueError("Index config not correct")


# A placeholder object for vector embeddings in queries
PLACEHOLDER = object()
