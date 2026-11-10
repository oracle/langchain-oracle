# Copyright (c) 2024, 2026, Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""
oraclevs.py

Provides integration between Oracle Vector Database and
LangChain for vector storage and search.
"""

from __future__ import annotations

import array
import hashlib
import inspect
import json
import logging
import re
import uuid
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

from numpy.typing import NDArray

if TYPE_CHECKING:
    from oracledb import (
        AsyncConnection,
        AsyncConnectionPool,
        Connection,
        ConnectionPool,
    )

import numpy as np
import oracledb
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_core.vectorstores.utils import maximal_marginal_relevance

from .._annotations import (
    TableType,
    abackfill_table_annotation,
    acreate_table_with_annotations,
    backfill_table_annotation,
    create_table_with_annotations,
)
from .._telemetry import tag_sql
from ..embeddings import OracleEmbeddings
from .utils import (
    DistanceStrategy,
    _aclear_session_proxy,
    _aget_connection,
    _ahandle_exceptions,
    _aindex_exists,
    _atable_exists,
    _clear_session_proxy,
    _get_connection,
    _get_index_name,
    _handle_exceptions,
    _index_exists,
    _quote_indentifier,
    _table_exists,
    adrop_table_purge,  # noqa: F401
    drop_table_purge,  # noqa: F401
    output_type_string_handler,
)

logger = logging.getLogger(__name__)

INTERNAL_ID_KEY = "__orcl_internal_doc_id"

_SIMPLE_IDENTIFIER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_$#]*$")


LOGICAL_MAP = {
    "$and": (" AND ", "({0})"),
    "$or": (" OR ", "({0})"),
    "$nor": (" OR ", "( NOT ({0}) )"),
}

COMPARISON_MAP = {
    "$exists": "",
    "$eq": "@ == {0}",
    "$ne": "@ != {0}",
    "$gt": "@ > {0}",
    "$lt": "@ < {0}",
    "$gte": "@ >= {0}",
    "$lte": "@ <= {0}",
    "$between": "",
    "$startsWith": "@  starts with {0}",
    "$hasSubstring": "@  has substring {0}",
    "$instr": "@  has substring {0}",
    "$regex": "@  like_regex {0}",
    "$like": "@  like {0}",
    "$in": "",
    "$nin": "",
    "$all": "",
    "$not": "",
}

# operations that may need negation
NOT_OPERS = ["$nin", "$not", "$exists"]


def _comparison_operand(value: Any, bind_variables: List[str]) -> tuple[str, str, str]:
    if isinstance(value, bool):
        # SQL BOOLEAN passing values can retain a preceding True in cached
        # statements. These two fixed JSON literals avoid that problem while
        # preserving the implicit boolean conversion of the SQL contract.
        return "@.boolean()", json.dumps(value), ""
    index = len(bind_variables)
    bind_variables.append(value)
    return "@", f"$val{index}", f':value{index} as "val{index}"'


def _passing_clause(passings: str) -> str:
    return " PASSING " + passings if passings else ""


def _get_comparison_string(
    oper: str, value: Any, bind_variables: List[str]
) -> tuple[str, str]:
    if oper not in COMPARISON_MAP:
        raise ValueError(f"Invalid operator: {oper}")

    # usual two sided operator case
    if COMPARISON_MAP[oper] != "":
        item, operand, passing = _comparison_operand(value, bind_variables)
        return COMPARISON_MAP[oper].replace("@", item).format(operand), passing

    # between - needs two bindings
    elif oper == "$between":
        if not isinstance(value, List) or len(value) != 2:
            raise ValueError(
                f"Invalid value for $between: {value}. "
                "It must be a list containing exactly 2 elements."
            )

        min_val, max_val = value
        if min_val is None and max_val is None:
            raise ValueError("At least one bound in $between must be non-null.")

        conditions = []
        passings = []
        if min_val is not None:
            item, operand, passing = _comparison_operand(min_val, bind_variables)
            conditions.append(f"{item} >= {operand}")
            if passing:
                passings.append(passing)

        if max_val is not None:
            item, operand, passing = _comparison_operand(max_val, bind_variables)
            conditions.append(f"{item} <= {operand}")
            if passing:
                passings.append(passing)

        passing_bind = ",".join(passings)

        return " && ".join(conditions), passing_bind

    # in/nin/all needs N bindings
    elif oper in ["$in", "$nin", "$all"]:
        if not isinstance(value, List):
            raise ValueError(
                f"Invalid value for $in: {value}. It must be a non-empty list."
            )

        value_binds = []
        passings = []
        comparisons = []
        for val in value:
            item, operand, passing = _comparison_operand(val, bind_variables)
            value_binds.append(operand)
            comparisons.append(f"{item} == {operand}")
            if passing:
                passings.append(passing)

        passing_bind = ",".join(passings)
        condition = ""

        if any(isinstance(val, bool) for val in value) and (
            oper == "$all" or all(isinstance(val, bool) for val in value)
        ):
            separator = " && " if oper == "$all" else " || "
            condition = "(" + separator.join(comparisons) + ")"
        elif oper == "$all":
            condition = "@ == " + " && @ == ".join(value_binds)

        else:
            value_bind = ",".join(value_binds)
            condition = f"@ in ({value_bind})"

        return condition, passing_bind

    else:
        raise ValueError(f"Invalid operator: {oper}. ")


def _validate_metadata_key(metadata_key: str) -> None:
    # Allow letters, digits, underscore, dot, brackets, comma, *, space (for 'to')
    pattern = re.compile(r"[a-zA-Z0-9_\.\[\],\s\*]*")

    if not pattern.fullmatch(metadata_key):
        raise ValueError(
            f"Invalid metadata key '{metadata_key}'. "
            "Only letters, numbers, underscores, nesting via '.', "
            "and array wildcards '[*]' are allowed."
        )


def _validate_int_param(
    config: dict[str, Any],
    key: str,
    min_value: int,
    max_value: Optional[int] = None,
) -> None:
    if key not in config:
        return

    value = config[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer.")
    if value < min_value:
        raise ValueError(f"{key} must be at least {min_value}.")
    if max_value is not None and value > max_value:
        raise ValueError(f"{key} must be at most {max_value}.")


def _validate_index_type(config: dict[str, Any], expected_type: str) -> None:
    if "idx_type" not in config:
        return

    idx_type = config["idx_type"]
    if not isinstance(idx_type, str) or idx_type.upper() != expected_type:
        raise ValueError(f"idx_type must be {expected_type}.")
    config["idx_type"] = expected_type


def _validate_vector_index_common(config: dict[str, Any]) -> None:
    config["idx_name"] = _quote_indentifier(config["idx_name"])
    _validate_int_param(config, "accuracy", 1, 100)
    _validate_int_param(config, "parallel", 1)


def _generate_condition(
    metadata_key: str, value: Any, bind_variables: List[str]
) -> str:
    # single check inside a JSON_EXISTS
    SINGLE_MASK = (
        "JSON_EXISTS(metadata, '$.{key}?(@ {oper} $val)' "
        'PASSING {value_bind} AS "val")'
    )
    # combined checks with multiple operators and passing values
    MULTIPLE_MASK = "JSON_EXISTS(metadata, '$.{key}?({filters})'{passes})"

    _validate_metadata_key(metadata_key)

    if not isinstance(value, (dict, list, tuple)):
        # scalar-equality Clause
        if isinstance(value, bool):
            condition, _ = _get_comparison_string("$eq", value, bind_variables)
            return MULTIPLE_MASK.format(key=metadata_key, filters=condition, passes="")
        bind = f":value{len(bind_variables)}"
        bind_variables.append(value)
        return SINGLE_MASK.format(key=metadata_key, oper="==", value_bind=bind)

    elif isinstance(value, dict):
        # all values are filters
        result: str
        passings: str

        # comparison operator keys
        if all(value_key.startswith("$") for value_key in value.keys()):
            not_dict = {}

            passing_values = []
            comparison_values = []

            for k, v in value.items():
                # if need to negate, cannot combine in single JSON_EXISTS
                if (
                    k in NOT_OPERS
                    or (k == "$eq" and isinstance(v, (list, dict)))
                    or (k == "$ne" and isinstance(v, (list, dict)))
                ):
                    not_dict[k] = v
                    continue

                result, passings = _get_comparison_string(k, v, bind_variables)

                comparison_values.append(result)
                passing_values.append(passings)

            # combine all operators in a single JSON_EXISTS
            all_conditions = []
            if len(comparison_values) != 0:
                all_conditions.append(
                    MULTIPLE_MASK.format(
                        key=metadata_key,
                        filters=" && ".join(comparison_values),
                        passes=_passing_clause(
                            " , ".join(p for p in passing_values if p)
                        ),
                    )
                )

            # handle negated filters one by one, one JSON_EXISTS for each
            for k, v in not_dict.items():
                if k == "$not":
                    condition = _generate_condition(metadata_key, v, bind_variables)
                    all_conditions.append(f"NOT ({condition})")

                elif k == "$exists":
                    if not isinstance(v, bool):
                        raise ValueError(
                            f"Invalid value for $exists: {value}. "
                            "It must be a boolean (true or false)."
                        )

                    if v:
                        all_conditions.append(
                            f"JSON_EXISTS(metadata, '$.{metadata_key}')"
                        )
                    else:
                        all_conditions.append(
                            f"NOT (JSON_EXISTS(metadata, '$.{metadata_key}'))"
                        )

                elif k == "$nin":  # for now only $nin
                    result, passings = _get_comparison_string(k, v, bind_variables)

                    all_conditions.append(
                        " NOT "
                        + MULTIPLE_MASK.format(
                            key=metadata_key,
                            filters=result,
                            passes=_passing_clause(passings),
                        )
                    )

                elif k == "$eq":
                    bind_l = len(bind_variables)
                    bind_variables.append(json.dumps(v))

                    all_conditions.append(
                        f"JSON_EQUAL("
                        f"    JSON_QUERY(metadata, '$.{metadata_key}' ),"
                        f"    JSON(:value{bind_l})"
                        ")"
                    )

                elif k == "$ne":
                    bind_l = len(bind_variables)
                    bind_variables.append(json.dumps(v))

                    all_conditions.append(
                        f"NOT (JSON_EQUAL("
                        f"    JSON_QUERY(metadata, '$.{metadata_key}' ),"
                        f"    JSON(:value{bind_l})"
                        "))"
                    )

            res = " AND ".join(all_conditions)

            if len(all_conditions) > 1:
                return "(" + res + ")"

            return res

        else:
            from ._filters import render_field

            def combine(operator: Optional[str], clauses: list[str]) -> str:
                result = " AND ".join(clauses)
                return "(" + result + ")" if len(clauses) > 1 else result

            return render_field(
                metadata_key,
                value,
                lambda key, operand: _generate_condition(key, operand, bind_variables),
                combine,
            )

    else:
        raise ValueError("Filter format is invalid.")


def _generate_where_clause(filter: dict, bind_variables: List[str]) -> str:
    from ._filters import render_filter

    def combine(operator: Optional[str], clauses: list[str]) -> str:
        if operator is not None:
            separator, template = LOGICAL_MAP[operator]
            return template.format(separator.join(clauses))
        result = " AND ".join(clauses)
        return "(" + result + ")" if len(clauses) > 1 else result

    return render_filter(
        filter,
        lambda key, value: _generate_condition(key, value, bind_variables),
        combine,
    )


def _get_distance_function(distance_strategy: DistanceStrategy) -> str:
    # dictionary to map distance strategies to their corresponding function
    # names
    distance_strategy2function = {
        DistanceStrategy.EUCLIDEAN_DISTANCE: "EUCLIDEAN",
        DistanceStrategy.DOT_PRODUCT: "DOT",
        DistanceStrategy.COSINE: "COSINE",
    }

    # attempt to return the corresponding distance function
    if distance_strategy in distance_strategy2function:
        return distance_strategy2function[distance_strategy]

    # if it's an unsupported distance strategy, raise an error
    raise ValueError(f"Unsupported distance strategy: {distance_strategy}")


def _get_table_dict(embedding_dim: int) -> Dict:
    cols_dict = {
        "id": "RAW(16) DEFAULT SYS_GUID() PRIMARY KEY",
        "text": "CLOB",
        "metadata": "JSON",
        "embedding": f"vector({embedding_dim}, FLOAT32)",
    }
    return cols_dict


def _create_table(
    connection: Connection,
    table_name: str,
    embedding_dim: int,
    table_type: TableType = "vector_store",
) -> None:
    cols_dict = _get_table_dict(embedding_dim)

    if not _table_exists(connection, table_name):
        with connection.cursor() as cursor:
            ddl_body = ", ".join(
                f"{col_name} {col_type}" for col_name, col_type in cols_dict.items()
            )
            ddl = f"CREATE TABLE {table_name} ({ddl_body})"
            create_table_with_annotations(cursor, ddl, table_type)
        logger.info(f"Table {table_name} created successfully...")
    else:
        logger.info(f"Table {table_name} already exists...")
        backfill_table_annotation(connection, table_name, table_type)


async def _acreate_table(
    connection: AsyncConnection,
    table_name: str,
    embedding_dim: int,
    table_type: TableType = "vector_store",
) -> None:
    cols_dict = _get_table_dict(embedding_dim)

    if not await _atable_exists(connection, table_name):
        with connection.cursor() as cursor:
            ddl_body = ", ".join(
                f"{col_name} {col_type}" for col_name, col_type in cols_dict.items()
            )
            ddl = f"CREATE TABLE {table_name} ({ddl_body})"
            await acreate_table_with_annotations(cursor, ddl, table_type)
        logger.info(f"Table {table_name} created successfully...")
    else:
        logger.info(f"Table {table_name} already exists...")
        await abackfill_table_annotation(connection, table_name, table_type)


@_handle_exceptions
def create_index(
    client: Union[Connection, ConnectionPool],
    vector_store: OracleVS,
    params: Optional[dict[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    """Ensure a vector index exists, or explicitly reconfigure a VecDB index.

    Existing VecDB indexes are unchanged when params is empty. Explicit params
    use LangChain HNSW/IVF options and rebuild an existing VecDB vector index.
    A VecDB operation returns its provider response; a no-op or SQL returns None.
    Use store.describe_vecdb_index() for readiness. Index maintenance requires
    an idle connection and caller-owned serialization.
    """
    if vector_store._vecdb is not None:
        return vector_store._vecdb.create_index(params)
    with _get_connection(client) as connection:
        if params:
            if "idx_name" in params:
                params["idx_name"] = _quote_indentifier(params["idx_name"])
            if params["idx_type"] == "HNSW":
                _create_hnsw_index(
                    connection,
                    vector_store.table_name,
                    vector_store.distance_strategy,
                    params,
                )
            elif params["idx_type"] == "IVF":
                _create_ivf_index(
                    connection,
                    vector_store.table_name,
                    vector_store.distance_strategy,
                    params,
                )
            else:
                _create_hnsw_index(
                    connection,
                    vector_store.table_name,
                    vector_store.distance_strategy,
                    params,
                )
        else:
            _create_hnsw_index(
                connection,
                vector_store.table_name,
                vector_store.distance_strategy,
                params,
            )
    return None


def _get_hnsw_index_ddl(
    table_name: str,
    distance_strategy: DistanceStrategy,
    params: Optional[dict[str, Any]] = None,
) -> Tuple[str, str]:
    defaults = {
        "idx_name": "HNSW",
        "idx_type": "HNSW",
        "neighbors": 32,
        "efConstruction": 200,
        "accuracy": 90,
        "parallel": 8,
    }

    if params:
        config = params.copy()
        # ensure compulsory parts are included
        for compulsory_key in ["idx_name", "parallel"]:
            if compulsory_key not in config:
                if compulsory_key == "idx_name":
                    config[compulsory_key] = _get_index_name(
                        str(defaults[compulsory_key])
                    )
                else:
                    config[compulsory_key] = defaults[compulsory_key]

        # validate keys in config against defaults
        for key in config:
            if key not in defaults:
                raise ValueError(f"Invalid parameter: {key}")
    else:
        config = defaults.copy()
        config["idx_name"] = _get_index_name(str(config["idx_name"]))

    if (
        "neighbors" in config or "efConstruction" in config
    ) and "idx_type" not in config:
        config["idx_type"] = defaults["idx_type"]
    _validate_index_type(config, "HNSW")
    _validate_int_param(config, "neighbors", 2, 2048)
    _validate_int_param(config, "efConstruction", 1, 65535)
    _validate_vector_index_common(config)

    # base SQL statement
    idx_name = config["idx_name"]
    base_sql = (
        f"create vector index {idx_name} on {table_name}(embedding) "
        "ORGANIZATION INMEMORY NEIGHBOR GRAPH"
    )

    # optional parts depending on parameters
    accuracy_part = " WITH TARGET ACCURACY {accuracy}" if ("accuracy" in config) else ""
    distance_part = f" DISTANCE {_get_distance_function(distance_strategy)}"

    parameters_part = ""
    if "neighbors" in config and "efConstruction" in config:
        parameters_part = (
            " parameters (type {idx_type}, neighbors {neighbors}, "
            "efConstruction {efConstruction})"
        )
    elif "neighbors" in config and "efConstruction" not in config:
        config["efConstruction"] = defaults["efConstruction"]
        parameters_part = (
            " parameters (type {idx_type}, neighbors {neighbors}, "
            "efConstruction {efConstruction})"
        )
    elif "neighbors" not in config and "efConstruction" in config:
        config["neighbors"] = defaults["neighbors"]
        parameters_part = (
            " parameters (type {idx_type}, neighbors {neighbors}, "
            "efConstruction {efConstruction})"
        )

    # always included part for parallel
    parallel_part = " parallel {parallel}"

    # combine all parts
    ddl_assembly = (
        base_sql + accuracy_part + distance_part + parameters_part + parallel_part
    )
    # format the SQL with values from the params dictionary
    ddl = ddl_assembly.format(**config)

    return idx_name, ddl


@_handle_exceptions
def _create_hnsw_index(
    connection: Connection,
    table_name: str,
    distance_strategy: DistanceStrategy,
    params: Optional[dict[str, Any]] = None,
) -> None:
    idx_name, ddl = _get_hnsw_index_ddl(table_name, distance_strategy, params)

    # check if the index exists
    if not _index_exists(connection, idx_name, table_name):
        with connection.cursor() as cursor:
            cursor.execute(ddl)
            logger.info(f"Index {idx_name} created successfully...")
    else:
        logger.info(f"Index {idx_name} already exists...")


def _get_ivf_index_ddl(
    table_name: str,
    distance_strategy: DistanceStrategy,
    params: Optional[dict[str, Any]] = None,
) -> Tuple[str, str]:
    # default configuration
    defaults = {
        "idx_name": "IVF",
        "idx_type": "IVF",
        "neighbor_part": 32,
        "accuracy": 90,
        "parallel": 8,
    }

    if params:
        config = params.copy()
        # ensure compulsory parts are included
        for compulsory_key in ["idx_name", "parallel"]:
            if compulsory_key not in config:
                if compulsory_key == "idx_name":
                    config[compulsory_key] = _get_index_name(
                        str(defaults[compulsory_key])
                    )
                else:
                    config[compulsory_key] = defaults[compulsory_key]

        # validate keys in config against defaults
        for key in config:
            if key not in defaults:
                raise ValueError(f"Invalid parameter: {key}")
    else:
        config = defaults.copy()
        config["idx_name"] = _get_index_name(str(config["idx_name"]))

    if "neighbor_part" in config and "idx_type" not in config:
        config["idx_type"] = defaults["idx_type"]
    _validate_index_type(config, "IVF")
    _validate_int_param(config, "neighbor_part", 1, 10000000)
    _validate_vector_index_common(config)

    # base SQL statement
    idx_name = config["idx_name"]
    base_sql = (
        f"CREATE VECTOR INDEX {idx_name} ON {table_name}(embedding) "
        "ORGANIZATION NEIGHBOR PARTITIONS"
    )

    # optional parts depending on parameters
    accuracy_part = " WITH TARGET ACCURACY {accuracy}" if ("accuracy" in config) else ""
    distance_part = f" DISTANCE {_get_distance_function(distance_strategy)}"

    parameters_part = ""
    if "idx_type" in config and "neighbor_part" in config:
        parameters_part = (
            f" PARAMETERS (type {config['idx_type']}, "
            f"neighbor partitions {config['neighbor_part']})"
        )

    # always included part for parallel
    parallel_part = f" PARALLEL {config['parallel']}"

    # combine all parts
    ddl_assembly = (
        base_sql + accuracy_part + distance_part + parameters_part + parallel_part
    )
    # format the SQL with values from the params dictionary
    ddl = ddl_assembly.format(**config)

    return idx_name, ddl


@_handle_exceptions
def _create_ivf_index(
    connection: Connection,
    table_name: str,
    distance_strategy: DistanceStrategy,
    params: Optional[dict[str, Any]] = None,
) -> None:
    idx_name, ddl = _get_ivf_index_ddl(table_name, distance_strategy, params)

    # check if the index exists
    if not _index_exists(connection, idx_name, table_name):
        with connection.cursor() as cursor:
            cursor.execute(ddl)
        logger.info(f"Index {idx_name} created successfully...")
    else:
        logger.info(f"Index {idx_name} already exists...")


@_ahandle_exceptions
async def acreate_index(
    client: Union[AsyncConnection, AsyncConnectionPool],
    vector_store: OracleVS,
    params: Optional[dict[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    """Ensure a vector index exists, or explicitly reconfigure a VecDB index.

    Existing VecDB indexes are unchanged when params is empty. Explicit params
    use LangChain HNSW/IVF options and rebuild an existing VecDB vector index.
    A VecDB operation returns its provider response; a no-op or SQL returns None.
    Use store.describe_vecdb_index() for readiness. Index maintenance requires
    an idle connection and caller-owned serialization.
    """
    if vector_store._vecdb is not None:
        return await vector_store._vecdb.run_async("create_index", params)
    async with _aget_connection(client) as connection:
        if params:
            if "idx_name" in params:
                params["idx_name"] = _quote_indentifier(params["idx_name"])
            if params["idx_type"] == "HNSW":
                await _acreate_hnsw_index(
                    connection,
                    vector_store.table_name,
                    vector_store.distance_strategy,
                    params,
                )
            elif params["idx_type"] == "IVF":
                await _acreate_ivf_index(
                    connection,
                    vector_store.table_name,
                    vector_store.distance_strategy,
                    params,
                )
            else:
                await _acreate_hnsw_index(
                    connection,
                    vector_store.table_name,
                    vector_store.distance_strategy,
                    params,
                )

        else:
            await _acreate_hnsw_index(
                connection,
                vector_store.table_name,
                vector_store.distance_strategy,
                params,
            )

    return None


async def _acreate_hnsw_index(
    connection: AsyncConnection,
    table_name: str,
    distance_strategy: DistanceStrategy,
    params: Optional[dict[str, Any]] = None,
) -> None:
    idx_name, ddl = _get_hnsw_index_ddl(table_name, distance_strategy, params)

    # check if the index exists
    if not await _aindex_exists(connection, idx_name, table_name):
        with connection.cursor() as cursor:
            await cursor.execute(ddl)
            logger.info(f"Index {idx_name} created successfully...")
    else:
        logger.info(f"Index {idx_name} already exists...")


async def _acreate_ivf_index(
    connection: AsyncConnection,
    table_name: str,
    distance_strategy: DistanceStrategy,
    params: Optional[dict[str, Any]] = None,
) -> None:
    idx_name, ddl = _get_ivf_index_ddl(table_name, distance_strategy, params)

    # check if the index exists
    if not await _aindex_exists(connection, idx_name, table_name):
        with connection.cursor() as cursor:
            await cursor.execute(ddl)
        logger.info(f"Index {idx_name} created successfully...")
    else:
        logger.info(f"Index {idx_name} already exists...")


@_handle_exceptions
def drop_index_if_exists(
    client: Union[Connection, ConnectionPool], index_name: str
) -> None:
    """Drop an index if it exists.

    Args:
        client: The OracleDB connection object.
        index_name: The name of the index to drop.

    Raises:
        RuntimeError: If an error occurs while dropping the index.
    """
    with _get_connection(client) as connection:
        index_name = _quote_indentifier(index_name)
        if _index_exists(connection, index_name):
            drop_query = f"DROP INDEX {index_name}"
            with connection.cursor() as cursor:
                cursor.execute(drop_query)
                logger.info(f"Index {index_name} has been dropped.")
        else:
            logger.exception(f"Index {index_name} does not exist.")
    return


@_ahandle_exceptions
async def adrop_index_if_exists(
    client: Union[AsyncConnection, AsyncConnectionPool], index_name: str
) -> None:
    """Drop an index if it exists.

    Args:
        client: The OracleDB connection object.
        index_name: The name of the index to drop.

    Raises:
        RuntimeError: If an error occurs while dropping the index.
    """
    index_name = _quote_indentifier(index_name)

    async with _aget_connection(client) as connection:
        if await _aindex_exists(connection, index_name):
            drop_query = f"DROP INDEX {index_name}"
            with connection.cursor() as cursor:
                await cursor.execute(drop_query)
                logger.info(f"Index {index_name} has been dropped.")
        else:
            logger.exception(f"Index {index_name} does not exist.")


def get_processed_ids(
    texts: Optional[Iterable[str]] = None,
    metadatas: Optional[List[Dict[Any, Any]]] = None,
    ids: Optional[List[str]] = None,
) -> Tuple[List[str], List[str]]:
    original_ids = None
    if ids:
        # if ids are provided, hash them to maintain consistency
        original_ids = [_id if _id else str(uuid.uuid4()) for _id in ids]
    elif metadatas and all("id" in metadata for metadata in metadatas):
        # if no ids are provided but metadatas with ids are, generate
        # ids from metadatas
        original_ids = [
            str(metadata["id"]) if metadata["id"] else str(uuid.uuid4())
            for metadata in metadatas
        ]
    else:
        input_seq = texts or metadatas or []
        # generate new ids if none are provided
        original_ids = [
            str(uuid.uuid4()) for _ in input_seq
        ]  # uuid4 is more standard for random UUIDs

    processed_ids = [
        hashlib.sha256(_id.encode()).hexdigest()[:16].upper() for _id in original_ids
    ]

    return processed_ids, original_ids


def _get_delete_ddl(
    table_name: str, ids: Optional[List[str]] = None
) -> Tuple[str, Dict]:
    if ids is None:
        raise ValueError("No ids provided to delete.")

    # compute SHA-256 hashes of the ids and truncate them
    hashed_ids = get_processed_ids(ids=[_id for _id in ids if _id])[0]

    # constructing the SQL statement with individual placeholders
    placeholders = ", ".join([":id" + str(i + 1) for i in range(len(hashed_ids))])

    ddl = f"DELETE FROM {table_name} WHERE id IN ({placeholders})"

    # preparing bind variables
    bind_vars = {f"id{i}": hashed_id for i, hashed_id in enumerate(hashed_ids, start=1)}

    return ddl, bind_vars


def _get_by_ids_select(
    table_name: str, ids: Optional[Sequence[str]] = None
) -> Tuple[str, Dict]:
    if ids is None:
        raise ValueError("No ids provided to select.")

    # compute SHA-256 hashes of the ids and truncate them
    hashed_ids = get_processed_ids(ids=[_id for _id in ids if _id])[0]

    # constructing the SQL statement with individual placeholders
    placeholders = ", ".join([":id" + str(i + 1) for i in range(len(hashed_ids))])

    ddl = f"SELECT text, metadata FROM {table_name} WHERE id IN ({placeholders})"

    # preparing bind variables
    bind_vars = {f"id{i}": hashed_id for i, hashed_id in enumerate(hashed_ids, start=1)}

    return ddl, bind_vars


def mmr_from_docs_embeddings(
    docs_scores_embeddings: List[Tuple[Document, float, NDArray[np.float32]]],
    embedding: List[float],
    k: int = 4,
    lambda_mult: float = 0.5,
) -> List[Tuple[Document, float]]:
    # if you need to split documents and scores for processing (e.g.,
    # for MMR calculation)
    documents, scores, embeddings = (
        zip(*docs_scores_embeddings) if docs_scores_embeddings else ([], [], [])
    )

    # assume maximal_marginal_relevance method accepts embeddings and
    # scores, and returns indices of selected docs
    mmr_selected_indices = maximal_marginal_relevance(
        np.array(embedding, dtype=np.float32),
        list(embeddings),
        k=k,
        lambda_mult=lambda_mult,
    )

    # filter documents based on MMR-selected indices and map scores
    mmr_selected_documents_with_scores = [
        (documents[i], scores[i]) for i in mmr_selected_indices
    ]

    return mmr_selected_documents_with_scores


def _get_similarity_search_query(
    table_name: str,
    distance_strategy: DistanceStrategy,
    k: int,
    filter: Optional[dict] = None,
    return_embeddings: bool = False,
) -> Tuple[str, list[str]]:
    where_clause = ""
    bind_variables: list[str] = []
    if filter:
        where_clause = _generate_where_clause(filter, bind_variables)

    # `VECTOR_INDEX_TRANSFORM` keeps the vector index in the plan even when a
    # JSON Search Index is also defined on the table. Without the hint, when
    # both indexes exist the optimizer picks the JSON Search Index for the
    # `JSON_EXISTS` filter and skips the vector index entirely, which kills
    # similarity-search latency. See issue #130.
    query = f"""
    SELECT /*+ VECTOR_INDEX_TRANSFORM({table_name}) */
        text,
        metadata,
        vector_distance(embedding, :embedding,
        {_get_distance_function(distance_strategy)}) as distance
        {",embedding" if return_embeddings else ""}
    FROM {table_name}
    {f"WHERE {where_clause}" if filter else ""}
    ORDER BY distance
    FETCH APPROX FIRST {k} ROWS ONLY
    """

    return tag_sql(query, "vector"), bind_variables


def _read_similarity_output(
    results: List, has_similarity_score: bool = False, has_embeddings: bool = False
) -> List:
    """
    Reads the SELECT queries, expects the following order:
    result[0] - text
    result[1] - metadata
    result[2] - distance (optional)
    result[3] - embedding (optional)
    """
    docs: Any = []
    for row in results:
        text, metadata, *extras = row
        metadata = metadata or {}

        # Extract internal ID if present
        doc_id = metadata.pop(INTERNAL_ID_KEY, None)

        # Ensure text is string
        if not isinstance(text, str):
            text = str(text or "")

        doc = Document(
            page_content=text,
            metadata=metadata,
            **({"id": doc_id} if doc_id is not None else {}),
        )

        # Basic case (no extra fields)
        if not has_similarity_score and not has_embeddings:
            docs.append(doc)
            continue

        # Extended result (score, embedding)
        result_parts: list[Any] = [doc]

        if has_similarity_score:
            result_parts.append(extras[0])

        if has_embeddings:
            # assuming result[3] is already in the correct format;
            # adjust if necessary
            current_embedding = (
                np.array(extras[1], dtype=np.float32)
                if extras[1]
                else np.empty(0, dtype=np.float32)
            )
            result_parts.append(current_embedding)

        docs.append(tuple(result_parts))

    return docs


# SQL queries to insert data into tables.
# INSERT_QUERY is used when we do not wish to update the row when there is duplicate id.
# MERGE_QUERY is used when we wish to update the row when there is duplicate id.
# both expect values in the order (id, embedding, metadata, text)

INSERT_QUERY = (
    "INSERT INTO {table_name} (id, embedding, metadata, text) "
    "VALUES ({id_value}, {embedding_value}, {metadata_value}, {text_value})"
)
MERGE_QUERY = """
MERGE INTO {table_name} t
USING (
    SELECT {id_value} AS id, {embedding_value} AS embedding,
           {metadata_value} AS metadata, {text_value} AS text
    FROM dual
) s
ON (t.id = s.id)
WHEN MATCHED THEN
    UPDATE SET 
        t.embedding = s.embedding,
        t.metadata = s.metadata,
        t.text = s.text
WHEN NOT MATCHED THEN
    INSERT (id, embedding, metadata, text)
    VALUES (s.id, s.embedding, s.metadata, s.text)
"""


class OracleVS(VectorStore):
    """`OracleVS` vector store.

    To use, you should have both:
    - the ``oracledb`` python package installed
    - a connection string associated with a OracleDBCluster having deployed an
       Search index

    - `mutate_on_duplicate` controls what happens when a document with an
        existing ID is provided.
        If False (default), the existing document is not updated.
        If True, the document with the same ID will be updated.

    Example:
        .. code-block:: python

            from langchain_oracledb.vectorstores import OracleVS
            from langchain.embeddings.openai import OpenAIEmbeddings
            import os
            import oracledb

            with oracledb.connect(dsn=os.environ["ORACLE_DB_DSN"]) as connection:
                print("Database version:", connection.version)
                embeddings = OpenAIEmbeddings()
                query = ""
                vectors = OracleVS(connection, embeddings, table_name, query)
    """

    def _annotation_type(self) -> TableType:
        return "vector_store"

    _vecdb: Any = None
    client: Any
    backend: str = "sql"

    @_handle_exceptions
    def __init__(
        self,
        client: Any,
        embedding_function: Union[
            Callable[[str], List[float]],
            Embeddings,
        ],
        table_name: str,  # case sensitive
        distance_strategy: DistanceStrategy = DistanceStrategy.EUCLIDEAN_DISTANCE,
        query: Optional[str] = "What is a Oracle database",
        params: Optional[Dict[str, Any]] = None,
        mutate_on_duplicate: Optional[bool] = False,
        *,
        backend: str = "auto",
        vector_index_params: Optional[dict[str, Any]] = None,
    ):
        """
        Initialize the OracleVS store.
        For an async version, use OracleVS.acreate() instead.

        vector_index_params accepts LangChain HNSW/IVF options for new VecDB
        tables only. Reopening preserves existing indexes. SQL stores reject
        this VecDB-only option.

        ``backend="auto"`` preserves existing SQL tables and uses VecDB only for
        compatible layouts and databases. Native VecDB package calls require
        unqualified uppercase table names; auto uses SQL for a new name outside
        that contract, while explicit ``backend="vecdb"`` rejects it. Inspect
        ``store.backend`` for the selected backend. SDK clients must use ORDS;
        for native VecDB pass an oracledb connection or pool.
        REST SDK clients require ``mutate_on_duplicate=True``; otherwise
        construction raises ValueError before preparing the table.
        """
        from ._vecdb import (
            NativeTransport,
            RestTransport,
            VecDBBackend,
            is_sdk_client,
            native_selection,
            table_identifier,
        )

        if backend not in {"auto", "sql", "vecdb"}:
            raise ValueError("backend must be auto, sql, or vecdb")
        self._vecdb = None
        self.backend = "sql"
        if is_sdk_client(client):
            if backend == "sql":
                raise ValueError("The SQL backend requires an oracledb connection")
            transport = RestTransport(client)
            if not mutate_on_duplicate:
                raise ValueError(
                    "oracle-vecdb REST SDK does not support insert-only writes; "
                    "set mutate_on_duplicate=True to enable upserts "
                    "(existing IDs will be overwritten)"
                )
            self.client = client
            self.embedding_function = embedding_function
            self.table_name = table_identifier(table_name)[0]
            self.query = query
            self.distance_strategy = distance_strategy
            self.params = params
            self.mutate_on_duplicate: Optional[bool] = mutate_on_duplicate
            self._vecdb = VecDBBackend(
                transport,
                table_name,
                distance_strategy,
                table_type=self._annotation_type(),
                vector_index_params=vector_index_params,
                embedding=embedding_function,
                mutate_on_duplicate=mutate_on_duplicate,
            )
            self._vecdb.prepare()
            self.backend = "vecdb"
            return

        with _get_connection(client) as connection:
            selected, version = native_selection(connection, table_name, backend)
            if selected == "sql" and vector_index_params is not None:
                raise ValueError("vector_index_params requires a VecDB backend")
            self._initialize(
                connection,
                client,
                embedding_function,
                table_name,
                distance_strategy,
                query,
                params,
                mutate_on_duplicate,
            )

            if selected == "vecdb":
                self._vecdb = VecDBBackend(
                    NativeTransport(connection, version),
                    table_name,
                    distance_strategy,
                    table_type=self._annotation_type(),
                    vector_index_params=vector_index_params,
                    embedding=embedding_function,
                    mutate_on_duplicate=mutate_on_duplicate,
                )
                self._vecdb.prepare()
                self._vecdb.transport.client = client
                self.backend = "vecdb"
            else:
                embedding_dim = self.get_embedding_dimension()
                _create_table(
                    connection, self.table_name, embedding_dim, self._annotation_type()
                )

    @classmethod
    def from_vecdb_config(
        cls, config: Any, table_name: str, embedding: Embeddings, **kwargs: Any
    ) -> OracleVS:
        """Create a REST VecDB store; install ``langchain-oracledb[vecdb]``.

        ``mutate_on_duplicate`` defaults to True because REST supports upserts
        only: writes update documents with existing IDs. Explicit False raises
        ValueError during construction because insert-only writes are unsupported.

        Operations not exposed by the REST SDK raise an unsupported-operation error.
        SDK clients supplied directly to OracleVS remain caller-owned.
        """
        try:
            from importlib import import_module

            OracleVecDB = import_module("oracle_vecdb").OracleVecDB
        except ImportError as exc:
            raise ImportError(
                "REST VecDB requires Python 3.10-3.15 and langchain-oracledb[vecdb]"
            ) from exc
        kwargs.setdefault("mutate_on_duplicate", True)
        return cls(
            client=OracleVecDB(config),
            embedding_function=embedding,
            table_name=table_name,
            **kwargs,
        )

    @classmethod
    @_ahandle_exceptions
    async def acreate(
        cls,
        client: Any,
        embedding_function: Union[
            Callable[[str], List[float]],
            Embeddings,
        ],
        table_name: str,
        distance_strategy: DistanceStrategy = DistanceStrategy.EUCLIDEAN_DISTANCE,
        query: Optional[str] = "What is a Oracle database",
        params: Optional[Dict[str, Any]] = None,
        mutate_on_duplicate: Optional[bool] = False,
        *,
        backend: str = "auto",
        vector_index_params: Optional[dict[str, Any]] = None,
    ) -> OracleVS:
        """
        Initialize the OracleVS store with async connection.

        vector_index_params follows the constructor's creation-only VecDB
        contract and accepts the same LangChain HNSW/IVF options.

        Uses the constructor's naming/backend contract. Native VecDB requires an
        unqualified uppercase name; new unsupported names use SQL in auto mode.
        Inspect the returned store's ``backend`` to see the selected backend.
        """

        from ._vecdb import (
            AsyncNativeTransport,
            VecDBBackend,
            anative_selection,
            is_sdk_client,
        )

        if backend not in {"auto", "sql", "vecdb"}:
            raise ValueError("backend must be auto, sql, or vecdb")
        if is_sdk_client(client):
            import asyncio

            return await asyncio.to_thread(
                cls,
                client,
                embedding_function,
                table_name,
                distance_strategy,
                query,
                params,
                mutate_on_duplicate,
                backend=backend,
                vector_index_params=vector_index_params,
            )
        self = cls.__new__(cls)
        self._vecdb = None
        self.backend = "sql"

        async with _aget_connection(client) as connection:
            selected, version = await anative_selection(connection, table_name, backend)
            if selected == "sql" and vector_index_params is not None:
                raise ValueError("vector_index_params requires a VecDB backend")
            self._initialize(
                connection,
                client,
                embedding_function,
                table_name,
                distance_strategy,
                query,
                params,
                mutate_on_duplicate,
            )

            if selected == "vecdb":
                self._vecdb = VecDBBackend(
                    AsyncNativeTransport(connection, version),
                    table_name,
                    distance_strategy,
                    table_type=self._annotation_type(),
                    vector_index_params=vector_index_params,
                    embedding=embedding_function,
                    mutate_on_duplicate=mutate_on_duplicate,
                )
                await self._vecdb.run_async("prepare")
                self._vecdb.transport.client = client
                self.backend = "vecdb"
            else:
                embedding_dim = await self.aget_embedding_dimension()
                await _acreate_table(
                    connection, self.table_name, embedding_dim, self._annotation_type()
                )

        return self

    def _initialize(
        self,
        connection: Any,
        client: Union[Connection, ConnectionPool, AsyncConnection, AsyncConnectionPool],
        embedding_function: Union[
            Callable[[str], List[float]],
            Embeddings,
        ],
        table_name: str,
        distance_strategy: DistanceStrategy = DistanceStrategy.EUCLIDEAN_DISTANCE,
        query: Optional[str] = "What is a Oracle database",
        params: Optional[Dict[str, Any]] = None,
        mutate_on_duplicate: Optional[bool] = False,
    ) -> None:
        if not (hasattr(connection, "thin") and connection.thin):
            if oracledb.clientversion()[:2] < (23, 4):
                raise Exception(
                    f"Oracle DB client driver version {oracledb.clientversion()} not \
                    supported, must be >=23.4 for vector support"
                )

        db_version = tuple([int(v) for v in connection.version.split(".")])

        if db_version < (23, 4):
            raise Exception(
                f"Oracle DB version {oracledb.__version__} not supported, \
                must be >=23.4 for vector support"
            )

        # initialize with oracledb client.
        self.client = client
        # initialize with necessary components.
        if not isinstance(embedding_function, Embeddings):
            logger.warning(
                "`embedding_function` is expected to be an Embeddings "
                "object, support "
                "for passing in a function will soon be removed."
            )
        self.embedding_function = embedding_function
        self.query = query
        self.table_name = _quote_indentifier(table_name)
        self.distance_strategy = distance_strategy
        self.params = params
        self.mutate_on_duplicate = mutate_on_duplicate

    def describe_vecdb_index(
        self, *, kind: str = "vector", job_name: Optional[str] = None
    ) -> dict[str, Any]:
        """Read VecDB index readiness or an ORDS job response without polling.

        ``ready`` is True/False when known, otherwise None; ``details`` retains
        provider status. A completed job alone never establishes index readiness.
        ``kind="metadata"`` inspects native text/hybrid indexes. This is a snapshot,
        not a synchronization guarantee; serialize index maintenance externally.
        """
        if self._vecdb is None:
            raise ValueError("Index status requires a VecDB backend")
        return self._vecdb.describe_index(kind, job_name)

    async def adescribe_vecdb_index(
        self, *, kind: str = "vector", job_name: Optional[str] = None
    ) -> dict[str, Any]:
        """Async variant of describe_vecdb_index with the same readiness contract."""
        if self._vecdb is None:
            raise ValueError("Index status requires a VecDB backend")
        return await self._vecdb.run_async("describe_index", kind, job_name)

    @property
    def embeddings(self) -> Optional[Embeddings]:
        """
        A property that returns an Embeddings instance embedding_function
        is an instance of Embeddings, otherwise returns None.

        Returns:
            Optional[Embeddings]: The embedding function if it's an instance of
            Embeddings, otherwise None.
        """
        return (
            self.embedding_function
            if isinstance(self.embedding_function, Embeddings)
            else None
        )

    def get_embedding_dimension(self) -> int:
        # embed the single document by wrapping it in a list
        embedded_document = self._embed_documents(
            [self.query if self.query is not None else ""]
        )

        # get the first (and only) embedding's dimension
        return len(embedded_document[0])

    async def aget_embedding_dimension(self) -> int:
        # embed the single document by wrapping it in a list
        embedded_document = await self._aembed_documents(
            [self.query if self.query is not None else ""]
        )

        # get the first (and only) embedding's dimension
        return len(embedded_document[0])

    def _embed_documents(self, texts: List[str]) -> List[List[float]]:
        if isinstance(self.embedding_function, Embeddings):
            return self.embedding_function.embed_documents(texts)
        elif callable(self.embedding_function):
            return [self.embedding_function(text) for text in texts]
        else:
            raise TypeError(
                "The embedding_function is neither Embeddings nor callable."
            )

    async def _aembed_documents(self, texts: List[str]) -> List[List[float]]:
        if isinstance(self.embedding_function, Embeddings):
            return await self.embedding_function.aembed_documents(texts)
        elif inspect.isawaitable(self.embedding_function):
            return [await self.embedding_function(text) for text in texts]  # type: ignore[unreachable]
        elif callable(self.embedding_function):
            return [self.embedding_function(text) for text in texts]
        else:
            raise TypeError(
                "The embedding_function is neither Embeddings nor callable."
            )

    def _embed_query(self, text: str) -> List[float]:
        if isinstance(self.embedding_function, Embeddings):
            return self.embedding_function.embed_query(text)
        else:
            return self.embedding_function(text)

    async def _aembed_query(self, text: str) -> List[float]:
        if isinstance(self.embedding_function, Embeddings):
            return await self.embedding_function.aembed_query(text)
        elif inspect.isawaitable(self.embedding_function):
            return await self.embedding_function(text)  # type: ignore[unreachable]
        else:
            return self.embedding_function(text)

    @staticmethod
    def _prepare_texts_from_documents(
        documents: List[Document],
        text_splitter: Optional[Any] = None,
        add_chunk_metadata: bool = True,
    ) -> Tuple[List[str], List[dict], List[int]]:
        """Prepare texts and metadatas for insertion.

        If `text_splitter` is provided, each document is split and every chunk
        is returned as a separate text row.
        """
        if text_splitter is not None and not hasattr(text_splitter, "split_text"):
            raise ValueError(
                "text_splitter must provide a split_text(text: str) method."
            )

        texts: List[str] = []
        metadatas: List[dict] = []
        source_doc_indices: List[int] = []

        for doc_index, doc in enumerate(documents):
            base_metadata = dict(doc.metadata) if doc.metadata else {}

            if text_splitter is None:
                texts.append(doc.page_content)
                metadatas.append(base_metadata)
                source_doc_indices.append(doc_index)
                continue

            chunks = text_splitter.split_text(doc.page_content)
            for chunk_index, chunk in enumerate(chunks):
                chunk_metadata = dict(base_metadata)
                if add_chunk_metadata:
                    chunk_metadata["source_doc_index"] = doc_index
                    chunk_metadata["chunk_index"] = chunk_index
                texts.append(chunk)
                metadatas.append(chunk_metadata)
                source_doc_indices.append(doc_index)

        return texts, metadatas, source_doc_indices

    @_handle_exceptions
    def add_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        """Add documents to the vector store.

        Optional kwargs:
            text_splitter: splitter object with split_text(str) -> List[str]
            add_chunk_metadata: when splitting, adds source_doc_index/chunk_index
            ids: optional IDs aligned to the input documents
        """
        text_splitter = kwargs.pop("text_splitter", None)
        add_chunk_metadata = kwargs.pop("add_chunk_metadata", True)
        ids = kwargs.pop("ids", None)

        if ids is None:
            doc_ids = [doc.id for doc in documents]
            if any(doc_ids):
                ids = doc_ids

        if ids is not None and len(ids) != len(documents):
            raise ValueError(
                f"Length mismatch: 'ids' has {len(ids)} items, "
                f"but 'documents' has {len(documents)} items."
            )

        texts, metadatas, source_doc_indices = self._prepare_texts_from_documents(
            documents,
            text_splitter=text_splitter,
            add_chunk_metadata=add_chunk_metadata,
        )

        if ids is not None:
            if text_splitter is None:
                return self.add_texts(
                    texts=texts,
                    metadatas=metadatas,
                    ids=ids,
                    **kwargs,
                )

            id_counts: Dict[int, int] = {}
            chunk_ids: List[str] = []
            for doc_index in source_doc_indices:
                current_idx = id_counts.get(doc_index, 0)
                chunk_ids.append(f"{ids[doc_index]}#chunk-{current_idx}")
                id_counts[doc_index] = current_idx + 1

            return self.add_texts(
                texts=texts,
                metadatas=metadatas,
                ids=chunk_ids,
                **kwargs,
            )

        return self.add_texts(texts=texts, metadatas=metadatas, **kwargs)

    @_ahandle_exceptions
    async def aadd_documents(
        self, documents: List[Document], **kwargs: Any
    ) -> List[str]:
        """Async version of add_documents with optional chunking."""
        text_splitter = kwargs.pop("text_splitter", None)
        add_chunk_metadata = kwargs.pop("add_chunk_metadata", True)
        ids = kwargs.pop("ids", None)

        if ids is None:
            doc_ids = [doc.id for doc in documents]
            if any(doc_ids):
                ids = doc_ids

        if ids is not None and len(ids) != len(documents):
            raise ValueError(
                f"Length mismatch: 'ids' has {len(ids)} items, "
                f"but 'documents' has {len(documents)} items."
            )

        texts, metadatas, source_doc_indices = self._prepare_texts_from_documents(
            documents,
            text_splitter=text_splitter,
            add_chunk_metadata=add_chunk_metadata,
        )

        if ids is not None:
            if text_splitter is None:
                return await self.aadd_texts(
                    texts=texts, metadatas=metadatas, ids=ids, **kwargs
                )

            id_counts: Dict[int, int] = {}
            chunk_ids: List[str] = []
            for doc_index in source_doc_indices:
                current_idx = id_counts.get(doc_index, 0)
                chunk_ids.append(f"{ids[doc_index]}#chunk-{current_idx}")
                id_counts[doc_index] = current_idx + 1

            return await self.aadd_texts(
                texts=texts,
                metadatas=metadatas,
                ids=chunk_ids,
                **kwargs,
            )

        return await self.aadd_texts(texts=texts, metadatas=metadatas, **kwargs)

    @_handle_exceptions
    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[list[dict]] = None,
        *,
        ids: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> list[str]:
        """Add more texts to the vectorstore index.

        Duplicate id handling behavior is controlled by the `mutate_on_duplicate`
            parameter passed when creating an `OracleVS` instance:
            If False: Existing rows with the same id are left unchanged;
                duplicate rows are skipped and their ids are not returned
                (i.e., they are not reported as successfully inserted).
            If True: Existing rows with the same id are updated (upsert behavior);
                returned ids include successful inserts and updates.

        Args:
          texts: Iterable of strings to add to the vectorstore.
          metadatas: Optional list of metadatas associated with the texts.
          ids: Optional list of ids for the texts that are being added to
          the vector store.
          kwargs: vectorstore specific parameters

        Returns:
          List[str]: The ids successfully inserted (and, when mutate_on_duplicate=True,
            also those successfully updated).
        """
        texts = list(texts)
        if ids and len(ids) != len(texts):
            raise ValueError(
                f"Length mismatch: 'ids' has {len(ids)} items, "
                f"but 'texts' has {len(texts)} items."
            )

        processed_ids, original_ids = get_processed_ids(texts, metadatas, ids)

        if not metadatas:
            metadatas = [{} for _ in texts]
        else:
            if len(metadatas) != len(texts):
                raise ValueError(
                    f"Length mismatch: 'metadatas' has {len(metadatas)} items, "
                    f"but 'texts' has {len(texts)} items."
                )

        for metadata in metadatas:
            if INTERNAL_ID_KEY in metadata:
                raise ValueError(
                    f'Metadata key "{INTERNAL_ID_KEY}" is '
                    "reserved and cannot used in metadata."
                )

        if self._vecdb is not None:
            vectors: Optional[List[List[float]]] = None
            if not self._vecdb.managed_embeddings:
                vectors = self._embed_documents(texts) if texts else []
            return self._vecdb.add(
                texts,
                vectors,
                metadatas,
                original_ids,
                self.mutate_on_duplicate,
            )

        for i, _id in enumerate(original_ids):
            metadatas[i][INTERNAL_ID_KEY] = _id

        # with OracleEmbeddings, embeddings are generated in the database during insert;
        # they are not sent back to Python to be written again.
        docs: Any
        if not isinstance(self.embeddings, OracleEmbeddings):
            embeddings = self._embed_documents(texts)

            docs = [
                (
                    id_,
                    array.array("f", embedding),
                    metadata,
                    text,
                )
                for id_, embedding, metadata, text in zip(
                    processed_ids, embeddings, metadatas, texts
                )
            ]
        else:
            docs = []
            for _1, _2, _3 in zip(processed_ids, metadatas, texts):
                docs.append(
                    {"id": _1, "meta": _2, "text": _3, "param": self.embeddings.params}
                )

        with _get_connection(self.client) as connection:
            error_indices = []
            try:
                with connection.cursor() as cursor:
                    # self.mutate_on_duplicate controls how inserts handle existing IDs.
                    # If False:
                    #   uses INSERT_QUERY.
                    #   existing rows having the same ID as the inserted row are not
                    #       updated.
                    #   with batcherrors=True, duplicate rows are skipped and their IDs
                    #       are not included in the `add_texts` return value (i.e., not
                    #       reported as successfully inserted).
                    #
                    # If True:
                    #   uses MERGE_QUERY.
                    #   existing rows having the same ID as the inserted row are updated
                    #       with the new data ("upsert").
                    #   the ID is included in the `add_texts` return value,
                    #       indicating a successful insert/update.
                    selected_query = (
                        INSERT_QUERY if not self.mutate_on_duplicate else MERGE_QUERY
                    )
                    if not isinstance(self.embeddings, OracleEmbeddings):
                        # Large MERGE text must use LOB locators. Keep the
                        # array uniformly typed. Leave bound LOB lifetimes to
                        # the driver: MERGE may consume their locators, making
                        # an explicit close unsafe for subsequent executions.
                        if self.mutate_on_duplicate and any(
                            len(row[3].encode("utf-8")) > 32767 for row in docs
                        ):
                            for i, row in enumerate(docs):
                                lob = connection.createlob(oracledb.DB_TYPE_CLOB)
                                lob.write(row[3])
                                docs[i] = (*row[:3], lob)
                        cursor.setinputsizes(
                            None,
                            oracledb.DB_TYPE_VECTOR,
                            oracledb.DB_TYPE_JSON,
                            oracledb.DB_TYPE_CLOB,
                        )
                        cursor.executemany(
                            selected_query.format(
                                table_name=self.table_name,
                                id_value=":1",
                                embedding_value=":2",
                                metadata_value=":3",
                                text_value=":4",
                            ),
                            docs,
                            batcherrors=True,
                        )
                        batch_errors = cursor.getbatcherrors() or []

                    else:
                        proxy_was_set = False
                        try:
                            if self.embeddings.proxy:
                                cursor.execute(
                                    "begin utl_http.set_proxy(:proxy); end;",
                                    proxy=self.embeddings.proxy,
                                )
                                proxy_was_set = True

                            cursor.setinputsizes(
                                meta=oracledb.DB_TYPE_JSON, param=oracledb.DB_TYPE_JSON
                            )

                            cursor.executemany(
                                selected_query.format(
                                    table_name=self.table_name,
                                    id_value=":id",
                                    embedding_value=(
                                        "dbms_vector_chain.utl_to_embedding(:text,json(:param))"  # noqa: E501
                                    ),
                                    metadata_value=":meta",
                                    text_value=":text",
                                ),
                                docs,
                                batcherrors=True,
                            )
                            batch_errors = cursor.getbatcherrors() or []
                        except BaseException:
                            if proxy_was_set:
                                try:
                                    _clear_session_proxy(cursor)
                                except Exception:
                                    logger.exception(
                                        "Failed to clear Oracle session proxy after "
                                        "add_texts failed"
                                    )
                            raise
                        else:
                            if proxy_was_set:
                                try:
                                    _clear_session_proxy(cursor)
                                except Exception:
                                    logger.warning(
                                        "Failed to clear Oracle session proxy after "
                                        "add_texts succeeded",
                                        exc_info=True,
                                    )

                    for error in batch_errors:
                        error_indices.append(error.offset)
                        logger.warning(
                            "Could not insert row at offset %s due to error: %s",
                            error.offset,
                            error.message,
                        )

                    connection.commit()
            finally:
                # do not change the input dict list
                for i in range(len(original_ids)):
                    metadatas[i].pop(INTERNAL_ID_KEY, None)

        inserted_ids = [i for j, i in enumerate(original_ids) if j not in error_indices]
        return inserted_ids

    @_ahandle_exceptions
    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[list[dict]] = None,
        *,
        ids: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> list[str]:
        """Add more texts to the vectorstore index, async.

        Duplicate id handling behavior is controlled by the `mutate_on_duplicate`
            parameter passed when creating an `OracleVS` instance:
            If False: Existing rows with the same id are left unchanged;
                duplicate rows are skipped and their ids are not returned
                (i.e., they are not reported as successfully inserted).
            If True: Existing rows with the same id are updated (upsert behavior);
                returned ids include successful inserts and updates.

        Args:
          texts: Iterable of strings to add to the vectorstore.
          metadatas: Optional list of metadatas associated with the texts.
          ids: Optional list of ids for the texts that are being added to
          the vector store.
          kwargs: vectorstore specific parameters

        Returns:
          List[str]: The ids successfully inserted (and, when mutate_on_duplicate=True,
            also those successfully updated).
        """
        texts = list(texts)
        if ids and len(ids) != len(texts):
            raise ValueError(
                f"Length mismatch: 'ids' has {len(ids)} items, "
                f"but 'texts' has {len(texts)} items."
            )

        processed_ids, original_ids = get_processed_ids(texts, metadatas, ids)

        if not metadatas:
            metadatas = [{} for _ in texts]
        else:
            if len(metadatas) != len(texts):
                raise ValueError(
                    f"Length mismatch: 'metadatas' has {len(metadatas)} items, "
                    f"but 'texts' has {len(texts)} items."
                )

        for metadata in metadatas:
            if INTERNAL_ID_KEY in metadata:
                raise ValueError(
                    f'Metadata key "{INTERNAL_ID_KEY}" is '
                    "reserved and cannot used in metadata."
                )

        if self._vecdb is not None:
            vectors: Optional[List[List[float]]] = None
            if not self._vecdb.managed_embeddings:
                vectors = await self._aembed_documents(texts) if texts else []
            return await self._vecdb.run_async(
                "add", texts, vectors, metadatas, original_ids, self.mutate_on_duplicate
            )

        for i, _id in enumerate(original_ids):
            metadatas[i][INTERNAL_ID_KEY] = _id

        # with OracleEmbeddings, embeddings are generated in the database during insert;
        # they are not sent back to Python to be written again.
        docs: Any
        if not isinstance(self.embeddings, OracleEmbeddings):
            embeddings = await self._aembed_documents(texts)

            docs = [
                (
                    id_,
                    array.array("f", embedding),
                    metadata,
                    text,
                )
                for id_, embedding, metadata, text in zip(
                    processed_ids, embeddings, metadatas, texts
                )
            ]
        else:
            docs = []
            for _1, _2, _3 in zip(processed_ids, metadatas, texts):
                docs.append(
                    {"id": _1, "meta": _2, "text": _3, "param": self.embeddings.params}
                )

        async with _aget_connection(self.client) as connection:
            if connection is None:
                raise ValueError("Failed to acquire a connection.")
            error_indices = []
            try:
                with connection.cursor() as cursor:
                    # self.mutate_on_duplicate controls how inserts handle existing IDs,
                    # behavior is identical to the synchronous `add_texts` method.
                    selected_query = (
                        INSERT_QUERY if not self.mutate_on_duplicate else MERGE_QUERY
                    )
                    if not isinstance(self.embeddings, OracleEmbeddings):
                        # Large MERGE text must use LOB locators. Keep the
                        # array uniformly typed. Leave bound LOB lifetimes to
                        # the driver: MERGE may consume their locators, making
                        # an explicit close unsafe for subsequent executions.
                        if self.mutate_on_duplicate and any(
                            len(row[3].encode("utf-8")) > 32767 for row in docs
                        ):
                            for i, row in enumerate(docs):
                                lob = await connection.createlob(oracledb.DB_TYPE_CLOB)
                                await lob.write(row[3])
                                docs[i] = (*row[:3], lob)
                        cursor.setinputsizes(
                            None,
                            oracledb.DB_TYPE_VECTOR,
                            oracledb.DB_TYPE_JSON,
                            oracledb.DB_TYPE_CLOB,
                        )
                        await cursor.executemany(
                            selected_query.format(
                                table_name=self.table_name,
                                id_value=":1",
                                embedding_value=":2",
                                metadata_value=":3",
                                text_value=":4",
                            ),
                            docs,
                            batcherrors=True,
                        )
                        batch_errors = cursor.getbatcherrors() or []
                    else:
                        proxy_was_set = False
                        try:
                            if self.embeddings.proxy:
                                await cursor.execute(
                                    "begin utl_http.set_proxy(:proxy); end;",
                                    proxy=self.embeddings.proxy,
                                )
                                proxy_was_set = True

                            cursor.setinputsizes(
                                meta=oracledb.DB_TYPE_JSON, param=oracledb.DB_TYPE_JSON
                            )

                            await cursor.executemany(
                                selected_query.format(
                                    table_name=self.table_name,
                                    id_value=":id",
                                    embedding_value=(
                                        "dbms_vector_chain.utl_to_embedding(:text,json(:param))"  # noqa: E501
                                    ),
                                    metadata_value=":meta",
                                    text_value=":text",
                                ),
                                docs,
                                batcherrors=True,
                            )
                            batch_errors = cursor.getbatcherrors() or []
                        except BaseException:
                            if proxy_was_set:
                                try:
                                    await _aclear_session_proxy(cursor)
                                except Exception:
                                    logger.exception(
                                        "Failed to clear Oracle session proxy after "
                                        "aadd_texts failed"
                                    )
                            raise
                        else:
                            if proxy_was_set:
                                try:
                                    await _aclear_session_proxy(cursor)
                                except Exception:
                                    logger.warning(
                                        "Failed to clear Oracle session proxy after "
                                        "aadd_texts succeeded",
                                        exc_info=True,
                                    )

                    for error in batch_errors:
                        error_indices.append(error.offset)
                        logger.warning(
                            "Could not insert row at offset %s due to error: %s",
                            error.offset,
                            error.message,
                        )

                    await connection.commit()
            finally:
                for i in range(len(original_ids)):
                    metadatas[i].pop(INTERNAL_ID_KEY, None)

            inserted_ids = [
                i for j, i in enumerate(original_ids) if j not in error_indices
            ]

        return inserted_ids

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        *,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to query."""
        embedding: List[float] = self._embed_query(query)

        documents = self.similarity_search_by_vector(
            embedding=embedding, k=k, filter=filter, **kwargs
        )
        return documents

    async def asimilarity_search(
        self,
        query: str,
        k: int = 4,
        *,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to query."""
        embedding: List[float] = await self._aembed_query(query)

        documents = await self.asimilarity_search_by_vector(
            embedding=embedding, k=k, filter=filter, **kwargs
        )
        return documents

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        *,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Document]:
        docs_and_scores = self.similarity_search_by_vector_with_relevance_scores(
            embedding=embedding, k=k, filter=filter, **kwargs
        )
        return [doc for doc, _ in docs_and_scores]

    async def asimilarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        *,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Document]:
        docs_and_scores = await self.asimilarity_search_by_vector_with_relevance_scores(
            embedding=embedding, k=k, filter=filter, **kwargs
        )
        return [doc for doc, _ in docs_and_scores]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        *,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query."""
        embedding: List[float] = self._embed_query(query)
        docs_and_scores = self.similarity_search_by_vector_with_relevance_scores(
            embedding=embedding, k=k, filter=filter, **kwargs
        )
        return docs_and_scores

    async def asimilarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        *,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query."""
        embedding: List[float] = await self._aembed_query(query)
        docs_and_scores = await self.asimilarity_search_by_vector_with_relevance_scores(
            embedding=embedding, k=k, filter=filter, **kwargs
        )
        return docs_and_scores

    @_handle_exceptions
    def similarity_search_by_vector_with_relevance_scores(
        self,
        embedding: List[float],
        k: int = 4,
        *,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        db_filter = kwargs.get("db_filter", None)
        if db_filter:
            if filter:
                raise ValueError(
                    "Specify only one of 'filter' or 'db_filter'; they are equivalent."
                )

            filter = db_filter

        if self._vecdb is not None:
            return self._vecdb.search(embedding, k, filter, **kwargs)
        docs_and_scores = []

        embedding_arr: Any = array.array("f", embedding)

        query, bind_variables = _get_similarity_search_query(
            self.table_name,
            self.distance_strategy,
            k,
            filter,
            return_embeddings=False,
        )

        # execute the query
        with _get_connection(self.client) as connection:
            with connection.cursor() as cursor:
                cursor.outputtypehandler = output_type_string_handler
                params = {"embedding": embedding_arr}
                for i, value in enumerate(bind_variables):
                    params[f"value{i}"] = value

                cursor.execute(query, **params)
                results = cursor.fetchall()

                docs_and_scores = _read_similarity_output(
                    results, has_similarity_score=True
                )

        return docs_and_scores

    @_ahandle_exceptions
    async def asimilarity_search_by_vector_with_relevance_scores(
        self,
        embedding: List[float],
        k: int = 4,
        *,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        db_filter = kwargs.get("db_filter", None)
        if db_filter:
            if filter:
                raise ValueError(
                    "Specify only one of 'filter' or 'db_filter'; they are equivalent."
                )

            filter = db_filter

        if self._vecdb is not None:
            return await self._vecdb.run_async("search", embedding, k, filter, **kwargs)
        embedding_arr: Any = array.array("f", embedding)

        query, bind_variables = _get_similarity_search_query(
            self.table_name,
            self.distance_strategy,
            k,
            filter,
            return_embeddings=False,
        )

        async with _aget_connection(self.client) as connection:
            # execute the query
            with connection.cursor() as cursor:
                cursor.outputtypehandler = output_type_string_handler
                params = {"embedding": embedding_arr}
                for i, value in enumerate(bind_variables):
                    params[f"value{i}"] = value

                await cursor.execute(query, **params)
                results = await cursor.fetchall()

                docs_and_scores = _read_similarity_output(
                    results, has_similarity_score=True
                )

        return docs_and_scores

    @_handle_exceptions
    def similarity_search_by_vector_returning_embeddings(
        self,
        embedding: List[float],
        k: int = 4,
        *,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float, NDArray[np.float32]]]:
        db_filter = kwargs.get("db_filter", None)
        if db_filter:
            if filter:
                raise ValueError(
                    "Specify only one of 'filter' or 'db_filter'; they are equivalent."
                )

            filter = db_filter

        if self._vecdb is not None:
            return self._vecdb.search(embedding, k, filter, True, **kwargs)
        embedding_arr: Any = array.array("f", embedding)

        documents = []

        query, bind_variables = _get_similarity_search_query(
            self.table_name,
            self.distance_strategy,
            k,
            filter,
            return_embeddings=True,
        )

        # execute the query
        with _get_connection(self.client) as connection:
            with connection.cursor() as cursor:
                cursor.outputtypehandler = output_type_string_handler
                params = {"embedding": embedding_arr}
                for i, value in enumerate(bind_variables):
                    params[f"value{i}"] = value

                cursor.execute(query, **params)
                results = cursor.fetchall()

                documents = _read_similarity_output(
                    results, has_similarity_score=True, has_embeddings=True
                )

        return documents

    @_ahandle_exceptions
    async def asimilarity_search_by_vector_returning_embeddings(
        self,
        embedding: List[float],
        k: int = 4,
        *,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float, NDArray[np.float32]]]:
        db_filter = kwargs.get("db_filter", None)
        if db_filter:
            if filter:
                raise ValueError(
                    "Specify only one of 'filter' or 'db_filter'; they are equivalent."
                )

            filter = db_filter

        if self._vecdb is not None:
            return await self._vecdb.run_async(
                "search", embedding, k, filter, True, **kwargs
            )
        embedding_arr: Any = array.array("f", embedding)

        query, bind_variables = _get_similarity_search_query(
            self.table_name,
            self.distance_strategy,
            k,
            filter,
            return_embeddings=True,
        )

        async with _aget_connection(self.client) as connection:
            # execute the query
            with connection.cursor() as cursor:
                cursor.outputtypehandler = output_type_string_handler
                params = {"embedding": embedding_arr}
                for i, value in enumerate(bind_variables):
                    params[f"value{i}"] = value

                await cursor.execute(query, **params)
                results = await cursor.fetchall()
                documents = _read_similarity_output(
                    results, has_similarity_score=True, has_embeddings=True
                )

        return documents

    def max_marginal_relevance_search_with_score_by_vector(
        self,
        embedding: List[float],
        *,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs and their similarity scores selected using the
        maximal marginal
            relevance.

        Maximal marginal relevance optimizes for similarity to query AND
        diversity
        among selected documents.

        Args:
          self: An instance of the class
          embedding: Embedding to look up documents similar to.
          k: Number of Documents to return. Defaults to 4.
          fetch_k: Number of Documents to fetch before filtering to
                   pass to MMR algorithm.
          filter: (Optional[dict]): Filter by metadata.
                                                                Defaults to None.
          lambda_mult: Number between 0 and 1 that determines the degree
                       of diversity among the results with 0 corresponding
                       to maximum diversity and 1 to minimum diversity.
                       Defaults to 0.5.
        Returns:
            List of Documents and similarity scores selected by maximal
            marginal
            relevance and score for each.
        """

        # fetch documents and their scores
        docs_scores_embeddings = self.similarity_search_by_vector_returning_embeddings(
            embedding, fetch_k, filter=filter, **kwargs
        )
        # assuming documents_with_scores is a list of tuples (Document, score)
        mmr_selected_documents_with_scores = mmr_from_docs_embeddings(
            docs_scores_embeddings, embedding, k, lambda_mult
        )

        return mmr_selected_documents_with_scores

    async def amax_marginal_relevance_search_with_score_by_vector(
        self,
        embedding: List[float],
        *,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs and their similarity scores selected using the
        maximal marginal
            relevance.

        Maximal marginal relevance optimizes for similarity to query AND
        diversity
        among selected documents.

        Args:
          self: An instance of the class
          embedding: Embedding to look up documents similar to.
          k: Number of Documents to return. Defaults to 4.
          fetch_k: Number of Documents to fetch before filtering to
                   pass to MMR algorithm.
          filter: (Optional[dict]): Filter by metadata.
                                                                Defaults to None.
          lambda_mult: Number between 0 and 1 that determines the degree
                       of diversity among the results with 0 corresponding
                       to maximum diversity and 1 to minimum diversity.
                       Defaults to 0.5.
        Returns:
            List of Documents and similarity scores selected by maximal
            marginal
            relevance and score for each.
        """

        # fetch documents and their scores
        docs_scores_embeddings = (
            await self.asimilarity_search_by_vector_returning_embeddings(
                embedding, fetch_k, filter=filter, **kwargs
            )
        )
        # assuming documents_with_scores is a list of tuples (Document, score)
        mmr_selected_documents_with_scores = mmr_from_docs_embeddings(
            docs_scores_embeddings, embedding, k, lambda_mult
        )

        return mmr_selected_documents_with_scores

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND
        diversity
        among selected documents.

        Args:
          self: An instance of the class
          embedding: Embedding to look up documents similar to.
          k: Number of Documents to return. Defaults to 4.
          fetch_k: Number of Documents to fetch to pass to MMR algorithm.
          lambda_mult: Number between 0 and 1 that determines the degree
                       of diversity among the results with 0 corresponding
                       to maximum diversity and 1 to minimum diversity.
                       Defaults to 0.5.
          filter: (Optional[dict]): Filter by metadata.
                                                                Defaults to None.
          **kwargs: Any
        Returns:
          List of Documents selected by maximal marginal relevance.
        """
        docs_and_scores = self.max_marginal_relevance_search_with_score_by_vector(
            embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
            **kwargs,
        )
        return [doc for doc, _ in docs_and_scores]

    async def amax_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND
        diversity
        among selected documents.

        Args:
          self: An instance of the class
          embedding: Embedding to look up documents similar to.
          k: Number of Documents to return. Defaults to 4.
          fetch_k: Number of Documents to fetch to pass to MMR algorithm.
          lambda_mult: Number between 0 and 1 that determines the degree
                       of diversity among the results with 0 corresponding
                       to maximum diversity and 1 to minimum diversity.
                       Defaults to 0.5.
          filter: (Optional[dict]): Filter by metadata.
                                                                Defaults to None.
          **kwargs: Any
        Returns:
          List of Documents selected by maximal marginal relevance.
        """
        docs_and_scores = (
            await self.amax_marginal_relevance_search_with_score_by_vector(
                embedding,
                k=k,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult,
                filter=filter,
                **kwargs,
            )
        )
        return [doc for doc, _ in docs_and_scores]

    @_handle_exceptions
    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND
        diversity
        among selected documents.

        Args:
          self: An instance of the class
          query: Text to look up documents similar to.
          k: Number of Documents to return. Defaults to 4.
          fetch_k: Number of Documents to fetch to pass to MMR algorithm.
          lambda_mult: Number between 0 and 1 that determines the degree
                       of diversity among the results with 0 corresponding
                       to maximum diversity and 1 to minimum diversity.
                       Defaults to 0.5.
          filter: (Optional[dict]): Filter by metadata.
                                                                Defaults to None.
          **kwargs
        Returns:
          List of Documents selected by maximal marginal relevance.

        `max_marginal_relevance_search` requires that `query` returns matched
        embeddings alongside the match documents.
        """
        embedding = self._embed_query(query)
        documents = self.max_marginal_relevance_search_by_vector(
            embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
            **kwargs,
        )
        return documents

    @_ahandle_exceptions
    async def amax_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND
        diversity
        among selected documents.

        Args:
          self: An instance of the class
          query: Text to look up documents similar to.
          k: Number of Documents to return. Defaults to 4.
          fetch_k: Number of Documents to fetch to pass to MMR algorithm.
          lambda_mult: Number between 0 and 1 that determines the degree
                       of diversity among the results with 0 corresponding
                       to maximum diversity and 1 to minimum diversity.
                       Defaults to 0.5.
          filter: (Optional[dict]): Filter by metadata.
                                                                Defaults to None.
          **kwargs
        Returns:
          List of Documents selected by maximal marginal relevance.

        `amax_marginal_relevance_search` requires that `query` returns matched
        embeddings alongside the match documents.
        """
        embedding = await self._aembed_query(query)
        documents = await self.amax_marginal_relevance_search_by_vector(
            embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
            **kwargs,
        )
        return documents

    @_handle_exceptions
    def get_by_ids(self, ids: Sequence[str], /) -> list[Document]:
        """Get documents by vector IDs.
        Args:
          self: An instance of the class
          ids: The ids of the documents to get.
          **kwargs
        """
        if self._vecdb is not None:
            return self._vecdb.get(list(ids))

        ddl, bind_vars = _get_by_ids_select(self.table_name, ids)

        with _get_connection(self.client) as connection:
            documents = []
            with connection.cursor() as cursor:
                cursor.outputtypehandler = output_type_string_handler
                cursor.execute(ddl, bind_vars)
                results = cursor.fetchall()

                documents = _read_similarity_output(results)

        return documents

    @_ahandle_exceptions
    async def aget_by_ids(self, ids: Sequence[str], /) -> list[Document]:
        """Get documents by vector IDs.
        Args:
          self: An instance of the class
          ids: The ids of the documents to get.
          **kwargs
        """
        if self._vecdb is not None:
            return await self._vecdb.run_async("get", list(ids))

        ddl, bind_vars = _get_by_ids_select(self.table_name, ids)

        async with _aget_connection(self.client) as connection:
            documents = []
            with connection.cursor() as cursor:
                cursor.outputtypehandler = output_type_string_handler
                await cursor.execute(ddl, bind_vars)
                results = await cursor.fetchall()

                documents = _read_similarity_output(results)

        return documents

    @_handle_exceptions
    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> None:
        """Delete by vector IDs.
        Args:
          self: An instance of the class
          ids: List of ids to delete.
          **kwargs
        """
        if not ids:
            raise ValueError("No ids provided to delete.")

        if self._vecdb is not None:
            return self._vecdb.delete(ids)

        ddl, bind_vars = _get_delete_ddl(self.table_name, ids)

        with _get_connection(self.client) as connection:
            with connection.cursor() as cursor:
                cursor.execute(ddl, bind_vars)
                connection.commit()

    @_ahandle_exceptions
    async def adelete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> None:
        """Delete by vector IDs.
        Args:
          self: An instance of the class
          ids: List of ids to delete.
          **kwargs
        """
        if not ids:
            raise ValueError("No ids provided to delete.")

        if self._vecdb is not None:
            return await self._vecdb.run_async("delete", ids)

        ddl, bind_vars = _get_delete_ddl(self.table_name, ids)

        async with _aget_connection(self.client) as connection:
            with connection.cursor() as cursor:
                await cursor.execute(ddl, bind_vars)
                await connection.commit()

    @classmethod
    def _from_texts_helper(
        cls: Type[OracleVS],
        **kwargs: Any,
    ) -> Tuple[
        Union[Connection, ConnectionPool, AsyncConnection, AsyncConnectionPool],
        str,
        DistanceStrategy,
        str,
        Dict,
    ]:
        client: Union[
            Connection, ConnectionPool, AsyncConnection, AsyncConnectionPool, None
        ] = kwargs.get("client", None)
        if client is None:
            raise ValueError("client parameter is required...")

        params = kwargs.get("params", {})

        table_name = str(kwargs.get("table_name", "langchain"))

        distance_strategy = kwargs.get(
            "distance_strategy", DistanceStrategy.EUCLIDEAN_DISTANCE
        )
        if not isinstance(distance_strategy, DistanceStrategy):
            try:
                # Accept an equivalent value (e.g. a DistanceStrategy from
                # another package or its string value) for backwards
                # compatibility now that the enum is vendored locally.
                distance_strategy = DistanceStrategy(distance_strategy)
            except (ValueError, TypeError):
                raise TypeError(
                    f"Expected DistanceStrategy got {type(distance_strategy).__name__} "
                )

        query = kwargs.get("query", "What is a Oracle database")

        return client, table_name, distance_strategy, query, params

    @classmethod
    @_handle_exceptions
    def from_texts(
        cls: Type[OracleVS],
        texts: Iterable[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> OracleVS:
        """Create and populate a store using the constructor's backend contract.

        With a native connection, the default lowercase ``langchain`` name uses
        SQL in auto mode. Native VecDB needs an unqualified uppercase name.
        ``backend="vecdb"`` rejects
        unsupported names; inspect ``store.backend`` after auto selection.
        """
        (
            client,
            table_name,
            distance_strategy,
            query,
            params,
        ) = OracleVS._from_texts_helper(**kwargs)

        vss = cls(
            client=client,
            embedding_function=embedding,
            table_name=table_name,
            distance_strategy=distance_strategy,
            query=query,
            params=params,
            backend=kwargs.get("backend", "auto"),
            vector_index_params=kwargs.get("vector_index_params"),
            mutate_on_duplicate=kwargs.get("mutate_on_duplicate", False),
        )

        vss.add_texts(texts=list(texts), metadatas=metadatas, ids=kwargs.get("ids"))
        return vss

    @classmethod
    @_ahandle_exceptions
    async def afrom_texts(
        cls: Type[OracleVS],
        texts: Iterable[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> OracleVS:
        """Create and populate a store using the constructor's backend contract.

        With a native connection, the default lowercase ``langchain`` name uses
        SQL in auto mode. Native VecDB needs an unqualified uppercase name.
        ``backend="vecdb"`` rejects
        unsupported names; inspect ``store.backend`` after auto selection.
        """
        (
            client,
            table_name,
            distance_strategy,
            query,
            params,
        ) = OracleVS._from_texts_helper(**kwargs)

        vss = await OracleVS.acreate(
            client=client,
            embedding_function=embedding,
            table_name=table_name,
            distance_strategy=distance_strategy,
            query=query,
            params=params,
            backend=kwargs.get("backend", "auto"),
            vector_index_params=kwargs.get("vector_index_params"),
            mutate_on_duplicate=kwargs.get("mutate_on_duplicate", False),
        )

        await vss.aadd_texts(
            texts=list(texts), metadatas=metadatas, ids=kwargs.get("ids")
        )
        return vss
