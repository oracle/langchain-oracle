# Copyright (c) 2024, 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""
oraclevs.py

Provides integration between Oracle Vector Database and
LangChain for vector storage and search.
"""

from __future__ import annotations

import array
import functools
import hashlib
import inspect
import json
import logging
import os
import re
import uuid
from collections.abc import Awaitable
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
    TypeVar,
    Union,
    cast,
)

from numpy.typing import NDArray

if TYPE_CHECKING:
    from oracledb import AsyncConnection, Connection

import numpy as np
import oracledb
from langchain_community.vectorstores.utils import (
    DistanceStrategy,
    maximal_marginal_relevance,
)
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from ..embeddings import OracleEmbeddings

logger = logging.getLogger(__name__)
log_level = os.getenv("LOG_LEVEL", "ERROR").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s - %(levelname)s - %(message)s",
)

INTERNAL_ID_KEY = "__orcl_internal_doc_id"

# define a type variable that can be any kind of function
T = TypeVar("T", bound=Callable[..., Any])

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


def _get_comparison_string(
    oper: str, value: Any, bind_variables: List[str]
) -> tuple[str, str]:
    if oper not in COMPARISON_MAP:
        raise ValueError(f"Invalid operator: {oper}")

    # usual two sided operator case
    if COMPARISON_MAP[oper] != "":
        bind_l = len(bind_variables)
        bind_variables.append(value)

        return (
            COMPARISON_MAP[oper].format(f"$val{bind_l}"),
            f':value{bind_l} as "val{bind_l}"',
        )

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
            bind_l = len(bind_variables)
            bind_variables.append(min_val)

            conditions.append(f"@ >= $val{bind_l}")
            passings.append(f':value{bind_l} as "val{bind_l}"')

        if max_val is not None:
            bind_l = len(bind_variables)
            bind_variables.append(max_val)

            conditions.append(f"@ <= $val{bind_l}")
            passings.append(f':value{bind_l} as "val{bind_l}"')

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
        for val in value:
            bind_l = len(bind_variables)
            bind_variables.append(val)

            value_binds.append(f"$val{bind_l}")
            passings.append(f':value{bind_l} as "val{bind_l}"')

        passing_bind = ",".join(passings)
        condition = ""

        if oper == "$all":
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


def _generate_condition(
    metadata_key: str, value: Any, bind_variables: List[str]
) -> str:
    # single check inside a JSON_EXISTS
    SINGLE_MASK = (
        "JSON_EXISTS(metadata, '$.{key}?(@ {oper} $val)' "
        'PASSING {value_bind} AS "val")'
    )
    # combined checks with multiple operators and passing values
    MULTIPLE_MASK = "JSON_EXISTS(metadata, '$.{key}?({filters})' PASSING {passes})"

    _validate_metadata_key(metadata_key)

    if not isinstance(value, (dict, list, tuple)):
        # scalar-equality Clause
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
                        passes=" , ".join(passing_values),
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
                            key=metadata_key, filters=result, passes=passings
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
            raise ValueError("Nested filters are not supported.")

    else:
        raise ValueError("Filter format is invalid.")


def _generate_where_clause(filter: dict, bind_variables: List[str]) -> str:
    if not isinstance(filter, dict):
        raise ValueError("Filter syntax is incorrect. Must be a dictionary.")

    all_conditions = []

    for key, value in filter.items():
        # must be a logical if on a high level
        if key.startswith("$"):
            if key not in LOGICAL_MAP.keys():
                raise ValueError(f"'{key}' is not a recognized logical operator.")

            filter_format = LOGICAL_MAP[key]

            if not isinstance(value, list):
                raise ValueError("Logical operators require an array of values.")

            combine_conditions = [
                _generate_where_clause(v, bind_variables) for v in value
            ]

            res = filter_format[1].format(filter_format[0].join(combine_conditions))

            all_conditions.append(res)

        else:
            # this is a metadata key - not an operator
            res = _generate_condition(key, value, bind_variables)
            all_conditions.append(res)

    # combine everything with AND
    res = " AND ".join(all_conditions)

    if len(all_conditions) > 1:
        res = "(" + res + ")"

    return res


def _get_connection(client: Any) -> Optional[Connection]:
    # check if ConnectionPool exists
    connection_pool_class = getattr(oracledb, "ConnectionPool", None)

    if isinstance(client, oracledb.Connection):
        return client
    elif connection_pool_class and isinstance(client, connection_pool_class):
        return client.acquire()
    else:
        valid_types = "oracledb.Connection"
        if connection_pool_class:
            valid_types += " or oracledb.ConnectionPool"
        raise TypeError(
            f"Expected client of type {valid_types}, got {type(client).__name__}"
        )


async def _aget_connection(client: Any) -> Optional[AsyncConnection]:
    # check if ConnectionPool exists
    connection_pool_class = getattr(oracledb, "AsyncConnectionPool", None)

    if isinstance(client, oracledb.AsyncConnection):
        return client
    elif connection_pool_class and isinstance(client, connection_pool_class):
        return await client.acquire()
    else:
        valid_types = "oracledb.AsyncConnection"
        if connection_pool_class:
            valid_types += " or oracledb.AsyncConnectionPool"
        raise TypeError(
            f"Expected client of type {valid_types}, got {type(client).__name__}"
        )


def _handle_exceptions(func: T) -> T:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except oracledb.Error as db_err:
            # Handle a known type of error (e.g., DB-related) specifically
            logger.exception("DB-related error occurred.")
            raise RuntimeError(
                "Failed due to a DB error: {}".format(db_err)
            ) from db_err
        except RuntimeError as runtime_err:
            # Handle a runtime error
            logger.exception("Runtime error occurred.")
            raise RuntimeError(
                "Failed due to a runtime error: {}".format(runtime_err)
            ) from runtime_err
        except ValueError as val_err:
            # Handle another known type of error specifically
            logger.exception("Validation error.")
            raise ValueError("Validation failed: {}".format(val_err)) from val_err
        except Exception as e:
            # Generic handler for all other exceptions
            logger.exception("An unexpected error occurred: {}".format(e))
            raise RuntimeError("Unexpected error: {}".format(e)) from e

    return cast(T, wrapper)


def _ahandle_exceptions(func: T) -> T:
    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return await func(*args, **kwargs)
        except oracledb.Error as db_err:
            # Handle a known type of error (e.g., DB-related) specifically
            logger.exception("DB-related error occurred.")
            raise RuntimeError(
                "Failed due to a DB error: {}".format(db_err)
            ) from db_err
        except RuntimeError as runtime_err:
            # Handle a runtime error
            logger.exception("Runtime error occurred.")
            raise RuntimeError(
                "Failed due to a runtime error: {}".format(runtime_err)
            ) from runtime_err
        except ValueError as val_err:
            # Handle another known type of error specifically
            logger.exception("Validation error.")
            raise ValueError("Validation failed: {}".format(val_err)) from val_err
        except Exception as e:
            # Generic handler for all other exceptions
            logger.exception("An unexpected error occurred: {}".format(e))
            raise RuntimeError("Unexpected error: {}".format(e)) from e

    return cast(T, wrapper)


def _table_exists(connection: Connection, table_name: str) -> bool:
    try:
        with connection.cursor() as cursor:
            cursor.execute(f"SELECT 1 FROM {table_name} WHERE ROWNUM < 1")
            return True
    except oracledb.DatabaseError as ex:
        err_obj = ex.args
        if err_obj[0].code == 942:
            return False
        raise


async def _atable_exists(connection: AsyncConnection, table_name: str) -> bool:
    try:
        with connection.cursor() as cursor:
            await cursor.execute(f"SELECT 1 FROM {table_name} WHERE ROWNUM < 1")
            return True
    except oracledb.DatabaseError as ex:
        err_obj = ex.args
        if err_obj[0].code == 942:
            return False
        raise


def _quote_indentifier(name: str) -> str:
    name = name.strip()
    reg = r'^(?:"[^"]+"|[^".]+)(?:\.(?:"[^"]+"|[^".]+))*$'
    pattern_validate = re.compile(reg)

    if not pattern_validate.match(name):
        raise ValueError(f"Identifier name {name} is not valid.")

    pattern_match = r'"([^"]+)"|([^".]+)'
    groups = re.findall(pattern_match, name)
    groups = [m[0] or m[1] for m in groups]
    groups = [f'"{g}"' for g in groups]

    return ".".join(groups)


@_handle_exceptions
def _index_exists(
    connection: Connection, index_name: str, table_name: Optional[str] = None
) -> bool:
    # check if the index exists
    query = f"""
        SELECT index_name 
        FROM all_indexes 
        WHERE index_name = :idx_name
        {"AND table_name = :table_name" if table_name else ""} 
        """

    # this is an internal method, index_name and table_name comes with double quotes
    index_name = index_name.replace('"', "")
    if table_name:
        table_name = table_name.replace('"', "")

    with connection.cursor() as cursor:
        # execute the query
        if table_name:
            cursor.execute(
                query,
                idx_name=index_name,
                table_name=table_name,
            )
        else:
            cursor.execute(query, idx_name=index_name)
        result = cursor.fetchone()

        # check if the index exists
    return result is not None


async def _aindex_exists(
    connection: AsyncConnection, index_name: str, table_name: Optional[str] = None
) -> bool:
    # check if the index exists
    query = f"""
        SELECT index_name,  table_name
        FROM all_indexes 
        WHERE index_name = :idx_name
        {"AND table_name = :table_name" if table_name else ""} 
        """

    # this is an internal method, index_name and table_name comes with double quotes
    index_name = index_name.replace('"', "")
    if table_name:
        table_name = table_name.replace('"', "")

    with connection.cursor() as cursor:
        # execute the query
        if table_name:
            await cursor.execute(
                query,
                idx_name=index_name,
                table_name=table_name,
            )
        else:
            await cursor.execute(query, idx_name=index_name)
        result = await cursor.fetchone()

        # check if the index exists
    return result is not None


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


def _get_index_name(base_name: str) -> str:
    unique_id = str(uuid.uuid4()).replace("-", "")
    return f'"{base_name}_{unique_id}"'


def _get_table_dict(embedding_dim: int) -> Dict:
    cols_dict = {
        "id": "RAW(16) DEFAULT SYS_GUID() PRIMARY KEY",
        "text": "CLOB",
        "metadata": "JSON",
        "embedding": f"vector({embedding_dim}, FLOAT32)",
    }
    return cols_dict


def _create_table(connection: Connection, table_name: str, embedding_dim: int) -> None:
    cols_dict = _get_table_dict(embedding_dim)

    if not _table_exists(connection, table_name):
        with connection.cursor() as cursor:
            ddl_body = ", ".join(
                f"{col_name} {col_type}" for col_name, col_type in cols_dict.items()
            )
            ddl = f"CREATE TABLE {table_name} ({ddl_body})"
            cursor.execute(ddl)
        logger.info(f"Table {table_name} created successfully...")
    else:
        logger.info(f"Table {table_name} already exists...")


async def _acreate_table(
    connection: AsyncConnection, table_name: str, embedding_dim: int
) -> None:
    cols_dict = _get_table_dict(embedding_dim)

    if not await _atable_exists(connection, table_name):
        with connection.cursor() as cursor:
            ddl_body = ", ".join(
                f"{col_name} {col_type}" for col_name, col_type in cols_dict.items()
            )
            ddl = f"CREATE TABLE {table_name} ({ddl_body})"
            await cursor.execute(ddl)
        logger.info(f"Table {table_name} created successfully...")
    else:
        logger.info(f"Table {table_name} already exists...")


@_handle_exceptions
def create_index(
    client: Any,
    vector_store: OracleVS,
    params: Optional[dict[str, Any]] = None,
) -> None:
    connection = _get_connection(client)
    if connection is None:
        raise ValueError("Failed to acquire a connection.")
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
            connection, vector_store.table_name, vector_store.distance_strategy, params
        )
    return


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
        config = defaults
        config["idx_name"] = _get_index_name(str(config["idx_name"]))

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
        config = defaults
        config["idx_name"] = _get_index_name(str(config["idx_name"]))

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
    client: Any,
    vector_store: OracleVS,
    params: Optional[dict[str, Any]] = None,
) -> None:
    async def context(connection: Any) -> None:
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

    await _handle_context(client, context)
    return


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
def drop_table_purge(client: Any, table_name: str) -> None:
    """Drop a table and purge it from the database.

    Args:
        client: oracledb connection object.
        table_name: The name of the table to drop.

    Raises:
        RuntimeError: If an error occurs while dropping the table.
    """
    connection = _get_connection(client)
    table_name = _quote_indentifier(table_name)
    if connection is None:
        raise ValueError("Failed to acquire a connection.")
    if _table_exists(connection, table_name):
        with connection.cursor() as cursor:
            ddl = f"DROP TABLE {table_name} PURGE"
            cursor.execute(ddl)
        logger.info(f"Table {table_name} dropped successfully...")
    else:
        logger.info(f"Table {table_name} not found...")
    return


@_ahandle_exceptions
async def adrop_table_purge(client: Any, table_name: str) -> None:
    """Drop a table and purge it from the database.

    Args:
        client: oracledb connection object.
        table_name: The name of the table to drop.

    Raises:
        RuntimeError: If an error occurs while dropping the table.
    """
    table_name = _quote_indentifier(table_name)

    async def context(connection: Any) -> None:
        if await _atable_exists(connection, table_name):
            with connection.cursor() as cursor:
                ddl = f"DROP TABLE {table_name} PURGE"
                await cursor.execute(ddl)
            logger.info(f"Table {table_name} dropped successfully...")
        else:
            logger.info(f"Table {table_name} not found...")

    await _handle_context(client, context)
    return


@_handle_exceptions
def drop_index_if_exists(client: Any, index_name: str) -> None:
    """Drop an index if it exists.

    Args:
        client: The OracleDB connection object.
        index_name: The name of the index to drop.

    Raises:
        RuntimeError: If an error occurs while dropping the index.
    """
    connection = _get_connection(client)
    index_name = _quote_indentifier(index_name)
    if connection is None:
        raise ValueError("Failed to acquire a connection.")
    if _index_exists(connection, index_name):
        drop_query = f"DROP INDEX {index_name}"
        with connection.cursor() as cursor:
            cursor.execute(drop_query)
            logger.info(f"Index {index_name} has been dropped.")
    else:
        logger.exception(f"Index {index_name} does not exist.")
    return


@_ahandle_exceptions
async def adrop_index_if_exists(client: Any, index_name: str) -> None:
    """Drop an index if it exists.

    Args:
        client: The OracleDB connection object.
        index_name: The name of the index to drop.

    Raises:
        RuntimeError: If an error occurs while dropping the index.
    """
    index_name = _quote_indentifier(index_name)

    async def context(connection: Any) -> None:
        if await _aindex_exists(connection, index_name):
            drop_query = f"DROP INDEX {index_name}"
            with connection.cursor() as cursor:
                await cursor.execute(drop_query)
                logger.info(f"Index {index_name} has been dropped.")
        else:
            logger.exception(f"Index {index_name} does not exist.")

    await _handle_context(client, context)
    return


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

    query = f"""
    SELECT 
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

    return query, bind_variables


async def _handle_context(
    client: Any,
    context: Callable[
        [Any],
        Awaitable[Any],
    ],
) -> Any:
    """
    AsyncConnectionPool connections are not released automatically,
    therefore needs to be used in a context manager
    """
    connection = await _aget_connection(client)
    if connection is None:
        raise ValueError("Failed to acquire a connection.")

    if connection._pool:
        async with connection as conn:
            return await context(conn)
    else:
        return await context(connection)


def output_type_string_handler(cursor: Any, metadata: Any) -> Any:
    if metadata.type_code is oracledb.DB_TYPE_CLOB:
        return cursor.var(oracledb.DB_TYPE_LONG, arraysize=cursor.arraysize)
    if metadata.type_code is oracledb.DB_TYPE_NCLOB:
        return cursor.var(oracledb.DB_TYPE_LONG_NVARCHAR, arraysize=cursor.arraysize)


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
    "INSERT INTO {table_name} (id, embedding, metadata, text) VALUES ({values})"
)
MERGE_QUERY = """
MERGE INTO {table_name} t
USING (
    VALUES ({values})
) s(id, embedding, metadata, text)
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
            import oracledb

            with oracledb.connect(user = user, password = pwd, dsn = dsn) as
            connection:
                print ("Database version:", connection.version)
                embeddings = OpenAIEmbeddings()
                query = ""
                vectors = OracleVS(connection, embeddings, table_name, query)
    """

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
    ):
        """
        Initialize the OracleVS store.
        For an async version, use OracleVS.acreate() instead.
        """
        connection = _get_connection(client)
        if connection is None:
            raise ValueError("Failed to acquire a connection.")

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

        embedding_dim = self.get_embedding_dimension()
        _create_table(connection, self.table_name, embedding_dim)

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
    ) -> OracleVS:
        """
        Initialize the OracleVS store with async connection.
        """

        self = cls.__new__(cls)

        async def context(connection: Any) -> None:
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

            embedding_dim = await self.aget_embedding_dimension()
            await _acreate_table(connection, self.table_name, embedding_dim)

        await _handle_context(client, context)

        return self

    def _initialize(
        self,
        connection: Any,
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
            return [await self.embedding_function(text) for text in texts]
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
            return await self.embedding_function(text)
        else:
            return self.embedding_function(text)

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

        for i, _id in enumerate(original_ids):
            if INTERNAL_ID_KEY in metadatas[i]:
                raise ValueError(
                    f'Metadata key "{INTERNAL_ID_KEY}" is '
                    "reserved and cannot used in metadata."
                )
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

        connection = _get_connection(self.client)
        if connection is None:
            raise ValueError("Failed to acquire a connection.")
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
                    cursor.setinputsizes(
                        None, oracledb.DB_TYPE_VECTOR, oracledb.DB_TYPE_JSON, None
                    )
                    cursor.executemany(
                        selected_query.format(
                            table_name=self.table_name, values=":1, :2, :3, :4"
                        ),
                        docs,
                        batcherrors=True,
                    )

                else:
                    if self.embeddings.proxy:
                        cursor.execute(
                            "begin utl_http.set_proxy(:proxy); end;",
                            proxy=self.embeddings.proxy,
                        )

                    cursor.setinputsizes(
                        meta=oracledb.DB_TYPE_JSON, param=oracledb.DB_TYPE_JSON
                    )

                    cursor.executemany(
                        selected_query.format(
                            table_name=self.table_name,
                            values=(
                                ":id, dbms_vector_chain.utl_to_embedding(:text,json(:param)), "  # noqa: E501
                                ":meta, :text"
                            ),
                        ),
                        docs,
                        batcherrors=True,
                    )

                for error in cursor.getbatcherrors():
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

        for i, _id in enumerate(original_ids):
            if INTERNAL_ID_KEY in metadatas[i]:
                raise ValueError(
                    f'Metadata key "{INTERNAL_ID_KEY}" is '
                    "reserved and cannot used in metadata."
                )
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

        async def context(connection: Any) -> List[str]:
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
                        cursor.setinputsizes(
                            None, oracledb.DB_TYPE_VECTOR, oracledb.DB_TYPE_JSON, None
                        )
                        await cursor.executemany(
                            selected_query.format(
                                table_name=self.table_name, values=":1, :2, :3, :4"
                            ),
                            docs,
                            batcherrors=True,
                        )
                    else:
                        if self.embeddings.proxy:
                            await cursor.execute(
                                "begin utl_http.set_proxy(:proxy); end;",
                                proxy=self.embeddings.proxy,
                            )

                        cursor.setinputsizes(
                            meta=oracledb.DB_TYPE_JSON, param=oracledb.DB_TYPE_JSON
                        )

                        await cursor.executemany(
                            selected_query.format(
                                table_name=self.table_name,
                                values=(
                                    ":id, dbms_vector_chain.utl_to_embedding(:text,json(:param)), "  # noqa: E501
                                    ":meta, :text"
                                ),
                            ),
                            docs,
                            batcherrors=True,
                        )

                    for error in cursor.getbatcherrors():
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

        inserted_ids = await _handle_context(self.client, context)

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
        docs_and_scores = []

        embedding_arr: Any = array.array("f", embedding)

        db_filter = kwargs.get("db_filter", None)
        if db_filter:
            if filter:
                raise ValueError(
                    "Specify only one of 'filter' or 'db_filter'; they are equivalent."
                )

            filter = db_filter

        query, bind_variables = _get_similarity_search_query(
            self.table_name,
            self.distance_strategy,
            k,
            filter,
            return_embeddings=False,
        )

        # execute the query
        connection = _get_connection(self.client)
        if connection is None:
            raise ValueError("Failed to acquire a connection.")
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
        embedding_arr: Any = array.array("f", embedding)

        db_filter = kwargs.get("db_filter", None)
        if db_filter:
            if filter:
                raise ValueError(
                    "Specify only one of 'filter' or 'db_filter'; they are equivalent."
                )

            filter = db_filter

        query, bind_variables = _get_similarity_search_query(
            self.table_name,
            self.distance_strategy,
            k,
            filter,
            return_embeddings=False,
        )

        async def context(connection: Any) -> List:
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

        return await _handle_context(self.client, context)

    @_handle_exceptions
    def similarity_search_by_vector_returning_embeddings(
        self,
        embedding: List[float],
        k: int = 4,
        *,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float, NDArray[np.float32]]]:
        embedding_arr: Any = array.array("f", embedding)

        documents = []

        db_filter = kwargs.get("db_filter", None)
        if db_filter:
            if filter:
                raise ValueError(
                    "Specify only one of 'filter' or 'db_filter'; they are equivalent."
                )

            filter = db_filter

        query, bind_variables = _get_similarity_search_query(
            self.table_name,
            self.distance_strategy,
            k,
            filter,
            return_embeddings=True,
        )

        # execute the query
        connection = _get_connection(self.client)
        if connection is None:
            raise ValueError("Failed to acquire a connection.")
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
        embedding_arr: Any = array.array("f", embedding)

        db_filter = kwargs.get("db_filter", None)
        if db_filter:
            if filter:
                raise ValueError(
                    "Specify only one of 'filter' or 'db_filter'; they are equivalent."
                )

            filter = db_filter

        query, bind_variables = _get_similarity_search_query(
            self.table_name,
            self.distance_strategy,
            k,
            filter,
            return_embeddings=True,
        )

        async def context(connection: Any) -> List:
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

        return await _handle_context(self.client, context)

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

        ddl, bind_vars = _get_by_ids_select(self.table_name, ids)

        connection = _get_connection(self.client)
        if connection is None:
            raise ValueError("Failed to acquire a connection.")

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

        ddl, bind_vars = _get_by_ids_select(self.table_name, ids)

        async def context(connection: Any) -> List:
            documents = []
            with connection.cursor() as cursor:
                cursor.outputtypehandler = output_type_string_handler
                await cursor.execute(ddl, bind_vars)
                results = await cursor.fetchall()

                documents = _read_similarity_output(results)

            return documents

        documents = await _handle_context(self.client, context)

        return documents

    @_handle_exceptions
    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> None:
        """Delete by vector IDs.
        Args:
          self: An instance of the class
          ids: List of ids to delete.
          **kwargs
        """

        ddl, bind_vars = _get_delete_ddl(self.table_name, ids)

        connection = _get_connection(self.client)
        if connection is None:
            raise ValueError("Failed to acquire a connection.")
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

        ddl, bind_vars = _get_delete_ddl(self.table_name, ids)

        async def context(connection: Any) -> None:
            with connection.cursor() as cursor:
                await cursor.execute(ddl, bind_vars)
                await connection.commit()

        await _handle_context(self.client, context)

    @classmethod
    def _from_texts_helper(
        cls: Type[OracleVS],
        **kwargs: Any,
    ) -> Tuple[Any, str, DistanceStrategy, str, Dict]:
        client: Any = kwargs.get("client", None)
        if client is None:
            raise ValueError("client parameter is required...")

        params = kwargs.get("params", {})

        table_name = str(kwargs.get("table_name", "langchain"))

        distance_strategy = cast(
            DistanceStrategy, kwargs.get("distance_strategy", None)
        )
        if not isinstance(distance_strategy, DistanceStrategy):
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
        )

        vss.add_texts(texts=list(texts), metadatas=metadatas)
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
        )

        await vss.aadd_texts(texts=list(texts), metadatas=metadatas)
        return vss
