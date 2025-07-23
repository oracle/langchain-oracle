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
import logging
import os
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
    Tuple,
    Type,
    TypedDict,
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

logger = logging.getLogger(__name__)
log_level = os.getenv("LOG_LEVEL", "ERROR").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# define a type variable that can be any kind of function
T = TypeVar("T", bound=Callable[..., Any])


class FilterCondition(TypedDict):
    key: str
    oper: str
    value: str


class FilterGroup(TypedDict, total=False):
    _and: Optional[List[Union["FilterCondition", "FilterGroup"]]]
    _or: Optional[List[Union["FilterCondition", "FilterGroup"]]]


def _convert_oper_to_sql(oper: str) -> str:
    oper_map = {"EQ": "==", "GT": ">", "LT": "<", "GTE": ">=", "LTE": "<="}
    if oper not in oper_map:
        raise ValueError("Filter operation {} not supported".format(oper))
    return oper_map.get(oper, "==")


def _generate_condition(condition: FilterCondition) -> str:
    key = condition["key"]
    oper = _convert_oper_to_sql(condition["oper"])
    value = condition["value"]
    if isinstance(value, str):
        value = f'"{value}"'
    return f"JSON_EXISTS(metadata, '$.{key}?(@ {oper} {value})')"


def _generate_where_clause(db_filter: Union[FilterCondition, FilterGroup]) -> str:
    if "key" in db_filter:  # identify as FilterCondition
        return _generate_condition(cast(FilterCondition, db_filter))

    if "_and" in db_filter and db_filter["_and"] is not None:
        and_conditions = [
            _generate_where_clause(cond)
            for cond in db_filter["_and"]
            if isinstance(cond, dict)
        ]
        return "(" + " AND ".join(and_conditions) + ")"

    if "_or" in db_filter and db_filter["_or"] is not None:
        or_conditions = [
            _generate_where_clause(cond)
            for cond in db_filter["_or"]
            if isinstance(cond, dict)
        ]
        return "(" + " OR ".join(or_conditions) + ")"

    raise ValueError(f"Invalid filter structure: {db_filter}")


def _get_connection(client: Any) -> Connection | None:
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


async def _aget_connection(client: Any) -> AsyncConnection | None:
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


def _normalize_oracle_identifier(name: str) -> str:
    name = name.strip()
    if name.startswith('"') and name.endswith('"'):
        return name
    else:
        return name.upper()


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

    with connection.cursor() as cursor:
        # execute the query
        if table_name:
            cursor.execute(
                query,
                idx_name=_normalize_oracle_identifier(index_name),
                table_name=_normalize_oracle_identifier(table_name),
            )
        else:
            cursor.execute(query, idx_name=_normalize_oracle_identifier(index_name))
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

    with connection.cursor() as cursor:
        # execute the query
        if table_name:
            await cursor.execute(
                query,
                idx_name=_normalize_oracle_identifier(index_name),
                table_name=_normalize_oracle_identifier(table_name),
            )
        else:
            await cursor.execute(
                query, idx_name=_normalize_oracle_identifier(index_name)
            )
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
    return f"{base_name}_{unique_id}"


def _get_table_dict(embedding_dim: int) -> Dict:
    cols_dict = {
        "id": "RAW(16) DEFAULT SYS_GUID() PRIMARY KEY",
        "text": "CLOB",
        "metadata": "JSON",
        "embedding": f"vector({embedding_dim}, FLOAT32)",
    }
    return cols_dict


@_handle_exceptions
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

    # base SQL statement
    idx_name = config["idx_name"]
    base_sql = (
        f"create vector index {idx_name} on {table_name}(embedding) "
        f"ORGANIZATION INMEMORY NEIGHBOR GRAPH"
    )

    # optional parts depending on parameters
    accuracy_part = " WITH TARGET ACCURACY {accuracy}" if ("accuracy" in config) else ""
    distance_part = f" DISTANCE {_get_distance_function(distance_strategy)}"

    parameters_part = ""
    if "neighbors" in config and "efConstruction" in config:
        parameters_part = (
            " parameters (type {idx_type}, neighbors {"
            "neighbors}, efConstruction {efConstruction})"
        )
    elif "neighbors" in config and "efConstruction" not in config:
        config["efConstruction"] = defaults["efConstruction"]
        parameters_part = (
            " parameters (type {idx_type}, neighbors {"
            "neighbors}, efConstruction {efConstruction})"
        )
    elif "neighbors" not in config and "efConstruction" in config:
        config["neighbors"] = defaults["neighbors"]
        parameters_part = (
            " parameters (type {idx_type}, neighbors {"
            "neighbors}, efConstruction {efConstruction})"
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

    # base SQL statement
    idx_name = config["idx_name"]
    base_sql = (
        f"CREATE VECTOR INDEX {idx_name} ON {table_name}(embedding) "
        f"ORGANIZATION NEIGHBOR PARTITIONS"
    )

    # optional parts depending on parameters
    accuracy_part = " WITH TARGET ACCURACY {accuracy}" if ("accuracy" in config) else ""
    distance_part = f" DISTANCE {_get_distance_function(distance_strategy)}"

    parameters_part = ""
    if "idx_type" in config and "neighbor_part" in config:
        parameters_part = (
            f" PARAMETERS (type {config['idx_type']}, neighbor"
            f" partitions {config['neighbor_part']})"
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
    texts: Iterable[str],
    metadatas: Optional[List[Dict[Any, Any]]] = None,
    ids: Optional[List[str]] = None,
) -> List[str]:
    if ids:
        # if ids are provided, hash them to maintain consistency
        processed_ids = [
            hashlib.sha256(_id.encode()).hexdigest()[:16].upper() for _id in ids
        ]
    elif metadatas and all("id" in metadata for metadata in metadatas):
        # if no ids are provided but metadatas with ids are, generate
        # ids from metadatas
        processed_ids = [
            hashlib.sha256(metadata["id"].encode()).hexdigest()[:16].upper()
            for metadata in metadatas
        ]
    else:
        # generate new ids if none are provided
        generated_ids = [
            str(uuid.uuid4()) for _ in texts
        ]  # uuid4 is more standard for random UUIDs
        processed_ids = [
            hashlib.sha256(_id.encode()).hexdigest()[:16].upper()
            for _id in generated_ids
        ]

    return processed_ids


def _get_delete_ddl(
    table_name: str, ids: Optional[List[str]] = None
) -> Tuple[str, Dict]:
    if ids is None:
        raise ValueError("No ids provided to delete.")

    # compute SHA-256 hashes of the ids and truncate them
    hashed_ids = [hashlib.sha256(_id.encode()).hexdigest()[:16].upper() for _id in ids]

    # constructing the SQL statement with individual placeholders
    placeholders = ", ".join([":id" + str(i + 1) for i in range(len(hashed_ids))])

    ddl = f"DELETE FROM {table_name} WHERE id IN ({placeholders})"

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
    db_filter: Optional[FilterGroup],
    return_embeddings: bool = False,
) -> str:
    where_clause = ""
    if db_filter:
        where_clause = _generate_where_clause(db_filter)

    query = f"""
    SELECT id,
        text,
        metadata,
        vector_distance(embedding, :embedding,
        {_get_distance_function(distance_strategy)}) as distance
        {",embedding" if return_embeddings else ""}
    FROM {table_name}
    {f"WHERE {where_clause}" if db_filter else ""}
    ORDER BY distance
    FETCH APPROX FIRST {k} ROWS ONLY
    """

    return query


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


class OracleVS(VectorStore):
    """`OracleVS` vector store.

    To use, you should have both:
    - the ``oracledb`` python package installed
    - a connection string associated with a OracleDBCluster having deployed an
       Search index

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
        table_name: str,
        distance_strategy: DistanceStrategy = DistanceStrategy.EUCLIDEAN_DISTANCE,
        query: Optional[str] = "What is a Oracle database",
        params: Optional[Dict[str, Any]] = None,
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
        )

        embedding_dim = self.get_embedding_dimension()
        _create_table(connection, table_name, embedding_dim)

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
            )

            embedding_dim = await self.aget_embedding_dimension()
            await _acreate_table(connection, table_name, embedding_dim)

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
        self.table_name = table_name
        self.distance_strategy = distance_strategy
        self.params = params

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
        metadatas: Optional[List[Dict[Any, Any]]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add more texts to the vectorstore index.
        Args:
          texts: Iterable of strings to add to the vectorstore.
          metadatas: Optional list of metadatas associated with the texts.
          ids: Optional list of ids for the texts that are being added to
          the vector store.
          kwargs: vectorstore specific parameters
        """

        texts = list(texts)
        processed_ids = get_processed_ids(texts, metadatas, ids)

        embeddings = self._embed_documents(texts)
        if not metadatas:
            metadatas = [{} for _ in texts]

        docs: List[Tuple[Any, Any, Any, Any]] = [
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

        connection = _get_connection(self.client)
        if connection is None:
            raise ValueError("Failed to acquire a connection.")
        with connection.cursor() as cursor:
            cursor.setinputsizes(None, None, oracledb.DB_TYPE_JSON, None)
            cursor.executemany(
                f"INSERT INTO {self.table_name} (id, embedding, metadata, "
                f"text) VALUES (:1, :2, :3, :4)",
                docs,
            )
            connection.commit()
        return processed_ids

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
        Args:
          texts: Iterable of strings to add to the vectorstore.
          metadatas: Optional list of metadatas associated with the texts.
          ids: Optional list of ids for the texts that are being added to
          the vector store.
          kwargs: vectorstore specific parameters
        """

        texts = list(texts)
        processed_ids = get_processed_ids(texts, metadatas, ids)

        embeddings = await self._aembed_documents(texts)
        if not metadatas:
            metadatas = [{} for _ in texts]

        docs: List[Tuple[Any, Any, Any, Any]] = [
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

        async def context(connection: Any) -> None:
            if connection is None:
                raise ValueError("Failed to acquire a connection.")
            with connection.cursor() as cursor:
                cursor.setinputsizes(None, None, oracledb.DB_TYPE_JSON, None)
                await cursor.executemany(
                    f"INSERT INTO {self.table_name} (id, embedding, metadata, "
                    f"text) VALUES (:1, :2, :3, :4)",
                    docs,
                )
                await connection.commit()

        await _handle_context(self.client, context)

        return processed_ids

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to query."""
        embedding: List[float] = []
        if isinstance(self.embedding_function, Embeddings):
            embedding = self.embedding_function.embed_query(query)
        documents = self.similarity_search_by_vector(
            embedding=embedding, k=k, filter=filter, **kwargs
        )
        return documents

    async def asimilarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to query."""
        embedding: List[float] = []
        if isinstance(self.embedding_function, Embeddings):
            embedding = await self.embedding_function.aembed_query(query)
        documents = await self.asimilarity_search_by_vector(
            embedding=embedding, k=k, filter=filter, **kwargs
        )
        return documents

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[dict[str, Any]] = None,
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
        filter: Optional[dict[str, Any]] = None,
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
        filter: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query."""
        embedding: List[float] = []
        if isinstance(self.embedding_function, Embeddings):
            embedding = self.embedding_function.embed_query(query)
        docs_and_scores = self.similarity_search_by_vector_with_relevance_scores(
            embedding=embedding, k=k, filter=filter, **kwargs
        )
        return docs_and_scores

    async def asimilarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query."""
        embedding: List[float] = []
        if isinstance(self.embedding_function, Embeddings):
            embedding = await self.embedding_function.aembed_query(query)
        docs_and_scores = await self.asimilarity_search_by_vector_with_relevance_scores(
            embedding=embedding, k=k, filter=filter, **kwargs
        )
        return docs_and_scores

    def _get_clob_value(self, result: Any) -> str:
        clob_value = ""
        if result:
            if isinstance(result, oracledb.LOB):
                raw_data = result.read()
                if isinstance(raw_data, bytes):
                    clob_value = raw_data.decode(
                        "utf-8"
                    )  # specify the correct encoding
                else:
                    clob_value = raw_data
            elif isinstance(result, str):
                clob_value = result
            else:
                raise Exception("Unexpected type:", type(result))
        return clob_value

    async def _aget_clob_value(self, result: Any) -> str:
        clob_value = ""
        if result:
            if isinstance(result, oracledb.AsyncLOB):
                raw_data = await result.read()
                if isinstance(raw_data, bytes):
                    clob_value = raw_data.decode(
                        "utf-8"
                    )  # specify the correct encoding
                else:
                    clob_value = raw_data
            elif isinstance(result, str):
                clob_value = result
            else:
                raise Exception("Unexpected type:", type(result))
        return clob_value

    @_handle_exceptions
    def similarity_search_by_vector_with_relevance_scores(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        docs_and_scores = []

        embedding_arr: Any = array.array("f", embedding)

        db_filter: Optional[FilterGroup] = kwargs.get("db_filter", None)
        query = _get_similarity_search_query(
            self.table_name,
            self.distance_strategy,
            k,
            db_filter,
            return_embeddings=False,
        )

        # execute the query
        connection = _get_connection(self.client)
        if connection is None:
            raise ValueError("Failed to acquire a connection.")
        with connection.cursor() as cursor:
            cursor.execute(query, embedding=embedding_arr)
            results = cursor.fetchall()

            # filter results if filter is provided
            for result in results:
                metadata = result[2] or {}

                doc = Document(
                    page_content=(
                        self._get_clob_value(result[1]) if result[1] is not None else ""
                    ),
                    metadata=metadata,
                )
                distance = result[3]

                # apply filtering based on the 'filter' dictionary
                if not filter or all(
                    metadata.get(key) in value for key, value in filter.items()
                ):
                    docs_and_scores.append((doc, distance))

        return docs_and_scores

    @_ahandle_exceptions
    async def asimilarity_search_by_vector_with_relevance_scores(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        docs_and_scores = []

        embedding_arr: Any = array.array("f", embedding)

        db_filter: Optional[FilterGroup] = kwargs.get("db_filter", None)
        query = _get_similarity_search_query(
            self.table_name,
            self.distance_strategy,
            k,
            db_filter,
            return_embeddings=False,
        )

        async def context(connection: Any) -> List:
            # execute the query
            with connection.cursor() as cursor:
                await cursor.execute(query, embedding=embedding_arr)
                results = await cursor.fetchall()

                # filter results if filter is provided
                for result in results:
                    metadata = result[2] or {}

                    doc = Document(
                        page_content=(
                            await self._aget_clob_value(result[1])
                            if result[1] is not None
                            else ""
                        ),
                        metadata=metadata,
                    )
                    distance = result[3]

                    # apply filtering based on the 'filter' dictionary
                    if not filter or all(
                        metadata.get(key) in value for key, value in filter.items()
                    ):
                        docs_and_scores.append((doc, distance))

            return docs_and_scores

        return await _handle_context(self.client, context)

    @_handle_exceptions
    def similarity_search_by_vector_returning_embeddings(
        self,
        embedding: List[float],
        k: int,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float, NDArray[np.float32]]]:
        embedding_arr: Any = array.array("f", embedding)

        documents = []

        db_filter: Optional[FilterGroup] = kwargs.get("db_filter", None)
        query = _get_similarity_search_query(
            self.table_name,
            self.distance_strategy,
            k,
            db_filter,
            return_embeddings=True,
        )

        # execute the query
        connection = _get_connection(self.client)
        if connection is None:
            raise ValueError("Failed to acquire a connection.")
        with connection.cursor() as cursor:
            cursor.execute(query, embedding=embedding_arr)
            results = cursor.fetchall()

            for result in results:
                page_content_str = self._get_clob_value(result[1])
                metadata = result[2] or {}

                # apply filter if provided and matches; otherwise, add all
                # documents
                if not filter or all(
                    metadata.get(key) in value for key, value in filter.items()
                ):
                    document = Document(
                        page_content=page_content_str, metadata=metadata
                    )
                    distance = result[3]

                    # assuming result[4] is already in the correct format;
                    # adjust if necessary
                    current_embedding = (
                        np.array(result[4], dtype=np.float32)
                        if result[4]
                        else np.empty(0, dtype=np.float32)
                    )

                    documents.append((document, distance, current_embedding))

        return documents

    @_ahandle_exceptions
    async def asimilarity_search_by_vector_returning_embeddings(
        self,
        embedding: List[float],
        k: int,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float, NDArray[np.float32]]]:
        embedding_arr: Any = array.array("f", embedding)

        documents = []

        db_filter: Optional[FilterGroup] = kwargs.get("db_filter", None)
        query = _get_similarity_search_query(
            self.table_name,
            self.distance_strategy,
            k,
            db_filter,
            return_embeddings=True,
        )

        async def context(connection: Any) -> List:
            # execute the query
            with connection.cursor() as cursor:
                await cursor.execute(query, embedding=embedding_arr)
                results = await cursor.fetchall()

                for result in results:
                    page_content_str = await self._aget_clob_value(result[1])
                    metadata = result[2] or {}

                    # apply filter if provided and matches; otherwise, add all
                    # documents
                    if not filter or all(
                        metadata.get(key) in value for key, value in filter.items()
                    ):
                        document = Document(
                            page_content=page_content_str, metadata=metadata
                        )
                        distance = result[3]

                        # assuming result[4] is already in the correct format;
                        # adjust if necessary
                        current_embedding = (
                            np.array(result[4], dtype=np.float32)
                            if result[4]
                            else np.empty(0, dtype=np.float32)
                        )

                        documents.append((document, distance, current_embedding))

            return documents

        return await _handle_context(self.client, context)

    def max_marginal_relevance_search_with_score_by_vector(
        self,
        embedding: List[float],
        *,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
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
          filter: (Optional[Dict[str, str]]): Filter by metadata. Defaults
          to None.
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
            embedding, fetch_k, filter=filter
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
        filter: Optional[Dict[str, Any]] = None,
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
          filter: (Optional[Dict[str, str]]): Filter by metadata. Defaults
          to None.
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
                embedding, fetch_k, filter=filter
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
        filter: Optional[Dict[str, Any]] = None,
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
          filter: Optional[Dict[str, Any]]
          **kwargs: Any
        Returns:
          List of Documents selected by maximal marginal relevance.
        """
        docs_and_scores = self.max_marginal_relevance_search_with_score_by_vector(
            embedding, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult, filter=filter
        )
        return [doc for doc, _ in docs_and_scores]

    async def amax_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
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
          filter: Optional[Dict[str, Any]]
          **kwargs: Any
        Returns:
          List of Documents selected by maximal marginal relevance.
        """
        docs_and_scores = (
            await self.amax_marginal_relevance_search_with_score_by_vector(
                embedding, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult, filter=filter
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
        filter: Optional[Dict[str, Any]] = None,
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
          filter: Optional[Dict[str, Any]]
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
        filter: Optional[Dict[str, Any]] = None,
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
          filter: Optional[Dict[str, Any]]
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
        texts: Iterable[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
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
                f"Expected DistanceStrategy got " f"{type(distance_strategy).__name__} "
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
        ) = OracleVS._from_texts_helper(texts, embedding, metadatas, **kwargs)

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
        ) = OracleVS._from_texts_helper(texts, embedding, metadatas, **kwargs)

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
