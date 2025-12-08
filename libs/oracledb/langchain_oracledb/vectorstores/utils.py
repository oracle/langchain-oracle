from __future__ import annotations

import functools
import logging
import re
import uuid
from contextlib import asynccontextmanager, contextmanager
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Callable,
    Iterator,
    Optional,
    TypeVar,
    cast,
)

import oracledb

if TYPE_CHECKING:
    from oracledb import AsyncConnection, Connection

logger = logging.getLogger(__name__)


# define a type variable that can be any kind of function
T = TypeVar("T", bound=Callable[..., Any])


@contextmanager
def _get_connection(client: Any) -> Iterator[Connection]:
    # check if ConnectionPool exists
    connection_pool_class = getattr(oracledb, "ConnectionPool", None)

    if isinstance(client, oracledb.Connection):
        yield client
    elif connection_pool_class and isinstance(client, connection_pool_class):
        with client.acquire() as connection:
            yield connection
    else:
        valid_types = "oracledb.Connection"
        if connection_pool_class:
            valid_types += " or oracledb.ConnectionPool"
        raise TypeError(
            f"Expected client of type {valid_types}, got {type(client).__name__}"
        )


@asynccontextmanager
async def _aget_connection(client: Any) -> AsyncIterator[AsyncConnection]:
    # check if ConnectionPool exists
    connection_pool_class = getattr(oracledb, "AsyncConnectionPool", None)

    if isinstance(client, oracledb.AsyncConnection):
        yield client
    elif connection_pool_class and isinstance(client, connection_pool_class):
        async with client.acquire() as connection:
            yield connection
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


@_handle_exceptions
def drop_table_purge(client: Any, table_name: str) -> None:
    """Drop a table and purge it from the database.

    Args:
        client: oracledb connection object.
        table_name: The name of the table to drop.

    Raises:
        RuntimeError: If an error occurs while dropping the table.
    """
    with _get_connection(client) as connection:
        table_name = _quote_indentifier(table_name)
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

    async with _aget_connection(client) as connection:
        if await _atable_exists(connection, table_name):
            with connection.cursor() as cursor:
                ddl = f"DROP TABLE {table_name} PURGE"
                await cursor.execute(ddl)
            logger.info(f"Table {table_name} dropped successfully...")
        else:
            logger.info(f"Table {table_name} not found...")


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


def _get_index_name(base_name: str) -> str:
    unique_id = str(uuid.uuid4()).replace("-", "")
    return f'"{base_name}_{unique_id}"'


@_handle_exceptions
def drop_index(client: Any, idx_name: str) -> None:
    """Drop a index and purge it from the database.

    Args:
        client: oracledb connection object.
        idx_name: The name of the table to drop.

    Raises:
        RuntimeError: If an error occurs while dropping the index.
    """
    idx_name = _quote_indentifier(idx_name)
    with _get_connection(client) as connection:
        if _index_exists(connection, idx_name):
            with connection.cursor() as cursor:
                ddl = f"DROP INDEX {idx_name}"
                cursor.execute(ddl)
            logger.info(f"Index {idx_name} dropped successfully...")
        else:
            logger.info(f"Index {idx_name} not found...")
    return


@_ahandle_exceptions
async def adrop_index(client: Any, idx_name: str) -> None:
    """Drop a index and purge it from the database.

    Args:
        client: oracledb async connection object.
        idx_name: The name of the table to drop.

    Raises:
        RuntimeError: If an error occurs while dropping the index.
    """
    idx_name = _quote_indentifier(idx_name)

    async with _aget_connection(client) as connection:
        if await _aindex_exists(connection, idx_name):
            with connection.cursor() as cursor:
                ddl = f"DROP INDEX {idx_name}"
                await cursor.execute(ddl)
            logger.info(f"Index {idx_name} dropped successfully...")
        else:
            logger.info(f"Index {idx_name} not found...")


def output_type_string_handler(cursor: Any, metadata: Any) -> Any:
    if metadata.type_code is oracledb.DB_TYPE_CLOB:
        return cursor.var(oracledb.DB_TYPE_LONG, arraysize=cursor.arraysize)
    if metadata.type_code is oracledb.DB_TYPE_NCLOB:
        return cursor.var(oracledb.DB_TYPE_LONG_NVARCHAR, arraysize=cursor.arraysize)
