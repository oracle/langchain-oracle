"""Shared utility functions for the Oracle checkpoint & storage classes."""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Union

from oracledb import Connection, ConnectionPool

Conn = Union[Connection, ConnectionPool]


@contextmanager
def get_connection(conn: Conn) -> Iterator[Connection]:
    """Get a connection from either a single connection or connection pool."""
    if isinstance(conn, Connection):
        yield conn
    elif isinstance(conn, ConnectionPool):
        with conn.acquire() as connection:
            yield connection
    else:
        raise TypeError(f"Invalid connection type: {type(conn)}")
