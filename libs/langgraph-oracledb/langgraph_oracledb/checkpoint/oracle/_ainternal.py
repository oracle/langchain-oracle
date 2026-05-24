from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Union

import oracledb
from oracledb import AsyncConnection, AsyncConnectionPool

# Type alias for connection types
Conn = Union[oracledb.AsyncConnection, AsyncConnectionPool]


@asynccontextmanager
async def get_connection(conn: Conn) -> AsyncIterator[oracledb.AsyncConnection]:
    """Get a connection from either a single connection or connection pool."""

    if isinstance(conn, AsyncConnection):
        yield conn
    elif isinstance(conn, AsyncConnectionPool):
        async with conn.acquire() as connection:
            yield connection
    else:
        raise TypeError(f"Invalid connection type: {type(conn)}")
