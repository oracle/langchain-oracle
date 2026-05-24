from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import oracledb


def with_blob_lobs(
    conn: oracledb.Connection, params: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Replace byte BLOB bind values with temporary LOBs for executemany."""
    lob_params = []
    for param in params:
        blob = param.get("blob")
        if isinstance(blob, (bytes, bytearray)):
            lob = conn.createlob(oracledb.DB_TYPE_BLOB)
            if blob:
                lob.write(bytes(blob))
            param = {**param, "blob": lob}
        lob_params.append(param)
    return lob_params


async def awith_blob_lobs(
    conn: oracledb.AsyncConnection, params: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Replace byte BLOB bind values with temporary LOBs for async executemany."""
    lob_params = []
    for param in params:
        blob = param.get("blob")
        if isinstance(blob, (bytes, bytearray)):
            lob = await conn.createlob(oracledb.DB_TYPE_BLOB)
            if blob:
                await lob.write(bytes(blob))
            param = {**param, "blob": lob}
        lob_params.append(param)
    return lob_params
