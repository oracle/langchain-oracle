# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""Native selection retrieves SEARCH capability and COMPATIBLE in one SELECT."""

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import oracledb
import pytest

from langchain_oracledb.vectorstores import _vecdb

from ._backend_fixtures import native_config


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.asyncio
async def test_selection_uses_one_combined_capability_query(asynchronous):
    config = native_config("vecdb")
    connection = (
        await oracledb.connect_async(**config)
        if asynchronous
        else oracledb.connect(**config)
    )
    try:
        with connection.cursor() as reference:
            version, compatible = (
                reference.var(str, size=128),
                reference.var(str, size=128),
            )
            if asynchronous:
                await reference.execute(
                    _vecdb.COMPATIBILITY_SQL,
                    db_version=version,
                    compatibility=compatible,
                )
            else:
                reference.execute(
                    _vecdb.COMPATIBILITY_SQL,
                    db_version=version,
                    compatibility=compatible,
                )
            expected = _vecdb._compatible_backend(
                _vecdb.parse_version(connection.version),
                compatible.getvalue(),
                "missing",
                "auto",
            )
        with connection.cursor() as cursor:
            observed = MagicMock(wraps=cursor)
            observed.__enter__.return_value = observed
            observed.__exit__.return_value = False
            if asynchronous:
                observed.execute = AsyncMock(wraps=cursor.execute)
                observed.fetchone = AsyncMock(wraps=cursor.fetchone)
                observed.fetchall = AsyncMock(wraps=cursor.fetchall)
            proxy = SimpleNamespace(version=connection.version, cursor=lambda: observed)
            name = "LCCOMP_" + uuid.uuid4().hex[:12].upper()
            result = (
                await _vecdb.anative_selection(proxy, name, "auto")
                if asynchronous
                else _vecdb.native_selection(proxy, name, "auto")
            )
            assert result[0] == expected
            calls = observed.execute.call_args_list
            assert (
                sum(call.args[0] == _vecdb.VECDB_CAPABILITY_SQL for call in calls) == 1
            )
            assert not any(call.args[0] == _vecdb.COMPATIBILITY_SQL for call in calls)
            observed.var.assert_not_called()
    finally:
        if asynchronous:
            await connection.close()
        else:
            connection.close()
