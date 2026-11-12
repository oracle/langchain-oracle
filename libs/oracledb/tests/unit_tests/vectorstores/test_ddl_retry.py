"""Test-only DDL waits retry just the requested Oracle error and stay bounded."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import oracledb
import pytest

from tests.integration_tests.vectorstores._backend_fixtures import aretry_ddl, retry_ddl

FIXTURES = "tests.integration_tests.vectorstores._backend_fixtures"


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("code", [54, 29868])
@pytest.mark.parametrize("outcome", ["success", "exhausted", "unrelated"])
@pytest.mark.asyncio
async def test_ddl_retry_is_specific_and_bounded(asynchronous, code, outcome):
    error = oracledb.DatabaseError(
        SimpleNamespace(code=942 if outcome == "unrelated" else code)
    )
    wrapped = RuntimeError("public helper wrapper")
    wrapped.__cause__ = error
    action = AsyncMock() if asynchronous else Mock()
    action.side_effect = [wrapped, "done"] if outcome == "success" else wrapped
    sleep = AsyncMock() if asynchronous else Mock()
    target = ".asyncio.sleep" if asynchronous else ".time.sleep"

    async def invoke():
        if asynchronous:
            return await aretry_ddl(action, code)
        return retry_ddl(action, code)

    with patch(FIXTURES + target, sleep):
        if outcome == "success":
            assert await invoke() == "done"
        else:
            with pytest.raises(RuntimeError) as caught:
                await invoke()
            assert caught.value is wrapped
    expected = {"success": 2, "exhausted": 11, "unrelated": 1}[outcome]
    assert action.call_count == expected
    assert sleep.call_count == expected - 1
    assert all(call.args == (0.5,) for call in sleep.call_args_list)


@pytest.mark.asyncio
async def test_async_ddl_wait_can_be_cancelled():
    action = AsyncMock(side_effect=oracledb.DatabaseError(SimpleNamespace(code=29868)))
    with patch(FIXTURES + ".asyncio.sleep", side_effect=asyncio.CancelledError):
        with pytest.raises(asyncio.CancelledError):
            await aretry_ddl(action, 29868)
    action.assert_awaited_once()
