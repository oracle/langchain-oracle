import pytest

from langgraph_oracledb.checkpoint.oracle import OracleSaver
from tests._checkpoint_test_utils import generate_checkpoint, generate_config


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method_name", "args"),
    [
        ("aget_tuple", (generate_config("thread-1"),)),
        ("aput", (generate_config("thread-1"), generate_checkpoint(), {}, {})),
        (
            "aput_writes",
            (
                {
                    "configurable": {
                        "thread_id": "thread-1",
                        "checkpoint_ns": "",
                        "checkpoint_id": "checkpoint-1",
                    }
                },
                [("channel", "value")],
                "task-1",
            ),
        ),
        ("adelete_thread", ("thread-1",)),
    ],
)
async def test_async_methods_raise_clear_error(
    method_name: str, args: tuple[object, ...]
) -> None:
    saver = OracleSaver(conn=object())  # type: ignore[arg-type]

    with pytest.raises(NotImplementedError, match="Use AsyncOracleSaver"):
        await getattr(saver, method_name)(*args)


@pytest.mark.asyncio
async def test_async_list_raises_clear_error() -> None:
    saver = OracleSaver(conn=object())  # type: ignore[arg-type]

    with pytest.raises(NotImplementedError, match="Use AsyncOracleSaver"):
        async for _ in saver.alist(generate_config("thread-1")):
            pass
