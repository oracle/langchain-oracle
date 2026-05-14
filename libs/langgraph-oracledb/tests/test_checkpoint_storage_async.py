from decimal import Decimal, InvalidOperation
from typing import Any
from unittest.mock import patch

import pytest
from langgraph.checkpoint.base import empty_checkpoint
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from langgraph_oracledb.checkpoint.oracle.aio import AsyncOracleSaver
from tests.conftest import DEFAULT_CONNECTION_INFO, create_connection_string
from tests.conftest_checkpointer import (
    _async_base_saver_with_serde,
    _async_cleanup_checkpoint_tables,
    _async_pool_saver_with_serde,
    _async_saver,
    _TestCustomObject,
)


@pytest.mark.parametrize("saver_name", ["base", "pool"])
async def test_small_json_objects_stay_in_json_column(saver_name: str) -> None:
    """Test that small JSON objects stay in the main checkpoints table."""
    async with _async_saver(saver_name) as saver:
        config = {
            "configurable": {
                "thread_id": "thread-json-test",
                "checkpoint_ns": "",
            }
        }

        checkpoint = empty_checkpoint()
        checkpoint["channel_values"] = {
            "small_dict": {"user": "john", "count": 5},
            "small_list": [1, 2, 3, 4, 5],
            "primitive": "test_string",
            "number": 42,
        }

        await saver.aput(
            config,
            checkpoint,
            {},
            {"small_dict": "v1", "small_list": "v2", "primitive": "v3", "number": "v4"},
        )

        result = await saver.aget_tuple(config)
        assert result is not None
        assert result.checkpoint["channel_values"]["small_dict"] == {
            "user": "john",
            "count": 5,
        }
        assert result.checkpoint["channel_values"]["small_list"] == [1, 2, 3, 4, 5]
        assert result.checkpoint["channel_values"]["primitive"] == "test_string"
        assert result.checkpoint["channel_values"]["number"] == 42

        async with saver._cursor() as cur:
            await cur.execute(
                "SELECT checkpoint FROM checkpoints WHERE thread_id = :1 AND checkpoint_id = :2",
                [config["configurable"]["thread_id"], checkpoint["id"]],
            )
            row = await cur.fetchone()
            assert row is not None

            if isinstance(row[0], str):
                import orjson

                checkpoint_data = orjson.loads(row[0])
            else:
                checkpoint_data = row[0]

            assert "small_dict" in checkpoint_data["channel_values"]
            assert "small_list" in checkpoint_data["channel_values"]


@pytest.mark.parametrize("saver_name", ["base", "pool"])
async def test_large_json_objects_go_to_blobs(saver_name: str) -> None:
    """Test that large JSON objects go to checkpoint_blobs table."""
    conn_string = create_connection_string(DEFAULT_CONNECTION_INFO)

    try:
        async with AsyncOracleSaver.from_conn_string(
            conn_string,
            json_size_threshold_mb=0.001,
        ) as saver:
            await saver.setup()

            config = {
                "configurable": {
                    "thread_id": "thread-blob-test",
                    "checkpoint_ns": "",
                }
            }

            large_data = {"data": "x" * 10000}

            checkpoint = empty_checkpoint()
            checkpoint["channel_values"] = {
                "large_object": large_data,
                "small_object": {"tiny": "data"},
            }

            new_config = await saver.aput(
                config, checkpoint, {}, {"large_object": "v1", "small_object": "v2"}
            )

            result = await saver.aget_tuple(new_config)
            assert result is not None
            assert result.checkpoint["channel_values"]["large_object"] == large_data
            assert result.checkpoint["channel_values"]["small_object"] == {
                "tiny": "data"
            }

            async with saver._cursor() as cur:
                await cur.execute(
                    "SELECT checkpoint FROM checkpoints WHERE thread_id = :1 AND checkpoint_id = :2",
                    [
                        new_config["configurable"]["thread_id"],
                        new_config["configurable"]["checkpoint_id"],
                    ],
                )
                row = await cur.fetchone()
                assert row is not None

                if isinstance(row[0], str):
                    import orjson

                    checkpoint_data = orjson.loads(row[0])
                else:
                    checkpoint_data = row[0]

                assert "large_object" not in checkpoint_data["channel_values"]
                assert "small_object" in checkpoint_data["channel_values"]

                await cur.execute(
                    "SELECT channel, type FROM checkpoint_blobs WHERE thread_id = :1 AND channel = :2",
                    [config["configurable"]["thread_id"], "large_object"],
                )
                blob_row = await cur.fetchone()
                assert blob_row is not None
                assert blob_row[0] == "large_object"

    finally:
        await _async_cleanup_checkpoint_tables()


@pytest.mark.parametrize("saver_name", ["base", "pool"])
async def test_checkpoint_blobs_preserve_empty_and_large_bytes(
    saver_name: str,
) -> None:
    async with _async_saver(saver_name) as saver:
        config = {
            "configurable": {
                "thread_id": "thread-byte-blob-test",
                "checkpoint_ns": "",
            }
        }
        large_bytes = b"x" * 32768
        checkpoint = empty_checkpoint()
        checkpoint["channel_values"] = {
            "empty_bytes": b"",
            "large_bytes": large_bytes,
        }

        new_config = await saver.aput(
            config,
            checkpoint,
            {},
            {"empty_bytes": "v1", "large_bytes": "v2"},
        )

        result = await saver.aget_tuple(new_config)
        assert result is not None
        assert result.checkpoint["channel_values"]["empty_bytes"] == b""
        assert result.checkpoint["channel_values"]["large_bytes"] == large_bytes


@pytest.mark.parametrize("saver_name", ["base", "pool"])
async def test_non_serializable_objects_go_to_blobs(saver_name: str) -> None:
    """Test that non-JSON-serializable objects always go to blobs."""
    serde = JsonPlusSerializer(pickle_fallback=True)

    if saver_name == "base":
        async with _async_base_saver_with_serde(serde) as saver:
            await _test_non_serializable_objects_logic(saver)
    elif saver_name == "pool":
        async with _async_pool_saver_with_serde(serde) as saver:
            await _test_non_serializable_objects_logic(saver)


async def _test_non_serializable_objects_logic(saver) -> None:
    """Shared test logic for non-serializable objects."""
    config = {
        "configurable": {
            "thread_id": "thread-nonserial-test",
            "checkpoint_ns": "",
        }
    }

    custom_obj = _TestCustomObject("test_value")

    checkpoint = empty_checkpoint()
    checkpoint["channel_values"] = {
        "custom_object": custom_obj,
        "json_object": {"serializable": "data"},
    }

    await saver.aput(
        config, checkpoint, {}, {"custom_object": "v1", "json_object": "v2"}
    )

    result = await saver.aget_tuple(config)
    assert result is not None
    assert "custom_object" in result.checkpoint["channel_values"]
    assert result.checkpoint["channel_values"]["json_object"] == {
        "serializable": "data"
    }


def _normalize_oracle_json_types(obj: Any) -> Any:
    """Convert Oracle-specific types to standard Python types."""
    if isinstance(obj, dict):
        return {k: _normalize_oracle_json_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_normalize_oracle_json_types(item) for item in obj]
    if isinstance(obj, Decimal):
        try:
            if obj % 1 == 0:
                return int(obj)
            return float(obj)
        except (InvalidOperation, OverflowError):
            return float(obj)
    return obj


@pytest.mark.parametrize("saver_name", ["base", "pool"])
async def test_primitive_values_stay_in_json(saver_name: str) -> None:
    """Test that primitive values always stay in JSON column."""
    async with _async_saver(saver_name) as saver:
        config = {
            "configurable": {
                "thread_id": "thread-primitive-test",
                "checkpoint_ns": "",
            }
        }

        checkpoint = empty_checkpoint()
        checkpoint["channel_values"] = {
            "string_val": "test_string",
            "int_val": 42,
            "float_val": 3.14,
            "bool_val": True,
            "none_val": None,
        }

        await saver.aput(
            config,
            checkpoint,
            {},
            {
                "string_val": "v1",
                "int_val": "v2",
                "float_val": "v3",
                "bool_val": "v4",
                "none_val": "v5",
            },
        )

        async with saver._cursor() as cur:
            await cur.execute(
                "SELECT checkpoint FROM checkpoints WHERE thread_id = :1 AND checkpoint_id = :2",
                [config["configurable"]["thread_id"], checkpoint["id"]],
            )
            row = await cur.fetchone()
            assert row is not None

            if isinstance(row[0], str):
                import orjson

                checkpoint_data = orjson.loads(row[0])
            else:
                checkpoint_data = _normalize_oracle_json_types(row[0])

            assert checkpoint_data["channel_values"]["string_val"] == "test_string"
            assert checkpoint_data["channel_values"]["int_val"] == 42
            assert checkpoint_data["channel_values"]["float_val"] == 3.14
            assert checkpoint_data["channel_values"]["bool_val"] is True
            assert checkpoint_data["channel_values"]["none_val"] is None


@pytest.mark.parametrize("saver_name", ["base", "pool"])
async def test_configurable_size_threshold(saver_name: str) -> None:
    """Test that size threshold can be configured."""
    conn_string = create_connection_string(DEFAULT_CONNECTION_INFO)

    try:
        async with AsyncOracleSaver.from_conn_string(
            conn_string,
            json_size_threshold_mb=0.01,
        ) as saver:
            await saver.setup()

            config = {
                "configurable": {
                    "thread_id": "thread-config-test",
                    "checkpoint_ns": "",
                }
            }

            medium_data = {"data": "x" * 50000}

            checkpoint = empty_checkpoint()
            checkpoint["channel_values"] = {
                "medium_object": medium_data,
            }

            await saver.aput(config, checkpoint, {}, {"medium_object": "v1"})

            async with saver._cursor() as cur:
                await cur.execute(
                    "SELECT checkpoint FROM checkpoints WHERE thread_id = :1 AND checkpoint_id = :2",
                    [config["configurable"]["thread_id"], checkpoint["id"]],
                )
                row = await cur.fetchone()
                assert row is not None

                if isinstance(row[0], str):
                    import orjson

                    checkpoint_data = orjson.loads(row[0])
                else:
                    checkpoint_data = row[0]

                assert "medium_object" not in checkpoint_data["channel_values"]

    finally:
        await _async_cleanup_checkpoint_tables()


@pytest.mark.parametrize("saver_name", ["base", "pool"])
@patch("json.dumps")
async def test_size_estimation_with_mocked_serialization(
    mock_dumps, saver_name: str
) -> None:
    """Test size estimation logic with mocked serialization."""
    async with _async_saver(saver_name) as saver:
        mock_dumps.return_value = b"x" * (3 * 1024 * 1024)
        assert not saver._should_use_blob({"small": "object"}, 5.0)

        mock_dumps.return_value = b"x" * (7 * 1024 * 1024)
        assert saver._should_use_blob({"large": "object"}, 5.0)

        mock_dumps.return_value = b"x" * (1 * 1024 * 1024)
        assert not saver._should_use_blob({"exact": "threshold"}, 1.0)

        mock_dumps.return_value = b"x" * (10 * 1024 * 1024)
        assert not saver._should_use_blob("primitive_string", 5.0)
        assert not saver._should_use_blob(42, 5.0)
        assert not saver._should_use_blob(3.14, 5.0)
        assert not saver._should_use_blob(True, 5.0)
        assert not saver._should_use_blob(None, 5.0)


@pytest.mark.parametrize("saver_name", ["base", "pool"])
async def test_checkpoint_data_stored_even_if_json(saver_name: str) -> None:
    """Test that checkpoint data is stored in CHECKPOINT even if it is JSON."""
    async with _async_saver(saver_name) as saver:
        config = {
            "configurable": {
                "thread_id": "thread-json-checkpoint-test",
                "checkpoint_ns": "",
            }
        }

        checkpoint = empty_checkpoint()
        checkpoint["channel_values"] = {
            "json_data": {
                "user": "alice",
                "preferences": {"theme": "dark", "lang": "en"},
            },
            "list_data": [1, 2, 3, {"nested": "value"}],
            "string_data": "simple string",
            "number_data": 123,
        }

        await saver.aput(
            config,
            checkpoint,
            {"test": "metadata"},
            {
                "json_data": "v1",
                "list_data": "v2",
                "string_data": "v3",
                "number_data": "v4",
            },
        )

        async with saver._cursor() as cur:
            await cur.execute(
                "SELECT checkpoint, metadata FROM checkpoints WHERE thread_id = :1 AND checkpoint_id = :2",
                [config["configurable"]["thread_id"], checkpoint["id"]],
            )
            row = await cur.fetchone()
            assert row is not None

            checkpoint_data = row[0]
            metadata_data = row[1]

            if isinstance(checkpoint_data, str):
                import orjson

                checkpoint_data = orjson.loads(checkpoint_data)

            assert checkpoint_data["id"] == checkpoint["id"]
            assert checkpoint_data["ts"] == checkpoint["ts"]
            assert "channel_values" in checkpoint_data

            if isinstance(metadata_data, str):
                import orjson

                metadata_data = orjson.loads(metadata_data)

            assert "test" in metadata_data

        result = await saver.aget_tuple(config)
        assert result is not None
        assert result.checkpoint["channel_values"]["json_data"] == {
            "user": "alice",
            "preferences": {"theme": "dark", "lang": "en"},
        }
        assert result.checkpoint["channel_values"]["list_data"] == [
            1,
            2,
            3,
            {"nested": "value"},
        ]
        assert result.checkpoint["channel_values"]["string_data"] == "simple string"
        assert result.checkpoint["channel_values"]["number_data"] == 123
