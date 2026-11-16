"""Channel versions must match blob rows regardless of their JSON type.

``checkpoints.channel_versions`` keeps whatever type the caller passed (the
LangGraph.js savers assign numeric versions by default), while
``checkpoint_blobs.version`` is a VARCHAR column. Matching them with a strict
``==`` silently drops every blob-backed channel value for such checkpoints.
"""

import json
from uuid import uuid4

import oracledb
import pytest
from langgraph.checkpoint.base import empty_checkpoint

from tests.conftest import DEFAULT_CONNECTION_INFO
from tests.conftest_checkpointer import _async_saver, _sync_saver

_JS_METADATA = {"source": "input", "step": 0, "parents": {}}


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}


def _seed_js_layout_checkpoint(thread_id: str, checkpoint_id: str) -> None:
    """Write a checkpoint the way the LangGraph.js Oracle saver does.

    Every channel value lives in ``checkpoint_blobs`` (``channel_values`` in the
    JSON document is empty), versions are numeric in the JSON document and text
    in the blob table, and a deleted channel is recorded as an ``empty`` row.
    """
    checkpoint = {
        "v": 1,
        "id": checkpoint_id,
        "ts": "2026-09-23T00:00:00.000Z",
        "channel_values": {},
        "channel_versions": {"messages": 7, "counter": 7, "removed": 8},
        "versions_seen": {},
    }
    blob_rows = [
        (thread_id, " ", "messages", "7", "json", json.dumps(["hi"]).encode()),
        (thread_id, " ", "counter", "7", "json", json.dumps(2).encode()),
        (thread_id, " ", "removed", "8", "empty", None),
    ]
    with oracledb.connect(**DEFAULT_CONNECTION_INFO) as conn:
        with conn.cursor() as cur:
            cur.setinputsizes(
                checkpoint=oracledb.DB_TYPE_JSON, metadata=oracledb.DB_TYPE_JSON
            )
            cur.execute(
                """
                INSERT INTO checkpoints
                    (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id,
                     checkpoint, metadata)
                VALUES (:thread_id, :checkpoint_ns, :checkpoint_id, NULL,
                        :checkpoint, :metadata)
                """,
                {
                    "thread_id": thread_id,
                    "checkpoint_ns": " ",
                    "checkpoint_id": checkpoint_id,
                    "checkpoint": checkpoint,
                    "metadata": _JS_METADATA,
                },
            )
            cur.executemany(
                """
                INSERT INTO checkpoint_blobs
                    (thread_id, checkpoint_ns, channel, version, type, blob)
                VALUES (:1, :2, :3, :4, :5, :6)
                """,
                blob_rows,
            )
        conn.commit()


@pytest.mark.parametrize("saver_name", ["base", "pool"])
def test_numeric_new_versions_round_trip_through_blobs(saver_name: str) -> None:
    """put() with numeric versions must read back its own blob-backed channels."""
    with _sync_saver(saver_name) as saver:
        saver.json_size_threshold_mb = 0  # route every non-primitive to blobs
        config = _config(f"numeric-versions-{uuid4()}")
        checkpoint = empty_checkpoint()
        checkpoint["channel_values"] = {
            "messages": ["hi"],
            "payload": b"\x01\x02",
            "counter": 2,
        }
        versions = {"messages": 7, "payload": 7, "counter": 7}

        saver.put(config, checkpoint, _JS_METADATA, versions)

        loaded = saver.get_tuple(config)
        assert loaded is not None
        assert loaded.checkpoint["channel_values"] == {
            "messages": ["hi"],
            "payload": b"\x01\x02",
            "counter": 2,
        }


@pytest.mark.parametrize("saver_name", ["base", "pool"])
def test_reads_checkpoint_written_by_langgraph_js_saver(saver_name: str) -> None:
    """A JS-layout checkpoint (all blobs, numeric versions) loads completely."""
    with _sync_saver(saver_name) as saver:
        thread_id = f"js-layout-{uuid4()}"
        checkpoint_id = str(uuid4())
        _seed_js_layout_checkpoint(thread_id, checkpoint_id)

        loaded = saver.get_tuple(_config(thread_id))
        assert loaded is not None
        assert loaded.checkpoint["id"] == checkpoint_id
        assert loaded.checkpoint["channel_values"] == {"messages": ["hi"], "counter": 2}
        assert "removed" not in loaded.checkpoint["channel_values"]

        listed = list(saver.list(_config(thread_id)))
        assert len(listed) == 1
        assert listed[0].checkpoint["channel_values"] == {
            "messages": ["hi"],
            "counter": 2,
        }


@pytest.mark.parametrize("saver_name", ["base", "pool"])
async def test_numeric_new_versions_round_trip_through_blobs_async(
    saver_name: str,
) -> None:
    async with _async_saver(saver_name) as saver:
        saver.json_size_threshold_mb = 0
        config = _config(f"numeric-versions-async-{uuid4()}")
        checkpoint = empty_checkpoint()
        checkpoint["channel_values"] = {
            "messages": ["hi"],
            "payload": b"\x01\x02",
            "counter": 2,
        }
        versions = {"messages": 7, "payload": 7, "counter": 7}

        await saver.aput(config, checkpoint, _JS_METADATA, versions)

        loaded = await saver.aget_tuple(config)
        assert loaded is not None
        assert loaded.checkpoint["channel_values"] == {
            "messages": ["hi"],
            "payload": b"\x01\x02",
            "counter": 2,
        }


@pytest.mark.parametrize("saver_name", ["base", "pool"])
async def test_reads_checkpoint_written_by_langgraph_js_saver_async(
    saver_name: str,
) -> None:
    async with _async_saver(saver_name) as saver:
        thread_id = f"js-layout-async-{uuid4()}"
        checkpoint_id = str(uuid4())
        _seed_js_layout_checkpoint(thread_id, checkpoint_id)

        loaded = await saver.aget_tuple(_config(thread_id))
        assert loaded is not None
        assert loaded.checkpoint["channel_values"] == {"messages": ["hi"], "counter": 2}
        assert "removed" not in loaded.checkpoint["channel_values"]

        listed = [item async for item in saver.alist(_config(thread_id))]
        assert len(listed) == 1
        assert listed[0].checkpoint["channel_values"] == {
            "messages": ["hi"],
            "counter": 2,
        }
