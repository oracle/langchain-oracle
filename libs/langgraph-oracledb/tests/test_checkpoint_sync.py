import re
from uuid import uuid4

import pytest
from langgraph.checkpoint.base import (
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    create_checkpoint,
    empty_checkpoint,
)
from langgraph.checkpoint.serde.types import ERROR, INTERRUPT, TASKS

from langgraph_oracledb.checkpoint.oracle import OracleSaver
from tests._checkpoint_test_utils import (
    generate_checkpoint,
    generate_config,
    generate_metadata,
    version_matches,
)
from tests.conftest_checkpointer import _exclude_keys, _sync_saver


def _setup_list_data(saver: OracleSaver) -> dict[str, object]:
    thread_id = str(uuid4())
    checkpoint_ids = []
    parent_config = None
    for step in range(4):
        config = generate_config(thread_id)
        if parent_config:
            config["configurable"]["checkpoint_id"] = parent_config["configurable"][
                "checkpoint_id"
            ]
        checkpoint = generate_checkpoint()
        metadata = generate_metadata(
            source="input" if step % 2 == 0 else "loop",
            step=step,
        )
        parent_config = saver.put(config, checkpoint, metadata, {})
        checkpoint_ids.append(checkpoint["id"])

    return {
        "thread_id": thread_id,
        "checkpoint_ids": checkpoint_ids,
        "latest_config": parent_config,
    }


@pytest.mark.parametrize("saver_name", ["base", "pool"])
def test_combined_metadata(saver_name: str, test_data) -> None:
    with _sync_saver(saver_name) as saver:
        config = {
            "configurable": {
                "thread_id": "thread-2",
                "checkpoint_ns": "",
                "__internal_config_marker": "internal_config_value",
            },
            "metadata": {"run_id": "my_run_id"},
        }
        checkpoint: Checkpoint = create_checkpoint(empty_checkpoint(), {}, 1)
        metadata: CheckpointMetadata = {
            "source": "loop",
            "step": 1,
            "writes": {"foo": "bar"},
            "score": None,
        }
        saver.put(config, checkpoint, metadata, {})
        loaded = saver.get_tuple(config)
        assert loaded.metadata == {
            **metadata,
            "run_id": "my_run_id",
        }


@pytest.mark.parametrize("saver_name", ["base", "pool"])
def test_search(saver_name: str, test_data) -> None:
    with _sync_saver(saver_name) as saver:
        configs = test_data["configs"]
        checkpoints = test_data["checkpoints"]
        metadata = test_data["metadata"]

        saver.put(configs[0], checkpoints[0], metadata[0], {})
        saver.put(configs[1], checkpoints[1], metadata[1], {})
        saver.put(configs[2], checkpoints[2], metadata[2], {})

        query_1 = {"source": "input"}
        query_2 = {"step": 1, "writes": {"foo": "bar"}}
        query_3 = {}
        query_4 = {"source": "update", "step": 1}

        search_results_1 = list(saver.list(None, filter=query_1))
        assert len(search_results_1) == 1
        assert search_results_1[0].metadata == {
            **_exclude_keys(configs[0]["configurable"]),
            **metadata[0],
        }

        search_results_2 = list(saver.list(None, filter=query_2))
        assert len(search_results_2) == 1
        assert search_results_2[0].metadata == {
            **_exclude_keys(configs[1]["configurable"]),
            **metadata[1],
        }

        assert len(list(saver.list(None, filter=query_3))) == 3
        assert len(list(saver.list(None, filter=query_4))) == 0

        search_results_5 = list(saver.list({"configurable": {"thread_id": "thread-2"}}))
        assert len(search_results_5) == 2
        assert {
            search_results_5[0].config["configurable"]["checkpoint_ns"],
            search_results_5[1].config["configurable"]["checkpoint_ns"],
        } == {"", "inner"}


@pytest.mark.parametrize("saver_name", ["base", "pool"])
def test_null_chars(saver_name: str, test_data) -> None:
    with _sync_saver(saver_name) as saver:
        config = saver.put(
            test_data["configs"][0],
            test_data["checkpoints"][0],
            {"my_key": "\x00abc"},
            {},
        )
        assert saver.get_tuple(config).metadata["my_key"] == "abc"  # type: ignore[index]
        assert (
            list(saver.list(None, filter={"my_key": "abc"}))[0].metadata["my_key"]
            == "abc"
        )


def test_nonnull_migrations() -> None:
    leading_comment_remover = re.compile(r"^/\*.*?\*/")
    for migration in OracleSaver.MIGRATIONS:
        statement = leading_comment_remover.sub("", migration).split()[0]
        assert statement.strip()


@pytest.mark.parametrize("saver_name", ["base", "pool"])
def test_pending_sends_migration(saver_name: str) -> None:
    with _sync_saver(saver_name) as saver:
        config = {
            "configurable": {
                "thread_id": "thread-1",
                "checkpoint_ns": "",
            }
        }

        checkpoint_0 = empty_checkpoint()
        config = saver.put(config, checkpoint_0, {}, {})
        saver.put_writes(
            config, [(TASKS, "send-1"), (TASKS, "send-2")], task_id="task-1"
        )
        saver.put_writes(config, [(TASKS, "send-3")], task_id="task-2")

        tuple_0 = saver.get_tuple(config)
        assert tuple_0.checkpoint["channel_values"] == {}
        assert tuple_0.checkpoint["channel_versions"] == {}

        checkpoint_1 = create_checkpoint(checkpoint_0, {}, 1)
        config = saver.put(config, checkpoint_1, {}, {})

        loaded = saver.get_tuple(config)
        assert loaded.checkpoint["channel_values"] == {
            TASKS: ["send-1", "send-2", "send-3"]
        }
        assert TASKS in loaded.checkpoint["channel_versions"]

        search_results = [
            c for c in saver.list({"configurable": {"thread_id": "thread-1"}})
        ]
        assert len(search_results) == 2
        assert search_results[-1].checkpoint["channel_values"] == {}
        assert search_results[-1].checkpoint["channel_versions"] == {}
        assert search_results[0].checkpoint["channel_values"] == {
            TASKS: ["send-1", "send-2", "send-3"]
        }
        assert TASKS in search_results[0].checkpoint["channel_versions"]


@pytest.mark.parametrize("saver_name", ["base", "pool"])
def test_list_includes_pending_writes(saver_name: str) -> None:
    with _sync_saver(saver_name) as saver:
        config = {
            "configurable": {
                "thread_id": "thread-pending-writes",
                "checkpoint_ns": "",
            }
        }

        checkpoint = empty_checkpoint()
        stored = saver.put(config, checkpoint, {}, {})
        saver.put_writes(stored, [("ch", "val")], task_id="task-1")

        results = list(
            saver.list({"configurable": {"thread_id": "thread-pending-writes"}})
        )

        assert len(results) == 1
        assert results[0].pending_writes is not None
        assert len(results[0].pending_writes) == 1
        assert results[0].pending_writes[0][0] == "task-1"
        assert results[0].pending_writes[0][1] == "ch"
        assert results[0].pending_writes[0][2] == "val"


@pytest.mark.parametrize("saver_name", ["base", "pool"])
def test_get_checkpoint_no_channel_values(monkeypatch, saver_name: str) -> None:
    with _sync_saver(saver_name) as saver:
        config = {
            "configurable": {
                "thread_id": "thread-2",
                "checkpoint_ns": "",
                "__internal_config_marker": "internal_config_value",
            },
        }
        checkpoint: Checkpoint = create_checkpoint(empty_checkpoint(), {}, 1)
        saver.put(config, checkpoint, {}, {})

        load_checkpoint_tuple = saver._load_checkpoint_tuple

        def patched_load_checkpoint_tuple(value):
            value["checkpoint"].pop("channel_values", None)
            return load_checkpoint_tuple(value)

        monkeypatch.setattr(
            saver, "_load_checkpoint_tuple", patched_load_checkpoint_tuple
        )

        loaded = saver.get_tuple(config)
        assert loaded.checkpoint["channel_values"] == {}


@pytest.mark.parametrize("saver_name", ["base", "pool"])
def test_put_returns_config(saver_name: str) -> None:
    with _sync_saver(saver_name) as saver:
        config = generate_config()
        checkpoint = generate_checkpoint(channel_values={"k": "v"})
        checkpoint["channel_versions"] = {"k": 1}

        result = saver.put(config, checkpoint, generate_metadata(), {"k": 1})

        assert "configurable" in result
        assert (
            result["configurable"]["thread_id"] == config["configurable"]["thread_id"]
        )
        assert result["configurable"]["checkpoint_ns"] == ""
        assert result["configurable"]["checkpoint_id"] == checkpoint["id"]


@pytest.mark.parametrize("saver_name", ["base", "pool"])
def test_put_roundtrip(saver_name: str) -> None:
    with _sync_saver(saver_name) as saver:
        config = generate_config()
        checkpoint = generate_checkpoint(channel_values={"msg": "hello"})
        checkpoint["channel_versions"] = {"msg": 1}

        stored = saver.put(
            config,
            checkpoint,
            generate_metadata(source="input", step=-1),
            {"msg": 1},
        )

        loaded = saver.get_tuple(stored)
        assert loaded is not None
        assert loaded.checkpoint["id"] == checkpoint["id"]
        assert loaded.checkpoint["channel_values"] == {"msg": "hello"}


@pytest.mark.parametrize("saver_name", ["base", "pool"])
def test_put_preserves_channel_values(saver_name: str) -> None:
    with _sync_saver(saver_name) as saver:
        values = {
            "str_val": "hello",
            "int_val": 42,
            "list_val": [1, 2, 3],
            "dict_val": {"nested": True},
        }
        checkpoint = generate_checkpoint(channel_values=values)
        versions = {key: 1 for key in values}
        checkpoint["channel_versions"] = versions

        stored = saver.put(
            config=generate_config(),
            checkpoint=checkpoint,
            metadata=generate_metadata(),
            new_versions=versions,
        )
        loaded = saver.get_tuple(stored)

        assert loaded is not None
        for key, value in values.items():
            assert loaded.checkpoint["channel_values"][key] == value


@pytest.mark.parametrize("saver_name", ["base", "pool"])
def test_put_preserves_channel_versions(saver_name: str) -> None:
    with _sync_saver(saver_name) as saver:
        versions: ChannelVersions = {"a": 1, "b": 2}
        checkpoint = generate_checkpoint(
            channel_values={"a": "x", "b": "y"},
            channel_versions=versions,
        )

        stored = saver.put(generate_config(), checkpoint, generate_metadata(), versions)
        loaded = saver.get_tuple(stored)

        assert loaded is not None
        for key, expected in versions.items():
            actual = loaded.checkpoint["channel_versions"].get(key)
            assert actual is not None
            assert version_matches(actual, expected)


@pytest.mark.parametrize("saver_name", ["base", "pool"])
def test_put_preserves_versions_seen(saver_name: str) -> None:
    with _sync_saver(saver_name) as saver:
        versions_seen = {"node1": {"ch": 1}, "node2": {"ch": 2}}
        checkpoint = generate_checkpoint(versions_seen=versions_seen)

        stored = saver.put(generate_config(), checkpoint, generate_metadata(), {})
        loaded = saver.get_tuple(stored)

        assert loaded is not None
        for node in versions_seen:
            assert node in loaded.checkpoint["versions_seen"]


@pytest.mark.parametrize("saver_name", ["base", "pool"])
def test_put_preserves_metadata(saver_name: str) -> None:
    with _sync_saver(saver_name) as saver:
        metadata = generate_metadata(source="loop", step=3, custom_key="custom_value")

        stored = saver.put(generate_config(), generate_checkpoint(), metadata, {})
        loaded = saver.get_tuple(stored)

        assert loaded is not None
        assert loaded.metadata["source"] == "loop"
        assert loaded.metadata["step"] == 3
        assert loaded.metadata["custom_key"] == "custom_value"


@pytest.mark.parametrize("saver_name", ["base", "pool"])
def test_put_root_namespace(saver_name: str) -> None:
    with _sync_saver(saver_name) as saver:
        stored = saver.put(
            generate_config(checkpoint_ns=""),
            generate_checkpoint(),
            generate_metadata(),
            {},
        )
        loaded = saver.get_tuple(stored)

        assert loaded is not None
        assert loaded.config["configurable"].get("checkpoint_ns", "") == ""


@pytest.mark.parametrize("saver_name", ["base", "pool"])
def test_put_child_namespace(saver_name: str) -> None:
    with _sync_saver(saver_name) as saver:
        stored = saver.put(
            generate_config(checkpoint_ns="child:abc"),
            generate_checkpoint(),
            generate_metadata(),
            {},
        )
        loaded = saver.get_tuple(stored)

        assert loaded is not None
        assert loaded.config["configurable"]["checkpoint_ns"] == "child:abc"


@pytest.mark.parametrize("saver_name", ["base", "pool"])
def test_put_default_namespace(saver_name: str) -> None:
    with _sync_saver(saver_name) as saver:
        thread_id = str(uuid4())
        config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}

        stored = saver.put(config, generate_checkpoint(), generate_metadata(), {})

        assert saver.get_tuple(stored) is not None


@pytest.mark.parametrize("saver_name", ["base", "pool"])
def test_put_multiple_checkpoints_same_thread(saver_name: str) -> None:
    with _sync_saver(saver_name) as saver:
        thread_id = str(uuid4())
        checkpoint_ids = []
        parent_config = None

        for step in range(3):
            config = generate_config(thread_id)
            if parent_config is not None:
                config["configurable"]["checkpoint_id"] = parent_config["configurable"][
                    "checkpoint_id"
                ]
            checkpoint = generate_checkpoint()
            parent_config = saver.put(
                config, checkpoint, generate_metadata(step=step), {}
            )
            checkpoint_ids.append(checkpoint["id"])

        for checkpoint_id in checkpoint_ids:
            loaded = saver.get_tuple(
                generate_config(thread_id, checkpoint_id=checkpoint_id)
            )
            assert loaded is not None
            assert loaded.checkpoint["id"] == checkpoint_id


@pytest.mark.parametrize("saver_name", ["base", "pool"])
def test_put_multiple_threads_isolated(saver_name: str) -> None:
    with _sync_saver(saver_name) as saver:
        thread_1, thread_2 = str(uuid4()), str(uuid4())
        checkpoint_1 = generate_checkpoint(channel_values={"x": "thread1"})
        checkpoint_1["channel_versions"] = {"x": 1}
        checkpoint_2 = generate_checkpoint(channel_values={"x": "thread2"})
        checkpoint_2["channel_versions"] = {"x": 1}

        saver.put(
            generate_config(thread_1), checkpoint_1, generate_metadata(), {"x": 1}
        )
        saver.put(
            generate_config(thread_2), checkpoint_2, generate_metadata(), {"x": 1}
        )

        loaded_1 = saver.get_tuple(generate_config(thread_1))
        loaded_2 = saver.get_tuple(generate_config(thread_2))

        assert loaded_1 is not None and loaded_2 is not None
        assert loaded_1.checkpoint["channel_values"]["x"] == "thread1"
        assert loaded_2.checkpoint["channel_values"]["x"] == "thread2"


@pytest.mark.parametrize("saver_name", ["base", "pool"])
def test_put_parent_config(saver_name: str) -> None:
    with _sync_saver(saver_name) as saver:
        thread_id = str(uuid4())
        first = saver.put(
            generate_config(thread_id),
            generate_checkpoint(),
            generate_metadata(step=0),
            {},
        )
        second_config = generate_config(thread_id)
        second_config["configurable"]["checkpoint_id"] = first["configurable"][
            "checkpoint_id"
        ]
        second = saver.put(
            second_config,
            generate_checkpoint(),
            generate_metadata(step=1),
            {},
        )

        loaded = saver.get_tuple(second)
        assert loaded is not None
        assert loaded.parent_config is not None
        assert (
            loaded.parent_config["configurable"]["checkpoint_id"]
            == first["configurable"]["checkpoint_id"]
        )


@pytest.mark.parametrize("saver_name", ["base", "pool"])
def test_put_incremental_channel_update(saver_name: str) -> None:
    with _sync_saver(saver_name) as saver:
        thread_id = str(uuid4())
        first_checkpoint = generate_checkpoint(
            channel_values={"a": "v1", "b": "v2"},
            channel_versions={"a": 1, "b": 1},
        )
        first = saver.put(
            generate_config(thread_id),
            first_checkpoint,
            generate_metadata(step=0),
            {"a": 1, "b": 1},
        )

        second_config = generate_config(thread_id)
        second_config["configurable"]["checkpoint_id"] = first["configurable"][
            "checkpoint_id"
        ]
        second_checkpoint = generate_checkpoint(
            channel_values={"a": "v1_updated", "b": "v2"},
            channel_versions={"a": 2, "b": 1},
        )
        second = saver.put(
            second_config,
            second_checkpoint,
            generate_metadata(step=1),
            {"a": 2},
        )

        loaded_second = saver.get_tuple(second)
        loaded_first = saver.get_tuple(first)

        assert loaded_second is not None
        assert loaded_second.checkpoint["channel_values"]["a"] == "v1_updated"
        assert loaded_second.checkpoint["channel_values"]["b"] == "v2"
        assert loaded_first is not None
        assert loaded_first.checkpoint["channel_values"]["a"] == "v1"
        assert loaded_first.checkpoint["channel_values"]["b"] == "v2"


@pytest.mark.parametrize("saver_name", ["base", "pool"])
def test_put_new_channel_added(saver_name: str) -> None:
    with _sync_saver(saver_name) as saver:
        thread_id = str(uuid4())
        first = saver.put(
            generate_config(thread_id),
            generate_checkpoint(
                channel_values={"a": "v1"},
                channel_versions={"a": 1},
            ),
            generate_metadata(step=0),
            {"a": 1},
        )

        second_config = generate_config(thread_id)
        second_config["configurable"]["checkpoint_id"] = first["configurable"][
            "checkpoint_id"
        ]
        second = saver.put(
            second_config,
            generate_checkpoint(
                channel_values={"a": "v1", "b": "new_channel"},
                channel_versions={"a": 1, "b": 1},
            ),
            generate_metadata(step=1),
            {"b": 1},
        )

        loaded = saver.get_tuple(second)
        assert loaded is not None
        assert loaded.checkpoint["channel_values"]["a"] == "v1"
        assert loaded.checkpoint["channel_values"]["b"] == "new_channel"


@pytest.mark.parametrize("saver_name", ["base", "pool"])
def test_put_channel_removed(saver_name: str) -> None:
    with _sync_saver(saver_name) as saver:
        thread_id = str(uuid4())
        first = saver.put(
            generate_config(thread_id),
            generate_checkpoint(
                channel_values={"a": "v1", "b": "v2"},
                channel_versions={"a": 1, "b": 1},
            ),
            generate_metadata(step=0),
            {"a": 1, "b": 1},
        )

        second_config = generate_config(thread_id)
        second_config["configurable"]["checkpoint_id"] = first["configurable"][
            "checkpoint_id"
        ]
        second = saver.put(
            second_config,
            generate_checkpoint(
                channel_values={"a": "v1_updated"},
                channel_versions={"a": 2},
            ),
            generate_metadata(step=1),
            {"a": 2},
        )

        loaded = saver.get_tuple(second)
        assert loaded is not None
        assert loaded.checkpoint["channel_values"]["a"] == "v1_updated"
        assert "b" not in loaded.checkpoint["channel_values"]


@pytest.mark.parametrize("saver_name", ["base", "pool"])
def test_put_preserves_run_id(saver_name: str) -> None:
    with _sync_saver(saver_name) as saver:
        run_id = str(uuid4())
        stored = saver.put(
            generate_config(),
            generate_checkpoint(),
            generate_metadata(source="loop", step=0, run_id=run_id),
            {},
        )
        loaded = saver.get_tuple(stored)

        assert loaded is not None
        assert loaded.metadata["run_id"] == run_id


@pytest.mark.parametrize("saver_name", ["base", "pool"])
def test_put_preserves_versions_seen_values(saver_name: str) -> None:
    with _sync_saver(saver_name) as saver:
        versions_seen = {
            "node1": {"ch_a": 1, "ch_b": 2},
            "node2": {"ch_a": 3},
        }
        stored = saver.put(
            generate_config(),
            generate_checkpoint(versions_seen=versions_seen),
            generate_metadata(),
            {},
        )
        loaded = saver.get_tuple(stored)

        assert loaded is not None
        for node, expected_versions in versions_seen.items():
            actual_versions = loaded.checkpoint["versions_seen"][node]
            for channel, expected in expected_versions.items():
                assert version_matches(actual_versions[channel], expected)


@pytest.mark.parametrize("saver_name", ["base", "pool"])
def test_get_tuple_nonexistent_returns_none(saver_name: str) -> None:
    with _sync_saver(saver_name) as saver:
        assert saver.get_tuple(generate_config(str(uuid4()))) is None


@pytest.mark.parametrize("saver_name", ["base", "pool"])
def test_get_tuple_latest_when_no_checkpoint_id(saver_name: str) -> None:
    with _sync_saver(saver_name) as saver:
        thread_id = str(uuid4())
        checkpoint_ids = []
        parent_config = None

        for step in range(3):
            config = generate_config(thread_id)
            if parent_config:
                config["configurable"]["checkpoint_id"] = parent_config["configurable"][
                    "checkpoint_id"
                ]
            checkpoint = generate_checkpoint()
            parent_config = saver.put(
                config, checkpoint, generate_metadata(step=step), {}
            )
            checkpoint_ids.append(checkpoint["id"])

        loaded = saver.get_tuple(generate_config(thread_id))
        assert loaded is not None
        assert loaded.checkpoint["id"] == checkpoint_ids[-1]
        assert loaded.metadata["step"] == 2


@pytest.mark.parametrize("saver_name", ["base", "pool"])
def test_get_tuple_specific_checkpoint_id(saver_name: str) -> None:
    with _sync_saver(saver_name) as saver:
        thread_id = str(uuid4())
        first = saver.put(
            generate_config(thread_id),
            generate_checkpoint(),
            generate_metadata(step=0),
            {},
        )
        second_config = generate_config(thread_id)
        second_config["configurable"]["checkpoint_id"] = first["configurable"][
            "checkpoint_id"
        ]
        saver.put(
            second_config,
            generate_checkpoint(),
            generate_metadata(step=1),
            {},
        )

        loaded = saver.get_tuple(first)
        assert loaded is not None
        assert loaded.checkpoint["id"] == first["configurable"]["checkpoint_id"]


@pytest.mark.parametrize("saver_name", ["base", "pool"])
def test_get_tuple_config_structure(saver_name: str) -> None:
    with _sync_saver(saver_name) as saver:
        thread_id = str(uuid4())
        checkpoint = generate_checkpoint()
        stored = saver.put(
            generate_config(thread_id),
            checkpoint,
            generate_metadata(),
            {},
        )
        loaded = saver.get_tuple(stored)

        assert loaded is not None
        assert loaded.config["configurable"]["thread_id"] == thread_id
        assert loaded.config["configurable"].get("checkpoint_ns", "") == ""
        assert loaded.config["configurable"]["checkpoint_id"] == checkpoint["id"]


@pytest.mark.parametrize("saver_name", ["base", "pool"])
def test_get_tuple_checkpoint_fields(saver_name: str) -> None:
    with _sync_saver(saver_name) as saver:
        checkpoint = generate_checkpoint(channel_values={"k": "v"})
        checkpoint["channel_versions"] = {"k": 1}
        stored = saver.put(
            generate_config(str(uuid4())),
            checkpoint,
            generate_metadata(),
            {"k": 1},
        )
        loaded = saver.get_tuple(stored)

        assert loaded is not None
        assert loaded.checkpoint["id"] == checkpoint["id"]
        assert loaded.checkpoint["v"] == 1
        assert loaded.checkpoint["ts"]
        assert loaded.checkpoint["channel_values"] == {"k": "v"}
        assert "channel_versions" in loaded.checkpoint
        assert "versions_seen" in loaded.checkpoint


@pytest.mark.parametrize("saver_name", ["base", "pool"])
def test_get_tuple_metadata(saver_name: str) -> None:
    with _sync_saver(saver_name) as saver:
        stored = saver.put(
            generate_config(str(uuid4())),
            generate_checkpoint(),
            generate_metadata(source="input", step=-1),
            {},
        )
        loaded = saver.get_tuple(stored)

        assert loaded is not None
        assert loaded.metadata["source"] == "input"
        assert loaded.metadata["step"] == -1


@pytest.mark.parametrize("saver_name", ["base", "pool"])
def test_get_tuple_parent_config(saver_name: str) -> None:
    with _sync_saver(saver_name) as saver:
        thread_id = str(uuid4())
        first = saver.put(
            generate_config(thread_id),
            generate_checkpoint(),
            generate_metadata(step=0),
            {},
        )
        first_loaded = saver.get_tuple(first)
        assert first_loaded is not None
        assert first_loaded.parent_config is None

        second_config = generate_config(thread_id)
        second_config["configurable"]["checkpoint_id"] = first["configurable"][
            "checkpoint_id"
        ]
        second = saver.put(
            second_config,
            generate_checkpoint(),
            generate_metadata(step=1),
            {},
        )
        second_loaded = saver.get_tuple(second)

        assert second_loaded is not None
        assert second_loaded.parent_config is not None
        assert (
            second_loaded.parent_config["configurable"]["checkpoint_id"]
            == first["configurable"]["checkpoint_id"]
        )


@pytest.mark.parametrize("saver_name", ["base", "pool"])
def test_get_tuple_parent_config_root_namespace_normalized(saver_name: str) -> None:
    with _sync_saver(saver_name) as saver:
        thread_id = str(uuid4())
        first = saver.put(
            generate_config(thread_id, checkpoint_ns=""),
            generate_checkpoint(),
            generate_metadata(step=0),
            {},
        )
        second_config = generate_config(thread_id, checkpoint_ns="")
        second_config["configurable"]["checkpoint_id"] = first["configurable"][
            "checkpoint_id"
        ]
        second = saver.put(
            second_config,
            generate_checkpoint(),
            generate_metadata(step=1),
            {},
        )

        loaded = saver.get_tuple(second)
        assert loaded is not None
        assert loaded.parent_config is not None
        assert loaded.parent_config["configurable"]["checkpoint_ns"] == ""


@pytest.mark.parametrize("saver_name", ["base", "pool"])
def test_get_tuple_pending_writes(saver_name: str) -> None:
    with _sync_saver(saver_name) as saver:
        stored = saver.put(
            generate_config(str(uuid4())),
            generate_checkpoint(),
            generate_metadata(),
            {},
        )
        task_id = str(uuid4())
        saver.put_writes(stored, [("ch", "val")], task_id)

        loaded = saver.get_tuple(stored)
        assert loaded is not None
        assert loaded.pending_writes is not None
        assert len(loaded.pending_writes) == 1
        assert loaded.pending_writes[0] == (task_id, "ch", "val")


@pytest.mark.parametrize("saver_name", ["base", "pool"])
def test_get_tuple_respects_namespace(saver_name: str) -> None:
    with _sync_saver(saver_name) as saver:
        thread_id = str(uuid4())
        root_checkpoint = generate_checkpoint()
        root_stored = saver.put(
            generate_config(thread_id, checkpoint_ns=""),
            root_checkpoint,
            generate_metadata(),
            {},
        )
        child_checkpoint = generate_checkpoint()
        child_stored = saver.put(
            generate_config(thread_id, checkpoint_ns="child:1"),
            child_checkpoint,
            generate_metadata(),
            {},
        )

        root_loaded = saver.get_tuple(root_stored)
        child_loaded = saver.get_tuple(child_stored)
        assert root_loaded is not None and child_loaded is not None
        assert root_loaded.checkpoint["id"] == root_checkpoint["id"]
        assert child_loaded.checkpoint["id"] == child_checkpoint["id"]


@pytest.mark.parametrize("saver_name", ["base", "pool"])
def test_get_tuple_nonexistent_checkpoint_id(saver_name: str) -> None:
    with _sync_saver(saver_name) as saver:
        thread_id = str(uuid4())
        saver.put(
            generate_config(thread_id),
            generate_checkpoint(),
            generate_metadata(),
            {},
        )

        assert (
            saver.get_tuple(generate_config(thread_id, checkpoint_id=str(uuid4())))
            is None
        )


@pytest.mark.parametrize("saver_name", ["base", "pool"])
def test_list_all(saver_name: str) -> None:
    with _sync_saver(saver_name) as saver:
        data = _setup_list_data(saver)

        results = list(saver.list(generate_config(data["thread_id"])))
        assert len(results) == 4


@pytest.mark.parametrize("saver_name", ["base", "pool"])
def test_list_by_thread(saver_name: str) -> None:
    with _sync_saver(saver_name) as saver:
        data = _setup_list_data(saver)

        assert len(list(saver.list(generate_config(str(uuid4()))))) == 0
        assert len(list(saver.list(generate_config(data["thread_id"])))) == 4


@pytest.mark.parametrize("saver_name", ["base", "pool"])
def test_list_by_namespace(saver_name: str) -> None:
    with _sync_saver(saver_name) as saver:
        thread_id = str(uuid4())
        saver.put(
            generate_config(thread_id, checkpoint_ns=""),
            generate_checkpoint(),
            generate_metadata(),
            {},
        )
        saver.put(
            generate_config(thread_id, checkpoint_ns="child:1"),
            generate_checkpoint(),
            generate_metadata(),
            {},
        )

        assert len(list(saver.list(generate_config(thread_id, checkpoint_ns="")))) == 1
        assert (
            len(list(saver.list(generate_config(thread_id, checkpoint_ns="child:1"))))
            == 1
        )


@pytest.mark.parametrize("saver_name", ["base", "pool"])
def test_list_ordering(saver_name: str) -> None:
    with _sync_saver(saver_name) as saver:
        data = _setup_list_data(saver)

        results = [
            checkpoint.checkpoint["id"]
            for checkpoint in saver.list(generate_config(data["thread_id"]))
        ]
        assert results == list(reversed(data["checkpoint_ids"]))


@pytest.mark.parametrize("saver_name", ["base", "pool"])
def test_list_metadata_filter_single_key(saver_name: str) -> None:
    with _sync_saver(saver_name) as saver:
        data = _setup_list_data(saver)

        results = list(
            saver.list(generate_config(data["thread_id"]), filter={"source": "input"})
        )
        assert len(results) == 2
        for result in results:
            assert result.metadata["source"] == "input"


@pytest.mark.parametrize("saver_name", ["base", "pool"])
def test_list_metadata_filter_step(saver_name: str) -> None:
    with _sync_saver(saver_name) as saver:
        data = _setup_list_data(saver)

        results = list(
            saver.list(generate_config(data["thread_id"]), filter={"step": 1})
        )
        assert len(results) == 1
        assert results[0].metadata["step"] == 1


@pytest.mark.parametrize("saver_name", ["base", "pool"])
def test_list_metadata_filter_multiple_keys(saver_name: str) -> None:
    with _sync_saver(saver_name) as saver:
        thread_id = str(uuid4())
        for source, step in [("input", 1), ("loop", 1), ("input", 2)]:
            saver.put(
                generate_config(thread_id),
                generate_checkpoint(),
                generate_metadata(source=source, step=step),
                {},
            )

        results = list(
            saver.list(
                generate_config(thread_id),
                filter={"source": "input", "step": 2},
            )
        )
        assert len(results) == 1
        assert results[0].metadata["source"] == "input"
        assert results[0].metadata["step"] == 2


@pytest.mark.parametrize("saver_name", ["base", "pool"])
def test_list_metadata_filter_no_match(saver_name: str) -> None:
    with _sync_saver(saver_name) as saver:
        data = _setup_list_data(saver)

        assert (
            len(
                list(
                    saver.list(
                        generate_config(data["thread_id"]),
                        filter={"source": "update", "step": 99},
                    )
                )
            )
            == 0
        )


@pytest.mark.parametrize("saver_name", ["base", "pool"])
def test_list_metadata_custom_keys(saver_name: str) -> None:
    with _sync_saver(saver_name) as saver:
        thread_id = str(uuid4())
        saver.put(
            generate_config(thread_id),
            generate_checkpoint(),
            generate_metadata(score=42, run_id="run-abc"),
            {},
        )
        saver.put(
            generate_config(thread_id),
            generate_checkpoint(),
            generate_metadata(score=99, run_id="run-xyz"),
            {},
        )

        results = list(saver.list(generate_config(thread_id), filter={"score": 42}))
        assert len(results) == 1
        assert results[0].metadata["score"] == 42
        assert results[0].metadata["run_id"] == "run-abc"


@pytest.mark.parametrize("saver_name", ["base", "pool"])
def test_list_before(saver_name: str) -> None:
    with _sync_saver(saver_name) as saver:
        data = _setup_list_data(saver)
        before = generate_config(
            data["thread_id"], checkpoint_id=data["checkpoint_ids"][2]
        )

        results = list(saver.list(generate_config(data["thread_id"]), before=before))
        result_ids = [result.checkpoint["id"] for result in results]

        assert data["checkpoint_ids"][2] not in result_ids
        assert data["checkpoint_ids"][3] not in result_ids
        assert set(result_ids) == {data["checkpoint_ids"][0], data["checkpoint_ids"][1]}


@pytest.mark.parametrize("saver_name", ["base", "pool"])
def test_list_limit(saver_name: str) -> None:
    with _sync_saver(saver_name) as saver:
        data = _setup_list_data(saver)

        assert len(list(saver.list(generate_config(data["thread_id"]), limit=1))) == 1
        assert len(list(saver.list(generate_config(data["thread_id"]), limit=2))) == 2


@pytest.mark.parametrize("saver_name", ["base", "pool"])
def test_list_limit_plus_before(saver_name: str) -> None:
    with _sync_saver(saver_name) as saver:
        data = _setup_list_data(saver)
        before = generate_config(
            data["thread_id"], checkpoint_id=data["checkpoint_ids"][3]
        )

        results = list(
            saver.list(generate_config(data["thread_id"]), before=before, limit=1)
        )
        assert len(results) == 1
        assert results[0].checkpoint["id"] == data["checkpoint_ids"][2]


@pytest.mark.parametrize("saver_name", ["base", "pool"])
def test_list_combined_thread_and_filter(saver_name: str) -> None:
    with _sync_saver(saver_name) as saver:
        data = _setup_list_data(saver)

        results = list(
            saver.list(generate_config(data["thread_id"]), filter={"source": "loop"})
        )
        assert len(results) == 2
        for result in results:
            assert result.metadata["source"] == "loop"


@pytest.mark.parametrize("saver_name", ["base", "pool"])
def test_list_empty_result(saver_name: str) -> None:
    with _sync_saver(saver_name) as saver:
        assert (
            len(
                list(
                    saver.list(
                        generate_config(str(uuid4())), filter={"source": "nonexistent"}
                    )
                )
            )
            == 0
        )


@pytest.mark.parametrize("saver_name", ["base", "pool"])
def test_list_multiple_namespaces(saver_name: str) -> None:
    with _sync_saver(saver_name) as saver:
        thread_id = str(uuid4())
        for namespace in ["", "child:1", "child:2"]:
            saver.put(
                generate_config(thread_id, checkpoint_ns=namespace),
                generate_checkpoint(),
                generate_metadata(),
                {},
            )

        results = list(saver.list(generate_config(thread_id, checkpoint_ns="")))
        assert len(results) == 1


@pytest.mark.parametrize("saver_name", ["base", "pool"])
def test_put_writes_basic(saver_name: str) -> None:
    with _sync_saver(saver_name) as saver:
        stored = saver.put(
            generate_config(str(uuid4())),
            generate_checkpoint(),
            generate_metadata(),
            {},
        )
        task_id = str(uuid4())
        saver.put_writes(stored, [("channel1", "value1")], task_id)

        loaded = saver.get_tuple(stored)
        assert loaded is not None
        matching = [
            write
            for write in (loaded.pending_writes or [])
            if write[0] == task_id and write[1] == "channel1"
        ]
        assert len(matching) == 1
        assert matching[0][2] == "value1"


@pytest.mark.parametrize("saver_name", ["base", "pool"])
def test_put_writes_multiple_writes_same_task(saver_name: str) -> None:
    with _sync_saver(saver_name) as saver:
        stored = saver.put(
            generate_config(str(uuid4())),
            generate_checkpoint(),
            generate_metadata(),
            {},
        )
        task_id = str(uuid4())
        writes = [("ch1", "v1"), ("ch2", "v2"), ("ch3", "v3")]
        saver.put_writes(stored, writes, task_id)

        loaded = saver.get_tuple(stored)
        assert loaded is not None
        assert loaded.pending_writes is not None
        assert len(loaded.pending_writes) == 3
        assert {write[1] for write in loaded.pending_writes} == {"ch1", "ch2", "ch3"}
        for channel, value in writes:
            matching = [
                write
                for write in loaded.pending_writes
                if write[0] == task_id and write[1] == channel
            ]
            assert len(matching) == 1
            assert matching[0][2] == value


@pytest.mark.parametrize("saver_name", ["base", "pool"])
def test_put_writes_multiple_tasks(saver_name: str) -> None:
    with _sync_saver(saver_name) as saver:
        stored = saver.put(
            generate_config(str(uuid4())),
            generate_checkpoint(),
            generate_metadata(),
            {},
        )
        task_1, task_2 = str(uuid4()), str(uuid4())
        saver.put_writes(stored, [("ch", "from_t1")], task_1)
        saver.put_writes(stored, [("ch", "from_t2")], task_2)

        loaded = saver.get_tuple(stored)
        assert loaded is not None
        assert loaded.pending_writes is not None
        assert len(loaded.pending_writes) == 2
        assert any(
            write == (task_1, "ch", "from_t1") for write in loaded.pending_writes
        )
        assert any(
            write == (task_2, "ch", "from_t2") for write in loaded.pending_writes
        )


@pytest.mark.parametrize("saver_name", ["base", "pool"])
def test_put_writes_preserves_task_id(saver_name: str) -> None:
    with _sync_saver(saver_name) as saver:
        stored = saver.put(
            generate_config(str(uuid4())),
            generate_checkpoint(),
            generate_metadata(),
            {},
        )
        task_id = str(uuid4())
        saver.put_writes(stored, [("ch", "val")], task_id)

        loaded = saver.get_tuple(stored)
        assert loaded is not None
        assert any(write[0] == task_id for write in (loaded.pending_writes or []))


@pytest.mark.parametrize("saver_name", ["base", "pool"])
def test_put_writes_preserves_channel_and_value(saver_name: str) -> None:
    with _sync_saver(saver_name) as saver:
        stored = saver.put(
            generate_config(str(uuid4())),
            generate_checkpoint(),
            generate_metadata(),
            {},
        )
        task_id = str(uuid4())
        saver.put_writes(stored, [("my_channel", {"data": 123})], task_id)

        loaded = saver.get_tuple(stored)
        assert loaded is not None
        matching = [
            write
            for write in (loaded.pending_writes or [])
            if write[0] == task_id and write[1] == "my_channel"
        ]
        assert len(matching) == 1
        assert matching[0][2] == {"data": 123}


@pytest.mark.parametrize("saver_name", ["base", "pool"])
def test_put_writes_task_path(saver_name: str) -> None:
    with _sync_saver(saver_name) as saver:
        stored = saver.put(
            generate_config(str(uuid4())),
            generate_checkpoint(),
            generate_metadata(),
            {},
        )
        saver.put_writes(stored, [("ch", "v")], str(uuid4()), task_path="a:b:c")

        loaded = saver.get_tuple(stored)
        assert loaded is not None
        assert len(loaded.pending_writes or []) == 1


@pytest.mark.parametrize("saver_name", ["base", "pool"])
def test_put_writes_idempotent(saver_name: str) -> None:
    with _sync_saver(saver_name) as saver:
        stored = saver.put(
            generate_config(str(uuid4())),
            generate_checkpoint(),
            generate_metadata(),
            {},
        )
        task_id = str(uuid4())
        saver.put_writes(stored, [("ch", "val")], task_id)
        saver.put_writes(stored, [("ch", "val")], task_id)

        loaded = saver.get_tuple(stored)
        assert loaded is not None
        matching = [
            write
            for write in (loaded.pending_writes or [])
            if write[0] == task_id and write[1] == "ch"
        ]
        assert len(loaded.pending_writes or []) == 1
        assert len(matching) == 1


@pytest.mark.parametrize("saver_name", ["base", "pool"])
def test_put_writes_special_channels(saver_name: str) -> None:
    with _sync_saver(saver_name) as saver:
        stored = saver.put(
            generate_config(str(uuid4())),
            generate_checkpoint(),
            generate_metadata(),
            {},
        )
        task_id = str(uuid4())
        saver.put_writes(
            stored,
            [(ERROR, "something went wrong"), (INTERRUPT, {"reason": "human_input"})],
            task_id,
        )

        loaded = saver.get_tuple(stored)
        assert loaded is not None
        writes = loaded.pending_writes or []
        assert any(
            write == (task_id, ERROR, "something went wrong") for write in writes
        )
        assert any(
            write == (task_id, INTERRUPT, {"reason": "human_input"}) for write in writes
        )


@pytest.mark.parametrize("saver_name", ["base", "pool"])
def test_put_writes_across_namespaces(saver_name: str) -> None:
    with _sync_saver(saver_name) as saver:
        thread_id = str(uuid4())
        root = saver.put(
            generate_config(thread_id, checkpoint_ns=""),
            generate_checkpoint(),
            generate_metadata(),
            {},
        )
        root_task = str(uuid4())
        saver.put_writes(root, [("ch", "root_val")], root_task)

        child = saver.put(
            generate_config(thread_id, checkpoint_ns="child:1"),
            generate_checkpoint(),
            generate_metadata(),
            {},
        )
        child_task = str(uuid4())
        saver.put_writes(child, [("ch", "child_val")], child_task)

        root_loaded = saver.get_tuple(root)
        child_loaded = saver.get_tuple(child)
        assert root_loaded is not None and child_loaded is not None
        assert [
            write for write in (root_loaded.pending_writes or []) if write[1] == "ch"
        ] == [(root_task, "ch", "root_val")]
        assert [
            write for write in (child_loaded.pending_writes or []) if write[1] == "ch"
        ] == [(child_task, "ch", "child_val")]


@pytest.mark.parametrize("saver_name", ["base", "pool"])
def test_put_writes_cleared_on_next_checkpoint(saver_name: str) -> None:
    with _sync_saver(saver_name) as saver:
        thread_id = str(uuid4())
        first = saver.put(
            generate_config(thread_id),
            generate_checkpoint(),
            generate_metadata(step=0),
            {},
        )
        saver.put_writes(first, [("ch", "old_write")], str(uuid4()))

        second_config = generate_config(thread_id)
        second_config["configurable"]["checkpoint_id"] = first["configurable"][
            "checkpoint_id"
        ]
        second = saver.put(
            second_config,
            generate_checkpoint(),
            generate_metadata(step=1),
            {},
        )

        loaded = saver.get_tuple(second)
        assert loaded is not None
        assert len(loaded.pending_writes or []) == 0


@pytest.mark.parametrize("saver_name", ["base", "pool"])
def test_delete_thread_removes_checkpoints(saver_name: str) -> None:
    with _sync_saver(saver_name) as saver:
        thread_id = str(uuid4())
        parent_config = None
        for step in range(3):
            config = generate_config(thread_id)
            if parent_config:
                config["configurable"]["checkpoint_id"] = parent_config["configurable"][
                    "checkpoint_id"
                ]
            parent_config = saver.put(
                config, generate_checkpoint(), generate_metadata(step=step), {}
            )

        assert saver.get_tuple(generate_config(thread_id)) is not None
        saver.delete_thread(thread_id)
        assert saver.get_tuple(generate_config(thread_id)) is None
        assert len(list(saver.list(generate_config(thread_id)))) == 0


@pytest.mark.parametrize("saver_name", ["base", "pool"])
def test_delete_thread_removes_writes(saver_name: str) -> None:
    with _sync_saver(saver_name) as saver:
        thread_id = str(uuid4())
        stored = saver.put(
            generate_config(thread_id),
            generate_checkpoint(),
            generate_metadata(),
            {},
        )
        saver.put_writes(stored, [("ch", "val")], str(uuid4()))

        pre_delete = saver.get_tuple(generate_config(thread_id))
        assert pre_delete is not None
        assert len(pre_delete.pending_writes or []) == 1

        saver.delete_thread(thread_id)
        assert saver.get_tuple(generate_config(thread_id)) is None


@pytest.mark.parametrize("saver_name", ["base", "pool"])
def test_delete_thread_removes_all_namespaces(saver_name: str) -> None:
    with _sync_saver(saver_name) as saver:
        thread_id = str(uuid4())
        for namespace in ["", "child:1"]:
            saver.put(
                generate_config(thread_id, checkpoint_ns=namespace),
                generate_checkpoint(),
                generate_metadata(),
                {},
            )

        for namespace in ["", "child:1"]:
            assert saver.get_tuple(generate_config(thread_id, checkpoint_ns=namespace))
        saver.delete_thread(thread_id)
        for namespace in ["", "child:1"]:
            assert (
                saver.get_tuple(generate_config(thread_id, checkpoint_ns=namespace))
                is None
            )


@pytest.mark.parametrize("saver_name", ["base", "pool"])
def test_delete_thread_preserves_other_threads(saver_name: str) -> None:
    with _sync_saver(saver_name) as saver:
        thread_1, thread_2 = str(uuid4()), str(uuid4())
        saver.put(
            generate_config(thread_1), generate_checkpoint(), generate_metadata(), {}
        )
        saver.put(
            generate_config(thread_2), generate_checkpoint(), generate_metadata(), {}
        )

        saver.delete_thread(thread_1)

        assert saver.get_tuple(generate_config(thread_1)) is None
        assert saver.get_tuple(generate_config(thread_2)) is not None


@pytest.mark.parametrize("saver_name", ["base", "pool"])
def test_delete_thread_nonexistent_noop(saver_name: str) -> None:
    with _sync_saver(saver_name) as saver:
        saver.delete_thread(str(uuid4()))


@pytest.mark.parametrize("saver_name", ["base", "pool"])
def test_single_space_namespace_is_rejected(saver_name: str) -> None:
    with _sync_saver(saver_name) as saver:
        with pytest.raises(ValueError, match="checkpoint_ns"):
            saver.put(
                generate_config(str(uuid4()), checkpoint_ns=" "),
                generate_checkpoint(),
                generate_metadata(),
                {},
            )


@pytest.mark.parametrize("saver_name", ["base", "pool"])
def test_single_space_task_path_is_rejected(saver_name: str) -> None:
    with _sync_saver(saver_name) as saver:
        stored = saver.put(
            generate_config(str(uuid4())),
            generate_checkpoint(),
            generate_metadata(),
            {},
        )
        with pytest.raises(ValueError, match="task_path"):
            saver.put_writes(stored, [("ch", "val")], str(uuid4()), task_path=" ")
