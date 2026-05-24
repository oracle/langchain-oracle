# type: ignore

import pytest
from langgraph.checkpoint.base import (
    empty_checkpoint,
)

from tests.conftest_checkpointer import (
    _create_config,
    _sync_saver,
)


@pytest.mark.parametrize("saver_name", ["base", "pool"])
def test_metadata_filtering_boolean_values(saver_name: str, test_data) -> None:
    """Test filtering by boolean metadata values."""
    with _sync_saver(saver_name) as saver:
        config = {
            "configurable": {
                "thread_id": "thread-bool-test",
                "checkpoint_ns": "",
            }
        }

        # Create checkpoints with boolean metadata
        checkpoint_1 = empty_checkpoint()
        checkpoint_2 = empty_checkpoint()

        metadata_true = {"active": True, "enabled": True}
        metadata_false = {"active": False, "enabled": False}

        saver.put(config, checkpoint_1, metadata_true, {})
        config_2 = dict(config)
        config_2["configurable"]["checkpoint_id"] = checkpoint_2["id"]
        saver.put(config_2, checkpoint_2, metadata_false, {})

        # Search for True values
        results_true = list(saver.list(None, filter={"active": True}))
        assert len(results_true) == 1
        assert results_true[0].metadata["active"] is True

        # Search for False values
        results_false = list(saver.list(None, filter={"active": False}))
        assert len(results_false) == 1
        assert results_false[0].metadata["active"] is False

        # Search by multiple boolean fields
        results_both_true = list(
            saver.list(None, filter={"active": True, "enabled": True})
        )
        assert len(results_both_true) == 1
        assert results_both_true[0].metadata["active"] is True
        assert results_both_true[0].metadata["enabled"] is True


@pytest.mark.parametrize("saver_name", ["base", "pool"])
def test_metadata_filtering_numeric_values(saver_name: str, test_data) -> None:
    """Test filtering by numeric metadata values (int and float)."""
    with _sync_saver(saver_name) as saver:
        config = {
            "configurable": {
                "thread_id": "thread-numeric-test",
                "checkpoint_ns": "",
            }
        }

        # Create checkpoints with numeric metadata
        checkpoint_1 = empty_checkpoint()
        checkpoint_2 = empty_checkpoint()
        checkpoint_3 = empty_checkpoint()

        metadata_int = {"count": 42, "priority": 1}
        metadata_float = {"rate": 3.14, "threshold": 0.85}
        metadata_mixed = {"count": 100, "rate": 2.5}

        saver.put(config, checkpoint_1, metadata_int, {})

        config_2 = dict(config)
        config_2["configurable"]["checkpoint_id"] = checkpoint_2["id"]
        saver.put(config_2, checkpoint_2, metadata_float, {})

        config_3 = dict(config)
        config_3["configurable"]["checkpoint_id"] = checkpoint_3["id"]
        saver.put(config_3, checkpoint_3, metadata_mixed, {})

        # Search for integer values
        results_int = list(saver.list(None, filter={"count": 42}))
        assert len(results_int) == 1
        assert results_int[0].metadata["count"] == 42

        # Search for float values
        results_float = list(saver.list(None, filter={"rate": 3.14}))
        assert len(results_float) == 1
        assert float(results_float[0].metadata["rate"]) == 3.14

        # Search for mixed numeric values
        results_mixed = list(saver.list(None, filter={"count": 100, "rate": 2.5}))
        assert len(results_mixed) == 1
        assert results_mixed[0].metadata["count"] == 100
        assert results_mixed[0].metadata["rate"] == 2.5

        # Search for non-existent numeric value
        results_none = list(saver.list(None, filter={"count": 999}))
        assert len(results_none) == 0


@pytest.mark.parametrize("saver_name", ["base", "pool"])
def test_metadata_filtering_null_values(saver_name: str, test_data) -> None:
    """Test filtering by null/None metadata values."""
    with _sync_saver(saver_name) as saver:
        config = {
            "configurable": {
                "thread_id": "thread-null-test",
                "checkpoint_ns": "",
            }
        }

        # Create checkpoints with null metadata
        checkpoint_1 = empty_checkpoint()
        checkpoint_2 = empty_checkpoint()

        metadata_with_null = {"status": "active", "optional": None}
        metadata_without_null = {"status": "inactive", "optional": "value"}

        saver.put(config, checkpoint_1, metadata_with_null, {})

        config_2 = dict(config)
        config_2["configurable"]["checkpoint_id"] = checkpoint_2["id"]
        saver.put(config_2, checkpoint_2, metadata_without_null, {})

        # Search for None values
        results_null = list(saver.list(None, filter={"optional": None}))
        assert len(results_null) == 1
        assert results_null[0].metadata["optional"] is None

        # Search for non-null values
        results_not_null = list(saver.list(None, filter={"optional": "value"}))
        assert len(results_not_null) == 1
        assert results_not_null[0].metadata["optional"] == "value"


@pytest.mark.parametrize("saver_name", ["base", "pool"])
def test_metadata_filtering_complex_objects(saver_name: str, test_data) -> None:
    """Test filtering by complex objects (dict/list) in metadata."""
    with _sync_saver(saver_name) as saver:
        config = {
            "configurable": {
                "thread_id": "thread-complex-test",
                "checkpoint_ns": "",
            }
        }

        # Create checkpoints with complex metadata
        checkpoint_1 = empty_checkpoint()
        checkpoint_2 = empty_checkpoint()
        checkpoint_3 = empty_checkpoint()

        complex_dict = {"nested": {"key": "value", "count": 5}}
        complex_list = [1, 2, {"item": "test"}]
        simple_dict = {"simple": "value"}

        metadata_dict = {"config": complex_dict, "type": "dict"}
        metadata_list = {"items": complex_list, "type": "list"}
        metadata_simple = {"config": simple_dict, "type": "simple"}

        saver.put(config, checkpoint_1, metadata_dict, {})

        config_2 = dict(config)
        config_2["configurable"]["checkpoint_id"] = checkpoint_2["id"]
        saver.put(config_2, checkpoint_2, metadata_list, {})

        config_3 = dict(config)
        config_3["configurable"]["checkpoint_id"] = checkpoint_3["id"]
        saver.put(config_3, checkpoint_3, metadata_simple, {})

        # Search for exact dict match
        results_dict = list(saver.list(None, filter={"config": complex_dict}))
        assert len(results_dict) == 1
        assert results_dict[0].metadata["config"] == complex_dict

        # Search for exact list match
        results_list = list(saver.list(None, filter={"items": complex_list}))
        assert len(results_list) == 1
        assert results_list[0].metadata["items"] == complex_list

        # Search for simple dict match
        results_simple = list(saver.list(None, filter={"config": simple_dict}))
        assert len(results_simple) == 1
        assert results_simple[0].metadata["config"] == simple_dict

        # Search for non-matching complex object
        non_matching = {"different": "structure"}
        results_none = list(saver.list(None, filter={"config": non_matching}))
        assert len(results_none) == 0


@pytest.mark.parametrize("saver_name", ["base", "pool"])
def test_before_parameter_filtering(saver_name: str, test_data) -> None:
    """Test filtering with before parameter for checkpoint_id ordering."""
    with _sync_saver(saver_name) as saver:
        config = {
            "configurable": {
                "thread_id": "thread-before-test",
                "checkpoint_ns": "",
            }
        }

        # Create multiple checkpoints with known IDs
        checkpoint_1 = empty_checkpoint()
        checkpoint_2 = empty_checkpoint()
        checkpoint_3 = empty_checkpoint()

        # Force specific checkpoint IDs for predictable ordering
        checkpoint_1["id"] = "checkpoint_001"
        checkpoint_2["id"] = "checkpoint_002"
        checkpoint_3["id"] = "checkpoint_003"

        config_1 = _create_config("thread-before-test", "", checkpoint_1["id"])
        config_2 = _create_config("thread-before-test", "", checkpoint_2["id"])
        config_3 = _create_config("thread-before-test", "", checkpoint_3["id"])

        metadata = {"test": "before"}

        saver.put(config_1, checkpoint_1, metadata, {})
        saver.put(config_2, checkpoint_2, metadata, {})
        saver.put(config_3, checkpoint_3, metadata, {})

        # Test before parameter - should return checkpoints with ID < "checkpoint_002"
        before_config = {
            "configurable": {
                "thread_id": "thread-before-test",
                "checkpoint_ns": "",
                "checkpoint_id": "checkpoint_002",
            }
        }

        results_before = list(saver.list(config, before=before_config))
        # Should only return checkpoint_001 (checkpoint_002 and checkpoint_003 excluded)
        assert len(results_before) == 1
        assert (
            results_before[0].config["configurable"]["checkpoint_id"]
            == "checkpoint_001"
        )

        # Test with no before parameter - should return all
        results_all = list(saver.list(config))
        assert len(results_all) == 3


@pytest.mark.parametrize("saver_name", ["base", "pool"])
def test_checkpoint_id_config_filtering(saver_name: str, test_data) -> None:
    """Test filtering by checkpoint_id in config."""
    with _sync_saver(saver_name) as saver:
        # Create multiple checkpoints
        checkpoint_1 = empty_checkpoint()
        checkpoint_2 = empty_checkpoint()

        config_1 = _create_config("thread-checkpoint-id-test", "", checkpoint_1["id"])
        config_2 = _create_config("thread-checkpoint-id-test", "", checkpoint_2["id"])

        metadata = {"test": "checkpoint_id"}

        saver.put(config_1, checkpoint_1, metadata, {})
        saver.put(config_2, checkpoint_2, metadata, {})

        # Search for specific checkpoint_id
        search_config = {
            "configurable": {
                "thread_id": "thread-checkpoint-id-test",
                "checkpoint_ns": "",
                "checkpoint_id": checkpoint_1["id"],
            }
        }

        results = list(saver.list(search_config))
        assert len(results) == 1
        assert results[0].config["configurable"]["checkpoint_id"] == checkpoint_1["id"]

        # Search for non-existent checkpoint_id
        search_config_none = {
            "configurable": {
                "thread_id": "thread-checkpoint-id-test",
                "checkpoint_ns": "",
                "checkpoint_id": "non-existent-id",
            }
        }

        results_none = list(saver.list(search_config_none))
        assert len(results_none) == 0


@pytest.mark.parametrize("saver_name", ["base", "pool"])
def test_combined_filtering_scenarios(saver_name: str, test_data) -> None:
    """Test combined filtering with multiple parameters."""
    with _sync_saver(saver_name) as saver:
        config = {
            "configurable": {
                "thread_id": "thread-combined-test",
                "checkpoint_ns": "",
            }
        }

        # Create checkpoints with varied metadata
        checkpoint_1 = empty_checkpoint()
        checkpoint_2 = empty_checkpoint()
        checkpoint_3 = empty_checkpoint()

        metadata_1 = {
            "status": "active",
            "priority": 1,
            "enabled": True,
            "config": {"mode": "production"},
            "optional": None,
        }
        metadata_2 = {
            "status": "active",
            "priority": 2,
            "enabled": False,
            "config": {"mode": "development"},
            "optional": "value",
        }
        metadata_3 = {
            "status": "inactive",
            "priority": 1,
            "enabled": True,
            "config": {"mode": "production"},
            "optional": None,
        }

        saver.put(config, checkpoint_1, metadata_1, {})

        config_2 = dict(config)
        config_2["configurable"]["checkpoint_id"] = checkpoint_2["id"]
        saver.put(config_2, checkpoint_2, metadata_2, {})

        config_3 = dict(config)
        config_3["configurable"]["checkpoint_id"] = checkpoint_3["id"]
        saver.put(config_3, checkpoint_3, metadata_3, {})

        # Test 1: Multiple primitive filters
        results_multi = list(
            saver.list(
                None, filter={"status": "active", "priority": 1, "enabled": True}
            )
        )
        assert len(results_multi) == 1
        assert results_multi[0].metadata["status"] == "active"
        assert results_multi[0].metadata["priority"] == 1
        assert results_multi[0].metadata["enabled"] is True

        # Test 2: Mixed types filter
        results_mixed = list(
            saver.list(
                None,
                filter={
                    "status": "active",
                    "enabled": False,
                    "config": {"mode": "development"},
                },
            )
        )
        assert len(results_mixed) == 1
        assert results_mixed[0].metadata["status"] == "active"
        assert results_mixed[0].metadata["enabled"] is False
        assert results_mixed[0].metadata["config"] == {"mode": "development"}

        # Test 3: Include null filter
        results_null = list(saver.list(None, filter={"priority": 1, "optional": None}))
        assert len(results_null) == 2  # Both metadata_1 and metadata_3 match

        # Test 4: No matches
        results_none = list(
            saver.list(
                None,
                filter={
                    "status": "active",
                    "priority": 999,  # Non-existent priority
                },
            )
        )
        assert len(results_none) == 0

        # Test 5: Config + metadata filter combined
        config_filter = {
            "configurable": {"thread_id": "thread-combined-test", "checkpoint_ns": ""}
        }
        results_config_meta = list(
            saver.list(config_filter, filter={"status": "active", "enabled": True})
        )
        assert len(results_config_meta) == 1
        assert results_config_meta[0].metadata["status"] == "active"
        assert results_config_meta[0].metadata["enabled"] is True
