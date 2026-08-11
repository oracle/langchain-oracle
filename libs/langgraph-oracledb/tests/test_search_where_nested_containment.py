# type: ignore

"""Nested-path containment matching for dict metadata filters (issue #290).

Dict filter values match by recursive containment — the filter
{"nested": {"b": {"c": True}}} matches metadata
{"nested": {"a": 1, "b": {"c": True}}} — aligning the Python saver with the
LangGraph.js Oracle saver and PostgresSaver (`metadata @> filter`). Lists
keep JSON_EQUAL exact-match (positional) semantics.
"""

import pytest
from langgraph.checkpoint.base import (
    empty_checkpoint,
)

from langgraph_oracledb.checkpoint.oracle.sync import OracleSaver
from tests.conftest_checkpointer import (
    _async_saver,
    _sync_saver,
)

# ---------------------------------------------------------------------------
# SQL generation (no database required)
# ---------------------------------------------------------------------------


def _where(filter_):
    saver = OracleSaver(conn=None)
    return saver._search_where(None, filter_)


def test_nested_dict_flattens_to_json_paths() -> None:
    where, params = _where({"nested": {"b": {"c": True}}})
    assert "JSON_VALUE(metadata, '$.nested.b.c') = 'true'" in where
    assert "JSON_EQUAL" not in where


def test_nested_leaf_types_use_typed_predicates() -> None:
    where, params = _where(
        {"level": {"depth": 3}, "meta": {"name": "run-1"}, "gone": {"x": None}}
    )
    assert "JSON_VALUE(metadata, '$.level.depth' RETURNING NUMBER)" in where
    assert "JSON_VALUE(metadata, '$.meta.name') = :" in where
    assert "JSON_VALUE(metadata, '$.gone.x') IS NULL" in where
    assert 3 in params.values()
    assert "run-1" in params.values()


def test_lists_keep_exact_match_semantics() -> None:
    where, params = _where({"tags": [1, 2], "nested": {"items": ["a"]}})
    assert "JSON_EQUAL(JSON_QUERY(metadata, '$.tags')" in where
    assert "JSON_EQUAL(JSON_QUERY(metadata, '$.nested.items')" in where


def test_empty_dict_requires_path_existence() -> None:
    where, _ = _where({"empty": {}})
    assert "JSON_EXISTS(metadata, '$.empty')" in where


def test_nested_keys_are_validated_against_path_injection() -> None:
    with pytest.raises(ValueError, match="Illegal metadata key"):
        _where({"nested": {"bad'key": 1}})


def test_scalar_filters_unchanged() -> None:
    where, params = _where({"source": "loop", "step": 2, "active": True})
    assert "JSON_VALUE(metadata, '$.source') = :" in where
    assert "JSON_VALUE(metadata, '$.step' RETURNING NUMBER) = :" in where
    assert "JSON_VALUE(metadata, '$.active') = 'true'" in where


# ---------------------------------------------------------------------------
# End-to-end against Oracle
# ---------------------------------------------------------------------------


def _put_checkpoints(saver, thread_id, metadatas):
    """Store one checkpoint per metadata dict on the given thread."""
    config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
    for metadata in metadatas:
        checkpoint = empty_checkpoint()
        cfg = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "",
                "checkpoint_id": checkpoint["id"],
            }
        }
        saver.put(cfg, checkpoint, metadata, {})
    return config


@pytest.mark.parametrize("saver_name", ["base", "pool"])
def test_partial_nested_filter_matches_containing_metadata(
    saver_name: str, test_data
) -> None:
    """The exact scenario from issue #290."""
    with _sync_saver(saver_name) as saver:
        _put_checkpoints(
            saver,
            "thread-containment",
            [
                {"nested": {"a": 1, "b": {"c": True}}, "kind": "match"},
                {"nested": {"a": 1, "b": {"c": False}}, "kind": "no-match"},
                {"nested": {"a": 1}, "kind": "missing-path"},
            ],
        )

        results = list(saver.list(None, filter={"nested": {"b": {"c": True}}}))
        assert len(results) == 1
        assert results[0].metadata["kind"] == "match"


@pytest.mark.parametrize("saver_name", ["base", "pool"])
def test_exact_dict_filter_still_matches_equal_metadata(
    saver_name: str, test_data
) -> None:
    """Exact filters are contained in equal metadata: old behavior preserved."""
    with _sync_saver(saver_name) as saver:
        complex_dict = {"nested": {"key": "value", "count": 5}}
        _put_checkpoints(
            saver,
            "thread-exact-still-works",
            [{"config": complex_dict}, {"config": {"other": True}}],
        )

        results = list(saver.list(None, filter={"config": complex_dict}))
        assert len(results) == 1
        assert results[0].metadata["config"] == complex_dict


@pytest.mark.parametrize("saver_name", ["base", "pool"])
def test_nested_filter_combined_with_scalar(saver_name: str, test_data) -> None:
    with _sync_saver(saver_name) as saver:
        _put_checkpoints(
            saver,
            "thread-combined",
            [
                {"source": "loop", "run": {"phase": "b", "attempt": 2}},
                {"source": "loop", "run": {"phase": "a", "attempt": 2}},
                {"source": "input", "run": {"phase": "b", "attempt": 2}},
            ],
        )

        results = list(
            saver.list(None, filter={"source": "loop", "run": {"phase": "b"}})
        )
        assert len(results) == 1
        assert results[0].metadata["run"]["phase"] == "b"
        assert results[0].metadata["source"] == "loop"


@pytest.mark.parametrize("saver_name", ["base", "pool"])
def test_list_values_still_require_exact_match(saver_name: str, test_data) -> None:
    with _sync_saver(saver_name) as saver:
        _put_checkpoints(
            saver,
            "thread-list-exact",
            [{"tags": [1, 2, 3]}, {"tags": [1, 2]}],
        )

        results = list(saver.list(None, filter={"tags": [1, 2]}))
        assert len(results) == 1
        assert results[0].metadata["tags"] == [1, 2]


@pytest.mark.parametrize("saver_name", ["base", "pool"])
async def test_async_partial_nested_filter(saver_name: str, test_data) -> None:
    """AsyncOracleSaver shares _search_where: containment works there too."""
    async with _async_saver(saver_name) as saver:
        thread_id = "thread-containment-async"
        for metadata in [
            {"nested": {"a": 1, "b": {"c": True}}, "kind": "match"},
            {"nested": {"b": {"c": False}}, "kind": "no-match"},
        ]:
            checkpoint = empty_checkpoint()
            cfg = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": "",
                    "checkpoint_id": checkpoint["id"],
                }
            }
            await saver.aput(cfg, checkpoint, metadata, {})

        results = [
            c async for c in saver.alist(None, filter={"nested": {"b": {"c": True}}})
        ]
        assert len(results) == 1
        assert results[0].metadata["kind"] == "match"
