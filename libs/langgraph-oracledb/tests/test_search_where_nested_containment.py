# type: ignore

"""Containment matching for checkpoint metadata filters (issues #290, #296).

Dict filter values match by recursive containment — the filter
{"nested": {"b": {"c": True}}} matches metadata
{"nested": {"a": 1, "b": {"c": True}}} — and list values match any stored
array containing every filter element regardless of order or duplicates,
aligning the Python saver with the LangGraph.js Oracle saver and
PostgresSaver (`metadata @> filter`).
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


def test_lists_use_containment_semantics() -> None:
    where, params = _where({"tags": [1, 2], "nested": {"items": ["a"]}})
    assert "JSON_EQUAL" not in where
    assert "JSON_EXISTS(metadata, '$.tags?(@.type() == \"array\")')" in where
    assert '\'$.tags[*]?(@.type() == "number" && @ == $FILTER_KEY_' in where
    assert "JSON_EXISTS(metadata, '$.nested.items?(@.type() == \"array\")')" in where
    assert '\'$.nested.items[*]?(@.type() == "string" && @ == $FILTER_KEY_' in where
    assert 1 in params.values()
    assert 2 in params.values()
    assert "a" in params.values()


def test_list_literal_elements_render_without_binds() -> None:
    where, params = _where({"flags": [True, None, ""]})
    assert '@.type() == "boolean" && @ == true' in where
    assert '@.type() == "null" && @ == null' in where
    assert '@.type() == "string" && @ == ""' in where
    assert params == {}


def test_list_dict_elements_compile_to_recursive_containment() -> None:
    where, params = _where({"events": [{"kind": "start", "attempt": 1}]})
    assert '@."kind"' in where
    assert '@."attempt"' in where
    assert "start" in params.values()
    assert 1 in params.values()


def test_list_dict_element_keys_are_validated_against_path_injection() -> None:
    with pytest.raises(ValueError, match="Illegal metadata key"):
        _where({"events": [{"bad'key": 1}]})


def test_list_non_finite_numbers_are_rejected() -> None:
    with pytest.raises(ValueError, match="must be finite"):
        _where({"values": [float("inf")]})


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
def test_list_values_match_by_containment(saver_name: str, test_data) -> None:
    """Issue #296: align list filters with LangGraph.js and PostgresSaver."""
    with _sync_saver(saver_name) as saver:
        _put_checkpoints(
            saver,
            "thread-list-containment",
            [{"tags": [1, 2, 3]}, {"tags": [1, 2]}, {"tags": "not-an-array"}],
        )

        # Subset and order-insensitive matches include both array rows.
        assert len(list(saver.list(None, filter={"tags": [1, 2]}))) == 2
        assert len(list(saver.list(None, filter={"tags": [2, 1]}))) == 2
        assert len(list(saver.list(None, filter={"tags": [1]}))) == 2
        # Elements absent from the stored arrays match nothing.
        assert len(list(saver.list(None, filter={"tags": [4]}))) == 0
        assert len(list(saver.list(None, filter={"tags": [1, 4]}))) == 0
        # An empty list matches any stored array, but never a scalar.
        assert len(list(saver.list(None, filter={"tags": []}))) == 2


@pytest.mark.parametrize("saver_name", ["base", "pool"])
def test_list_containment_matches_typed_elements(saver_name: str, test_data) -> None:
    with _sync_saver(saver_name) as saver:
        _put_checkpoints(
            saver,
            "thread-list-typed",
            [
                {"mixed": ["a", 2, True, None, {"kind": "start", "attempt": 1}]},
                {"mixed": ["b", 3, False]},
            ],
        )

        assert len(list(saver.list(None, filter={"mixed": ["a"]}))) == 1
        assert len(list(saver.list(None, filter={"mixed": [2]}))) == 1
        assert len(list(saver.list(None, filter={"mixed": [True]}))) == 1
        assert len(list(saver.list(None, filter={"mixed": [None]}))) == 1
        # Dict elements match by recursive containment, not exact equality.
        assert len(list(saver.list(None, filter={"mixed": [{"kind": "start"}]}))) == 1
        assert len(list(saver.list(None, filter={"mixed": [{"kind": "stop"}]}))) == 0
        # The boolean True must not match the number 2 or the string "a".
        assert len(list(saver.list(None, filter={"mixed": [False]}))) == 1


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
