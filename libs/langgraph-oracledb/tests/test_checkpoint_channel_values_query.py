"""Unit tests for the query that loads the blobs a checkpoint references."""

from __future__ import annotations

import json
from typing import Any

import pytest

from langgraph_oracledb.checkpoint.oracle import base as oracle_base
from langgraph_oracledb.checkpoint.oracle.base import BaseOracleSaver


def _saver() -> BaseOracleSaver:
    # Query building needs no connection.
    return BaseOracleSaver.__new__(BaseOracleSaver)


def _value(
    channel_versions: dict[str, Any], thread_id: str = "t1", checkpoint_ns: str = " "
) -> dict[str, Any]:
    return {
        "thread_id": thread_id,
        "checkpoint_ns": checkpoint_ns,
        "checkpoint": {"channel_versions": channel_versions},
    }


def _pairs(queries: list[tuple[str, dict[str, str]]]) -> list[dict[str, str]]:
    return [
        entry
        for _, params in queries
        for entry in json.loads(params["channel_versions_json"])
    ]


def test_requests_only_the_versions_the_checkpoint_references() -> None:
    queries = _saver()._channel_values_queries([_value({"messages": "7.0.1", "n": 3})])

    assert len(queries) == 1
    sql, _ = queries[0]
    assert "JSON_TABLE" in sql
    assert "bl.version = cv.version" in sql
    # Versions are sent as text: checkpoint_blobs.version is a VARCHAR2 and
    # numeric versions must keep matching their own blobs.
    assert _pairs(queries) == [
        {"t": "t1", "n": " ", "c": "messages", "v": "7.0.1"},
        {"t": "t1", "n": " ", "c": "n", "v": "3"},
    ]


def test_pairs_shared_by_several_checkpoints_are_requested_once() -> None:
    queries = _saver()._channel_values_queries(
        [
            _value({"messages": "2", "summary": "1"}),
            _value({"messages": "3", "summary": "1"}),
            _value({"messages": "3"}, thread_id="t2"),
        ]
    )

    assert [(p["t"], p["c"], p["v"]) for p in _pairs(queries)] == [
        ("t1", "messages", "2"),
        ("t1", "summary", "1"),
        ("t1", "messages", "3"),
        ("t2", "messages", "3"),
    ]


@pytest.mark.parametrize(
    "values",
    [
        [],
        [_value({})],
        [{"thread_id": "t1", "checkpoint_ns": " ", "checkpoint": {}}],
        [{"thread_id": "t1", "checkpoint_ns": " ", "checkpoint": None}],
    ],
)
def test_no_query_when_no_versions_are_referenced(values: list[dict]) -> None:
    assert _saver()._channel_values_queries(values) == []


def test_large_requests_are_split_below_the_bind_limit(monkeypatch) -> None:
    monkeypatch.setattr(oracle_base, "_CHANNEL_VERSIONS_BIND_MAX_BYTES", 200)
    versions = {f"channel_{i}": str(i) for i in range(25)}

    queries = _saver()._channel_values_queries([_value(versions)])

    assert len(queries) > 1
    for _, params in queries:
        assert len(params["channel_versions_json"].encode("utf-8")) <= 200
    assert {p["c"]: p["v"] for p in _pairs(queries)} == versions


def test_non_ascii_identifiers_are_sized_in_bytes(monkeypatch) -> None:
    monkeypatch.setattr(oracle_base, "_CHANNEL_VERSIONS_BIND_MAX_BYTES", 120)
    versions = {f"canal_ñ_{i}": "1" for i in range(6)}

    queries = _saver()._channel_values_queries([_value(versions, thread_id="hilo_é")])

    for _, params in queries:
        assert len(params["channel_versions_json"].encode("utf-8")) <= 120
    assert {p["c"] for p in _pairs(queries)} == set(versions)
    assert {p["t"] for p in _pairs(queries)} == {"hilo_é"}
