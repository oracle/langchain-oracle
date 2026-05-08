# type: ignore

"""Oracle checkpoint search WHERE clause key coverage."""

import itertools
import json
from contextlib import contextmanager

import oracledb
import pytest

from langgraph_oracledb.checkpoint.oracle import OracleSaver
from tests.conftest import DEFAULT_CONNECTION_INFO, skip_if_no_oracle

_connection_pool = None


def get_connection_pool():
    """Get or create a connection pool for real execution checks."""
    global _connection_pool
    if _connection_pool is None:
        if "dsn" in DEFAULT_CONNECTION_INFO:
            dsn = DEFAULT_CONNECTION_INFO["dsn"]
        else:
            dsn = (
                f"{DEFAULT_CONNECTION_INFO['host']}:"
                f"{DEFAULT_CONNECTION_INFO['port']}/"
                f"{DEFAULT_CONNECTION_INFO['service_name']}"
            )

        _connection_pool = oracledb.create_pool(
            user=DEFAULT_CONNECTION_INFO["user"],
            password=DEFAULT_CONNECTION_INFO["password"],
            dsn=dsn,
            min=2,
            max=10,
            increment=1,
        )
    return _connection_pool


def _mock_saver():
    """Create a saver instance without a live DB connection."""
    saver = OracleSaver.__new__(OracleSaver)
    saver.conn = None
    saver.is_setup = False
    return saver


@contextmanager
def _test_saver_with_db():
    """Create a real saver for execution checks."""
    pool = get_connection_pool()
    conn = pool.acquire()

    try:
        saver = OracleSaver(conn)
        saver.setup()
        yield saver, conn
    finally:
        from tests.conftest_checkpointer import _cleanup_checkpoint_tables

        _cleanup_checkpoint_tables()
        pool.release(conn)


def generate_key_parameter_combinations():
    """Generate smaller high-signal combinations for _search_where."""
    config_cases = [
        None,
        {"configurable": {"thread_id": "test-thread"}},
        {"configurable": {"thread_id": "test-thread", "checkpoint_ns": ""}},
        {
            "configurable": {
                "thread_id": "test-thread",
                "checkpoint_id": "checkpoint_001",
            }
        },
    ]

    filter_cases = [
        None,
        {},
        {"string_key": "test_value"},
        {"int_key": 42},
        {"bool_true": True},
        {"bool_false": False},
        {"null_key": None},
        {"dict_key": {"nested": "value"}},
        {"list_key": [1, 2, "three"]},
        {"str": "value", "bool": True, "num": 10},
        {"bool1": True, "bool2": False},
    ]

    before_cases = [
        None,
        {
            "configurable": {
                "thread_id": "test-thread",
                "checkpoint_id": "before_001",
            }
        },
    ]

    return [
        {"config": config, "filter": filter_dict, "before": before}
        for config, filter_dict, before in itertools.product(
            config_cases, filter_cases, before_cases
        )
    ]


def _assert_search_where_shape(saver, config, filter_dict, before) -> None:
    where_clause, param_values = saver._search_where(config, filter_dict, before)

    if config or filter_dict or before:
        assert where_clause.startswith("WHERE ")
    else:
        assert where_clause == ""

    if config:
        configurable = config["configurable"]
        assert param_values["thread_id"] == configurable["thread_id"]
        if "checkpoint_ns" in configurable:
            assert param_values["checkpoint_ns"] == saver._encode_checkpoint_ns(
                configurable["checkpoint_ns"]
            )
        if "checkpoint_id" in configurable:
            assert param_values["checkpoint_id"] == configurable["checkpoint_id"]

    if filter_dict:
        for key, value in filter_dict.items():
            if value is None:
                assert f"JSON_VALUE(metadata, '$.{key}') IS NULL" in where_clause
            elif isinstance(value, bool):
                assert (
                    f"JSON_VALUE(metadata, '$.{key}') = '{str(value).lower()}'"
                ) in where_clause
            elif isinstance(value, dict | list):
                assert f"JSON_EQUAL(JSON_QUERY(metadata, '$.{key}')" in where_clause
                assert json.dumps(value) in param_values.values()
            elif isinstance(value, int | float):
                assert (
                    f"JSON_VALUE(metadata, '$.{key}' RETURNING NUMBER) ="
                    in where_clause
                )
                assert value in param_values.values()
            else:
                assert f"JSON_VALUE(metadata, '$.{key}') =" in where_clause
                assert value in param_values.values()

    if before:
        assert (
            param_values["before_checkpoint_id"]
            == before["configurable"]["checkpoint_id"]
        )


@pytest.mark.parametrize("params", generate_key_parameter_combinations())
def test_search_where_key_combinations(params) -> None:
    """Exercise high-signal key combinations without snapshots."""
    saver = _mock_saver()
    _assert_search_where_shape(
        saver,
        params["config"],
        params["filter"],
        params["before"],
    )


@skip_if_no_oracle()
def test_known_problematic_cases() -> None:
    """Validate previously risky cases against a real Oracle cursor."""
    problematic_cases = [
        {
            "name": "single_boolean_true",
            "config": {"configurable": {"thread_id": "bool-test"}},
            "filter": {"active": True},
            "before": None,
        },
        {
            "name": "single_boolean_false",
            "config": {"configurable": {"thread_id": "bool-test"}},
            "filter": {"active": False},
            "before": None,
        },
        {
            "name": "multiple_booleans",
            "config": {"configurable": {"thread_id": "bool-test"}},
            "filter": {"active": True, "enabled": False, "visible": True},
            "before": None,
        },
        {
            "name": "before_parameter",
            "config": {"configurable": {"thread_id": "before-test"}},
            "filter": None,
            "before": {
                "configurable": {
                    "thread_id": "before-test",
                    "checkpoint_id": "checkpoint_002",
                }
            },
        },
        {
            "name": "checkpoint_id_config",
            "config": {
                "configurable": {
                    "thread_id": "checkpoint-test",
                    "checkpoint_id": "specific_checkpoint",
                }
            },
            "filter": None,
            "before": None,
        },
        {
            "name": "complex_mixed_types",
            "config": {
                "configurable": {
                    "thread_id": "mixed-test",
                    "checkpoint_ns": "",
                }
            },
            "filter": {
                "status": "active",
                "enabled": True,
                "count": 42,
                "data": {"key": "value"},
                "optional": None,
            },
            "before": None,
        },
    ]

    with _test_saver_with_db() as (saver, conn):
        for case in problematic_cases:
            where_clause, param_values = saver._search_where(
                case["config"], case["filter"], case["before"]
            )

            assert where_clause.startswith("WHERE ")
            full_query = "SELECT checkpoint_id FROM checkpoints " + where_clause

            with conn.cursor() as cur:
                cur.execute(full_query, param_values)
                cur.fetchall()
