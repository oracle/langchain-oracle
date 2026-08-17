# type: ignore

"""Oracle checkpoint search WHERE clause parameter coverage."""

import itertools
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


def generate_parameter_combinations():
    """Generate a reduced set of important _search_where combinations."""
    config_cases = [
        None,
        {"configurable": {"thread_id": "test-thread"}},
        {"configurable": {"thread_id": "test-thread", "checkpoint_ns": ""}},
        {
            "configurable": {
                "thread_id": "test-thread",
                "checkpoint_ns": "inner",
                "checkpoint_id": "checkpoint_002",
            }
        },
        # {
        #     "configurable": {
        #         "thread_id": "test-thread",
        #         "checkpoint_id": "checkpoint_001",
        #     }
        # },
        # {
        #     "configurable": {
        #         "thread_id": "test-thread",
        #         "checkpoint_ns": "",
        #         "checkpoint_id": "checkpoint_001",
        #     }
        # },
    ]

    filter_cases = [
        None,
        {},
        {"string_key": "test_value"},
        {"int_key": 42},
        {"bool_true": True},
        {"null_key": None},
        {"dict_key": {"nested": "value"}},
        {"mixed": "active", "count": 1, "enabled": False},
        {"complex_dict": {"nested": {"deep": {"value": 123}}}},
        {"dict": {"key": "val"}, "list": [1, 2]},
        {"list_key": [1, 2, "three"]},
        {"complex_list": [{"item": 1}, {"item": 2}, [3, 4]]},
        # {"float_key": 3.14},
        # {"bool_false": False},
        # {"empty_string": ""},
        # {"zero_int": 0},
        # {"zero_float": 0.0},
        # {"str": "value", "num": 10},
        # {"bool": True, "null": None},
        # {"dict": {"key": "val"}, "list": [1, 2]},
        # {
        #     "str": "test",
        #     "num": 42,
        #     "bool": False,
        #     "null": None,
        #     "dict": {"nested": "value"},
        # },
    ]

    before_cases = [
        None,
        {"configurable": {"thread_id": "test-thread", "checkpoint_id": "before_001"}},
        # {
        #     "configurable": {
        #         "thread_id": "test-thread",
        #         "checkpoint_ns": "",
        #         "checkpoint_id": "before_002",
        #     }
        # },
        # {
        #     "configurable": {
        #         "thread_id": "different-thread",
        #         "checkpoint_id": "before_003",
        #     }
        # },
    ]

    return [
        {"config": config, "filter": filter_dict, "before": before}
        for config, filter_dict, before in itertools.product(
            config_cases, filter_cases, before_cases
        )
    ]


def _count_leaf_params(value) -> int:
    """Bind-parameter count for one filter value under containment flattening."""
    if value is None or isinstance(value, bool):
        return 0
    if isinstance(value, dict):
        return sum(_count_leaf_params(v) for v in value.values())
    if isinstance(value, list):
        return sum(_count_list_element_params(v) for v in value)
    return 1


def _count_list_element_params(value) -> int:
    """Bind count for one list element under containment (issue #296)."""
    if value is None or isinstance(value, bool) or value == "":
        return 0
    if isinstance(value, dict):
        return sum(_count_list_element_params(v) for v in value.values())
    if isinstance(value, list):
        return sum(_count_list_element_params(v) for v in value)
    return 1


def _expected_filter_param_count(filter_dict) -> int:
    if not filter_dict:
        return 0
    return sum(_count_leaf_params(value) for value in filter_dict.values())


def _assert_list_containment(where_clause, param_values, path, value) -> None:
    """Assert containment predicates for list filter values (issue #296)."""
    assert f"JSON_EXISTS(metadata, '$.{path}?(@.type() == \"array\")')" in where_clause
    if value:
        assert f"'$.{path}[*]?(" in where_clause
    for element in value:
        if isinstance(element, (dict, list)):
            continue  # nested elements are covered by their bound leaf params
        if element is None or isinstance(element, bool) or element == "":
            continue  # rendered as path literals, not binds
        assert element in param_values.values()


def _assert_containment_paths(where_clause, param_values, path, value) -> None:
    """Assert flattened containment predicates for dict filter values."""
    if not value:
        assert f"JSON_EXISTS(metadata, '$.{path}')" in where_clause
        return
    for sub_key, sub_value in value.items():
        sub_path = f"{path}.{sub_key}"
        if isinstance(sub_value, dict):
            _assert_containment_paths(where_clause, param_values, sub_path, sub_value)
        elif sub_value is None:
            assert f"JSON_VALUE(metadata, '$.{sub_path}') IS NULL" in where_clause
        elif isinstance(sub_value, bool):
            bool_str = "true" if sub_value else "false"
            assert (
                f"JSON_VALUE(metadata, '$.{sub_path}') = '{bool_str}'" in where_clause
            )
        elif isinstance(sub_value, int | float):
            assert (
                f"JSON_VALUE(metadata, '$.{sub_path}' RETURNING NUMBER) ="
                in where_clause
            )
            assert sub_value in param_values.values()
        elif isinstance(sub_value, list):
            _assert_list_containment(where_clause, param_values, sub_path, sub_value)
        else:
            assert f"JSON_VALUE(metadata, '$.{sub_path}') =" in where_clause
            assert sub_value in param_values.values()


def _assert_search_where_shape(saver, config, filter_dict, before) -> None:
    where_clause, param_values = saver._search_where(config, filter_dict, before)

    if config or filter_dict or before:
        assert where_clause.startswith("WHERE ")
    else:
        assert where_clause == ""

    assert isinstance(param_values, dict)

    if config:
        configurable = config["configurable"]
        assert "thread_id = :thread_id" in where_clause
        assert param_values["thread_id"] == configurable["thread_id"]

        if "checkpoint_ns" in configurable:
            assert "checkpoint_ns = :checkpoint_ns" in where_clause
            assert param_values["checkpoint_ns"] == saver._encode_checkpoint_ns(
                configurable["checkpoint_ns"]
            )

        if "checkpoint_id" in configurable:
            assert "checkpoint_id = :checkpoint_id" in where_clause
            assert param_values["checkpoint_id"] == configurable["checkpoint_id"]

    if filter_dict:
        filter_param_names = [
            key
            for key in param_values
            if key.startswith("filter_key_") or key.startswith("filter_json_")
        ]
        assert len(filter_param_names) == _expected_filter_param_count(filter_dict)

        for key, value in filter_dict.items():
            if value is None:
                assert f"JSON_VALUE(metadata, '$.{key}') IS NULL" in where_clause
            elif isinstance(value, bool):
                bool_str = "true" if value else "false"
                assert f"JSON_VALUE(metadata, '$.{key}') = '{bool_str}'" in where_clause
            elif isinstance(value, list):
                _assert_list_containment(where_clause, param_values, key, value)
            elif isinstance(value, dict):
                # dict filters match by containment: flattened per-leaf paths
                _assert_containment_paths(where_clause, param_values, key, value)
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
        assert "checkpoint_id < :before_checkpoint_id" in where_clause
        assert (
            param_values["before_checkpoint_id"]
            == before["configurable"]["checkpoint_id"]
        )


@pytest.mark.parametrize("params", generate_parameter_combinations())
def test_search_where_parameter_combinations(params) -> None:
    """Exercise representative _search_where combinations without snapshots."""
    saver = _mock_saver()
    _assert_search_where_shape(
        saver,
        params["config"],
        params["filter"],
        params["before"],
    )


@skip_if_no_oracle()
def test_search_where_edge_cases() -> None:
    """Validate risky cases against a real Oracle cursor."""
    edge_cases = [
        {
            "name": "boolean_true_only",
            "config": {"configurable": {"thread_id": "bool-test"}},
            "filter": {"active": True},
            "before": None,
        },
        {
            "name": "boolean_false_only",
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
            "name": "before_parameter_basic",
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
            "name": "complex_combination",
            "config": {
                "configurable": {
                    "thread_id": "complex-test",
                    "checkpoint_ns": "inner",
                    "checkpoint_id": "checkpoint_001",
                }
            },
            "filter": {
                "status": "active",
                "enabled": True,
                "count": 42,
                "config": {"nested": "value"},
                "optional": None,
            },
            "before": {
                "configurable": {
                    "thread_id": "complex-test",
                    "checkpoint_id": "checkpoint_002",
                }
            },
        },
        {
            "name": "unicode_and_special_chars",
            "config": {"configurable": {"thread_id": "unicode-тест-🚀"}},
            "filter": {"description": "测试-αβγ-emoji🎉", "count": 42},
            "before": None,
        },
        {
            "name": "long_strings",
            "config": {"configurable": {"thread_id": "x" * 100}},
            "filter": {"long_key": "y" * 1000, "normal": "short"},
            "before": None,
        },
    ]

    with _test_saver_with_db() as (saver, conn):
        for case in edge_cases:
            where_clause, param_values = saver._search_where(
                case["config"], case["filter"], case["before"]
            )

            assert where_clause.startswith("WHERE ")
            assert isinstance(param_values, dict)

            full_query = "SELECT checkpoint_id FROM checkpoints " + where_clause
            with conn.cursor() as cur:
                cur.execute(full_query, param_values)
                cur.fetchall()
