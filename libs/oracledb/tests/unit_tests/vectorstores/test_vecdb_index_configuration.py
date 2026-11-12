# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""LangChain index options map to VecDB creation and reconfiguration requests."""

import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from langchain_oracledb.vectorstores._vecdb import (
    CURRENT_COLUMNS,
    AsyncNativeTransport,
    NativeTransport,
    RestTransport,
    VecDBBackend,
)
from langchain_oracledb.vectorstores._vecdb_adapter import SQLNetAdapter
from langchain_oracledb.vectorstores.oraclevs import acreate_index, create_index
from langchain_oracledb.vectorstores.utils import DistanceStrategy


@pytest.fixture(params=["legacy", "current", "rest"])
def policy_transport(request, asynchronous):
    """Record policy payloads without assuming a server's index lifecycle."""
    if request.param == "rest":
        transport = RestTransport.__new__(RestTransport)
    else:
        version = (23, 26, 2) if request.param == "legacy" else (23, 26, 3)
        transport = (AsyncNativeTransport if asynchronous else NativeTransport)(
            object(), version
        )
    transport.call = MagicMock(return_value={"message": "done"})
    transport.acall = AsyncMock(return_value={"message": "done"})
    transport.describe_index = MagicMock()
    transport.adescribe_index = AsyncMock()
    transport.columns = MagicMock(return_value=CURRENT_COLUMNS)
    transport.acolumns = AsyncMock(return_value=CURRENT_COLUMNS)
    transport.discover = MagicMock()
    transport.adiscover = AsyncMock()
    return transport


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "options, automatic",
    [
        pytest.param(None, False, id="missing-index-request"),
        pytest.param({}, None, id="empty-index-request"),
        pytest.param(
            {"idx_type": "IVF", "neighbor_part": 8}, None, id="omitted-policy"
        ),
        pytest.param({"auto_index": False}, False, id="manual-policy"),
        pytest.param({"auto_index": True}, True, id="automatic-policy"),
    ],
)
@pytest.mark.asyncio
async def test_table_creation_preserves_automatic_policy(
    policy_transport, asynchronous, options, automatic
):
    transport = policy_transport
    original = deepcopy(options)
    backend = VecDBBackend(
        transport, "DOCS", DistanceStrategy.COSINE, vector_index_params=options
    )
    if isinstance(transport, RestTransport):
        missing = RuntimeError("missing table")
        missing.status = 404
        # The second prepare reopens an existing table without changing policy.
        transport.call.side_effect = [missing, {}, {"name": "DOCS"}]
    for columns in (set(), CURRENT_COLUMNS):
        if asynchronous:
            await backend.run_async("prepare", columns=columns)
        else:
            backend.prepare(columns=columns)
    call = (
        transport.acall
        if isinstance(transport, AsyncNativeTransport)
        else transport.call
    )
    creates = [c for c in call.call_args_list if c.args == ("create_vector_table",)]
    assert len(creates) == 1
    request = creates[0].kwargs
    mapped = request.get("index_params")
    if options == {}:
        assert "index_params" not in request
    elif automatic is None:
        assert "indexing" not in mapped
        assert "auto_index" not in mapped.get("vector_index_params", mapped)
    elif getattr(transport, "version", (23, 26, 3)) < (23, 26, 3):
        assert mapped["indexing"] == ("auto" if automatic else "manual")
        assert "auto_index" not in mapped
    else:
        assert mapped["vector_index_params"]["auto_index"] is automatic
        assert "indexing" not in mapped["vector_index_params"]
    if isinstance(transport, NativeTransport):
        _, binds = SQLNetAdapter(transport.version).statement(
            "create_vector_table", request
        )
        if mapped is None:
            assert "index_params" not in binds
        else:
            assert json.loads(binds["index_params"]) == mapped
    assert not any(
        c.args[0] in {"create_index", "rebuild_index"} for c in call.call_args_list
    )
    assert options == original


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("exists", [False, True])
@pytest.mark.asyncio
async def test_manual_table_allows_explicit_create_or_rebuild(
    policy_transport, asynchronous, exists
):
    transport = policy_transport
    backend = VecDBBackend(
        transport,
        "DOCS",
        DistanceStrategy.COSINE,
        vector_index_params={"auto_index": False},
    )
    status = {"details": {"indexes": [{"index_name": "GENERATED"}] if exists else []}}
    transport.describe_index.return_value = status
    transport.adescribe_index.return_value = status

    def rest_call(operation, **kwargs):
        if operation == "describe_index":
            if exists:
                return {"index_status": "READY"}
            missing = RuntimeError("missing index")
            missing.status = 404
            raise missing
        return {"message": "done"}

    if isinstance(transport, RestTransport):
        transport.call.side_effect = rest_call
    store = SimpleNamespace(_vecdb=backend, client=object())
    options = {"idx_type": "IVF"}
    if asynchronous:
        result = await acreate_index(store.client, store, options)
    else:
        result = create_index(store.client, store, options)
    assert result == {"message": "done"}
    call = (
        transport.acall
        if isinstance(transport, AsyncNativeTransport)
        else transport.call
    )
    assert call.call_args.args == ("rebuild_index" if exists else "create_index",)
    mapped = call.call_args.kwargs["index_params"]
    assert "indexing" not in mapped
    assert "auto_index" not in mapped.get("vector_index_params", mapped)
    if getattr(transport, "version", (23, 26, 3)) < (23, 26, 3):
        assert backend._creation_index_params["indexing"] == "manual"
    else:
        assert (
            backend._creation_index_params["vector_index_params"]["auto_index"] is False
        )
    if exists:
        assert mapped["index_type"] == "vector"


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "options, message",
    [
        ({"auto_index": False}, "auto_index is a table-creation policy"),
        ({"auto_index": True}, "auto_index is a table-creation policy"),
        ({"auto_index": "manual"}, "auto_index is a table-creation policy"),
        ({"auto_index": "false"}, "auto_index is a table-creation policy"),
        ({"auto_index": 0}, "auto_index is a table-creation policy"),
        ({"auto_index": 1}, "auto_index is a table-creation policy"),
        ({"auto_index": None}, "auto_index is a table-creation policy"),
        ({"idx_name": "MY_INDEX"}, "idx_name is unsupported"),
        ({"idx_name": None}, "idx_name is unsupported"),
    ],
)
@pytest.mark.asyncio
async def test_explicit_helpers_reject_policy_and_names_before_index_actions(
    policy_transport, asynchronous, options, message
):
    transport = policy_transport
    backend = VecDBBackend(transport, "DOCS", DistanceStrategy.COSINE)
    store = SimpleNamespace(_vecdb=backend, client=object())
    original = deepcopy(options)
    with pytest.raises(ValueError, match=message):
        if asynchronous:
            await acreate_index(store.client, store, options)
        else:
            create_index(store.client, store, options)
    transport.call.assert_not_called()
    transport.acall.assert_not_called()
    transport.describe_index.assert_not_called()
    transport.adescribe_index.assert_not_called()
    assert options == original


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "options, message",
    [
        ({"auto_index": "manual"}, "auto_index must be a boolean"),
        ({"auto_index": "false"}, "auto_index must be a boolean"),
        ({"auto_index": 0}, "auto_index must be a boolean"),
        ({"auto_index": 1}, "auto_index must be a boolean"),
        ({"auto_index": None}, "auto_index must be a boolean"),
        ({"idx_name": "MY_INDEX"}, "idx_name is unsupported"),
    ],
)
def test_invalid_table_policy_and_names_fail_before_creation(
    policy_transport, options, message
):
    original = deepcopy(options)
    with pytest.raises(ValueError, match=message):
        VecDBBackend(
            policy_transport,
            "DOCS",
            DistanceStrategy.COSINE,
            vector_index_params=options,
        )
    policy_transport.call.assert_not_called()
    policy_transport.acall.assert_not_called()
    assert options == original


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("exists", [False, True])
@pytest.mark.parametrize(
    "params", [None, {}, {"idx_type": "HNSW"}, {"idx_type": "IVF", "neighbor_part": 8}]
)
@pytest.mark.asyncio
async def test_native_ensure_or_reconfigure(asynchronous, exists, params):
    transport = (AsyncNativeTransport if asynchronous else NativeTransport)(
        object(), (23, 26, 3)
    )
    # An unusable index still exists; readiness cannot authorize another CREATE.
    status = {
        "ready": False,
        "details": {"indexes": [{"index_name": "GENERATED"}] if exists else []},
    }
    transport.describe_index = MagicMock(return_value=status)
    transport.adescribe_index = AsyncMock(return_value=status)
    transport.call = MagicMock(return_value={"message": "done"})
    transport.acall = AsyncMock(return_value={"message": "done"})
    backend = VecDBBackend(transport, "DOCS", DistanceStrategy.COSINE)
    original = deepcopy(params)
    result = (
        await backend.run_async("create_index", params)
        if asynchronous
        else backend.create_index(params)
    )
    call = transport.acall if asynchronous else transport.call
    if exists and not params:
        assert result is None
        call.assert_not_called()
    else:
        assert result == {"message": "done"}
        (operation,) = call.call_args.args
        request = call.call_args.kwargs
        assert operation == ("rebuild_index" if exists else "create_index")
        assert request["table_name"] == '"DOCS"'
        if params:
            mapped = request["index_params"]
            assert (mapped.get("index_type") == "vector") == exists
            vector = mapped["vector_index_params"]
            assert "distribute_params" not in vector
            assert vector["distance_metric"] == "COSINE"
            if params["idx_type"] == "IVF":
                assert vector["advanced_params"] == {"partitions": 8}
            sql, _ = SQLNetAdapter((23, 26, 3)).statement(operation, request)
            assert "dbms_vector_database." + operation in sql.lower()
        else:
            assert "index_params" not in request
    assert params == original


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("kind", ["HNSW", "IVF"])
@pytest.mark.asyncio
async def test_creation_options_only_used_for_new_native_table(asynchronous, kind):
    transport = (AsyncNativeTransport if asynchronous else NativeTransport)(
        object(), (23, 26, 3)
    )
    columns = (
        AsyncMock(side_effect=[set(), CURRENT_COLUMNS, CURRENT_COLUMNS])
        if asynchronous
        else MagicMock(side_effect=[set(), CURRENT_COLUMNS, CURRENT_COLUMNS])
    )
    setattr(transport, "acolumns" if asynchronous else "columns", columns)
    setattr(
        transport,
        "adiscover" if asynchronous else "discover",
        AsyncMock() if asynchronous else MagicMock(),
    )
    call = AsyncMock(return_value={}) if asynchronous else MagicMock(return_value={})
    setattr(transport, "acall" if asynchronous else "call", call)
    options = {"idx_type": kind, "accuracy": 85, "parallel": 2}
    backend = VecDBBackend(
        transport, "DOCS", DistanceStrategy.COSINE, vector_index_params=options
    )
    for _ in range(2):
        if asynchronous:
            await backend.run_async("prepare")
        else:
            backend.prepare()
    call.assert_called_once()
    assert call.call_args.args == ("create_vector_table",)
    mapped = call.call_args.kwargs["index_params"]
    assert mapped["parallel_creation"] == 2
    assert mapped["vector_index_params"]["accuracy"] == 85
    assert "distribute_params" not in mapped["vector_index_params"]
    assert options == {"idx_type": kind, "accuracy": 85, "parallel": 2}


@pytest.mark.parametrize(
    "state", ["READY", "BUILDING", "PENDING", "FAILED", "UNUSABLE"]
)
def test_rest_existing_default_is_noop_and_explicit_options_rebuild(state):
    transport = RestTransport.__new__(RestTransport)
    transport.client = SimpleNamespace(
        describe_index=MagicMock(return_value={"index_status": state}),
        rebuild_index=MagicMock(return_value={"job_id": "rebuild"}),
        create_index=MagicMock(),
    )
    backend = VecDBBackend(transport, "DOCS", DistanceStrategy.COSINE)
    assert backend.create_index(None) is None
    assert backend.create_index({"idx_type": "IVF"}) == {"job_id": "rebuild"}
    request = transport.client.rebuild_index.call_args.kwargs["index_params"]
    assert request["index_type"] == "vector"
    assert request["vector_index_params"]["advanced_params"] == {"partitions": 32}
    transport.client.create_index.assert_not_called()


def test_rest_unknown_status_is_not_treated_as_missing():
    transport = RestTransport.__new__(RestTransport)
    transport.client = SimpleNamespace(
        describe_index=MagicMock(return_value={"index_status": "NEW_STATE"}),
        create_index=MagicMock(),
    )
    backend = VecDBBackend(transport, "DOCS", DistanceStrategy.COSINE)
    with pytest.raises(ValueError, match="existence"):
        backend.create_index(None)
    transport.client.create_index.assert_not_called()


@pytest.mark.parametrize("status", [401, 403, 404, 500])
def test_rest_only_missing_index_with_existing_table_allows_create(status):
    error = RuntimeError("provider failure")
    error.status = status
    transport = RestTransport.__new__(RestTransport)
    transport.client = SimpleNamespace(
        describe_index=MagicMock(side_effect=error),
        describe_vector_table=MagicMock(return_value={"table_name": "DOCS"}),
        create_index=MagicMock(return_value={"job_id": "create"}),
    )
    backend = VecDBBackend(transport, "DOCS", DistanceStrategy.COSINE)
    if status == 404:
        assert backend.create_index(None) == {"job_id": "create"}
        transport.client.describe_vector_table.assert_called_once_with(name="DOCS")
        transport.client.create_index.assert_called_once_with(table_name="DOCS")
    else:
        with pytest.raises(RuntimeError, match="provider failure"):
            backend.create_index(None)
        transport.client.create_index.assert_not_called()


@pytest.mark.parametrize(
    "operation", ["create_vector_table", "create_index", "rebuild_index"]
)
def test_hnsw_distribution_is_optional_but_validated_when_supplied(operation):
    from langchain_oracledb._vecdb_sdk.parameter_validation import (
        validate_operation_arguments,
    )

    vector = {"organization": "INMEMORY GRAPH"}
    params = {"vector_index_params": vector}
    validate_operation_arguments(operation, {"index_params": params})
    vector["distribute_params"] = {"distribute_method": "invalid"}
    with pytest.raises(ValueError, match="distribute_method"):
        validate_operation_arguments(operation, {"index_params": params})


def test_missing_rest_table_error_is_not_replaced_by_index_creation():
    error = RuntimeError("missing")
    error.status = 404
    transport = RestTransport.__new__(RestTransport)
    transport.client = SimpleNamespace(
        describe_index=MagicMock(side_effect=error),
        describe_vector_table=MagicMock(side_effect=error),
        create_index=MagicMock(),
    )
    backend = VecDBBackend(transport, "DOCS", DistanceStrategy.COSINE)
    with pytest.raises(RuntimeError, match="missing"):
        backend.create_index(None)
    transport.client.create_index.assert_not_called()


def test_index_metric_cannot_diverge_from_store_strategy():
    backend = VecDBBackend(
        NativeTransport(object(), (23, 26, 3)), "DOCS", DistanceStrategy.COSINE
    )
    with pytest.raises(ValueError, match="distance_strategy"):
        backend.index_params({"idx_type": "HNSW", "distance_metric": "EUCLIDEAN"})


@pytest.mark.parametrize("params", [[], "HNSW", 0])
def test_malformed_creation_options_rejected_before_database_access(params):
    with pytest.raises(TypeError, match="dictionary"):
        VecDBBackend(
            NativeTransport(object(), (23, 26, 3)),
            "DOCS",
            DistanceStrategy.COSINE,
            vector_index_params=params,
        )


def test_null_distribution_is_omitted():
    backend = VecDBBackend(
        NativeTransport(object(), (23, 26, 3)), "DOCS", DistanceStrategy.COSINE
    )
    result = backend.index_params({"idx_type": "HNSW", "distribute_params": None})
    assert "distribute_params" not in result["vector_index_params"]


@pytest.mark.parametrize("kind", ["HNSW", "IVF"])
def test_rest_named_index_completion_is_recognized(kind):
    from langchain_oracledb.vectorstores._vecdb_capabilities import sdk_index_status

    response = {
        "index_status": f'"OWNER"."VECIDX$_123_DENSE" {kind} Index Creation Completed'
    }
    status = sdk_index_status(response, False)
    assert status["ready"] is True
    assert status["details"] is response
    assert sdk_index_status(response, True)["ready"] is None
    transport = RestTransport.__new__(RestTransport)
    transport.client = SimpleNamespace(
        describe_index=MagicMock(return_value=response), create_index=MagicMock()
    )
    backend = VecDBBackend(transport, "DOCS", DistanceStrategy.COSINE)
    assert backend.create_index(None) is None
    transport.client.create_index.assert_not_called()
