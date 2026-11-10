# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""Shared index aliases and SQL constructor index lifecycle."""

from copy import deepcopy
from unittest.mock import AsyncMock, MagicMock, patch

import oracledb
import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from langchain_oracledb import OracleVS
from langchain_oracledb.vectorstores import oraclevs
from langchain_oracledb.vectorstores._index_params import normalize_index_params
from langchain_oracledb.vectorstores._vecdb import NativeTransport, VecDBBackend
from langchain_oracledb.vectorstores.utils import DistanceStrategy


@pytest.mark.parametrize(
    ("legacy", "vecdb"),
    [
        (
            {"idx_type": "HNSW", "neighbors": 16, "efConstruction": 100, "parallel": 2},
            {
                "organization": "INMEMORY GRAPH",
                "advanced_params": {"neighbors": 16, "efConstruction": 100},
                "parallel_creation": 2,
            },
        ),
        (
            {"idx_type": "IVF", "neighbor_part": 8},
            {"organization": "PARTITIONS", "advanced_params": {"partitions": 8}},
        ),
        (
            {"idx_type": "IVF", "neighbor_part": 8},
            {"organization": "PARTITIONS", "partitions": 8},
        ),
    ],
)
def test_aliases_produce_identical_sql_and_vecdb_requests(legacy, vecdb):
    original = deepcopy(vecdb)
    assert normalize_index_params(vecdb) == legacy
    backend = VecDBBackend(
        NativeTransport(object(), (23, 26, 3)), "DOCS", DistanceStrategy.COSINE
    )
    assert backend.index_params(vecdb) == backend.index_params(legacy)
    # Pin the generated name to compare complete SQL DDL.
    assert oraclevs._sql_index_ddl(
        "DOCS", DistanceStrategy.COSINE, {**vecdb, "idx_name": "MY_IDX"}
    ) == oraclevs._sql_index_ddl(
        "DOCS", DistanceStrategy.COSINE, {**legacy, "idx_name": "MY_IDX"}
    )
    assert vecdb == original


@pytest.mark.parametrize(
    "params",
    [
        {"idx_type": "HNSW", "organization": "PARTITIONS"},
        {"parallel": 2, "parallel_creation": 3},
        {"neighbors": 16, "advanced_params": {"neighbors": 32}},
        {"idx_type": "IVF", "neighbor_part": 4, "advanced_params": {"partitions": 8}},
        {"idx_type": "IVF", "partitions": 8, "advanced_params": {"partitions": 4}},
    ],
)
def test_conflicting_aliases_rejected(params):
    with pytest.raises(ValueError, match="Conflicting"):
        normalize_index_params(params)


def test_matching_aliases_and_case_are_accepted():
    assert normalize_index_params(
        {
            "idx_type": "hnsw",
            "organization": "inmemory graph",
            "neighbors": 16,
            "advanced_params": {"neighbors": 16},
            "distance_metric": "cosine",
        },
        metric="COSINE",
    ) == {"idx_type": "HNSW", "neighbors": 16, "distance_metric": "COSINE"}


@pytest.mark.parametrize(
    "params, error",
    [
        ({"advanced_params": []}, TypeError),
        ({"organization": "INVALID"}, ValueError),
        (
            {"organization": "PARTITIONS", "advanced_params": {"neighbors": 16}},
            ValueError,
        ),
        ({"advanced_params": {"partitions": 8}}, ValueError),
        ({"distance_metric": "DOT"}, ValueError),
        ({"advanced_params": {"neighbors": True}}, ValueError),
    ],
)
def test_invalid_options_rejected(params, error):
    backend = VecDBBackend(
        NativeTransport(object(), (23, 26, 3)), "DOCS", DistanceStrategy.COSINE
    )
    with pytest.raises(error):
        backend.index_params(params)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "index_request",
    [None, {}, {"organization": "PARTITIONS", "advanced_params": {"partitions": 8}}],
)
@pytest.mark.asyncio
async def test_sql_constructor_ensures_requested_index(asynchronous, index_request):
    conn = MagicMock(
        spec=oracledb.AsyncConnection if asynchronous else oracledb.Connection
    )
    prefix = "langchain_oracledb.vectorstores."
    with (
        patch(prefix + "_vecdb._inspect_columns", return_value=set()),
        patch(prefix + "_vecdb._ainspect_columns", new=AsyncMock(return_value=set())),
        patch(prefix + "_vecdb.native_selection", return_value=("sql", (23, 26, 3))),
        patch(
            prefix + "_vecdb.anative_selection",
            new=AsyncMock(return_value=("sql", (23, 26, 3))),
        ),
        patch.object(OracleVS, "_initialize"),
        patch.object(OracleVS, "get_embedding_dimension", return_value=3),
        patch.object(
            OracleVS, "aget_embedding_dimension", new=AsyncMock(return_value=3)
        ),
        patch.object(oraclevs, "_create_table") as create,
        patch.object(oraclevs, "_acreate_table", new_callable=AsyncMock) as acreate,
        patch.object(oraclevs, "_ensure_sql_constructor_index") as ensure,
        patch.object(
            oraclevs, "_aensure_sql_constructor_index", new_callable=AsyncMock
        ) as aensure,
    ):
        # _initialize is isolated here to avoid unrelated version/annotation queries.
        def initialize(self, *args):
            self.table_name = "DOCS"

        with patch.object(OracleVS, "_initialize", initialize):
            options = (
                {} if index_request is None else {"vector_index_params": index_request}
            )
            kwargs = dict(
                client=conn,
                embedding_function=DeterministicFakeEmbedding(size=3),
                table_name="DOCS",
                backend="sql",
                params=options,
            )
            if asynchronous:
                await OracleVS.acreate(**kwargs)
            else:
                OracleVS(**kwargs)
        creator, ensurer = (acreate, aensure) if asynchronous else (create, ensure)
        creator.assert_called_once()
        if index_request is None:
            ensurer.assert_not_called()
        else:
            ensurer.assert_called_once()
            ddl = ensurer.call_args.args[2][1]
            assert ("NEIGHBOR PARTITIONS" in ddl) == bool(index_request)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("exists", [False, True])
@pytest.mark.asyncio
async def test_constructor_preserves_existing_sql_index(asynchronous, exists):
    conn = MagicMock()
    cursor = conn.cursor.return_value.__enter__.return_value
    cursor.fetchone = (
        AsyncMock(return_value=("EXISTING",) if exists else None)
        if asynchronous
        else MagicMock(return_value=("EXISTING",) if exists else None)
    )
    if asynchronous:
        cursor.execute = AsyncMock()
        await oraclevs._aensure_sql_constructor_index(
            conn, "DOCS", ("NEW", "CREATE VECTOR INDEX ...")
        )
    else:
        oraclevs._ensure_sql_constructor_index(
            conn, "DOCS", ("NEW", "CREATE VECTOR INDEX ...")
        )
    assert cursor.execute.call_count == (1 if exists else 2)
    assert cursor.execute.call_args_list[0].kwargs == {"table_name": "DOCS"}
    if not exists:
        assert cursor.execute.call_args.args == ("CREATE VECTOR INDEX ...",)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_sql_invalid_options_fail_before_table_creation(asynchronous):
    conn = MagicMock(
        spec=oracledb.AsyncConnection if asynchronous else oracledb.Connection
    )
    with (
        patch(
            "langchain_oracledb.vectorstores._vecdb._inspect_columns",
            return_value=set(),
        ),
        patch(
            "langchain_oracledb.vectorstores._vecdb._ainspect_columns",
            new=AsyncMock(return_value=set()),
        ),
        patch(
            "langchain_oracledb.vectorstores._vecdb.native_selection",
            return_value=("sql", (23, 26, 3)),
        ),
        patch(
            "langchain_oracledb.vectorstores._vecdb.anative_selection",
            new=AsyncMock(return_value=("sql", (23, 26, 3))),
        ),
        patch.object(oraclevs, "_create_table") as create,
        patch.object(oraclevs, "_acreate_table", new_callable=AsyncMock) as acreate,
    ):
        with pytest.raises(ValueError, match="Invalid"):
            args = (conn, DeterministicFakeEmbedding(size=3), "DOCS")
            kwargs = {
                "params": {
                    "vector_index_params": {"advanced_params": {"algorithm": "CUSTOM"}}
                },
                "backend": "sql",
            }
            if asynchronous:
                await OracleVS.acreate(*args, **kwargs)
            else:
                OracleVS(*args, **kwargs)
        create.assert_not_called()
        acreate.assert_not_called()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("kind", ["HNSW", "IVF"])
@pytest.mark.asyncio
async def test_public_sql_index_api_normalizes_without_mutation(asynchronous, kind):
    conn = MagicMock(
        spec=oracledb.AsyncConnection if asynchronous else oracledb.Connection
    )
    store = MagicMock()
    store._vecdb = None
    store.table_name = '"DOCS"'
    store.distance_strategy = DistanceStrategy.COSINE
    options = {
        "organization": "INMEMORY GRAPH" if kind == "HNSW" else "PARTITIONS",
        "advanced_params": {"neighbors": 16} if kind == "HNSW" else {"partitions": 8},
        "parallel_creation": 2,
    }
    original = deepcopy(options)
    name = ("_acreate_" if asynchronous else "_create_") + kind.lower() + "_index"
    mock = AsyncMock() if asynchronous else MagicMock()
    with patch.object(oraclevs, name, mock):
        if asynchronous:
            await oraclevs.acreate_index(conn, store, options)
        else:
            oraclevs.create_index(conn, store, options)
    mock.assert_called_once_with(
        conn,
        '"DOCS"',
        DistanceStrategy.COSINE,
        {
            "idx_type": kind,
            "neighbors" if kind == "HNSW" else "neighbor_part": 16
            if kind == "HNSW"
            else 8,
            "parallel": 2,
        },
    )
    assert options == original


def test_removed_factory_keyword_is_not_silently_ignored():
    with pytest.raises(TypeError, match="inside params"):
        OracleVS._from_texts_helper(client=object(), vector_index_params={})
