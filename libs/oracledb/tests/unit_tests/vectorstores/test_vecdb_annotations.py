# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""VecDB creation markers use the shared contract on every transport."""

import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from langchain_oracledb._annotations import table_annotations
from langchain_oracledb.cache import _SemanticCacheVectorStore
from langchain_oracledb.vectorstores._vecdb import (
    CURRENT_COLUMNS,
    AsyncNativeTransport,
    NativeTransport,
    VecDBBackend,
)
from langchain_oracledb.vectorstores.utils import DistanceStrategy


@pytest.mark.parametrize("kind", ["vector_store", "semantic_cache"])
@pytest.mark.parametrize("version", [(23, 26, 1), (23, 26, 3)])
@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_native_creation_and_reopen(
    kind, version, existing, asynchronous, caplog
):
    caplog.set_level(logging.INFO, logger="langchain_oracledb.vectorstores._vecdb")
    transport = (AsyncNativeTransport if asynchronous else NativeTransport)(
        object(), version
    )
    mock = AsyncMock if asynchronous else MagicMock
    columns = "acolumns" if asynchronous else "columns"
    call = "acall" if asynchronous else "call"
    discover = "adiscover" if asynchronous else "discover"
    with (
        patch.object(
            transport,
            columns,
            mock(
                side_effect=[CURRENT_COLUMNS] if existing else [set(), CURRENT_COLUMNS]
            ),
        ),
        patch.object(transport, call, mock()) as create,
        patch.object(transport, discover, mock()),
    ):
        backend = VecDBBackend(
            transport, "DOCS", DistanceStrategy.COSINE, table_type=kind
        )
        if asynchronous:
            await backend.run_async("prepare")
        else:
            backend.prepare()
        message = (
            'Table "DOCS" already exists...'
            if existing
            else 'Table "DOCS" created successfully...'
        )
        assert caplog.records[-1].levelno == logging.INFO
        assert caplog.records[-1].getMessage() == message
        if existing:
            create.assert_not_called()
        else:
            create.assert_called_once_with(
                "create_vector_table",
                name='"DOCS"',
                table_params={"auto_generate_id": False},
                annotations=table_annotations(kind, "vecdb"),
            )
            # Both package generations bind annotations as JSON, never SQL text.
            sql, binds = transport._statement(
                "create_vector_table", create.call_args.kwargs
            )
            assert "annotations => JSON(:annotations)" in sql
            assert json.loads(binds["annotations"]) == table_annotations(kind, "vecdb")


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_rest_semantic_cache_creation_and_reopen(asynchronous, caplog):
    caplog.set_level(logging.INFO, logger="langchain_oracledb.vectorstores._vecdb")
    from langchain_core.embeddings import DeterministicFakeEmbedding

    sdk = pytest.importorskip("oracle_vecdb")
    client = sdk.OracleVecDB(
        sdk.Configuration(
            rest_url="https://example.com/ords/test/_/db-api/stable/vecdb/",
            access_token="test",
        )
    )
    error = RuntimeError("missing")
    error.status = 404
    client.describe_vector_table = MagicMock(side_effect=[error, {}])
    client.create_vector_table = MagicMock(return_value={})
    with patch("langchain_oracledb._annotations.version", return_value="1.6.0"):
        for _ in range(2):
            if asynchronous:
                await _SemanticCacheVectorStore.acreate(
                    client, DeterministicFakeEmbedding(size=3), "Tb1"
                )
            else:
                _SemanticCacheVectorStore(
                    client, DeterministicFakeEmbedding(size=3), "Tb1"
                )
    assert [
        record.getMessage()
        for record in caplog.records
        if record.name == "langchain_oracledb.vectorstores._vecdb"
        and record.levelno == logging.INFO
    ] == [
        'Table "Tb1" created successfully...',
        'Table "Tb1" already exists...',
    ]
    client.create_vector_table.assert_called_once_with(
        name="Tb1",
        table_params={"auto_generate_id": False},
        annotations={
            "LANGCHAIN_ORACLE": (
                "format=1;package=langchain-oracledb;version=1.6.0;"
                "type=semantic_cache;backend=vecdb"
            )
        },
    )


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_failed_native_creation_does_not_log_success(asynchronous, caplog):
    caplog.set_level(logging.INFO, logger="langchain_oracledb.vectorstores._vecdb")
    transport = (AsyncNativeTransport if asynchronous else NativeTransport)(
        object(), (23, 26, 3)
    )
    mock = AsyncMock if asynchronous else MagicMock
    with (
        patch.object(
            transport,
            "acolumns" if asynchronous else "columns",
            mock(return_value=set()),
        ),
        patch.object(
            transport,
            "acall" if asynchronous else "call",
            mock(side_effect=RuntimeError("creation failed")),
        ),
    ):
        backend = VecDBBackend(transport, "DOCS", DistanceStrategy.COSINE)
        with pytest.raises(RuntimeError, match="creation failed"):
            if asynchronous:
                await backend.run_async("prepare")
            else:
                backend.prepare()
    assert not [
        record
        for record in caplog.records
        if record.name == "langchain_oracledb.vectorstores._vecdb"
        and record.levelno == logging.INFO
    ]
