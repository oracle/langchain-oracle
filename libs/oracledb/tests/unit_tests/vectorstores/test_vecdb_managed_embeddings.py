# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""Managed ingestion keeps in-database embeddings out of Python write payloads."""

from contextlib import ExitStack
from copy import deepcopy
from unittest.mock import AsyncMock, MagicMock, patch

import oracledb
import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from langchain_oracledb import OracleVS
from langchain_oracledb.embeddings import OracleEmbeddings
from langchain_oracledb.vectorstores._vecdb import (
    CURRENT_COLUMNS,
    AsyncNativeTransport,
    NativeTransport,
    VecDBBackend,
)
from langchain_oracledb.vectorstores._vecdb_adapter import (
    LOGICAL_ID_KEY,
    TEXT_KEY,
    SQLNetAdapter,
    storage_id,
)
from langchain_oracledb.vectorstores.utils import DistanceStrategy

EMBED_PARAMS = {"model": "MODEL", "embed_metadata_jsonpath": TEXT_KEY}


@pytest.fixture(params=["native", "native-async", "rest"])
def environment(request):
    kind = request.param
    existing = {}
    requests = []

    def call(operation, **params):
        requests.append((operation, deepcopy(params)))
        if operation == "create_vector_table":
            existing.update(params)
        return {"embed_params": existing.get("embed_params"), "upserted_count": 1}

    with ExitStack() as stack:
        if kind == "rest":
            sdk = pytest.importorskip("oracle_vecdb")
            client = sdk.OracleVecDB(
                sdk.Configuration(
                    rest_url="https://example.com/ords/test/_/db-api/stable/vecdb/",
                    access_token="x",
                )
            )

            def describe(**params):
                if not existing:
                    error = RuntimeError("missing table")
                    error.status = 404
                    raise error
                return call("describe_vector_table", **params)

            client.describe_vector_table = MagicMock(side_effect=describe)
            client.create_vector_table = MagicMock(
                side_effect=lambda **params: call("create_vector_table", **params)
            )
            client.upsert_vectors = MagicMock(
                side_effect=lambda **params: call("upsert_vectors", **params)
            )
        else:
            asynchronous = kind == "native-async"
            client = MagicMock(
                spec=oracledb.AsyncConnection if asynchronous else oracledb.Connection,
                thin=True,
                version="23.26.3",
            )
            if asynchronous:
                client.__aenter__.return_value = client
            else:
                client.__enter__.return_value = client
            transport = AsyncNativeTransport if asynchronous else NativeTransport
            prefix = "a" if asynchronous else ""
            mock_type = AsyncMock if asynchronous else MagicMock
            selection = "anative_selection" if asynchronous else "native_selection"
            stack.enter_context(
                patch(
                    "langchain_oracledb.vectorstores._vecdb." + selection,
                    new=mock_type(return_value=("vecdb", (23, 26, 3))),
                )
            )
            stack.enter_context(
                patch.object(
                    transport,
                    prefix + "columns",
                    new=mock_type(
                        side_effect=lambda name: CURRENT_COLUMNS if existing else set()
                    ),
                )
            )
            stack.enter_context(
                patch.object(transport, prefix + "discover", new=mock_type())
            )
            stack.enter_context(
                patch.object(
                    transport, prefix + "call", new=mock_type(side_effect=call)
                )
            )
            stack.enter_context(
                patch.object(
                    transport,
                    prefix + "insert",
                    new=mock_type(return_value=[storage_id("a")]),
                )
            )
        yield kind, client, existing, requests


async def create_store(environment, embedding, mutate=True):
    kind, client, _, _ = environment
    if kind == "native-async":
        return await OracleVS.acreate(
            client, embedding, "DOCS", backend="vecdb", mutate_on_duplicate=mutate
        )
    return OracleVS(
        client, embedding, "DOCS", backend="vecdb", mutate_on_duplicate=mutate
    )


@pytest.mark.asyncio
async def test_managed_create_upsert_and_reopen_without_python_embeddings(environment):
    kind, client, existing, requests = environment
    embedding = OracleEmbeddings(
        conn=client, params={"provider": "database", "model": "MODEL"}
    )
    with (
        patch.object(
            OracleEmbeddings,
            "embed_documents",
            side_effect=AssertionError("Python embedding called"),
        ),
        patch.object(
            OracleEmbeddings,
            "aembed_documents",
            side_effect=AssertionError("Python embedding called"),
        ),
    ):
        store = await create_store(environment, embedding)
        assert existing["embed_params"] == EMBED_PARAMS
        assert existing["table_params"] == {"auto_generate_id": False}
        metadata = {"tenant": "one"}
        for text in ["original", "replacement"]:
            if kind == "native":
                assert store.add_texts([text], [metadata], ids=["a"]) == ["a"]
            else:
                assert await store.aadd_texts([text], [metadata], ids=["a"]) == ["a"]
            operation, payload = requests[-1]
            assert operation == "upsert_vectors"
            assert payload["vectors"] == [
                {
                    "id": storage_id("a"),
                    "metadata": {"tenant": "one", TEXT_KEY: text, LOGICAL_ID_KEY: "a"},
                }
            ]
        assert metadata == {"tenant": "one"}
        reopened = await create_store(environment, embedding)
        assert reopened._vecdb.managed_embeddings
        assert sum(operation == "create_vector_table" for operation, _ in requests) == 1


@pytest.mark.parametrize(
    "stored",
    [
        None,
        {**EMBED_PARAMS, "model": "OTHER"},
        {**EMBED_PARAMS, "embed_metadata_jsonpath": "text"},
    ],
)
@pytest.mark.asyncio
async def test_managed_reopen_rejects_mismatched_configuration(environment, stored):
    _, client, existing, requests = environment
    existing.update(name="DOCS", embed_params=stored)
    embedding = OracleEmbeddings(
        conn=client, params={"provider": "database", "model": "MODEL"}
    )
    with pytest.raises(ValueError, match="embedding configuration does not match"):
        await create_store(environment, embedding)
    assert all(operation == "describe_vector_table" for operation, _ in requests)


@pytest.mark.parametrize(
    "provider,extra", [("ocigenai", {}), ("database", {"batch_size": 2})]
)
@pytest.mark.asyncio
async def test_other_embedding_parameters_keep_explicit_vectors(
    environment, provider, extra
):
    kind, client, existing, requests = environment
    embedding = OracleEmbeddings(
        conn=client, params={"provider": provider, "model": "MODEL", **extra}
    )
    with patch.object(
        OracleEmbeddings, "embed_documents", return_value=[[1.0, 0.0]]
    ) as embed:
        store = await create_store(environment, embedding)
        if kind == "native":
            store.add_texts(["text"], ids=["a"])
        else:
            await store.aadd_texts(["text"], ids=["a"])
        embed.assert_called_once_with(["text"])
    assert "embed_params" not in existing
    assert requests[-1][1]["vectors"][0]["dense_vector"] == [1.0, 0.0]


@pytest.mark.asyncio
async def test_external_embeddings_keep_explicit_vectors(environment):
    kind, _, existing, requests = environment
    store = await create_store(environment, DeterministicFakeEmbedding(size=3))
    if kind == "native":
        store.add_texts(["text"], ids=["a"])
    else:
        await store.aadd_texts(["text"], ids=["a"])
    assert "embed_params" not in existing
    assert len(requests[-1][1]["vectors"][0]["dense_vector"]) == 3


@pytest.mark.asyncio
async def test_insert_only_oracle_embeddings_keep_explicit_vectors(environment):
    kind, client, existing, _ = environment
    if kind == "rest":
        pytest.skip("REST insert-only writes remain unsupported")
    embedding = OracleEmbeddings(
        conn=client, params={"provider": "database", "model": "MODEL"}
    )
    with patch.object(
        OracleEmbeddings, "embed_documents", return_value=[[1.0, 0.0]]
    ) as embed:
        store = await create_store(environment, embedding, mutate=False)
        if kind == "native":
            assert store.add_texts(["text"], ids=["a"]) == ["a"]
            insert = store._vecdb.transport.insert
        else:
            assert await store.aadd_texts(["text"], ids=["a"]) == ["a"]
            insert = store._vecdb.transport.ainsert
        embed.assert_called_once_with(["text"])
    assert "embed_params" not in existing
    assert insert.call_args.args[1][0]["dense_vector"] == [1.0, 0.0]


@pytest.mark.parametrize("model", ["model", '"MODEL"'])
@pytest.mark.asyncio
async def test_reopen_normalizes_unquoted_model_names(environment, model):
    _, client, existing, _ = environment
    existing.update(name="DOCS", embed_params=EMBED_PARAMS)
    embedding = OracleEmbeddings(
        conn=client, params={"provider": "database", "model": model}
    )
    store = await create_store(environment, embedding)
    assert store._vecdb.managed_embeddings


@pytest.mark.parametrize("version,mutate", [((23, 26, 2), True), ((23, 26, 3), False)])
def test_legacy_and_insert_only_stores_keep_byov(version, mutate):
    embedding = OracleEmbeddings(params={"provider": "database", "model": "MODEL"})
    backend = VecDBBackend(
        NativeTransport(object(), version),
        "DOCS",
        DistanceStrategy.COSINE,
        embedding=embedding,
        mutate_on_duplicate=mutate,
    )
    assert not backend.managed_embeddings
    assert "embed_params" not in backend._table_creation_params()


def test_managed_parameters_reach_native_package_as_json():
    statement, binds = SQLNetAdapter((23, 26, 3)).statement(
        "create_vector_table", {"name": "DOCS", "embed_params": EMBED_PARAMS}
    )
    assert "embed_params => JSON(:embed_params)" in statement
    assert '"model": "MODEL"' in binds["embed_params"]
