# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""Catalog discovery is scoped to explicit features, never routine API reads."""

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from langchain_oracledb import OracleVS

from .test_vecdb_pool_sessions import GUIDE, use_one_session
from .test_vecdb_pool_sessions import sessions as sessions


def catalogs(log):
    return [sql for sql, _ in log if " FROM all_" in sql or " JOIN all_" in sql]


async def open_store(s, params=None):
    embedding = DeterministicFakeEmbedding(size=1)
    if s.asynchronous:
        return await OracleVS.acreate(
            s.pool, embedding, "DOCS", mutate_on_duplicate=True, params=params
        )
    return OracleVS(s.pool, embedding, "DOCS", mutate_on_duplicate=True, params=params)


@pytest.mark.parametrize("metadata", ["METADATA", "CONTENT_METADATA"])
@pytest.mark.asyncio
async def test_existing_native_store_has_no_catalog_preflights_or_read_overhead(
    sessions, metadata
):
    s = sessions
    use_one_session(s)
    s.a.columns = {"ID", "DENSE_VECTOR", metadata}
    store = await open_store(s)
    assert store.backend == "vecdb"
    assert catalogs(s.a.log) == []
    assert not any("NAME_RESOLVE" in sql for sql, _ in s.a.log)
    s.a.log.clear()
    operations = (
        ("similarity_search", ("oracle",), {}),
        ("similarity_search_by_vector", ([1.0],), {}),
        ("get_by_ids", (["SCHEMA_A"],), {}),
        ("delete", (["one"],), {}),
        ("add_texts", (["oracle"],), {"ids": ["one"]}),
    )
    for _ in range(2):
        for method, args, kwargs in operations:
            before = len(s.a.log)
            result = (
                await getattr(store, "a" + method)(*args, **kwargs)
                if s.asynchronous
                else getattr(store, method)(*args, **kwargs)
            )
            if method != "delete":
                assert result is not None
            assert len(s.a.log) == before + 1, (method, s.a.log[before:])
    assert catalogs(s.a.log) == []
    assert not s.b.log


@pytest.mark.asyncio
async def test_requested_vector_index_uses_provider_status_without_catalog(sessions):
    s = sessions
    use_one_session(s)
    store = await open_store(s, {"vector_index_params": {}})
    assert store.backend == "vecdb"
    assert sum("INDEX_BUILD_STATUS" in sql for sql, _ in s.a.log) == 1
    assert catalogs(s.a.log) == []


@pytest.mark.asyncio
async def test_hybrid_discovery_is_lazy_and_reused_by_subsequent_queries(sessions):
    s = sessions
    use_one_session(s)
    s.a.parameters = "SIMPLIFIED_JSON_HYBRID " + GUIDE
    store = await open_store(s)
    assert catalogs(s.a.log) == []
    s.a.log.clear()
    transport = store._vecdb.transport
    request = {"text": {"search_text": "oracle"}}
    if s.asynchronous:
        docs = await transport.ahybrid_search(store.table_name, request, False)
    else:
        docs = transport.hybrid_search(store.table_name, request, False)
    assert docs[0].page_content == "SCHEMA_A"
    assert len(catalogs(s.a.log)) == 2
    for _ in range(2):
        s.a.log.clear()
        if s.asynchronous:
            await transport.ahybrid_search(store.table_name, request, False)
        else:
            transport.hybrid_search(store.table_name, request, False)
        assert len(s.a.log) == 1
        assert catalogs(s.a.log) == []


@pytest.mark.parametrize("missing_api", [True, False, 1418])
@pytest.mark.asyncio
async def test_vector_index_catalog_fallback_only_when_api_cannot_prove_existence(
    sessions, missing_api
):
    from types import SimpleNamespace

    import oracledb

    s = sessions
    use_one_session(s)
    store = await open_store(s)
    original = s.a.execute

    def execute(statement, parameters=None, **binds):
        original(statement, parameters, **binds)
        if "INDEX_BUILD_STATUS" in statement:
            if missing_api:
                raise oracledb.DatabaseError(
                    SimpleNamespace(
                        code=904 if missing_api is True else 1418,
                        message='ORA-00904: "INDEX_BUILD_STATUS": invalid identifier',
                    )
                )
            s.a.one = ({"Index Status": "Index build in progress: 75%"},)

    s.a.cursor.execute.side_effect = execute
    s.a.log.clear()
    if s.asynchronous:
        await store._vecdb.run_async("create_index", None)
    else:
        store._vecdb.create_index(None)
    assert len(catalogs(s.a.log)) == 1
    assert not any(
        "create_index(" in sql.lower() or "rebuild_index(" in sql.lower()
        for sql, _ in s.a.log
    )


@pytest.mark.asyncio
async def test_unrelated_index_status_error_does_not_trigger_catalog_fallback(sessions):
    from types import SimpleNamespace

    import oracledb

    s = sessions
    use_one_session(s)
    store = await open_store(s)
    original = s.a.execute
    error = oracledb.DatabaseError(
        SimpleNamespace(
            code=904, message='ORA-00904: "OTHER_COLUMN": invalid identifier'
        )
    )

    def execute(statement, parameters=None, **binds):
        original(statement, parameters, **binds)
        if "INDEX_BUILD_STATUS" in statement:
            raise error

    s.a.cursor.execute.side_effect = execute
    s.a.log.clear()
    with pytest.raises(oracledb.DatabaseError) as caught:
        if s.asynchronous:
            await store._vecdb.run_async("create_index", None)
        else:
            store._vecdb.create_index(None)
    assert caught.value is error
    assert catalogs(s.a.log) == []
