# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""Case-preserving identifiers through public factories, CRUD, and cleanup."""

import uuid

import oracledb
import pytest

from langchain_oracledb import OracleVS
from langchain_oracledb.vectorstores.utils import (
    _aget_connection,
    _get_connection,
    _quote_indentifier,
    adrop_table_purge,
    drop_table_purge,
)

from ._backend_fixtures import async_backend_store, backend_store, native_config
from .test_backend_contract import ContractEmbeddings


@pytest.mark.parametrize(
    "style",
    [
        "uppercase",
        "quoted-uppercase",
        "lowercase",
        "mixed",
        "quoted-lowercase",
        "literal-dot",
        "apostrophe",
    ],
)
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.asyncio
async def test_identifier_crud_and_case_isolation(oracle_backend, asynchronous, style):
    if oracle_backend == "rest" and style != "uppercase":
        pytest.skip(
            "The REST SDK rejects double quotes in resource names; "
            "case-sensitive native package identifiers require those quotes"
        )
    suffix = uuid.uuid4().hex[:12]
    names = {
        "uppercase": f"LCID_{suffix.upper()}",
        "quoted-uppercase": f'"LCID_{suffix.upper()}"',
        "lowercase": f"lcid_{suffix}",
        "mixed": f"LCid_{suffix}",
        "quoted-lowercase": f'"lcid_{suffix}"',
        "literal-dot": f'"lcid.{suffix}"',
        "apostrophe": f'"lcid_o\'brien_{suffix}"',
    }
    name = names[style]
    quoted = _quote_indentifier(name)
    embeddings = ContractEmbeddings()

    if oracle_backend.startswith("vecdb") and style != "uppercase":
        with oracledb.connect(**native_config(oracle_backend)) as connection:
            version = tuple(int(part) for part in connection.version.split(".")[:3])
            if version < (23, 27, 2):
                # A rejected name must leave neither the exact nor folded table.
                with pytest.raises(ValueError, match="unquoted uppercase.*23.27.2"):
                    if asynchronous:
                        async with async_backend_store(
                            oracle_backend, embeddings, table_name=name
                        ):
                            pytest.fail("Unsupported VecDB name was accepted")
                    else:
                        with backend_store(oracle_backend, embeddings, table_name=name):
                            pytest.fail("Unsupported VecDB name was accepted")
                with connection.cursor() as cursor:
                    cursor.execute(
                        "SELECT table_name FROM user_tables "
                        "WHERE table_name IN (:exact,:folded)",
                        exact=quoted[1:-1],
                        folded=quoted[1:-1].upper(),
                    )
                    assert cursor.fetchall() == []
                return

    async def check(store):
        assert store.table_name == quoted
        if oracle_backend != "rest":
            sql = "SELECT COUNT(*) FROM user_tables WHERE table_name=:name"
            if asynchronous:
                async with _aget_connection(store.client) as connection:
                    with connection.cursor() as cursor:
                        await cursor.execute(sql, name=quoted[1:-1])
                        assert (await cursor.fetchone())[0] == 1
            else:
                with _get_connection(store.client) as connection:
                    with connection.cursor() as cursor:
                        cursor.execute(sql, name=quoted[1:-1])
                        assert cursor.fetchone()[0] == 1
        kwargs = dict(
            client=store.client,
            table_name=name,
            backend=store.backend,
            mutate_on_duplicate=True,
        )
        if asynchronous:
            factory = await OracleVS.afrom_texts(
                ["oracle original"], embeddings, ids=["one"], **kwargs
            )
            assert await factory.aadd_texts(["oracle updated"], ids=["one"]) == ["one"]
            reopened = await OracleVS.acreate(
                store.client,
                embeddings,
                name,
                backend="auto",
                mutate_on_duplicate=store.mutate_on_duplicate,
            )
            assert (await reopened.aget_by_ids(["one"]))[
                0
            ].page_content == "oracle updated"
            assert (await reopened.asimilarity_search("oracle", k=1))[0].id == "one"
            await reopened.adelete(["one"])
            assert await reopened.aget_by_ids(["one"]) == []
        else:
            factory = OracleVS.from_texts(
                ["oracle original"], embeddings, ids=["one"], **kwargs
            )
            assert factory.add_texts(["oracle updated"], ids=["one"]) == ["one"]
            reopened = OracleVS(
                store.client,
                embeddings,
                name,
                backend="auto",
                mutate_on_duplicate=store.mutate_on_duplicate,
            )
            assert reopened.get_by_ids(["one"])[0].page_content == "oracle updated"
            assert reopened.similarity_search("oracle", k=1)[0].id == "one"
            reopened.delete(["one"])
            assert reopened.get_by_ids(["one"]) == []
        assert factory.table_name == reopened.table_name == quoted
        assert factory.backend == reopened.backend == store.backend

        if style != "lowercase":
            return
        # The same letters in uppercase identify a different table. Reusing the
        # same document ID must not overwrite a document in the original table.
        other_name = '"' + name.strip('"').upper() + '"'
        try:
            if asynchronous:
                await store.aadd_texts(["oracle original"], ids=["one"])
                other = await OracleVS.afrom_texts(
                    ["oracle uppercase"],
                    embeddings,
                    client=store.client,
                    table_name=other_name,
                    backend=store.backend,
                    mutate_on_duplicate=True,
                    ids=["one"],
                )
                assert (await other.aget_by_ids(["one"]))[
                    0
                ].page_content == "oracle uppercase"
                assert (await store.aget_by_ids(["one"]))[
                    0
                ].page_content == "oracle original"
            else:
                store.add_texts(["oracle original"], ids=["one"])
                other = OracleVS.from_texts(
                    ["oracle uppercase"],
                    embeddings,
                    client=store.client,
                    table_name=other_name,
                    backend=store.backend,
                    mutate_on_duplicate=True,
                    ids=["one"],
                )
                assert other.get_by_ids(["one"])[0].page_content == "oracle uppercase"
                assert store.get_by_ids(["one"])[0].page_content == "oracle original"
        finally:
            if asynchronous:
                await adrop_table_purge(store.client, other_name, backend=store.backend)
            else:
                drop_table_purge(store.client, other_name, backend=store.backend)

    if asynchronous:
        async with async_backend_store(
            oracle_backend, embeddings, table_name=name
        ) as store:
            await check(store)
    else:
        with backend_store(oracle_backend, embeddings, table_name=name) as store:
            await check(store)
