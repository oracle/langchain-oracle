# Copyright (c) 2024, 2026, Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""
test_oraclevs.py

Test Oracle AI Vector Search functionality integration
with OracleVS.

Tests asserting physical SQL index names explicitly select the SQL backend.
Package-managed VecDB indexes are covered by test_vecdb_managed_indexes.py.
Portable behavior uses the same backend fixtures for native SQL, VecDB, and REST.
"""

# import required modules
import asyncio
import logging
import os
import sys
import threading

import numpy as np
import oracledb
import pytest
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_oracledb.embeddings import OracleEmbeddings
from langchain_oracledb.vectorstores import DistanceStrategy
from langchain_oracledb.vectorstores.oraclevs import (
    INTERNAL_ID_KEY,
    OracleVS,
    _acreate_table,
    _aindex_exists,
    _atable_exists,
    _create_table,
    _index_exists,
    _quote_indentifier,
    _table_exists,
    acreate_index,
    adrop_index_if_exists,
    adrop_table_purge,
    create_index,
    drop_index_if_exists,
    drop_table_purge,
)

from ._backend_fixtures import async_backend_store, backend_store, native_config
from .test_backend_contract import ContractEmbeddings

username = os.environ.get("VECDB_USER")
password = os.environ.get("VECDB_PASS")
dsn = os.environ.get("VECDB_HOST")


@pytest.fixture(scope="module")
def _legacy_native_available():
    config = native_config()
    with oracledb.connect(**config):
        pass


@pytest.fixture(autouse=True)
def _legacy_native_only(request):
    if (
        "oracle_backend" in request.fixturenames
        or request.node.originalname == "test_quote_identifier"
    ):
        return
    if request.config.getoption("--oracle-backends").split(",") == ["rest"]:
        pytest.skip(
            "Requires a native connection: SQL DDL, pools, or in-database embeddings"
        )
    request.getfixturevalue("_legacy_native_available")


@pytest.fixture(scope="module")
def legacy_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


@pytest.fixture(scope="module")
def search_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-mpnet-base-v2"
    )


SEARCH_INDEX_CASES = [
    pytest.param(strategy, kind, id=f"{strategy.name.lower()}-{kind.lower()}")
    for kind in ("HNSW", "IVF")
    for strategy in (
        DistanceStrategy.EUCLIDEAN_DISTANCE,
        DistanceStrategy.DOT_PRODUCT,
        DistanceStrategy.COSINE,
    )
]


############################
####### table_exists #######
############################


def test_table_exists_test() -> None:
    try:
        connection = oracledb.connect(user=username, password=password, dsn=dsn)
    except Exception:
        sys.exit(1)
    # 1. Existing Table:(all capital letters)
    # expectation:True
    _table_exists(connection, _quote_indentifier("V$TRANSACTION"))

    # 2. Existing Table:(all small letters)
    # expectation:True
    _table_exists(connection, _quote_indentifier("v$transaction"))

    # 3. Non-Existing Table
    # expectation:false
    _table_exists(connection, _quote_indentifier("Hello"))

    # 4. Invalid Table Name
    # Expectation:ORA-00903: invalid table name
    with pytest.raises(oracledb.Error):
        _table_exists(connection, "123")

    # 5. Empty String
    # Expectation:ORA-00903: invalid table name
    with pytest.raises(ValueError):
        _table_exists(connection, _quote_indentifier(""))

    """# 6. Special Character
    # Expectation:ORA-00911: #: invalid character after FROM
    with pytest.raises(oracledb.Error):
        _table_exists(connection, "##4")"""

    # 7. Table name length > 128
    # Expectation:ORA-00972: The identifier XXXXXXXXXX...XXXXXXXXXX...
    # exceeds the maximum length of 128 bytes.
    with pytest.raises(oracledb.Error):
        _table_exists(connection, _quote_indentifier("x" * 129))

    # 8. <Schema_Name.Table_Name>
    # Expectation:True
    _create_table(connection, _quote_indentifier("TB1"), 65535)
    assert _table_exists(connection, _quote_indentifier("TB1"))

    # 9. Toggle Case (like TaBlE)
    # Expectation:False - case sensitive
    assert not _table_exists(connection, _quote_indentifier("Tb1"))
    drop_table_purge(connection, _quote_indentifier("TB1"))

    # 10. Table_Name→ "हिन्दी"
    # Expectation:True
    _create_table(connection, _quote_indentifier('"हिन्दी"'), 545)
    assert _table_exists(connection, _quote_indentifier('"हिन्दी"'))
    drop_table_purge(connection, _quote_indentifier('"हिन्दी"'))


@pytest.mark.asyncio
async def test_table_exists_test_async() -> None:
    try:
        connection = await oracledb.connect_async(
            user=username, password=password, dsn=dsn
        )
    except Exception:
        sys.exit(1)
    # 1. Existing Table:(all capital letters)
    # expectation:True
    await _atable_exists(connection, _quote_indentifier("V$TRANSACTION"))

    # 2. Existing Table:(all small letters)
    # expectation:True
    await _atable_exists(connection, _quote_indentifier("v$transaction"))

    # 3. Non-Existing Table
    # expectation:false
    await _atable_exists(connection, _quote_indentifier("Hello"))

    # 4. Invalid Table Name
    # Expectation:ORA-00903: invalid table name
    with pytest.raises(oracledb.Error):
        await _atable_exists(connection, "123")

    # 5. Empty String
    # Expectation:ORA-00903: invalid table name
    with pytest.raises(ValueError):
        await _atable_exists(connection, _quote_indentifier(""))

    """# 6. Special Character
    # Expectation:ORA-00911: #: invalid character after FROM
    with pytest.raises(oracledb.Error):
        await _atable_exists(connection, "##4")"""

    # 7. Table name length > 128
    # Expectation:ORA-00972: The identifier XXXXXXXXXX...XXXXXXXXXX...
    # exceeds the maximum length of 128 bytes.
    with pytest.raises(oracledb.Error):
        await _atable_exists(connection, _quote_indentifier("x" * 129))

    # 8. <Schema_Name.Table_Name>
    # Expectation:True
    await _acreate_table(connection, _quote_indentifier("TB1"), 65535)
    assert await _atable_exists(connection, _quote_indentifier("TB1"))

    # 9. Toggle Case (like TaBlE)
    # Expectation:False - case sensitive
    assert not await _atable_exists(connection, _quote_indentifier("Tb1"))
    await adrop_table_purge(connection, _quote_indentifier("TB1"))

    # 10. Table_Name→ "हिन्दी"
    # Expectation:True
    await _acreate_table(connection, _quote_indentifier('"हिन्दी"'), 545)
    assert await _atable_exists(connection, _quote_indentifier('"हिन्दी"'))
    await adrop_table_purge(connection, _quote_indentifier('"हिन्दी"'))


############################
####### create_table #######
############################


def test_create_table_test() -> None:
    try:
        connection = oracledb.connect(user=username, password=password, dsn=dsn)
    except Exception:
        sys.exit(1)

    # 1. New table - HELLO
    #    Dimension - 100
    # Expectation:table is created
    _create_table(connection, _quote_indentifier("HELLO"), 100)

    # 2. Existing table name
    #    HELLO
    #    Dimension - 110
    # Expectation:Nothing happens
    _create_table(connection, _quote_indentifier("HELLO"), 110)
    drop_table_purge(connection, _quote_indentifier("HELLO"))

    """# 3. New Table - 123 # Quoted names not valid anymore
    #    Dimension - 100
    # Expectation:ORA-00903: invalid table name
    with pytest.raises(oracledb.Error):
        _create_table(connection, _quote_indentifier("123"), 100)
        drop_table_purge(connection, _quote_indentifier("123"))"""

    # 4. New Table - Hello123
    #    Dimension - 65535
    # Expectation:table is created
    _create_table(connection, _quote_indentifier("Hello123"), 65535)
    drop_table_purge(connection, _quote_indentifier("Hello123"))

    # 5. New Table - T1
    #    Dimension - 65536
    # Expectation:ORA-51801: VECTOR column type specification
    # has an unsupported dimension count ('65536').
    with pytest.raises(oracledb.Error):
        _create_table(connection, _quote_indentifier("T1"), 65536)
        drop_table_purge(connection, _quote_indentifier("T1"))

    # 6. New Table - T1
    #    Dimension - 0
    # Expectation:ORA-51801: VECTOR column type specification has
    # an unsupported dimension count (0).
    with pytest.raises(oracledb.Error):
        _create_table(connection, _quote_indentifier("T1"), 0)
        drop_table_purge(connection, _quote_indentifier("T1"))

    # 7. New Table - T1
    #    Dimension - -1
    # Expectation:ORA-51801: VECTOR column type specification has
    # an unsupported dimension count ('-').
    with pytest.raises(oracledb.Error):
        _create_table(connection, _quote_indentifier("T1"), -1)
        drop_table_purge(connection, _quote_indentifier("T1"))

    # 8. New Table - T2
    #     Dimension - '1000'
    # Expectation:table is created
    _create_table(connection, _quote_indentifier("T2"), int("1000"))
    drop_table_purge(connection, _quote_indentifier("T2"))

    # 9. New Table - T3
    #     Dimension - 100 passed as a variable
    # Expectation:table is created
    val = 100
    _create_table(connection, _quote_indentifier("T3"), val)
    drop_table_purge(connection, _quote_indentifier("T3"))

    '''# 10.
    # Expectation:ORA-00922: missing or invalid option
    val2 = """H
    ello"""
    with pytest.raises(oracledb.Error):
        _create_table(connection, _quote_indentifier(val2), 545)
        drop_table_purge(connection, _quote_indentifier(val2))'''

    # 11. New Table - हिन्दी
    #     Dimension - 545
    # Expectation:table is created
    _create_table(connection, _quote_indentifier('"हिन्दी"'), 545)
    drop_table_purge(connection, _quote_indentifier('"हिन्दी"'))

    # 12. <schema_name.table_name>
    # Expectation:failure - user does not exist
    with pytest.raises(oracledb.Error):
        _create_table(connection, _quote_indentifier("U1.TB4"), 128)
        drop_table_purge(connection, _quote_indentifier("U1.TB4"))

    # 13.
    # Expectation:table is created
    _create_table(connection, _quote_indentifier('"T5"'), 128)
    drop_table_purge(connection, _quote_indentifier('"T5"'))

    """# 14. Toggle Case
    # Expectation:table creation fails
    with pytest.raises(oracledb.Error):
        _create_table(connection, _quote_indentifier("TaBlE"), 128)
        drop_table_purge(connection, _quote_indentifier("TaBlE"))"""

    # 15. table_name as empty_string
    # Expectation: ORA-00903: invalid table name
    with pytest.raises(ValueError):
        _create_table(connection, _quote_indentifier(""), 128)
        drop_table_purge(connection, _quote_indentifier(""))
        _create_table(connection, _quote_indentifier('""'), 128)
        drop_table_purge(connection, _quote_indentifier('""'))

    # 16. Arithmetic Operations in dimension parameter
    # Expectation:table is created
    n = 1
    _create_table(connection, _quote_indentifier("T10"), n + 500)
    drop_table_purge(connection, _quote_indentifier("T10"))

    # 17. String Operations in table_name&dimension parameter
    # Expectation:table is created
    _create_table(connection, _quote_indentifier("YaSh".replace("aS", "ok")), 500)
    drop_table_purge(connection, _quote_indentifier("YaSh".replace("aS", "ok")))


@pytest.mark.asyncio
async def test_create_table_test_async() -> None:
    try:
        connection = await oracledb.connect_async(
            user=username, password=password, dsn=dsn
        )
    except Exception:
        sys.exit(1)

    # 1. New table - HELLO
    #    Dimension - 100
    # Expectation:table is created
    await _acreate_table(connection, _quote_indentifier("HELLO"), 100)

    # 2. Existing table name
    #    HELLO
    #    Dimension - 110
    # Expectation:Nothing happens
    await _acreate_table(connection, _quote_indentifier("HELLO"), 110)
    await adrop_table_purge(connection, _quote_indentifier("HELLO"))

    """# 3. New Table - 123 # Quoted names not valid anymore
    #    Dimension - 100
    # Expectation:ORA-00903: invalid table name
    with pytest.raises(oracledb.Error):
        await _acreate_table(connection, _quote_indentifier("123"), 100)
        await adrop_table_purge(connection, _quote_indentifier("123"))"""

    # 4. New Table - Hello123
    #    Dimension - 65535
    # Expectation:table is created
    await _acreate_table(connection, _quote_indentifier("Hello123"), 65535)
    await adrop_table_purge(connection, _quote_indentifier("Hello123"))

    # 5. New Table - T1
    #    Dimension - 65536
    # Expectation:ORA-51801: VECTOR column type specification
    # has an unsupported dimension count ('65536').
    with pytest.raises(oracledb.Error):
        await _acreate_table(connection, _quote_indentifier("T1"), 65536)
        await adrop_table_purge(connection, _quote_indentifier("T1"))

    # 6. New Table - T1
    #    Dimension - 0
    # Expectation:ORA-51801: VECTOR column type specification has
    # an unsupported dimension count (0).
    with pytest.raises(oracledb.Error):
        await _acreate_table(connection, _quote_indentifier("T1"), 0)
        await adrop_table_purge(connection, _quote_indentifier("T1"))

    # 7. New Table - T1
    #    Dimension - -1
    # Expectation:ORA-51801: VECTOR column type specification has
    # an unsupported dimension count ('-').
    with pytest.raises(oracledb.Error):
        await _acreate_table(connection, _quote_indentifier("T1"), -1)
        await adrop_table_purge(connection, _quote_indentifier("T1"))

    # 8. New Table - T2
    #     Dimension - '1000'
    # Expectation:table is created
    await _acreate_table(connection, _quote_indentifier("T2"), int("1000"))
    await adrop_table_purge(connection, _quote_indentifier("T2"))

    # 9. New Table - T3
    #     Dimension - 100 passed as a variable
    # Expectation:table is created
    val = 100
    await _acreate_table(connection, _quote_indentifier("T3"), val)
    await adrop_table_purge(connection, _quote_indentifier("T3"))

    '''# 10.
    # Expectation:ORA-00922: missing or invalid option
    val2 = """H
    ello"""
    with pytest.raises(oracledb.Error):
        await _acreate_table(connection, _quote_indentifier(val2), 545)
        await adrop_table_purge(connection, _quote_indentifier(val2))'''

    # 11. New Table - हिन्दी
    #     Dimension - 545
    # Expectation:table is created
    await _acreate_table(connection, _quote_indentifier('"हिन्दी"'), 545)
    await adrop_table_purge(connection, _quote_indentifier('"हिन्दी"'))

    # 12. <schema_name.table_name>
    # Expectation:failure - user does not exist
    with pytest.raises(oracledb.Error):
        await _acreate_table(connection, _quote_indentifier("U1.TB4"), 128)
        await adrop_table_purge(connection, _quote_indentifier("U1.TB4"))

    # 13.
    # Expectation:table is created
    await _acreate_table(connection, _quote_indentifier('"T5"'), 128)
    await adrop_table_purge(connection, _quote_indentifier('"T5"'))

    """# 14. Toggle Case
    # Expectation:table creation fails
    with pytest.raises(oracledb.Error):
        await _acreate_table(connection, _quote_indentifier("TaBlE"), 128)
        await adrop_table_purge(connection, _quote_indentifier("TaBlE"))"""

    # 15. table_name as empty_string
    # Expectation: ORA-00903: invalid table name
    with pytest.raises(ValueError):
        await _acreate_table(connection, _quote_indentifier(""), 128)
        await adrop_table_purge(connection, _quote_indentifier(""))
        await _acreate_table(connection, _quote_indentifier('""'), 128)
        await adrop_table_purge(connection, _quote_indentifier('""'))

    # 16. Arithmetic Operations in dimension parameter
    # Expectation:table is created
    n = 1
    await _acreate_table(connection, _quote_indentifier("T10"), n + 500)
    await adrop_table_purge(connection, _quote_indentifier("T10"))

    # 17. String Operations in table_name&dimension parameter
    # Expectation:table is created
    await _acreate_table(
        connection, _quote_indentifier("YaSh".replace("aS", "ok")), 500
    )
    await adrop_table_purge(connection, _quote_indentifier("YaSh".replace("aS", "ok")))


##################################
####### create_hnsw_index #######
##################################


def test_create_hnsw_index_test() -> None:
    try:
        connection = oracledb.connect(user=username, password=password, dsn=dsn)
    except Exception:
        sys.exit(1)
    # 1. Table_name - TB1
    #    New Index
    #    distance_strategy - DistanceStrategy.Dot_product
    # Expectation:Index created
    model1 = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-mpnet-base-v2"
    )
    vs = OracleVS(
        connection, model1, "TB1", DistanceStrategy.EUCLIDEAN_DISTANCE, backend="sql"
    )
    create_index(connection, vs)

    # 2. Creating same index again
    #    Table_name - TB1
    # Expectation:Nothing happens
    with pytest.raises(RuntimeError, match="such column list already indexed"):
        create_index(connection, vs)
        drop_index_if_exists(connection, "HNSW")
    drop_table_purge(connection, "TB1")

    # 3. Create index with following parameters:
    #    idx_name - hnsw_idx2
    #    idx_type - HNSW
    # Expectation:Index created
    vs = OracleVS(
        connection, model1, "TB2", DistanceStrategy.EUCLIDEAN_DISTANCE, backend="sql"
    )
    create_index(connection, vs, params={"idx_name": "hnsw_idx2", "idx_type": "HNSW"})
    drop_index_if_exists(connection, "hnsw_idx2")
    drop_table_purge(connection, "TB2")

    # 4. Table Name - TB1
    #    idx_name - "हिन्दी"
    #    idx_type - HNSW
    # Expectation:Index created
    vs = OracleVS(
        connection, model1, "TB3", DistanceStrategy.EUCLIDEAN_DISTANCE, backend="sql"
    )
    create_index(connection, vs, params={"idx_name": '"हिन्दी"', "idx_type": "HNSW"})
    drop_index_if_exists(connection, '"हिन्दी"')
    drop_table_purge(connection, "TB3")

    # 5. idx_name passed empty
    # Expectation:ORA-01741: illegal zero-length identifier
    with pytest.raises(ValueError):
        vs = OracleVS(
            connection,
            model1,
            "TB4",
            DistanceStrategy.EUCLIDEAN_DISTANCE,
            backend="sql",
        )
        create_index(connection, vs, params={"idx_name": '""', "idx_type": "HNSW"})
        drop_index_if_exists(connection, '""')
    drop_table_purge(connection, "TB4")

    # 6. idx_type left empty
    # Expectation: rejected by _validate_index_type (added in #215)
    vs = OracleVS(
        connection, model1, "TB5", DistanceStrategy.EUCLIDEAN_DISTANCE, backend="sql"
    )
    with pytest.raises(ValueError, match="idx_type must be HNSW"):
        create_index(connection, vs, params={"idx_name": "Hello", "idx_type": ""})
    drop_table_purge(connection, "TB5")

    # 7. efconstruction passed as parameter but not neighbours
    # Expectation:Index created
    vs = OracleVS(
        connection, model1, "TB7", DistanceStrategy.EUCLIDEAN_DISTANCE, backend="sql"
    )
    create_index(
        connection,
        vs,
        params={"idx_name": "idx11", "efConstruction": 100, "idx_type": "HNSW"},
    )
    drop_index_if_exists(connection, "idx11")
    drop_table_purge(connection, "TB7")

    # 8. efconstruction passed as parameter as well as neighbours
    # (for this idx_type parameter is also necessary)
    # Expectation:Index created
    vs = OracleVS(
        connection, model1, "TB8", DistanceStrategy.EUCLIDEAN_DISTANCE, backend="sql"
    )
    create_index(
        connection,
        vs,
        params={
            "idx_name": "idx11",
            "efConstruction": 100,
            "neighbors": 80,
            "idx_type": "HNSW",
        },
    )
    drop_index_if_exists(connection, "idx11")
    drop_table_purge(connection, "TB8")

    #  9. Limit of Values for(integer values):
    #     parallel
    #     efConstruction
    #     Neighbors
    #     Accuracy
    #     0<Accuracy<=100
    #     0<Neighbour<=2048
    #     0<efConstruction<=65535
    #     0<parallel<=255
    # Expectation:Index created
    drop_table_purge(connection, "TB9")
    vs = OracleVS(
        connection, model1, "TB9", DistanceStrategy.EUCLIDEAN_DISTANCE, backend="sql"
    )
    create_index(
        connection,
        vs,
        params={
            "idx_name": "idx11",
            "efConstruction": 65535,
            "neighbors": 2048,
            "idx_type": "HNSW",
            "parallel": 255,
        },
    )
    drop_index_if_exists(connection, "idx11")
    drop_table_purge(connection, "TB9")
    # index not created:
    with pytest.raises(ValueError):
        vs = OracleVS(
            connection,
            model1,
            "TB10",
            DistanceStrategy.EUCLIDEAN_DISTANCE,
            backend="sql",
        )
        create_index(
            connection,
            vs,
            params={
                "idx_name": "idx11",
                "efConstruction": 0,
                "neighbors": 2048,
                "idx_type": "HNSW",
                "parallel": 255,
            },
        )
        drop_index_if_exists(connection, "idx11")
    # index not created:
    with pytest.raises(ValueError):
        vs = OracleVS(
            connection,
            model1,
            "TB11",
            DistanceStrategy.EUCLIDEAN_DISTANCE,
            backend="sql",
        )
        create_index(
            connection,
            vs,
            params={
                "idx_name": "idx11",
                "efConstruction": 100,
                "neighbors": 0,
                "idx_type": "HNSW",
                "parallel": 255,
            },
        )
        drop_index_if_exists(connection, "idx11")
    # index not created
    with pytest.raises(ValueError):
        vs = OracleVS(
            connection,
            model1,
            "TB12",
            DistanceStrategy.EUCLIDEAN_DISTANCE,
            backend="sql",
        )
        create_index(
            connection,
            vs,
            params={
                "idx_name": "idx11",
                "efConstruction": 100,
                "neighbors": 100,
                "idx_type": "HNSW",
                "parallel": 0,
            },
        )
        drop_index_if_exists(connection, "idx11")
    # index not created
    with pytest.raises(ValueError):
        vs = OracleVS(
            connection,
            model1,
            "TB13",
            DistanceStrategy.EUCLIDEAN_DISTANCE,
            backend="sql",
        )
        create_index(
            connection,
            vs,
            params={
                "idx_name": "idx11",
                "efConstruction": 10,
                "neighbors": 100,
                "idx_type": "HNSW",
                "parallel": 10,
                "accuracy": 120,
            },
        )
        drop_index_if_exists(connection, "idx11")
    # With negative or out-of-bound values for all 4 of them, we get the same errors.
    # Expectation:Index not created
    with pytest.raises(ValueError):
        vs = OracleVS(
            connection,
            model1,
            "TB14",
            DistanceStrategy.EUCLIDEAN_DISTANCE,
            backend="sql",
        )
        create_index(
            connection,
            vs,
            params={
                "idx_name": "idx11",
                "efConstruction": 200,
                "neighbors": 100,
                "idx_type": "HNSW",
                "parallel": "hello",
                "accuracy": 10,
            },
        )
        drop_index_if_exists(connection, "idx11")
    drop_table_purge(connection, "TB10")
    drop_table_purge(connection, "TB11")
    drop_table_purge(connection, "TB12")
    drop_table_purge(connection, "TB13")
    drop_table_purge(connection, "TB14")

    # 10. Table_name as <schema_name.table_name>
    # Expectation:Index created
    vs = OracleVS(
        connection, model1, "TB15", DistanceStrategy.EUCLIDEAN_DISTANCE, backend="sql"
    )
    create_index(
        connection,
        vs,
        params={
            "idx_name": "idx11",
            "efConstruction": 200,
            "neighbors": 100,
            "idx_type": "HNSW",
            "parallel": 8,
            "accuracy": 10,
        },
    )
    drop_index_if_exists(connection, "idx11")
    drop_table_purge(connection, "TB15")

    # 11. index_name as <schema_name.index_name>
    # Expectation:U1 not present
    with pytest.raises(RuntimeError):
        vs = OracleVS(
            connection,
            model1,
            "U1.TB16",
            DistanceStrategy.EUCLIDEAN_DISTANCE,
            backend="sql",
        )
        create_index(
            connection,
            vs,
            params={
                "idx_name": "U1.idx11",
                "efConstruction": 200,
                "neighbors": 100,
                "idx_type": "HNSW",
                "parallel": 8,
                "accuracy": 10,
            },
        )
        drop_index_if_exists(connection, "U1.idx11")
        drop_table_purge(connection, "TB16")

    # 12. Index_name size >129
    # Expectation:Index not created
    with pytest.raises(RuntimeError):
        vs = OracleVS(
            connection,
            model1,
            "TB17",
            DistanceStrategy.EUCLIDEAN_DISTANCE,
            backend="sql",
        )
        create_index(connection, vs, params={"idx_name": "x" * 129, "idx_type": "HNSW"})
        drop_index_if_exists(connection, "x" * 129)
    drop_table_purge(connection, "TB17")

    # 13. Index_name size 128
    # Expectation:Index created
    vs = OracleVS(
        connection, model1, "TB18", DistanceStrategy.EUCLIDEAN_DISTANCE, backend="sql"
    )
    create_index(connection, vs, params={"idx_name": "x" * 128, "idx_type": "HNSW"})
    drop_index_if_exists(connection, "x" * 128)
    drop_table_purge(connection, "TB18")


@pytest.mark.asyncio
async def test_create_hnsw_index_test_async() -> None:
    try:
        connection = await oracledb.connect_async(
            user=username, password=password, dsn=dsn
        )
    except Exception:
        sys.exit(1)
    # 1. Table_name - TB1
    #    New Index
    #    distance_strategy - DistanceStrategy.Dot_product
    # Expectation:Index created
    model1 = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-mpnet-base-v2"
    )
    vs = await OracleVS.acreate(
        connection, model1, "TB1", DistanceStrategy.EUCLIDEAN_DISTANCE, backend="sql"
    )
    await acreate_index(connection, vs)

    # 2. Creating same index again
    #    Table_name - TB1
    # Expectation:Without index name, error happens
    with pytest.raises(RuntimeError, match="such column list already indexed"):
        await acreate_index(connection, vs)
        await adrop_index_if_exists(connection, "HNSW")
    await adrop_table_purge(connection, "TB1")

    # 3. Create index with following parameters:
    #    idx_name - hnsw_idx2
    #    idx_type - HNSW
    # Expectation:Index created
    vs = await OracleVS.acreate(
        connection, model1, "TB2", DistanceStrategy.EUCLIDEAN_DISTANCE, backend="sql"
    )
    await acreate_index(
        connection, vs, params={"idx_name": "hnsw_idx2", "idx_type": "HNSW"}
    )
    await adrop_index_if_exists(connection, "hnsw_idx2")
    await adrop_table_purge(connection, "TB2")

    # 4. Table Name - TB1
    #    idx_name - "हिन्दी"
    #    idx_type - HNSW
    # Expectation:Index created
    vs = await OracleVS.acreate(
        connection, model1, "TB3", DistanceStrategy.EUCLIDEAN_DISTANCE, backend="sql"
    )
    await acreate_index(
        connection, vs, params={"idx_name": '"हिन्दी"', "idx_type": "HNSW"}
    )
    await adrop_index_if_exists(connection, '"हिन्दी"')
    await adrop_table_purge(connection, "TB3")

    # 5. idx_name passed empty
    # Expectation:ORA-01741: illegal zero-length identifier
    with pytest.raises(ValueError):
        vs = await OracleVS.acreate(
            connection,
            model1,
            "TB4",
            DistanceStrategy.EUCLIDEAN_DISTANCE,
            backend="sql",
        )
        await acreate_index(
            connection, vs, params={"idx_name": '""', "idx_type": "HNSW"}
        )
        await adrop_index_if_exists(connection, '""')
    await adrop_table_purge(connection, "TB4")

    # 6. idx_type left empty
    vs = await OracleVS.acreate(
        connection, model1, "TB5", DistanceStrategy.EUCLIDEAN_DISTANCE, backend="sql"
    )
    with pytest.raises(ValueError, match="idx_type must be HNSW"):
        await acreate_index(
            connection, vs, params={"idx_name": "Hello", "idx_type": ""}
        )
    await adrop_table_purge(connection, "TB5")

    # 7. efconstruction passed as parameter but not neighbours
    # Expectation:Index created
    vs = await OracleVS.acreate(
        connection, model1, "TB7", DistanceStrategy.EUCLIDEAN_DISTANCE, backend="sql"
    )
    await acreate_index(
        connection,
        vs,
        params={"idx_name": "idx11", "efConstruction": 100, "idx_type": "HNSW"},
    )
    await adrop_index_if_exists(connection, "idx11")
    await adrop_table_purge(connection, "TB7")

    # 8. efconstruction passed as parameter as well as neighbours
    # (for this idx_type parameter is also necessary)
    # Expectation:Index created
    vs = await OracleVS.acreate(
        connection, model1, "TB8", DistanceStrategy.EUCLIDEAN_DISTANCE, backend="sql"
    )
    await acreate_index(
        connection,
        vs,
        params={
            "idx_name": "idx11",
            "efConstruction": 100,
            "neighbors": 80,
            "idx_type": "HNSW",
        },
    )
    await adrop_index_if_exists(connection, "idx11")
    await adrop_table_purge(connection, "TB8")

    #  9. Limit of Values for(integer values):
    #     parallel
    #     efConstruction
    #     Neighbors
    #     Accuracy
    #     0<Accuracy<=100
    #     0<Neighbour<=2048
    #     0<efConstruction<=65535
    #     0<parallel<=255
    # Expectation:Index created
    await adrop_table_purge(connection, "TB9")
    vs = await OracleVS.acreate(
        connection, model1, "TB9", DistanceStrategy.EUCLIDEAN_DISTANCE, backend="sql"
    )
    await acreate_index(
        connection,
        vs,
        params={
            "idx_name": "idx11",
            "efConstruction": 65535,
            "neighbors": 2048,
            "idx_type": "HNSW",
            "parallel": 255,
        },
    )
    await adrop_index_if_exists(connection, "idx11")
    await adrop_table_purge(connection, "TB9")
    # index not created:
    with pytest.raises(ValueError):
        vs = await OracleVS.acreate(
            connection,
            model1,
            "TB10",
            DistanceStrategy.EUCLIDEAN_DISTANCE,
            backend="sql",
        )
        await acreate_index(
            connection,
            vs,
            params={
                "idx_name": "idx11",
                "efConstruction": 0,
                "neighbors": 2048,
                "idx_type": "HNSW",
                "parallel": 255,
            },
        )
        await adrop_index_if_exists(connection, "idx11")

    # index not created:
    with pytest.raises(ValueError):
        vs = await OracleVS.acreate(
            connection,
            model1,
            "TB11",
            DistanceStrategy.EUCLIDEAN_DISTANCE,
            backend="sql",
        )
        await acreate_index(
            connection,
            vs,
            params={
                "idx_name": "idx11",
                "efConstruction": 100,
                "neighbors": 0,
                "idx_type": "HNSW",
                "parallel": 255,
            },
        )
        await adrop_index_if_exists(connection, "idx11")
    # index not created
    with pytest.raises(ValueError):
        vs = await OracleVS.acreate(
            connection,
            model1,
            "TB12",
            DistanceStrategy.EUCLIDEAN_DISTANCE,
            backend="sql",
        )
        await acreate_index(
            connection,
            vs,
            params={
                "idx_name": "idx11",
                "efConstruction": 100,
                "neighbors": 100,
                "idx_type": "HNSW",
                "parallel": 0,
            },
        )
        await adrop_index_if_exists(connection, "idx11")
    # index not created
    with pytest.raises(ValueError):
        vs = await OracleVS.acreate(
            connection,
            model1,
            "TB13",
            DistanceStrategy.EUCLIDEAN_DISTANCE,
            backend="sql",
        )
        await acreate_index(
            connection,
            vs,
            params={
                "idx_name": "idx11",
                "efConstruction": 10,
                "neighbors": 100,
                "idx_type": "HNSW",
                "parallel": 10,
                "accuracy": 120,
            },
        )
        await adrop_index_if_exists(connection, "idx11")
    # with negative values/out-of-bound values for all 4 of them, we get the same errors
    # Expectation:Index not created
    with pytest.raises(ValueError):
        vs = await OracleVS.acreate(
            connection,
            model1,
            "TB14",
            DistanceStrategy.EUCLIDEAN_DISTANCE,
            backend="sql",
        )
        await acreate_index(
            connection,
            vs,
            params={
                "idx_name": "idx11",
                "efConstruction": 200,
                "neighbors": 100,
                "idx_type": "HNSW",
                "parallel": "hello",
                "accuracy": 10,
            },
        )
        await adrop_index_if_exists(connection, "idx11")
    await adrop_table_purge(connection, "TB10")
    await adrop_table_purge(connection, "TB11")
    await adrop_table_purge(connection, "TB12")
    await adrop_table_purge(connection, "TB13")
    await adrop_table_purge(connection, "TB14")

    # 10. Table_name as <schema_name.table_name>
    # Expectation:Index created
    vs = await OracleVS.acreate(
        connection, model1, "TB15", DistanceStrategy.EUCLIDEAN_DISTANCE, backend="sql"
    )
    await acreate_index(
        connection,
        vs,
        params={
            "idx_name": "idx11",
            "efConstruction": 200,
            "neighbors": 100,
            "idx_type": "HNSW",
            "parallel": 8,
            "accuracy": 10,
        },
    )
    await adrop_index_if_exists(connection, "idx11")
    await adrop_table_purge(connection, "TB15")

    # 11. index_name as <schema_name.index_name>
    # Expectation:U1 not present
    with pytest.raises(RuntimeError):
        vs = await OracleVS.acreate(
            connection,
            model1,
            "U1.TB16",
            DistanceStrategy.EUCLIDEAN_DISTANCE,
            backend="sql",
        )
        await acreate_index(
            connection,
            vs,
            params={
                "idx_name": "U1.idx11",
                "efConstruction": 200,
                "neighbors": 100,
                "idx_type": "HNSW",
                "parallel": 8,
                "accuracy": 10,
            },
        )
        await adrop_index_if_exists(connection, "U1.idx11")
        await adrop_table_purge(connection, "TB16")

    # 12. Index_name size >129
    # Expectation:Index not created
    with pytest.raises(RuntimeError):
        vs = await OracleVS.acreate(
            connection,
            model1,
            "TB17",
            DistanceStrategy.EUCLIDEAN_DISTANCE,
            backend="sql",
        )
        await acreate_index(
            connection, vs, params={"idx_name": "x" * 129, "idx_type": "HNSW"}
        )
        await adrop_index_if_exists(connection, "x" * 129)
    await adrop_table_purge(connection, "TB17")

    # 13. Index_name size 128
    # Expectation:Index created
    vs = await OracleVS.acreate(
        connection, model1, "TB18", DistanceStrategy.EUCLIDEAN_DISTANCE, backend="sql"
    )
    await acreate_index(
        connection, vs, params={"idx_name": "x" * 128, "idx_type": "HNSW"}
    )
    await adrop_index_if_exists(connection, "x" * 128)
    await adrop_table_purge(connection, "TB18")


##################################
####### index_exists #############
##################################


def test_index_exists_test() -> None:
    try:
        connection = oracledb.connect(user=username, password=password, dsn=dsn)
    except Exception:
        sys.exit(1)
    model1 = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-mpnet-base-v2"
    )
    # 1. Existing Index:(all capital letters)
    # Expectation:true
    vs = OracleVS(
        connection, model1, "TB1", DistanceStrategy.EUCLIDEAN_DISTANCE, backend="sql"
    )
    create_index(connection, vs, params={"idx_name": "idx11", "idx_type": "HNSW"})
    assert not _index_exists(connection, _quote_indentifier("IDX11"))

    # 2. Existing Table:(all small letters)
    # Expectation:true
    assert _index_exists(connection, _quote_indentifier("idx11"))

    # 3. Non-Existing Index
    # Expectation:False
    assert not _index_exists(connection, _quote_indentifier("Hello"))

    """# 4. Invalid Index Name # Qutoted not invalid
    # Expectation:Error
    with pytest.raises(RuntimeError):
        _index_exists(connection, _quote_indentifier("123"))"""

    # 5. Empty String
    # Expectation:Error
    with pytest.raises(ValueError):
        _index_exists(connection, _quote_indentifier(""))

    """# 6. Special Character
    # Expectation:Error
    with pytest.raises(RuntimeError):
        _index_exists(connection, _quote_indentifier("##4"))"""

    """# 7. Index name length > 128
    # Expectation:Error
    with pytest.raises(oracledb.Error):
        _index_exists(connection, _quote_indentifier("x" * 129))"""

    # 8. <Schema_Name.Index_Name>
    # Expectation:true
    _index_exists(connection, _quote_indentifier("ONNXUSER.idx11"))

    # 9. Toggle Case (like iDx11)
    # Expectation:true
    assert not _index_exists(connection, _quote_indentifier("IdX11"))

    # 10. Index_Name→ "हिन्दी"
    # Expectation:true
    drop_index_if_exists(connection, _quote_indentifier("idx11"))
    create_index(
        connection,
        vs,
        params={"idx_name": _quote_indentifier('"हिन्दी"'), "idx_type": "HNSW"},
    )
    assert _index_exists(connection, _quote_indentifier('"हिन्दी"'))
    drop_table_purge(connection, _quote_indentifier("TB1"))


@pytest.mark.asyncio
async def test_index_exists_test_async() -> None:
    try:
        connection = await oracledb.connect_async(
            user=username, password=password, dsn=dsn
        )
    except Exception:
        sys.exit(1)
    model1 = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-mpnet-base-v2"
    )
    # 1. Existing Index:(all capital letters)
    # Expectation:true
    vs = await OracleVS.acreate(
        connection, model1, "TB1", DistanceStrategy.EUCLIDEAN_DISTANCE, backend="sql"
    )
    await acreate_index(
        connection, vs, params={"idx_name": "idx11", "idx_type": "HNSW"}
    )
    assert not await _aindex_exists(connection, _quote_indentifier("IDX11"))

    # 2. Existing Table:(all small letters)
    # Expectation:true
    assert await _aindex_exists(connection, _quote_indentifier("idx11"))

    # 3. Non-Existing Index
    # Expectation:False
    assert not await _aindex_exists(connection, _quote_indentifier("Hello"))

    """# 4. Invalid Index Name # Qutoted not invalid
    # Expectation:Error
    with pytest.raises(RuntimeError):
        await _aindex_exists(connection, _quote_indentifier("123"))"""

    # 5. Empty String
    # Expectation:Error
    with pytest.raises(ValueError):
        await _aindex_exists(connection, _quote_indentifier(""))
    with pytest.raises(ValueError):
        await _aindex_exists(connection, _quote_indentifier(""))

    """# 6. Special Character
    # Expectation:Error
    with pytest.raises(RuntimeError):
        await _aindex_exists(connection, _quote_indentifier("##4"))"""

    """# 7. Index name length > 128
    # Expectation:Error
    with pytest.raises(oracledb.Error):
        await _aindex_exists(connection, _quote_indentifier("x" * 129))"""

    # 8. <Schema_Name.Index_Name>
    # Expectation:true
    await _aindex_exists(connection, _quote_indentifier("ONNXUSER.idx11"))

    # 9. Toggle Case (like iDx11)
    # Expectation:true
    assert not await _aindex_exists(connection, _quote_indentifier("IdX11"))

    # 10. Index_Name→ "हिन्दी"
    # Expectation:true
    await adrop_index_if_exists(connection, _quote_indentifier("idx11"))
    await acreate_index(
        connection,
        vs,
        params={"idx_name": _quote_indentifier('"हिन्दी"'), "idx_type": "HNSW"},
    )
    assert await _aindex_exists(connection, _quote_indentifier('"हिन्दी"'))
    await adrop_table_purge(connection, _quote_indentifier("TB1"))


##################################
####### add_texts ################
##################################


def test_add_texts_test() -> None:
    try:
        connection = oracledb.connect(user=username, password=password, dsn=dsn)
    except Exception:
        sys.exit(1)
    # 1. Add 2 records to table
    # Expectation:Successful
    texts = ["Rohan", "Shailendra"]
    metadata = [
        {"id": "100", "link": "Document Example Test 1"},
        {"id": "101", "link": "Document Example Test 2"},
    ]
    model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vs_obj = OracleVS(connection, model, "TB1", DistanceStrategy.EUCLIDEAN_DISTANCE)
    vs_obj.add_texts(texts, metadata)
    drop_table_purge(connection, "TB1")

    # 2. Add record but metadata is not there
    # Expectation:An exception occurred :: Either specify an 'ids' list or
    # 'metadatas' with an 'id' attribute for each element.
    model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vs_obj = OracleVS(connection, model, "TB2", DistanceStrategy.EUCLIDEAN_DISTANCE)
    texts2 = ["Sri Ram", "Krishna"]
    vs_obj.add_texts(texts2)
    drop_table_purge(connection, "TB2")

    # 3. Add record with ids option
    #    ids are passed as string
    #    ids are passed as empty string
    #    ids are passed as multi-line string
    #    ids are passed as "<string>"
    # Expectations:
    # Successful
    # Successful
    # Successful
    # Successful

    vs_obj = OracleVS(connection, model, "TB4", DistanceStrategy.EUCLIDEAN_DISTANCE)
    ids3 = ["114", "124"]
    vs_obj.add_texts(texts2, ids=ids3)
    drop_table_purge(connection, "TB4")

    vs_obj = OracleVS(connection, model, "TB5", DistanceStrategy.EUCLIDEAN_DISTANCE)
    ids4 = ["", "134"]
    vs_obj.add_texts(texts2, ids=ids4)
    drop_table_purge(connection, "TB5")

    vs_obj = OracleVS(connection, model, "TB6", DistanceStrategy.EUCLIDEAN_DISTANCE)
    ids5 = [
        """Good afternoon
    my friends""",
        "India",
    ]
    vs_obj.add_texts(texts2, ids=ids5)
    drop_table_purge(connection, "TB6")

    vs_obj = OracleVS(connection, model, "TB7", DistanceStrategy.EUCLIDEAN_DISTANCE)
    ids6 = ['"Good afternoon"', '"India"']
    vs_obj.add_texts(texts2, ids=ids6)
    assert len(vs_obj.add_texts(texts2, ids=ids6)) == 0
    drop_table_purge(connection, "TB7")

    # 4. Add records with ids and metadatas
    # Expectation:Successful
    vs_obj = OracleVS(connection, model, "TB8", DistanceStrategy.EUCLIDEAN_DISTANCE)
    texts3 = ["Sri Ram 6", "Krishna 6"]
    ids7 = ["1", "2"]
    metadata = [
        {"id": "102", "link": "Document Example", "stream": "Science"},
        {"id": "104", "link": "Document Example 45"},
    ]
    vs_obj.add_texts(texts3, metadata, ids=ids7)
    drop_table_purge(connection, "TB8")

    # 5. Add 10000 records
    # Expectation:Successful
    vs_obj = OracleVS(connection, model, "TB9", DistanceStrategy.EUCLIDEAN_DISTANCE)
    texts4 = ["Sri Ram{0}".format(i) for i in range(1, 10000)]
    ids8 = ["Hello{0}".format(i) for i in range(1, 10000)]
    vs_obj.add_texts(texts4, ids=ids8)
    drop_table_purge(connection, "TB9")

    # PyTorch + Transformers are NOT thread-safe during initialization
    # Initialize model creation OUTSIDE threads
    model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    # Create ONCE (this creates the table safely)
    vs_obj = OracleVS(connection, model, "TB10", DistanceStrategy.EUCLIDEAN_DISTANCE)

    # 6. Add 2 different record concurrently
    # Expectation:Successful
    def add(val: str) -> None:
        texts5 = [val]
        ids9 = texts5
        vs_obj.add_texts(texts5, ids=ids9)

    thread_1 = threading.Thread(target=add, args=("Sri Ram",))
    thread_2 = threading.Thread(target=add, args=("Sri Krishna",))
    thread_1.start()
    thread_2.start()
    thread_1.join()
    thread_2.join()
    drop_table_purge(connection, "TB10")

    # 8. create object with table name of type <schema_name.table_name>
    # Expectation:U1 does not exist
    with pytest.raises(RuntimeError):
        vs_obj = OracleVS(connection, model, "U1.TB14", DistanceStrategy.DOT_PRODUCT)
        for i in range(1, 10):
            texts7 = ["Yash{0}".format(i)]
            ids13 = ["1234{0}".format(i)]
            vs_obj.add_texts(texts7, ids=ids13)
        drop_table_purge(connection, "TB14")


@pytest.mark.asyncio
async def test_add_texts_test_async() -> None:
    try:
        connection = await oracledb.connect_async(
            user=username, password=password, dsn=dsn
        )
    except Exception:
        sys.exit(1)
    # 1. Add 2 records to table
    # Expectation:Successful
    texts = ["Rohan", "Shailendra"]
    metadata = [
        {"id": "100", "link": "Document Example Test 1"},
        {"id": "101", "link": "Document Example Test 2"},
    ]
    model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vs_obj = await OracleVS.acreate(
        connection, model, "TB1", DistanceStrategy.EUCLIDEAN_DISTANCE
    )
    await vs_obj.aadd_texts(texts, metadata)
    await adrop_table_purge(connection, "TB1")

    # 2. Add record but metadata is not there
    # Expectation:An exception occurred :: Either specify an 'ids' list or
    # 'metadatas' with an 'id' attribute for each element.
    model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vs_obj = await OracleVS.acreate(
        connection, model, "TB2", DistanceStrategy.EUCLIDEAN_DISTANCE
    )
    texts2 = ["Sri Ram", "Krishna"]
    await vs_obj.aadd_texts(texts2)
    await adrop_table_purge(connection, "TB2")

    # 3. Add record with ids option
    #    ids are passed as string
    #    ids are passed as empty string
    #    ids are passed as multi-line string
    #    ids are passed as "<string>"
    # Expectations:
    # Successful
    # Successful
    # Successful
    # Successful

    vs_obj = await OracleVS.acreate(
        connection, model, "TB4", DistanceStrategy.EUCLIDEAN_DISTANCE
    )
    ids3 = ["114", "124"]
    await vs_obj.aadd_texts(texts2, ids=ids3)
    await adrop_table_purge(connection, "TB4")

    vs_obj = await OracleVS.acreate(
        connection, model, "TB5", DistanceStrategy.EUCLIDEAN_DISTANCE
    )
    ids4 = ["", "134"]
    await vs_obj.aadd_texts(texts2, ids=ids4)
    await adrop_table_purge(connection, "TB5")

    vs_obj = await OracleVS.acreate(
        connection, model, "TB6", DistanceStrategy.EUCLIDEAN_DISTANCE
    )
    ids5 = [
        """Good afternoon
    my friends""",
        "India",
    ]
    await vs_obj.aadd_texts(texts2, ids=ids5)
    await adrop_table_purge(connection, "TB6")

    vs_obj = await OracleVS.acreate(
        connection, model, "TB7", DistanceStrategy.EUCLIDEAN_DISTANCE
    )
    ids6 = ['"Good afternoon"', '"India"']
    await vs_obj.aadd_texts(texts2, ids=ids6)
    assert len(await vs_obj.aadd_texts(texts2, ids=ids6)) == 0
    await adrop_table_purge(connection, "TB7")

    # 4. Add records with ids and metadatas
    # Expectation:Successful
    vs_obj = await OracleVS.acreate(
        connection, model, "TB8", DistanceStrategy.EUCLIDEAN_DISTANCE
    )
    texts3 = ["Sri Ram 6", "Krishna 6"]
    ids7 = ["1", "2"]
    metadata = [
        {"id": "102", "link": "Document Example", "stream": "Science"},
        {"id": "104", "link": "Document Example 45"},
    ]
    await vs_obj.aadd_texts(texts3, metadata, ids=ids7)
    await adrop_table_purge(connection, "TB8")

    # 5. Add 10000 records
    # Expectation:Successful
    vs_obj = await OracleVS.acreate(
        connection, model, "TB9", DistanceStrategy.EUCLIDEAN_DISTANCE
    )
    texts4 = ["Sri Ram{0}".format(i) for i in range(1, 10000)]
    ids8 = ["Hello{0}".format(i) for i in range(1, 10000)]
    await vs_obj.aadd_texts(texts4, ids=ids8)
    await adrop_table_purge(connection, "TB9")

    # 6. Add 2 different record concurrently
    # Expectation:Successful
    await OracleVS.acreate(
        connection, model, "TB10", DistanceStrategy.EUCLIDEAN_DISTANCE
    )

    async def add(val: str) -> None:
        model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        vs_obj = await OracleVS.acreate(
            connection, model, "TB10", DistanceStrategy.EUCLIDEAN_DISTANCE
        )
        texts5 = [val]
        ids9 = texts5
        await vs_obj.aadd_texts(texts5, ids=ids9)

    task_1 = asyncio.create_task(add("Sri Ram"))
    task_2 = asyncio.create_task(add("Sri Krishna"))

    await asyncio.gather(task_1, task_2)
    await adrop_table_purge(connection, "TB10")

    # 8. create object with table name of type <schema_name.table_name>
    # Expectation:U1 does not exist
    with pytest.raises(RuntimeError):
        vs_obj = await OracleVS.acreate(
            connection, model, "U1.TB14", DistanceStrategy.DOT_PRODUCT
        )
        for i in range(1, 10):
            texts7 = ["Yash{0}".format(i)]
            ids13 = ["1234{0}".format(i)]
            await vs_obj.aadd_texts(texts7, ids=ids13)
        await adrop_table_purge(connection, "TB14")


@pytest.mark.parametrize(
    "case",
    [
        "metadata-only",
        "no-metadata",
        "ids",
        "multiline-ids",
        "quoted-ids",
        "metadata-with-ids",
    ],
)
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.asyncio
async def test_add_texts_portable(
    oracle_backend, legacy_embeddings, asynchronous, case
):
    texts = ["Rohan", "Shailendra"]
    metadata = None
    ids = None
    if case == "metadata-only":
        metadata = [
            {"id": "100", "link": "Document Example Test 1"},
            {"id": "101", "link": "Document Example Test 2"},
        ]
    elif case == "ids":
        ids = ["114", "124"]
    elif case == "multiline-ids":
        ids = ["Good afternoon\n    my friends", "India"]
    elif case == "quoted-ids":
        ids = ['"Good afternoon"', '"India"']
    elif case == "metadata-with-ids":
        texts = ["Sri Ram 6", "Krishna 6"]
        ids = ["1", "2"]
        metadata = [
            {"id": "102", "link": "Document Example", "stream": "Science"},
            {"id": "104", "link": "Document Example 45"},
        ]
    expected_metadata = (
        [dict(value) for value in metadata] if metadata is not None else None
    )

    async def check(store):
        kwargs = dict(
            client=store.client,
            table_name=store.table_name,
            backend=store.backend,
            mutate_on_duplicate=True,
            distance_strategy=store.distance_strategy,
            ids=ids,
        )
        factory = (
            await OracleVS.afrom_texts(texts, legacy_embeddings, metadata, **kwargs)
            if asynchronous
            else OracleVS.from_texts(texts, legacy_embeddings, metadata, **kwargs)
        )
        assert factory.backend == store.backend
        if ids is not None:
            documents = (
                await factory.aget_by_ids(ids)
                if asynchronous
                else factory.get_by_ids(ids)
            )
        else:
            documents = (
                await factory.asimilarity_search(texts[0], k=2)
                if asynchronous
                else factory.similarity_search(texts[0], k=2)
            )
        assert len(documents) == len(texts)
        by_text = {doc.page_content: doc for doc in documents}
        assert set(by_text) == set(texts)
        if ids is not None:
            assert {doc.id for doc in documents} == set(ids)
        if expected_metadata is not None:
            for text, expected in zip(texts, expected_metadata):
                for key, value in expected.items():
                    assert by_text[text].metadata[key] == value
        target_id = by_text[texts[0]].id
        assert target_id is not None
        updated_text = "Updated " + texts[0]
        returned = (
            await factory.aadd_texts(
                [updated_text], [{"updated": True}], ids=[target_id]
            )
            if asynchronous
            else factory.add_texts([updated_text], [{"updated": True}], ids=[target_id])
        )
        assert returned == [target_id]
        updated = (
            await factory.aget_by_ids([target_id])
            if asynchronous
            else factory.get_by_ids([target_id])
        )
        assert len(updated) == 1
        assert updated[0].page_content == updated_text
        assert updated[0].metadata["updated"] is True

    if asynchronous:
        async with async_backend_store(oracle_backend, legacy_embeddings) as store:
            await check(store)
    else:
        with backend_store(oracle_backend, legacy_embeddings) as store:
            await check(store)


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.asyncio
async def test_add_texts_large_batch_portable(oracle_backend, asynchronous):
    # Keep the original 9,999-row workload; avoid coupling storage volume to
    # transformer inference time by using the shared deterministic embedder.
    texts = [f"Sri Ram{i}" for i in range(1, 10000)]
    ids = [f"Hello{i}" for i in range(1, 10000)]

    async def check(store):
        returned = (
            await store.aadd_texts(texts, ids=ids)
            if asynchronous
            else store.add_texts(texts, ids=ids)
        )
        assert returned == ids
        documents = (
            await store.aget_by_ids(ids) if asynchronous else store.get_by_ids(ids)
        )
        assert len(documents) == len(texts)
        assert {doc.id: doc.page_content for doc in documents} == dict(zip(ids, texts))

    if asynchronous:
        async with async_backend_store(oracle_backend, ContractEmbeddings()) as store:
            await check(store)
    else:
        with backend_store(oracle_backend, ContractEmbeddings()) as store:
            await check(store)


##################################
####### embed_documents(text) ####
##################################
def test_embed_documents_test(oracle_backend, legacy_embeddings) -> None:
    with backend_store(oracle_backend, legacy_embeddings) as store:
        vectors = store._embed_documents(["hello", "yash"])
        assert len(vectors) == 2
        assert len(vectors[0]) > 0
        assert len(vectors[0]) == len(vectors[1])


@pytest.mark.asyncio
async def test_embed_documents_test_async(oracle_backend, legacy_embeddings) -> None:
    async with async_backend_store(oracle_backend, legacy_embeddings) as store:
        vectors = await store._aembed_documents(["hello", "yash"])
        assert len(vectors) == 2
        assert len(vectors[0]) > 0
        assert len(vectors[0]) == len(vectors[1])


##################################
####### embed_query(text) ########
##################################
def test_embed_query_test(oracle_backend, legacy_embeddings) -> None:
    with backend_store(oracle_backend, legacy_embeddings) as store:
        vector = store._embed_query("Sri Ram")
        empty_query_vector = store._embed_query("")
        assert len(vector) > 0
        assert len(empty_query_vector) == len(vector)


@pytest.mark.asyncio
async def test_embed_query_test_async(oracle_backend, legacy_embeddings) -> None:
    async with async_backend_store(oracle_backend, legacy_embeddings) as store:
        vector = await store._aembed_query("Sri Ram")
        empty_query_vector = await store._aembed_query("")
        assert len(vector) > 0
        assert len(empty_query_vector) == len(vector)


##################################
####### create_index #############
##################################
def test_create_index_test() -> None:
    try:
        connection = oracledb.connect(user=username, password=password, dsn=dsn)
    except Exception:
        sys.exit(1)
    # 1. No optional parameters passed
    # Expectation:Successful
    model1 = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-mpnet-base-v2"
    )
    vs = OracleVS(
        connection, model1, "TB1", DistanceStrategy.EUCLIDEAN_DISTANCE, backend="sql"
    )
    create_index(connection, vs)
    drop_index_if_exists(connection, "HNSW")
    drop_table_purge(connection, "TB1")

    # 2. ivf index
    # Expectation:Successful
    vs = OracleVS(
        connection, model1, "TB2", DistanceStrategy.EUCLIDEAN_DISTANCE, backend="sql"
    )
    create_index(connection, vs, {"idx_type": "IVF", "idx_name": "IVF"})
    drop_index_if_exists(connection, "IVF")
    drop_table_purge(connection, "TB2")

    # 3. ivf index with neighbour_part passed as parameter
    # Expectation:Successful
    vs = OracleVS(
        connection, model1, "TB3", DistanceStrategy.EUCLIDEAN_DISTANCE, backend="sql"
    )
    create_index(connection, vs, {"idx_type": "IVF", "neighbor_part": 10})
    drop_index_if_exists(connection, "IVF")
    drop_table_purge(connection, "TB3")

    # 4. ivf index with neighbour_part and accuracy passed as parameter
    # Expectation:Successful
    vs = OracleVS(
        connection, model1, "TB4", DistanceStrategy.EUCLIDEAN_DISTANCE, backend="sql"
    )
    create_index(
        connection, vs, {"idx_type": "IVF", "neighbor_part": 10, "accuracy": 90}
    )
    drop_index_if_exists(connection, "IVF")
    drop_table_purge(connection, "TB4")

    # 5. ivf index with neighbour_part and parallel passed as parameter
    # Expectation:Successful
    vs = OracleVS(
        connection, model1, "TB5", DistanceStrategy.EUCLIDEAN_DISTANCE, backend="sql"
    )
    create_index(
        connection, vs, {"idx_type": "IVF", "neighbor_part": 10, "parallel": 90}
    )
    drop_index_if_exists(connection, "IVF")
    drop_table_purge(connection, "TB5")

    # 6. ivf index and then perform dml(insert)
    # Expectation:Successful
    vs = OracleVS(
        connection, model1, "TB6", DistanceStrategy.EUCLIDEAN_DISTANCE, backend="sql"
    )
    create_index(connection, vs, {"idx_type": "IVF", "idx_name": "IVF"})
    texts = ["Sri Ram", "Krishna"]
    vs.add_texts(texts)
    # perform delete
    vs.delete(["hello"])
    drop_index_if_exists(connection, "IVF")
    drop_table_purge(connection, "TB6")

    # 7. ivf index with neighbour_part,parallel and accuracy passed as parameter
    # Expectation:Successful
    vs = OracleVS(
        connection, model1, "TB7", DistanceStrategy.EUCLIDEAN_DISTANCE, backend="sql"
    )
    create_index(
        connection,
        vs,
        {"idx_type": "IVF", "neighbor_part": 10, "parallel": 90, "accuracy": 99},
    )
    drop_index_if_exists(connection, "IVF")
    drop_table_purge(connection, "TB7")


@pytest.mark.asyncio
async def test_create_index_test_async() -> None:
    try:
        connection = await oracledb.connect_async(
            user=username, password=password, dsn=dsn
        )
    except Exception:
        sys.exit(1)
    # 1. No optional parameters passed
    # Expectation:Successful
    model1 = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-mpnet-base-v2"
    )
    vs = await OracleVS.acreate(
        connection, model1, "TB1", DistanceStrategy.EUCLIDEAN_DISTANCE, backend="sql"
    )
    await acreate_index(connection, vs)
    await adrop_index_if_exists(connection, "HNSW")
    await adrop_table_purge(connection, "TB1")

    # 2. ivf index
    # Expectation:Successful
    vs = await OracleVS.acreate(
        connection, model1, "TB2", DistanceStrategy.EUCLIDEAN_DISTANCE, backend="sql"
    )
    await acreate_index(connection, vs, {"idx_type": "IVF", "idx_name": "IVF"})
    await adrop_index_if_exists(connection, "IVF")
    await adrop_table_purge(connection, "TB2")

    # 3. ivf index with neighbour_part passed as parameter
    # Expectation:Successful
    vs = await OracleVS.acreate(
        connection, model1, "TB3", DistanceStrategy.EUCLIDEAN_DISTANCE, backend="sql"
    )
    await acreate_index(connection, vs, {"idx_type": "IVF", "neighbor_part": 10})
    await adrop_index_if_exists(connection, "IVF")
    await adrop_table_purge(connection, "TB3")

    # 4. ivf index with neighbour_part and accuracy passed as parameter
    # Expectation:Successful
    vs = await OracleVS.acreate(
        connection, model1, "TB4", DistanceStrategy.EUCLIDEAN_DISTANCE, backend="sql"
    )
    await acreate_index(
        connection, vs, {"idx_type": "IVF", "neighbor_part": 10, "accuracy": 90}
    )
    await adrop_index_if_exists(connection, "IVF")
    await adrop_table_purge(connection, "TB4")

    # 5. ivf index with neighbour_part and parallel passed as parameter
    # Expectation:Successful
    vs = await OracleVS.acreate(
        connection, model1, "TB5", DistanceStrategy.EUCLIDEAN_DISTANCE, backend="sql"
    )
    await acreate_index(
        connection, vs, {"idx_type": "IVF", "neighbor_part": 10, "parallel": 90}
    )
    await adrop_index_if_exists(connection, "IVF")
    await adrop_table_purge(connection, "TB5")

    # 6. ivf index and then perform dml(insert)
    # Expectation:Successful
    vs = await OracleVS.acreate(
        connection, model1, "TB6", DistanceStrategy.EUCLIDEAN_DISTANCE, backend="sql"
    )
    await acreate_index(connection, vs, {"idx_type": "IVF", "idx_name": "IVF"})
    texts = ["Sri Ram", "Krishna"]
    await vs.aadd_texts(texts)
    # perform delete
    await vs.adelete(["hello"])
    await adrop_index_if_exists(connection, "IVF")
    await adrop_table_purge(connection, "TB6")

    # 7. ivf index with neighbour_part,parallel and accuracy passed as parameter
    # Expectation:Successful
    vs = await OracleVS.acreate(
        connection, model1, "TB7", DistanceStrategy.EUCLIDEAN_DISTANCE, backend="sql"
    )
    await acreate_index(
        connection,
        vs,
        {"idx_type": "IVF", "neighbor_part": 10, "parallel": 90, "accuracy": 99},
    )
    await adrop_index_if_exists(connection, "IVF")
    await adrop_table_purge(connection, "TB7")


##################################
####### perform_search ###########
##################################
@pytest.mark.parametrize("distance_strategy,index_type", SEARCH_INDEX_CASES)
def test_perform_search_test(
    oracle_backend, search_embeddings, distance_strategy, index_type
) -> None:
    with backend_store(
        oracle_backend, search_embeddings, distance_strategy=distance_strategy
    ) as store:
        texts = ["Yash", "Varanasi", "Yashaswi", "Mumbai", "BengaluruYash"]
        metadatas = [{"id": value} for value in ["hello", "105", "106", "yash", "108"]]
        assert len(store.add_texts(texts, metadatas)) == len(texts)
        create_index(
            store.client,
            store,
            {"idx_type": index_type, "idx_name": "IDX_" + store.table_name.strip('"')},
        )
        query = "YashB"
        filter_ = {"$or": [{"id": "106"}, {"id": "108"}, {"id": "yash"}]}
        assert len(store.similarity_search(query, 2)) == 2
        hits = store.similarity_search(query, 2, filter=filter_)
        assert 0 < len(hits) <= 2
        assert {doc.metadata["id"] for doc in hits} <= {"106", "108", "yash"}
        scored = store.similarity_search_with_score(query, 2)
        assert len(scored) == 2
        assert all(isinstance(score, float) for _, score in scored)
        assert scored[0][1] <= scored[1][1]
        scored = store.similarity_search_with_score(query, 2, filter=filter_)
        assert 0 < len(scored) <= 2
        assert {doc.metadata["id"] for doc, _ in scored} <= {"106", "108", "yash"}
        assert (
            len(
                store.max_marginal_relevance_search(
                    query, 2, fetch_k=20, lambda_mult=0.5
                )
            )
            == 2
        )
        hits = store.max_marginal_relevance_search(
            query, 2, fetch_k=20, lambda_mult=0.5, filter=filter_
        )
        assert 0 < len(hits) <= 2
        assert {doc.metadata["id"] for doc in hits} <= {"106", "108", "yash"}


@pytest.mark.parametrize("distance_strategy,index_type", SEARCH_INDEX_CASES)
@pytest.mark.asyncio
async def test_perform_search_test_async(
    oracle_backend, search_embeddings, distance_strategy, index_type
) -> None:
    async with async_backend_store(
        oracle_backend, search_embeddings, distance_strategy=distance_strategy
    ) as store:
        texts = ["Yash", "Varanasi", "Yashaswi", "Mumbai", "BengaluruYash"]
        metadatas = [{"id": value} for value in ["hello", "105", "106", "yash", "108"]]
        assert len(await store.aadd_texts(texts, metadatas)) == len(texts)
        await acreate_index(
            store.client,
            store,
            {"idx_type": index_type, "idx_name": "IDX_" + store.table_name.strip('"')},
        )
        query = "YashB"
        filter_ = {"$or": [{"id": "106"}, {"id": "108"}, {"id": "yash"}]}
        assert len(await store.asimilarity_search(query, 2)) == 2
        hits = await store.asimilarity_search(query, 2, filter=filter_)
        assert 0 < len(hits) <= 2
        assert {doc.metadata["id"] for doc in hits} <= {"106", "108", "yash"}
        scored = await store.asimilarity_search_with_score(query, 2)
        assert len(scored) == 2
        assert all(isinstance(score, float) for _, score in scored)
        assert scored[0][1] <= scored[1][1]
        scored = await store.asimilarity_search_with_score(query, 2, filter=filter_)
        assert 0 < len(scored) <= 2
        assert {doc.metadata["id"] for doc, _ in scored} <= {"106", "108", "yash"}
        assert (
            len(
                await store.amax_marginal_relevance_search(
                    query, 2, fetch_k=20, lambda_mult=0.5
                )
            )
            == 2
        )
        hits = await store.amax_marginal_relevance_search(
            query, 2, fetch_k=20, lambda_mult=0.5, filter=filter_
        )
        assert 0 < len(hits) <= 2
        assert {doc.metadata["id"] for doc in hits} <= {"106", "108", "yash"}


##################################
##### perform_filter_search ######
##################################

FILTERED_FUNCTIONS = [
    "similarity_search",
    "similarity_search_by_vector",
    "similarity_search_with_score",
    "similarity_search_by_vector_with_relevance_scores",
    "similarity_search_by_vector_returning_embeddings",
    "max_marginal_relevance_search_with_score_by_vector",
    "max_marginal_relevance_search_by_vector",
    "max_marginal_relevance_search",
]


@pytest.mark.parametrize("distance_strategy,index_type", SEARCH_INDEX_CASES)
def test_db_filter_test(
    oracle_backend, search_embeddings, distance_strategy, index_type
) -> None:
    with backend_store(
        oracle_backend, search_embeddings, distance_strategy=distance_strategy
    ) as store:
        texts = ["Strawberry", "Banana", "Blueberry", "Grape", "Watermelon"]
        metadatas = [
            {"id": value, "order": order}
            for order, value in enumerate(["st", "ba", "bl", "gr", "wa"], start=1)
        ]
        assert len(store.add_texts(texts, metadatas)) == len(texts)
        create_index(
            store.client,
            store,
            {"idx_type": index_type, "idx_name": "IDX_" + store.table_name.strip('"')},
        )
        query = "Strawberry"
        filter_ = {"id": {"$eq": "bl"}}
        nested = {
            "$or": [{"id": "ba"}, {"$and": [{"order": {"$lte": 4}}, {"id": "st"}]}]
        }
        for filtered_function in FILTERED_FUNCTIONS:
            method = getattr(store, filtered_function)
            query_value = (
                search_embeddings.embed_query(query)
                if "_by_vector" in filtered_function
                else query
            )
            result = method(query_value, k=1)
            assert len(result) == 1
            document = result[0][0] if isinstance(result[0], tuple) else result[0]
            assert document.metadata["id"] == "st"
            for keyword in ("filter", "db_filter"):
                result = method(query_value, k=5, **{keyword: filter_})
                assert len(result) == 1
                document = result[0][0] if isinstance(result[0], tuple) else result[0]
                assert document.metadata["id"] == "bl"
            result = method(query_value, k=5, filter=nested)
            assert len(result) == 2
            documents = [
                item[0] if isinstance(item, tuple) else item for item in result
            ]
            assert {doc.metadata["id"] for doc in documents} == {"ba", "st"}
            assert documents[0].metadata["id"] == "st"
        for invalid in (
            {"_xor": [{"order": {"$lte": 4}}, {"id": "st"}]},
            {"$or": [{"order": {"$xeq": 4}}, {"id": "st"}]},
        ):
            with pytest.raises(ValueError):
                store.similarity_search(query, 1, filter=invalid)


@pytest.mark.parametrize("distance_strategy,index_type", SEARCH_INDEX_CASES)
@pytest.mark.asyncio
async def test_db_filter_test_async(
    oracle_backend, search_embeddings, distance_strategy, index_type
) -> None:
    async with async_backend_store(
        oracle_backend, search_embeddings, distance_strategy=distance_strategy
    ) as store:
        texts = ["Strawberry", "Banana", "Blueberry", "Grape", "Watermelon"]
        metadatas = [
            {"id": value, "order": order}
            for order, value in enumerate(["st", "ba", "bl", "gr", "wa"], start=1)
        ]
        assert len(await store.aadd_texts(texts, metadatas)) == len(texts)
        await acreate_index(
            store.client,
            store,
            {"idx_type": index_type, "idx_name": "IDX_" + store.table_name.strip('"')},
        )
        query = "Strawberry"
        filter_ = {"id": {"$eq": "bl"}}
        nested = {
            "$or": [{"id": "ba"}, {"$and": [{"order": {"$lte": 4}}, {"id": "st"}]}]
        }
        for filtered_function in FILTERED_FUNCTIONS:
            method = getattr(store, "a" + filtered_function)
            query_value = (
                search_embeddings.embed_query(query)
                if "_by_vector" in filtered_function
                else query
            )
            result = await method(query_value, k=1)
            assert len(result) == 1
            document = result[0][0] if isinstance(result[0], tuple) else result[0]
            assert document.metadata["id"] == "st"
            for keyword in ("filter", "db_filter"):
                result = await method(query_value, k=5, **{keyword: filter_})
                assert len(result) == 1
                document = result[0][0] if isinstance(result[0], tuple) else result[0]
                assert document.metadata["id"] == "bl"
            result = await method(query_value, k=5, filter=nested)
            assert len(result) == 2
            documents = [
                item[0] if isinstance(item, tuple) else item for item in result
            ]
            assert {doc.metadata["id"] for doc in documents} == {"ba", "st"}
            assert documents[0].metadata["id"] == "st"
        for invalid in (
            {"_xor": [{"order": {"$lte": 4}}, {"id": "st"}]},
            {"$or": [{"order": {"$xeq": 4}}, {"id": "st"}]},
        ):
            with pytest.raises(ValueError):
                await store.asimilarity_search(query, 1, filter=invalid)


##################################
##### test_pool_connections ######
##################################


@pytest.mark.asyncio
async def test_add_texts_pool_test() -> None:
    POOLS_MAX = 4

    try:
        connection = oracledb.create_pool(
            user=username, password=password, dsn=dsn, min=1, max=POOLS_MAX, increment=1
        )
    except Exception:
        sys.exit(1)

    # 1. Add different records concurrently
    # Expectation:Successful
    async def add(order: int) -> None:
        model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        vs_obj = OracleVS(
            connection, model, "TB10", DistanceStrategy.EUCLIDEAN_DISTANCE
        )
        texts = ["Sri Ram" + str(order)]
        ids = texts
        vs_obj.add_texts(texts, ids=ids)

    tasks = []

    for i in range(POOLS_MAX + 2):
        task = asyncio.create_task(add(i))
        tasks.append(task)

    await asyncio.gather(*tasks, return_exceptions=True)

    assert connection.busy == 0

    with connection.acquire() as _conn:
        with _conn.cursor() as _conncursor:
            _conncursor.execute("select count(*) from TB10")
            count = _conncursor.fetchone()

    assert count[0] == POOLS_MAX + 2

    drop_table_purge(connection, "TB10")

    connection.close()


@pytest.mark.asyncio
async def test_add_texts_pool_test_async() -> None:
    POOLS_MAX = 4

    try:
        connection = oracledb.create_pool_async(
            user=username, password=password, dsn=dsn, min=1, max=4, increment=1
        )
    except Exception:
        sys.exit(1)

    # 1. Add different records concurrently
    # Expectation:Successful
    model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    await OracleVS.acreate(
        connection, model, "TB10", DistanceStrategy.EUCLIDEAN_DISTANCE
    )

    async def add(order: int) -> None:
        model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        vs_obj = await OracleVS.acreate(
            connection, model, "TB10", DistanceStrategy.EUCLIDEAN_DISTANCE
        )
        texts = ["Sri Ram" + str(order)]
        ids = texts
        await vs_obj.aadd_texts(texts, ids=ids)

    tasks = []

    for i in range(POOLS_MAX + 2):
        task = asyncio.create_task(add(i))
        tasks.append(task)

    await asyncio.gather(*tasks)

    assert connection.busy == 0

    async with connection.acquire() as _conn:
        count = await _conn.fetchone("select count(*) from TB10")

    assert count[0] == POOLS_MAX + 2

    await adrop_table_purge(connection, "TB10")

    await connection.close()


##################################
##### test_from_texts_lobs  ######
##################################


def test_from_texts_lobs() -> None:
    try:
        connection = oracledb.connect(user=username, password=password, dsn=dsn)
    except Exception:
        sys.exit(1)

    model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-mpnet-base-v2"
    )

    texts = [
        "If the answer to any preceding questions is yes, then the database stops \
        the search and allocates space from the specified tablespace; otherwise, \
        space is allocated from the database default shared temporary tablespace.",
        "A tablespace can be online (accessible) or offline (not accessible) whenever \
        the database is open.\nA tablespace is usually online so that its data is \
        available to users. The SYSTEM tablespace and temporary tablespaces cannot \
        be taken offline.",
    ]

    metadatas = [
        {
            "id": "cncpt_15.5.3.2.2_P4",
            "link": "https://docs.oracle.com/en/database/oracle/oracle-database/23/cncpt/logical-storage-structures.html#GUID-5387D7B2-C0CA-4C1E-811B-C7EB9B636442",
        },
        {
            "id": "cncpt_15.5.5_P1",
            "link": "https://docs.oracle.com/en/database/oracle/oracle-database/23/cncpt/logical-storage-structures.html#GUID-D02B2220-E6F5-40D9-AFB5-BC69BCEF6CD4",
        },
    ]

    vs = OracleVS.from_texts(
        texts,
        model,
        metadatas,
        client=connection,
        table_name="TB10",
        distance_strategy=DistanceStrategy.COSINE,
    )

    create_index(connection, vs, {"idx_type": "HNSW", "idx_name": "IDX1"})

    query = "What is a tablespace?"

    # 1. Test when oracledb.defaults.fetch_lobs is set to False
    # Expectation:Successful
    oracledb.defaults.fetch_lobs = False
    # similarity_search without filter
    res = vs.similarity_search(query, 2)

    assert len(res) == 2
    assert any("tablespace can be online" in str(r.page_content) for r in res)

    drop_table_purge(connection, "TB10")

    oracledb.defaults.fetch_lobs = True


@pytest.mark.asyncio
async def test_from_texts_lobs_async() -> None:
    try:
        connection = await oracledb.connect_async(
            user=username, password=password, dsn=dsn
        )
    except Exception:
        sys.exit(1)

    model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-mpnet-base-v2"
    )

    texts = [
        "If the answer to any preceding questions is yes, then the database stops \
        the search and allocates space from the specified tablespace; otherwise, \
        space is allocated from the database default shared temporary tablespace.",
        "A tablespace can be online (accessible) or offline (not accessible) whenever \
        the database is open.\nA tablespace is usually online so that its data is \
        available to users. The SYSTEM tablespace and temporary tablespaces cannot \
        be taken offline.",
    ]

    metadatas = [
        {
            "id": "cncpt_15.5.3.2.2_P4",
            "link": "https://docs.oracle.com/en/database/oracle/oracle-database/23/cncpt/logical-storage-structures.html#GUID-5387D7B2-C0CA-4C1E-811B-C7EB9B636442",
        },
        {
            "id": "cncpt_15.5.5_P1",
            "link": "https://docs.oracle.com/en/database/oracle/oracle-database/23/cncpt/logical-storage-structures.html#GUID-D02B2220-E6F5-40D9-AFB5-BC69BCEF6CD4",
        },
    ]

    vs = await OracleVS.afrom_texts(
        texts,
        model,
        metadatas,
        client=connection,
        table_name="TB10",
        distance_strategy=DistanceStrategy.COSINE,
    )

    query = "What is a tablespace?"

    # 1. Test when oracledb.defaults.fetch_lobs is set to False
    oracledb.defaults.fetch_lobs = False
    # similarity_search without filter
    res = await vs.asimilarity_search(query, 2)

    assert len(res) == 2
    assert any("tablespace can be online" in str(r.page_content) for r in res)

    await adrop_table_purge(connection, "TB10")

    oracledb.defaults.fetch_lobs = True


##################################
##### test_index_table_case  #####
##################################


def test_index_table_case(caplog: pytest.LogCaptureFixture) -> None:
    try:
        connection = oracledb.connect(user=username, password=password, dsn=dsn)
    except Exception:
        sys.exit(1)

    drop_table_purge(connection, "TB1")
    drop_table_purge(connection, "Tb1")
    drop_table_purge(connection, "TB2")

    # LOGGER = logging.getLogger(__name__)

    texts = ["Rohan", "Shailendra"]
    metadata = [
        {"id": "100", "link": "Document Example Test 1"},
        {"id": "101", "link": "Document Example Test 2"},
    ]
    model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    with caplog.at_level(logging.INFO):
        vs_obj = OracleVS(
            connection, model, "TB1", DistanceStrategy.EUCLIDEAN_DISTANCE, backend="sql"
        )

    assert 'Table "TB1" created successfully...' in caplog.records[-1].message

    with caplog.at_level(logging.INFO):
        OracleVS(
            connection,
            model,
            '"TB1"',
            DistanceStrategy.EUCLIDEAN_DISTANCE,
            backend="sql",
        )

    assert 'Table "TB1" already exists...' in caplog.records[-1].message

    with caplog.at_level(logging.INFO):
        OracleVS(
            connection, model, "Tb1", DistanceStrategy.EUCLIDEAN_DISTANCE, backend="sql"
        )

    assert 'Table "Tb1" created successfully...' in caplog.records[-1].message

    with caplog.at_level(logging.INFO):
        vs_obj2 = OracleVS(
            connection, model, "TB2", DistanceStrategy.EUCLIDEAN_DISTANCE, backend="sql"
        )

    assert 'Table "TB2" created successfully...' in caplog.records[-1].message

    vs_obj.add_texts(texts, metadata)

    with caplog.at_level(logging.INFO):
        create_index(
            connection, vs_obj, params={"idx_name": "hnsw_idx2", "idx_type": "HNSW"}
        )

    assert 'Index "hnsw_idx2" created successfully...' in caplog.records[-1].message

    with pytest.raises(RuntimeError, match="such column list already indexed"):
        create_index(
            connection, vs_obj, params={"idx_name": "HNSW_idx2", "idx_type": "HNSW"}
        )

    with pytest.raises(
        RuntimeError, match="name is already used by an existing object"
    ):
        create_index(
            connection, vs_obj2, params={"idx_name": "hnsw_idx2", "idx_type": "HNSW"}
        )

    with pytest.raises(RuntimeError, match="such column list already indexed"):
        create_index(
            connection, vs_obj, params={"idx_name": "HNSW_idx2", "idx_type": "HNSW"}
        )

    with caplog.at_level(logging.INFO):
        drop_index_if_exists(connection, "hnsw_idx2")

    assert 'Index "hnsw_idx2" has been dropped.' in caplog.records[-1].message

    with caplog.at_level(logging.INFO):
        create_index(
            connection, vs_obj, params={"idx_name": '"hnsw_idx2"', "idx_type": "HNSW"}
        )

    assert 'Index "hnsw_idx2" created successfully...' in caplog.records[-1].message

    drop_table_purge(connection, "TB1")
    drop_table_purge(connection, "Tb1")
    drop_table_purge(connection, "TB2")


@pytest.mark.asyncio
async def test_index_table_case_async(caplog: pytest.LogCaptureFixture) -> None:
    try:
        connection = await oracledb.connect_async(
            user=username, password=password, dsn=dsn
        )
    except Exception:
        sys.exit(1)

    await adrop_table_purge(connection, "TB1")
    await adrop_table_purge(connection, "Tb1")
    await adrop_table_purge(connection, "TB2")

    # LOGGER = logging.getLogger(__name__)

    texts = ["Rohan", "Shailendra"]
    metadata = [
        {"id": "100", "link": "Document Example Test 1"},
        {"id": "101", "link": "Document Example Test 2"},
    ]
    model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    with caplog.at_level(logging.INFO):
        vs_obj = await OracleVS.acreate(
            connection, model, "TB1", DistanceStrategy.EUCLIDEAN_DISTANCE, backend="sql"
        )

    assert 'Table "TB1" created successfully...' in caplog.records[-1].message

    with caplog.at_level(logging.INFO):
        await OracleVS.acreate(
            connection,
            model,
            '"TB1"',
            DistanceStrategy.EUCLIDEAN_DISTANCE,
            backend="sql",
        )

    assert 'Table "TB1" already exists...' in caplog.records[-1].message

    with caplog.at_level(logging.INFO):
        await OracleVS.acreate(
            connection, model, "Tb1", DistanceStrategy.EUCLIDEAN_DISTANCE, backend="sql"
        )

    assert 'Table "Tb1" created successfully...' in caplog.records[-1].message

    with caplog.at_level(logging.INFO):
        vs_obj2 = await OracleVS.acreate(
            connection, model, "TB2", DistanceStrategy.EUCLIDEAN_DISTANCE, backend="sql"
        )

    assert 'Table "TB2" created successfully...' in caplog.records[-1].message

    await vs_obj.aadd_texts(texts, metadata)

    with caplog.at_level(logging.INFO):
        await acreate_index(
            connection, vs_obj, params={"idx_name": "hnsw_idx2", "idx_type": "HNSW"}
        )

    assert 'Index "hnsw_idx2" created successfully...' in caplog.records[-1].message

    with pytest.raises(RuntimeError, match="such column list already indexed"):
        await acreate_index(
            connection, vs_obj, params={"idx_name": "HNSW_idx2", "idx_type": "HNSW"}
        )

    with pytest.raises(
        RuntimeError, match="name is already used by an existing object"
    ):
        await acreate_index(
            connection, vs_obj2, params={"idx_name": "hnsw_idx2", "idx_type": "HNSW"}
        )

    with pytest.raises(RuntimeError, match="such column list already indexed"):
        await acreate_index(
            connection, vs_obj, params={"idx_name": "HNSW_idx2", "idx_type": "HNSW"}
        )

    with caplog.at_level(logging.INFO):
        await adrop_index_if_exists(connection, "hnsw_idx2")

    assert 'Index "hnsw_idx2" has been dropped.' in caplog.records[-1].message

    with caplog.at_level(logging.INFO):
        await acreate_index(
            connection, vs_obj, params={"idx_name": '"hnsw_idx2"', "idx_type": "HNSW"}
        )

    assert 'Index "hnsw_idx2" created successfully...' in caplog.records[-1].message

    await adrop_table_purge(connection, "TB1")
    await adrop_table_purge(connection, "Tb1")
    await adrop_table_purge(connection, "TB2")


##################################
##### test_oracle_embeddings  ####
##################################


def test_oracle_embeddings() -> None:
    try:
        connection = oracledb.connect(user=username, password=password, dsn=dsn)
    except Exception:
        sys.exit(1)

    drop_table_purge(connection, "TB1")

    texts = ["Database Document", "Code Document"]
    metadata = [
        {"id": "100", "link": "Document Example Test 1"},
        {"id": "101", "link": "Document Example Test 2"},
    ]
    embedder_params = {"provider": "database", "model": "allminilm"}
    proxy = ""

    # instance
    model = OracleEmbeddings(conn=connection, params=embedder_params, proxy=proxy)

    vs_obj = OracleVS(connection, model, "TB1", DistanceStrategy.EUCLIDEAN_DISTANCE)

    vs_obj.add_texts(texts, metadata)
    res = vs_obj.similarity_search("database", 1)

    assert "Database" in res[0].page_content
    assert "100" == res[0].id

    embedding = model.embed_query("Database Document")
    res = vs_obj.similarity_search_by_vector_returning_embeddings(embedding, 1)  # type: ignore

    # distance
    assert all(np.isclose([res[0][1]], [0]))  # type: ignore
    assert all(np.isclose(res[0][2], embedding))  # type: ignore

    drop_table_purge(connection, "TB1")

    connection.close()


@pytest.mark.asyncio
async def test_oracle_embeddings_async(caplog: pytest.LogCaptureFixture) -> None:
    try:
        connection = await oracledb.connect_async(
            user=username, password=password, dsn=dsn
        )

        connection_sync = oracledb.connect(user=username, password=password, dsn=dsn)
    except Exception:
        sys.exit(1)

    await adrop_table_purge(connection, "TB1")

    texts = ["Database Document", "Code Document"]
    metadata = [
        {"id": "100", "link": "Document Example Test 1"},
        {"id": "101", "link": "Document Example Test 2"},
    ]
    embedder_params = {"provider": "database", "model": "allminilm"}
    proxy = ""

    # instance
    model = OracleEmbeddings(conn=connection_sync, params=embedder_params, proxy=proxy)

    vs_obj = await OracleVS.acreate(
        connection, model, "TB1", DistanceStrategy.EUCLIDEAN_DISTANCE
    )

    await vs_obj.aadd_texts(texts, metadata)
    res = await vs_obj.asimilarity_search("database", 1)

    assert "Database" in res[0].page_content
    assert "100" == res[0].id

    embedding = model.embed_query("Database Document")
    res = await vs_obj.asimilarity_search_by_vector_returning_embeddings(embedding, 1)  # type: ignore

    # distance
    assert all(np.isclose([res[0][1]], [0]))  # type: ignore
    assert all(np.isclose(res[0][2], embedding))  # type: ignore

    await adrop_table_purge(connection, "TB1")

    await connection.close()


##################################
##### test_quote_identifier  #####
##################################


def test_quote_identifier() -> None:
    # unquoted
    assert _quote_indentifier("hello") == '"hello"'
    assert _quote_indentifier("--") == '"--"'
    assert _quote_indentifier("U1.table") == '"U1"."table"'
    assert _quote_indentifier("hnsw_idx2") == '"hnsw_idx2"'
    assert _quote_indentifier("'") == '"\'"'

    with pytest.raises(ValueError):
        _quote_indentifier('hnsw_"idx2')

    with pytest.raises(ValueError):
        _quote_indentifier('"')

    with pytest.raises(ValueError):
        _quote_indentifier('"--')

    with pytest.raises(ValueError):
        _quote_indentifier(" ")

    # quoted
    assert _quote_indentifier('"U1.table"') == '"U1.table"'
    assert _quote_indentifier('"U1"."table"') == '"U1"."table"'
    assert _quote_indentifier('"he".--') == '"he"."--"'

    with pytest.raises(ValueError):
        assert _quote_indentifier('"he"llo"')

    with pytest.raises(ValueError):
        assert _quote_indentifier('"he"--')

    # mixed
    assert _quote_indentifier('"U1".table') == '"U1"."table"'


##################################
########## test_filters  #########
##################################


##################################
####### test_reserved  ######
##################################


def test_reserved(oracle_backend, legacy_embeddings) -> None:
    with backend_store(oracle_backend, legacy_embeddings) as store:
        texts = ["Database Document", "Code Document"]
        metadata = [
            {
                "id": "100",
                "link": "Document Example Test 1",
                INTERNAL_ID_KEY: "my_temp_id",
            },
            {"id": "101", "link": "Document Example Test 2"},
        ]
        with pytest.raises(ValueError, match="reserved"):
            store.add_texts(texts, metadata, ids=["1", "2"])


@pytest.mark.asyncio
async def test_reserved_async(oracle_backend, legacy_embeddings) -> None:
    async with async_backend_store(oracle_backend, legacy_embeddings) as store:
        texts = ["Database Document", "Code Document"]
        metadata = [
            {
                "id": "100",
                "link": "Document Example Test 1",
                INTERNAL_ID_KEY: "my_temp_id",
            },
            {"id": "101", "link": "Document Example Test 2"},
        ]
        with pytest.raises(ValueError, match="reserved"):
            await store.aadd_texts(texts, metadata, ids=["1", "2"])
