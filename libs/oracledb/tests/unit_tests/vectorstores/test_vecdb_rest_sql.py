"""REST hybrid contracts through the real SDK serializer and response parser."""

import json
import re
from unittest.mock import MagicMock

import pytest

from langchain_oracledb.vectorstores._vecdb import (
    CURRENT_COLUMNS,
    RestTransport,
    _hybrid_sql,
    _TableInfo,
)
from langchain_oracledb.vectorstores._vecdb_rest_sql import ORDSSQLConnection


def envelope(columns=(), values=(), **extra):
    return {
        "items": [
            {
                "resultSet": {
                    "metadata": [
                        {"columnName": name, "jsonColumnName": name.lower()}
                        for name in columns
                    ],
                    "items": [dict(zip(columns, row)) for row in values],
                    **extra,
                }
            }
        ]
    }


@pytest.fixture
def sdk():
    oracle_vecdb = pytest.importorskip("oracle_vecdb")
    return oracle_vecdb.OracleVecDB(
        oracle_vecdb.Configuration(
            rest_url="https://example.com/ords/test/_/db-api/stable/vecdb/",
            access_token="test",
        )
    )


def mock_http(monkeypatch, sdk, responses, status=200):
    from oracle_vecdb.services.ords.rest import RESTResponse
    from urllib3 import HTTPResponse

    def respond(*args, **kwargs):
        payload = responses.pop(0)
        return RESTResponse(
            HTTPResponse(
                body=json.dumps(payload).encode(),
                status=status,
                headers={"content-type": "application/json"},
            )
        )

    call = MagicMock(side_effect=respond)
    monkeypatch.setattr(sdk.api_client, "call_api", call)
    return call


def test_sdk_sql_transport_uses_authenticated_endpoint_and_typed_binds(
    sdk, monkeypatch
):
    call = mock_http(monkeypatch, sdk, [envelope(["X"], [(1,)])])
    cursor = ORDSSQLConnection(sdk).cursor()
    cursor.setinputsizes(search_params=object())
    cursor.execute(
        "select :search_params as x from dual", search_params="sensitive query"
    )
    assert cursor.fetchall() == [(1,)]
    args = call.call_args.args
    assert args[0] == "POST"
    assert args[1] == "https://example.com/ords/test/_/sql"
    body = args[3]
    assert body["binds"] == [
        {
            "name": "search_params",
            "value": "sensitive query",
            "data_type": "CLOB",
            "mode": "in",
            "batch": False,
        }
    ]
    assert args[2]["Authorization"] == "Bearer test"
    assert args[2]["Content-Type"] == "application/json"


@pytest.mark.parametrize(
    "payload,exception",
    [
        (
            {"items": [{"errorCode": 942, "errorMessage": "ORA-00942: missing table"}]},
            RuntimeError,
        ),
        (
            {"items": [{"errorDetails": "ORA-01031: insufficient privileges"}]},
            RuntimeError,
        ),
        ({}, ValueError),
        (envelope(["X"], [(1,)], hasMore=True), ValueError),
    ],
)
def test_sql_errors_and_truncated_results_are_not_success(
    sdk, monkeypatch, payload, exception
):
    mock_http(monkeypatch, sdk, [payload])
    with pytest.raises(exception):
        ORDSSQLConnection(sdk).cursor().execute("select 1 from dual")


def test_http_errors_propagate(sdk, monkeypatch):
    from oracle_vecdb.services.ords.exceptions import ApiException

    mock_http(monkeypatch, sdk, [{"message": "forbidden"}], status=403)
    with pytest.raises(ApiException):
        ORDSSQLConnection(sdk).cursor().execute("select 1 from dual")


@pytest.mark.parametrize("keyword_only", [False, True])
def test_rest_uses_native_hybrid_sql_and_document_mapping(
    sdk, monkeypatch, keyword_only
):
    transport = RestTransport(sdk)
    shared = transport.hybrid_transport()
    shared._remember_table(
        "DOCS", _TableInfo(frozenset(CURRENT_COLUMNS), "PHYSICAL_HVI")
    )
    call = mock_http(
        monkeypatch,
        sdk,
        [
            envelope(
                ["ID", "CONTENT_METADATA", "SCORE", "TEXT_SCORE", "VECTOR_SCORE"],
                [
                    (
                        "stored-id",
                        json.dumps(
                            {
                                "__orcl_internal_content": "hello",
                                "__orcl_internal_doc_id": "public-id",
                                "tenant": "a",
                            }
                        ),
                        9,
                        2,
                        7,
                    )
                ],
            )
        ],
    )
    params = {"text": {"contains": "x' OR 1=1"}, "return": {"topN": 3}}
    if not keyword_only:
        params["vector"] = {"search_vector": [1, 0]}
    docs = transport.hybrid_search("DOCS", params, True)
    assert docs[0].id == "public-id"
    assert docs[0].page_content == "hello"
    assert docs[0].metadata == {
        "tenant": "a",
        "score": 9,
        "text_score": 2,
        "vector_score": 0 if keyword_only else 7,
    }
    body = call.call_args.args[3]
    assert _hybrid_sql("DOCS", set(CURRENT_COLUMNS)) in re.sub(
        r" /\*.*?\*/", "", body["statementText"]
    )
    assert "x' OR 1=1" not in body["statementText"]
    request = json.loads(body["binds"][0]["value"])
    assert request["hybrid_index_name"] == '"PHYSICAL_HVI"'
    assert request["text"]["contains"] == "x' OR 1=1"


def test_rest_index_replacement_restores_original_after_failure(sdk, monkeypatch):
    transport = RestTransport(sdk).hybrid_transport()
    monkeypatch.setattr(transport, "columns", lambda name: set(CURRENT_COLUMNS))
    parameters = (
        "SIMPLIFIED_JSON DATAGUIDE ON CHANGE "
        "SYS.DBMS_VECTOR_DATABASE.HANDLE_MVI_INDEX "
        'SYNC (EVERY "freq=secondly; interval=30")'
    )
    call = mock_http(
        monkeypatch,
        sdk,
        [
            envelope(
                ["INDEX_NAME", "PARAMETERS", "OWNER"],
                [("EXISTING", parameters, "TEST")],
            ),
            {"items": [{}]},
            {"items": [{"errorMessage": "ORA-29855: failed index creation"}]},
            envelope(["INDEX_NAME"], [("EXISTING",)]),
            {"items": [{}]},
            {"items": [{}]},
        ],
    )

    def build(index, params, column):
        return (
            f"CREATE HYBRID VECTOR INDEX {index} ON DOCS({column}) "
            f"PARAMETERS ('VECTORIZER PREF {params}')"
        )

    with pytest.raises(RuntimeError, match="ORA-29855"):
        transport.create_metadata_index("DOCS", "IGNORED", hybrid=True, build=build)
    statements = [call.args[3]["statementText"] for call in call.call_args_list]
    assert statements[1] == 'DROP INDEX "TEST"."EXISTING"'
    assert statements[2].startswith('CREATE HYBRID VECTOR INDEX "TEST"."EXISTING"')
    assert (
        "DATAGUIDE ON CHANGE SYS.DBMS_VECTOR_DATABASE.HANDLE_MVI_INDEX" in statements[2]
    )
    assert 'SYNC (EVERY "freq=secondly; interval=30")' in statements[2]
    assert statements[4] == 'DROP INDEX "TEST"."EXISTING" FORCE'
    assert statements[5].startswith('CREATE SEARCH INDEX "TEST"."EXISTING"')


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_public_rest_hybrid_lifecycle(sdk, monkeypatch, asynchronous):
    from types import SimpleNamespace

    from langchain_oracledb.embeddings import OracleEmbeddings
    from langchain_oracledb.retrievers.hybrid_search import (
        OracleVectorizerPreference,
        acreate_hybrid_index,
        create_hybrid_index,
    )
    from langchain_oracledb.vectorstores._vecdb import VecDBBackend
    from langchain_oracledb.vectorstores.utils import DistanceStrategy

    backend = VecDBBackend(
        RestTransport(sdk), "DOCS", DistanceStrategy.COSINE, mutate_on_duplicate=True
    )
    store = SimpleNamespace(
        _vecdb=backend,
        client=sdk,
        table_name="DOCS",
        embedding_function=OracleEmbeddings(
            conn=sdk, params={"provider": "database", "model": "EMBED_MODEL"}
        ),
    )
    shared = backend.hybrid_transport()
    monkeypatch.setattr(shared, "columns", lambda name: set(CURRENT_COLUMNS))
    parameters = (
        "SIMPLIFIED_JSON DATAGUIDE ON CHANGE "
        "SYS.DBMS_VECTOR_DATABASE.HANDLE_MVI_INDEX "
        'SYNC (EVERY "freq=secondly; interval=30")'
    )
    call = mock_http(
        monkeypatch,
        sdk,
        [
            {"items": [{}]},
            envelope(
                ["INDEX_NAME", "PARAMETERS", "OWNER"],
                [("META_IDX", parameters, "TEST")],
            ),
            {"items": [{}]},
            {"items": [{}]},
            {"items": [{}]},
        ],
    )
    if asynchronous:
        await acreate_hybrid_index(sdk, "REQUESTED", vector_store=store)
    else:
        create_hybrid_index(sdk, "REQUESTED", vector_store=store)
    bodies = [call.args[3] for call in call.call_args_list]
    assert "CREATE_PREFERENCE" in bodies[0]["statementText"]
    pref_params = json.loads(bodies[0]["binds"][1]["value"])
    assert pref_params["model"] == "EMBED_MODEL"
    assert pref_params["paths"][0]["path_list"] == ["$.__orcl_internal_content"]
    assert bodies[2]["statementText"] == 'DROP INDEX "TEST"."META_IDX"'
    assert 'CREATE HYBRID VECTOR INDEX "TEST"."META_IDX"' in bodies[3]["statementText"]
    assert "DROP_PREFERENCE" in bodies[4]["statementText"]
    # Also exercise standalone preference creation/drop through public async APIs.
    mock_http(monkeypatch, sdk, [{"items": [{}]}, {"items": [{}]}])
    if asynchronous:
        pref = await OracleVectorizerPreference.acreate_preference(store)
        await pref.adrop_preference()
    else:
        pref = OracleVectorizerPreference.create_preference(store)
        pref.drop_preference()


@pytest.mark.parametrize("legacy", [False, True])
def test_rest_catalog_discovery_uses_actual_layout(sdk, monkeypatch, legacy):
    columns = {"ID", "DENSE_VECTOR", "METADATA"} if legacy else set(CURRENT_COLUMNS)
    call = mock_http(
        monkeypatch,
        sdk,
        [
            envelope(["COLUMN_NAME"], [(column,) for column in sorted(columns)]),
            envelope(
                ["INDEX_NAME", "PARAMETERS", "OWNER"],
                [("META_IDX", "SIMPLIFIED_JSON_HYBRID", "TEST")],
            ),
        ],
    )
    info = RestTransport(sdk).hybrid_transport().discover('"Mixed.Table"')
    assert info.metadata_column == ("METADATA" if legacy else "CONTENT_METADATA")
    assert info.hybrid_index == '"META_IDX"'
    assert call.call_args_list[0].args[3]["binds"][0]["value"] == "Mixed.Table"


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_rest_explicit_index_and_preference_drop(sdk, monkeypatch, asynchronous):
    from langchain_oracledb.retrievers.hybrid_search import (
        adrop_preference,
        drop_preference,
    )
    from langchain_oracledb.vectorstores.utils import adrop_index, drop_index

    call = mock_http(
        monkeypatch,
        sdk,
        [
            envelope(["INDEX_NAME"], [("Physical.Idx",)]),
            {"items": [{}]},
            {"items": [{}]},
        ],
    )
    if asynchronous:
        await adrop_index(sdk, '"Owner"."Physical.Idx"')
        await adrop_preference(sdk, "pref")
    else:
        drop_index(sdk, '"Owner"."Physical.Idx"')
        drop_preference(sdk, "pref")
    bodies = [call.args[3] for call in call.call_args_list]
    assert bodies[1]["statementText"] == 'DROP INDEX "Owner"."Physical.Idx"'
    assert bodies[2]["binds"][0]["value"] == "pref"


def test_rest_drop_missing_index_is_noop(sdk, monkeypatch):
    from langchain_oracledb.vectorstores.utils import drop_index

    call = mock_http(monkeypatch, sdk, [envelope(["INDEX_NAME"])])
    drop_index(sdk, "MISSING")
    assert call.call_count == 1


@pytest.mark.parametrize(
    "params",
    [
        {"return": {"topN": 0}},
        {"vector": {"search_vector": [float("nan")]}},
        {"text": {"inpath": ["not-a-json-path"]}},
    ],
)
def test_rest_hybrid_validation_rejects_inputs_before_http(sdk, monkeypatch, params):
    call = mock_http(monkeypatch, sdk, [])
    with pytest.raises((TypeError, ValueError)):
        RestTransport(sdk).hybrid_search("DOCS", params, False)
    call.assert_not_called()
