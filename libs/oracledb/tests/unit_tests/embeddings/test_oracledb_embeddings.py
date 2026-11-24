# Copyright (c) 2024, 2026, Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""
unit_tests/embeddings/test_oracledb_embeddings.py

Unit tests for OracleEmbeddings, exercising the full embeddings contract
without any real database connection. All DB interactions are intercepted
via mock conn and cursor objects at exact call sites.

Covers:
- Constructor attribute storage and defaults
    (conn, params, proxy, extra field rejection)
- load_onnx_model PL/SQL execution, bind variable correctness, and cursor cleanup
- load_onnx_model None input validation
    (None conn, dir, onnx file, model name all raise)
- embed_documents output shape, vector parsing, and multi-text batching
- embed_documents internals (fetch_lobs flag, setinputsizes, utl_to_embeddings SQL,
  SYS.VECTOR_ARRAY_T type fetch, chunk id and data construction)
- Proxy execute path triggered only when proxy is set
- Cursor cleanup on both success and failure paths
- embed_query delegation to embed_documents and correct result unwrapping

Run:
    pytest tests/unit_tests/embeddings/test_oracledb_embeddings.py

Authors:
    - Diego Ascencio (diegoascencioqa)
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import oracledb
import pytest

from langchain_oracledb.embeddings.oracleai import OracleEmbeddings

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def make_conn(rows=None, execute_side_effect=None):
    """Return (mock_conn, mock_cursor) with optional row iteration and
    optional execute side-effect for error path tests."""
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value = cursor

    if execute_side_effect is not None:
        cursor.execute.side_effect = execute_side_effect
    if rows is not None:
        cursor.__iter__ = MagicMock(return_value=iter(rows))

    vector_type = MagicMock()
    vector_type.newobject.return_value = MagicMock()
    conn.gettype.return_value = vector_type

    return conn, cursor


def make_embedder(conn, params=None, proxy=None):
    """Return an OracleEmbeddings instance with sensible defaults."""
    return OracleEmbeddings(
        conn=conn,
        params=params or {"provider": "database", "model": "demo_model"},
        **({"proxy": proxy} if proxy is not None else {}),
    )


def embed_row(chunk_id: int, vector: list) -> tuple:
    """Build a fake cursor row matching Oracle's utl_to_embeddings output."""
    payload = json.dumps(
        {
            "embed_id": chunk_id,
            "embed_vector": json.dumps(vector),
        }
    )
    return (payload,)


# ===========================================================================
# Constructor
# ===========================================================================


class TestConstructor:
    def test_basic_construction(self):
        conn, cursor = make_conn()
        embedder = make_embedder(conn)
        assert embedder is not None

    def test_conn_stored(self):
        conn, cursor = make_conn()
        embedder = make_embedder(conn)
        assert embedder.conn is conn

    def test_params_stored(self):
        conn, cursor = make_conn()
        params = {"provider": "database", "model": "allminilm"}
        embedder = make_embedder(conn, params=params)
        assert embedder.params == params

    def test_proxy_stored(self):
        conn, cursor = make_conn()
        embedder = make_embedder(conn, proxy="http://proxy:80")
        assert embedder.proxy == "http://proxy:80"

    def test_proxy_is_none_by_default(self):
        conn, cursor = make_conn()
        embedder = make_embedder(conn)
        assert embedder.proxy is None

    def test_extra_fields_forbidden(self):
        """model_config = extra='forbid' — unknown fields must raise."""
        conn, cursor = make_conn()
        with pytest.raises(Exception):
            OracleEmbeddings(conn=conn, params={}, unknown_field="x")


# ===========================================================================
# load_onnx_model
# ===========================================================================


class TestLoadOnnxModel:
    @pytest.fixture
    def rest_client(self):
        sdk = pytest.importorskip("oracle_vecdb")
        client = sdk.OracleVecDB(
            sdk.Configuration(
                rest_url="https://example.com/ords/test/_/db-api/stable/vecdb/",
                access_token="test",
            )
        )
        client.load_model = MagicMock(return_value={"model_name": "MY_MODEL"})
        client.close = MagicMock()
        return client

    @pytest.mark.parametrize(
        "model_params", [None, {"metadata": {"function": "embedding"}}]
    )
    def test_rest_delegates_url_loading(self, rest_client, model_params):
        url = "https://example.com/model.onnx"
        result = OracleEmbeddings.load_onnx_model(
            rest_client, model_name="MY_MODEL", url=url, model_params=model_params
        )
        assert result is None
        rest_client.load_model.assert_called_once_with(
            model_name="MY_MODEL", url=url, model_params=model_params
        )
        rest_client.close.assert_not_called()

    def test_rest_directory_loading_rejected(self, rest_client):
        with pytest.raises(ValueError, match="REST SDK model loading requires url"):
            OracleEmbeddings.load_onnx_model(
                rest_client, "MY_DIR", "model.onnx", "MY_MODEL"
            )
        rest_client.load_model.assert_not_called()
        rest_client.close.assert_not_called()

    def test_rest_propagates_load_error(self, rest_client):
        error = RuntimeError("model download failed")
        rest_client.load_model.side_effect = error
        with pytest.raises(RuntimeError) as raised:
            OracleEmbeddings.load_onnx_model(
                rest_client, model_name="MY_MODEL", url="https://example.com/model.onnx"
            )
        assert raised.value is error
        rest_client.load_model.assert_called_once()
        rest_client.close.assert_not_called()

    def test_sqlnet_sdk_rejected(self, rest_client):
        from types import SimpleNamespace

        rest_client.config = SimpleNamespace(
            has_sqlnet_connection=True, has_rest_connection=False
        )
        with pytest.raises(ValueError, match="SQLNet-configured"):
            OracleEmbeddings.load_onnx_model(
                rest_client, model_name="MY_MODEL", url="https://example.com/model.onnx"
            )
        rest_client.load_model.assert_not_called()

    def test_executes_plsql_block(self):
        conn, cursor = make_conn()
        OracleEmbeddings.load_onnx_model(conn, "MY_DIR", "model.onnx", "MY_MODEL")
        assert cursor.execute.called

    def test_passes_correct_bind_vars(self):
        conn, cursor = make_conn()
        OracleEmbeddings.load_onnx_model(conn, "MY_DIR", "model.onnx", "MY_MODEL")
        _, kwargs = cursor.execute.call_args
        assert kwargs["path"] == "MY_DIR"
        assert kwargs["filename"] == "model.onnx"
        assert kwargs["model"] == "MY_MODEL"

    def test_cursor_closed_on_success(self):
        conn, cursor = make_conn()
        OracleEmbeddings.load_onnx_model(conn, "MY_DIR", "model.onnx", "MY_MODEL")
        cursor.close.assert_called_once()

    def test_cursor_closed_on_failure(self):
        conn, cursor = make_conn(execute_side_effect=Exception("DB error"))
        with pytest.raises(Exception):
            OracleEmbeddings.load_onnx_model(conn, "MY_DIR", "model.onnx", "MY_MODEL")
        cursor.close.assert_called_once()

    def test_raises_on_none_conn(self):
        """None conn must raise Exception with 'Invalid input' message."""
        with pytest.raises(Exception, match="Invalid input"):
            OracleEmbeddings.load_onnx_model(None, "MY_DIR", "model.onnx", "MY_MODEL")

    def test_raises_on_none_dir(self):
        """None dir must raise Exception with 'Invalid input' message."""
        conn, cursor = make_conn()
        with pytest.raises(Exception, match="Invalid input"):
            OracleEmbeddings.load_onnx_model(conn, None, "model.onnx", "MY_MODEL")

    def test_raises_on_none_onnx_file(self):
        """None onnx file must raise Exception with 'Invalid input' message."""
        conn, cursor = make_conn()
        with pytest.raises(Exception, match="Invalid input"):
            OracleEmbeddings.load_onnx_model(conn, "MY_DIR", None, "MY_MODEL")

    def test_raises_on_none_model_name(self):
        """None model name must raise Exception with 'Invalid input' message."""
        conn, cursor = make_conn()
        with pytest.raises(Exception, match="Invalid input"):
            OracleEmbeddings.load_onnx_model(conn, "MY_DIR", "model.onnx", None)

    def test_reraises_db_exception(self):
        conn, cursor = make_conn(execute_side_effect=RuntimeError("ORA-00942"))
        with pytest.raises(RuntimeError, match="ORA-00942"):
            OracleEmbeddings.load_onnx_model(conn, "MY_DIR", "model.onnx", "MY_MODEL")

    @pytest.mark.parametrize(
        "model_params",
        [
            None,
            {},
            {
                "credential": "OBJ_STORE_CRED",
                "metadata": {
                    "function": "embedding",
                    "embeddingOutput": "embedding",
                    "input": {"input": ["DATA"]},
                },
            },
        ],
    )
    def test_loads_url_with_bound_parameters(self, model_params):
        conn, cursor = make_conn()
        url = "https://example.com/model.onnx?token=it's-a-token"
        model_name = "MODEL_WITH_'_QUOTE"

        result = OracleEmbeddings.load_onnx_model(
            conn, model_name=model_name, url=url, model_params=model_params
        )

        assert result is None
        cursor.execute.assert_called_once()
        (sql,), binds = cursor.execute.call_args
        assert "DBMS_VECTOR_DATABASE.LOAD_MODEL(" in sql
        assert "model_params => json(:model_params)" in sql
        assert "drop_model" not in sql.lower()
        assert "DBMS_VECTOR.load_onnx_model" not in sql
        assert url not in sql
        assert model_name not in sql
        assert binds["url"] == url
        assert binds["model"] == model_name
        if model_params is None:
            assert binds["model_params"] is None
        else:
            assert json.loads(binds["model_params"]) == model_params
        cursor.var.assert_called_once_with(oracledb.DB_TYPE_CLOB)
        assert binds["result"] is cursor.var.return_value
        cursor.close.assert_called_once()

    @pytest.mark.parametrize(
        "message",
        [
            "PLS-00302: component 'LOAD_MODEL' must be declared",
            "ORA-01031: insufficient privileges",
            "model already exists",
            "model download failed",
        ],
    )
    def test_url_propagates_database_error_without_retry(self, message):
        error = oracledb.DatabaseError(message)
        conn, cursor = make_conn(execute_side_effect=error)
        with pytest.raises(oracledb.DatabaseError) as raised:
            OracleEmbeddings.load_onnx_model(
                conn, model_name="MY_MODEL", url="https://example.com/model.onnx"
            )
        assert raised.value is error
        cursor.execute.assert_called_once()
        cursor.close.assert_called_once()

    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {"url": ""},
            {"url": "https://example.com/model.onnx", "dir": "MY_DIR"},
            {"url": "https://example.com/model.onnx", "onnx_file": "model.onnx"},
            {"url": "https://example.com/model.onnx", "model_params": []},
            {"dir": "MY_DIR", "onnx_file": "model.onnx", "model_params": {}},
            {"dir": "MY_DIR"},
            {"onnx_file": "model.onnx"},
        ],
    )
    def test_invalid_source_rejected_before_database_access(self, kwargs):
        conn, _ = make_conn()
        with pytest.raises(ValueError, match="Invalid input"):
            OracleEmbeddings.load_onnx_model(conn, model_name="MY_MODEL", **kwargs)
        conn.cursor.assert_not_called()

    @pytest.mark.parametrize("missing", ["conn", "model_name"])
    def test_url_requires_connection_and_model_name(self, missing):
        conn, _ = make_conn()
        kwargs = {
            "conn": conn,
            "model_name": "MY_MODEL",
            "url": "https://example.com/model.onnx",
        }
        kwargs[missing] = None
        with pytest.raises(ValueError, match="Invalid input"):
            OracleEmbeddings.load_onnx_model(**kwargs)
        conn.cursor.assert_not_called()


# ===========================================================================
# embed_documents
# ===========================================================================


class TestRestEmbeddings:
    @pytest.fixture
    def client(self):
        sdk = pytest.importorskip("oracle_vecdb")
        client = sdk.OracleVecDB(
            sdk.Configuration(
                rest_url="https://example.com/ords/test/_/db-api/stable/vecdb/",
                access_token="test",
            )
        )
        client.generate_embedding = MagicMock(
            return_value=SimpleNamespace(
                to_dict=MagicMock(
                    return_value={"data": [{"text": "hello", "embedding": [1, "0.5"]}]}
                )
            )
        )
        client.close = MagicMock()
        return client

    def test_query_and_documents_delegate_to_rest(self, client):
        embedder = make_embedder(client)
        assert embedder.embed_documents(["hello"]) == [[1.0, 0.5]]
        assert embedder.embed_query("hello") == [1.0, 0.5]
        client.generate_embedding.assert_called_with(
            model_name="demo_model", inputs=["hello"]
        )
        assert client.generate_embedding.call_count == 2
        client.close.assert_not_called()

    def test_default_provider(self, client):
        assert make_embedder(client, params={"model": "M"}).embed_query("hello") == [
            1.0,
            0.5,
        ]

    def test_empty_input_does_not_request(self, client):
        assert make_embedder(client).embed_documents([]) == []
        client.generate_embedding.assert_not_called()

    @pytest.mark.asyncio
    async def test_async_interfaces(self, client):
        embedder = make_embedder(client)
        assert await embedder.aembed_documents(["hello"]) == [[1.0, 0.5]]
        assert await embedder.aembed_query("hello") == [1.0, 0.5]

    @pytest.mark.parametrize(
        "params,proxy",
        [
            ({"provider": "cohere", "model": "M"}, None),
            ({"provider": "database"}, None),
            ({"model": ""}, None),
            ({"model": "M", "credential_name": "CRED"}, None),
            ({"model": "M"}, "proxy.example.com"),
        ],
    )
    def test_unsupported_configuration_fails_before_request(
        self, client, params, proxy
    ):
        with pytest.raises(ValueError):
            make_embedder(client, params=params, proxy=proxy).embed_documents(["hello"])
        client.generate_embedding.assert_not_called()

    @pytest.mark.parametrize(
        "payload",
        [
            {},
            {"data": []},
            {"data": [{"embedding": []}]},
            {"data": [{"text": "different", "embedding": [1]}]},
            {"data": [{"embedding": [True]}]},
            {"data": [{"embedding": [None]}]},
            {"data": [{"embedding": ["NaN"]}]},
            {"data": [{"embedding": ["Infinity"]}]},
        ],
    )
    def test_malformed_response_rejected(self, client, payload):
        client.generate_embedding.return_value.to_dict.return_value = payload
        with pytest.raises(ValueError):
            make_embedder(client).embed_documents(["hello"])

    def test_inconsistent_dimensions_rejected(self, client):
        client.generate_embedding.return_value.to_dict.return_value = {
            "data": [{"embedding": [1]}, {"embedding": [1, 2]}]
        }
        with pytest.raises(ValueError, match="inconsistent dimensions"):
            make_embedder(client).embed_documents(["hello", "world"])

    def test_sdk_error_propagates_without_retry(self, client):
        error = RuntimeError("model missing")
        client.generate_embedding.side_effect = error
        with pytest.raises(RuntimeError) as raised:
            make_embedder(client).embed_query("hello")
        assert raised.value is error
        client.generate_embedding.assert_called_once()


class TestEmbedDocuments:
    @pytest.mark.parametrize(
        "element_type", [oracledb.DB_TYPE_VARCHAR, oracledb.DB_TYPE_CLOB]
    )
    def test_collection_input_follows_database_element_type(self, element_type):
        conn, _ = make_conn(rows=[embed_row(1, [0.1, 0.2])])
        vector_type = conn.gettype.return_value
        vector_type.element_type = element_type
        lob = conn.createlob.return_value
        assert make_embedder(conn).embed_documents(["hello"]) == [[0.1, 0.2]]
        chunk = json.dumps({"chunk_id": 1, "chunk_data": "hello"})
        if element_type == oracledb.DB_TYPE_CLOB:
            conn.createlob.assert_called_once_with(oracledb.DB_TYPE_CLOB, chunk)
            vector_type.newobject.assert_called_once_with([lob])
        else:
            conn.createlob.assert_not_called()
            vector_type.newobject.assert_called_once_with([chunk])

    def test_empty_input_does_not_require_a_database_model(self):
        conn, _ = make_conn()
        assert make_embedder(conn).embed_documents([]) == []
        conn.cursor.assert_not_called()

    def test_returns_list_of_lists(self):
        conn, _ = make_conn(rows=[embed_row(1, [0.1, 0.2, 0.3])])
        embedder = make_embedder(conn)
        with patch("oracledb.defaults"):
            result = embedder.embed_documents(["hello world"])
        assert isinstance(result, list)
        assert isinstance(result[0], list)

    def test_vector_values_parsed_correctly(self):
        vector = [0.1, 0.2, 0.3]
        conn, _ = make_conn(rows=[embed_row(1, vector)])
        embedder = make_embedder(conn)
        with patch("oracledb.defaults"):
            result = embedder.embed_documents(["hello"])
        assert result[0] == pytest.approx(vector)

    def test_multiple_texts_return_multiple_embeddings(self):
        rows = [embed_row(1, [0.1, 0.2]), embed_row(2, [0.3, 0.4])]
        conn, _ = make_conn(rows=rows)
        embedder = make_embedder(conn)
        with patch("oracledb.defaults"):
            result = embedder.embed_documents(["text1", "text2"])
        assert len(result) == 2
        assert result[0] == pytest.approx([0.1, 0.2])
        assert result[1] == pytest.approx([0.3, 0.4])

    def test_empty_input_returns_empty_list(self):
        conn, _ = make_conn(rows=[])
        embedder = make_embedder(conn)
        with patch("oracledb.defaults"):
            result = embedder.embed_documents([])
        assert result == []

    def test_none_row_appends_empty_vector(self):
        """Source appends [] when row is None — documents actual behaviour."""
        conn, cursor = make_conn()
        cursor.__iter__ = MagicMock(return_value=iter([None]))
        embedder = make_embedder(conn)
        with patch("oracledb.defaults"):
            result = embedder.embed_documents(["hello"])
        assert result == [[]]

    def test_fetch_lobs_set_to_false(self):
        """oracledb.defaults.fetch_lobs must be False before querying."""
        conn, _ = make_conn(rows=[embed_row(1, [0.1])])
        embedder = make_embedder(conn)
        with patch("oracledb.defaults") as mock_defaults:
            embedder.embed_documents(["hello"])
        assert mock_defaults.fetch_lobs is False

    def test_setinputsizes_called_with_json_type(self):
        """Source calls cursor.setinputsizes(None, oracledb.DB_TYPE_JSON)
        before execute so Oracle knows the params bind var is JSON."""
        conn, cursor = make_conn(rows=[embed_row(1, [0.1])])
        embedder = make_embedder(conn)
        with patch("oracledb.defaults"):
            embedder.embed_documents(["hello"])
        cursor.setinputsizes.assert_called_once_with(None, oracledb.DB_TYPE_JSON)

    def test_sql_contains_utl_to_embeddings(self):
        """The execute call must reference Oracle's utl_to_embeddings function."""
        conn, cursor = make_conn(rows=[embed_row(1, [0.1])])
        embedder = make_embedder(conn)
        with patch("oracledb.defaults"):
            embedder.embed_documents(["hello"])
        sql = cursor.execute.call_args.args[0]
        assert "utl_to_embeddings" in sql

    def test_params_passed_as_second_bind_variable(self):
        """Source passes [inputs, self.params] — params dict must be second."""
        params = {"provider": "database", "model": "allminilm"}
        conn, cursor = make_conn(rows=[embed_row(1, [0.1])])
        embedder = make_embedder(conn, params=params)
        with patch("oracledb.defaults"):
            embedder.embed_documents(["hello"])
        bind_args = cursor.execute.call_args.args[1]
        assert bind_args[1] == params

    def test_vector_array_type_fetched_from_connection(self):
        """Source calls conn.gettype('SYS.VECTOR_ARRAY_T') to build the input."""
        conn, _ = make_conn(rows=[embed_row(1, [0.1])])
        embedder = make_embedder(conn)
        with patch("oracledb.defaults"):
            embedder.embed_documents(["hello"])
        conn.gettype.assert_called_once_with("SYS.VECTOR_ARRAY_T")

    def test_chunks_built_with_one_based_ids_and_correct_data(self):
        """Each chunk must have chunk_id starting at 1 and chunk_data matching input."""
        conn, cursor = make_conn(rows=[embed_row(1, [0.1]), embed_row(2, [0.2])])
        vector_type = conn.gettype.return_value
        embedder = make_embedder(conn)
        with patch("oracledb.defaults"):
            embedder.embed_documents(["first", "second"])
        chunks_passed = vector_type.newobject.call_args.args[0]
        parsed = [json.loads(c) for c in chunks_passed]
        assert parsed[0]["chunk_id"] == 1
        assert parsed[0]["chunk_data"] == "first"
        assert parsed[1]["chunk_id"] == 2
        assert parsed[1]["chunk_data"] == "second"

    def test_proxy_execute_called_when_proxy_set(self):
        conn, cursor = make_conn(rows=[embed_row(1, [0.1])])
        embedder = make_embedder(conn, proxy="http://myproxy:80")
        with patch("oracledb.defaults"):
            embedder.embed_documents(["hello"])
        calls = cursor.execute.call_args_list
        assert any("utl_http" in str(c) for c in calls)

    def test_proxy_execute_not_called_when_none(self):
        conn, cursor = make_conn(rows=[embed_row(1, [0.1])])
        embedder = make_embedder(conn, proxy=None)
        with patch("oracledb.defaults"):
            embedder.embed_documents(["hello"])
        calls = cursor.execute.call_args_list
        assert not any("utl_http" in str(c) for c in calls)

    def test_cursor_closed_on_success(self):
        conn, cursor = make_conn(rows=[embed_row(1, [0.1, 0.2])])
        embedder = make_embedder(conn)
        with patch("oracledb.defaults"):
            embedder.embed_documents(["hello"])
        cursor.close.assert_called_once()

    def test_cursor_closed_on_failure(self):
        conn, cursor = make_conn(execute_side_effect=Exception("DB error"))
        embedder = make_embedder(conn)
        with patch("oracledb.defaults"):
            with pytest.raises(Exception):
                embedder.embed_documents(["hello"])
        cursor.close.assert_called_once()

    def test_reraises_db_exception(self):
        conn, _ = make_conn(execute_side_effect=RuntimeError("ORA-12345"))
        embedder = make_embedder(conn)
        with patch("oracledb.defaults"):
            with pytest.raises(RuntimeError, match="ORA-12345"):
                embedder.embed_documents(["hello"])


# ===========================================================================
# embed_query
# ===========================================================================


class TestEmbedQuery:
    def test_returns_single_list(self):
        vector = [0.5, 0.6, 0.7]
        conn, _ = make_conn(rows=[embed_row(1, vector)])
        embedder = make_embedder(conn)
        with patch("oracledb.defaults"):
            result = embedder.embed_query("hello world")
        assert isinstance(result, list)
        assert result == pytest.approx(vector)

    def test_delegates_to_embed_documents_with_single_element_list(self):
        """embed_query must call embed_documents(["text"]) and return result[0]."""
        conn, cursor = make_conn()
        embedder = make_embedder(conn)
        with patch.object(
            OracleEmbeddings, "embed_documents", return_value=[[0.1, 0.2]]
        ) as mock_embed:
            result = embedder.embed_query("test input")
        mock_embed.assert_called_once_with(["test input"])
        assert result == [0.1, 0.2]

    def test_returns_first_element_of_embed_documents_result(self):
        """embed_query is embed_documents([text])[0] — must unwrap correctly."""
        vector = [0.9, 0.8, 0.7]
        conn, _ = make_conn(rows=[embed_row(1, vector)])
        embedder = make_embedder(conn)
        with patch("oracledb.defaults"):
            result = embedder.embed_query("hello")
        assert result == pytest.approx(vector)

    def test_reraises_exception_from_embed_documents(self):
        conn, _ = make_conn(execute_side_effect=RuntimeError("fail"))
        embedder = make_embedder(conn)
        with patch("oracledb.defaults"):
            with pytest.raises(RuntimeError):
                embedder.embed_query("hello")
