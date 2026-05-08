# Copyright (c) 2024, 2026, Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""Unit tests for OracleAutonomousDatabaseLoader with patched DB calls.

Unit tests for OracleAutonomousDatabaseLoader, exercising the full loader
contract without any real database connection. All calls to oracledb.connect
are patched at the module level.

Run:
    pytest tests/unit_tests/document_loaders/test_oracleadb_loader.py

Authors:
    - Diego Ascencio (diegoascencioqa)
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from langchain_oracledb.document_loaders.oracleadb_loader import (
    OracleAutonomousDatabaseLoader,
)

# Patch the oracledb module exactly where the loader imports it.
# The loader uses its own local reference to `oracledb`, so patching the
# global package would have no effect. This path ensures all DB calls
# inside the loader are replaced with our mock during tests.
PATCH_TARGET = "langchain_oracledb.document_loaders.oracleadb_loader.oracledb"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_loader(
    query="SELECT * FROM T",
    user=None,
    password=None,
    dsn=None,
    schema=None,
    config_dir=None,
    wallet_location=None,
    wallet_password=None,
    metadata=None,
    parameter=None,
):
    if user is None:
        user = f"user_{uuid.uuid4().hex[:8]}"
    if password is None:
        password = uuid.uuid4().hex
    if dsn is None:
        dsn = f"db_{uuid.uuid4().hex[:8]}"
    return OracleAutonomousDatabaseLoader(
        query=query,
        user=user,
        password=password,
        dsn=dsn,
        schema=schema,
        config_dir=config_dir,
        wallet_location=wallet_location,
        wallet_password=wallet_password,
        metadata=metadata,
        parameter=parameter,
    )


def make_mock_oracledb(rows, columns):
    """Return a patched oracledb module with a mock connection and cursor."""
    mock_oracledb = MagicMock()

    cursor = MagicMock()
    cursor.description = [(col,) for col in columns]
    cursor.fetchall.return_value = rows

    connection = MagicMock()
    connection.cursor.return_value = cursor
    mock_oracledb.connect.return_value = connection

    # LOB is never used in happy paths — make isinstance(x, oracledb.LOB) False
    mock_oracledb.LOB = type("LOB", (), {})

    return mock_oracledb, connection, cursor


# ===========================================================================
# Constructor
# ===========================================================================


class TestConstructor:
    def test_query_stored(self):
        loader = make_loader(query="SELECT 1 FROM DUAL")
        assert loader.query == "SELECT 1 FROM DUAL"

    def test_user_stored(self):
        db_user = f"user_{uuid.uuid4().hex[:8]}"
        loader = make_loader(user=db_user)
        assert loader.user == db_user

    def test_password_stored(self):
        auth_value = uuid.uuid4().hex
        loader = make_loader(password=auth_value)
        assert loader.password == auth_value

    def test_dsn_stored(self):
        db_dsn = f"db_{uuid.uuid4().hex[:8]}"
        loader = make_loader(dsn=db_dsn)
        assert loader.dsn == db_dsn

    def test_schema_defaults_to_none(self):
        loader = make_loader()
        assert loader.schema is None

    def test_schema_stored(self):
        loader = make_loader(schema="HR")
        assert loader.schema == "HR"

    def test_config_dir_defaults_to_none(self):
        loader = make_loader()
        assert loader.config_dir is None

    def test_config_dir_stored(self):
        loader = make_loader(config_dir="/path/to/config")
        assert loader.config_dir == "/path/to/config"

    def test_wallet_location_defaults_to_none(self):
        loader = make_loader()
        assert loader.wallet_location is None

    def test_wallet_password_defaults_to_none(self):
        loader = make_loader()
        assert loader.wallet_password is None

    def test_wallet_params_stored(self):
        loader = make_loader(wallet_location="/wallet", wallet_password="wpwd")
        assert loader.wallet_location == "/wallet"
        assert loader.wallet_password == "wpwd"

    def test_metadata_defaults_to_none(self):
        loader = make_loader()
        assert loader.metadata is None

    def test_metadata_stored(self):
        loader = make_loader(metadata=["ID", "NAME"])
        assert loader.metadata == ["ID", "NAME"]

    def test_parameter_defaults_to_none(self):
        loader = make_loader()
        assert loader.parameter is None

    def test_parameter_stored(self):
        loader = make_loader(parameter={"id": 42})
        assert loader.parameter == {"id": 42}


# ===========================================================================
# _run_query — connection parameters
# ===========================================================================


class TestRunQueryConnection:
    def test_connects_with_user_password_dsn(self):
        db_user = f"user_{uuid.uuid4().hex[:8]}"
        auth_value = uuid.uuid4().hex
        db_dsn = f"db_{uuid.uuid4().hex[:8]}"
        loader = make_loader(user=db_user, password=auth_value, dsn=db_dsn)
        mock_db, conn, cursor = make_mock_oracledb([], [])
        with patch(PATCH_TARGET, mock_db):
            loader._run_query()
        mock_db.connect.assert_called_once()
        call_kwargs = mock_db.connect.call_args.kwargs
        assert call_kwargs["user"] == db_user
        assert call_kwargs["password"] == auth_value
        assert call_kwargs["dsn"] == db_dsn

    def test_config_dir_added_when_set(self):
        loader = make_loader(config_dir="/etc/oracle/config")
        mock_db, conn, cursor = make_mock_oracledb([], [])
        with patch(PATCH_TARGET, mock_db):
            loader._run_query()
        assert mock_db.connect.call_args.kwargs["config_dir"] == "/etc/oracle/config"

    def test_config_dir_omitted_when_none(self):
        loader = make_loader(config_dir=None)
        mock_db, conn, cursor = make_mock_oracledb([], [])
        with patch(PATCH_TARGET, mock_db):
            loader._run_query()
        assert "config_dir" not in mock_db.connect.call_args.kwargs

    def test_wallet_params_added_when_both_set(self):
        loader = make_loader(wallet_location="/wallet", wallet_password="wpwd")
        mock_db, conn, cursor = make_mock_oracledb([], [])
        with patch(PATCH_TARGET, mock_db):
            loader._run_query()
        kw = mock_db.connect.call_args.kwargs
        assert kw["wallet_location"] == "/wallet"
        assert kw["wallet_password"] == "wpwd"

    def test_wallet_params_omitted_when_missing(self):
        """wallet_location without wallet_password — neither should be passed."""
        loader = make_loader(wallet_location="/wallet", wallet_password=None)
        mock_db, conn, cursor = make_mock_oracledb([], [])
        with patch(PATCH_TARGET, mock_db):
            loader._run_query()
        kw = mock_db.connect.call_args.kwargs
        assert "wallet_location" not in kw
        assert "wallet_password" not in kw

    def test_schema_set_on_connection(self):
        loader = make_loader(schema="HR")
        mock_db, conn, cursor = make_mock_oracledb([], [])
        with patch(PATCH_TARGET, mock_db):
            loader._run_query()
        assert conn.current_schema == "HR"

    def test_schema_not_set_when_none(self):
        loader = make_loader(schema=None)
        mock_db, conn, cursor = make_mock_oracledb([], [])
        with patch(PATCH_TARGET, mock_db):
            loader._run_query()
        assert "current_schema" not in conn._mock_children

    def test_cursor_closed_after_query(self):
        loader = make_loader()
        mock_db, conn, cursor = make_mock_oracledb([], [])
        with patch(PATCH_TARGET, mock_db):
            loader._run_query()
        cursor.close.assert_called_once()

    def test_connection_closed_after_query(self):
        loader = make_loader()
        mock_db, conn, cursor = make_mock_oracledb([], [])
        with patch(PATCH_TARGET, mock_db):
            loader._run_query()
        conn.close.assert_called_once()


# ===========================================================================
# _run_query — query execution
# ===========================================================================


class TestRunQueryExecution:
    def test_executes_query_without_params(self):
        loader = make_loader(query="SELECT * FROM T", parameter=None)
        mock_db, conn, cursor = make_mock_oracledb([], [])
        with patch(PATCH_TARGET, mock_db):
            loader._run_query()
        cursor.execute.assert_called_once_with("SELECT * FROM T")

    def test_executes_query_with_params(self):
        loader = make_loader(query="SELECT * FROM T WHERE ID=:id", parameter={"id": 1})
        mock_db, conn, cursor = make_mock_oracledb([], [])
        with patch(PATCH_TARGET, mock_db):
            loader._run_query()
        cursor.execute.assert_called_once_with(
            "SELECT * FROM T WHERE ID=:id", {"id": 1}
        )

    def test_executes_query_with_list_params(self):
        loader = make_loader(query="SELECT * FROM T WHERE ID=:1", parameter=[42])
        mock_db, conn, cursor = make_mock_oracledb([], [])
        with patch(PATCH_TARGET, mock_db):
            loader._run_query()
        cursor.execute.assert_called_once_with("SELECT * FROM T WHERE ID=:1", [42])

    def test_returns_empty_list_on_db_error(self):
        """When connect() fails, the connection error must propagate."""
        loader = make_loader()
        mock_db = MagicMock()
        mock_db.DatabaseError = Exception
        mock_db.LOB = type("LOB", (), {})
        # Simulate connect() failing with a real Oracle error
        mock_db.connect.side_effect = Exception("ORA-01017: invalid credentials")
        with patch(PATCH_TARGET, mock_db):
            with pytest.raises(Exception, match="ORA-01017"):
                loader._run_query()

    def test_returns_list_of_dicts(self):
        loader = make_loader()
        mock_db, conn, cursor = make_mock_oracledb(
            rows=[("Alice", 30), ("Bob", 25)],
            columns=["NAME", "AGE"],
        )
        with patch(PATCH_TARGET, mock_db):
            result = loader._run_query()
        assert result == [
            {"NAME": "Alice", "AGE": 30},
            {"NAME": "Bob", "AGE": 25},
        ]

    def test_empty_result_returns_empty_list(self):
        loader = make_loader()
        mock_db, conn, cursor = make_mock_oracledb(rows=[], columns=["ID"])
        with patch(PATCH_TARGET, mock_db):
            result = loader._run_query()
        assert result == []

    def test_no_description_returns_empty_columns(self):
        """cursor.description is None when no rows are returned by some drivers."""
        loader = make_loader()
        mock_db, conn, cursor = make_mock_oracledb(rows=[], columns=[])
        cursor.description = None
        cursor.fetchall.return_value = []
        with patch(PATCH_TARGET, mock_db):
            result = loader._run_query()
        assert result == []

    def test_lob_values_are_read(self):
        """LOB column values must be replaced by their .read() content."""
        loader = make_loader()
        mock_db = MagicMock()

        class FakeLOB:
            def read(self):
                return "lob content"

        mock_db.LOB = FakeLOB
        mock_db.DatabaseError = Exception

        lob_val = FakeLOB()
        cursor = MagicMock()
        cursor.description = [("TEXT",)]
        cursor.fetchall.return_value = [(lob_val,)]
        conn = MagicMock()
        conn.cursor.return_value = cursor
        mock_db.connect.return_value = conn

        with patch(PATCH_TARGET, mock_db):
            result = loader._run_query()
        assert result == [{"TEXT": "lob content"}]


# ===========================================================================
# load
# ===========================================================================


class TestLoad:
    def test_returns_list_of_documents(self):
        loader = make_loader()
        mock_db, conn, cursor = make_mock_oracledb(rows=[("Alice",)], columns=["NAME"])
        with patch(PATCH_TARGET, mock_db):
            docs = loader.load()
        assert isinstance(docs, list)
        assert all(isinstance(d, Document) for d in docs)

    def test_one_document_per_row(self):
        loader = make_loader()
        mock_db, conn, cursor = make_mock_oracledb(
            rows=[("Alice",), ("Bob",), ("Carol",)], columns=["NAME"]
        )
        with patch(PATCH_TARGET, mock_db):
            docs = loader.load()
        assert len(docs) == 3

    def test_page_content_is_str_of_row(self):
        """page_content must be str(row_dict) — documents the exact behaviour."""
        loader = make_loader()
        mock_db, conn, cursor = make_mock_oracledb(
            rows=[("Alice", 30)], columns=["NAME", "AGE"]
        )
        with patch(PATCH_TARGET, mock_db):
            docs = loader.load()
        assert docs[0].page_content == str({"NAME": "Alice", "AGE": 30})

    def test_metadata_columns_extracted(self):
        """Columns listed in self.metadata must appear in doc.metadata."""
        loader = make_loader(metadata=["ID"])
        mock_db, conn, cursor = make_mock_oracledb(
            rows=[(1, "Alice")], columns=["ID", "NAME"]
        )
        with patch(PATCH_TARGET, mock_db):
            docs = loader.load()
        assert docs[0].metadata == {"ID": 1}

    def test_non_metadata_columns_not_in_metadata(self):
        """Columns not listed in self.metadata must not appear in doc.metadata."""
        loader = make_loader(metadata=["ID"])
        mock_db, conn, cursor = make_mock_oracledb(
            rows=[(1, "Alice")], columns=["ID", "NAME"]
        )
        with patch(PATCH_TARGET, mock_db):
            docs = loader.load()
        assert "NAME" not in docs[0].metadata

    def test_empty_metadata_when_none(self):
        """When metadata=None, doc.metadata must be an empty dict."""
        loader = make_loader(metadata=None)
        mock_db, conn, cursor = make_mock_oracledb(
            rows=[(1, "Alice")], columns=["ID", "NAME"]
        )
        with patch(PATCH_TARGET, mock_db):
            docs = loader.load()
        assert docs[0].metadata == {}

    def test_multiple_metadata_columns(self):
        loader = make_loader(metadata=["ID", "DEPT"])
        mock_db, conn, cursor = make_mock_oracledb(
            rows=[(1, "Alice", "ENG")], columns=["ID", "NAME", "DEPT"]
        )
        with patch(PATCH_TARGET, mock_db):
            docs = loader.load()
        assert docs[0].metadata == {"ID": 1, "DEPT": "ENG"}

    def test_empty_result_returns_empty_list(self):
        loader = make_loader()
        mock_db, conn, cursor = make_mock_oracledb(rows=[], columns=["ID"])
        with patch(PATCH_TARGET, mock_db):
            docs = loader.load()
        assert docs == []

    def test_db_error_returns_empty_list(self):
        """When connect() fails, the connection error must propagate."""
        loader = make_loader()
        mock_db = MagicMock()
        mock_db.DatabaseError = Exception
        mock_db.LOB = type("LOB", (), {})
        # Simulate connect() failing with a real Oracle error
        mock_db.connect.side_effect = Exception("ORA-12541: no listener")
        with patch(PATCH_TARGET, mock_db):
            with pytest.raises(Exception, match="ORA-12541"):
                loader.load()
