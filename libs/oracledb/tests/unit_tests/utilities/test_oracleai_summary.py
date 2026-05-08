# Copyright (c) 2024, 2026, Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""
unit_tests/utilities/test_oracleai_summary.py

Unit tests for OracleSummary, exercising all input type branches and the
validation layer without any real database connection. All DB interactions
are intercepted via mock conn and cursor objects at exact call sites.

Covers:
- Constructor attribute storage and defaults (conn, params, proxy)
- get_summary(None) -> empty list with no DB calls
- get_summary(str) SQL content, bind variables, CLOB handling, and cursor cleanup
- get_summary(Document) page_content extraction, CLOB handling, and cursor cleanup
- get_summary(List[str]) executemany path, row structure, and cursor cleanup
- get_summary(List[Document]) page_content extraction in batch mode
- Invalid top-level types raising (int, dict, float) with cursor cleanup
- Invalid items inside lists raising (int, dict mixed with valid items)
- Proxy path triggering utl_http execute and correct proxy value forwarding
- Edge cases (empty string input, single-item list, empty list)
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from langchain_oracledb.utilities.oracleai import OracleSummary

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_conn():
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value = cursor
    return conn, cursor


def make_clob_var(value):
    var = MagicMock()
    var.getvalue.return_value = value
    return var


def setup_list_cursor(cursor, values):
    """Configure a cursor var for executemany list mode."""
    var = MagicMock()
    var.actual_elements = len(values)
    var.getvalue.side_effect = lambda i: values[i]
    cursor.var.return_value = var
    return var


# ===========================================================================
# Constructor
# ===========================================================================


class TestConstructor:
    def test_conn_stored(self):
        conn, _ = make_conn()
        s = OracleSummary(conn=conn, params={"provider": "database"})
        assert s.conn is conn

    def test_params_stored(self):
        conn, _ = make_conn()
        params = {"provider": "database", "glevel": "S"}
        s = OracleSummary(conn=conn, params=params)
        assert s.summary_params == params

    def test_proxy_defaults_to_none(self):
        conn, _ = make_conn()
        s = OracleSummary(conn=conn, params={})
        assert s.proxy is None

    def test_proxy_stored(self):
        conn, _ = make_conn()
        s = OracleSummary(conn=conn, params={}, proxy="http://proxy:80")
        assert s.proxy == "http://proxy:80"


# ===========================================================================
# get_summary — None input
# ===========================================================================


class TestGetSummaryNone:
    def test_none_returns_empty_list(self):
        conn, cursor = make_conn()
        s = OracleSummary(conn=conn, params={"provider": "database"})
        assert s.get_summary(None) == []

    def test_none_makes_no_db_call(self):
        conn, cursor = make_conn()
        s = OracleSummary(conn=conn, params={})
        s.get_summary(None)
        cursor.execute.assert_not_called()
        cursor.executemany.assert_not_called()


# ===========================================================================
# get_summary — str input
# ===========================================================================


class TestGetSummaryStr:
    def test_str_returns_single_item_list(self):
        conn, cursor = make_conn()
        cursor.var.return_value = make_clob_var("A summary.")
        s = OracleSummary(conn=conn, params={"provider": "database"})
        result = s.get_summary("Some text.")
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] == "A summary."

    def test_str_none_clob_yields_empty_string(self):
        conn, cursor = make_conn()
        cursor.var.return_value = make_clob_var(None)
        s = OracleSummary(conn=conn, params={})
        assert s.get_summary("text") == [""]

    def test_str_sql_contains_utl_to_summary(self):
        conn, cursor = make_conn()
        cursor.var.return_value = make_clob_var("ok")
        s = OracleSummary(conn=conn, params={})
        s.get_summary("input")
        assert "utl_to_summary" in cursor.execute.call_args.args[0]

    def test_str_data_kwarg_matches_input(self):
        conn, cursor = make_conn()
        cursor.var.return_value = make_clob_var("ok")
        s = OracleSummary(conn=conn, params={})
        s.get_summary("my input text")
        assert cursor.execute.call_args.kwargs["data"] == "my input text"

    def test_str_params_kwarg_is_json_of_summary_params(self):
        conn, cursor = make_conn()
        cursor.var.return_value = make_clob_var("ok")
        params = {"provider": "database", "glevel": "S"}
        s = OracleSummary(conn=conn, params=params)
        s.get_summary("text")
        assert cursor.execute.call_args.kwargs["params"] == json.dumps(params)

    def test_str_cursor_closed_on_success(self):
        conn, cursor = make_conn()
        cursor.var.return_value = make_clob_var("ok")
        s = OracleSummary(conn=conn, params={})
        s.get_summary("text")
        cursor.close.assert_called_once()

    def test_str_cursor_closed_and_reraises_on_db_error(self):
        conn, cursor = make_conn()
        cursor.var.return_value = MagicMock()
        cursor.execute.side_effect = RuntimeError("ORA-12345")
        s = OracleSummary(conn=conn, params={})
        with pytest.raises(RuntimeError, match="ORA-12345"):
            s.get_summary("text")
        cursor.close.assert_called_once()


# ===========================================================================
# get_summary — Document input
# ===========================================================================


class TestGetSummaryDocument:
    def test_document_returns_single_item_list(self):
        conn, cursor = make_conn()
        cursor.var.return_value = make_clob_var("Doc summary.")
        s = OracleSummary(conn=conn, params={})
        result = s.get_summary(Document(page_content="Full text."))
        assert len(result) == 1
        assert result[0] == "Doc summary."

    def test_document_uses_page_content_as_data(self):
        conn, cursor = make_conn()
        cursor.var.return_value = make_clob_var("ok")
        s = OracleSummary(conn=conn, params={})
        s.get_summary(Document(page_content="the content"))
        assert cursor.execute.call_args.kwargs["data"] == "the content"

    def test_document_none_clob_yields_empty_string(self):
        conn, cursor = make_conn()
        cursor.var.return_value = make_clob_var(None)
        s = OracleSummary(conn=conn, params={})
        assert s.get_summary(Document(page_content="text")) == [""]

    def test_document_cursor_closed_on_success(self):
        conn, cursor = make_conn()
        cursor.var.return_value = make_clob_var("ok")
        s = OracleSummary(conn=conn, params={})
        s.get_summary(Document(page_content="text"))
        cursor.close.assert_called_once()

    def test_document_cursor_closed_and_reraises_on_db_error(self):
        conn, cursor = make_conn()
        cursor.var.return_value = MagicMock()
        cursor.execute.side_effect = RuntimeError("ORA-99999")
        s = OracleSummary(conn=conn, params={})
        with pytest.raises(RuntimeError, match="ORA-99999"):
            s.get_summary(Document(page_content="text"))
        cursor.close.assert_called_once()


# ===========================================================================
# get_summary — List[str] input
# ===========================================================================


class TestGetSummaryListStr:
    def test_list_str_returns_one_summary_per_item(self):
        """Each input string should map to its own summary result."""
        conn, cursor = make_conn()
        setup_list_cursor(cursor, ["Sum A", "Sum B"])
        s = OracleSummary(conn=conn, params={})
        result = s.get_summary(["text A", "text B"])
        assert len(result) == 2
        assert result[0] == "Sum A"
        assert result[1] == "Sum B"

    def test_list_str_none_values_become_empty_string(self):
        """Each list slot should preserve its own NULL-to-empty conversion."""
        conn, cursor = make_conn()
        setup_list_cursor(cursor, [None, "ok"])
        s = OracleSummary(conn=conn, params={})
        result = s.get_summary(["t1", "t2"])
        assert result[0] == ""
        assert result[1] == "ok"

    def test_list_str_uses_executemany(self):
        conn, cursor = make_conn()
        setup_list_cursor(cursor, ["s1"])
        s = OracleSummary(conn=conn, params={})
        s.get_summary(["text"])
        cursor.executemany.assert_called_once()

    def test_list_str_sql_contains_utl_to_summary(self):
        conn, cursor = make_conn()
        setup_list_cursor(cursor, ["s1", "s2"])
        s = OracleSummary(conn=conn, params={})
        s.get_summary(["a", "b"])
        assert "utl_to_summary" in cursor.executemany.call_args.args[0]

    def test_list_str_rows_contain_text_and_params_json(self):
        conn, cursor = make_conn()
        setup_list_cursor(cursor, ["s1", "s2"])
        params = {"provider": "database", "glevel": "S"}
        s = OracleSummary(conn=conn, params=params)
        s.get_summary(["text one", "text two"])
        rows = cursor.executemany.call_args.args[1]
        assert rows[0][0] == "text one"
        assert rows[0][1] == json.dumps(params)
        assert rows[1][0] == "text two"

    def test_list_str_cursor_closed_on_success(self):
        conn, cursor = make_conn()
        setup_list_cursor(cursor, ["s1"])
        s = OracleSummary(conn=conn, params={})
        s.get_summary(["text"])
        cursor.close.assert_called_once()

    def test_list_str_cursor_closed_and_reraises_on_db_error(self):
        conn, cursor = make_conn()
        cursor.var.return_value = MagicMock()
        cursor.executemany.side_effect = RuntimeError("ORA-DB-ERR")
        s = OracleSummary(conn=conn, params={})
        with pytest.raises(RuntimeError, match="ORA-DB-ERR"):
            s.get_summary(["text"])
        cursor.close.assert_called_once()


# ===========================================================================
# get_summary — List[Document] input
# ===========================================================================


class TestGetSummaryListDocument:
    def test_list_document_returns_one_summary_per_doc(self):
        """Each input Document should map to its own summary result."""
        conn, cursor = make_conn()
        setup_list_cursor(cursor, ["Sum1", "Sum2"])
        s = OracleSummary(conn=conn, params={})
        docs = [Document(page_content="doc one"), Document(page_content="doc two")]
        result = s.get_summary(docs)
        assert len(result) == 2
        assert result[0] == "Sum1"
        assert result[1] == "Sum2"

    def test_list_document_uses_page_content_in_rows(self):
        conn, cursor = make_conn()
        setup_list_cursor(cursor, ["s1"])
        s = OracleSummary(conn=conn, params={})
        s.get_summary([Document(page_content="the content")])
        rows = cursor.executemany.call_args.args[1]
        assert rows[0][0] == "the content"

    def test_list_invalid_item_type_raises(self):
        """
        A list containing a non-str/non-Document item must raise 'Invalid input type'.
        """
        conn, cursor = make_conn()
        var = MagicMock()
        var.actual_elements = 0
        cursor.var.return_value = var
        s = OracleSummary(conn=conn, params={})
        with pytest.raises(Exception, match="Invalid input type"):
            s.get_summary(["valid", 42])

    def test_list_dict_item_raises(self):
        conn, cursor = make_conn()
        var = MagicMock()
        var.actual_elements = 0
        cursor.var.return_value = var
        s = OracleSummary(conn=conn, params={})
        with pytest.raises(Exception, match="Invalid input type"):
            s.get_summary([{"not": "valid"}])


# ===========================================================================
# get_summary — invalid top-level types
# ===========================================================================


class TestGetSummaryInvalidTopLevelType:
    def test_int_raises_invalid_input(self):
        conn, _ = make_conn()
        s = OracleSummary(conn=conn, params={})
        with pytest.raises(Exception, match="Invalid input type"):
            s.get_summary(42)

    def test_dict_raises_invalid_input(self):
        conn, _ = make_conn()
        s = OracleSummary(conn=conn, params={})
        with pytest.raises(Exception, match="Invalid input type"):
            s.get_summary({"text": "value"})

    def test_float_raises_invalid_input(self):
        conn, _ = make_conn()
        s = OracleSummary(conn=conn, params={})
        with pytest.raises(Exception, match="Invalid input type"):
            s.get_summary(3.14)

    def test_invalid_type_cursor_closed(self):
        """Cursor must be closed even when top-level type is invalid."""
        conn, cursor = make_conn()
        s = OracleSummary(conn=conn, params={})
        with pytest.raises(Exception):
            s.get_summary(42)
        cursor.close.assert_called_once()


# ===========================================================================
# get_summary — proxy
# ===========================================================================


class TestGetSummaryProxy:
    def test_proxy_calls_utl_http_set_proxy(self):
        conn, cursor = make_conn()
        cursor.var.return_value = make_clob_var("summary")
        s = OracleSummary(conn=conn, params={}, proxy="http://myproxy:80")
        s.get_summary("text")
        calls = [str(c) for c in cursor.execute.call_args_list]
        assert any("utl_http" in c for c in calls)

    def test_proxy_value_passed_correctly(self):
        conn, cursor = make_conn()
        cursor.var.return_value = make_clob_var("summary")
        s = OracleSummary(conn=conn, params={}, proxy="http://myproxy:80")
        s.get_summary("text")
        first_call = cursor.execute.call_args_list[0]
        assert first_call.kwargs.get("proxy") == "http://myproxy:80"

    def test_no_proxy_does_not_call_utl_http(self):
        conn, cursor = make_conn()
        cursor.var.return_value = make_clob_var("summary")
        s = OracleSummary(conn=conn, params={}, proxy=None)
        s.get_summary("text")
        calls = [str(c) for c in cursor.execute.call_args_list]
        assert not any("utl_http" in c for c in calls)


# ===========================================================================
# get_summary — edge cases
# ===========================================================================


class TestGetSummaryEdgeCases:
    def test_empty_list_returns_empty_list(self):
        """Empty list should short-circuit to an empty result without DB calls."""
        conn, cursor = make_conn()
        s = OracleSummary(conn=conn, params={})
        assert s.get_summary([]) == []
        cursor.execute.assert_not_called()
        cursor.executemany.assert_not_called()

    def test_single_item_list_returns_one_result(self):
        conn, cursor = make_conn()
        var = MagicMock()
        var.actual_elements = 1
        var.getvalue.return_value = "single summary"
        cursor.var.return_value = var
        s = OracleSummary(conn=conn, params={})
        result = s.get_summary(["one item"])
        assert len(result) == 1
        assert result[0] == "single summary"

    def test_empty_string_input_is_valid(self):
        """Empty string is still a valid str input — not None."""
        conn, cursor = make_conn()
        cursor.var.return_value = make_clob_var("")
        s = OracleSummary(conn=conn, params={})
        result = s.get_summary("")
        assert isinstance(result, list)
        assert len(result) == 1
