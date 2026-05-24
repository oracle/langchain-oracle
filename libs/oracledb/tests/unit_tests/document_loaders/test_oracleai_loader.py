# Copyright (c) 2024, 2026, Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""
unit_tests/document_loaders/test_oracleai_loader.py

Unit tests for OracleDocLoader, OracleTextSplitter, OracleDocReader and
ParseOracleDocMetadata, exercising the full loader contract without any real
database connection. All DB interactions are intercepted via mock conn and
cursor objects at exact call sites.

Covers:
- ParseOracleDocMetadata HTML parsing (meta name/content, title tag, edge cases)
- OracleDocReader.generate_object_id format and uniqueness
- OracleDocReader.read_file — page_content, metadata (_oid, _file), cursor cleanup,
  HTML metadata extraction, error handling (file not found, DB error)
- OracleDocLoader constructor (conn stored, params deep-copied)
- OracleDocLoader.load — file mode (single file, None result skipped)
- OracleDocLoader.load — dir mode (empty dir, valid file, None skipped,
  nonexistent dir raises, subdirs ignored)
- OracleDocLoader.load — table mode (missing owner/colname raises validation
  error, normal rows, None rows, rowid in metadata, HTML metadata parsed,
  cursor closed on error)
- OracleDocLoader.load — mdata_cols (limit exceeded raises, columns in SELECT,
  unsupported type raises, projected metadata mapped correctly)
- OracleTextSplitter — params stored, split_text SQL content, bind variables,
  setinputsizes, multiple chunks, empty text, DB error reraised

Run:
    pytest tests/unit_tests/document_loaders/test_oracleai_loader.py

Authors:
    - Diego Ascencio (diegoascencioqa)
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import oracledb
import pytest
from langchain_core.documents import Document

from langchain_oracledb.document_loaders.oracleai import (
    OracleDocLoader,
    OracleDocReader,
    OracleTextSplitter,
    ParseOracleDocMetadata,
    _quote_identifier,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def make_conn(username: str = "vector_user") -> tuple:
    """Return (mock_conn, mock_cursor). cursor is what conn.cursor() returns."""
    conn = MagicMock()
    conn.username = username
    cursor = MagicMock()
    conn.cursor.return_value = cursor
    return conn, cursor


# ===========================================================================
# ParseOracleDocMetadata
# ===========================================================================


class TestParseOracleDocMetadata:
    def test_empty_html_yields_empty_dict(self):
        parser = ParseOracleDocMetadata()
        parser.feed("<html></html>")
        assert parser.get_metadata() == {}

    def test_single_meta_name_content(self):
        parser = ParseOracleDocMetadata()
        parser.feed('<meta name="author" content="Alice">')
        assert parser.get_metadata()["author"] == "Alice"

    def test_multiple_meta_tags(self):
        parser = ParseOracleDocMetadata()
        parser.feed(
            '<meta name="author" content="Alice"><meta name="subject" content="AI">'
        )
        metadata = parser.get_metadata()
        assert metadata["author"] == "Alice"
        assert metadata["subject"] == "AI"

    def test_title_tag_captured(self):
        parser = ParseOracleDocMetadata()
        parser.feed("<html><head><title>My Doc</title></head></html>")
        assert parser.get_metadata()["title"] == "My Doc"

    def test_title_and_meta_together(self):
        parser = ParseOracleDocMetadata()
        parser.feed(
            '<html><head><title>T</title><meta name="kw" content="v"></head></html>'
        )
        metadata = parser.get_metadata()
        assert metadata["title"] == "T"
        assert metadata["kw"] == "v"

    def test_meta_without_name_ignored(self):
        parser = ParseOracleDocMetadata()
        parser.feed('<meta content="orphan">')
        assert parser.get_metadata() == {}

    def test_meta_without_content_ignored(self):
        parser = ParseOracleDocMetadata()
        parser.feed('<meta name="author">')
        assert "author" not in parser.get_metadata()

    def test_match_flag_reset_after_title_data(self):
        parser = ParseOracleDocMetadata()
        parser.feed("<title>First</title>")
        assert parser.match is False

    def test_get_metadata_returns_dict_type(self):
        parser = ParseOracleDocMetadata()
        assert isinstance(parser.get_metadata(), dict)

    def test_doctype_html_prefix_does_not_break_parsing(self):
        parser = ParseOracleDocMetadata()
        html = (
            '<!DOCTYPE html<html><head><meta name="creator" '
            'content="Bob"></head></html>'
        )
        parser.feed(html)
        assert parser.get_metadata().get("creator") == "Bob"


# ===========================================================================
# OracleDocReader.generate_object_id
# ===========================================================================


class TestGenerateObjectId:
    def test_returns_32_char_string(self):
        oid = OracleDocReader.generate_object_id("test")
        assert isinstance(oid, str) and len(oid) == 32

    def test_valid_hex(self):
        oid = OracleDocReader.generate_object_id("hello")
        int(oid, 16)  # raises ValueError if not valid hex

    def test_no_input_still_returns_32_chars(self):
        oid = OracleDocReader.generate_object_id()
        assert len(oid) == 32

    def test_two_calls_with_same_input_both_valid(self):
        """Timestamp + random counter means results may differ,
        but both must be valid 32-char hex strings."""
        oid_a = OracleDocReader.generate_object_id("x")
        oid_b = OracleDocReader.generate_object_id("x")
        assert len(oid_a) == 32 and len(oid_b) == 32


# ===========================================================================
# OracleDocReader.read_file
# ===========================================================================


class TestOracleDocReaderReadFile:
    def setup_cursor_vars(self, mdata_val, text_val):
        conn, cursor = make_conn()
        mdata_var = MagicMock()
        mdata_var.getvalue.return_value = mdata_val
        text_var = MagicMock()
        text_var.getvalue.return_value = text_val
        cursor.var.side_effect = [mdata_var, text_var]
        return conn, cursor

    def fake_open(self, data: bytes):
        file_handle = MagicMock()
        file_handle.__enter__ = MagicMock(return_value=file_handle)
        file_handle.__exit__ = MagicMock(return_value=False)
        file_handle.read.return_value = data
        return file_handle

    def test_returns_document_with_page_content(self):
        conn, cursor = self.setup_cursor_vars(None, "hello world")
        with (
            patch("builtins.open", return_value=self.fake_open(b"bytes")),
            patch("oracledb.defaults"),
        ):
            doc = OracleDocReader.read_file(conn, "/tmp/t.txt", {})
        assert isinstance(doc, Document)
        assert doc.page_content == "hello world"

    def test_metadata_always_has_oid_and_file(self):
        conn, cursor = self.setup_cursor_vars(None, "text")
        with (
            patch("builtins.open", return_value=self.fake_open(b"bytes")),
            patch("oracledb.defaults"),
        ):
            doc = OracleDocReader.read_file(conn, "/tmp/doc.txt", {})
        assert "_oid" in doc.metadata
        assert doc.metadata["_file"] == "/tmp/doc.txt"

    def test_none_text_var_yields_empty_page_content(self):
        conn, cursor = self.setup_cursor_vars(None, None)
        with (
            patch("builtins.open", return_value=self.fake_open(b"bytes")),
            patch("oracledb.defaults"),
        ):
            doc = OracleDocReader.read_file(conn, "/tmp/t.txt", {})
        assert doc.page_content == ""

    def test_empty_file_returns_document_with_empty_content(self):
        conn, cursor = self.setup_cursor_vars(None, None)
        with (
            patch("builtins.open", return_value=self.fake_open(b"")),
            patch("oracledb.defaults"),
        ):
            doc = OracleDocReader.read_file(conn, "/tmp/empty.txt", {})
        assert isinstance(doc, Document)
        assert doc.page_content == ""

    def test_html_mdata_parsed_into_metadata(self):
        html = (
            '<!DOCTYPE html<html><head><meta name="author" content="Eve"></head></html>'
        )
        conn, cursor = self.setup_cursor_vars(html, "body text")
        with (
            patch("builtins.open", return_value=self.fake_open(b"bytes")),
            patch("oracledb.defaults"),
        ):
            doc = OracleDocReader.read_file(conn, "/tmp/d.html", {})
        assert doc.metadata.get("author") == "Eve"

    def test_html_mdata_starting_with_HTML_tag_also_parsed(self):
        html = '<HTML><HEAD><meta name="creator" content="Oracle"></HEAD></HTML>'
        conn, cursor = self.setup_cursor_vars(html, "body text")
        with (
            patch("builtins.open", return_value=self.fake_open(b"bytes")),
            patch("oracledb.defaults"),
        ):
            doc = OracleDocReader.read_file(conn, "/tmp/d.html", {})
        assert doc.metadata.get("creator") == "Oracle"

    def test_cursor_closed_on_success(self):
        conn, cursor = self.setup_cursor_vars(None, "text")
        with (
            patch("builtins.open", return_value=self.fake_open(b"bytes")),
            patch("oracledb.defaults"),
        ):
            OracleDocReader.read_file(conn, "/tmp/t.txt", {})
        cursor.close.assert_called_once()

    def test_oid_seeded_with_username_and_filepath(self):
        conn, cursor = self.setup_cursor_vars(None, "text")
        conn.username = "scott"
        with (
            patch("builtins.open", return_value=self.fake_open(b"bytes")),
            patch("oracledb.defaults"),
            patch.object(
                OracleDocReader, "generate_object_id", return_value="ABC123"
            ) as mock_gen,
        ):
            doc = OracleDocReader.read_file(conn, "/tmp/f.txt", {})
        mock_gen.assert_called_once_with("scott$/tmp/f.txt")
        assert doc.metadata["_oid"] == "ABC123"

    def test_returns_none_on_file_open_error(self):
        conn, cursor = make_conn()
        with (
            patch("builtins.open", side_effect=IOError("permission denied")),
            patch("oracledb.defaults"),
        ):
            doc = OracleDocReader.read_file(conn, "/bad/path.txt", {})
        assert doc is None

    def test_cursor_closed_on_exception(self):
        conn, cursor = make_conn()
        cursor.var.side_effect = Exception("ORA-01234")
        with (
            patch("builtins.open", return_value=self.fake_open(b"bytes")),
            patch("oracledb.defaults"),
        ):
            result = OracleDocReader.read_file(conn, "/tmp/t.txt", {})
        cursor.close.assert_called_once()
        assert result is None


# ===========================================================================
# OracleDocLoader — constructor
# ===========================================================================


class TestOracleDocLoaderConstructor:
    def test_stores_conn(self):
        conn, cursor = make_conn()
        loader = OracleDocLoader(conn=conn, params={})
        assert loader.conn is conn

    def test_params_deep_copied(self):
        conn, cursor = make_conn()
        params = {"tablename": "T", "owner": "U", "colname": "C"}
        loader = OracleDocLoader(conn=conn, params=params)
        params["tablename"] = "CHANGED"
        assert loader.params["tablename"] == "T"

    def test_accepts_empty_params(self):
        conn, cursor = make_conn()
        loader = OracleDocLoader(conn=conn, params={})
        assert loader is not None


# ===========================================================================
# OracleDocLoader.load — FILE mode
# ===========================================================================


class TestOracleDocLoaderFileMode:
    def test_single_file_returns_one_document(self, tmp_path):
        file_path = tmp_path / "doc.txt"
        file_path.write_text("content")
        expected_doc = Document(
            page_content="content", metadata={"_oid": "x", "_file": str(file_path)}
        )
        conn, cursor = make_conn()
        loader = OracleDocLoader(conn=conn, params={"file": str(file_path)})
        with (
            patch(
                "langchain_oracledb.document_loaders.oracleai.OracleDocReader.read_file",
                return_value=expected_doc,
            ),
            patch("oracledb.defaults"),
        ):
            docs = loader.load()
        assert len(docs) == 1
        assert docs[0].page_content == "content"

    def test_returns_empty_when_read_file_returns_none(self, tmp_path):
        file_path = tmp_path / "doc.txt"
        file_path.write_text("x")
        conn, cursor = make_conn()
        loader = OracleDocLoader(conn=conn, params={"file": str(file_path)})
        with (
            patch(
                "langchain_oracledb.document_loaders.oracleai.OracleDocReader.read_file",
                return_value=None,
            ),
            patch("oracledb.defaults"),
        ):
            docs = loader.load()
        assert docs == []


# ===========================================================================
# OracleDocLoader.load — DIR mode
# ===========================================================================


class TestOracleDocLoaderDirMode:
    def test_empty_directory_returns_empty_list(self, tmp_path):
        conn, cursor = make_conn()
        loader = OracleDocLoader(conn=conn, params={"dir": str(tmp_path)})
        with (
            patch(
                "langchain_oracledb.document_loaders.oracleai.OracleDocReader.read_file",
                return_value=None,
            ),
            patch("oracledb.defaults"),
        ):
            docs = loader.load()
        assert docs == []

    def test_one_valid_file_produces_one_document(self, tmp_path):
        (tmp_path / "a.txt").write_text("hello")
        expected_doc = Document(
            page_content="hello", metadata={"_oid": "y", "_file": "a.txt"}
        )
        conn, cursor = make_conn()
        loader = OracleDocLoader(conn=conn, params={"dir": str(tmp_path)})
        with (
            patch(
                "langchain_oracledb.document_loaders.oracleai.OracleDocReader.read_file",
                return_value=expected_doc,
            ),
            patch("oracledb.defaults"),
        ):
            docs = loader.load()
        assert len(docs) == 1

    def test_skips_none_results_from_read_file(self, tmp_path):
        (tmp_path / "a.txt").write_text("x")
        (tmp_path / "b.txt").write_text("y")
        good_doc = Document(page_content="x", metadata={"_oid": "1", "_file": "a.txt"})
        conn, cursor = make_conn()
        loader = OracleDocLoader(conn=conn, params={"dir": str(tmp_path)})
        with (
            patch(
                "langchain_oracledb.document_loaders.oracleai.OracleDocReader.read_file",
                side_effect=[good_doc, None],
            ),
            patch("oracledb.defaults"),
        ):
            docs = loader.load()
        assert len(docs) == 1

    def test_nonexistent_dir_raises(self):
        conn, cursor = make_conn()
        loader = OracleDocLoader(conn=conn, params={"dir": "/no/such/dir/xyz"})
        with patch("oracledb.defaults"):
            with pytest.raises(Exception):
                loader.load()

    def test_only_files_processed_not_subdirs(self, tmp_path):
        (tmp_path / "sub").mkdir()
        (tmp_path / "doc.txt").write_text("text")
        expected_doc = Document(
            page_content="text", metadata={"_oid": "z", "_file": "doc.txt"}
        )
        conn, cursor = make_conn()
        loader = OracleDocLoader(conn=conn, params={"dir": str(tmp_path)})
        with (
            patch(
                "langchain_oracledb.document_loaders.oracleai.OracleDocReader.read_file",
                return_value=expected_doc,
            ) as mock_read_file,
            patch("oracledb.defaults"),
        ):
            loader.load()
        assert mock_read_file.call_count == 1


# ===========================================================================
# OracleDocLoader.load — TABLE mode
# ===========================================================================


class TestOracleDocLoaderTableMode:
    def loader_with_rows(self, rows, mdata_cols=None, fetchall_rows=None):
        conn, cursor = make_conn()
        cursor.__iter__ = MagicMock(return_value=iter(rows))
        # fetchall is used by the loader to look up mdata_col types from Oracle's
        # data dictionary. Only needed when mdata_cols is set; defaults to empty.
        cursor.fetchall.return_value = fetchall_rows or []
        params = {"tablename": "MY_TABLE", "owner": "SCOTT", "colname": "TEXT_COL"}
        if mdata_cols is not None:
            params["mdata_cols"] = mdata_cols
        loader = OracleDocLoader(conn=conn, params=params)
        return loader, cursor

    def test_raises_when_owner_missing(self):
        """Missing owner must raise with 'Missing owner or column name' message."""
        conn, cursor = make_conn()
        loader = OracleDocLoader(conn=conn, params={"tablename": "T", "colname": "C"})
        with patch("oracledb.defaults"):
            with pytest.raises(Exception, match="Missing owner or column name"):
                loader.load()

    def test_raises_when_colname_missing(self):
        """Missing colname must raise with 'Missing owner or column name' message."""
        conn, cursor = make_conn()
        loader = OracleDocLoader(conn=conn, params={"tablename": "T", "owner": "U"})
        with patch("oracledb.defaults"):
            with pytest.raises(Exception, match="Missing owner or column name"):
                loader.load()

    def test_normal_row_produces_document(self):
        rows = [(None, "Hello Oracle", "ROWID001")]
        loader, cursor = self.loader_with_rows(rows)
        with patch("oracledb.defaults"):
            docs = loader.load()
        assert len(docs) == 1
        assert docs[0].page_content == "Hello Oracle"

    def test_multiple_rows_produce_multiple_documents(self):
        rows = [(None, "First", "R1"), (None, "Second", "R2"), (None, "Third", "R3")]
        loader, cursor = self.loader_with_rows(rows)
        with patch("oracledb.defaults"):
            docs = loader.load()
        assert len(docs) == 3
        assert [d.page_content for d in docs] == ["First", "Second", "Third"]

    def test_none_text_yields_empty_page_content(self):
        rows = [(None, None, "R1")]
        loader, cursor = self.loader_with_rows(rows)
        with patch("oracledb.defaults"):
            docs = loader.load()
        assert docs[0].page_content == ""

    def test_none_row_yields_document_with_oid(self):
        rows = [None]
        loader, cursor = self.loader_with_rows(rows)
        with patch("oracledb.defaults"):
            docs = loader.load()
        assert len(docs) == 1
        assert "_oid" in docs[0].metadata
        assert docs[0].page_content == ""

    def test_rowid_stored_in_metadata(self):
        rows = [(None, "text", "ROWID_XYZ")]
        loader, cursor = self.loader_with_rows(rows)
        with patch("oracledb.defaults"):
            docs = loader.load()
        assert docs[0].metadata["_rowid"] == "ROWID_XYZ"

    def test_html_mdata_parsed_from_row(self):
        html = (
            '<!DOCTYPE html<html><head><meta name="author" content="Bob"></head></html>'
        )
        rows = [(html, "plain text", "R1")]
        loader, cursor = self.loader_with_rows(rows)
        with patch("oracledb.defaults"):
            docs = loader.load()
        assert docs[0].metadata.get("author") == "Bob"

    def test_html_uppercase_tag_also_parsed(self):
        html = '<HTML><HEAD><meta name="source" content="DB"></HEAD></HTML>'
        rows = [(html, "text", "R1")]
        loader, cursor = self.loader_with_rows(rows)
        with patch("oracledb.defaults"):
            docs = loader.load()
        assert docs[0].metadata.get("source") == "DB"

    def test_cursor_execute_called(self):
        rows = [(None, "some text", "R1")]
        loader, cursor = self.loader_with_rows(rows)
        with patch("oracledb.defaults"):
            loader.load()
        assert cursor.execute.called

    def test_quoted_table_identifiers_preserve_case_in_select(self):
        conn, cursor = make_conn()
        cursor.__iter__ = MagicMock(return_value=iter([]))
        cursor.fetchall.return_value = []
        loader = OracleDocLoader(
            conn=conn,
            params={
                "tablename": '"MixedTable"',
                "owner": '"MixedOwner"',
                "colname": '"bodyCol"',
            },
        )
        with patch("oracledb.defaults"):
            loader.load()

        sql = cursor.execute.call_args.args[0]
        assert 't."bodyCol"' in sql
        assert 'from "MixedOwner"."MixedTable" t' in sql

    def test_quoted_metadata_column_preserves_case(self):
        conn, cursor = make_conn()
        cursor.__iter__ = MagicMock(return_value=iter([]))
        cursor.fetchall.return_value = [("MetaCol", "VARCHAR2")]
        loader = OracleDocLoader(
            conn=conn,
            params={
                "tablename": '"MixedTable"',
                "owner": '"MixedOwner"',
                "colname": '"bodyCol"',
                "mdata_cols": ['"MetaCol"'],
            },
        )
        with patch("oracledb.defaults"):
            loader.load()

        dictionary_call = cursor.execute.call_args_list[0]
        assert dictionary_call.kwargs["ownername"] == "MixedOwner"
        assert dictionary_call.kwargs["tablename"] == "MixedTable"

        sql = cursor.execute.call_args_list[-1].args[0]
        assert 't."MetaCol"' in sql

    @pytest.mark.parametrize(
        "identifier",
        ["bad name", '"bad name"', "bad,name", "bad)"],
    )
    def test_quote_identifier_rejects_invalid_identifier(self, identifier):
        with pytest.raises(ValueError, match="valid Oracle identifier"):
            _quote_identifier(identifier, "column name")

    def test_cursor_closed_on_db_error(self):
        conn, cursor = make_conn()
        cursor.__iter__ = MagicMock(return_value=iter([]))
        cursor.fetchall.return_value = []
        cursor.execute.side_effect = Exception("ORA-00942: table not found")
        loader = OracleDocLoader(
            conn=conn, params={"tablename": "X", "owner": "U", "colname": "C"}
        )
        with patch("oracledb.defaults"):
            with pytest.raises(Exception, match="ORA-00942"):
                loader.load()
        cursor.close.assert_called()

    def test_mdata_cols_limit_exceeded_raises(self):
        conn, cursor = make_conn()
        cursor.__iter__ = MagicMock(return_value=iter([]))
        cursor.fetchall.return_value = []
        loader = OracleDocLoader(
            conn=conn,
            params={
                "tablename": "T",
                "owner": "U",
                "colname": "C",
                "mdata_cols": ["A", "B", "C", "D"],
            },
        )
        with patch("oracledb.defaults"):
            with pytest.raises(Exception, match="Exceeds the max number"):
                loader.load()

    def test_mdata_cols_included_in_select(self):
        conn, cursor = make_conn()
        cursor.__iter__ = MagicMock(return_value=iter([]))
        cursor.fetchall.return_value = [("TITLE", "VARCHAR2"), ("AUTHOR", "VARCHAR2")]
        loader = OracleDocLoader(
            conn=conn,
            params={
                "tablename": "T",
                "owner": "U",
                "colname": "C",
                "mdata_cols": ["TITLE", "AUTHOR"],
            },
        )
        with patch("oracledb.defaults"):
            loader.load()
        last_sql = str(cursor.execute.call_args_list[-1])
        assert "TITLE" in last_sql
        assert "AUTHOR" in last_sql

    def test_mdata_col_unsupported_type_raises(self):
        conn, cursor = make_conn()
        cursor.__iter__ = MagicMock(return_value=iter([]))
        cursor.fetchall.return_value = [("CONTENT", "BLOB")]
        loader = OracleDocLoader(
            conn=conn,
            params={
                "tablename": "T",
                "owner": "U",
                "colname": "C",
                "mdata_cols": ["CONTENT"],
            },
        )
        with patch("oracledb.defaults"):
            with pytest.raises(Exception, match="datatype.*not supported"):
                loader.load()

    def test_extra_cols_stored_in_metadata(self):
        """Each mdata_col value must appear in metadata keyed by its column name."""
        conn, cursor = make_conn()
        rows = [(None, "body", "R1", "Alice", "Science")]
        cursor.__iter__ = MagicMock(return_value=iter(rows))
        cursor.fetchall.return_value = [("AUTHOR", "VARCHAR2"), ("SUBJECT", "VARCHAR2")]
        loader = OracleDocLoader(
            conn=conn,
            params={
                "tablename": "T",
                "owner": "U",
                "colname": "C",
                "mdata_cols": ["AUTHOR", "SUBJECT"],
            },
        )
        with patch("oracledb.defaults"):
            docs = loader.load()
        assert docs[0].metadata["AUTHOR"] == "Alice"
        assert docs[0].metadata["SUBJECT"] == "Science"


# ===========================================================================
# OracleTextSplitter
# ===========================================================================


class TestOracleTextSplitter:
    def make_splitter(self, fetchone_side_effect, params=None):
        conn, cursor = make_conn()
        cursor.fetchone.side_effect = fetchone_side_effect
        splitter = OracleTextSplitter(
            conn=conn,
            params=params or {"by": "words", "max": "1000"},
        )
        return splitter, cursor

    def chunk_row(self, text: str) -> tuple:
        payload = json.dumps(
            {
                "chunk_id": 1,
                "chunk_offset": 0,
                "chunk_length": len(text),
                "chunk_data": text,
            }
        )
        return (payload,)

    def test_stores_params(self):
        conn, cursor = make_conn()
        params = {"by": "chars", "max": "500"}
        splitter = OracleTextSplitter(conn=conn, params=params)
        assert splitter.params == params

    def test_returns_list_of_strings(self):
        splitter, cursor = self.make_splitter([self.chunk_row("hello"), None])
        with patch("oracledb.defaults"):
            result = splitter.split_text("hello world")
        assert result == ["hello"]

    def test_multiple_chunks_returned(self):
        splitter, cursor = self.make_splitter(
            [self.chunk_row("first"), self.chunk_row("second"), None]
        )
        with patch("oracledb.defaults"):
            result = splitter.split_text("first second")
        assert result == ["first", "second"]

    def test_empty_text_returns_empty_list(self):
        splitter, cursor = self.make_splitter([None])
        with patch("oracledb.defaults"):
            result = splitter.split_text("")
        assert result == []

    def test_setinputsizes_called_with_clob(self):
        splitter, cursor = self.make_splitter([None])
        with patch("oracledb.defaults"):
            splitter.split_text("text")
        cursor.setinputsizes.assert_called_once_with(content=oracledb.CLOB)

    def test_execute_receives_content_and_params_json(self):
        params = {"by": "words", "max": "500"}
        splitter, cursor = self.make_splitter([None], params=params)
        with patch("oracledb.defaults"):
            splitter.split_text("my text")
        call_kwargs = cursor.execute.call_args
        assert call_kwargs.kwargs["content"] == "my text"
        assert call_kwargs.kwargs["params"] == json.dumps(params)

    def test_sql_contains_utl_to_chunks(self):
        splitter, cursor = self.make_splitter([None])
        with patch("oracledb.defaults"):
            splitter.split_text("some text")
        sql = cursor.execute.call_args.args[0]
        assert "utl_to_chunks" in sql

    def test_by_chars_params(self):
        params = {"by": "chars", "max": "4000", "overlap": "800", "split": "NEWLINE"}
        splitter, cursor = self.make_splitter([None], params=params)
        with patch("oracledb.defaults"):
            splitter.split_text("text")
        assert json.dumps(params) in cursor.execute.call_args.kwargs["params"]

    def test_by_vocabulary_params(self):
        params = {
            "by": "vocabulary",
            "vocabulary": "MYVOCAB",
            "max": "200",
            "overlap": "0",
        }
        splitter, cursor = self.make_splitter([None], params=params)
        with patch("oracledb.defaults"):
            splitter.split_text("text")
        assert cursor.execute.called

    def test_db_exception_is_reraised(self):
        conn, cursor = make_conn()
        cursor.execute.side_effect = Exception("ORA-30584: invalid chunking param")
        splitter = OracleTextSplitter(conn=conn, params={"by": "bad"})
        with patch("oracledb.defaults"):
            with pytest.raises(Exception, match="ORA-30584"):
                splitter.split_text("text")
