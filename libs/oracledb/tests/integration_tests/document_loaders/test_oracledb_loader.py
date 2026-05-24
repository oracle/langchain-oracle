# Copyright (c) 2024, 2026, Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""
integration_tests/document_loaders/test_oracledb_loader.py


Integration tests for OracleDocLoader, OracleTextSplitter, ParseOracleDocMetadata,
and OracleAutonomousDatabaseLoader.

Covers:
- Loading documents from Oracle tables, files, and directories
- Metadata extraction (_oid, _rowid, _file) across all load modes
- NULL value handling and error conditions (missing table, missing params)
- Text splitting strategies (by words, by chars) with overlap and normalization
- Unicode content passing through Oracle intact
- HTML metadata parsing (author, subject, title) via ParseOracleDocMetadata
- Full load -> split pipeline with metadata preservation
- OracleAutonomousDatabaseLoader with multi-column queries and selective metadata
- Sync code paths throughout

Required environment variables:
    VECDB_HOST   — DSN / host string  (e.g. cdb1_pdb1)
    VECDB_USER   — database username  (e.g. vector_user)
    VECDB_PASS   — database password

Set them before running:
    export VECDB_HOST=cdb1_pdb1
    export VECDB_USER=vector_user
    export VECDB_PASS=Linux1234#Solaris1234#

Run:
    pytest tests/integration_tests/document_loaders/test_oracledb_loader.py

Authors:
    - Diego Ascencio (diegoascencioqa)
"""

from __future__ import annotations

import os
import pathlib

import oracledb
import pytest
from langchain_core.documents import Document

from langchain_oracledb.document_loaders.oracleadb_loader import (
    OracleAutonomousDatabaseLoader,
)
from langchain_oracledb.document_loaders.oracleai import (
    OracleDocLoader,
    OracleTextSplitter,
    ParseOracleDocMetadata,
)
from langchain_oracledb.vectorstores.oraclevs import (
    _table_exists,
    drop_table_purge,
)

# ---------------------------------------------------------------------------
# Credentials from environment variables
# ---------------------------------------------------------------------------

USERNAME = os.environ.get("VECDB_USER")
PASSWORD = os.environ.get("VECDB_PASS")
DSN = os.environ.get("VECDB_HOST")

# ---------------------------------------------------------------------------
# Skip entire module if env vars are missing or DB is unreachable
# ---------------------------------------------------------------------------

if not all([USERNAME, PASSWORD, DSN]):
    pytest.skip(
        allow_module_level=True,
        reason="VECDB_USER, VECDB_PASS, VECDB_HOST environment variables not set.",
    )

try:
    oracledb.connect(user=USERNAME, password=PASSWORD, dsn=DSN)
except Exception as e:
    pytest.skip(
        allow_module_level=True,
        reason=f"Database connection failed: {e}",
    )

# ---------------------------------------------------------------------------
# Fixtures (integration tests — real DB)
# ---------------------------------------------------------------------------


@pytest.fixture
def connection():
    """Fresh connection for each test. Closed after test completes."""
    conn = oracledb.connect(user=USERNAME, password=PASSWORD, dsn=DSN)
    yield conn
    conn.close()


@pytest.fixture
def demo_table(connection):
    """
    Creates LANGCHAIN_DEMO with 7 rows before the test.
    Drops it after the test regardless of pass or fail.
    """
    if _table_exists(connection, "LANGCHAIN_DEMO"):
        drop_table_purge(connection, "LANGCHAIN_DEMO")

    cursor = connection.cursor()
    cursor.execute("CREATE TABLE langchain_demo(id number, text varchar2(25))")
    cursor.executemany(
        "INSERT INTO langchain_demo(id, text) VALUES (:1, :2)",
        [
            (1, "First"),
            (2, "Second"),
            (3, "Third"),
            (4, "Fourth"),
            (5, "Fifth"),
            (6, "Sixth"),
            (7, "Seventh"),
        ],
    )
    connection.commit()
    cursor.close()

    yield "LANGCHAIN_DEMO"

    drop_table_purge(connection, "LANGCHAIN_DEMO")


# ---------------------------------------------------------------------------
# Example files already in the repo
# ---------------------------------------------------------------------------

_REPO_ROOT = pathlib.Path(__file__).parents[5]
EXAMPLE_DATA_DIR = (
    _REPO_ROOT / "libs" / "js" / "langchain-oracledb" / "src" / "tests" / "example_data"
)
EXAMPLE_PDF = EXAMPLE_DATA_DIR / "1706.03762.pdf"
EXAMPLE_DOCX = EXAMPLE_DATA_DIR / "attention.docx"


# ===========================================================================
# Integration — OracleDocLoader table mode
# ===========================================================================


class TestOracleDocLoaderTableMode:
    def test_loads_all_rows_from_table(self, connection, demo_table):
        loader = OracleDocLoader(
            conn=connection,
            params={
                "owner": USERNAME,
                "tablename": "LANGCHAIN_DEMO",
                "colname": "TEXT",
            },
        )
        docs = loader.load()
        assert len(docs) == 7

    def test_each_doc_has_page_content(self, connection, demo_table):
        loader = OracleDocLoader(
            conn=connection,
            params={
                "owner": USERNAME,
                "tablename": "LANGCHAIN_DEMO",
                "colname": "TEXT",
            },
        )
        for doc in loader.load():
            assert isinstance(doc.page_content, str)
            assert len(doc.page_content) > 0

    def test_each_doc_has_oid_and_rowid_in_metadata(self, connection, demo_table):
        loader = OracleDocLoader(
            conn=connection,
            params={
                "owner": USERNAME,
                "tablename": "LANGCHAIN_DEMO",
                "colname": "TEXT",
            },
        )
        for doc in loader.load():
            assert "_oid" in doc.metadata
            assert "_rowid" in doc.metadata

    def test_all_expected_values_present(self, connection, demo_table):
        loader = OracleDocLoader(
            conn=connection,
            params={
                "owner": USERNAME,
                "tablename": "LANGCHAIN_DEMO",
                "colname": "TEXT",
            },
        )
        contents = [doc.page_content for doc in loader.load()]
        for expected in [
            "First",
            "Second",
            "Third",
            "Fourth",
            "Fifth",
            "Sixth",
            "Seventh",
        ]:
            assert expected in contents

    def test_nonexistent_table_raises_oracle_error(self, connection):
        loader = OracleDocLoader(
            conn=connection,
            params={
                "owner": USERNAME,
                "tablename": "THIS_TABLE_DOES_NOT_EXIST",
                "colname": "TEXT",
            },
        )
        with pytest.raises(Exception, match="ORA-00942|table or view does not exist"):
            loader.load()

    def test_missing_owner_raises(self, connection):
        loader = OracleDocLoader(
            conn=connection,
            params={
                "tablename": "LANGCHAIN_DEMO",
                "colname": "TEXT",
            },
        )
        with pytest.raises(Exception):
            loader.load()

    def test_missing_colname_raises(self, connection):
        loader = OracleDocLoader(
            conn=connection,
            params={
                "owner": USERNAME,
                "tablename": "LANGCHAIN_DEMO",
            },
        )
        with pytest.raises(Exception):
            loader.load()

    def test_mdata_cols_limit_exceeded_raises(self, connection, demo_table):
        loader = OracleDocLoader(
            conn=connection,
            params={
                "owner": USERNAME,
                "tablename": "LANGCHAIN_DEMO",
                "colname": "TEXT",
                "mdata_cols": ["A", "B", "C", "D"],
            },
        )
        with pytest.raises(Exception, match="Exceeds the max number"):
            loader.load()


# ===========================================================================
# Integration — OracleDocLoader file mode
# ===========================================================================


class TestOracleDocLoaderFileMode:
    def test_load_pdf_returns_document_with_content(self, connection):
        if not EXAMPLE_PDF.exists():
            pytest.skip(f"Example file not found: {EXAMPLE_PDF}")
        loader = OracleDocLoader(conn=connection, params={"file": str(EXAMPLE_PDF)})
        docs = loader.load()
        assert len(docs) == 1
        assert isinstance(docs[0].page_content, str)
        assert "_oid" in docs[0].metadata
        assert docs[0].metadata["_file"] == str(EXAMPLE_PDF)

    def test_load_docx_returns_document_with_content(self, connection):
        if not EXAMPLE_DOCX.exists():
            pytest.skip(f"Example file not found: {EXAMPLE_DOCX}")
        loader = OracleDocLoader(conn=connection, params={"file": str(EXAMPLE_DOCX)})
        docs = loader.load()
        assert len(docs) == 1
        assert isinstance(docs[0].page_content, str)
        assert "_oid" in docs[0].metadata

    def test_nonexistent_file_returns_empty(self, connection):
        loader = OracleDocLoader(
            conn=connection, params={"file": "/no/such/file/xyz.txt"}
        )
        docs = loader.load()
        assert docs == []


# ===========================================================================
# Integration — OracleDocLoader dir mode
# ===========================================================================


class TestOracleDocLoaderDirMode:
    def test_load_example_data_dir_returns_one_doc_per_file(self, connection):
        if not EXAMPLE_DATA_DIR.exists():
            pytest.skip(f"Example data dir not found: {EXAMPLE_DATA_DIR}")
        loader = OracleDocLoader(conn=connection, params={"dir": str(EXAMPLE_DATA_DIR)})
        docs = loader.load()
        assert len(docs) >= 1
        for doc in docs:
            assert isinstance(doc.page_content, str)
            assert "_oid" in doc.metadata
            assert "_file" in doc.metadata

    def test_each_doc_file_path_is_inside_example_dir(self, connection):
        if not EXAMPLE_DATA_DIR.exists():
            pytest.skip(f"Example data dir not found: {EXAMPLE_DATA_DIR}")
        loader = OracleDocLoader(conn=connection, params={"dir": str(EXAMPLE_DATA_DIR)})
        for doc in loader.load():
            assert str(EXAMPLE_DATA_DIR) in doc.metadata["_file"]

    def test_nonexistent_dir_raises(self, connection):
        loader = OracleDocLoader(conn=connection, params={"dir": "/no/such/dir/xyz"})
        with pytest.raises(Exception):
            loader.load()

    def test_empty_dir_returns_empty_list(self, connection, tmp_path):
        loader = OracleDocLoader(conn=connection, params={"dir": str(tmp_path)})
        assert loader.load() == []


# ===========================================================================
# Integration — OracleTextSplitter happy paths
# ===========================================================================

SAMPLE_DOC = (
    "Langchain is a wonderful framework to load, split, chunk and embed your data!!"
)


class TestOracleTextSplitterHappyPaths:
    def test_split_by_words(self, connection):
        splitter = OracleTextSplitter(
            conn=connection,
            params={
                "by": "words",
                "max": "1000",
                "overlap": "200",
                "split": "custom",
                "custom_list": [","],
                "extended": "true",
                "normalize": "all",
            },
        )
        chunks = splitter.split_text(SAMPLE_DOC)
        assert isinstance(chunks, list) and len(chunks) > 0
        assert all(isinstance(c, str) for c in chunks)

    def test_split_by_chars(self, connection):
        splitter = OracleTextSplitter(
            conn=connection,
            params={
                "by": "chars",
                "max": "4000",
                "overlap": "800",
                "split": "NEWLINE",
                "normalize": "all",
            },
        )
        assert len(splitter.split_text(SAMPLE_DOC)) > 0

    def test_split_by_words_small_max(self, connection):
        splitter = OracleTextSplitter(
            conn=connection,
            params={
                "by": "words",
                "max": "10",
                "overlap": "2",
                "split": "SENTENCE",
            },
        )
        assert len(splitter.split_text(SAMPLE_DOC)) > 0

    def test_split_by_chars_small_max(self, connection):
        splitter = OracleTextSplitter(
            conn=connection,
            params={
                "by": "chars",
                "max": "50",
                "overlap": "10",
                "split": "SPACE",
                "normalize": "all",
            },
        )
        assert len(splitter.split_text(SAMPLE_DOC)) > 0

    def test_chunks_contain_original_words(self, connection):
        splitter = OracleTextSplitter(
            conn=connection, params={"by": "words", "max": "1000"}
        )
        combined = " ".join(splitter.split_text(SAMPLE_DOC))
        for word in ["Langchain", "framework", "embed", "data"]:
            assert word in combined

    def test_empty_string_returns_empty_list(self, connection):
        splitter = OracleTextSplitter(
            conn=connection, params={"by": "words", "max": "1000"}
        )
        assert splitter.split_text("") == []


# ===========================================================================
# Integration — OracleTextSplitter bad inputs
# ===========================================================================


class TestOracleTextSplitterBadInputs:
    def test_invalid_by_param_raises(self, connection):
        """ORA-20003: invalid value xyz for BY parameter."""
        splitter = OracleTextSplitter(conn=connection, params={"by": "xyz"})
        with pytest.raises(Exception, match="ORA-20003"):
            splitter.split_text(SAMPLE_DOC)

    def test_chars_max_too_small_raises(self, connection):
        splitter = OracleTextSplitter(
            conn=connection,
            params={
                "by": "chars",
                "max": "10",
                "overlap": "2",
                "split": "SPACE",
                "normalize": "all",
            },
        )
        with pytest.raises(Exception, match="ORA-|invalid"):
            splitter.split_text(SAMPLE_DOC)

    def test_words_max_too_small_raises(self, connection):
        splitter = OracleTextSplitter(
            conn=connection,
            params={
                "by": "words",
                "max": "5",
                "overlap": "2",
                "split": "SPACE",
                "normalize": "all",
            },
        )
        with pytest.raises(Exception, match="ORA-|invalid"):
            splitter.split_text(SAMPLE_DOC)

    def test_invalid_by_value_raises(self, connection):
        """ORA-20003: completely unknown value for BY parameter raises."""
        splitter = OracleTextSplitter(conn=connection, params={"by": "nonsense_value"})
        with pytest.raises(Exception, match="ORA-|invalid"):
            splitter.split_text(SAMPLE_DOC)


# ===========================================================================
# Functional — ParseOracleDocMetadata
# Pure Python HTML parsing — no DB needed even in integration context.
# ===========================================================================


class TestFunctionalParseMetadata:
    def test_author_value_matches_html(self):
        parser = ParseOracleDocMetadata()
        parser.feed('<meta name="author" content="Alice">')
        assert parser.metadata["author"] == "Alice"

    def test_multiple_fields_all_correct(self):
        parser = ParseOracleDocMetadata()
        parser.feed(
            '<meta name="author" content="Alice">'
            '<meta name="subject" content="Science">'
            "<title>My Paper</title>"
        )
        assert parser.metadata["author"] == "Alice"
        assert parser.metadata["subject"] == "Science"
        assert parser.metadata["title"] == "My Paper"

    def test_parser_reuse_does_not_bleed_state(self):
        """
        Two separate parser instances must not share metadata
        (reset() does not clear it).
        """
        parser1 = ParseOracleDocMetadata()
        parser1.feed('<meta name="author" content="Alice">')
        assert parser1.metadata["author"] == "Alice"

        parser2 = ParseOracleDocMetadata()
        parser2.feed('<meta name="subject" content="Physics">')
        assert "author" not in parser2.metadata
        assert parser2.metadata["subject"] == "Physics"

    def test_whitespace_in_content_preserved(self):
        parser = ParseOracleDocMetadata()
        parser.feed('<meta name="title" content="  spaced  ">')
        assert parser.metadata["title"] == "  spaced  "


# ===========================================================================
# Functional — OracleDocLoader table mode (real DB)
# Verifies that the right *data* comes out of a real Oracle query,
# not just that the right methods were called.
# ===========================================================================


class TestFunctionalDocLoaderTableMode:
    def test_page_content_matches_row_text(self, connection, demo_table):
        """
        Each loaded document's page_content must equal the value in the TEXT column.
        """
        loader = OracleDocLoader(
            conn=connection,
            params={
                "owner": USERNAME,
                "tablename": "LANGCHAIN_DEMO",
                "colname": "TEXT",
            },
        )
        docs = loader.load()
        contents = [doc.page_content for doc in docs]
        assert "First" in contents
        assert "Seventh" in contents

    def test_rowid_is_non_empty_string(self, connection, demo_table):
        """Every document must have a non-empty _rowid from Oracle."""
        loader = OracleDocLoader(
            conn=connection,
            params={
                "owner": USERNAME,
                "tablename": "LANGCHAIN_DEMO",
                "colname": "TEXT",
            },
        )
        for doc in loader.load():
            assert isinstance(doc.metadata["_rowid"], str)
            assert len(doc.metadata["_rowid"]) > 0

    def test_seven_rows_produce_seven_docs_with_correct_content(
        self, connection, demo_table
    ):
        """All 7 expected values must appear in the loaded documents in any order."""
        loader = OracleDocLoader(
            conn=connection,
            params={
                "owner": USERNAME,
                "tablename": "LANGCHAIN_DEMO",
                "colname": "TEXT",
            },
        )
        docs = loader.load()
        contents = sorted([doc.page_content for doc in docs])
        expected = sorted(
            ["First", "Second", "Third", "Fourth", "Fifth", "Sixth", "Seventh"]
        )
        assert contents == expected

    def test_each_doc_has_unique_oid(self, connection, demo_table):
        """Every document must have a distinct _oid — no two rows share an ID."""
        loader = OracleDocLoader(
            conn=connection,
            params={
                "owner": USERNAME,
                "tablename": "LANGCHAIN_DEMO",
                "colname": "TEXT",
            },
        )
        docs = loader.load()
        oids = [doc.metadata["_oid"] for doc in docs]
        assert len(oids) == len(set(oids))

    def test_null_text_rows_stored_as_empty_string(self, connection):
        """When a row has NULL in the text column, Oracle returns it as empty string."""
        cursor = connection.cursor()
        try:
            if _table_exists(connection, "LANGCHAIN_NULL_TEST"):
                drop_table_purge(connection, "LANGCHAIN_NULL_TEST")
            cursor.execute(
                "CREATE TABLE langchain_null_test(id number, text varchar2(25))"
            )
            cursor.execute(
                "INSERT INTO langchain_null_test(id, text) VALUES (1, 'present')"
            )
            cursor.execute("INSERT INTO langchain_null_test(id, text) VALUES (2, NULL)")
            connection.commit()

            loader = OracleDocLoader(
                conn=connection,
                params={
                    "owner": USERNAME,
                    "tablename": "LANGCHAIN_NULL_TEST",
                    "colname": "TEXT",
                },
            )
            docs = loader.load()
            assert len(docs) == 2
            contents = {doc.page_content for doc in docs}
            assert "present" in contents
            assert "" in contents  # NULL -> "" (source behaviour)
        finally:
            cursor.close()
            drop_table_purge(connection, "LANGCHAIN_NULL_TEST")


# ===========================================================================
# Functional — OracleTextSplitter (real DB)
# Verifies that Oracle's utl_to_chunks actually returns the right content,
# not just that the Python wrapper called the right methods.
# ===========================================================================


class TestFunctionalTextSplitter:
    def test_single_short_text_returns_one_chunk(self, connection):
        """A short text with a large max must come back as a single chunk."""
        text = "hello world"
        splitter = OracleTextSplitter(
            conn=connection, params={"by": "words", "max": "1000"}
        )
        chunks = splitter.split_text(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_content_contains_original_words(self, connection):
        """
        Key words from the input must appear somewhere in the chunks Oracle returns.
        """
        text = (
            "Langchain is a wonderful framework to load split chunk and embed your data"
        )
        splitter = OracleTextSplitter(
            conn=connection, params={"by": "words", "max": "1000"}
        )
        combined = " ".join(splitter.split_text(text))
        for word in ["Langchain", "wonderful", "framework", "embed", "data"]:
            assert word in combined

    def test_small_max_produces_multiple_chunks(self, connection):
        """With a small max, a longer text must be split into more than one chunk."""
        text = "one two three four five six seven eight nine ten eleven twelve"
        splitter = OracleTextSplitter(
            conn=connection, params={"by": "words", "max": "10", "overlap": "0"}
        )
        chunks = splitter.split_text(text)
        assert len(chunks) > 1
        assert all(isinstance(c, str) and len(c) > 0 for c in chunks)

    def test_chunk_content_survives_unicode(self, connection):
        """Unicode characters must not be corrupted passing through Oracle."""
        text = "Héllo wörld — this is a unicode test string with accents"
        splitter = OracleTextSplitter(
            conn=connection, params={"by": "words", "max": "1000"}
        )
        chunks = splitter.split_text(text)
        combined = " ".join(chunks)
        assert "Héllo" in combined
        assert "wörld" in combined

    def test_empty_string_returns_empty_list(self, connection):
        """Empty input must return an empty list from the real Oracle call."""
        splitter = OracleTextSplitter(
            conn=connection, params={"by": "words", "max": "1000"}
        )
        assert splitter.split_text("") == []


# ===========================================================================
# Functional — Pipeline (real DB, two classes working together)
# These exercise load() -> split_text() end to end against a real Oracle database.
# ===========================================================================


class TestFunctionalPipeline:
    def test_load_then_split_roundtrip(self, connection, demo_table):
        """Content loaded from a real table must survive split_text() intact."""
        loader = OracleDocLoader(
            conn=connection,
            params={
                "owner": USERNAME,
                "tablename": "LANGCHAIN_DEMO",
                "colname": "TEXT",
            },
        )
        docs = loader.load()
        assert len(docs) == 7

        splitter = OracleTextSplitter(
            conn=connection, params={"by": "words", "max": "1000"}
        )
        all_chunks = []
        for doc in docs:
            all_chunks.extend(splitter.split_text(doc.page_content))

        # Every original value must appear somewhere in the chunks
        chunk_text = " ".join(all_chunks)
        for expected in [
            "First",
            "Second",
            "Third",
            "Fourth",
            "Fifth",
            "Sixth",
            "Seventh",
        ]:
            assert expected in chunk_text

    def test_each_doc_splits_independently(self, connection, demo_table):
        """Splitting docs one at a time must produce the same total output
        as splitting them together — no state leaks between calls."""
        loader = OracleDocLoader(
            conn=connection,
            params={
                "owner": USERNAME,
                "tablename": "LANGCHAIN_DEMO",
                "colname": "TEXT",
            },
        )
        docs = loader.load()
        splitter = OracleTextSplitter(
            conn=connection, params={"by": "words", "max": "1000"}
        )

        all_chunks = []
        for doc in docs:
            chunks = splitter.split_text(doc.page_content)
            assert isinstance(chunks, list)
            assert all(isinstance(c, str) for c in chunks)
            all_chunks.extend(chunks)

        assert len(all_chunks) == len(docs)  # 1 chunk per short row at max=1000

    def test_load_pdf_then_split_produces_chunks(self, connection):
        """A real PDF loaded from disk must produce non-empty chunks after splitting."""
        if not EXAMPLE_PDF.exists():
            pytest.skip(f"Example file not found: {EXAMPLE_PDF}")

        loader = OracleDocLoader(conn=connection, params={"file": str(EXAMPLE_PDF)})
        docs = loader.load()
        assert len(docs) == 1
        assert len(docs[0].page_content) > 0

        splitter = OracleTextSplitter(
            conn=connection,
            params={
                "by": "words",
                "max": "100",
                "overlap": "10",
            },
        )
        chunks = splitter.split_text(docs[0].page_content)
        assert len(chunks) > 1
        assert all(isinstance(c, str) and len(c) > 0 for c in chunks)

    def test_oid_and_rowid_present_after_load_before_split(
        self, connection, demo_table
    ):
        """Metadata set during load must still be on the Document when it reaches
        the splitter — nothing in the pipeline should strip it."""
        loader = OracleDocLoader(
            conn=connection,
            params={
                "owner": USERNAME,
                "tablename": "LANGCHAIN_DEMO",
                "colname": "TEXT",
            },
        )
        docs = loader.load()

        for doc in docs:
            assert "_oid" in doc.metadata
            assert "_rowid" in doc.metadata
            # split_text should not touch metadata
            splitter = OracleTextSplitter(
                conn=connection, params={"by": "words", "max": "1000"}
            )
            splitter.split_text(doc.page_content)
            assert "_oid" in doc.metadata
            assert "_rowid" in doc.metadata


# ===========================================================================
# Integration — OracleAutonomousDatabaseLoader
# ===========================================================================


class TestADBLoaderFunctionalPipeline:
    def test_load_then_check_all_rows_present(self):
        """Load 5 rows via CONNECT BY, verify all page_content values are present."""
        loader = OracleAutonomousDatabaseLoader(
            query="SELECT level AS n FROM DUAL CONNECT BY level <= 5",
            user=USERNAME,
            password=PASSWORD,
            dsn=DSN,
        )
        docs = loader.load()
        assert len(docs) == 5
        assert all(isinstance(d, Document) for d in docs)

    def test_multiple_columns_all_in_page_content(self):
        """All column values must appear in the str(row_dict) page_content."""
        loader = OracleAutonomousDatabaseLoader(
            query="SELECT 'hello' AS greeting, 'world' AS target FROM DUAL",
            user=USERNAME,
            password=PASSWORD,
            dsn=DSN,
        )
        docs = loader.load()
        assert "hello" in docs[0].page_content
        assert "world" in docs[0].page_content

    def test_multiple_metadata_columns(self):
        loader = OracleAutonomousDatabaseLoader(
            query="SELECT 1 AS id, 'Alice' AS name, 'ENG' AS dept FROM DUAL",
            user=USERNAME,
            password=PASSWORD,
            dsn=DSN,
            metadata=["ID", "DEPT"],
        )
        docs = loader.load()
        assert docs[0].metadata == {"ID": 1, "DEPT": "ENG"}
        assert "NAME" not in docs[0].metadata

    def test_empty_query_result_returns_empty_list(self):
        loader = OracleAutonomousDatabaseLoader(
            query="SELECT 1 AS n FROM DUAL WHERE 1=0",
            user=USERNAME,
            password=PASSWORD,
            dsn=DSN,
        )
        assert loader.load() == []

    def test_repeated_load_returns_same_result(self):
        """Calling load() twice must return the same documents."""
        loader = OracleAutonomousDatabaseLoader(
            query="SELECT 42 AS val FROM DUAL",
            user=USERNAME,
            password=PASSWORD,
            dsn=DSN,
        )
        docs1 = loader.load()
        docs2 = loader.load()
        assert len(docs1) == len(docs2)
        assert docs1[0].page_content == docs2[0].page_content
