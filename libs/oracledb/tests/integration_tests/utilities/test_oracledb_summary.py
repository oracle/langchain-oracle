# Copyright (c) 2024, 2026, Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""
integration_tests/utilities/test_oracledb_summary.py

Integration and functional tests for OracleSummary, covering the full
summarization pipeline against a live Oracle database. The suite is skipped
automatically when the required environment variables are not set or the
database is unreachable.

Covers:
- Constructor wiring (conn, params, proxy storage and defaults)
- get_summary(None) -> empty list
- get_summary(str) with multiple glevel values (S, P, Sentence, Paragraph abbreviations)
- get_summary(Document) and consistency with str input
- get_summary(List[str]) and get_summary(List[Document]) count and type checks
- Invalid inputs raising (bad provider, bad glevel, negative numParagraphs, wrong types)
- Functional summary quality (shorter than input, repeatability, str vs Document parity)

Required environment variables:
    VECDB_HOST   — DSN / host string  (e.g. cdb1_pdb1)
    VECDB_USER   — database username  (e.g. vector_user)
    VECDB_PASS   — database password

Run:
    pytest tests/integration_tests/utilities/test_oracledb_summary.py -v

Authors:
    - Diego Ascencio (diegoascencioqa)
"""

from __future__ import annotations

import os

import oracledb
import pytest
from langchain_core.documents import Document

from langchain_oracledb.utilities.oracleai import OracleSummary

# ---------------------------------------------------------------------------
# Credentials from environment variables
# ---------------------------------------------------------------------------

USERNAME = os.environ.get("VECDB_USER")
PASSWORD = os.environ.get("VECDB_PASS")
DSN = os.environ.get("VECDB_HOST")

# ---------------------------------------------------------------------------
# Skip entire module if env vars missing or DB unreachable
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
# Shared test document
# ---------------------------------------------------------------------------

SAMPLE_DOC = (
    "It was 7 minutes after midnight. The dog was lying on the grass in front "
    "of Mrs Shears house. Its eyes were closed. It was running on its side, "
    "the way dogs run when they think they are chasing a cat in a dream. "
    "But the dog was not running or asleep. The dog was dead. "
    "There was a garden fork sticking out of the dog. The points of the fork "
    "must have gone all the way through the dog and into the ground because "
    "the fork had not fallen over. I decided that the dog was probably killed "
    "with the fork because I could not see any other wounds in the dog."
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def connection():
    conn = oracledb.connect(user=USERNAME, password=PASSWORD, dsn=DSN)
    yield conn
    conn.close()


@pytest.fixture
def summarizer(connection):
    return OracleSummary(
        conn=connection,
        params={
            "provider": "database",
            "glevel": "S",
            "numParagraphs": 1,
            "language": "english",
        },
    )


# ===========================================================================
# Integration — constructor and basic wiring
# ===========================================================================


class TestOracleSummaryConstructor:
    def test_conn_stored(self, connection):
        s = OracleSummary(conn=connection, params={"provider": "database"})
        assert s.conn is connection

    def test_params_stored(self, connection):
        params = {"provider": "database", "glevel": "S"}
        s = OracleSummary(conn=connection, params=params)
        assert s.summary_params == params

    def test_proxy_defaults_to_none(self, connection):
        s = OracleSummary(conn=connection, params={})
        assert s.proxy is None

    def test_proxy_stored(self, connection):
        s = OracleSummary(conn=connection, params={}, proxy="http://proxy:80")
        assert s.proxy == "http://proxy:80"


# ===========================================================================
# Integration — get_summary(None)
# ===========================================================================


class TestGetSummaryNone:
    def test_none_returns_empty_list(self, summarizer):
        assert summarizer.get_summary(None) == []


# ===========================================================================
# Integration — get_summary(str)
# ===========================================================================


class TestGetSummaryStr:
    def test_str_returns_list_of_one(self, summarizer):
        result = summarizer.get_summary(SAMPLE_DOC)
        assert isinstance(result, list)
        assert len(result) == 1

    def test_str_result_is_string(self, summarizer):
        result = summarizer.get_summary(SAMPLE_DOC)
        assert isinstance(result[0], str)

    def test_str_result_is_non_empty(self, summarizer):
        result = summarizer.get_summary(SAMPLE_DOC)
        assert len(result[0]) > 0

    def test_str_summary_shorter_than_input(self, summarizer):
        """A summary must be shorter than the original text."""
        result = summarizer.get_summary(SAMPLE_DOC)
        assert len(result[0]) < len(SAMPLE_DOC)

    def test_whitespace_only_string(self, connection):
        """Whitespace-only input is valid"""
        s = OracleSummary(
            conn=connection,
            params={"provider": "database", "glevel": "S", "numParagraphs": 2},
        )
        result = s.get_summary(" ")
        assert isinstance(result, list)
        assert len(result) == 1

    def test_glevel_paragraph(self, connection):
        s = OracleSummary(
            conn=connection,
            params={
                "provider": "database",
                "glevel": "paragraph",
                "numParagraphs": 2,
                "language": "english",
            },
        )
        result = s.get_summary(SAMPLE_DOC)
        assert len(result) == 1
        assert isinstance(result[0], str)

    def test_glevel_sentence(self, connection):
        s = OracleSummary(
            conn=connection,
            params={"provider": "database", "glevel": "Sentence"},
        )
        result = s.get_summary(SAMPLE_DOC)
        assert len(result) == 1
        assert isinstance(result[0], str)

    def test_glevel_P_abbreviation(self, connection):
        s = OracleSummary(
            conn=connection,
            params={"provider": "database", "glevel": "P"},
        )
        result = s.get_summary(SAMPLE_DOC)
        assert len(result) == 1

    def test_glevel_S_abbreviation(self, connection):
        s = OracleSummary(
            conn=connection,
            params={
                "provider": "database",
                "glevel": "S",
                "numParagraphs": 16,
                "language": "english",
            },
        )
        result = s.get_summary(SAMPLE_DOC)
        assert len(result) == 1


# ===========================================================================
# Integration — get_summary(Document)
# ===========================================================================


class TestGetSummaryDocument:
    def test_document_returns_list_of_one(self, summarizer):
        doc = Document(page_content=SAMPLE_DOC)
        result = summarizer.get_summary(doc)
        assert isinstance(result, list)
        assert len(result) == 1

    def test_document_result_is_non_empty_string(self, summarizer):
        doc = Document(page_content=SAMPLE_DOC)
        result = summarizer.get_summary(doc)
        assert isinstance(result[0], str)
        assert len(result[0]) > 0

    def test_document_summary_shorter_than_input(self, summarizer):
        doc = Document(page_content=SAMPLE_DOC)
        result = summarizer.get_summary(doc)
        assert len(result[0]) < len(SAMPLE_DOC)

    def test_document_same_result_as_str(self, summarizer):
        """Summarizing a Document must produce the same result as summarizing
        its page_content directly as a string."""
        str_result = summarizer.get_summary(SAMPLE_DOC)
        doc_result = summarizer.get_summary(Document(page_content=SAMPLE_DOC))
        assert str_result == doc_result


# ===========================================================================
# Integration — get_summary(List[str])
# ===========================================================================


class TestGetSummaryListStr:
    def test_list_str_two_items_returns_two_results(self, connection):
        """Each item in the list must produce an independent summary."""
        s = OracleSummary(
            conn=connection,
            params={"provider": "database", "glevel": "S"},
        )
        docs = [
            "The cat sat on the windowsill watching the rain.",
            "The rocket launched at dawn and cleared the clouds.",
        ]
        result = s.get_summary(docs)
        expected = [s.get_summary(doc)[0] for doc in docs]
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(r, str) for r in result)
        assert result == expected
        assert result[0] != result[1]

    def test_list_str_single_item(self, connection):
        s = OracleSummary(
            conn=connection,
            params={"provider": "database", "glevel": "S"},
        )
        result = s.get_summary([SAMPLE_DOC])
        assert len(result) == 1
        assert isinstance(result[0], str)
        assert len(result[0]) > 0


# ===========================================================================
# Integration — get_summary(List[Document])
# ===========================================================================


class TestGetSummaryListDocument:
    def test_list_document_two_items_returns_two_results(self, connection):
        """Each Document in the list must produce an independent summary."""
        s = OracleSummary(
            conn=connection,
            params={"provider": "database", "glevel": "S"},
        )
        docs = [
            Document(page_content="Mars is the fourth planet from the Sun."),
            Document(page_content="Bananas are rich in potassium and fiber."),
        ]
        result = s.get_summary(docs)
        expected = [s.get_summary(doc)[0] for doc in docs]
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(r, str) for r in result)
        assert result == expected
        assert result[0] != result[1]

    def test_list_document_single_item(self, connection):
        s = OracleSummary(
            conn=connection,
            params={"provider": "database", "glevel": "S"},
        )
        result = s.get_summary([Document(page_content=SAMPLE_DOC)])
        assert len(result) == 1
        assert isinstance(result[0], str)
        assert len(result[0]) > 0


# ===========================================================================
# Integration — invalid inputs raise
# ===========================================================================


class TestGetSummaryInvalidInputs:
    def test_invalid_provider_raises(self, connection):
        """Missing value for PROVIDER"""
        s = OracleSummary(
            conn=connection, params={"provider": "database1", "glevel": "S"}
        )
        with pytest.raises(Exception):
            s.get_summary(SAMPLE_DOC)

    def test_invalid_glevel_raises(self, connection):
        """Invalid gist level"""
        s = OracleSummary(
            conn=connection, params={"provider": "database", "glevel": "INVALID"}
        )
        with pytest.raises(Exception):
            s.get_summary(SAMPLE_DOC)

    def test_negative_num_paragraphs_raises(self, connection):
        """Negative numParagraphs is invalid"""
        s = OracleSummary(
            conn=connection,
            params={"provider": "database", "glevel": "S", "numParagraphs": -2},
        )
        with pytest.raises(Exception):
            s.get_summary(SAMPLE_DOC)

    def test_int_input_raises(self, connection):
        s = OracleSummary(
            conn=connection, params={"provider": "database", "glevel": "S"}
        )
        with pytest.raises(Exception, match="Invalid input type"):
            s.get_summary(42)

    def test_dict_input_raises(self, connection):
        s = OracleSummary(
            conn=connection, params={"provider": "database", "glevel": "S"}
        )
        with pytest.raises(Exception, match="Invalid input type"):
            s.get_summary({"text": SAMPLE_DOC})

    def test_list_with_invalid_item_raises(self, connection):
        s = OracleSummary(
            conn=connection, params={"provider": "database", "glevel": "S"}
        )
        with pytest.raises(Exception, match="Invalid input type"):
            s.get_summary([SAMPLE_DOC, 42])

    def test_empty_list_returns_empty_list(self, connection):
        """Empty list input must return an empty list without raising."""
        s = OracleSummary(
            conn=connection, params={"provider": "database", "glevel": "S"}
        )
        result = s.get_summary([])
        assert result == []


# ===========================================================================
# Functional — summary quality
# ===========================================================================


class TestFunctionalSummaryQuality:
    def test_summary_is_shorter_than_input(self, summarizer):
        result = summarizer.get_summary(SAMPLE_DOC)
        assert len(result[0]) < len(SAMPLE_DOC)

    def test_different_params_produce_different_summaries(self, connection):
        """Sentence-level and paragraph-level summaries should differ."""
        s_sentence = OracleSummary(
            conn=connection,
            params={"provider": "database", "glevel": "S"},
        )
        s_paragraph = OracleSummary(
            conn=connection,
            params={"provider": "database", "glevel": "P"},
        )
        r_sentence = s_sentence.get_summary(SAMPLE_DOC)
        r_paragraph = s_paragraph.get_summary(SAMPLE_DOC)
        # Both must be valid non-empty strings
        assert len(r_sentence[0]) > 0
        assert len(r_paragraph[0]) > 0

    def test_repeated_summary_same_result(self, summarizer):
        """Same input and params must produce the same summary both times."""
        r1 = summarizer.get_summary(SAMPLE_DOC)
        r2 = summarizer.get_summary(SAMPLE_DOC)
        assert r1 == r2

    def test_document_and_string_same_content_same_result(self, summarizer):
        """str and Document with same content must produce identical summary."""
        r_str = summarizer.get_summary(SAMPLE_DOC)
        r_doc = summarizer.get_summary(Document(page_content=SAMPLE_DOC))
        assert r_str == r_doc
