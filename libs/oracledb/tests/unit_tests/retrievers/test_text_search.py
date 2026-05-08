# Copyright (c) 2024, 2026, Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""
unit_tests/retrievers/test_text_search.py

Unit tests for OracleTextSearchRetriever and its supporting pure functions,
exercising the full query-building and validation contract without any real
database connection. All DB interactions are replaced with mock objects.

Covers:
- _generate_accum_query tokenization, ACCUM joining, punctuation handling,
  fuzzy mode (True/False), empty/whitespace input, case preservation,
  numeric tokens, newlines and tabs
- _get_text_index_ddl DDL construction for OracleVS and raw table paths,
  all error branches (both inputs, neither input, invalid column), and
  case-insensitive "TEXT" column acceptance
- OracleTextSearchRetriever Pydantic model_validator: mutual exclusivity of
  vector_store and table_name, column_name restrictions with vector_store,
  client requirement for raw table, returned_columns deduplication and
  defaults, fuzzy+operator_search warning via caplog, all field defaults
- OracleTextSearchRetriever._get_result_documents: score present/absent based
  on return_scores, page_content population, raw table row handling,
  empty row list

Run:
    pytest tests/unit_tests/retrievers/test_text_search.py

Authors:
    - Diego Ascencio (diegoascencioqa)
"""

import logging
from unittest.mock import MagicMock

import pytest

from langchain_oracledb.retrievers.text_search import (
    OracleTextSearchRetriever,
    _generate_accum_query,
    _get_text_index_ddl,
)
from langchain_oracledb.vectorstores.oraclevs import OracleVS

# ---------------------------------------------------------------------------
# _generate_accum_query
# ---------------------------------------------------------------------------


class TestGenerateAccumQuery:
    """Tests for the _generate_accum_query pure function."""

    # --- Basic tokenization ---
    def test_single_word(self):
        assert _generate_accum_query("hello") == '"hello"'

    def test_two_words(self):
        assert _generate_accum_query("refund policy") == '"refund" ACCUM "policy"'

    def test_three_words(self):
        result = _generate_accum_query("refund policy plan")
        assert result == '"refund" ACCUM "policy" ACCUM "plan"'

    # --- Punctuation is treated as a word boundary ---
    def test_comma_as_boundary(self):
        # "quick, brown" should tokenize to ["quick", "brown"]
        assert _generate_accum_query("quick, brown") == '"quick" ACCUM "brown"'

    def test_exclamation_as_boundary(self):
        assert _generate_accum_query("hello!") == '"hello"'

    def test_mixed_punctuation(self):
        result = _generate_accum_query("refund, policy for (premium) ???")
        assert result == '"refund" ACCUM "policy" ACCUM "for" ACCUM "premium"'

    def test_leading_and_trailing_punctuation(self):
        assert _generate_accum_query("!hello!") == '"hello"'

    def test_only_punctuation_returns_empty(self):
        # re.split on non-word chars then filter empties -> no tokens
        assert _generate_accum_query("!!!") == ""

    # --- Empty / whitespace input ---
    def test_empty_string(self):
        assert _generate_accum_query("") == ""

    def test_whitespace_only(self):
        assert _generate_accum_query("   ") == ""

    # --- Fuzzy mode ---
    def test_single_word_fuzzy(self):
        assert _generate_accum_query("hello", fuzzy=True) == 'fuzzy("hello")'

    def test_two_words_fuzzy(self):
        result = _generate_accum_query("refund policy", fuzzy=True)
        assert result == 'fuzzy("refund") ACCUM fuzzy("policy")'

    def test_fuzzy_false_default(self):
        """fuzzy defaults to False; output should use quoted tokens."""
        result = _generate_accum_query("tablespace")
        assert result == '"tablespace"'
        assert "fuzzy" not in result

    def test_punctuation_fuzzy(self):
        result = _generate_accum_query("quick, brown!", fuzzy=True)
        assert result == 'fuzzy("quick") ACCUM fuzzy("brown")'

    # --- Case preservation ---
    def test_preserves_case(self):
        """Tokens are not lowercased; Oracle Text is case-insensitive anyway."""
        result = _generate_accum_query("TableSpace")
        assert '"TableSpace"' in result

    # --- Numeric tokens ---
    def test_numeric_tokens(self):
        result = _generate_accum_query("version 2 update")
        assert '"version" ACCUM "2" ACCUM "update"' == result

    # --- Multiple spaces ---
    def test_multiple_spaces_between_words(self):
        result = _generate_accum_query("refund   policy")
        assert result == '"refund" ACCUM "policy"'

    # --- Newlines and tabs ---
    def test_newline_as_boundary(self):
        result = _generate_accum_query("refund\npolicy")
        assert result == '"refund" ACCUM "policy"'

    def test_tab_as_boundary(self):
        result = _generate_accum_query("refund\tpolicy")
        assert result == '"refund" ACCUM "policy"'


# ---------------------------------------------------------------------------
# _get_text_index_ddl
# ---------------------------------------------------------------------------


class TestGetTextIndexDdl:
    """Tests for the _get_text_index_ddl helper."""

    def _make_vs(self, table_name="MY_TABLE"):
        vs = MagicMock(spec=OracleVS)
        vs.table_name = table_name
        return vs

    # --- Happy paths ---
    def test_vector_store_path_default_column(self):
        vs = self._make_vs("DOCS")
        ddl, resolved_table = _get_text_index_ddl('"IDX_DOCS"', vs, None)
        assert "CREATE SEARCH INDEX" in ddl
        assert '"IDX_DOCS"' in ddl
        assert "DOCS" in ddl
        assert "(text)" in ddl
        assert resolved_table == "DOCS"

    def test_vector_store_path_explicit_text_column(self):
        vs = self._make_vs("DOCS")
        ddl, resolved_table = _get_text_index_ddl('"IDX"', vs, None, column_name="text")
        assert "(text)" in ddl

    def test_table_name_path(self):
        ddl, resolved_table = _get_text_index_ddl(
            '"IDX_RAW"', None, "MY_TABLE", column_name="body"
        )
        assert '"MY_TABLE"' in ddl
        assert '("BODY")' in ddl
        assert resolved_table == '"MY_TABLE"'

    def test_table_name_path_preserves_quoted_case(self):
        ddl, resolved_table = _get_text_index_ddl(
            '"IDX_RAW"', None, '"MyTable"', column_name='"Body"'
        )
        assert '"MyTable"' in ddl
        assert '("Body")' in ddl
        assert resolved_table == '"MyTable"'

    # --- Error cases ---
    def test_both_vector_store_and_table_name_raises(self):
        vs = self._make_vs()
        with pytest.raises(
            ValueError, match="Only give one of vector_store or table_name"
        ):
            _get_text_index_ddl('"IDX"', vs, "OTHER_TABLE")

    def test_neither_vector_store_nor_table_name_raises(self):
        with pytest.raises(
            ValueError, match="Provide either vector_store or table_name"
        ):
            _get_text_index_ddl('"IDX"', None, None)

    def test_vector_store_with_non_text_column_raises(self):
        vs = self._make_vs()
        with pytest.raises(ValueError, match="column_name must be 'text'"):
            _get_text_index_ddl('"IDX"', vs, None, column_name="metadata")

    def test_vector_store_with_non_text_column_title_raises(self):
        vs = self._make_vs()
        with pytest.raises(ValueError, match="column_name must be 'text'"):
            _get_text_index_ddl('"IDX"', vs, None, column_name="title")

    def test_table_name_without_column_name_raises(self):
        with pytest.raises(ValueError, match="column_name must be provided"):
            _get_text_index_ddl('"IDX"', None, "MY_TABLE", column_name=None)

    def test_table_name_without_column_name_empty_string_raises(self):
        with pytest.raises(ValueError):
            _get_text_index_ddl('"IDX"', None, "MY_TABLE", column_name="")

    def test_table_name_rejects_sql_fragment(self):
        with pytest.raises(ValueError):
            _get_text_index_ddl('"IDX"', None, "DOCS WHERE 1=1 --", column_name="body")

    def test_column_name_rejects_sql_fragment(self):
        with pytest.raises(ValueError):
            _get_text_index_ddl(
                '"IDX"', None, "DOCS", column_name="body) PARAMETERS('x') --"
            )

    # --- Column name is case-insensitive for vector store ---
    def test_vector_store_text_column_uppercase_accepted(self):
        """'TEXT' (uppercase) should be accepted as equivalent to 'text'."""
        vs = self._make_vs("DOCS")
        ddl, _ = _get_text_index_ddl('"IDX"', vs, None, column_name="TEXT")
        assert "text" in ddl

    # --- DDL structure ---
    def test_ddl_contains_create_search_index(self):
        vs = self._make_vs("T")
        ddl, _ = _get_text_index_ddl('"I"', vs, None)
        assert ddl.strip().startswith("CREATE SEARCH INDEX")


# ---------------------------------------------------------------------------
# OracleTextSearchRetriever — Pydantic validation (no DB)
# ---------------------------------------------------------------------------


class TestOracleTextSearchRetrieverValidation:
    """Tests for OracleTextSearchRetriever model_validator without DB calls."""

    def _make_vs(self, table_name="DOCS"):
        vs = MagicMock(spec=OracleVS)
        vs.table_name = table_name
        vs.client = MagicMock()
        return vs

    # --- Mutual exclusivity ---
    def test_both_vector_store_and_table_name_raises(self):
        vs = self._make_vs()
        with pytest.raises(
            ValueError, match="Only give one of vector_store or table_name"
        ):
            OracleTextSearchRetriever(
                vector_store=vs,
                client=MagicMock(),
                table_name="OTHER",
                column_name="text",
            )

    def test_neither_vector_store_nor_table_name_raises(self):
        with pytest.raises(
            ValueError, match="Provide either vector_store or table_name"
        ):
            OracleTextSearchRetriever()

    def test_table_name_without_client_raises(self):
        with pytest.raises(ValueError, match="client must be provided"):
            OracleTextSearchRetriever(table_name="MY_TABLE", column_name="body")

    # --- column_name restrictions with vector_store ---
    def test_vector_store_with_metadata_column_raises(self):
        vs = self._make_vs()
        with pytest.raises(ValueError, match="column_name must be 'text'"):
            OracleTextSearchRetriever(vector_store=vs, column_name="metadata")

    def test_vector_store_with_title_column_raises(self):
        vs = self._make_vs()
        with pytest.raises(ValueError, match="column_name must be 'text'"):
            OracleTextSearchRetriever(vector_store=vs, column_name="title")

    # --- Happy paths ---
    def test_vector_store_text_column_accepted(self):
        vs = self._make_vs()
        r = OracleTextSearchRetriever(vector_store=vs, column_name="text")
        assert r.column_name == "text"

    def test_vector_store_default_column_is_text(self):
        vs = self._make_vs()
        r = OracleTextSearchRetriever(vector_store=vs)
        assert r.column_name == "text"

    def test_raw_table_accepted(self):
        r = OracleTextSearchRetriever(
            client=MagicMock(),
            table_name="MY_TABLE",
            column_name="body",
        )
        assert r.column_name == '"BODY"'
        assert r.table_name == '"MY_TABLE"'

    def test_raw_table_preserves_quoted_case(self):
        r = OracleTextSearchRetriever(
            client=MagicMock(),
            table_name='"MyTable"',
            column_name='"Body"',
            returned_columns=['"Title"'],
        )
        assert r.column_name == '"Body"'
        assert r.table_name == '"MyTable"'
        assert r.returned_columns == ['"Title"']

    def test_raw_table_rejects_sql_fragment(self):
        with pytest.raises(ValueError):
            OracleTextSearchRetriever(
                client=MagicMock(),
                table_name="MY_TABLE WHERE 1=1 --",
                column_name="body",
            )

    def test_returned_columns_reject_sql_fragment(self):
        with pytest.raises(ValueError):
            OracleTextSearchRetriever(
                client=MagicMock(),
                table_name="T",
                column_name="body",
                returned_columns=["metadata FROM other_table --"],
            )

    # --- returned_columns deduplication ---
    def test_returned_columns_deduplicates_main_column_vs(self):
        vs = self._make_vs()
        r = OracleTextSearchRetriever(
            vector_store=vs, column_name="text", returned_columns=["text", "metadata"]
        )
        assert "text" not in r.returned_columns
        assert '"METADATA"' in r.returned_columns

    def test_returned_columns_deduplicates_main_column_raw(self):
        r = OracleTextSearchRetriever(
            client=MagicMock(),
            table_name="T",
            column_name="body",
            returned_columns=["title", "body"],
        )
        assert '"BODY"' not in r.returned_columns
        assert '"TITLE"' in r.returned_columns

    def test_returned_columns_defaults_to_metadata_for_vs(self):
        """When vector_store is used and returned_columns is None,
        default to ['metadata']."""
        vs = self._make_vs()
        r = OracleTextSearchRetriever(vector_store=vs)
        assert r.returned_columns == ['"METADATA"']

    def test_returned_columns_defaults_to_empty_for_raw(self):
        r = OracleTextSearchRetriever(
            client=MagicMock(), table_name="T", column_name="body"
        )
        assert r.returned_columns == []

    # --- fuzzy + operator_search warning ---
    def test_fuzzy_and_operator_search_emits_warning(self, caplog):
        vs = self._make_vs()
        with caplog.at_level(logging.WARNING):
            OracleTextSearchRetriever(
                vector_store=vs,
                fuzzy=True,
                operator_search=True,
            )
        assert any("ignored" in record.message.lower() for record in caplog.records)

    # --- Defaults ---
    def test_default_k(self):
        vs = self._make_vs()
        r = OracleTextSearchRetriever(vector_store=vs)
        assert r.k == 4

    def test_default_fuzzy_false(self):
        vs = self._make_vs()
        r = OracleTextSearchRetriever(vector_store=vs)
        assert r.fuzzy is False

    def test_default_operator_search_false(self):
        vs = self._make_vs()
        r = OracleTextSearchRetriever(vector_store=vs)
        assert r.operator_search is False

    def test_default_return_scores_false(self):
        vs = self._make_vs()
        r = OracleTextSearchRetriever(vector_store=vs)
        assert r.return_scores is False

    # --- _get_result_documents: score absent when return_scores=False ---

    def test_get_result_documents_no_score_when_disabled(self):
        vs = self._make_vs()
        r = OracleTextSearchRetriever(vector_store=vs, return_scores=False)
        rows = [{"text": "hello world", "score": 42.0, "metadata": {"id": "x"}}]
        docs = r._get_result_documents(rows)
        assert len(docs) == 1
        assert "score" not in docs[0].metadata

    def test_get_result_documents_score_present_when_enabled(self):
        vs = self._make_vs()
        r = OracleTextSearchRetriever(vector_store=vs, return_scores=True)
        rows = [{"text": "hello world", "score": 42.0, "metadata": {"id": "x"}}]
        docs = r._get_result_documents(rows)
        assert docs[0].metadata["score"] == 42.0

    def test_get_result_documents_page_content_populated(self):
        vs = self._make_vs()
        r = OracleTextSearchRetriever(vector_store=vs)
        rows = [{"text": "hello world", "score": 1.0, "metadata": {}}]
        docs = r._get_result_documents(rows)
        assert docs[0].page_content == "hello world"

    def test_get_result_documents_raw_table(self):
        r = OracleTextSearchRetriever(
            client=MagicMock(),
            table_name="T",
            column_name="body",
            return_scores=True,
            returned_columns=["title"],
        )
        rows = [{"body": "some content", "title": "My Title", "score": 7.5}]
        docs = r._get_result_documents(rows)
        assert docs[0].page_content == "some content"
        assert docs[0].metadata["title"] == "My Title"
        assert docs[0].metadata["score"] == 7.5

    def test_get_result_documents_empty_rows(self):
        vs = self._make_vs()
        r = OracleTextSearchRetriever(vector_store=vs)
        assert r._get_result_documents([]) == []

    def test_get_relevant_documents_rejects_non_int_k_before_db(self):
        r = OracleTextSearchRetriever(
            client=MagicMock(),
            table_name="T",
            column_name="body",
        )
        with pytest.raises(ValueError, match="k must be a positive integer"):
            r._get_relevant_documents("query", k="1 OR 1=1")
