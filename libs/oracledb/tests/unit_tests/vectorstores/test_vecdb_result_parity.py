"""Result types and ranking contracts at the VecDB boundary."""

from decimal import Decimal

import pytest

from langchain_oracledb.vectorstores._vecdb_adapter import (
    _hybrid_documents,
    _text_documents,
    as_dict,
    document,
)


def test_numeric_metadata_matches_sql_without_mutating_input():
    metadata = {"lc_oracle_content": "oracle", "count": 2, "nested": [1.25, True, None]}
    result = document({"id": "a", "metadata": metadata})
    assert type(result.metadata["count"]) is Decimal
    assert type(result.metadata["nested"][0]) is Decimal
    assert result.metadata["nested"][1] is True
    assert result.metadata["nested"][2] is None
    assert type(metadata["count"]) is int


def test_json_response_preserves_decimal_precision():
    row = as_dict(
        '{"id":"a","metadata":{"lc_oracle_content":"oracle",'
        '"number":123456789.123456789,"count":9007199254740993}}'
    )
    result = document(row)
    assert result.metadata["number"] == Decimal("123456789.123456789")
    assert result.metadata["count"] == Decimal("9007199254740993")


@pytest.mark.parametrize("scores", [False, True])
def test_keyword_result_uses_database_score_and_order(scores):
    docs = _text_documents(
        [
            ("z", {"lc_oracle_content": "most relevant"}, 37),
            ("a", {"lc_oracle_content": "less relevant"}, 12),
        ],
        scores,
    )
    assert [doc.id for doc in docs] == ["z", "a"]
    assert [doc.metadata for doc in docs] == (
        [{"score": 37}, {"score": 12}] if scores else [{}, {}]
    )


@pytest.mark.parametrize("keyword_only", [False, True])
def test_keyword_hybrid_vector_contribution_is_zero(keyword_only):
    result = _hybrid_documents(
        [("a", {"lc_oracle_content": "oracle"}, 9, 2, 58.16)],
        True,
        keyword_only=keyword_only,
    )
    assert result[0].metadata["vector_score"] == (0 if keyword_only else 58.16)
    assert result[0].metadata["score"] == 9
    assert result[0].metadata["text_score"] == 2


@pytest.mark.parametrize("hybrid", [False, True])
def test_text_search_results_preserve_user_text_metadata(hybrid):
    metadata = {"lc_oracle_content": "document body", "text": "user label"}
    result = (
        _hybrid_documents([("a", metadata, 9, 2, 7)], False)
        if hybrid
        else _text_documents([("a", metadata, 9)], False)
    )
    assert result[0].page_content == "document body"
    assert result[0].metadata == {"text": "user label"}
    assert metadata == {"lc_oracle_content": "document body", "text": "user label"}
