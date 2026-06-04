# Copyright (c) 2026, Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Unit tests for the vendored ``DistanceStrategy`` enum.

``DistanceStrategy`` used to be imported from ``langchain_community``. It is now
vendored in :mod:`langchain_oracledb.vectorstores.utils` so the package no longer
depends on ``langchain-community``. These tests lock in its values and the
backwards-compatible coercion in ``OracleVS._from_texts_helper``.
"""

from enum import Enum

import pytest

from langchain_oracledb.vectorstores import DistanceStrategy, OracleVS
from langchain_oracledb.vectorstores import utils as vs_utils


def test_is_str_enum() -> None:
    assert issubclass(DistanceStrategy, str)
    assert issubclass(DistanceStrategy, Enum)


def test_member_values() -> None:
    # These values are sent to Oracle (see ``_get_distance_function``) and must
    # match the historical ``langchain_community`` values for compatibility.
    assert {s.name: s.value for s in DistanceStrategy} == {
        "EUCLIDEAN_DISTANCE": "EUCLIDEAN_DISTANCE",
        "MAX_INNER_PRODUCT": "MAX_INNER_PRODUCT",
        "DOT_PRODUCT": "DOT_PRODUCT",
        "JACCARD": "JACCARD",
        "COSINE": "COSINE",
    }


def test_exported_from_package() -> None:
    assert DistanceStrategy is vs_utils.DistanceStrategy
    from langchain_oracledb import vectorstores

    assert "DistanceStrategy" in vectorstores.__all__


def test_value_coercion() -> None:
    assert DistanceStrategy("COSINE") is DistanceStrategy.COSINE
    assert DistanceStrategy(DistanceStrategy.COSINE) is DistanceStrategy.COSINE


def test_from_texts_helper_accepts_native_enum() -> None:
    _, _, strategy, _, _ = OracleVS._from_texts_helper(
        client=object(), distance_strategy=DistanceStrategy.DOT_PRODUCT
    )
    assert strategy is DistanceStrategy.DOT_PRODUCT


def test_from_texts_helper_coerces_foreign_enum() -> None:
    # Simulate a ``DistanceStrategy`` from another package (e.g. the old
    # ``langchain_community`` one): a distinct str-Enum with the same value.
    class ForeignDistanceStrategy(str, Enum):
        COSINE = "COSINE"

    _, _, strategy, _, _ = OracleVS._from_texts_helper(
        client=object(), distance_strategy=ForeignDistanceStrategy.COSINE
    )
    assert strategy is DistanceStrategy.COSINE
    assert isinstance(strategy, DistanceStrategy)


def test_from_texts_helper_coerces_string() -> None:
    _, _, strategy, _, _ = OracleVS._from_texts_helper(
        client=object(), distance_strategy="EUCLIDEAN_DISTANCE"
    )
    assert strategy is DistanceStrategy.EUCLIDEAN_DISTANCE


@pytest.mark.parametrize("bad", ["NOT_A_STRATEGY", None, 123])
def test_from_texts_helper_rejects_invalid(bad: object) -> None:
    with pytest.raises(TypeError):
        OracleVS._from_texts_helper(client=object(), distance_strategy=bad)
