# Copyright (c) 2025, 2026, Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.outputs import Generation

from langchain_oracledb.cache import (
    OracleSemanticCache,
    _dumps_generations,
    _loads_generations,
)


def test_cache_generation_serialization_round_trip() -> None:
    generations = [Generation(text="hello"), Generation(text="world")]

    serialized = _dumps_generations(generations)

    assert _loads_generations(serialized) == generations


def test_cache_loads_legacy_generation_format() -> None:
    legacy_payload = '[{"text": "legacy", "generation_info": {"kind": "old"}}]'

    assert _loads_generations(legacy_payload) == [
        Generation(text="legacy", generation_info={"kind": "old"})
    ]


def test_cache_loads_returns_none_for_malformed_payload() -> None:
    assert _loads_generations("not-json") is None


def test_cache_validates_constructor_arguments() -> None:
    embedding = DeterministicFakeEmbedding(size=6)

    with pytest.raises(ValueError, match="client must be provided"):
        OracleSemanticCache(client=None, embedding=embedding)

    with pytest.raises(ValueError, match="score_threshold must be non-negative"):
        OracleSemanticCache(
            client=object(),
            embedding=embedding,
            score_threshold=-0.1,
        )

    with pytest.raises(ValueError, match='Identifier name bad"name is not valid'):
        OracleSemanticCache(
            client=object(),
            embedding=embedding,
            table_name='bad"name',
        )

    with pytest.raises(ValueError, match='Identifier name bad"name is not valid'):
        OracleSemanticCache(
            client=object(),
            embedding=embedding,
            index_name='bad"name',
        )
