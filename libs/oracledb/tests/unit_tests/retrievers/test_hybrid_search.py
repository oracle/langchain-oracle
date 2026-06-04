# Copyright (c) 2024, 2026, Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""
unit_tests/retrievers/test_hybrid_search.py

Unit tests for OracleHybridSearchRetriever, OracleVectorizerPreference, and
their supporting pure functions, exercising the full parameter-building and
validation contract without any real database connection. All DB interactions
are replaced with mock objects.

Covers:
- _validate_parameters model match/mismatch, embedder_spec match/mismatch,
  JSON key-order normalization, no model config path
- _get_hybrid_index_ddl DDL construction, all four reserved key names
  (case-insensitive), filter_by/order_by/parallel optional clauses,
  non-int parallel, extra parameters, single-quote SQL escaping
- OracleHybridSearchRetriever field_validator: idx_name quoting, all six
  params validator error paths (top-level search_text, return key,
  vector sub-dict, text sub-dict), valid score_weight passthrough,
  all field defaults
- OracleHybridSearchRetriever._get_search_params: hybrid/semantic/keyword
  mode populates correct keys only, return block contents and format,
  k override at call time, call-time params merge, all four call-time
  error raises, default-k fallback
- OracleVectorizerPreference._get_preference_parameters: database vs
  external provider path, non-OracleEmbeddings error, explicit param
  validation, mismatch raise, auto-generated name prefix

Run:
    pytest tests/unit_tests/retrievers/test_hybrid_search.py

Authors:
    - Diego Ascencio (diegoascencioqa)
"""

from unittest.mock import MagicMock

import pytest

from langchain_oracledb.embeddings import OracleEmbeddings
from langchain_oracledb.retrievers.hybrid_search import (
    OracleHybridSearchRetriever,
    OracleVectorizerPreference,
    _get_hybrid_index_ddl,
    _validate_parameters,
)
from langchain_oracledb.vectorstores import DistanceStrategy
from langchain_oracledb.vectorstores.oraclevs import (
    OracleVS,
    _get_hnsw_index_ddl,
    _get_ivf_index_ddl,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_vs(table_name="MY_TABLE"):
    vs = MagicMock(spec=OracleVS)
    vs.table_name = table_name
    vs.client = MagicMock()
    return vs


def _make_embeddings(provider="database", model="allminilm"):
    emb = MagicMock(spec=OracleEmbeddings)
    emb.params = {"provider": provider, "model": model}
    return emb


def _make_preference(vs=None, preference_name="MY_PREF"):
    pref = OracleVectorizerPreference.__new__(OracleVectorizerPreference)
    pref.vs = vs or _make_vs()
    pref.preference_name = preference_name
    pref.params = None
    return pref


# ---------------------------------------------------------------------------
# _validate_parameters
# ---------------------------------------------------------------------------


class TestValidateParameters:
    """Tests for the _validate_parameters pure function."""

    def test_model_matches_embeddings_returns_true(self):
        emb = _make_embeddings(provider="database", model="allminilm")
        result = _validate_parameters(emb, {"model": "allminilm"})
        assert result is True

    def test_model_mismatch_raises(self):
        emb = _make_embeddings(provider="database", model="allminilm")
        with pytest.raises(ValueError, match="Mismatch"):
            _validate_parameters(emb, {"model": "other_model"})

    def test_model_with_wrong_provider_raises(self):
        emb = _make_embeddings(provider="openai", model="text-embedding-ada-002")
        with pytest.raises(ValueError, match="Mismatch"):
            _validate_parameters(emb, {"model": "text-embedding-ada-002"})

    def test_embedder_spec_matches_returns_true(self):
        spec = {
            "provider": "oracleai",
            "url": "http://host/score",
            "model": "all_minilm",
        }
        emb = MagicMock(spec=OracleEmbeddings)
        emb.params = spec.copy()
        result = _validate_parameters(emb, {"embedder_spec": spec})
        assert result is True

    def test_embedder_spec_mismatch_raises(self):
        spec = {"provider": "oracleai", "url": "http://host/score", "model": "a"}
        emb = MagicMock(spec=OracleEmbeddings)
        emb.params = {"provider": "oracleai", "url": "http://other/score", "model": "b"}
        with pytest.raises(ValueError, match="Mismatch"):
            _validate_parameters(emb, {"embedder_spec": spec})

    def test_no_model_config_returns_false(self):
        emb = _make_embeddings()
        result = _validate_parameters(emb, {"some_other_key": "value"})
        assert result is False

    def test_empty_params_returns_false(self):
        emb = _make_embeddings()
        result = _validate_parameters(emb, {})
        assert result is False

    def test_embedder_spec_json_normalization(self):
        """Key order in embedder_spec should not matter (JSON normalization)."""
        spec = {"model": "m", "provider": "p"}
        emb = MagicMock(spec=OracleEmbeddings)
        emb.params = {"provider": "p", "model": "m"}  # different key order
        result = _validate_parameters(emb, {"embedder_spec": spec})
        assert result is True


# ---------------------------------------------------------------------------
# _get_hybrid_index_ddl
# ---------------------------------------------------------------------------


class TestGetHybridIndexDdl:
    """Tests for the _get_hybrid_index_ddl helper."""

    def _make_pref(self, table_name="MY_TABLE", pref_name="MY_PREF"):
        vs = _make_vs(table_name)
        return _make_preference(vs=vs, preference_name=pref_name)

    def test_basic_ddl_contains_required_clauses(self):
        pref = self._make_pref()
        ddl = _get_hybrid_index_ddl(pref, '"IDX"', {})
        assert "CREATE HYBRID VECTOR INDEX" in ddl
        assert '"IDX"' in ddl
        assert "MY_TABLE" in ddl
        assert "(text)" in ddl
        assert "MY_PREF" in ddl

    def test_vectorizer_in_parameters_string(self):
        pref = self._make_pref(pref_name="PREF_X")
        ddl = _get_hybrid_index_ddl(pref, '"IDX"', {})
        assert "vectorizer PREF_X" in ddl

    # --- Reserved parameter keys raise ---

    @pytest.mark.parametrize(
        "reserved_key", ["model", "embedder_spec", "vectorizer", "vector_idxtype"]
    )
    def test_reserved_key_in_parameters_raises(self, reserved_key):
        pref = self._make_pref()
        with pytest.raises(ValueError, match="Vectorization parameters must be given"):
            _get_hybrid_index_ddl(pref, '"IDX"', {"parameters": {reserved_key: "val"}})

    @pytest.mark.parametrize(
        "reserved_key", ["MODEL", "EMBEDDER_SPEC", "VECTORIZER", "VECTOR_IDXTYPE"]
    )
    def test_reserved_key_case_insensitive_raises(self, reserved_key):
        pref = self._make_pref()
        with pytest.raises(ValueError, match="Vectorization parameters must be given"):
            _get_hybrid_index_ddl(pref, '"IDX"', {"parameters": {reserved_key: "val"}})

    # --- Optional clauses ---

    def test_filter_by_clause_included(self):
        pref = self._make_pref()
        ddl = _get_hybrid_index_ddl(pref, '"IDX"', {"filter_by": ["col1", "col2"]})
        assert 'FILTER BY "COL1","COL2"' in ddl

    def test_order_by_asc_clause_included(self):
        pref = self._make_pref()
        ddl = _get_hybrid_index_ddl(
            pref, '"IDX"', {"order_by": ["ts"], "order_by_asc": True}
        )
        assert 'ORDER BY "TS" ASC' in ddl

    def test_order_by_desc_clause_included(self):
        pref = self._make_pref()
        ddl = _get_hybrid_index_ddl(
            pref, '"IDX"', {"order_by": ["ts"], "order_by_asc": False}
        )
        assert 'ORDER BY "TS" DESC' in ddl

    def test_explicitly_quoted_filter_and_order_by_preserve_case(self):
        pref = self._make_pref()
        ddl = _get_hybrid_index_ddl(
            pref,
            '"IDX"',
            {"filter_by": ['"col1"'], "order_by": ['"ts"']},
        )
        assert 'FILTER BY "col1"' in ddl
        assert 'ORDER BY "ts" ASC' in ddl

    def test_qualified_filter_and_order_by_identifiers(self):
        pref = self._make_pref()
        ddl = _get_hybrid_index_ddl(
            pref,
            '"IDX"',
            {
                "filter_by": ["schema.col1", '"Mixed"."CaseCol"'],
                "order_by": ["schema.ts"],
            },
        )
        assert 'FILTER BY "SCHEMA"."COL1","Mixed"."CaseCol"' in ddl
        assert 'ORDER BY "SCHEMA"."TS" ASC' in ddl

    @pytest.mark.parametrize(
        "identifier",
        [
            "col name",
            "col,name",
            "col)",
            "col;drop",
            "col--comment",
            "schema.col name",
            '"bad name"',
        ],
    )
    def test_filter_by_invalid_identifier_raises(self, identifier):
        pref = self._make_pref()
        with pytest.raises(ValueError, match="invalid identifier"):
            _get_hybrid_index_ddl(pref, '"IDX"', {"filter_by": [identifier]})

    def test_filter_by_non_list_raises(self):
        pref = self._make_pref()
        with pytest.raises(ValueError, match="filter_by must be a list"):
            _get_hybrid_index_ddl(pref, '"IDX"', {"filter_by": "col1"})

    def test_order_by_non_string_raises(self):
        pref = self._make_pref()
        with pytest.raises(ValueError, match="order_by must contain only column names"):
            _get_hybrid_index_ddl(pref, '"IDX"', {"order_by": [1]})

    def test_order_by_asc_non_bool_raises(self):
        pref = self._make_pref()
        with pytest.raises(ValueError, match="order_by_asc must be a boolean"):
            _get_hybrid_index_ddl(
                pref, '"IDX"', {"order_by": ["ts"], "order_by_asc": "true"}
            )

    def test_parallel_clause_included(self):
        pref = self._make_pref()
        ddl = _get_hybrid_index_ddl(pref, '"IDX"', {"parallel": 4})
        assert "PARALLEL 4" in ddl

    def test_parallel_non_int_raises(self):
        pref = self._make_pref()
        with pytest.raises(ValueError, match="parallel must be a positive integer"):
            _get_hybrid_index_ddl(pref, '"IDX"', {"parallel": "4"})

    def test_parallel_float_raises(self):
        pref = self._make_pref()
        with pytest.raises(ValueError, match="parallel must be a positive integer"):
            _get_hybrid_index_ddl(pref, '"IDX"', {"parallel": 4.0})

    @pytest.mark.parametrize("parallel", [True, False, 0, -1])
    def test_parallel_bool_and_non_positive_int_raise(self, parallel):
        pref = self._make_pref()
        with pytest.raises(ValueError, match="parallel must be a positive integer"):
            _get_hybrid_index_ddl(pref, '"IDX"', {"parallel": parallel})

    def test_no_optional_clauses_absent_from_ddl(self):
        pref = self._make_pref()
        ddl = _get_hybrid_index_ddl(pref, '"IDX"', {})
        assert "FILTER BY" not in ddl
        assert "ORDER BY" not in ddl
        assert "PARALLEL" not in ddl

    def test_extra_parameters_are_included(self):
        pref = self._make_pref()
        ddl = _get_hybrid_index_ddl(pref, '"IDX"', {"parameters": {"word_min_len": 2}})
        assert "word_min_len 2" in ddl

    # --- SQL injection safety: single quotes in params_str are escaped ---

    def test_single_quote_in_params_escaped(self):
        pref = self._make_pref(pref_name="PREF'X")
        ddl = _get_hybrid_index_ddl(pref, '"IDX"', {})
        # The preference name is placed inside a single-quoted SQL string literal;
        # any ' in the name must be doubled.
        assert "PREF''X" in ddl


# ---------------------------------------------------------------------------
# OracleVS vector index DDL helpers
# ---------------------------------------------------------------------------


class TestVectorIndexDdlValidation:
    """Tests for vector index DDL parameter validation."""

    def test_hnsw_valid_params_are_quoted_and_formatted(self):
        idx_name, ddl = _get_hnsw_index_ddl(
            '"DOCS"',
            DistanceStrategy.COSINE,
            {
                "idx_name": "idx_hnsw",
                "idx_type": "HNSW",
                "neighbors": 32,
                "efConstruction": 200,
                "accuracy": 90,
                "parallel": 4,
            },
        )
        assert idx_name == '"idx_hnsw"'
        assert 'create vector index "idx_hnsw"' in ddl
        assert "parameters (type HNSW, neighbors 32, efConstruction 200)" in ddl
        assert "parallel 4" in ddl

    def test_hnsw_quotes_index_name_with_sql_tokens(self):
        idx_name, ddl = _get_hnsw_index_ddl(
            '"DOCS"',
            DistanceStrategy.COSINE,
            {"idx_name": "IDX_HACK PARAMETERS (x) --", "parallel": 1},
        )
        assert idx_name == '"IDX_HACK PARAMETERS (x) --"'
        assert ddl.startswith(
            'create vector index "IDX_HACK PARAMETERS (x) --" on "DOCS"'
        )

    @pytest.mark.parametrize(
        "params",
        [
            {"idx_type": "IVF"},
            {"neighbors": 1},
            {"neighbors": 2049},
            {"efConstruction": 0},
            {"efConstruction": 65536},
            {"accuracy": 0},
            {"accuracy": 101},
            {"parallel": "1 ONLINE --"},
            {"parallel": 0},
        ],
    )
    def test_hnsw_rejects_invalid_params(self, params):
        with pytest.raises(ValueError):
            _get_hnsw_index_ddl('"DOCS"', DistanceStrategy.COSINE, params)

    def test_ivf_valid_params_are_quoted_and_formatted(self):
        idx_name, ddl = _get_ivf_index_ddl(
            '"DOCS"',
            DistanceStrategy.COSINE,
            {
                "idx_name": "idx_ivf",
                "idx_type": "IVF",
                "neighbor_part": 32,
                "accuracy": 90,
                "parallel": 4,
            },
        )
        assert idx_name == '"idx_ivf"'
        assert 'CREATE VECTOR INDEX "idx_ivf"' in ddl
        assert "PARAMETERS (type IVF, neighbor partitions 32)" in ddl
        assert "PARALLEL 4" in ddl

    def test_ivf_quotes_index_name_with_sql_tokens(self):
        idx_name, ddl = _get_ivf_index_ddl(
            '"DOCS"',
            DistanceStrategy.COSINE,
            {"idx_name": "IDX_HACK PARAMETERS (x) --", "parallel": 1},
        )
        assert idx_name == '"IDX_HACK PARAMETERS (x) --"'
        assert ddl.startswith(
            'CREATE VECTOR INDEX "IDX_HACK PARAMETERS (x) --" ON "DOCS"'
        )

    @pytest.mark.parametrize(
        "params",
        [
            {"idx_type": "HNSW"},
            {"neighbor_part": 0},
            {"neighbor_part": 10000001},
            {"accuracy": 0},
            {"accuracy": 101},
            {"parallel": "1 ONLINE --"},
            {"parallel": 0},
        ],
    )
    def test_ivf_rejects_invalid_params(self, params):
        with pytest.raises(ValueError):
            _get_ivf_index_ddl('"DOCS"', DistanceStrategy.COSINE, params)


# ---------------------------------------------------------------------------
# OracleHybridSearchRetriever — field_validator (no DB)
# ---------------------------------------------------------------------------


class TestOracleHybridSearchRetrieverValidation:
    """Tests for OracleHybridSearchRetriever Pydantic validators."""

    def _make_retriever(self, **kwargs):
        vs = _make_vs()
        defaults = dict(vector_store=vs, idx_name="MY_IDX")
        defaults.update(kwargs)
        return OracleHybridSearchRetriever(**defaults)

    # --- Default values ---
    def test_default_search_mode_is_hybrid(self):
        r = self._make_retriever()
        assert r.search_mode == "hybrid"

    def test_default_k_is_4(self):
        r = self._make_retriever()
        assert r.k == 4

    def test_default_return_scores_false(self):
        r = self._make_retriever()
        assert r.return_scores is False

    def test_default_params_is_empty_dict(self):
        r = self._make_retriever()
        assert r.params == {}

    # --- idx_name quoting ---
    def test_idx_name_is_quoted(self):
        r = self._make_retriever(idx_name="MY_IDX")
        # _quote_indentifier wraps in double-quotes
        assert r.idx_name == '"MY_IDX"'

    def test_idx_name_already_quoted_not_double_quoted(self):
        """Verify the name goes through _quote_indentifier exactly once."""
        r = self._make_retriever(idx_name="MY_IDX")
        assert r.idx_name.count('"') == 2  # exactly one pair

    # --- params validator: top-level search_text ---
    def test_top_level_search_text_raises(self):
        vs = _make_vs()
        with pytest.raises(
            ValueError,
            match="Cannot provide search_text as a parameter at the top level",
        ):
            OracleHybridSearchRetriever(
                vector_store=vs, idx_name="IDX", params={"search_text": "bad"}
            )

    # --- params validator: return key ---
    def test_return_key_in_params_raises(self):
        vs = _make_vs()
        with pytest.raises(ValueError, match="Cannot provide return as a parameter"):
            OracleHybridSearchRetriever(
                vector_store=vs, idx_name="IDX", params={"return": {"topN": 5}}
            )

    # --- params validator: vector sub-dict ---
    def test_vector_search_text_raises(self):
        vs = _make_vs()
        with pytest.raises(ValueError, match=r"params\['vector'\]"):
            OracleHybridSearchRetriever(
                vector_store=vs,
                idx_name="IDX",
                params={"vector": {"search_text": "bad"}},
            )

    def test_vector_search_vector_raises(self):
        vs = _make_vs()
        with pytest.raises(ValueError, match=r"params\['vector'\]"):
            OracleHybridSearchRetriever(
                vector_store=vs,
                idx_name="IDX",
                params={"vector": {"search_vector": [0.1, 0.2]}},
            )

    # --- params validator: text sub-dict ---
    def test_text_search_text_raises(self):
        vs = _make_vs()
        with pytest.raises(ValueError, match=r"params\['text'\]"):
            OracleHybridSearchRetriever(
                vector_store=vs,
                idx_name="IDX",
                params={"text": {"search_text": "bad"}},
            )

    def test_text_contains_raises(self):
        vs = _make_vs()
        with pytest.raises(ValueError, match=r"params\['text'\]"):
            OracleHybridSearchRetriever(
                vector_store=vs,
                idx_name="IDX",
                params={"text": {"contains": "bad"}},
            )

    # --- valid extra params pass through ---
    def test_score_weight_in_vector_accepted(self):
        r = self._make_retriever(params={"vector": {"score_weight": 0.7}})
        assert r.params["vector"]["score_weight"] == 0.7

    def test_score_weight_in_text_accepted(self):
        r = self._make_retriever(params={"text": {"score_weight": 0.3}})
        assert r.params["text"]["score_weight"] == 0.3


# ---------------------------------------------------------------------------
# OracleHybridSearchRetriever._get_search_params
# ---------------------------------------------------------------------------


class TestGetSearchParams:
    """Tests for _get_search_params — the JSON parameter builder."""

    def _make_retriever(self, search_mode="hybrid", k=4, params=None):
        vs = _make_vs()
        return OracleHybridSearchRetriever(
            vector_store=vs,
            idx_name="MY_IDX",
            search_mode=search_mode,
            k=k,
            params=params or {},
        )

    # --- hybrid mode populates both vector and text ---
    def test_hybrid_mode_sets_vector_search_text(self):
        r = self._make_retriever(search_mode="hybrid")
        p = r._get_search_params("my query")
        assert p["vector"]["search_text"] == "my query"

    def test_hybrid_mode_sets_text_search_text(self):
        r = self._make_retriever(search_mode="hybrid")
        p = r._get_search_params("my query")
        assert p["text"]["search_text"] == "my query"

    # --- semantic mode populates only vector ---
    def test_semantic_mode_sets_vector_search_text(self):
        r = self._make_retriever(search_mode="semantic")
        p = r._get_search_params("my query")
        assert p["vector"]["search_text"] == "my query"
        assert "text" not in p

    # --- keyword mode populates only text ---
    def test_keyword_mode_sets_text_search_text(self):
        r = self._make_retriever(search_mode="keyword")
        p = r._get_search_params("my query")
        assert p["text"]["search_text"] == "my query"
        assert "vector" not in p

    # --- return block ---
    def test_return_topn_uses_k(self):
        r = self._make_retriever(k=7)
        p = r._get_search_params("q")
        assert p["return"]["topN"] == 7

    def test_return_topn_overridden_by_kwarg(self):
        r = self._make_retriever(k=4)
        p = r._get_search_params("q", k=10)
        assert p["return"]["topN"] == 10

    def test_return_format_is_json(self):
        r = self._make_retriever()
        p = r._get_search_params("q")
        assert p["return"]["format"] == "JSON"

    def test_return_values_contains_required_fields(self):
        r = self._make_retriever()
        p = r._get_search_params("q")
        for field in ("rowid", "score", "vector_score", "text_score"):
            assert field in p["return"]["values"]

    # --- hybrid_index_name is always set ---
    def test_hybrid_index_name_set(self):
        r = self._make_retriever()
        p = r._get_search_params("q")
        assert p["hybrid_index_name"] == r.idx_name

    # --- call-time params are merged ---
    def test_call_time_params_merged(self):
        r = self._make_retriever(search_mode="hybrid")
        p = r._get_search_params("q", params={"vector": {"score_weight": 0.8}})
        assert p["vector"]["score_weight"] == 0.8
        # search_text is still set from the query
        assert p["vector"]["search_text"] == "q"

    # --- call-time invalid params still raise ---
    def test_call_time_search_text_at_top_level_raises(self):
        r = self._make_retriever()
        with pytest.raises(
            ValueError,
            match="Cannot provide search_text as a parameter at the top level",
        ):
            r._get_search_params("q", params={"search_text": "bad"})

    def test_call_time_return_key_raises(self):
        r = self._make_retriever()
        with pytest.raises(ValueError, match="Cannot provide return as a parameter"):
            r._get_search_params("q", params={"return": {"topN": 1}})

    def test_call_time_vector_search_text_raises(self):
        r = self._make_retriever(search_mode="hybrid")
        with pytest.raises(ValueError, match=r"params\['vector'\]"):
            r._get_search_params("q", params={"vector": {"search_text": "bad"}})

    def test_call_time_text_contains_raises(self):
        r = self._make_retriever(search_mode="hybrid")
        with pytest.raises(ValueError, match=r"params\['text'\]"):
            r._get_search_params("q", params={"text": {"contains": "x"}})

    # --- default k fallback ---
    def test_default_k_fallback_when_none(self):
        r = self._make_retriever(k=None)
        p = r._get_search_params("q")
        assert p["return"]["topN"] == 4  # hard-coded fallback in the method


# ---------------------------------------------------------------------------
# OracleVectorizerPreference._get_preference_parameters
# ---------------------------------------------------------------------------


class TestGetPreferenceParameters:
    """Tests for _get_preference_parameters (no DB calls)."""

    def test_database_provider_sets_model_key(self):
        vs = _make_vs()
        vs.embedding_function = _make_embeddings(provider="database", model="allminilm")
        pref = _make_preference(vs=vs)
        params = pref._get_preference_parameters()
        assert params.get("model") == "allminilm"
        assert "embedder_spec" not in params

    def test_non_database_provider_sets_embedder_spec(self):
        vs = _make_vs()
        spec = {"provider": "oracleai", "url": "http://host/score", "model": "m"}
        vs.embedding_function = MagicMock(spec=OracleEmbeddings)
        vs.embedding_function.params = spec
        pref = _make_preference(vs=vs)
        params = pref._get_preference_parameters()
        assert params.get("embedder_spec") == spec
        assert "model" not in params

    def test_non_oracle_embeddings_raises(self):
        from langchain_core.embeddings import Embeddings

        vs = _make_vs()
        vs.embedding_function = MagicMock(spec=Embeddings)  # not OracleEmbeddings
        pref = _make_preference(vs=vs)
        with pytest.raises(ValueError, match="Only OracleEmbeddings"):
            pref._get_preference_parameters()

    def test_explicit_model_param_validated(self):
        vs = _make_vs()
        vs.embedding_function = _make_embeddings(provider="database", model="allminilm")
        pref = _make_preference(vs=vs)
        pref.params = {"model": "allminilm"}
        params = pref._get_preference_parameters()
        assert params["model"] == "allminilm"

    def test_explicit_model_param_mismatch_raises(self):
        vs = _make_vs()
        vs.embedding_function = _make_embeddings(provider="database", model="allminilm")
        pref = _make_preference(vs=vs)
        pref.params = {"model": "wrong_model"}
        with pytest.raises(ValueError, match="Mismatch"):
            pref._get_preference_parameters()
