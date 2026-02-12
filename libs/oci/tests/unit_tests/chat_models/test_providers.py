# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Unit tests for OCI Generative AI providers."""

import warnings

from langchain_oci.chat_models.providers import (
    CohereProvider,
    GeminiProvider,
    GenericProvider,
    MetaProvider,
    Provider,
)


class TestProviderBaseClass:
    """Tests for the Provider base class."""

    def test_normalize_params_no_transforms(self) -> None:
        """Test normalize_params returns params unchanged when no transforms defined."""
        provider = GenericProvider()
        params = {"temperature": 0.5, "max_tokens": 100}
        result = provider.normalize_params(params)
        assert result == params

    def test_normalize_params_does_not_mutate_input(self) -> None:
        """Test normalize_params does not mutate the input dictionary."""
        provider = GeminiProvider()
        params = {"max_output_tokens": 100, "temperature": 0.5}
        original_params = params.copy()
        provider.normalize_params(params)
        assert params == original_params


class TestGeminiProvider:
    """Tests for the GeminiProvider class."""

    def test_inherits_from_generic_provider(self) -> None:
        """Test GeminiProvider inherits from GenericProvider."""
        provider = GeminiProvider()
        assert isinstance(provider, GenericProvider)
        assert isinstance(provider, Provider)

    def test_normalize_params_maps_max_output_tokens(self) -> None:
        """Test max_output_tokens is mapped to max_tokens."""
        provider = GeminiProvider()
        params = {"max_output_tokens": 100, "temperature": 0.5}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = provider.normalize_params(params)

            assert "max_output_tokens" not in result
            assert result["max_tokens"] == 100
            assert result["temperature"] == 0.5

            # Should emit warning
            mapping_warnings = [x for x in w if "max_output_tokens" in str(x.message)]
            assert len(mapping_warnings) == 1
            assert "Mapped" in str(mapping_warnings[0].message)

    def test_normalize_params_prefers_max_tokens_when_both_provided(self) -> None:
        """Test max_tokens is preferred when both are provided."""
        provider = GeminiProvider()
        params = {"max_tokens": 50, "max_output_tokens": 100, "temperature": 0.5}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = provider.normalize_params(params)

            assert result["max_tokens"] == 50  # Prefer max_tokens
            assert "max_output_tokens" not in result
            assert result["temperature"] == 0.5

            # Should emit warning about both being provided
            both_warnings = [x for x in w if "Both" in str(x.message)]
            assert len(both_warnings) == 1
            assert "ignoring" in str(both_warnings[0].message).lower()

    def test_normalize_params_no_changes_when_only_max_tokens(self) -> None:
        """Test no changes when only max_tokens is provided."""
        provider = GeminiProvider()
        params = {"max_tokens": 100, "temperature": 0.5}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = provider.normalize_params(params)

            assert result == params
            # Should not emit any warnings
            mapping_warnings = [x for x in w if "max_output_tokens" in str(x.message)]
            assert len(mapping_warnings) == 0

    def test_normalize_params_empty_params(self) -> None:
        """Test normalize_params handles empty params."""
        provider = GeminiProvider()
        params: dict = {}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = provider.normalize_params(params)

            assert result == {}
            assert len(w) == 0

    def test_stop_sequence_key(self) -> None:
        """Test stop_sequence_key returns correct value."""
        provider = GeminiProvider()
        assert provider.stop_sequence_key == "stop"


class TestMetaProvider:
    """Tests for the MetaProvider class."""

    def test_inherits_from_generic_provider(self) -> None:
        """Test MetaProvider inherits from GenericProvider."""
        provider = MetaProvider()
        assert isinstance(provider, GenericProvider)

    def test_normalize_params_unchanged(self) -> None:
        """Test normalize_params returns params unchanged."""
        provider = MetaProvider()
        params = {"max_tokens": 100, "temperature": 0.5}
        result = provider.normalize_params(params)
        assert result == params


class TestCohereProvider:
    """Tests for the CohereProvider class."""

    def test_inherits_from_provider(self) -> None:
        """Test CohereProvider inherits from Provider."""
        provider = CohereProvider()
        assert isinstance(provider, Provider)

    def test_stop_sequence_key(self) -> None:
        """Test stop_sequence_key returns correct value for Cohere."""
        provider = CohereProvider()
        assert provider.stop_sequence_key == "stop_sequences"


class TestGenericProvider:
    """Tests for the GenericProvider class."""

    def test_inherits_from_provider(self) -> None:
        """Test GenericProvider inherits from Provider."""
        provider = GenericProvider()
        assert isinstance(provider, Provider)

    def test_stop_sequence_key(self) -> None:
        """Test stop_sequence_key returns correct value."""
        provider = GenericProvider()
        assert provider.stop_sequence_key == "stop"

    def test_normalize_params_unchanged(self) -> None:
        """Test normalize_params returns params unchanged by default."""
        provider = GenericProvider()
        params = {"max_tokens": 100, "temperature": 0.5}
        result = provider.normalize_params(params)
        assert result == params
