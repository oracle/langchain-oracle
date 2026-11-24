# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""Unit tests for the ``timeout`` and ``max_retries`` client settings."""

from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import oci
import pytest

from langchain_oci.chat_models import ChatOCIGenAI
from langchain_oci.common.auth import (
    _resolve_retry_strategy,
    create_oci_client_kwargs,
)


def _max_attempts(strategy: Any) -> Optional[int]:
    for checker in strategy.checkers.checkers:
        if hasattr(checker, "max_attempts"):
            return checker.max_attempts
    return None


def _client_kwargs(**kwargs: Any) -> Dict[str, Any]:
    with patch("oci.config.from_file", return_value={}):
        return create_oci_client_kwargs(auth_type="API_KEY", **kwargs)


class TestResolveRetryStrategy:
    def test_none_keeps_the_sdk_default(self) -> None:
        assert _resolve_retry_strategy(None) is oci.retry.DEFAULT_RETRY_STRATEGY

    def test_zero_disables_retries(self) -> None:
        assert isinstance(_resolve_retry_strategy(0), oci.retry.NoneRetryStrategy)

    @pytest.mark.parametrize("max_retries", [1, 2, 5])
    def test_counts_retries_after_the_first_attempt(self, max_retries: int) -> None:
        strategy = _resolve_retry_strategy(max_retries)

        assert _max_attempts(strategy) == max_retries + 1

    def test_retries_the_same_errors_as_the_sdk_default(self) -> None:
        def service_checker(strategy: Any) -> Dict[str, Any]:
            return next(
                vars(checker)
                for checker in strategy.checkers.checkers
                if hasattr(checker, "service_error_retry_config")
            )

        assert service_checker(_resolve_retry_strategy(2)) == service_checker(
            oci.retry.DEFAULT_RETRY_STRATEGY
        )

    @pytest.mark.parametrize("max_retries", [-1, 1.5, "3", True])
    def test_rejects_invalid_values(self, max_retries: Any) -> None:
        with pytest.raises(ValueError, match="max_retries"):
            _resolve_retry_strategy(max_retries)


class TestCreateClientKwargs:
    def test_defaults_are_unchanged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OCI_REQUEST_TIMEOUT", raising=False)

        kwargs = _client_kwargs()

        assert kwargs["timeout"] == (10, 240)
        assert kwargs["retry_strategy"] is oci.retry.DEFAULT_RETRY_STRATEGY

    def test_timeout_and_max_retries_are_applied(self) -> None:
        kwargs = _client_kwargs(timeout=30, max_retries=2)

        assert kwargs["timeout"] == (10, 30)
        assert _max_attempts(kwargs["retry_strategy"]) == 3

    def test_connect_and_read_timeout_tuple(self) -> None:
        assert _client_kwargs(timeout=(5, 60))["timeout"] == (5, 60)


class TestChatModelConstructor:
    def _build(self, **kwargs: Any) -> Dict[str, Any]:
        with patch("oci.config.from_file", return_value={}):
            with patch(
                "oci.generative_ai_inference.GenerativeAiInferenceClient"
            ) as client_cls:
                ChatOCIGenAI(
                    model_id="meta.llama-3.3-70b-instruct",
                    compartment_id="ocid1.compartment.oc1..example",
                    service_endpoint="https://inference.example.com",
                    **kwargs,
                )
        return dict(client_cls.call_args.kwargs)

    def test_passes_timeout_and_max_retries_to_the_client(self) -> None:
        client_kwargs = self._build(timeout=45, max_retries=1)

        assert client_kwargs["timeout"] == (10, 45)
        assert _max_attempts(client_kwargs["retry_strategy"]) == 2

    def test_max_retries_zero_disables_retries(self) -> None:
        client_kwargs = self._build(max_retries=0)

        assert isinstance(client_kwargs["retry_strategy"], oci.retry.NoneRetryStrategy)

    def test_unset_keeps_the_existing_defaults(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OCI_REQUEST_TIMEOUT", raising=False)

        client_kwargs = self._build()

        assert client_kwargs["timeout"] == (10, 240)
        assert client_kwargs["retry_strategy"] is oci.retry.DEFAULT_RETRY_STRATEGY

    def test_invalid_max_retries_is_reported(self) -> None:
        with pytest.raises(ValueError, match="max_retries"):
            self._build(max_retries=-1)

    def test_fields_are_accepted_with_an_injected_client(self) -> None:
        llm = ChatOCIGenAI(
            model_id="meta.llama-3.3-70b-instruct",
            client=MagicMock(),
            timeout=(5, 20),
            max_retries=3,
        )

        assert llm.timeout == (5, 20)
        assert llm.max_retries == 3
