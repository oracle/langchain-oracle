# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""Unit tests for translating OCI errors into the standard model exceptions."""

from typing import Any, Optional
from unittest.mock import MagicMock

import langchain_core.exceptions as core_exceptions
import pytest
from oci.exceptions import ServiceError

from langchain_oci.chat_models import ChatOCIGenAI
from langchain_oci.common import errors
from langchain_oci.common.async_support import OCIAsyncRequestError
from langchain_oci.common.errors import raise_translated, translate_oci_error

pytestmark = pytest.mark.skipif(
    not hasattr(core_exceptions, "ModelError"),
    reason="langchain-core predates the standard model exceptions",
)

# Error texts returned by OCI Generative AI for inputs larger than the model's
# context window.
OPENAI_OVERFLOW = (
    '{ "error": { "message": "This model\'s maximum context length is 1047576 '
    "tokens. However, your messages resulted in 1400008 tokens. Please reduce "
    'the length of the messages.", "type": "invalid_request_error", "param": '
    '"messages", "code": "context_length_exceeded" } }'
)
OPENAI_INPUT_LIMIT = (
    '{ "error": { "message": "Input tokens exceed the configured limit of 272000 '
    'tokens. Your messages resulted in 1400007 tokens.", "type": '
    '"invalid_request_error", "param": "messages", "code": '
    '"context_length_exceeded" } }'
)
GOOGLE_OVERFLOW = (
    '{ "error": { "code": 400, "message": "The input token count exceeds the '
    'maximum number of tokens allowed (1048576); input token count: 2000025.", '
    '"status": "INVALID_ARGUMENT" } }'
)
COHERE_OVERFLOW = (
    '{"id":"abc","message":"too many tokens: size limit exceeded by 2112516 '
    "tokens. Try using shorter or fewer inputs. The limit for this model is "
    '288000 tokens."}'
)
BAD_PARAMETER = "Invalid 'temperature': Value is greater than maximum: 2.0"


def _service_error(status: int, message: str = "boom", **kwargs: Any) -> ServiceError:
    return ServiceError(
        status, str(status), {"opc-request-id": "req-1"}, message, **kwargs
    )


def _standard(name: str) -> Optional[type]:
    return getattr(core_exceptions, name, None)


@pytest.mark.parametrize(
    ("status", "message", "expected"),
    [
        (400, OPENAI_OVERFLOW, "ContextOverflowError"),
        (400, OPENAI_INPUT_LIMIT, "ContextOverflowError"),
        (400, GOOGLE_OVERFLOW, "ContextOverflowError"),
        (400, COHERE_OVERFLOW, "ContextOverflowError"),
        (400, BAD_PARAMETER, "ModelInvalidRequestError"),
        # Some models answer an oversized input with an empty message.
        (400, "", "ModelInvalidRequestError"),
        (
            401,
            "The required information to complete authentication was not "
            "provided or was incorrect.",
            "ModelAuthenticationError",
        ),
        (403, "forbidden", "ModelPermissionDeniedError"),
        (
            404,
            "Entity with key meta.llama-does-not-exist not found",
            "ModelNotFoundError",
        ),
        (
            404,
            "Authorization failed or requested resource not found.",
            "ModelNotFoundError",
        ),
        (429, "Too many requests", "ModelRateLimitError"),
        (500, "Internal server error", "ModelAPIError"),
        (503, "Service unavailable", "ModelAPIError"),
    ],
)
class TestStatusMapping:
    def test_service_error(self, status: int, message: str, expected: str) -> None:
        original = _service_error(status, message)

        translated = translate_oci_error(original)

        assert isinstance(translated, _standard(expected))  # type: ignore[arg-type]
        assert isinstance(translated, core_exceptions.ModelError)
        # Still the SDK error, so existing handlers keep working.
        assert isinstance(translated, ServiceError)

    def test_async_request_error(
        self, status: int, message: str, expected: str
    ) -> None:
        original = OCIAsyncRequestError(status, message)

        translated = translate_oci_error(original)

        assert isinstance(translated, _standard(expected))  # type: ignore[arg-type]
        assert isinstance(translated, OCIAsyncRequestError)
        assert isinstance(translated, RuntimeError)


class TestTranslatedErrors:
    def test_service_error_details_are_preserved(self) -> None:
        original = _service_error(
            429,
            "Too many requests",
            operation_name="chat",
            request_endpoint="POST https://inference.example.com/20231130/actions/chat",
            target_service="generative_ai_inference",
        )

        translated = translate_oci_error(original)

        assert isinstance(translated, ServiceError)
        assert translated.status == 429
        assert translated.code == "429"
        assert translated.message == "Too many requests"
        assert translated.headers == {"opc-request-id": "req-1"}
        assert translated.request_id == "req-1"
        assert translated.operation_name == "chat"
        assert translated.request_endpoint == original.request_endpoint
        assert translated.target_service == "generative_ai_inference"
        assert "Too many requests" in str(translated)

    def test_async_error_details_are_preserved(self) -> None:
        translated = translate_oci_error(OCIAsyncRequestError(400, OPENAI_OVERFLOW))

        assert isinstance(translated, OCIAsyncRequestError)
        assert translated.status == 400
        assert translated.body == OPENAI_OVERFLOW
        assert "status 400" in str(translated)

    @pytest.mark.parametrize(
        ("status", "retryable"),
        [(400, False), (401, False), (404, False), (429, True), (500, True)],
    )
    def test_retryability_follows_the_standard_exception(
        self, status: int, retryable: bool
    ) -> None:
        translated = translate_oci_error(_service_error(status))

        assert translated.is_retryable is retryable  # type: ignore[attr-defined]

    def test_class_names_identify_the_condition(self) -> None:
        sync_error = translate_oci_error(_service_error(429))
        async_error = translate_oci_error(OCIAsyncRequestError(429, "slow down"))

        assert type(sync_error).__name__ == "OCIModelRateLimitError"
        assert type(async_error).__name__ == "OCIAsyncModelRateLimitError"
        # The same condition always yields the same class.
        assert type(translate_oci_error(_service_error(429))) is type(sync_error)

    def test_translation_is_idempotent(self) -> None:
        translated = translate_oci_error(_service_error(500))

        assert translate_oci_error(translated) is translated

    @pytest.mark.parametrize("status", [302, 409, 412, None, "400"])
    def test_statuses_without_a_standard_error_are_left_alone(
        self, status: Any
    ) -> None:
        original = ServiceError(status, "x", {}, "message")

        assert translate_oci_error(original) is original

    def test_other_exceptions_are_left_alone(self) -> None:
        original = ValueError("not an OCI error")

        assert translate_oci_error(original) is original

    def test_older_langchain_core_keeps_the_original_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(errors, "_standard_error", lambda name: None)
        original = _service_error(429)

        assert translate_oci_error(original) is original


class TestRaiseTranslated:
    def test_raises_the_translated_error_chained_to_the_original(self) -> None:
        original = _service_error(429)

        with pytest.raises(core_exceptions.ModelRateLimitError) as exc_info:
            try:
                raise original
            except ServiceError as e:
                raise_translated(e)

        assert exc_info.value.__cause__ is original

    def test_reraises_an_untranslatable_error_unchanged(self) -> None:
        original = ServiceError(409, "IncorrectState", {}, "conflict")

        with pytest.raises(ServiceError) as exc_info:
            try:
                raise original
            except ServiceError as e:
                raise_translated(e)

        assert exc_info.value is original
        assert original.__cause__ is None


class TestChatModel:
    def _llm(self, error: Exception) -> ChatOCIGenAI:
        client = MagicMock()
        client.chat.side_effect = error
        return ChatOCIGenAI(model_id="meta.llama-3.3-70b-instruct", client=client)

    def test_invoke_raises_the_standard_error(self) -> None:
        llm = self._llm(_service_error(400, OPENAI_OVERFLOW))

        with pytest.raises(core_exceptions.ContextOverflowError) as exc_info:
            llm.invoke("hello")

        assert isinstance(exc_info.value, ServiceError)
        assert exc_info.value.status == 400

    def test_stream_raises_the_standard_error(self) -> None:
        llm = self._llm(_service_error(429, "Too many requests"))

        with pytest.raises(core_exceptions.ModelRateLimitError):
            list(llm.stream("hello"))

    def test_existing_service_error_handlers_still_catch_it(self) -> None:
        llm = self._llm(_service_error(404, "Entity with key x not found"))

        with pytest.raises(ServiceError) as exc_info:
            llm.invoke("hello")

        assert isinstance(exc_info.value, core_exceptions.ModelNotFoundError)
