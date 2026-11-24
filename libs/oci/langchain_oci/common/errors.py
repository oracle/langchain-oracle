# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Translate OCI Generative AI errors into LangChain's standard model exceptions.

``langchain-core`` defines one exception type per failure condition
(``ModelRateLimitError``, ``ContextOverflowError``, ...) so that middleware can
decide whether to retry, fall back or summarise without knowing the provider.

The translated exceptions also inherit from the error type that was raised
before (``oci.exceptions.ServiceError`` on the sync paths,
:class:`~langchain_oci.common.async_support.OCIAsyncRequestError` on the async
ones), so existing ``except`` clauses and attribute access keep working.

``langchain-core`` releases that predate the standard exceptions are still
supported: there the original error is raised unchanged.
"""

from __future__ import annotations

import re
from typing import Any, Dict, NoReturn, Optional, Tuple, Type

import langchain_core.exceptions as _core_exceptions
from oci.exceptions import ServiceError

from langchain_oci.common.async_support import OCIAsyncRequestError

# Messages OCI Generative AI returns (HTTP 400) when the input does not fit the
# model's context window, as observed per model family:
#   OpenAI: "This model's maximum context length is N tokens. However, ..."
#           "Input tokens exceed the configured limit of N tokens."
#           (both with code "context_length_exceeded")
#   Google: "The input token count exceeds the maximum number of tokens allowed"
#   Cohere: "too many tokens: size limit exceeded by N tokens."
_CONTEXT_OVERFLOW_PATTERN = re.compile(
    r"context_length_exceeded"
    r"|maximum context length"
    r"|input tokens exceed"
    r"|input token count exceeds"
    r"|too many tokens",
    re.IGNORECASE,
)

_SERVICE_ERROR_KWARGS = (
    "original_request",
    "operation_name",
    "api_reference_link",
    "target_service",
    "request_endpoint",
    "client_version",
    "timestamp",
)

_translated_types: Dict[Tuple[type, type], type] = {}


def _standard_error(name: str) -> Optional[Type[Exception]]:
    """Return a standard exception class, or None on older langchain-core."""
    return getattr(_core_exceptions, name, None)


def _standard_error_for(status: Any, text: str) -> Optional[Type[Exception]]:
    """Pick the standard exception for an HTTP status and error text."""
    if not isinstance(status, int):
        return None
    if status == 400:
        if _CONTEXT_OVERFLOW_PATTERN.search(text or ""):
            overflow = _standard_error("ContextOverflowError")
            if overflow is not None:
                return overflow
        return _standard_error("ModelInvalidRequestError")
    if status == 401:
        return _standard_error("ModelAuthenticationError")
    if status == 403:
        return _standard_error("ModelPermissionDeniedError")
    if status == 404:
        # OCI also answers 404 when the caller is not authorized for the
        # resource ("Authorization failed or requested resource not found").
        return _standard_error("ModelNotFoundError")
    if status == 429:
        return _standard_error("ModelRateLimitError")
    if status >= 500:
        return _standard_error("ModelAPIError")
    return None


def _translated_type(original: type, standard: type) -> type:
    """Build (once) a class that is both the original and the standard error."""
    key = (original, standard)
    if key not in _translated_types:
        prefix = "OCIAsync" if original is OCIAsyncRequestError else "OCI"
        _translated_types[key] = type(
            f"{prefix}{standard.__name__}",
            (original, standard),
            {"__module__": __name__, "__doc__": standard.__doc__},
        )
    return _translated_types[key]


def translate_oci_error(error: BaseException) -> BaseException:
    """Return the standard-exception form of an OCI error.

    The result is an instance of both the original error type and the matching
    ``langchain_core.exceptions.ModelError`` subclass. ``error`` itself is
    returned when it is not an OCI request error, when its status has no
    standard counterpart, or when the installed ``langchain-core`` does not
    define the standard exceptions.
    """
    if _standard_error("ModelError") is None:
        return error
    if isinstance(error, _core_exceptions.ModelError):
        return error

    if isinstance(error, ServiceError):
        standard = _standard_error_for(error.status, f"{error.code} {error.message}")
        if standard is None:
            return error
        kwargs = {
            name: getattr(error, name, None)
            for name in _SERVICE_ERROR_KWARGS
            if getattr(error, name, None) is not None
        }
        translated: BaseException = _translated_type(ServiceError, standard)(
            error.status, error.code, error.headers, error.message, **kwargs
        )
        return translated

    if isinstance(error, OCIAsyncRequestError):
        standard = _standard_error_for(error.status, error.body)
        if standard is None:
            return error
        translated = _translated_type(OCIAsyncRequestError, standard)(
            error.status, error.body
        )
        return translated

    return error


def raise_translated(error: BaseException) -> NoReturn:
    """Raise the standard-exception form of ``error``.

    Call from inside the ``except`` block that caught ``error``: an error with
    no standard counterpart is re-raised as is.
    """
    translated = translate_oci_error(error)
    if translated is error:
        raise error
    raise translated from error
