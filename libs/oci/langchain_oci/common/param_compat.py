# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Reactive parameter-compatibility handling for OCI GenAI 400 responses.

Some models reject request parameters at call time rather than by schema —
e.g. legacy ``openai.gpt-5`` returns ``400 unsupported_value`` for
non-default ``temperature``/``top_p``. Instead of hardcoding per-model
predicates, these helpers parse the structured error OCI returns, remove
(or rename) the rejected parameter on the outgoing request, and let the
caller retry. This works for any current or future model.

Shared by the sync path (SDK ``ChatRequest`` objects raising
``oci.exceptions.ServiceError``) and the async path (camelCase wire dicts
raising :class:`~langchain_oci.common.async_support.OCIAsyncRequestError`).
"""

import json
import warnings
from typing import Any, Optional, Tuple

from oci.exceptions import ServiceError

# Retry budget for parameter-compatibility retries on 400 responses.
PARAM_RETRY_ATTEMPTS = 3

# Wire parameter names (camelCase, as OCI reports them in error payloads)
# mapped to the SDK ChatRequest attribute names used on the sync path.
_OCI_PARAM_TO_ATTR = {
    "topP": "top_p",
    "topK": "top_k",
    "maxTokens": "max_tokens",
    "maxCompletionTokens": "max_completion_tokens",
    "frequencyPenalty": "frequency_penalty",
    "presencePenalty": "presence_penalty",
    "streamOptions": "stream_options",
}

# When OCI rejects a parameter outright, prefer renaming its value to the
# supported equivalent over dropping it (wire-name space).
_OCI_PARAM_RENAME = {
    "maxTokens": "maxCompletionTokens",
    "max_tokens": "maxCompletionTokens",
}


# Lower-cased fragment of the 400 that newer OpenAI reasoning models return
# when a chat-completions request combines function tools with reasoning.
_FUNCTION_TOOLS_REASONING_MARKER = "function tools with reasoning_effort"


def extract_unsupported_param(payload: Any) -> Tuple[Optional[str], Optional[str]]:
    """Extract ``(param, code)`` from an OCI 400 error payload.

    Accepts an ``oci.exceptions.ServiceError``, a response-body string, or
    a parsed body dict. Returns ``(None, None)`` when the error carries no
    structured parameter info; never raises on missing or ``None`` fields.

    Note: for OpenAI-style bodies (``{"error": {"param": ..., "code": ...}}``)
    the OCI SDK does not promote the nested payload into
    ``ServiceError.message`` (which is then ``None``); the deserialized body
    is merged into the exception's ``args[0]`` details dict instead.
    """
    body: Any = payload
    if isinstance(payload, ServiceError):
        details = payload.args[0] if payload.args else None
        body = details if isinstance(details, dict) else {}
        if "error" not in body and isinstance(payload.message, str):
            body = payload.message
    if isinstance(body, str):
        try:
            body = json.loads(body)
        except ValueError:
            return None, None
    if not isinstance(body, dict):
        return None, None
    err = body.get("error")
    if not isinstance(err, dict):
        # OCI envelope: {"code": "400", "message": "<json string>"} where the
        # OpenAI-style error is double-encoded inside the message string
        # (observed live on the async REST path).
        inner = body.get("message")
        if isinstance(inner, str):
            return extract_unsupported_param(inner)
        return None, None
    param = err.get("param")
    code = err.get("code")
    return (
        param if isinstance(param, str) else None,
        code if isinstance(code, str) else None,
    )


def drop_unsupported_param(chat_request: Any, param: str) -> bool:
    """Remove or rename a rejected parameter on the outgoing request.

    Works on both the sync SDK ``ChatRequest`` object and the async
    camelCase wire dict. Returns True if the request changed.
    """
    rename = _OCI_PARAM_RENAME.get(param)
    if isinstance(chat_request, dict):
        if chat_request.get(param) is None:
            return False
        value = chat_request.pop(param)
        if rename and chat_request.get(rename) is None:
            chat_request[rename] = value
        return True
    attr = _OCI_PARAM_TO_ATTR.get(param, param)
    if getattr(chat_request, attr, None) is None:
        return False
    value = getattr(chat_request, attr)
    setattr(chat_request, attr, None)
    if rename:
        rename_attr = _OCI_PARAM_TO_ATTR.get(rename, rename)
        if hasattr(chat_request, rename_attr) and (
            getattr(chat_request, rename_attr) is None
        ):
            setattr(chat_request, rename_attr, value)
    return True


def _get_request_field(chat_request: Any, attr: str, wire_name: str) -> Any:
    if isinstance(chat_request, dict):
        return chat_request.get(wire_name)
    return getattr(chat_request, attr, None)


def disable_reasoning_for_function_tools(error_payload: Any, chat_request: Any) -> bool:
    """Turn reasoning off after a "function tools with reasoning_effort" 400.

    Newer OpenAI reasoning models refuse function tools on chat completions
    unless ``reasoning_effort`` is ``none``, and reason by default, so a plain
    ``bind_tools`` call is rejected. When the request carries tools and
    reasoning is not already off, set ``reasoning_effort`` to ``NONE`` so the
    caller can retry. Returns True if the request changed.
    """
    if _FUNCTION_TOOLS_REASONING_MARKER not in str(error_payload).lower():
        return False
    if not _get_request_field(chat_request, "tools", "tools"):
        return False
    current = _get_request_field(chat_request, "reasoning_effort", "reasoningEffort")
    if isinstance(current, str) and current.upper() == "NONE":
        # Already off and still rejected: nothing left to adjust.
        return False
    if isinstance(chat_request, dict):
        chat_request["reasoningEffort"] = "NONE"
    elif hasattr(chat_request, "reasoning_effort"):
        chat_request.reasoning_effort = "NONE"
    else:
        return False
    was = f"'{current}'" if current else "the model default"
    warnings.warn(
        "This model does not accept function tools together with reasoning on "
        f"chat completions; changed reasoning_effort from {was} to 'NONE' and "
        "retrying the request. Set reasoning_effort='NONE' to avoid the extra "
        "round trip, or use ChatOCIOpenAI (Responses API) to keep reasoning with "
        "tools.",
        UserWarning,
        stacklevel=3,
    )
    return True


def adjust_request_for_param_error(error_payload: Any, chat_request: Any) -> bool:
    """Fix the request in place after a parameter-compatibility 400.

    Returns True if the request was modified and the call should be
    retried; False when the error isn't a recognizable parameter
    rejection (caller should re-raise).
    """
    param, code = extract_unsupported_param(error_payload)
    if param == "reasoning_effort" and disable_reasoning_for_function_tools(
        error_payload, chat_request
    ):
        return True
    if not param or code not in ("unsupported_value", "unsupported_parameter"):
        return False
    if not drop_unsupported_param(chat_request, param):
        return False
    action = (
        f"renamed to '{_OCI_PARAM_RENAME[param]}'"
        if code == "unsupported_parameter" and param in _OCI_PARAM_RENAME
        else "removed"
    )
    warnings.warn(
        f"OCI rejected parameter '{param}' ({code}) for this model; "
        f"{action} and retrying the request.",
        UserWarning,
        stacklevel=2,
    )
    return True
