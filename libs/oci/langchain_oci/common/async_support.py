# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Async support utilities for OCI Generative AI.

This module provides async HTTP request handling for OCI services,
enabling true async/await support without thread pool wrappers.

When the installed ``oci`` SDK ships the native
``AsyncGenerativeAiInferenceClient`` (aiohttp-based), all operations
delegate to it so signing, retries, and error handling come from the SDK.
Older SDKs fall back to the aiohttp transport implemented here.
"""

import json
import logging
import ssl
from contextlib import asynccontextmanager
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    Optional,
)

import aiohttp
import certifi
import requests
from oci.exceptions import ServiceError

logger = logging.getLogger(__name__)


def _load_native_async_client_cls() -> Optional[type]:
    """Return the SDK's native async GenAI client class, or None.

    ``AsyncGenerativeAiInferenceClient`` was added to the ``oci`` SDK after
    2.184.2; on older SDKs the import fails and callers use the fallback
    aiohttp transport in this module.
    """
    try:
        from oci.generative_ai_inference import AsyncGenerativeAiInferenceClient
    except (ImportError, AttributeError):
        return None
    return AsyncGenerativeAiInferenceClient


def _service_error_body(error: ServiceError) -> str:
    """Rebuild the OCI error envelope from a ServiceError.

    Produces the same ``{"code": ..., "message": ...}`` JSON the REST
    fallback surfaces as :attr:`OCIAsyncRequestError.body`, so structured
    handling (e.g. the parameter-compatibility retry in
    ``common/param_compat.py``) works identically on both transports.
    """
    return json.dumps({"code": error.code, "message": error.message})


def _get_oci_genai_api_version() -> str:
    """Get OCI GenAI API version, attempting to detect from SDK.

    Falls back to known version if detection fails.
    """
    try:
        from oci.generative_ai_inference import GenerativeAiInferenceClient

        # Try to get version from client class if available
        if hasattr(GenerativeAiInferenceClient, "API_VERSION"):
            return GenerativeAiInferenceClient.API_VERSION
    except ImportError:
        pass
    # Fallback to known version
    # See: oci/generative_ai_inference/generative_ai_inference_client.py
    return "20231130"


OCI_GENAI_API_VERSION = _get_oci_genai_api_version()


class OCIAsyncRequestError(RuntimeError):
    """Non-200 response from the async OCI GenAI REST call.

    Subclasses ``RuntimeError`` (the exception previously raised here) so
    existing ``except RuntimeError`` handlers keep working, while exposing
    the HTTP ``status`` and raw response ``body`` for structured handling —
    e.g. the parameter-compatibility retry in the async chat paths.
    """

    def __init__(self, status: int, body: str):
        super().__init__(f"OCI GenAI request failed with status {status}: {body}")
        self.status = status
        self.body = body


class OCIAsyncClient:
    """Async HTTP client for OCI Generative AI services.

    When the installed ``oci`` SDK provides the native
    ``AsyncGenerativeAiInferenceClient``, every operation delegates to it
    (constructed with ``skip_deserialization=True`` so responses stay the
    camelCase wire dicts this module has always yielded). On older SDKs the
    client signs and sends requests itself with aiohttp.

    The client reuses aiohttp.ClientSession for connection pooling and
    performance. Call close() or use as async context manager to cleanup.

    Note: per-call ``timeout`` arguments only apply on the fallback
    transport; the native client uses the SDK's client-level timeout
    (same 300s default).
    """

    def __init__(
        self,
        service_endpoint: str,
        signer: Any,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the async client.

        Args:
            service_endpoint: The OCI service endpoint URL.
            signer: OCI signer for request authentication.
            config: OCI config dictionary (used for API_KEY auth when signer is None).
        """
        self.service_endpoint = service_endpoint.rstrip("/")
        self.signer = signer
        self.config = config or {}
        self._session: Optional[aiohttp.ClientSession] = None
        self._ensure_signer()
        self._native = self._build_native_client()

    def _build_native_client(self) -> Optional[Any]:
        """Construct the SDK's native async client when available.

        Mirrors the arguments the sync client was built with (same config,
        signer, and endpoint), so authentication behaves identically. Any
        construction failure falls back to the aiohttp transport rather
        than breaking async support.
        """
        native_cls = _load_native_async_client_cls()
        if native_cls is None:
            return None
        try:
            return native_cls(
                dict(self.config),
                signer=self.signer,
                service_endpoint=self.service_endpoint,
                skip_deserialization=True,
            )
        except Exception as e:
            logger.debug(
                "Falling back to built-in async transport; native "
                "AsyncGenerativeAiInferenceClient construction failed: %s",
                e,
            )
            return None

    async def _native_call(
        self,
        op: Callable[[Dict[str, Any]], Awaitable[Any]],
        details: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run a non-streaming native operation, translating errors.

        Returns the raw response dict (``skip_deserialization=True`` keeps
        the SDK from turning it into model objects).
        """
        try:
            response = await op(details)
        except ServiceError as e:
            raise OCIAsyncRequestError(e.status, _service_error_body(e)) from e
        data: Dict[str, Any] = response.data
        return data

    async def _native_stream(
        self,
        resource_path: str,
        body: Dict[str, Any],
        operation_name: str,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream SSE events through the native client, translating errors.

        The SDK's generated operations are non-streaming; SSE is exposed on
        ``AsyncBaseClient.call_api_stream``, which yields parsed event dicts
        (the same shape ``_parse_sse_async`` yields on the fallback path).
        """
        assert self._native is not None
        stream = self._native.async_base_client.call_api_stream(
            resource_path=resource_path,
            method="POST",
            body=body,
            operation_name=operation_name,
        )
        try:
            async for event in stream:
                yield event
        except ServiceError as e:
            raise OCIAsyncRequestError(e.status, _service_error_body(e)) from e

    def _ensure_signer(self) -> None:
        """Ensure we have a signer for request signing."""
        if self.signer is not None:
            return

        # For API_KEY auth, create signer from config using SDK's from_config
        if self.config:
            try:
                from oci.signer import Signer

                self.signer = Signer.from_config(self.config)
            except Exception as e:
                raise ValueError(f"Failed to create OCI signer from config: {e}") from e

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create a reusable aiohttp ClientSession.

        Reusing sessions improves performance via connection pooling
        and avoids repeated SSL handshake overhead.
        """
        if self._session is None or self._session.closed:
            ssl_context = ssl.create_default_context(cafile=certifi.where())
            connector = aiohttp.TCPConnector(ssl=ssl_context)
            self._session = aiohttp.ClientSession(connector=connector)
        return self._session

    async def close(self) -> None:
        """Close the HTTP session and release resources."""
        if self._native is not None:
            await self._native.close()
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None

    async def __aenter__(self) -> "OCIAsyncClient":
        """Enter async context manager."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit async context manager, closing the session."""
        await self.close()

    def _sign_headers(
        self,
        method: str,
        url: str,
        body: Optional[Dict[str, Any]] = None,
        stream: bool = False,
    ) -> Dict[str, str]:
        """Sign request headers using OCI signer.

        Args:
            method: HTTP method (POST, GET, etc.).
            url: Request URL.
            body: Request body as dictionary.
            stream: Whether this is a streaming request.

        Returns:
            Dictionary of signed headers.
        """
        # Create a requests.Request to sign
        req = requests.Request(method, url, json=body)
        prepared = req.prepare()

        # Sign the request
        signed = self.signer(prepared)

        # Extract signed headers
        headers = dict(signed.headers)

        # Add streaming headers if needed
        if stream:
            headers["Accept"] = "text/event-stream"

        return headers

    @asynccontextmanager
    async def _arequest(
        self,
        method: str,
        url: str,
        headers: Dict[str, str],
        json_body: Optional[Dict[str, Any]] = None,
        timeout: int = 300,
    ) -> AsyncGenerator[aiohttp.ClientResponse, None]:
        """Make an async HTTP request.

        Args:
            method: HTTP method.
            url: Request URL.
            headers: Request headers (should be signed).
            json_body: Request body as dictionary.
            timeout: Request timeout in seconds.

        Yields:
            aiohttp.ClientResponse object.
        """
        session = await self._get_session()
        client_timeout = aiohttp.ClientTimeout(total=timeout)
        async with session.request(
            method,
            url,
            headers=headers,
            json=json_body,
            timeout=client_timeout,
        ) as response:
            yield response

    async def _parse_sse_async(
        self,
        content: aiohttp.StreamReader,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Parse Server-Sent Events from async stream.

        Args:
            content: aiohttp StreamReader object.

        Yields:
            Parsed JSON objects from SSE data lines.

        Note:
            Newlines within LLM response content don't affect parsing because:
            1. SSE uses double newline (\\n\\n) as event delimiter, not single \\n
            2. We only process lines starting with "data:" prefix
            3. Content is JSON-encoded, so literal newlines become escaped \\n
        """
        async for line in content:
            line = line.strip()
            if not line:
                continue

            decoded = line.decode("utf-8")

            # SSE format: "data: {...}"
            if decoded.lower().startswith("data:"):
                data = decoded[5:].strip()
                if data and not data.startswith("[DONE]"):
                    try:
                        yield json.loads(data)
                    except json.JSONDecodeError:
                        continue

    async def chat_async(
        self,
        compartment_id: str,
        chat_request_dict: Dict[str, Any],
        serving_mode_dict: Dict[str, Any],
        stream: bool = False,
        timeout: int = 300,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Make async chat request to OCI GenAI.

        Args:
            compartment_id: OCI compartment OCID.
            chat_request_dict: Chat request as dictionary.
            serving_mode_dict: Serving mode as dictionary.
            stream: Whether to stream the response.
            timeout: Request timeout in seconds.

        Yields:
            For streaming: SSE event dictionaries.
            For non-streaming: Single response dictionary.
        """
        body = {
            "compartmentId": compartment_id,
            "servingMode": serving_mode_dict,
            "chatRequest": chat_request_dict,
        }

        if self._native is not None:
            if stream:
                async for event in self._native_stream("/actions/chat", body, "chat"):
                    yield event
            else:
                yield await self._native_call(self._native.chat, body)
            return

        url = f"{self.service_endpoint}/{OCI_GENAI_API_VERSION}/actions/chat"
        headers = self._sign_headers("POST", url, body, stream=stream)

        async with self._arequest("POST", url, headers, body, timeout) as response:
            if response.status != 200:
                error_text = await response.text()
                raise OCIAsyncRequestError(response.status, error_text)

            if stream:
                async for event in self._parse_sse_async(response.content):
                    yield event
            else:
                data = await response.json()
                yield data

    async def embed_text_async(
        self,
        embed_text_details_dict: Dict[str, Any],
        timeout: int = 300,
    ) -> Dict[str, Any]:
        """Make async embed-text request to OCI GenAI.

        Args:
            embed_text_details_dict: Serialized EmbedTextDetails (includes
                compartmentId, servingMode, inputs, ...), as produced by the
                SDK's ``sanitize_for_serialization``.
            timeout: Request timeout in seconds.

        Returns:
            The embed-text response dictionary (``embeddings`` key holds the
            vectors).
        """
        if self._native is not None:
            return await self._native_call(
                self._native.embed_text, embed_text_details_dict
            )

        url = f"{self.service_endpoint}/{OCI_GENAI_API_VERSION}/actions/embedText"
        headers = self._sign_headers("POST", url, embed_text_details_dict)

        async with self._arequest(
            "POST", url, headers, embed_text_details_dict, timeout
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise OCIAsyncRequestError(response.status, error_text)
            data: Dict[str, Any] = await response.json()
            return data

    async def rerank_text_async(
        self,
        rerank_text_details_dict: Dict[str, Any],
        timeout: int = 300,
    ) -> Dict[str, Any]:
        """Make async rerank request to OCI GenAI.

        Args:
            rerank_text_details_dict: Serialized RerankTextDetails (includes
                compartmentId, servingMode, input, documents, ...), as
                produced by the SDK's ``sanitize_for_serialization``.
            timeout: Request timeout in seconds.

        Returns:
            The rerank response dictionary (``documentRanks`` key holds the
            per-document scores).
        """
        if self._native is not None:
            return await self._native_call(
                self._native.rerank_text, rerank_text_details_dict
            )

        url = f"{self.service_endpoint}/{OCI_GENAI_API_VERSION}/actions/rerankText"
        headers = self._sign_headers("POST", url, rerank_text_details_dict)

        async with self._arequest(
            "POST", url, headers, rerank_text_details_dict, timeout
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise OCIAsyncRequestError(response.status, error_text)
            data: Dict[str, Any] = await response.json()
            return data

    async def generate_text_async(
        self,
        generate_text_details_dict: Dict[str, Any],
        stream: bool = False,
        timeout: int = 300,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Make async generate-text (completion) request to OCI GenAI.

        Args:
            generate_text_details_dict: Serialized GenerateTextDetails
                (includes compartmentId, servingMode, inferenceRequest), as
                produced by the SDK's ``sanitize_for_serialization``.
            stream: Whether to stream the response.
            timeout: Request timeout in seconds.

        Yields:
            For streaming: SSE event dictionaries.
            For non-streaming: a single response dictionary.
        """
        if self._native is not None:
            if stream:
                async for event in self._native_stream(
                    "/actions/generateText", generate_text_details_dict, "generateText"
                ):
                    yield event
            else:
                yield await self._native_call(
                    self._native.generate_text, generate_text_details_dict
                )
            return

        url = f"{self.service_endpoint}/{OCI_GENAI_API_VERSION}/actions/generateText"
        headers = self._sign_headers(
            "POST", url, generate_text_details_dict, stream=stream
        )

        async with self._arequest(
            "POST", url, headers, generate_text_details_dict, timeout
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise OCIAsyncRequestError(response.status, error_text)

            if stream:
                async for event in self._parse_sse_async(response.content):
                    yield event
            else:
                data = await response.json()
                yield data
