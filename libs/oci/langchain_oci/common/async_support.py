# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Async support utilities for OCI Generative AI.

This module provides async HTTP request handling for OCI services,
enabling true async/await support without thread pool wrappers.
"""

import json
import ssl
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, AsyncIterator, Dict, Optional

import aiohttp
import certifi
import requests

# OCI GenAI API version - must match oci.generative_ai_inference SDK version
# See: oci/generative_ai_inference/generative_ai_inference_client.py line 112
OCI_GENAI_API_VERSION = "20231130"


def _snake_to_camel(name: str) -> str:
    """Convert snake_case to camelCase."""
    components = name.split("_")
    return components[0] + "".join(x.title() for x in components[1:])


def _convert_keys_to_camel(obj: Any) -> Any:
    """Recursively convert dict keys from snake_case to camelCase."""
    if isinstance(obj, dict):
        return {_snake_to_camel(k): _convert_keys_to_camel(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_keys_to_camel(item) for item in obj]
    return obj


class OCIAsyncClient:
    """Async HTTP client for OCI Generative AI services.

    This client handles OCI request signing and async HTTP requests
    using aiohttp, enabling true async support for LLM operations.

    The client reuses aiohttp.ClientSession for connection pooling and
    performance. Call close() or use as async context manager to cleanup.
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
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None

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
        url = f"{self.service_endpoint}/{OCI_GENAI_API_VERSION}/actions/chat"

        # Convert snake_case keys to camelCase for OCI REST API
        body = {
            "compartmentId": compartment_id,
            "servingMode": _convert_keys_to_camel(serving_mode_dict),
            "chatRequest": _convert_keys_to_camel(chat_request_dict),
        }

        headers = self._sign_headers("POST", url, body, stream=stream)

        async with self._arequest("POST", url, headers, body, timeout) as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(
                    f"OCI GenAI request failed with status "
                    f"{response.status}: {error_text}"
                )

            if stream:
                async for event in self._parse_sse_async(response.content):
                    yield event
            else:
                data = await response.json()
                yield data
