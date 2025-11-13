# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Common utility functions for LLM APIs."""

import re
from typing import List, Dict

import httpx
from openai import DefaultHttpxClient, DefaultAsyncHttpxClient


def enforce_stop_tokens(text: str, stop: List[str]) -> str:
    """Cut off the text as soon as any stop words occur."""
    return re.split("|".join(stop), text, maxsplit=1)[0]


def get_base_url(region: str, override_url: str = "") -> str:
    return (
        override_url
        if override_url
        else f"https://inference.generativeai.{region}.oci.oraclecloud.com/openai/v1"
    )


def get_sync_httpx_client(
    auth: httpx.Auth, headers: Dict[str, str]
) -> httpx.Client:
    return DefaultHttpxClient(
        auth=auth,
        headers=headers,
    )


def get_async_httpx_client(
    auth: httpx.Auth, headers: Dict[str, str]
) -> httpx.AsyncClient:
    return DefaultAsyncHttpxClient(
        auth=auth,
        headers=headers,
    )
