# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Common utilities for OCI agents."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Union

from langchain_oci.common.auth import OCIAuthType


@dataclass(frozen=True)
class OCIConfig:
    """Resolved OCI configuration for agent creation."""

    compartment_id: str
    service_endpoint: str
    auth_type: str
    auth_profile: str
    auth_file_location: str

    @classmethod
    def resolve(
        cls,
        compartment_id: str | None = None,
        service_endpoint: str | None = None,
        auth_type: Union[str, OCIAuthType] = OCIAuthType.API_KEY,
        auth_profile: str = "DEFAULT",
        auth_file_location: str = "~/.oci/config",
        default_region: str = "us-chicago-1",
        require_compartment: bool = True,
    ) -> "OCIConfig":
        """Resolve OCI configuration from parameters and environment.

        Args:
            compartment_id: OCI compartment OCID, or None to read from env.
            service_endpoint: Service endpoint, or None to construct from region.
            auth_type: Authentication type as string or enum.
            auth_profile: OCI config profile name.
            auth_file_location: Path to OCI config file.
            default_region: Default region if OCI_REGION not set.
            require_compartment: If True, raise ValueError when compartment missing.

        Returns:
            Resolved OCIConfig instance.

        Raises:
            ValueError: If compartment_id is required but not available.
        """
        # Resolve compartment_id
        resolved_compartment = compartment_id or os.environ.get("OCI_COMPARTMENT_ID")
        if require_compartment and not resolved_compartment:
            raise ValueError(
                "compartment_id must be provided or set via "
                "OCI_COMPARTMENT_ID environment variable"
            )

        # Resolve service_endpoint
        resolved_endpoint = service_endpoint or os.environ.get("OCI_SERVICE_ENDPOINT")
        if not resolved_endpoint:
            region = os.environ.get("OCI_REGION", default_region)
            resolved_endpoint = (
                f"https://inference.generativeai.{region}.oci.oraclecloud.com"
            )

        # Normalize auth_type to string
        if isinstance(auth_type, OCIAuthType):
            auth_type_str = auth_type.name
        else:
            auth_type_str = auth_type

        return cls(
            compartment_id=resolved_compartment or "",
            service_endpoint=resolved_endpoint,
            auth_type=auth_type_str,
            auth_profile=auth_profile,
            auth_file_location=auth_file_location,
        )


def filter_none(**kwargs: Any) -> dict[str, Any]:
    """Filter out None values from keyword arguments.

    This is useful for building kwargs dicts where None means "use default".

    Example:
        >>> filter_none(a=1, b=None, c="hello")
        {'a': 1, 'c': 'hello'}
    """
    return {k: v for k, v in kwargs.items() if v is not None}


def merge_model_kwargs(
    base_kwargs: dict[str, Any],
    temperature: float | None = None,
    max_tokens: int | None = None,
    model_id: str | None = None,
) -> dict[str, Any] | None:
    """Merge temperature and max_tokens into model kwargs.

    Returns None if the result is empty (to avoid passing empty dict).
    """
    result = {**base_kwargs}
    if temperature is not None:
        result["temperature"] = temperature
    if max_tokens is not None:
        if model_id and model_id.startswith("openai."):
            result["max_completion_tokens"] = max_tokens
        else:
            result["max_tokens"] = max_tokens
    return result or None
