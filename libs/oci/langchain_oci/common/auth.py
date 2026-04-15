# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Shared OCI authentication utilities."""

import os
from enum import Enum
from typing import Any, Dict, Optional, Tuple, Union

# Default timeout: (connect_timeout, read_timeout) in seconds.
# Override via OCI_REQUEST_TIMEOUT env var (read timeout only)
# or pass timeout= to create_oci_client_kwargs.
_DEFAULT_CONNECT_TIMEOUT = 10
_DEFAULT_READ_TIMEOUT = 240


class OCIAuthType(Enum):
    """OCI authentication types as enumerator."""

    API_KEY = 1
    SECURITY_TOKEN = 2
    INSTANCE_PRINCIPAL = 3
    RESOURCE_PRINCIPAL = 4


def _resolve_timeout(
    timeout: Optional[Union[int, float, Tuple[int, int]]] = None,
) -> Tuple[int, int]:
    """Resolve timeout from parameter or OCI_REQUEST_TIMEOUT env var."""
    if timeout is not None:
        if isinstance(timeout, tuple):
            return timeout
        return (_DEFAULT_CONNECT_TIMEOUT, int(timeout))

    env_timeout = os.environ.get("OCI_REQUEST_TIMEOUT")
    if env_timeout:
        return (_DEFAULT_CONNECT_TIMEOUT, int(env_timeout))

    return (_DEFAULT_CONNECT_TIMEOUT, _DEFAULT_READ_TIMEOUT)


def create_oci_client_kwargs(
    auth_type: str,
    service_endpoint: Optional[str] = None,
    auth_file_location: str = "~/.oci/config",
    auth_profile: str = "DEFAULT",
    timeout: Optional[Union[int, float, Tuple[int, int]]] = None,
) -> Dict[str, Any]:
    """Create OCI client kwargs based on authentication type.

    This function consolidates the authentication logic that was duplicated
    across multiple modules (llms, embeddings, chat_models).

    Args:
        auth_type: The authentication type (API_KEY, SECURITY_TOKEN,
                   INSTANCE_PRINCIPAL, or RESOURCE_PRINCIPAL).
        service_endpoint: The OCI service endpoint URL.
        auth_file_location: Path to the OCI config file.
        auth_profile: The profile name in the OCI config file.
        timeout: Request timeout. Can be an int/float (read timeout in seconds),
                 or a tuple of (connect_timeout, read_timeout). Defaults to
                 (10, 240). Override globally via OCI_REQUEST_TIMEOUT env var.

    Returns:
        Dict with 'config' and/or 'signer' keys ready for OCI client initialization.

    Raises:
        ImportError: If the oci package is not installed.
        ValueError: If an invalid auth_type is provided.
    """
    try:
        import oci
    except ImportError as ex:
        raise ImportError(
            "Could not import oci python package. "
            "Please make sure you have the oci package installed."
        ) from ex

    client_kwargs: Dict[str, Any] = {
        "config": {},
        "signer": None,
        "service_endpoint": service_endpoint,
        "retry_strategy": oci.retry.DEFAULT_RETRY_STRATEGY,
        "timeout": _resolve_timeout(timeout),
    }

    if auth_type == OCIAuthType.API_KEY.name:
        client_kwargs["config"] = oci.config.from_file(
            file_location=auth_file_location,
            profile_name=auth_profile,
        )
        client_kwargs.pop("signer", None)
    elif auth_type == OCIAuthType.SECURITY_TOKEN.name:

        def make_security_token_signer(oci_config: Dict[str, Any]) -> Any:
            key_file = oci_config["key_file"]
            security_token_file = oci_config["security_token_file"]
            pk = oci.signer.load_private_key_from_file(key_file, None)
            with open(security_token_file, encoding="utf-8") as f:
                st_string = f.read()
            return oci.auth.signers.SecurityTokenSigner(st_string, pk)

        client_kwargs["config"] = oci.config.from_file(
            file_location=auth_file_location,
            profile_name=auth_profile,
        )
        client_kwargs["signer"] = make_security_token_signer(
            oci_config=client_kwargs["config"]
        )
    elif auth_type == OCIAuthType.INSTANCE_PRINCIPAL.name:
        client_kwargs["signer"] = (
            oci.auth.signers.InstancePrincipalsSecurityTokenSigner()
        )
    elif auth_type == OCIAuthType.RESOURCE_PRINCIPAL.name:
        client_kwargs["signer"] = oci.auth.signers.get_resource_principals_signer()
    else:
        raise ValueError(
            f"Please provide valid value to auth_type, '{auth_type}' is not valid. "
            f"Valid values are: {[e.name for e in OCIAuthType]}"
        )

    return client_kwargs
