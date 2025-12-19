# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Shared OCI authentication utilities."""

from enum import Enum
from typing import Any, Dict, Optional


class OCIAuthType(Enum):
    """OCI authentication types as enumerator."""

    API_KEY = 1
    SECURITY_TOKEN = 2
    INSTANCE_PRINCIPAL = 3
    RESOURCE_PRINCIPAL = 4


def create_oci_client_kwargs(
    auth_type: str,
    service_endpoint: Optional[str] = None,
    auth_file_location: str = "~/.oci/config",
    auth_profile: str = "DEFAULT",
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
        "timeout": (10, 240),  # default timeout config for OCI Gen AI service
    }

    if auth_type == OCIAuthType.API_KEY.name:
        client_kwargs["config"] = oci.config.from_file(
            file_location=auth_file_location,
            profile_name=auth_profile,
        )
        client_kwargs.pop("signer", None)
    elif auth_type == OCIAuthType.SECURITY_TOKEN.name:

        def make_security_token_signer(oci_config: Dict[str, Any]) -> Any:
            pk = oci.signer.load_private_key_from_file(
                oci_config.get("key_file"), None
            )
            with open(
                oci_config.get("security_token_file"), encoding="utf-8"
            ) as f:
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
