# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Shared fixtures and configuration for agent integration tests.

All configuration is provided via environment variables.

## Required Environment Variables

### OpenSearch
- OPENSEARCH_ENDPOINT: OpenSearch endpoint URL
- OPENSEARCH_INDEX: Index name to test
- OPENSEARCH_USERNAME: Username
- OPENSEARCH_PASSWORD: Password
- OPENSEARCH_EMBEDDING_MODEL: Embedding model matching index dimensions

### ADB
- ADB_DSN: Database connection string
- ADB_USER: Database user
- ADB_PASSWORD: Database password
- ADB_EMBEDDING_MODEL: Embedding model matching table dimensions

### OCI
- OCI_COMPARTMENT_ID: OCI compartment OCID
"""

import os

import pytest

# =============================================================================
# Configuration Helpers
# =============================================================================


def get_opensearch_config() -> dict:
    """Get OpenSearch configuration from environment variables."""
    return {
        "endpoint": os.environ.get("OPENSEARCH_ENDPOINT"),
        "index_name": os.environ.get("OPENSEARCH_INDEX"),
        "username": os.environ.get("OPENSEARCH_USERNAME"),
        "password": os.environ.get("OPENSEARCH_PASSWORD"),
        "use_ssl": os.environ.get("OPENSEARCH_USE_SSL", "true").lower() == "true",
        "verify_certs": (
            os.environ.get("OPENSEARCH_VERIFY_CERTS", "false").lower() == "true"
        ),
        "vector_field": os.environ.get("OPENSEARCH_VECTOR_FIELD", "vector_field"),
        "search_fields": os.environ.get("OPENSEARCH_SEARCH_FIELDS", "text").split(","),
        "hint": os.environ.get("OPENSEARCH_HINT", ""),
        "embedding_model": os.environ.get("OPENSEARCH_EMBEDDING_MODEL"),
    }


def get_adb_config() -> dict:
    """Get ADB configuration from environment variables."""
    wallet_loc = os.environ.get("ADB_WALLET_LOCATION")
    return {
        "dsn": os.environ.get("ADB_DSN"),
        "wallet_location": os.path.expanduser(wallet_loc) if wallet_loc else None,
        "user": os.environ.get("ADB_USER"),
        "password": os.environ.get("ADB_PASSWORD"),
        "table_name": os.environ.get("ADB_TABLE_NAME", "VECTOR_DOCUMENTS"),
        "hint": os.environ.get("ADB_HINT", ""),
        "embedding_model": os.environ.get("ADB_EMBEDDING_MODEL"),
    }


def get_oci_config() -> dict:
    """Get OCI configuration from environment variables."""
    region = os.environ.get("OCI_REGION", "us-chicago-1")
    default_endpoint = f"https://inference.generativeai.{region}.oci.oraclecloud.com"
    max_output = os.environ.get("OCI_DEEP_RESEARCH_MAX_OUTPUT_TOKENS")

    return {
        "compartment_id": os.environ.get("OCI_COMPARTMENT_ID"),
        "service_endpoint": os.environ.get("OCI_SERVICE_ENDPOINT", default_endpoint),
        "auth_type": os.environ.get("OCI_AUTH_TYPE", "API_KEY"),
        "auth_profile": os.environ.get("OCI_CONFIG_PROFILE", "DEFAULT"),
        "chat_model": os.environ.get("OCI_CHAT_MODEL"),
        "deep_research_model": os.environ.get("OCI_DEEP_RESEARCH_MODEL"),
        "deep_research_max_output_tokens": int(max_output) if max_output else None,
    }


# =============================================================================
# Skip Condition Helpers
# =============================================================================


def has_opensearch_config() -> bool:
    """Check if required OpenSearch environment variables are set."""
    config = get_opensearch_config()
    required = ["endpoint", "index_name", "username", "password", "embedding_model"]
    return all(config.get(k) for k in required)


def has_adb_config() -> bool:
    """Check if required ADB environment variables are set."""
    config = get_adb_config()
    required = ["dsn", "user", "password", "embedding_model"]
    return all(config.get(k) for k in required)


def has_oci_config() -> bool:
    """Check if required OCI environment variables are set."""
    config = get_oci_config()
    return bool(config.get("compartment_id"))


def opensearch_is_reachable() -> bool:
    """Check if OpenSearch is reachable."""
    if not has_opensearch_config():
        return False
    try:
        import urllib3

        urllib3.disable_warnings()
        import requests

        config = get_opensearch_config()
        response = requests.get(
            config["endpoint"],
            auth=(config["username"], config["password"]),
            verify=False,
            timeout=5,
        )
        return response.status_code == 200
    except Exception:
        return False


def adb_is_reachable() -> bool:
    """Check if ADB is reachable."""
    if not has_adb_config():
        return False
    try:
        import oracledb

        config = get_adb_config()
        connect_kwargs = {
            "user": config["user"],
            "password": config["password"],
            "dsn": config["dsn"],
        }
        if config.get("wallet_location"):
            connect_kwargs.update(
                {
                    "config_dir": config["wallet_location"],
                    "wallet_location": config["wallet_location"],
                    "wallet_password": os.environ.get(
                        "ADB_WALLET_PASSWORD", config["password"]
                    ),
                }
            )

        conn = oracledb.connect(
            **connect_kwargs,
        )
        conn.close()
        return True
    except Exception:
        return False


# =============================================================================
# Factory Helpers
# =============================================================================


def create_embedding_model(model_id: str):
    """Create an OCI embedding model with the given model ID."""
    from langchain_oci import OCIGenAIEmbeddings

    config = get_oci_config()
    return OCIGenAIEmbeddings(
        model_id=model_id,
        compartment_id=config["compartment_id"],
        service_endpoint=config["service_endpoint"],
        auth_type=config["auth_type"],
        auth_profile=config["auth_profile"],
    )


def create_opensearch_store():
    """Create an OpenSearch store from configuration."""
    from langchain_oci.datastores import OpenSearch

    config = get_opensearch_config()
    return OpenSearch(
        endpoint=config["endpoint"],
        index_name=config["index_name"],
        username=config["username"],
        password=config["password"],
        use_ssl=config["use_ssl"],
        verify_certs=config["verify_certs"],
        vector_field=config["vector_field"],
        search_fields=config["search_fields"],
        datastore_description=config["hint"],
    )


def create_adb_store():
    """Create an ADB store from configuration."""
    from langchain_oci.datastores import ADB

    config = get_adb_config()
    return ADB(
        dsn=config["dsn"],
        user=config["user"],
        password=config["password"],
        wallet_location=config["wallet_location"],
        table_name=config["table_name"],
        datastore_description=config["hint"],
    )


# =============================================================================
# Pytest Fixtures
# =============================================================================


@pytest.fixture
def opensearch_config():
    """Get OpenSearch configuration."""
    return get_opensearch_config()


@pytest.fixture
def adb_config():
    """Get ADB configuration."""
    return get_adb_config()


@pytest.fixture
def oci_config():
    """Get OCI configuration."""
    return get_oci_config()
