# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Shared fixtures and configuration for agent integration tests.

All configuration is provided via environment variables. In CI/CD, populate
these via the platform's secret store (GitHub Actions secrets, OCI DevOps
parameters, etc.). Tests skip automatically when their required group is
absent, so partial environments still produce a clean run.

## Required Environment Variables

### OCI (required for any agent / chat test)
- OCI_COMPARTMENT_ID: OCI compartment OCID
- OCI_CONFIG_PROFILE: profile name in ~/.oci/config (default "DEFAULT")
- OCI_AUTH_TYPE: "API_KEY" or "SECURITY_TOKEN" (default "API_KEY")
- OCI_REGION: OCI region (default "us-chicago-1")
- OCI_DEEPAGENTS_MODEL: model id for deepagents tests (e.g.
  "google.gemini-2.5-flash"; pro recommended for structured-output tests)
- OCI_CHAT_MODEL: model id for chat / react tests

### OpenSearch (skipped if any missing)
- OPENSEARCH_ENDPOINT, OPENSEARCH_INDEX, OPENSEARCH_USERNAME,
  OPENSEARCH_PASSWORD, OPENSEARCH_EMBEDDING_MODEL

### ADB (skipped if any missing; data auto-seeded by the
``_seed_adb_test_table`` session fixture below)
- ADB_DSN, ADB_USER, ADB_PASSWORD, ADB_EMBEDDING_MODEL
- ADB_WALLET_LOCATION, ADB_WALLET_PASSWORD (for wallet auth)
- ADB_TABLE_NAME (default ``VECTOR_DOCUMENTS_LC`` — the test fixture
  creates this with the OracleVS schema if absent and seeds it with a
  small set of sample documents on the first run)
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
        "table_name": os.environ.get("ADB_TABLE_NAME", "VECTOR_DOCUMENTS_LC"),
        "hint": os.environ.get("ADB_HINT", ""),
        "embedding_model": os.environ.get("ADB_EMBEDDING_MODEL"),
    }


def get_oci_config() -> dict:
    """Get OCI configuration from environment variables."""
    region = os.environ.get("OCI_REGION", "us-chicago-1")
    default_endpoint = f"https://inference.generativeai.{region}.oci.oraclecloud.com"
    max_output = os.environ.get("OCI_DEEPAGENTS_MAX_OUTPUT_TOKENS")

    return {
        "compartment_id": os.environ.get("OCI_COMPARTMENT_ID"),
        "service_endpoint": os.environ.get("OCI_SERVICE_ENDPOINT", default_endpoint),
        "auth_type": os.environ.get("OCI_AUTH_TYPE", "API_KEY"),
        "auth_profile": os.environ.get("OCI_CONFIG_PROFILE", "DEFAULT"),
        "chat_model": os.environ.get("OCI_CHAT_MODEL"),
        "deepagents_model": os.environ.get("OCI_DEEPAGENTS_MODEL"),
        "deepagents_max_output_tokens": int(max_output) if max_output else None,
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


# =============================================================================
# Session-scoped autouse: bootstrap the ADB test table for CI/CD
# =============================================================================

# Sample documents used to bootstrap an empty ADB test table so that
# downstream tests have something to query. Keeping the corpus small and
# textual keeps the seeding step well under a few seconds of OCI embedding
# calls per session.
_ADB_TEST_DOCS: list[tuple[str, dict]] = [
    (
        "Oracle Autonomous Database provides self-driving, self-securing, "
        "self-repairing cloud database services for diverse workloads.",
        {"source": "oracle-docs", "topic": "adb"},
    ),
    (
        "Vector search in Oracle Database 23ai uses HNSW or IVF Flat indexes "
        "for approximate nearest neighbor search over embedded text.",
        {"source": "oracle-docs", "topic": "vector"},
    ),
    (
        "LangChain is a framework for developing applications powered by "
        "large language models, supporting chains, agents, and retrieval "
        "augmented generation.",
        {"source": "langchain-docs", "topic": "framework"},
    ),
    (
        "OCI Generative AI offers managed access to foundation models from "
        "Cohere, Meta, Google, xAI, and OpenAI for text generation and "
        "embeddings.",
        {"source": "oci-docs", "topic": "genai"},
    ),
    (
        "Deepagents is a Python library for building research and planning "
        "agents on top of LangGraph with built-in subagent spawning and "
        "filesystem tools.",
        {"source": "deepagents", "topic": "agent"},
    ),
    (
        "Retrieval augmented generation combines vector search over a "
        "document corpus with an LLM to produce grounded, citation-supported "
        "answers.",
        {"source": "general", "topic": "rag"},
    ),
]


@pytest.fixture(scope="session", autouse=True)
def _seed_adb_test_table():
    """Idempotently bootstrap the ADB test table with sample documents.

    Skips entirely if ADB is unreachable or OCI credentials are absent so
    the suite still runs in partial environments. When ADB is reachable
    but the configured table is empty or missing, this fixture creates
    the table via the OracleVS schema, embeds the sample corpus with the
    configured OCI embedding model, and inserts the rows. A CTXSYS.CONTEXT
    text index is added so the hybrid-search test path exercises the
    full lexical + semantic merge instead of the no-text-index fallback.

    The table persists between runs so the bootstrap is fast (cost: one
    SELECT COUNT(*) on subsequent runs). To force a re-seed, drop the
    table externally.
    """
    if not adb_is_reachable() or not has_oci_config():
        yield
        return

    try:
        import oracledb

        from langchain_oracledb.vectorstores.oraclevs import (
            DistanceStrategy,
            OracleVS,
        )
    except ImportError:
        # Optional deps missing; let the requires-marked tests skip.
        yield
        return

    adb_cfg = get_adb_config()
    table = adb_cfg["table_name"]

    connect_kwargs = {
        "user": adb_cfg["user"],
        "password": adb_cfg["password"],
        "dsn": adb_cfg["dsn"],
    }
    if adb_cfg.get("wallet_location"):
        connect_kwargs.update(
            {
                "config_dir": adb_cfg["wallet_location"],
                "wallet_location": adb_cfg["wallet_location"],
                "wallet_password": os.environ.get(
                    "ADB_WALLET_PASSWORD", adb_cfg["password"]
                ),
            }
        )

    conn = oracledb.connect(**connect_kwargs)
    cur = conn.cursor()

    # Check if the test table already has rows; if so, skip the work.
    try:
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        count = cur.fetchone()[0]
    except oracledb.DatabaseError:
        # Table does not exist yet — OracleVS.from_texts will create it.
        count = 0

    if count > 0:
        conn.close()
        yield
        return

    embeddings = create_embedding_model(adb_cfg["embedding_model"])
    OracleVS.from_texts(
        [text for text, _ in _ADB_TEST_DOCS],
        embeddings,
        metadatas=[meta for _, meta in _ADB_TEST_DOCS],
        client=conn,
        table_name=table,
        distance_strategy=DistanceStrategy.COSINE,
    )

    # Add a CTXSYS.CONTEXT index so hybrid search exercises the full path
    # rather than the no-text-index fallback. Ignored if it already exists.
    idx_name = f"{table}_TEXT_IDX"[:30]
    try:
        cur.execute(
            f"CREATE INDEX {idx_name} ON {table} (TEXT) "
            f"INDEXTYPE IS CTXSYS.CONTEXT"
        )
        conn.commit()
    except oracledb.DatabaseError:
        # Index already exists, or text index unavailable on this ADB tier.
        pass

    conn.close()
    yield
