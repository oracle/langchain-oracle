# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""Shared native/REST configuration for embedding contract tests."""

import os
from contextlib import contextmanager

import oracledb
import pytest

EMBEDDING_BACKEND = os.environ.get("ORACLE_EMBEDDINGS_BACKEND", "sql")
EMBEDDING_MODEL = os.environ.get("ORACLE_EMBEDDINGS_MODEL", "allminilm")


@contextmanager
def embedding_connection():
    if EMBEDDING_BACKEND == "rest":
        sdk = pytest.importorskip("oracle_vecdb")
        url = os.environ.get("ORACLE_VECDB_REST_URL")
        token = os.environ.get("ORACLE_VECDB_ACCESS_TOKEN")
        user = os.environ.get("ORACLE_VECDB_USERNAME")
        password = os.environ.get("ORACLE_VECDB_PASSWORD")
        if not url or not (token or (user and password)):
            pytest.skip(
                "REST embedding configuration is missing", allow_module_level=True
            )
        auth = (
            {"access_token": token}
            if token
            else {"username": user, "password": password}
        )
        yield sdk.OracleVecDB(
            sdk.Configuration(
                rest_url=url,
                ssl_ca_cert=os.environ.get("ORACLE_VECDB_CA_CERT"),
                timeout=60,
                **auth,
            )
        )
    elif EMBEDDING_BACKEND == "sql":
        keys = ("VECDB_USER", "VECDB_PASS", "VECDB_HOST")
        if not all(os.environ.get(key) for key in keys):
            pytest.skip(
                "Native embedding configuration is missing", allow_module_level=True
            )
        with oracledb.connect(
            user=os.environ[keys[0]],
            password=os.environ[keys[1]],
            dsn=os.environ[keys[2]],
        ) as connection:
            yield connection
    else:
        raise ValueError("ORACLE_EMBEDDINGS_BACKEND must be sql or rest")


def check_embedding_configuration():
    try:
        with embedding_connection():
            pass
    except oracledb.Error:
        pytest.skip("Native embedding database is unreachable", allow_module_level=True)
