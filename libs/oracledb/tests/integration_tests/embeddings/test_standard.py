import os
from typing import Type

import oracledb
import pytest
from langchain_tests.integration_tests import EmbeddingsIntegrationTests

from langchain_oracledb import OracleEmbeddings

username = os.environ.get("VECDB_USER")
password = os.environ.get("VECDB_PASS")
dsn = os.environ.get("VECDB_HOST")

try:
    with oracledb.connect(user=username, password=password, dsn=dsn):
        pass
except Exception as e:
    pytest.skip(
        allow_module_level=True,
        reason=f"Database connection failed: {e}, skipping tests.",
    )


class TestOracleEmbeddingsModelIntegration(EmbeddingsIntegrationTests):
    @pytest.fixture(autouse=True)
    def embedding_connection(self):
        with oracledb.connect(user=username, password=password, dsn=dsn) as conn:
            self._embedding_conn = conn
            yield

    @property
    def embeddings_class(self) -> Type[OracleEmbeddings]:
        # Return the embeddings model class to test here
        return OracleEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        # Return initialization parameters for the model.
        return {
            "conn": self._embedding_conn,
            "params": {"provider": "database", "model": "allminilm"},
        }
