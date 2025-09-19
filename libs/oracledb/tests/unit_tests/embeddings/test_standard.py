from typing import Type

import oracledb
import pytest
from langchain_tests.unit_tests import EmbeddingsUnitTests

from langchain_oracledb import OracleEmbeddings

username = ""
password = ""
dsn = ""

try:
    oracledb.connect(user=username, password=password, dsn=dsn)
except Exception as e:
    pytest.skip(
        allow_module_level=True,
        reason=f"Database connection failed: {e}, skipping tests.",
    )


class TestOracleEmbeddingsModelIntegration(EmbeddingsUnitTests):
    @property
    def embeddings_class(self) -> Type[OracleEmbeddings]:
        # Return the embeddings model class to test here
        return OracleEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        # Return initialization parameters for the model.
        conn = oracledb.connect(user=username, password=password, dsn=dsn)
        return {"conn": conn, "params": {"provider": "database", "model": "allminilm"}}
