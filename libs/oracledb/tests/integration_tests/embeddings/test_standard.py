from typing import Type

import pytest
from langchain_tests.integration_tests import EmbeddingsIntegrationTests

from langchain_oracledb import OracleEmbeddings
from tests._embedding_config import (
    EMBEDDING_MODEL,
    check_embedding_configuration,
    embedding_connection,
)

check_embedding_configuration()


class TestOracleEmbeddingsModelIntegration(EmbeddingsIntegrationTests):
    @pytest.fixture(autouse=True)
    def embedding_connection(self):
        with embedding_connection() as conn:
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
            "params": {"provider": "database", "model": EMBEDDING_MODEL},
        }
