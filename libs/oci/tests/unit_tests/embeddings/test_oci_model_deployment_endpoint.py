"""Test OCI Data Science Model Deployment Endpoint."""

import pytest
import responses
from pytest_mock import MockerFixture

from langchain_oci.embeddings import OCIModelDeploymentEndpointEmbeddings


@pytest.mark.requires("ads")
@responses.activate
def test_embedding_call(mocker: MockerFixture) -> None:
    """Test valid call to oci model deployment endpoint."""
    endpoint = "https://MD_OCID/predict"
    documents = ["Hello", "World"]
    expected_output = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    responses.add(
        responses.POST,
        endpoint,
        json={
            "data": [{"embedding": expected_output}],
        },
        status=200,
    )
    mocker.patch("ads.common.auth.default_signer", return_value=dict(signer=None))

    embeddings = OCIModelDeploymentEndpointEmbeddings(
        endpoint=endpoint,
    )

    output = embeddings.embed_documents(documents)
    assert output == expected_output
