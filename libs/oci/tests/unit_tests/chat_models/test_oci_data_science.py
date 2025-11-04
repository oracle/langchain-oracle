"""Test Chat model for OCI Data Science Model Deployment Endpoint."""

from unittest import mock

import httpx
import pytest

from langchain_oci import ChatOCIModelDeployment


@pytest.mark.requires("ads")
@pytest.mark.requires("langchain_openai")
@mock.patch("ads.aqua.client.client.HttpxOCIAuth")
def test_authentication_with_ads(*args):
    with mock.patch(
        "ads.common.auth.default_signer", return_value=dict(signer=mock.MagicMock())
    ) as ads_auth:
        llm = ChatOCIModelDeployment(endpoint="<endpoint>", model="my-model")
        ads_auth.assert_called_once()
        assert llm.openai_api_base == "<endpoint>/v1"
        assert llm.model_name == "my-model"


@pytest.mark.requires("langchain_openai")
def test_authentication_with_custom_client():
    http_client = httpx.Client()
    llm = ChatOCIModelDeployment(base_url="<endpoint>/v1", http_client=http_client)
    assert llm.http_client == http_client
    assert llm.openai_api_base == "<endpoint>/v1"
    assert llm.model_name == "odsc-llm"
