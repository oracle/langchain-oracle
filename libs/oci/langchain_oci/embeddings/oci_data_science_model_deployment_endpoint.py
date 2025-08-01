# Copyright (c) 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from typing import Any, Callable, Dict, List, Mapping, Optional

import requests
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.llms import create_base_retry_decorator
from langchain_core.utils import get_from_dict_or_env
from pydantic import BaseModel, Field, model_validator

DEFAULT_HEADER = {
    "Content-Type": "application/json",
}


class TokenExpiredError(Exception):
    pass


def _create_retry_decorator(llm) -> Callable[[Any], Any]:
    """Creates a retry decorator."""
    errors = [requests.exceptions.ConnectTimeout, TokenExpiredError]
    decorator = create_base_retry_decorator(
        error_types=errors, max_retries=llm.max_retries
    )
    return decorator


class OCIModelDeploymentEndpointEmbeddings(BaseModel, Embeddings):
    """Embedding model deployed on OCI Data Science Model Deployment.

    Example:

        .. code-block:: python

            from langchain_oci.embeddings import OCIModelDeploymentEndpointEmbeddings

            embeddings = OCIModelDeploymentEndpointEmbeddings(
                endpoint="https://modeldeployment.us-ashburn-1.oci.customer-oci.com/<md_ocid>/predict",
            )
    """  # noqa: E501

    auth: dict = Field(default_factory=dict, exclude=True)
    """ADS auth dictionary for OCI authentication:
    https://accelerated-data-science.readthedocs.io/en/latest/user_guide/cli/authentication.html.
    This can be generated by calling `ads.common.auth.api_keys()`
    or `ads.common.auth.resource_principal()`. If this is not
    provided then the `ads.common.default_signer()` will be used."""

    endpoint: str = ""
    """The uri of the endpoint from the deployed Model Deployment model."""

    model_kwargs: Optional[Dict] = None
    """Keyword arguments to pass to the model."""

    endpoint_kwargs: Optional[Dict] = None
    """Optional attributes (except for headers) passed to the request.post
    function. 
    """

    max_retries: int = 1
    """The maximum number of retries to make when generating."""

    @model_validator(mode="before")
    def validate_environment(  # pylint: disable=no-self-argument
        cls, values: Dict
    ) -> Dict:
        """Validate that python package exists in environment."""
        try:
            import ads

        except ImportError as ex:
            raise ImportError(
                "Could not import ads python package. "
                "Please install it with `pip install oracle_ads`."
            ) from ex
        if not values.get("auth", None):
            values["auth"] = ads.common.auth.default_signer()
        values["endpoint"] = get_from_dict_or_env(
            values,
            "endpoint",
            "OCI_LLM_ENDPOINT",
        )
        return values

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        _model_kwargs = self.model_kwargs or {}
        return {
            **{"endpoint": self.endpoint},
            **{"model_kwargs": _model_kwargs},
        }

    def _embed_with_retry(self, **kwargs) -> Any:
        """Use tenacity to retry the call."""
        retry_decorator = _create_retry_decorator(self)

        @retry_decorator
        def _completion_with_retry(**kwargs: Any) -> Any:
            try:
                response = requests.post(self.endpoint, **kwargs)
                response.raise_for_status()
                return response
            except requests.exceptions.HTTPError as http_err:
                if response.status_code == 401 and self._refresh_signer():
                    raise TokenExpiredError() from http_err
                else:
                    raise ValueError(
                        f"Server error: {str(http_err)}. Message: {response.text}"
                    ) from http_err
            except Exception as e:
                raise ValueError(f"Error occurs by inference endpoint: {str(e)}") from e

        return _completion_with_retry(**kwargs)

    def _embedding(self, texts: List[str]) -> List[List[float]]:
        """Call out to OCI Data Science Model Deployment Endpoint.

        Args:
            texts: A list of texts to embed.

        Returns:
            A list of list of floats representing the embeddings, or None if an
            error occurs.
        """
        _model_kwargs = self.model_kwargs or {}
        body = self._construct_request_body(texts, _model_kwargs)
        request_kwargs = self._construct_request_kwargs(body)
        response = self._embed_with_retry(**request_kwargs)
        return self._proceses_response(response)

    def _construct_request_kwargs(self, body: Any) -> dict:
        """Constructs the request kwargs as a dictionary."""
        from ads.model.common.utils import _is_json_serializable

        _endpoint_kwargs = self.endpoint_kwargs or {}
        headers = _endpoint_kwargs.pop("headers", DEFAULT_HEADER)
        return (
            dict(
                headers=headers,
                json=body,
                auth=self.auth.get("signer"),
                **_endpoint_kwargs,
            )
            if _is_json_serializable(body)
            else dict(
                headers=headers,
                data=body,
                auth=self.auth.get("signer"),
                **_endpoint_kwargs,
            )
        )

    def _construct_request_body(self, texts: List[str], params: dict) -> Any:
        """Constructs the request body."""
        return {"input": texts}

    def _proceses_response(self, response: requests.Response) -> List[List[float]]:
        """Extracts results from requests.Response."""
        try:
            res_json = response.json()
            embeddings = res_json["data"][0]["embedding"]
        except Exception as e:
            raise ValueError(
                f"Error raised by inference API: {e}.\nResponse: {response.text}"
            )
        return embeddings

    def embed_documents(
        self,
        texts: List[str],
        chunk_size: Optional[int] = None,
    ) -> List[List[float]]:
        """Compute doc embeddings using OCI Data Science Model Deployment Endpoint.

        Args:
            texts: The list of texts to embed.
            chunk_size: The chunk size defines how many input texts will
                be grouped together as request. If None, will use the
                chunk size specified by the class.

        Returns:
            List of embeddings, one for each text.
        """
        results = []
        _chunk_size = (
            len(texts) if (not chunk_size or chunk_size > len(texts)) else chunk_size
        )
        for i in range(0, len(texts), _chunk_size):
            response = self._embedding(texts[i : i + _chunk_size])
            results.extend(response)
        return results

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using OCI Data Science Model Deployment Endpoint.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        return self._embedding([text])[0]
