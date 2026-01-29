# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from typing import Any, Dict, Iterator, List, Mapping, Optional

from langchain_core.embeddings import Embeddings
from langchain_core.utils import pre_init
from pydantic import BaseModel, ConfigDict

from langchain_oci.common.auth import create_oci_client_kwargs
from langchain_oci.common.utils import CUSTOM_ENDPOINT_PREFIX
from langchain_oci.embeddings.image import ImageEmbeddingMixin


class OCIGenAIEmbeddings(BaseModel, Embeddings, ImageEmbeddingMixin):
    """OCI embedding models with text and image support.

    To authenticate, the OCI client uses the methods described in
    https://docs.oracle.com/en-us/iaas/Content/API/Concepts/sdk_authentication_methods.htm

    The authentifcation method is passed through auth_type and should be one of:
    API_KEY (default), SECURITY_TOKEN, INSTANCE_PRINCIPLE, RESOURCE_PRINCIPLE

    Make sure you have the required policies (profile/roles) to
    access the OCI Generative AI service. If a specific config profile is used,
    you must pass the name of the profile (~/.oci/config) through auth_profile.
    If a specific config file location is used, you must pass
    the file location where profile name configs present
    through auth_file_location

    To use, you must provide the compartment id
    along with the endpoint url, and model id
    as named parameters to the constructor.

    Example:
        .. code-block:: python

            from langchain_oci import OCIGenAIEmbeddings

            # Text embeddings
            embeddings = OCIGenAIEmbeddings(
                model_id="cohere.embed-v4.0",
                service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
                compartment_id="MY_OCID",
            )
            vectors = embeddings.embed_documents(["Hello world"])

            # Image embeddings (same model, same vector space)
            vectors = embeddings.embed_images(["./photo.jpg"])

    For image embedding, use a multimodal model like ``cohere.embed-v4.0``
    which embeds text and images into the same vector space, enabling
    cross-modal retrieval (search images with text queries and vice versa).

    Image inputs can be file paths, raw bytes, or data URIs. The
    ``to_data_uri``, ``load_image``, and ``encode_image`` utilities from
    ``langchain_oci.utils.vision`` can also be used to prepare image data.
    """

    client: Any = None  #: :meta private:

    service_models: Any = None  #: :meta private:

    auth_type: Optional[str] = "API_KEY"
    """Authentication type, could be

    API_KEY,
    SECURITY_TOKEN,
    INSTANCE_PRINCIPLE,
    RESOURCE_PRINCIPLE

    If not specified, API_KEY will be used
    """

    auth_profile: Optional[str] = "DEFAULT"
    """The name of the profile in ~/.oci/config
    If not specified , DEFAULT will be used
    """

    auth_file_location: Optional[str] = "~/.oci/config"
    """Path to the config file.
    If not specified, ~/.oci/config will be used
    """

    model_id: Optional[str] = None
    """Id of the model to call, e.g., cohere.embed-v4.0"""

    model_kwargs: Optional[Dict] = None
    """Keyword arguments to pass to the model"""

    service_endpoint: Optional[str] = None
    """service endpoint url"""

    compartment_id: Optional[str] = None
    """OCID of compartment"""

    truncate: Optional[str] = "END"
    """Truncate embeddings that are too long
    from start or end ("NONE"|"START"|"END")"""

    input_type: Optional[str] = None
    """Input type for the embedding request. Valid values:

    SEARCH_DOCUMENT - For documents to be searched (default if not set)
    SEARCH_QUERY - For search queries
    CLASSIFICATION - For text classification
    CLUSTERING - For text clustering
    IMAGE - For image inputs (use embed_image/embed_images methods instead)

    If not specified, the OCI API default is used.
    """

    output_dimensions: Optional[int] = None
    """Number of output embedding dimensions. Only supported by embed-v4.0+.
    Valid values: 256, 512, 1024, 1536. If not specified, the model default
    is used (1536 for embed-v4.0).
    """

    batch_size: int = 96
    """Batch size of OCI GenAI embedding requests. OCI GenAI may
    handle up to 96 texts per request"""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    @pre_init
    def validate_environment(  # pylint: disable=no-self-argument
        cls, values: Dict
    ) -> Dict:
        """Validate that OCI config and python package exists in environment."""

        # Skip creating new client if passed in constructor
        if values["client"] is not None:
            return values

        try:
            import oci

            client_kwargs = create_oci_client_kwargs(
                auth_type=values["auth_type"],
                service_endpoint=values["service_endpoint"],
                auth_file_location=values["auth_file_location"],
                auth_profile=values["auth_profile"],
            )

            values["client"] = oci.generative_ai_inference.GenerativeAiInferenceClient(
                **client_kwargs
            )

        except ImportError as ex:
            raise ImportError(
                "Could not import oci python package. "
                "Please make sure you have the oci package installed."
            ) from ex
        except Exception as e:
            raise ValueError(
                "Could not authenticate with OCI client. "
                "If INSTANCE_PRINCIPAL or RESOURCE_PRINCIPAL is used, "
                "please check the specified auth_profile, "
                "auth_file_location and auth_type are valid.",
                e,
            ) from e

        return values

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        _model_kwargs = self.model_kwargs or {}
        return {
            **{"model_kwargs": _model_kwargs},
        }

    def _get_serving_mode(self) -> Any:
        """Get the serving mode for the model."""
        from oci.generative_ai_inference import models

        if not self.model_id:
            raise ValueError("Model ID is required")

        if self.model_id.startswith(CUSTOM_ENDPOINT_PREFIX):
            return models.DedicatedServingMode(endpoint_id=self.model_id)
        return models.OnDemandServingMode(model_id=self.model_id)

    def _build_embed_request(
        self,
        inputs: List[str],
        input_type: Optional[str] = None,
    ) -> Any:
        """Build an EmbedTextDetails request.

        Args:
            inputs: List of text strings or data URIs.
            input_type: Override for self.input_type.
        """
        from oci.generative_ai_inference import models

        kwargs: Dict[str, Any] = {
            "serving_mode": self._get_serving_mode(),
            "compartment_id": self.compartment_id,
            "truncate": self.truncate,
            "inputs": inputs,
        }

        resolved_type = input_type or self.input_type
        if resolved_type:
            kwargs["input_type"] = resolved_type

        if self.output_dimensions is not None:
            if hasattr(models.EmbedTextDetails, "output_dimensions"):
                kwargs["output_dimensions"] = self.output_dimensions
            else:
                raise ValueError(
                    "output_dimensions requires a newer version of the "
                    "OCI SDK. Please upgrade: pip install --upgrade oci"
                )

        return models.EmbedTextDetails(**kwargs)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Call out to OCIGenAI's embedding endpoint.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        embeddings: List[List[float]] = []

        def split_texts() -> Iterator[List[str]]:
            for i in range(0, len(texts), self.batch_size):
                yield texts[i : i + self.batch_size]

        for chunk in split_texts():
            invocation_obj = self._build_embed_request(chunk)
            response = self.client.embed_text(invocation_obj)
            embeddings.extend(response.data.embeddings)

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Call out to OCIGenAI's embedding endpoint.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        return self.embed_documents([text])[0]
