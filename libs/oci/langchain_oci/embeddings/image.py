# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Image embedding mixin for OCIGenAIEmbeddings.

Provides embed_image() and embed_image_batch() methods that convert images
to data URIs and call the OCI embed_text API with input_type="IMAGE".

This mixin is mixed into OCIGenAIEmbeddings so that the core embeddings
module stays focused on text embedding and OCI client setup.
"""

from pathlib import Path
from typing import Any, List, Optional, Sequence, Union

from langchain_oci.utils.vision import to_data_uri


class ImageEmbeddingMixin:
    """Mixin that adds image embedding methods to an embeddings class.

    Expects the host class to provide:
    - ``_build_embed_request(inputs, input_type=None)``
    - ``client.embed_text(request)``
    """

    # Stubs so mypy understands the host-class interface.
    client: Any

    def _build_embed_request(
        self,
        inputs: List[str],
        input_type: Optional[str] = None,
    ) -> Any:
        """Build an embed request (provided by host class)."""
        ...

    def embed_image(
        self,
        image: Union[str, bytes, Path],
        mime_type: str = "image/png",
    ) -> List[float]:
        """Embed a single image.

        Requires a multimodal embedding model (e.g., ``cohere.embed-v4.0``)
        that supports ``input_type="IMAGE"``. The image is converted to a
        data URI via ``langchain_oci.utils.vision.to_data_uri`` and sent
        to the OCI embed_text API.

        The resulting vector lives in the same space as text embeddings,
        enabling cross-modal retrieval.

        Args:
            image: File path (str or Path), raw image bytes,
                or a data URI string (``data:image/png;base64,...``).
            mime_type: MIME type when *image* is raw bytes.
                Ignored for file paths (auto-detected) and data URIs.

        Returns:
            Embedding vector for the image.
        """
        return self.embed_image_batch([image], mime_type=mime_type)[0]

    def embed_image_batch(
        self,
        images: Sequence[Union[str, bytes, Path]],
        mime_type: str = "image/png",
    ) -> List[List[float]]:
        """Embed multiple images in a batch.

        Each image is embedded individually (the OCI API accepts one
        image per request). Requires a multimodal embedding model.

        Args:
            images: List of file paths, raw bytes, or data URI strings.
            mime_type: Default MIME type for raw bytes inputs.

        Returns:
            List of embedding vectors, one per image.
        """
        embeddings: List[List[float]] = []
        for image in images:
            data_uri = to_data_uri(image, mime_type=mime_type)
            invocation_obj = self._build_embed_request(
                inputs=[data_uri],
                input_type="IMAGE",
            )
            response = self.client.embed_text(invocation_obj)
            embeddings.extend(response.data.embeddings)
        return embeddings
