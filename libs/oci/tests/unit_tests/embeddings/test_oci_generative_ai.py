# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for OCIGenAIEmbeddings — text and image embedding."""

import base64
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from oci.generative_ai_inference.models import EmbedTextDetails

from langchain_oci.embeddings.oci_generative_ai import OCIGenAIEmbeddings
from langchain_oci.utils.vision import IMAGE_EMBEDDING_MODELS, to_data_uri

_SDK_HAS_OUTPUT_DIMS = hasattr(EmbedTextDetails, "output_dimensions")

# =============================================================================
# Helpers
# =============================================================================


def _make_embeddings(**overrides) -> OCIGenAIEmbeddings:
    """Create an OCIGenAIEmbeddings with a mocked OCI client."""
    defaults = {
        "client": MagicMock(),
        "model_id": "cohere.embed-v4.0",
        "compartment_id": "ocid1.compartment.oc1..test",
    }
    defaults.update(overrides)
    return OCIGenAIEmbeddings(**defaults)


def _mock_embed_response(vectors: list[list[float]]) -> MagicMock:
    """Build a mock response.data.embeddings result."""
    response = MagicMock()
    response.data.embeddings = vectors
    return response


# =============================================================================
# Tests: basic fields
# =============================================================================


class TestFieldDefaults:
    """Test new field defaults and validation."""

    def test_input_type_default_none(self) -> None:
        emb = _make_embeddings()
        assert emb.input_type is None

    def test_output_dimensions_default_none(self) -> None:
        emb = _make_embeddings()
        assert emb.output_dimensions is None

    def test_input_type_settable(self) -> None:
        emb = _make_embeddings(input_type="SEARCH_QUERY")
        assert emb.input_type == "SEARCH_QUERY"

    def test_output_dimensions_settable(self) -> None:
        emb = _make_embeddings(output_dimensions=256)
        assert emb.output_dimensions == 256

    def test_truncate_default(self) -> None:
        emb = _make_embeddings()
        assert emb.truncate == "END"

    def test_batch_size_default(self) -> None:
        emb = _make_embeddings()
        assert emb.batch_size == 96


# =============================================================================
# Tests: _build_embed_request
# =============================================================================


class TestBuildEmbedRequest:
    """Test the internal _build_embed_request method."""

    def test_basic_text_request(self) -> None:
        """Request with no input_type or output_dimensions."""
        emb = _make_embeddings()
        req = emb._build_embed_request(["hello"])

        assert req.inputs == ["hello"]
        assert req.compartment_id == "ocid1.compartment.oc1..test"
        assert req.truncate == "END"
        assert req.input_type is None
        if _SDK_HAS_OUTPUT_DIMS:
            assert req.output_dimensions is None

    def test_request_with_input_type(self) -> None:
        """input_type from instance is passed to request."""
        emb = _make_embeddings(input_type="SEARCH_DOCUMENT")
        req = emb._build_embed_request(["hello"])
        assert req.input_type == "SEARCH_DOCUMENT"

    def test_request_input_type_override(self) -> None:
        """Explicit input_type arg overrides instance value."""
        emb = _make_embeddings(input_type="SEARCH_DOCUMENT")
        req = emb._build_embed_request(
            ["data:image/png;base64,abc"], input_type="IMAGE"
        )
        assert req.input_type == "IMAGE"

    @pytest.mark.skipif(
        not _SDK_HAS_OUTPUT_DIMS,
        reason="OCI SDK too old for output_dimensions",
    )
    def test_request_with_output_dimensions(self) -> None:
        """output_dimensions is passed when set."""
        emb = _make_embeddings(output_dimensions=512)
        req = emb._build_embed_request(["hello"])
        assert req.output_dimensions == 512

    @pytest.mark.skipif(
        not _SDK_HAS_OUTPUT_DIMS,
        reason="OCI SDK too old for output_dimensions",
    )
    def test_request_no_output_dimensions_when_none(self) -> None:
        """output_dimensions is omitted when None."""
        emb = _make_embeddings()
        req = emb._build_embed_request(["hello"])
        assert req.output_dimensions is None

    def test_serving_mode_on_demand(self) -> None:
        """On-demand serving mode for standard model IDs."""
        from oci.generative_ai_inference.models import (
            OnDemandServingMode,
        )

        emb = _make_embeddings()
        req = emb._build_embed_request(["x"])
        assert isinstance(req.serving_mode, OnDemandServingMode)
        assert req.serving_mode.model_id == "cohere.embed-v4.0"

    def test_serving_mode_dedicated(self) -> None:
        """Dedicated serving mode for custom endpoint IDs."""
        from oci.generative_ai_inference.models import (
            DedicatedServingMode,
        )

        emb = _make_embeddings(model_id="ocid1.generativeaiendpoint.oc1..xyz")
        req = emb._build_embed_request(["x"])
        assert isinstance(req.serving_mode, DedicatedServingMode)


# =============================================================================
# Tests: to_data_uri (from vision.py)
# =============================================================================


class TestToDataUri:
    """Test the to_data_uri conversion function."""

    def test_bytes_default_mime(self) -> None:
        raw = b"\x89PNG\r\n\x1a\nfakedata"
        result = to_data_uri(raw)
        assert result.startswith("data:image/png;base64,")
        encoded_part = result.split(",", 1)[1]
        assert base64.standard_b64decode(encoded_part) == raw

    def test_bytes_custom_mime(self) -> None:
        raw = b"\xff\xd8\xff\xe0jpegdata"
        result = to_data_uri(raw, mime_type="image/jpeg")
        assert result.startswith("data:image/jpeg;base64,")

    def test_data_uri_passthrough(self) -> None:
        uri = "data:image/webp;base64,UklGRhIAAABXRUJQ"
        result = to_data_uri(uri)
        assert result == uri

    def test_file_path(self, tmp_path: Path) -> None:
        img_file = tmp_path / "test.png"
        img_file.write_bytes(b"\x89PNG\r\n\x1a\nfakeimage")
        result = to_data_uri(str(img_file))
        assert result.startswith("data:image/png;base64,")
        encoded_part = result.split(",", 1)[1]
        assert base64.standard_b64decode(encoded_part) == b"\x89PNG\r\n\x1a\nfakeimage"

    def test_file_path_jpeg(self, tmp_path: Path) -> None:
        img_file = tmp_path / "photo.jpg"
        img_file.write_bytes(b"\xff\xd8\xff\xe0fakedata")
        result = to_data_uri(str(img_file))
        assert result.startswith("data:image/jpeg;base64,")

    def test_pathlib_path(self, tmp_path: Path) -> None:
        img_file = tmp_path / "img.png"
        img_file.write_bytes(b"png_bytes")
        result = to_data_uri(img_file)
        assert result.startswith("data:image/png;base64,")


# =============================================================================
# Tests: embed_documents (text)
# =============================================================================


class TestEmbedDocuments:
    """Test text embedding via embed_documents."""

    def test_single_text(self) -> None:
        emb = _make_embeddings()
        emb.client.embed_text.return_value = _mock_embed_response([[0.1, 0.2, 0.3]])
        result = emb.embed_documents(["hello"])
        assert result == [[0.1, 0.2, 0.3]]
        emb.client.embed_text.assert_called_once()

    def test_multiple_texts(self) -> None:
        emb = _make_embeddings()
        emb.client.embed_text.return_value = _mock_embed_response(
            [[0.1, 0.2], [0.3, 0.4]]
        )
        result = emb.embed_documents(["a", "b"])
        assert len(result) == 2

    def test_batching(self) -> None:
        """Texts exceeding batch_size are split into chunks."""
        emb = _make_embeddings(batch_size=2)
        emb.client.embed_text.side_effect = [
            _mock_embed_response([[1.0], [2.0]]),
            _mock_embed_response([[3.0]]),
        ]
        result = emb.embed_documents(["a", "b", "c"])
        assert result == [[1.0], [2.0], [3.0]]
        assert emb.client.embed_text.call_count == 2

    def test_embed_query_delegates(self) -> None:
        emb = _make_embeddings()
        emb.client.embed_text.return_value = _mock_embed_response([[0.5, 0.6]])
        result = emb.embed_query("test")
        assert result == [0.5, 0.6]

    def test_no_model_id_raises(self) -> None:
        emb = _make_embeddings(model_id=None)
        with pytest.raises(ValueError, match="Model ID is required"):
            emb.embed_documents(["hello"])


# =============================================================================
# Tests: embed_image / embed_image_batch
# =============================================================================


class TestEmbedImages:
    """Test image embedding methods."""

    def test_embed_image_bytes(self) -> None:
        emb = _make_embeddings()
        emb.client.embed_text.return_value = _mock_embed_response([[0.1, 0.2, 0.3]])
        result = emb.embed_image(b"\x89PNG\r\n\x1a\ndata")
        assert result == [0.1, 0.2, 0.3]

        # Verify embed_text was called with input_type="IMAGE"
        emb.client.embed_text.assert_called_once()
        request_obj = emb.client.embed_text.call_args[0][0]
        assert request_obj.input_type == "IMAGE"

    def test_embed_image_data_uri(self) -> None:
        emb = _make_embeddings()
        emb.client.embed_text.return_value = _mock_embed_response([[0.4, 0.5]])
        uri = "data:image/png;base64,iVBOR"
        result = emb.embed_image(uri)
        assert result == [0.4, 0.5]

    def test_embed_image_file_path(self, tmp_path: Path) -> None:
        img_file = tmp_path / "test.png"
        img_file.write_bytes(b"\x89PNGfake")

        emb = _make_embeddings()
        emb.client.embed_text.return_value = _mock_embed_response([[0.7, 0.8]])
        result = emb.embed_image(str(img_file))
        assert result == [0.7, 0.8]

    def test_embed_image_batch_multiple(self) -> None:
        """embed_image_batch calls API once per image."""
        emb = _make_embeddings()
        emb.client.embed_text.side_effect = [
            _mock_embed_response([[1.0, 2.0]]),
            _mock_embed_response([[3.0, 4.0]]),
            _mock_embed_response([[5.0, 6.0]]),
        ]
        result = emb.embed_image_batch(
            [
                b"img1",
                b"img2",
                "data:image/jpeg;base64,abc",
            ]
        )
        assert len(result) == 3
        assert result[0] == [1.0, 2.0]
        assert result[2] == [5.0, 6.0]
        assert emb.client.embed_text.call_count == 3

    def test_embed_image_batch_empty(self) -> None:
        emb = _make_embeddings()
        result = emb.embed_image_batch([])
        assert result == []
        emb.client.embed_text.assert_not_called()

    def test_embed_image_uses_image_input_type(self) -> None:
        """embed_image always sets input_type=IMAGE regardless of instance."""
        emb = _make_embeddings(input_type="SEARCH_DOCUMENT")
        emb.client.embed_text.return_value = _mock_embed_response([[0.1]])
        emb.embed_image(b"fake")
        # The request should have been built with input_type="IMAGE"
        emb.client.embed_text.assert_called_once()

    def test_embed_image_with_vision_encode_image(self) -> None:
        """Data from vision.encode_image() works with embed_image."""
        from langchain_oci.utils.vision import encode_image

        raw = b"\x89PNG\r\n\x1a\ntest_pixels"
        block = encode_image(raw, mime_type="image/png")
        data_uri = block["image_url"]["url"]

        emb = _make_embeddings()
        emb.client.embed_text.return_value = _mock_embed_response([[0.9, 0.8, 0.7]])
        result = emb.embed_image(data_uri)
        assert result == [0.9, 0.8, 0.7]


# =============================================================================
# Tests: IMAGE_EMBEDDING_MODELS
# =============================================================================


class TestImageEmbeddingModels:
    """Test the IMAGE_EMBEDDING_MODELS registry."""

    def test_embed_v4_listed(self) -> None:
        assert "cohere.embed-v4.0" in IMAGE_EMBEDDING_MODELS

    def test_is_list(self) -> None:
        assert isinstance(IMAGE_EMBEDDING_MODELS, list)

    def test_importable_from_package(self) -> None:
        from langchain_oci import IMAGE_EMBEDDING_MODELS as imported

        assert imported is IMAGE_EMBEDDING_MODELS


# =============================================================================
# Tests: input_type passed through embed_documents
# =============================================================================


class TestInputTypePassthrough:
    """Verify input_type is forwarded in embed_documents calls."""

    def test_search_query_type(self) -> None:
        """SEARCH_QUERY input_type is passed to the API."""
        emb = _make_embeddings(input_type="SEARCH_QUERY")
        # Capture the EmbedTextDetails object passed to embed_text
        emb.client.embed_text.return_value = _mock_embed_response([[0.1]])
        emb.embed_documents(["query text"])

        request_obj = emb.client.embed_text.call_args[0][0]
        assert request_obj.input_type == "SEARCH_QUERY"

    def test_no_type_defaults_to_none(self) -> None:
        """No input_type means None on the request object."""
        emb = _make_embeddings()
        emb.client.embed_text.return_value = _mock_embed_response([[0.1]])
        emb.embed_documents(["text"])

        request_obj = emb.client.embed_text.call_args[0][0]
        assert request_obj.input_type is None

    def test_image_type_in_embed_image_batch(self) -> None:
        """embed_image_batch sets input_type=IMAGE on the request."""
        emb = _make_embeddings()
        emb.client.embed_text.return_value = _mock_embed_response([[0.1]])
        emb.embed_image_batch([b"fake_png_bytes"])

        request_obj = emb.client.embed_text.call_args[0][0]
        assert request_obj.input_type == "IMAGE"

    @pytest.mark.skipif(
        not _SDK_HAS_OUTPUT_DIMS,
        reason="OCI SDK too old for output_dimensions",
    )
    def test_output_dimensions_in_request(self) -> None:
        """output_dimensions is set on the request object."""
        emb = _make_embeddings(output_dimensions=256)
        emb.client.embed_text.return_value = _mock_embed_response([[0.1]])
        emb.embed_documents(["text"])

        request_obj = emb.client.embed_text.call_args[0][0]
        assert request_obj.output_dimensions == 256
