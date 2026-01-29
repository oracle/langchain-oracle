# Copyright (c) 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Vision and image utilities for OCI Generative AI.

This module provides helper functions for working with images across
OCI Generative AI — both vision-capable chat models and multimodal
embedding models. It is the single source of truth for image encoding
and model registries.

Core functions:
- ``to_data_uri`` — Convert any image input to a data URI string.
- ``load_image`` — Load a file as a content block for chat models.
- ``encode_image`` — Encode bytes as a content block for chat models.
- ``is_vision_model`` — Check if a model supports vision inputs.

Model registries:
- ``VISION_MODELS`` — Chat models that accept image inputs.
- ``IMAGE_EMBEDDING_MODELS`` — Embedding models that support images.

Example:
    >>> from langchain_oci import ChatOCIGenAI, load_image
    >>> from langchain_core.messages import HumanMessage
    >>>
    >>> llm = ChatOCIGenAI(model_id="meta.llama-3.2-90b-vision-instruct")
    >>> message = HumanMessage(
    ...     content=[
    ...         {"type": "text", "text": "What's in this image?"},
    ...         load_image("./photo.jpg"),
    ...     ]
    ... )
    >>> response = llm.invoke([message])
"""

import base64
import mimetypes
from pathlib import Path
from typing import List, Union

# =============================================================================
# Model registries
# =============================================================================

# Vision-capable chat models available in OCI Generative AI
VISION_MODELS: List[str] = [
    # Meta Llama models
    "meta.llama-3.2-90b-vision-instruct",
    "meta.llama-3.2-11b-vision-instruct",
    "meta.llama-4-scout-17b-16e-instruct",
    "meta.llama-4-maverick-17b-128e-instruct-fp8",
    # Google Gemini models
    "google.gemini-2.5-flash",
    "google.gemini-2.5-pro",
    "google.gemini-2.5-flash-lite",
    # xAI Grok models (Grok 4 and later)
    "xai.grok-4",
    "xai.grok-4-1-fast-reasoning",
    "xai.grok-4-1-fast-non-reasoning",
    "xai.grok-4-fast-reasoning",
    "xai.grok-4-fast-non-reasoning",
    # Cohere models
    "cohere.command-a-vision",
]

# Embedding models that support image input via input_type="IMAGE".
# These models embed text and images into the same vector space.
IMAGE_EMBEDDING_MODELS: List[str] = [
    "cohere.embed-v4.0",
    "cohere.embed-multilingual-image-v3.0",
]

# =============================================================================
# Core image encoding
# =============================================================================


def to_data_uri(
    image: Union[str, bytes, Path],
    mime_type: str = "image/png",
) -> str:
    """Convert an image input to a ``data:`` URI string.

    This is the core encoding function used by ``load_image``,
    ``encode_image``, and ``OCIGenAIEmbeddings.embed_image``.

    Accepts:
    - **Raw bytes** — base64-encodes with the given *mime_type*.
    - **A data URI string** (``data:image/...;base64,...``) — returned as-is.
    - **A file path** (str or Path) — reads the file, auto-detects MIME type.

    Args:
        image: File path, raw bytes, or data URI string.
        mime_type: MIME type when *image* is raw bytes.
            Ignored for file paths (auto-detected) and data URIs.

    Returns:
        A ``data:mime;base64,...`` string.

    Example:
        >>> uri = to_data_uri(b"\\x89PNG...", mime_type="image/png")
        >>> uri = to_data_uri("photo.jpg")
        >>> uri = to_data_uri("data:image/png;base64,iVBOR...")  # passthrough
    """
    if isinstance(image, bytes):
        encoded = base64.standard_b64encode(image).decode("utf-8")
        return f"data:{mime_type};base64,{encoded}"

    image_str = str(image)

    if image_str.startswith("data:"):
        return image_str

    # File path
    path = Path(image_str)
    detected_mime = (
        mimetypes.guess_type(str(path))[0] or "image/png"
    )
    with open(path, "rb") as f:
        encoded = base64.standard_b64encode(f.read()).decode(
            "utf-8"
        )
    return f"data:{detected_mime};base64,{encoded}"


# =============================================================================
# Content block helpers (for chat model HumanMessage content)
# =============================================================================


def load_image(file_path: Union[str, Path]) -> dict:
    """Load a local image file and return it as a content block for vision models.

    This function reads an image file from disk and returns a dict that can be
    used directly in HumanMessage content for ChatOCIGenAI vision models.

    Args:
        file_path: Path to the image file. Supports PNG, JPEG, GIF, and WebP formats.

    Returns:
        A dict in the format {"type": "image_url", "image_url": {"url": "data:..."}}
        ready to use in HumanMessage content.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        PermissionError: If the file cannot be read due to permissions.

    Example:
        >>> from langchain_oci import ChatOCIGenAI, load_image
        >>> llm = ChatOCIGenAI(model_id="meta.llama-3.2-90b-vision-instruct")
        >>> message = HumanMessage(
        ...     content=[
        ...         {"type": "text", "text": "Describe this image"},
        ...         load_image("./photo.jpg"),
        ...     ]
        ... )
        >>> response = llm.invoke([message])
    """
    data_url = to_data_uri(file_path)
    return {"type": "image_url", "image_url": {"url": data_url}}


def encode_image(image_bytes: bytes, mime_type: str = "image/png") -> dict:
    """Encode image bytes as a content block for vision models.

    This function takes raw image bytes and returns a dict that can be
    used directly in HumanMessage content for ChatOCIGenAI vision models.

    Args:
        image_bytes: Raw image bytes (e.g., from an HTTP response or PIL image).
        mime_type: The MIME type of the image. Defaults to "image/png".
            Common values: "image/png", "image/jpeg", "image/gif", "image/webp".

    Returns:
        A dict in the format {"type": "image_url", "image_url": {"url": "data:..."}}

    Example:
        >>> # From HTTP response
        >>> import requests
        >>> from langchain_core.messages import HumanMessage
        >>> response = requests.get("https://example.com/image.png")
        >>> message = HumanMessage(
        ...     content=[
        ...         {"type": "text", "text": "What's in this image?"},
        ...         encode_image(response.content, "image/png"),
        ...     ]
        ... )
    """
    data_url = to_data_uri(image_bytes, mime_type=mime_type)
    return {"type": "image_url", "image_url": {"url": data_url}}


# =============================================================================
# Model detection
# =============================================================================


def is_vision_model(model_id: str) -> bool:
    """Check if a model supports vision/image inputs.

    This function determines whether a given model ID corresponds to a
    vision-capable model that can process image inputs.

    Note: For dedicated endpoints with custom OCIDs, this function may return
    False even if the underlying model supports vision. In such cases, you can
    still use vision features - this function is just a convenience helper.

    Args:
        model_id: The OCI Generative AI model ID to check.

    Returns:
        True if the model supports vision inputs, False otherwise.

    Example:
        >>> is_vision_model("meta.llama-3.2-90b-vision-instruct")
        True
        >>> is_vision_model("meta.llama-3.3-70b-instruct")
        False
        >>> is_vision_model("meta.llama-4-scout-17b-16e-instruct")
        True
    """
    return model_id in VISION_MODELS
