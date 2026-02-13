# Copyright (c) 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from langchain_oci.agents.react import create_oci_agent
from langchain_oci.chat_models.oci_data_science import (
    ChatOCIModelDeployment,
    ChatOCIModelDeploymentTGI,
    ChatOCIModelDeploymentVLLM,
)
from langchain_oci.chat_models.oci_generative_ai import ChatOCIGenAI, ChatOCIOpenAI
from langchain_oci.common.auth import OCIAuthType
from langchain_oci.embeddings.oci_data_science_model_deployment_endpoint import (
    OCIModelDeploymentEndpointEmbeddings,
)
from langchain_oci.embeddings.oci_generative_ai import OCIGenAIEmbeddings
from langchain_oci.llms.oci_data_science_model_deployment_endpoint import (
    BaseOCIModelDeployment,
    OCIModelDeploymentLLM,
    OCIModelDeploymentTGI,
    OCIModelDeploymentVLLM,
)
from langchain_oci.llms.oci_generative_ai import OCIGenAI, OCIGenAIBase
from langchain_oci.utils.vision import (
    IMAGE_EMBEDDING_MODELS,
    VISION_MODELS,
    encode_image,
    is_vision_model,
    load_image,
    to_data_uri,
)

__all__ = [
    "ChatOCIGenAI",
    "ChatOCIModelDeployment",
    "ChatOCIModelDeploymentTGI",
    "ChatOCIModelDeploymentVLLM",
    "ChatOCIOpenAI",
    "OCIAuthType",
    "OCIGenAIEmbeddings",
    "OCIModelDeploymentEndpointEmbeddings",
    "OCIGenAIBase",
    "OCIGenAI",
    "BaseOCIModelDeployment",
    "OCIModelDeploymentLLM",
    "OCIModelDeploymentTGI",
    "OCIModelDeploymentVLLM",
    "create_oci_agent",
    # Vision / image utilities
    "load_image",
    "encode_image",
    "is_vision_model",
    "to_data_uri",
    "VISION_MODELS",
    "IMAGE_EMBEDDING_MODELS",
]
