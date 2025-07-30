# Copyright (c) 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from langchain_oci.embeddings.oci_data_science_model_deployment_endpoint import (
    OCIModelDeploymentEndpointEmbeddings,
)
from langchain_oci.embeddings.oci_generative_ai import OCIGenAIEmbeddings

__all__ = ["OCIModelDeploymentEndpointEmbeddings", "OCIGenAIEmbeddings"]
