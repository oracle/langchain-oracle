# Copyright (c) 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


from langchain_oci.llms.oci_generative_ai import OCIGenAI, OCIGenAIBase

try:
    from langchain_oci.llms.oci_data_science_model_deployment_endpoint import (
        BaseOCIModelDeployment,
        OCIModelDeploymentLLM,
    )
except ModuleNotFoundError as ex:

    if ex.name == "langchain_openai":
        message = (
            "No module named langchain_openai. "
            "Please install it with `pip install langchain_openai`"
        )
    else:
        message = ex.msg

    class BaseOCIModelDeployment:
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError(message)

    class OCIModelDeploymentLLM(BaseOCIModelDeployment):
        pass


__all__ = [
    "OCIGenAIBase",
    "OCIGenAI",
    "BaseOCIModelDeployment",
    "OCIModelDeploymentLLM",
]
