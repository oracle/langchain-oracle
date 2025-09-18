# Copyright (c) 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from langchain_oci.chat_models.oci_generative_ai import ChatOCIGenAI

try:
    from langchain_oci.chat_models.oci_data_science import ChatOCIModelDeployment

except ModuleNotFoundError as ex:
    if ex.name == "langchain_openai":
        message = (
            "No module named langchain_openai. "
            "Please install it with `pip install langchain_openai`"
        )
    else:
        message = ex.msg

    class ChatOCIModelDeployment:
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError(message)


__all__ = [
    "ChatOCIGenAI",
    "ChatOCIModelDeployment",
]
