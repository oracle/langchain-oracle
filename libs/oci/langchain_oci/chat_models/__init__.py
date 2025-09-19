# Copyright (c) 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from langchain_oci.chat_models.oci_generative_ai import ChatOCIGenAI

try:
    from langchain_oci.chat_models.oci_data_science import ChatOCIModelDeployment

except ModuleNotFoundError as ex:
    # Default message
    message = ex.msg
    # For langchain_openai, show the message with pip install command.
    if ex.name == "langchain_openai":
        message = (
            "No module named langchain_openai. "
            "Please install it with `pip install langchain_openai`"
        )

    # Create a placeholder class here so that
    # users can import the class without error.
    # Users will see the error message when they try to initialize an instance.
    class ChatOCIModelDeployment:
        """Placeholder class for ChatOCIModelDeployment
        when langchain-openai is not installed."""

        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError(message)


__all__ = [
    "ChatOCIGenAI",
    "ChatOCIModelDeployment",
]
