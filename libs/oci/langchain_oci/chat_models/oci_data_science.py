# Copyright (c) 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Chat model for OCI data science model deployment endpoint."""
from langchain_openai import ChatOpenAI

from langchain_oci.llms import BaseOCIModelDeployment


class ChatOCIModelDeployment(BaseOCIModelDeployment, ChatOpenAI):
    """OCI Data Science Model Deployment chat model integration.

    This class inherits from ChatOpenAI LangChain client.
    You can use all the parameters supported by ChatOpenAI LangChain client.

    Prerequisite
        The OCI Model Deployment plugins are installable only on
        python version 3.9 and above. If you're working inside the notebook,
        try installing the python 3.10 based conda pack and running the
        following setup.


    Setup:
        Install ``oracle-ads`` and ``langchain-openai``.

        .. code-block:: bash

            pip install -U oracle-ads langchain-openai

        Use `ads.set_auth()` to configure authentication.
        For example, to use OCI resource_principal for authentication:

        .. code-block:: python

            import ads
            ads.set_auth("resource_principal")

        For more details on authentication, see:
        https://accelerated-data-science.readthedocs.io/en/latest/user_guide/cli/authentication.html

        Make sure to have the required policies to access the OCI Data
        Science Model Deployment endpoint. See:
        https://docs.oracle.com/en-us/iaas/data-science/using/model-dep-policies-auth.htm


    Example:

        .. code-block:: python

            from langchain_oci import ChatOCIModelDeployment

            llm = ChatOCIModelDeployment(
                endpoint="https://modeldeployment.us-ashburn-1.oci.customer-oci.com/<ocid>/predictWithResponseStream",
                model="odsc-llm",
            )
            llm.stream("tell me a joke.")


    Key init args:
        endpoint: str
            The OCI model deployment endpoint. For example:
            Non-streaming: https://modeldeployment.<region>.oci.customer-oci.com/<OCID>/predict
            Streaming: https://modeldeployment.<region>.oci.customer-oci.com/<OCID>/predictWithResponseStream
        auth: dict
            ADS auth dictionary for OCI authentication.
    """

    pass
