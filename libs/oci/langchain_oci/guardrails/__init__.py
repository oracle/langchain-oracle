# Copyright (c) 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""OCI Generative AI Guardrails integration for LangChain."""

from langchain_oci.guardrails.oci_guardrails import OCIGuardrails

__all__ = ["OCIGuardrails"]

# Agent middleware requires langchain >= 1.0 (AgentMiddleware / create_agent).
# On the Python 3.9 / langchain 0.3.x matrix it is unavailable; the OCIGuardrails
# Runnable above still works there.
try:
    from langchain_oci.guardrails.middleware import (
        OCIGuardrailsMiddleware,
        OCIGuardrailsViolationError,
    )

    __all__ += ["OCIGuardrailsMiddleware", "OCIGuardrailsViolationError"]
except ImportError:
    pass
