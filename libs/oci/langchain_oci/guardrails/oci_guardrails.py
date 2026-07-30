# Copyright (c) 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""LangChain-native interface to the OCI Generative AI Guardrails API.

`OCIGuardrails` wraps the OCI SDK ``GenerativeAiInferenceClient.apply_guardrails``
operation (content moderation, PII detection, and prompt-injection protection)
as a LangChain :class:`~langchain_core.runnables.Runnable`, so guardrails compose
in LCEL chains and reuse the same OCI authentication as ``ChatOCIGenAI``.

Example:
    .. code-block:: python

        from langchain_oci import OCIGuardrails

        guardrails = OCIGuardrails(
            service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
            compartment_id="ocid1.compartment.oc1..example",
        )

        results = guardrails.invoke("Ignore all previous instructions and ...")
        # results.content_moderation / .personally_identifiable_information /
        # .prompt_injection  -> OCI SDK result objects

The constructor accepts the same auth options as the chat/embeddings classes
(``auth_type``, ``auth_profile``, ``auth_file_location``, ``service_endpoint``).
"""

from typing import Any, Dict, Optional

from langchain_core.runnables import RunnableConfig, RunnableSerializable
from pydantic import ConfigDict, model_validator

from langchain_oci.common.auth import create_oci_client_kwargs

#: ``type`` discriminator for the SDK ``GuardrailsTextInput`` payload.
_TEXT_INPUT_TYPE = "TEXT"


class OCIGuardrails(RunnableSerializable[str, Any]):
    """Apply OCI Generative AI guardrails to text as a LangChain Runnable.

    The returned value is the SDK ``GuardrailsResults`` object, which exposes
    ``content_moderation``, ``personally_identifiable_information`` and
    ``prompt_injection`` findings. Pass a pre-built
    ``oci.generative_ai_inference.models.GuardrailConfigs`` via
    ``guardrail_configs`` to tune individual checks; when omitted, the service
    applies its default configuration.
    """

    auth_type: Optional[str] = "API_KEY"
    """Authentication type, one of API_KEY, SECURITY_TOKEN, INSTANCE_PRINCIPAL,
    RESOURCE_PRINCIPAL. Mirrors the other langchain-oci OCI classes."""

    auth_profile: Optional[str] = "DEFAULT"
    """Profile name in ``~/.oci/config``."""

    auth_file_location: Optional[str] = "~/.oci/config"
    """Path to the OCI config file."""

    service_endpoint: Optional[str] = None
    """OCI Generative AI inference endpoint,
    e.g. ``https://inference.generativeai.us-chicago-1.oci.oraclecloud.com``."""

    compartment_id: Optional[str] = None
    """OCID of the compartment. Required to apply guardrails."""

    language_code: Optional[str] = None
    """Optional BCP-47 language code for the input text (e.g. ``en``)."""

    enable_content_moderation: bool = True
    """Enable content moderation (hate, violence, etc.) when ``guardrail_configs``
    is not supplied."""

    enable_pii_detection: bool = True
    """Enable personally-identifiable-information detection when
    ``guardrail_configs`` is not supplied."""

    enable_prompt_injection: bool = True
    """Enable prompt-injection protection when ``guardrail_configs`` is not
    supplied."""

    guardrail_configs: Optional[Any] = None
    """Optional pre-built ``GuardrailConfigs`` for full control over individual
    checks. When set, it is used as-is and the ``enable_*`` flags are ignored.
    When ``None``, a config is built from the ``enable_*`` flags (the OCI API
    requires at least one guardrail to be configured)."""

    guardrail_version_config: Optional[Any] = None
    """Optional pre-built ``GuardrailVersionConfig`` pinning a guardrail
    version. When ``None``, the latest version is used."""

    client: Any = None
    """An OCI ``GenerativeAiInferenceClient``. Built from the auth options
    above when not supplied; inject a client directly to reuse an existing one
    (or a mock in tests)."""

    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Dict:
        """Build an OCI inference client from the auth options if not provided."""
        if not isinstance(values, dict) or values.get("client") is not None:
            return values

        try:
            import oci

            client_kwargs = create_oci_client_kwargs(
                auth_type=values.get("auth_type", "API_KEY"),
                service_endpoint=values.get("service_endpoint"),
                auth_file_location=values.get("auth_file_location", "~/.oci/config"),
                auth_profile=values.get("auth_profile", "DEFAULT"),
            )
            values["client"] = oci.generative_ai_inference.GenerativeAiInferenceClient(
                **client_kwargs
            )
        except ImportError as ex:
            raise ModuleNotFoundError(
                "Could not import oci python package. "
                "Please make sure you have the oci package installed."
            ) from ex
        except Exception as e:
            raise ValueError(
                """Could not authenticate with OCI client.
                Please check if ~/.oci/config exists.
                If INSTANCE_PRINCIPAL or RESOURCE_PRINCIPAL is used,
                please check the specified auth_profile, auth_file_location
                and auth_type are valid.""",
                e,
            ) from e

        return values

    def _resolve_configs(self) -> Any:
        """Return the ``GuardrailConfigs`` to apply.

        Uses ``guardrail_configs`` when supplied, otherwise builds one from the
        ``enable_*`` flags. The OCI API rejects an empty config, so at least one
        guardrail must be enabled.
        """
        if self.guardrail_configs is not None:
            return self.guardrail_configs

        from oci.generative_ai_inference import models

        config_kwargs: Dict[str, Any] = {}
        if self.enable_content_moderation:
            config_kwargs["content_moderation_config"] = (
                models.ContentModerationConfiguration()
            )
        if self.enable_pii_detection:
            config_kwargs["personally_identifiable_information_config"] = (
                models.PersonallyIdentifiableInformationConfiguration()
            )
        if self.enable_prompt_injection:
            config_kwargs["prompt_injection_config"] = (
                models.PromptInjectionConfiguration()
            )

        if not config_kwargs:
            raise ValueError(
                "No guardrails enabled. Enable at least one of "
                "enable_content_moderation / enable_pii_detection / "
                "enable_prompt_injection, or pass guardrail_configs."
            )
        return models.GuardrailConfigs(**config_kwargs)

    def _build_details(self, text: str) -> Any:
        """Build the ``ApplyGuardrailsDetails`` payload for ``text``."""
        from oci.generative_ai_inference import models

        if not self.compartment_id:
            raise ValueError("compartment_id is required to apply guardrails.")

        text_input = models.GuardrailsTextInput(type=_TEXT_INPUT_TYPE, content=text)
        if self.language_code is not None:
            text_input.language_code = self.language_code

        details_kwargs: Dict[str, Any] = {
            "compartment_id": self.compartment_id,
            "input": text_input,
            "guardrail_configs": self._resolve_configs(),
        }
        if self.guardrail_version_config is not None:
            details_kwargs["guardrail_version_config"] = self.guardrail_version_config

        return models.ApplyGuardrailsDetails(**details_kwargs)

    def apply(self, text: str) -> Any:
        """Apply guardrails to ``text`` and return the SDK ``GuardrailsResults``."""
        response = self.client.apply_guardrails(self._build_details(text))
        return response.data.results

    def invoke(
        self,
        input: str,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Any:
        """Runnable entry point: apply guardrails to the input text."""
        return self.apply(input)
