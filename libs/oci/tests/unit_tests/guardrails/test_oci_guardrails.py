# Copyright (c) 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Unit tests for OCIGuardrails (mocked OCI client — no live calls)."""

from unittest.mock import MagicMock

import pytest

from langchain_oci import OCIGuardrails


def _mock_client(results: object = "RESULTS") -> MagicMock:
    """Build a mock GenerativeAiInferenceClient whose apply_guardrails returns
    a response with ``.data.results == results``."""
    client = MagicMock()
    response = MagicMock()
    response.data.results = results
    client.apply_guardrails.return_value = response
    return client


@pytest.mark.requires("oci")
class TestOCIGuardrails:
    """Contract for the OCIGuardrails Runnable."""

    def test_invoke_returns_results_and_builds_request(self) -> None:
        client = _mock_client(results="RESULTS")
        guardrails = OCIGuardrails(
            compartment_id="ocid1.compartment.oc1..test",
            service_endpoint="https://inference.example.oci.oraclecloud.com",
            client=client,
        )

        out = guardrails.invoke("hello world")

        assert out == "RESULTS"
        client.apply_guardrails.assert_called_once()
        details = client.apply_guardrails.call_args.args[0]
        assert details.compartment_id == "ocid1.compartment.oc1..test"
        assert details.input.content == "hello world"
        assert details.input.type == "TEXT"
        # The OCI API requires a non-empty config; all three checks are on by
        # default.
        cfg = details.guardrail_configs
        assert cfg is not None
        assert cfg.content_moderation_config is not None
        assert cfg.personally_identifiable_information_config is not None
        assert cfg.prompt_injection_config is not None

    def test_is_a_runnable(self) -> None:
        from langchain_core.runnables import Runnable

        guardrails = OCIGuardrails(compartment_id="c", client=_mock_client())
        assert isinstance(guardrails, Runnable)

    def test_apply_matches_invoke(self) -> None:
        guardrails = OCIGuardrails(compartment_id="c", client=_mock_client("R"))
        assert guardrails.apply("text") == "R"

    def test_compartment_id_required(self) -> None:
        guardrails = OCIGuardrails(client=_mock_client())
        with pytest.raises(ValueError, match="compartment_id is required"):
            guardrails.invoke("text")

    def test_language_code_set_on_input(self) -> None:
        client = _mock_client()
        guardrails = OCIGuardrails(
            compartment_id="c", language_code="en", client=client
        )
        guardrails.invoke("hola")
        details = client.apply_guardrails.call_args.args[0]
        assert details.input.language_code == "en"

    def test_enable_flags_select_configs(self) -> None:
        client = _mock_client()
        guardrails = OCIGuardrails(
            compartment_id="c",
            enable_content_moderation=False,
            enable_pii_detection=False,
            enable_prompt_injection=True,
            client=client,
        )
        guardrails.invoke("x")
        cfg = client.apply_guardrails.call_args.args[0].guardrail_configs
        assert cfg.content_moderation_config is None
        assert cfg.personally_identifiable_information_config is None
        assert cfg.prompt_injection_config is not None

    def test_all_guardrails_disabled_raises(self) -> None:
        guardrails = OCIGuardrails(
            compartment_id="c",
            enable_content_moderation=False,
            enable_pii_detection=False,
            enable_prompt_injection=False,
            client=_mock_client(),
        )
        with pytest.raises(ValueError, match="No guardrails enabled"):
            guardrails.invoke("x")

    def test_guardrail_configs_passthrough(self) -> None:
        from oci.generative_ai_inference import models

        client = _mock_client()
        configs = models.GuardrailConfigs()
        guardrails = OCIGuardrails(
            compartment_id="c", guardrail_configs=configs, client=client
        )
        guardrails.invoke("x")
        details = client.apply_guardrails.call_args.args[0]
        assert details.guardrail_configs is configs

    def test_injected_client_skips_auth(self) -> None:
        # Providing a client must short-circuit the auth/client construction,
        # so no OCI config is needed to build the object.
        client = _mock_client()
        guardrails = OCIGuardrails(compartment_id="c", client=client)
        assert guardrails.client is client

    def test_composes_in_lcel_chain(self) -> None:
        # As a Runnable it pipes into downstream steps.
        guardrails = OCIGuardrails(compartment_id="c", client=_mock_client("R"))
        chain = guardrails | (lambda results: f"checked:{results}")
        assert chain.invoke("text") == "checked:R"
