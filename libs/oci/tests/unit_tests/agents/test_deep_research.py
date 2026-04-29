# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Unit tests for create_deep_research_agent helper function."""

import sys
from contextlib import contextmanager
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.tools import tool

from langchain_oci.common.auth import OCIAuthType


@tool
def dummy_tool(x: str) -> str:
    """A dummy tool for testing."""
    return f"Result: {x}"


@contextmanager
def mock_deepagents() -> Generator[MagicMock, None, None]:
    """Mock the deepagents module and its create_deep_agent function."""
    mock_create = MagicMock(return_value=MagicMock())

    # Create a mock deepagents module
    mock_module = MagicMock()
    mock_module.create_deep_agent = mock_create

    # Store original if exists
    original = sys.modules.get("deepagents")

    # Insert mock into sys.modules
    sys.modules["deepagents"] = mock_module

    try:
        yield mock_create
    finally:
        # Restore original state
        if original is not None:
            sys.modules["deepagents"] = original
        elif "deepagents" in sys.modules:
            del sys.modules["deepagents"]


# deepagents requires Python 3.11+; Python 3.14 not yet tested
_SKIP_DEEPAGENTS_314 = sys.version_info >= (3, 14)


@pytest.mark.requires("oci", "langgraph", "deepagents")
@pytest.mark.skipif(_SKIP_DEEPAGENTS_314, reason="deepagents not supported on 3.14+")
class TestCreateDeepResearchAgent:
    """Tests for create_deep_research_agent function."""

    def test_creates_agent_with_minimal_args(self) -> None:
        """Test agent creation with just tools and compartment_id."""
        from langchain_oci.agents.deep_research.agent import create_deep_research_agent

        with patch.dict("os.environ", {"OCI_COMPARTMENT_ID": "test-compartment"}):
            with patch(
                "langchain_oci.chat_models.oci_generative_ai.ChatOCIGenAI"
            ) as mock_llm_class:
                with mock_deepagents() as mock_create:
                    mock_llm_instance = MagicMock()
                    mock_llm_class.return_value = mock_llm_instance

                    agent = create_deep_research_agent(tools=[dummy_tool])

                    # Verify ChatOCIGenAI was created with correct params
                    mock_llm_class.assert_called_once()
                    call_kwargs = mock_llm_class.call_args.kwargs
                    expected_model = "google.gemini-2.5-pro"
                    assert call_kwargs["model_id"] == expected_model
                    assert call_kwargs["compartment_id"] == "test-compartment"

                    # Verify create_deep_agent was called
                    mock_create.assert_called_once()
                    assert agent is not None

    def test_datastore_path_uses_lightweight_agent(self) -> None:
        """Datastore-backed helper should avoid deepagents planning middleware."""
        from langchain_oci.agents.deep_research.agent import create_deep_research_agent

        fake_store = MagicMock()
        fake_store.name = "opensearch"

        with patch.dict("os.environ", {"OCI_COMPARTMENT_ID": "test-compartment"}):
            with patch(
                "langchain_oci.chat_models.oci_generative_ai.ChatOCIGenAI"
            ) as mock_llm_class:
                with patch(
                    "langchain_oci.agents.deep_research.agent.create_datastore_tools",
                    return_value=[],
                ):
                    with patch("langchain.agents.create_agent") as mock_create_agent:
                        mock_llm_class.return_value = MagicMock()
                        mock_create_agent.return_value = MagicMock()

                        create_deep_research_agent(datastores={"runbooks": fake_store})

                        mock_create_agent.assert_called_once()

    def test_empty_middleware_uses_lightweight_agent(self) -> None:
        """Explicitly disabling middleware should avoid deepagents planning."""
        from langchain_oci.agents.deep_research.agent import create_deep_research_agent

        with patch.dict("os.environ", {"OCI_COMPARTMENT_ID": "test-compartment"}):
            with patch(
                "langchain_oci.chat_models.oci_generative_ai.ChatOCIGenAI"
            ) as mock_llm_class:
                with patch("langchain.agents.create_agent") as mock_create_agent:
                    mock_llm_class.return_value = MagicMock()
                    mock_create_agent.return_value = MagicMock()

                    create_deep_research_agent(
                        tools=[dummy_tool],
                        middleware=[],
                    )

                    mock_create_agent.assert_called_once()

    def test_backend_forces_deep_agent_path(self) -> None:
        """Backend param should force deep agent path even with datastores."""
        from langchain_oci.agents.deep_research.agent import create_deep_research_agent

        fake_store = MagicMock()
        fake_store.name = "opensearch"

        with patch.dict("os.environ", {"OCI_COMPARTMENT_ID": "test"}):
            with patch("langchain_oci.chat_models.oci_generative_ai.ChatOCIGenAI"):
                with patch(
                    "langchain_oci.agents.deep_research.agent.create_datastore_tools",
                    return_value=[],
                ):
                    with mock_deepagents() as mock_create:
                        create_deep_research_agent(
                            datastores={"runbooks": fake_store},
                            backend=MagicMock(),
                        )
                        # Should use create_deep_agent, not lightweight
                        mock_create.assert_called_once()
                        assert "backend" in mock_create.call_args.kwargs

    def test_cache_forces_deep_agent_path(self) -> None:
        """Cache param should force deep agent path."""
        from langchain_oci.agents.deep_research.agent import create_deep_research_agent

        with patch.dict("os.environ", {"OCI_COMPARTMENT_ID": "test"}):
            with patch("langchain_oci.chat_models.oci_generative_ai.ChatOCIGenAI"):
                with mock_deepagents() as mock_create:
                    create_deep_research_agent(
                        tools=[dummy_tool],
                        middleware=[],
                        cache=MagicMock(),
                    )
                    # middleware=[] alone would use lightweight,
                    # but cache forces deep agent
                    mock_create.assert_called_once()
                    assert "cache" in mock_create.call_args.kwargs

    def test_interrupt_on_forces_deep_agent_path(self) -> None:
        """interrupt_on should force deep agent path."""
        from langchain_oci.agents.deep_research.agent import create_deep_research_agent

        with patch.dict("os.environ", {"OCI_COMPARTMENT_ID": "test"}):
            with patch("langchain_oci.chat_models.oci_generative_ai.ChatOCIGenAI"):
                with mock_deepagents() as mock_create:
                    create_deep_research_agent(
                        tools=[dummy_tool],
                        middleware=[],
                        interrupt_on={"edit_file": True},
                    )
                    mock_create.assert_called_once()
                    assert "interrupt_on" in mock_create.call_args.kwargs

    def test_raises_without_compartment_id(self) -> None:
        """Test that error is raised when no compartment_id available."""
        from langchain_oci.agents.deep_research.agent import create_deep_research_agent

        with patch.dict("os.environ", {}, clear=True):
            with mock_deepagents():
                with pytest.raises(ValueError, match="compartment_id must be provided"):
                    create_deep_research_agent(tools=[dummy_tool])

    def test_raises_without_deepagents_installed(self) -> None:
        """Test that ImportError is raised when deepagents not installed."""
        from langchain_oci.agents.deep_research.agent import create_deep_research_agent

        with patch.dict("os.environ", {"OCI_COMPARTMENT_ID": "test"}):
            # Remove deepagents from sys.modules to simulate not installed
            original = sys.modules.get("deepagents")
            if "deepagents" in sys.modules:
                del sys.modules["deepagents"]

            # Make import fail by setting to None (intentional for testing)
            sys.modules["deepagents"] = None  # type: ignore[assignment]

            try:
                with pytest.raises(ImportError, match="deepagents"):
                    create_deep_research_agent(tools=[dummy_tool])
            finally:
                if original is not None:
                    sys.modules["deepagents"] = original
                elif "deepagents" in sys.modules:
                    del sys.modules["deepagents"]

    def test_passes_system_prompt(self) -> None:
        """Test that system_prompt is passed through as-is."""
        from langchain_oci.agents.deep_research.agent import create_deep_research_agent

        with patch.dict("os.environ", {"OCI_COMPARTMENT_ID": "test"}):
            with patch("langchain_oci.chat_models.oci_generative_ai.ChatOCIGenAI"):
                with mock_deepagents() as mock_create:
                    create_deep_research_agent(
                        tools=[dummy_tool],
                        system_prompt="You are helpful.",
                    )

                    call_kwargs = mock_create.call_args.kwargs
                    assert call_kwargs["system_prompt"] == "You are helpful."

    def test_passes_checkpointer(self) -> None:
        """Test that checkpointer is passed through."""
        from langchain_oci.agents.deep_research.agent import create_deep_research_agent

        with patch.dict("os.environ", {"OCI_COMPARTMENT_ID": "test"}):
            with patch("langchain_oci.chat_models.oci_generative_ai.ChatOCIGenAI"):
                with mock_deepagents() as mock_create:
                    mock_checkpointer = MagicMock()

                    create_deep_research_agent(
                        tools=[dummy_tool],
                        checkpointer=mock_checkpointer,
                    )

                    call_kwargs = mock_create.call_args.kwargs
                    assert call_kwargs["checkpointer"] == mock_checkpointer

    def test_passes_oci_specific_options(self) -> None:
        """Test OCI-specific options are passed to ChatOCIGenAI."""
        from langchain_oci.agents.deep_research.agent import create_deep_research_agent

        with patch.dict("os.environ", {"OCI_COMPARTMENT_ID": "test"}):
            with patch(
                "langchain_oci.chat_models.oci_generative_ai.ChatOCIGenAI"
            ) as mock_llm_class:
                with mock_deepagents():
                    create_deep_research_agent(
                        tools=[dummy_tool],
                        compartment_id="explicit-compartment",
                        auth_profile="CUSTOM",
                        temperature=0.5,
                        max_tokens=1024,
                    )

                    call_kwargs = mock_llm_class.call_args.kwargs
                    assert call_kwargs["compartment_id"] == "explicit-compartment"
                    assert call_kwargs["auth_profile"] == "CUSTOM"
                    assert call_kwargs["model_kwargs"]["temperature"] == 0.5
                    assert call_kwargs["model_kwargs"]["max_tokens"] == 1024

    def test_auth_type_as_enum(self) -> None:
        """Test that auth_type can be passed as OCIAuthType enum."""
        from langchain_oci.agents.deep_research.agent import create_deep_research_agent

        with patch.dict("os.environ", {"OCI_COMPARTMENT_ID": "test"}):
            with patch(
                "langchain_oci.chat_models.oci_generative_ai.ChatOCIGenAI"
            ) as mock_llm_class:
                with mock_deepagents():
                    create_deep_research_agent(
                        tools=[dummy_tool],
                        auth_type=OCIAuthType.SECURITY_TOKEN,
                    )

                    call_kwargs = mock_llm_class.call_args.kwargs
                    assert call_kwargs["auth_type"] == "SECURITY_TOKEN"

    def test_auth_type_as_string(self) -> None:
        """Test that auth_type can be passed as string."""
        from langchain_oci.agents.deep_research.agent import create_deep_research_agent

        with patch.dict("os.environ", {"OCI_COMPARTMENT_ID": "test"}):
            with patch(
                "langchain_oci.chat_models.oci_generative_ai.ChatOCIGenAI"
            ) as mock_llm_class:
                with mock_deepagents():
                    create_deep_research_agent(
                        tools=[dummy_tool],
                        auth_type="INSTANCE_PRINCIPAL",
                    )

                    call_kwargs = mock_llm_class.call_args.kwargs
                    assert call_kwargs["auth_type"] == "INSTANCE_PRINCIPAL"

    def test_service_endpoint_from_region(self) -> None:
        """Test that service_endpoint is constructed from OCI_REGION."""
        from langchain_oci.agents.deep_research.agent import create_deep_research_agent

        with patch.dict(
            "os.environ",
            {"OCI_COMPARTMENT_ID": "test", "OCI_REGION": "us-ashburn-1"},
        ):
            with patch(
                "langchain_oci.chat_models.oci_generative_ai.ChatOCIGenAI"
            ) as mock_llm_class:
                with mock_deepagents():
                    create_deep_research_agent(tools=[dummy_tool])

                    call_kwargs = mock_llm_class.call_args.kwargs
                    expected = (
                        "https://inference.generativeai."
                        "us-ashburn-1.oci.oraclecloud.com"
                    )
                    assert call_kwargs["service_endpoint"] == expected

    def test_explicit_service_endpoint(self) -> None:
        """Test that explicit service_endpoint takes precedence."""
        from langchain_oci.agents.deep_research.agent import create_deep_research_agent

        with patch.dict(
            "os.environ",
            {"OCI_COMPARTMENT_ID": "test", "OCI_REGION": "us-chicago-1"},
        ):
            with patch(
                "langchain_oci.chat_models.oci_generative_ai.ChatOCIGenAI"
            ) as mock_llm_class:
                with mock_deepagents():
                    create_deep_research_agent(
                        tools=[dummy_tool],
                        service_endpoint="https://custom.endpoint.com",
                    )

                    call_kwargs = mock_llm_class.call_args.kwargs
                    expected = "https://custom.endpoint.com"
                    assert call_kwargs["service_endpoint"] == expected

    def test_passes_deep_agent_options(self) -> None:
        """Test that deep agent specific options are passed through."""
        from langchain_oci.agents.deep_research.agent import create_deep_research_agent

        with patch.dict("os.environ", {"OCI_COMPARTMENT_ID": "test"}):
            with patch("langchain_oci.chat_models.oci_generative_ai.ChatOCIGenAI"):
                with mock_deepagents() as mock_create:
                    mock_subagents = [MagicMock()]
                    mock_store = MagicMock()

                    create_deep_research_agent(
                        tools=[dummy_tool],
                        subagents=mock_subagents,
                        skills=["research", "writing"],
                        memory=["conversations"],
                        store=mock_store,
                        debug=True,
                        name="test_agent",
                    )

                    call_kwargs = mock_create.call_args.kwargs
                    assert call_kwargs["subagents"] == mock_subagents
                    assert call_kwargs["skills"] == ["research", "writing"]
                    assert call_kwargs["memory"] == ["conversations"]
                    assert call_kwargs["store"] == mock_store
                    assert call_kwargs["debug"] is True
                    assert call_kwargs["name"] == "test_agent"

    def test_passes_backend(self) -> None:
        """Test that backend is passed to create_deep_agent, not to ChatOCIGenAI."""
        from langchain_oci.agents.deep_research.agent import create_deep_research_agent

        with patch.dict("os.environ", {"OCI_COMPARTMENT_ID": "test"}):
            with patch(
                "langchain_oci.chat_models.oci_generative_ai.ChatOCIGenAI"
            ) as mock_llm_class:
                with mock_deepagents() as mock_create:
                    mock_backend = MagicMock()

                    create_deep_research_agent(
                        tools=[dummy_tool],
                        backend=mock_backend,
                    )

                    # backend should reach create_deep_agent
                    agent_kwargs = mock_create.call_args.kwargs
                    assert agent_kwargs["backend"] == mock_backend

                    # backend should NOT leak into ChatOCIGenAI model_kwargs
                    llm_kwargs = mock_llm_class.call_args.kwargs
                    model_kw = llm_kwargs.get("model_kwargs") or {}
                    assert "backend" not in model_kw

    def test_passes_response_format(self) -> None:
        """Test that response_format is passed to create_deep_agent."""
        from langchain_oci.agents.deep_research.agent import create_deep_research_agent

        with patch.dict("os.environ", {"OCI_COMPARTMENT_ID": "test"}):
            with patch("langchain_oci.chat_models.oci_generative_ai.ChatOCIGenAI"):
                with mock_deepagents() as mock_create:
                    mock_format = MagicMock()

                    create_deep_research_agent(
                        tools=[dummy_tool],
                        response_format=mock_format,
                    )

                    agent_kwargs = mock_create.call_args.kwargs
                    assert agent_kwargs["response_format"] == mock_format

    def test_passes_context_schema(self) -> None:
        """Test that context_schema is passed to create_deep_agent."""
        from langchain_oci.agents.deep_research.agent import create_deep_research_agent

        with patch.dict("os.environ", {"OCI_COMPARTMENT_ID": "test"}):
            with patch("langchain_oci.chat_models.oci_generative_ai.ChatOCIGenAI"):
                with mock_deepagents() as mock_create:
                    create_deep_research_agent(
                        tools=[dummy_tool],
                        context_schema=dict,
                    )

                    agent_kwargs = mock_create.call_args.kwargs
                    assert agent_kwargs["context_schema"] is dict

    def test_passes_cache(self) -> None:
        """Test that cache is passed to create_deep_agent."""
        from langchain_oci.agents.deep_research.agent import create_deep_research_agent

        with patch.dict("os.environ", {"OCI_COMPARTMENT_ID": "test"}):
            with patch("langchain_oci.chat_models.oci_generative_ai.ChatOCIGenAI"):
                with mock_deepagents() as mock_create:
                    mock_cache = MagicMock()

                    create_deep_research_agent(
                        tools=[dummy_tool],
                        cache=mock_cache,
                    )

                    agent_kwargs = mock_create.call_args.kwargs
                    assert agent_kwargs["cache"] == mock_cache

    def test_passes_interrupt_on(self) -> None:
        """Test that interrupt_on is passed to create_deep_agent."""
        from langchain_oci.agents.deep_research.agent import create_deep_research_agent

        with patch.dict("os.environ", {"OCI_COMPARTMENT_ID": "test"}):
            with patch("langchain_oci.chat_models.oci_generative_ai.ChatOCIGenAI"):
                with mock_deepagents() as mock_create:
                    interrupt_config = {"edit_file": True, "write_file": True}

                    create_deep_research_agent(
                        tools=[dummy_tool],
                        interrupt_on=interrupt_config,
                    )

                    agent_kwargs = mock_create.call_args.kwargs
                    assert agent_kwargs["interrupt_on"] == interrupt_config

    def test_agent_params_not_leaked_to_model(self) -> None:
        """Test that agent-only params don't leak into ChatOCIGenAI."""
        from langchain_oci.agents.deep_research.agent import create_deep_research_agent

        with patch.dict("os.environ", {"OCI_COMPARTMENT_ID": "test"}):
            with patch(
                "langchain_oci.chat_models.oci_generative_ai.ChatOCIGenAI"
            ) as mock_llm_class:
                with mock_deepagents():
                    create_deep_research_agent(
                        tools=[dummy_tool],
                        backend=MagicMock(),
                        cache=MagicMock(),
                        response_format=MagicMock(),
                        context_schema=dict,
                        interrupt_on={"edit_file": True},
                    )

                    llm_kwargs = mock_llm_class.call_args.kwargs
                    model_kw = llm_kwargs.get("model_kwargs") or {}
                    for key in (
                        "backend",
                        "cache",
                        "response_format",
                        "context_schema",
                        "interrupt_on",
                    ):
                        assert key not in model_kw, f"{key} leaked into model_kwargs"
                        assert key not in llm_kwargs, f"{key} leaked into ChatOCIGenAI"

    def test_none_params_omitted_from_deep_agent(self) -> None:
        """Test that None-valued params are omitted (preserving deepagents defaults)."""
        from langchain_oci.agents.deep_research.agent import create_deep_research_agent

        with patch.dict("os.environ", {"OCI_COMPARTMENT_ID": "test"}):
            with patch("langchain_oci.chat_models.oci_generative_ai.ChatOCIGenAI"):
                with mock_deepagents() as mock_create:
                    create_deep_research_agent(tools=[dummy_tool])

                    agent_kwargs = mock_create.call_args.kwargs
                    for key in (
                        "backend",
                        "cache",
                        "response_format",
                        "context_schema",
                        "interrupt_on",
                        "system_prompt",
                        "subagents",
                        "skills",
                        "memory",
                        "middleware",
                        "checkpointer",
                        "store",
                    ):
                        assert key not in agent_kwargs, (
                            f"{key} should be omitted when None"
                        )

    def test_custom_model_id(self) -> None:
        """Test that custom model_id is used."""
        from langchain_oci.agents.deep_research.agent import create_deep_research_agent

        with patch.dict("os.environ", {"OCI_COMPARTMENT_ID": "test"}):
            with patch(
                "langchain_oci.chat_models.oci_generative_ai.ChatOCIGenAI"
            ) as mock_llm_class:
                with mock_deepagents():
                    create_deep_research_agent(
                        tools=[dummy_tool],
                        model_id="google.gemini-2.5-flash",
                    )

                    call_kwargs = mock_llm_class.call_args.kwargs
                    assert call_kwargs["model_id"] == "google.gemini-2.5-flash"

    def test_openai_models_pass_max_tokens_through(self) -> None:
        """OpenAI-compatible models receive ``max_tokens`` from the agent layer.

        The ``max_tokens -> max_completion_tokens`` remap happens at
        request-prep time in ``OpenAIProvider.normalize_params`` (see
        ``chat_models/providers/generic.py``), keeping the agent layer
        model-agnostic.
        """
        from langchain_oci.agents.deep_research.agent import create_deep_research_agent

        with patch.dict("os.environ", {"OCI_COMPARTMENT_ID": "test"}):
            with patch(
                "langchain_oci.chat_models.oci_generative_ai.ChatOCIGenAI"
            ) as mock_llm_class:
                with mock_deepagents():
                    create_deep_research_agent(
                        tools=[dummy_tool],
                        model_id="openai.gpt-5",
                        max_tokens=1024,
                    )

                    call_kwargs = mock_llm_class.call_args.kwargs
                    assert call_kwargs["model_kwargs"]["max_tokens"] == 1024
                    assert "max_completion_tokens" not in call_kwargs["model_kwargs"]

    def test_tools_passed_to_create_deep_agent(self) -> None:
        """Test that tools are passed to create_deep_agent."""
        from langchain_oci.agents.deep_research.agent import create_deep_research_agent

        with patch.dict("os.environ", {"OCI_COMPARTMENT_ID": "test"}):
            with patch("langchain_oci.chat_models.oci_generative_ai.ChatOCIGenAI"):
                with mock_deepagents() as mock_create:
                    create_deep_research_agent(tools=[dummy_tool])

                    call_kwargs = mock_create.call_args.kwargs
                    assert len(call_kwargs["tools"]) == 1

    def test_extra_model_kwargs(self) -> None:
        """Test that extra model kwargs are passed through."""
        from langchain_oci.agents.deep_research.agent import create_deep_research_agent

        with patch.dict("os.environ", {"OCI_COMPARTMENT_ID": "test"}):
            with patch(
                "langchain_oci.chat_models.oci_generative_ai.ChatOCIGenAI"
            ) as mock_llm_class:
                with mock_deepagents():
                    create_deep_research_agent(
                        tools=[dummy_tool],
                        top_p=0.9,
                        frequency_penalty=0.5,
                    )

                    call_kwargs = mock_llm_class.call_args.kwargs
                    assert call_kwargs["model_kwargs"]["top_p"] == 0.9
                    assert call_kwargs["model_kwargs"]["frequency_penalty"] == 0.5


@pytest.mark.requires("oci", "langgraph")
def test_import_from_package() -> None:
    """Test that create_deep_research_agent can be imported from langchain_oci."""
    from langchain_oci import create_deep_research_agent as imported_func

    assert imported_func is not None
    assert callable(imported_func)


@pytest.mark.requires("oci", "langgraph")
def test_import_from_agents_module() -> None:
    """Test that create_deep_research_agent can be imported from agents module."""
    from langchain_oci.agents import create_deep_research_agent as imported_func

    assert imported_func is not None
    assert callable(imported_func)


@pytest.mark.requires("oci", "langgraph")
class TestVectorDataStoreImports:
    """Tests for VectorDataStore and datastore imports."""

    def test_import_from_datastores(self) -> None:
        """Test datastores can be imported from langchain_oci.datastores."""
        from langchain_oci.datastores import (
            ADB,
            OpenSearch,
            VectorDataStore,
            create_datastore_tools,
        )

        assert VectorDataStore is not None
        assert OpenSearch is not None
        assert ADB is not None
        assert create_datastore_tools is not None
        assert callable(create_datastore_tools)

    def test_import_from_top_level(self) -> None:
        """Test datastores can be imported from langchain_oci."""
        from langchain_oci import (
            ADB,
            OpenSearch,
            VectorDataStore,
            create_datastore_tools,
        )

        assert VectorDataStore is not None
        assert OpenSearch is not None
        assert ADB is not None
        assert create_datastore_tools is not None

    def test_import_from_vectorstores_submodule(self) -> None:
        """Test direct import from vectorstores submodule."""
        from langchain_oci.datastores.vectorstores import (
            ADB,
            OpenSearch,
            VectorDataStore,
        )

        assert VectorDataStore is not None
        assert OpenSearch is not None
        assert ADB is not None
