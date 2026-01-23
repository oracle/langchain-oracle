# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Unit tests for create_oci_agent helper function."""

from contextlib import contextmanager
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.tools import tool

from langchain_oci.agents.react import create_oci_agent
from langchain_oci.common.auth import OCIAuthType


@tool
def dummy_tool(x: str) -> str:
    """A dummy tool for testing."""
    return f"Result: {x}"


@contextmanager
def mock_create_agent() -> Generator[MagicMock, None, None]:
    """Mock the agent creation function (works with both new and legacy APIs)."""
    mock = MagicMock(return_value=MagicMock())

    # Try to determine which API is available
    try:
        from langchain.agents import create_agent  # noqa: F401

        has_new_api = True
    except (ImportError, AttributeError):
        has_new_api = False

    if has_new_api:
        with patch("langchain.agents.create_agent", mock):
            yield mock
    else:
        with patch("langgraph.prebuilt.create_react_agent", mock):
            yield mock


@pytest.mark.requires("oci", "langgraph")
class TestCreateOCIReactAgent:
    """Tests for create_oci_agent function."""

    def test_creates_agent_with_minimal_args(self) -> None:
        """Test agent creation with just model_id and tools."""
        with patch.dict("os.environ", {"OCI_COMPARTMENT_ID": "test-compartment"}):
            with patch("langchain_oci.agents.react.ChatOCIGenAI") as mock_llm_class:
                with mock_create_agent() as mock_create:
                    mock_llm_instance = MagicMock()
                    mock_llm_class.return_value = mock_llm_instance

                    agent = create_oci_agent(
                        model_id="meta.llama-4-scout-17b-16e-instruct",
                        tools=[dummy_tool],
                    )

                    # Verify ChatOCIGenAI was created with correct params
                    mock_llm_class.assert_called_once()
                    call_kwargs = mock_llm_class.call_args.kwargs
                    expected_model = "meta.llama-4-scout-17b-16e-instruct"
                    assert call_kwargs["model_id"] == expected_model
                    assert call_kwargs["compartment_id"] == "test-compartment"

                    # Verify agent creation function was called
                    mock_create.assert_called_once()
                    assert agent is not None

    def test_raises_without_compartment_id(self) -> None:
        """Test that error is raised when no compartment_id available."""
        with patch.dict("os.environ", {}, clear=True):
            # Clear any env vars that might be set
            with patch("os.environ.get", return_value=None):
                with pytest.raises(ValueError, match="compartment_id must be provided"):
                    create_oci_agent(
                        model_id="meta.llama-4-scout-17b-16e-instruct",
                        tools=[dummy_tool],
                    )

    def test_passes_system_prompt(self) -> None:
        """Test that system_prompt is passed to the agent creation function."""
        with patch.dict("os.environ", {"OCI_COMPARTMENT_ID": "test"}):
            with patch("langchain_oci.agents.react.ChatOCIGenAI"):
                with mock_create_agent() as mock_create:
                    create_oci_agent(
                        model_id="meta.llama-4-scout-17b-16e-instruct",
                        tools=[dummy_tool],
                        system_prompt="You are helpful.",
                    )

                    call_kwargs = mock_create.call_args.kwargs
                    # Legacy API uses "prompt", new API uses "system_prompt"
                    prompt_value = call_kwargs.get("prompt") or call_kwargs.get(
                        "system_prompt"
                    )
                    assert prompt_value == "You are helpful."

    def test_passes_checkpointer(self) -> None:
        """Test that checkpointer is passed through."""
        with patch.dict("os.environ", {"OCI_COMPARTMENT_ID": "test"}):
            with patch("langchain_oci.agents.react.ChatOCIGenAI"):
                with mock_create_agent() as mock_create:
                    mock_checkpointer = MagicMock()

                    create_oci_agent(
                        model_id="meta.llama-4-scout-17b-16e-instruct",
                        tools=[dummy_tool],
                        checkpointer=mock_checkpointer,
                    )

                    call_kwargs = mock_create.call_args.kwargs
                    assert call_kwargs["checkpointer"] == mock_checkpointer

    def test_passes_oci_specific_options(self) -> None:
        """Test OCI-specific options are passed to ChatOCIGenAI."""
        with patch.dict("os.environ", {"OCI_COMPARTMENT_ID": "test"}):
            with patch("langchain_oci.agents.react.ChatOCIGenAI") as mock_llm_class:
                with mock_create_agent():
                    create_oci_agent(
                        model_id="meta.llama-4-scout-17b-16e-instruct",
                        tools=[dummy_tool],
                        compartment_id="explicit-compartment",
                        auth_profile="CUSTOM",
                        max_sequential_tool_calls=10,
                        temperature=0.5,
                    )

                    call_kwargs = mock_llm_class.call_args.kwargs
                    assert call_kwargs["compartment_id"] == "explicit-compartment"
                    assert call_kwargs["auth_profile"] == "CUSTOM"
                    assert call_kwargs["max_sequential_tool_calls"] == 10
                    assert call_kwargs["model_kwargs"]["temperature"] == 0.5

    def test_auth_type_as_enum(self) -> None:
        """Test that auth_type can be passed as OCIAuthType enum."""
        with patch.dict("os.environ", {"OCI_COMPARTMENT_ID": "test"}):
            with patch("langchain_oci.agents.react.ChatOCIGenAI") as mock_llm_class:
                with mock_create_agent():
                    create_oci_agent(
                        model_id="meta.llama-4-scout-17b-16e-instruct",
                        tools=[dummy_tool],
                        auth_type=OCIAuthType.SECURITY_TOKEN,
                    )

                    call_kwargs = mock_llm_class.call_args.kwargs
                    assert call_kwargs["auth_type"] == "SECURITY_TOKEN"

    def test_auth_type_as_string(self) -> None:
        """Test that auth_type can be passed as string."""
        with patch.dict("os.environ", {"OCI_COMPARTMENT_ID": "test"}):
            with patch("langchain_oci.agents.react.ChatOCIGenAI") as mock_llm_class:
                with mock_create_agent():
                    create_oci_agent(
                        model_id="meta.llama-4-scout-17b-16e-instruct",
                        tools=[dummy_tool],
                        auth_type="INSTANCE_PRINCIPAL",
                    )

                    call_kwargs = mock_llm_class.call_args.kwargs
                    assert call_kwargs["auth_type"] == "INSTANCE_PRINCIPAL"

    def test_service_endpoint_from_region(self) -> None:
        """Test that service_endpoint is constructed from OCI_REGION."""
        with patch.dict(
            "os.environ",
            {"OCI_COMPARTMENT_ID": "test", "OCI_REGION": "us-chicago-1"},
        ):
            with patch("langchain_oci.agents.react.ChatOCIGenAI") as mock_llm_class:
                with mock_create_agent():
                    create_oci_agent(
                        model_id="meta.llama-4-scout-17b-16e-instruct",
                        tools=[dummy_tool],
                    )

                    call_kwargs = mock_llm_class.call_args.kwargs
                    expected = "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
                    assert call_kwargs["service_endpoint"] == expected

    def test_explicit_service_endpoint(self) -> None:
        """Test that explicit service_endpoint takes precedence."""
        with patch.dict(
            "os.environ",
            {"OCI_COMPARTMENT_ID": "test", "OCI_REGION": "us-chicago-1"},
        ):
            with patch("langchain_oci.agents.react.ChatOCIGenAI") as mock_llm_class:
                with mock_create_agent():
                    create_oci_agent(
                        model_id="meta.llama-4-scout-17b-16e-instruct",
                        tools=[dummy_tool],
                        service_endpoint="https://custom.endpoint.com",
                    )

                    call_kwargs = mock_llm_class.call_args.kwargs
                    assert (
                        call_kwargs["service_endpoint"] == "https://custom.endpoint.com"
                    )

    def test_passes_interrupt_options(self) -> None:
        """Test that interrupt_before and interrupt_after are passed."""
        with patch.dict("os.environ", {"OCI_COMPARTMENT_ID": "test"}):
            with patch("langchain_oci.agents.react.ChatOCIGenAI"):
                with mock_create_agent() as mock_create:
                    create_oci_agent(
                        model_id="meta.llama-4-scout-17b-16e-instruct",
                        tools=[dummy_tool],
                        interrupt_before=["tools"],
                        interrupt_after=["agent"],
                    )

                    call_kwargs = mock_create.call_args.kwargs
                    assert call_kwargs["interrupt_before"] == ["tools"]
                    assert call_kwargs["interrupt_after"] == ["agent"]

    def test_passes_debug_and_name(self) -> None:
        """Test that debug and name are passed."""
        with patch.dict("os.environ", {"OCI_COMPARTMENT_ID": "test"}):
            with patch("langchain_oci.agents.react.ChatOCIGenAI"):
                with mock_create_agent() as mock_create:
                    create_oci_agent(
                        model_id="meta.llama-4-scout-17b-16e-instruct",
                        tools=[dummy_tool],
                        debug=True,
                        name="my_custom_agent",
                    )

                    call_kwargs = mock_create.call_args.kwargs
                    assert call_kwargs["debug"] is True
                    assert call_kwargs["name"] == "my_custom_agent"

    def test_passes_max_tokens(self) -> None:
        """Test that max_tokens is passed to model_kwargs."""
        with patch.dict("os.environ", {"OCI_COMPARTMENT_ID": "test"}):
            with patch("langchain_oci.agents.react.ChatOCIGenAI") as mock_llm_class:
                with mock_create_agent():
                    create_oci_agent(
                        model_id="meta.llama-4-scout-17b-16e-instruct",
                        tools=[dummy_tool],
                        max_tokens=1024,
                    )

                    call_kwargs = mock_llm_class.call_args.kwargs
                    assert call_kwargs["model_kwargs"]["max_tokens"] == 1024

    def test_passes_store(self) -> None:
        """Test that store is passed through."""
        with patch.dict("os.environ", {"OCI_COMPARTMENT_ID": "test"}):
            with patch("langchain_oci.agents.react.ChatOCIGenAI"):
                with mock_create_agent() as mock_create:
                    mock_store = MagicMock()

                    create_oci_agent(
                        model_id="meta.llama-4-scout-17b-16e-instruct",
                        tools=[dummy_tool],
                        store=mock_store,
                    )

                    call_kwargs = mock_create.call_args.kwargs
                    assert call_kwargs["store"] == mock_store

    def test_accepts_callable_tools(self) -> None:
        """Test that callable functions can be passed as tools."""
        with patch.dict("os.environ", {"OCI_COMPARTMENT_ID": "test"}):
            with patch("langchain_oci.agents.react.ChatOCIGenAI"):
                with mock_create_agent() as mock_create:

                    def my_func(x: str) -> str:
                        """A function tool."""
                        return x

                    create_oci_agent(
                        model_id="meta.llama-4-scout-17b-16e-instruct",
                        tools=[my_func],
                    )

                    call_kwargs = mock_create.call_args.kwargs
                    # Tools should be converted to a list
                    assert len(call_kwargs["tools"]) == 1

    def test_extra_model_kwargs(self) -> None:
        """Test that extra model kwargs are passed through."""
        with patch.dict("os.environ", {"OCI_COMPARTMENT_ID": "test"}):
            with patch("langchain_oci.agents.react.ChatOCIGenAI") as mock_llm_class:
                with mock_create_agent():
                    create_oci_agent(
                        model_id="meta.llama-4-scout-17b-16e-instruct",
                        tools=[dummy_tool],
                        top_p=0.9,
                        top_k=50,
                    )

                    call_kwargs = mock_llm_class.call_args.kwargs
                    assert call_kwargs["model_kwargs"]["top_p"] == 0.9
                    assert call_kwargs["model_kwargs"]["top_k"] == 50


@pytest.mark.requires("oci", "langgraph")
def test_import_from_package() -> None:
    """Test that create_oci_agent can be imported from langchain_oci."""
    from langchain_oci import create_oci_agent as imported_func

    assert imported_func is not None
    assert callable(imported_func)
