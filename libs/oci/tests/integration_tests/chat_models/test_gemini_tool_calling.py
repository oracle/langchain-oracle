# Copyright (c) 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Integration tests for Gemini tool calling via OCI GenAI.

This module tests the GeminiProvider workaround for tool calling with
Google Gemini models on OCI. The OCI GenAI service's translation layer
doesn't properly handle ToolMessage and tool_calls for Gemini, so the
GeminiProvider converts these to regular messages.

Issue: https://github.com/oracle/langchain-oracle/issues/78

Prerequisites:
    1. OCI Authentication with security token:
       ```bash
       oci session authenticate --profile BOAT-OC1
       ```

    2. Environment variables:
       ```bash
       export OCI_COMPARTMENT_ID="your-compartment-ocid"
       export OCI_CONFIG_PROFILE="BOAT-OC1"
       export OCI_AUTH_TYPE="SECURITY_TOKEN"
       export OCI_GENAI_ENDPOINT="https://inference.generativeai.us-ashburn-1.oci.oraclecloud.com"
       ```

Running tests:
    ```bash
    pytest tests/integration_tests/chat_models/test_gemini_tool_calling.py -v
    ```
"""

import os

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field

from langchain_oci.chat_models import ChatOCIGenAI

# Test configuration from environment
COMPARTMENT_ID = os.getenv(
    "OCI_COMPARTMENT_ID",
    "ocid1.compartment.oc1..aaaaaaaahpwdqajkowoqh4d2q66l4umblh32vbjkh3qnpfdmzcrb7am6jyuq",
)
SERVICE_ENDPOINT = os.getenv(
    "OCI_GENAI_ENDPOINT",
    "https://inference.generativeai.us-ashburn-1.oci.oraclecloud.com",
)
AUTH_TYPE = os.getenv("OCI_AUTH_TYPE", "SECURITY_TOKEN")
AUTH_PROFILE = os.getenv("OCI_CONFIG_PROFILE", "BOAT-OC1")

# Models to test
GEMINI_MODELS = [
    pytest.param("google.gemini-2.5-flash", id="gemini-2.5-flash"),
    pytest.param("google.gemini-2.5-pro", id="gemini-2.5-pro"),
]

COMPARISON_MODELS = [
    pytest.param("xai.grok-3-mini-fast", id="xai-grok-3-mini-fast"),
]


def create_llm(model_id: str, **kwargs) -> ChatOCIGenAI:
    """Create a ChatOCIGenAI instance for testing.

    Args:
        model_id: The model identifier (e.g., "google.gemini-2.5-flash")
        **kwargs: Additional keyword arguments for ChatOCIGenAI

    Returns:
        Configured ChatOCIGenAI instance
    """
    return ChatOCIGenAI(
        auth_type=AUTH_TYPE,
        auth_profile=AUTH_PROFILE,
        model_id=model_id,
        service_endpoint=SERVICE_ENDPOINT,
        compartment_id=COMPARTMENT_ID,
        is_stream=False,
        model_kwargs={
            "temperature": 0.0,
            "max_tokens": 512,
            **kwargs.pop("model_kwargs", {}),
        },
        **kwargs,
    )


class GetWeather(BaseModel):
    """Get the current weather for a city."""

    city: str = Field(description="The city name")
    units: str = Field(
        default="metric", description="Temperature units (metric/imperial)"
    )


class GetStockPrice(BaseModel):
    """Get the current stock price for a ticker symbol."""

    ticker: str = Field(description="The stock ticker symbol (e.g., AAPL)")


@pytest.mark.requires("oci")
@pytest.mark.parametrize("model_id", GEMINI_MODELS)
class TestGeminiToolResultProcessing:
    """Tests for processing tool results with Gemini models."""

    def test_single_tool_result(self, model_id: str) -> None:
        """Test that Gemini can process a single tool result.

        This reproduces the issue from GitHub issue #78 where sending
        a ToolMessage to Gemini resulted in a 500 error after timeout.
        """
        llm = create_llm(model_id)

        messages = [
            HumanMessage(content="What's the weather in Rome?"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_weather_001",
                        "name": "get_weather",
                        "args": {"city": "Rome", "units": "metric"},
                    }
                ],
            ),
            ToolMessage(
                content='{"temperature": 22, "condition": "sunny", "humidity": 45}',
                tool_call_id="call_weather_001",
                name="get_weather",
            ),
        ]

        response = llm.invoke(messages)

        assert response is not None
        assert response.content
        # Verify the response mentions the weather data
        content_lower = response.content.lower()
        assert any(
            term in content_lower for term in ["22", "sunny", "rome", "temperature"]
        ), f"Response should mention weather data: {response.content}"

    def test_tool_result_with_system_message(self, model_id: str) -> None:
        """Test tool result processing with a system message."""
        llm = create_llm(model_id)

        messages = [
            SystemMessage(content="You are a helpful weather assistant. Be concise."),
            HumanMessage(content="What's the weather in Tokyo?"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_tokyo_weather",
                        "name": "get_weather",
                        "args": {"city": "Tokyo", "units": "metric"},
                    }
                ],
            ),
            ToolMessage(
                content='{"temperature": 18, "condition": "cloudy", "humidity": 70}',
                tool_call_id="call_tokyo_weather",
                name="get_weather",
            ),
        ]

        response = llm.invoke(messages)

        assert response is not None
        assert response.content
        content_lower = response.content.lower()
        assert any(term in content_lower for term in ["18", "tokyo", "cloudy"]), (
            f"Response should mention Tokyo weather: {response.content}"
        )

    def test_multiple_sequential_tool_results(self, model_id: str) -> None:
        """Test processing multiple tool results in sequence."""
        llm = create_llm(model_id)

        messages = [
            HumanMessage(content="Compare the weather in Paris and London."),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_paris",
                        "name": "get_weather",
                        "args": {"city": "Paris"},
                    }
                ],
            ),
            ToolMessage(
                content='{"temperature": 15, "condition": "rainy"}',
                tool_call_id="call_paris",
                name="get_weather",
            ),
            AIMessage(
                content="Paris is rainy at 15°C. Let me check London.",
                tool_calls=[
                    {
                        "id": "call_london",
                        "name": "get_weather",
                        "args": {"city": "London"},
                    }
                ],
            ),
            ToolMessage(
                content='{"temperature": 12, "condition": "foggy"}',
                tool_call_id="call_london",
                name="get_weather",
            ),
        ]

        response = llm.invoke(messages)

        assert response is not None
        assert response.content
        # Should mention both cities or their weather
        content_lower = response.content.lower()
        assert any(
            term in content_lower
            for term in ["paris", "london", "15", "12", "rainy", "foggy", "compare"]
        ), f"Response should mention both cities: {response.content}"


@pytest.mark.requires("oci")
@pytest.mark.parametrize("model_id", GEMINI_MODELS)
class TestGeminiToolBinding:
    """Tests for tool binding with Gemini models."""

    def test_bind_single_tool(self, model_id: str) -> None:
        """Test binding a single tool to Gemini."""
        llm = create_llm(model_id)
        llm_with_tools = llm.bind_tools([GetWeather])

        response = llm_with_tools.invoke("What's the weather in Berlin?")

        assert response is not None
        # Model should either call the tool or respond
        assert response.content or response.tool_calls

    def test_bind_multiple_tools(self, model_id: str) -> None:
        """Test binding multiple tools to Gemini."""
        llm = create_llm(model_id)
        llm_with_tools = llm.bind_tools([GetWeather, GetStockPrice])

        response = llm_with_tools.invoke("What's the weather in NYC?")

        assert response is not None
        assert response.content or response.tool_calls


@pytest.mark.requires("oci")
@pytest.mark.parametrize("model_id", GEMINI_MODELS + COMPARISON_MODELS)
class TestCrossModelCompatibility:
    """Tests to verify Gemini works the same as other models."""

    def test_tool_result_processing_parity(self, model_id: str) -> None:
        """Verify all models can process tool results similarly."""
        llm = create_llm(model_id)

        messages = [
            HumanMessage(content="What's the weather in Sydney?"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_sydney",
                        "name": "get_weather",
                        "args": {"city": "Sydney", "units": "metric"},
                    }
                ],
            ),
            ToolMessage(
                content='{"temperature": 25, "condition": "clear"}',
                tool_call_id="call_sydney",
                name="get_weather",
            ),
        ]

        response = llm.invoke(messages)

        assert response is not None
        assert response.content
        # All models should provide a coherent response about the weather
        assert len(response.content) > 10, "Response should be substantive"


@pytest.mark.requires("oci")
@pytest.mark.parametrize("model_id", GEMINI_MODELS)
class TestGeminiProviderSelection:
    """Tests to verify GeminiProvider is correctly selected."""

    def test_provider_is_gemini_provider(self, model_id: str) -> None:
        """Verify that Gemini models use GeminiProvider."""
        llm = create_llm(model_id)

        provider = llm._provider
        provider_class_name = type(provider).__name__

        assert provider_class_name == "GeminiProvider", (
            f"Expected GeminiProvider for {model_id}, got {provider_class_name}"
        )

    def test_simple_chat_still_works(self, model_id: str) -> None:
        """Verify simple chat (no tools) still works with GeminiProvider."""
        llm = create_llm(model_id)

        response = llm.invoke("What is 2 + 2? Reply with just the number.")

        assert response is not None
        assert response.content
        # Model should return "4" or mention "four"
        assert "4" in response.content or "four" in response.content.lower()


@pytest.mark.requires("oci")
class TestGeminiEdgeCases:
    """Edge case tests for Gemini tool calling."""

    @pytest.mark.parametrize("model_id", GEMINI_MODELS)
    def test_empty_tool_result(self, model_id: str) -> None:
        """Test handling of empty tool result."""
        llm = create_llm(model_id)

        messages = [
            HumanMessage(content="Check my calendar."),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_calendar",
                        "name": "check_calendar",
                        "args": {"date": "today"},
                    }
                ],
            ),
            ToolMessage(
                content="{}",  # Empty result
                tool_call_id="call_calendar",
                name="check_calendar",
            ),
        ]

        response = llm.invoke(messages)

        assert response is not None
        # Should handle empty result gracefully
        assert response.content is not None

    @pytest.mark.parametrize("model_id", GEMINI_MODELS)
    def test_json_tool_result(self, model_id: str) -> None:
        """Test handling of complex JSON tool result."""
        llm = create_llm(model_id)

        complex_result = {
            "data": {
                "temperature": 20,
                "conditions": ["partly cloudy", "windy"],
                "forecast": [
                    {"day": "Monday", "high": 22, "low": 15},
                    {"day": "Tuesday", "high": 24, "low": 16},
                ],
            },
            "metadata": {"source": "weather_api", "timestamp": "2025-01-01T12:00:00Z"},
        }

        messages = [
            HumanMessage(content="Give me a detailed weather forecast."),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_forecast",
                        "name": "get_forecast",
                        "args": {"city": "Madrid", "days": 3},
                    }
                ],
            ),
            ToolMessage(
                content=str(complex_result),
                tool_call_id="call_forecast",
                name="get_forecast",
            ),
        ]

        response = llm.invoke(messages)

        assert response is not None
        assert response.content
        # Should reference some of the forecast data
        content_lower = response.content.lower()
        assert any(
            term in content_lower
            for term in ["monday", "tuesday", "22", "24", "forecast", "weather"]
        )

    @pytest.mark.parametrize("model_id", GEMINI_MODELS)
    def test_tool_result_with_special_characters(self, model_id: str) -> None:
        """Test tool result containing special characters."""
        llm = create_llm(model_id)

        messages = [
            HumanMessage(content="Get the café menu."),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_menu",
                        "name": "get_menu",
                        "args": {"restaurant": "café"},
                    }
                ],
            ),
            ToolMessage(
                content=(
                    '{"items": ["Café Latte €4.50", '
                    '"Croissant €2.00", "Crème brûlée €6.50"]}'
                ),
                tool_call_id="call_menu",
                name="get_menu",
            ),
        ]

        response = llm.invoke(messages)

        assert response is not None
        assert response.content
        # Should handle special characters (accents, currency symbols)
        assert len(response.content) > 5


@pytest.mark.requires("oci")
@pytest.mark.parametrize("model_id", GEMINI_MODELS)
class TestGeminiParallelToolCalling:
    """Tests for parallel tool calling with Gemini models.

    These tests verify that Gemini can handle multiple tool calls
    in a single turn, which is common in agent workflows.
    """

    def test_parallel_tool_results_same_function(self, model_id: str) -> None:
        """Test processing multiple parallel calls to the same function.

        Scenario: User asks about weather in multiple cities simultaneously.
        """
        llm = create_llm(model_id)

        messages = [
            HumanMessage(content="What's the weather in New York, London, and Tokyo?"),
            AIMessage(
                content="I'll check the weather in all three cities.",
                tool_calls=[
                    {
                        "id": "call_ny",
                        "name": "get_weather",
                        "args": {"city": "New York", "units": "metric"},
                    },
                    {
                        "id": "call_london",
                        "name": "get_weather",
                        "args": {"city": "London", "units": "metric"},
                    },
                    {
                        "id": "call_tokyo",
                        "name": "get_weather",
                        "args": {"city": "Tokyo", "units": "metric"},
                    },
                ],
            ),
            ToolMessage(
                content='{"temperature": 8, "condition": "cloudy", "humidity": 65}',
                tool_call_id="call_ny",
                name="get_weather",
            ),
            ToolMessage(
                content='{"temperature": 12, "condition": "rainy", "humidity": 80}',
                tool_call_id="call_london",
                name="get_weather",
            ),
            ToolMessage(
                content='{"temperature": 18, "condition": "sunny", "humidity": 50}',
                tool_call_id="call_tokyo",
                name="get_weather",
            ),
        ]

        response = llm.invoke(messages)

        assert response is not None
        assert response.content
        content_lower = response.content.lower()
        # Should mention at least some of the cities or their weather data
        mentioned_cities = sum(
            1 for city in ["new york", "london", "tokyo", "ny"] if city in content_lower
        )
        mentioned_temps = sum(1 for temp in ["8", "12", "18"] if temp in content_lower)
        assert mentioned_cities >= 1 or mentioned_temps >= 1, (
            f"Response should mention cities or temperatures: {response.content}"
        )

    def test_parallel_tool_results_different_functions(self, model_id: str) -> None:
        """Test processing parallel calls to different functions.

        Scenario: User asks for both weather and stock price simultaneously.
        """
        llm = create_llm(model_id)

        messages = [
            HumanMessage(
                content="What's the weather in SF and the current price of AAPL stock?"
            ),
            AIMessage(
                content="I'll check both for you.",
                tool_calls=[
                    {
                        "id": "call_weather",
                        "name": "get_weather",
                        "args": {"city": "San Francisco", "units": "imperial"},
                    },
                    {
                        "id": "call_stock",
                        "name": "get_stock_price",
                        "args": {"ticker": "AAPL"},
                    },
                ],
            ),
            ToolMessage(
                content='{"temperature": 65, "condition": "foggy", "humidity": 75}',
                tool_call_id="call_weather",
                name="get_weather",
            ),
            ToolMessage(
                content='{"price": 195.50, "change": "+2.30", "volume": "45M"}',
                tool_call_id="call_stock",
                name="get_stock_price",
            ),
        ]

        response = llm.invoke(messages)

        assert response is not None
        assert response.content
        content_lower = response.content.lower()
        # Should reference both weather and stock data
        has_weather_ref = any(
            term in content_lower
            for term in ["65", "foggy", "san francisco", "sf", "weather"]
        )
        has_stock_ref = any(
            term in content_lower for term in ["195", "aapl", "apple", "stock", "price"]
        )
        assert has_weather_ref or has_stock_ref, (
            f"Response should mention weather or stock: {response.content}"
        )

    def test_parallel_five_tool_calls(self, model_id: str) -> None:
        """Test processing five parallel tool calls.

        Scenario: Complex query requiring multiple data points.
        """
        llm = create_llm(model_id)

        messages = [
            HumanMessage(content="Give me a global weather summary for major cities."),
            AIMessage(
                content="Checking weather across major cities worldwide.",
                tool_calls=[
                    {
                        "id": "call_1",
                        "name": "get_weather",
                        "args": {"city": "New York"},
                    },
                    {
                        "id": "call_2",
                        "name": "get_weather",
                        "args": {"city": "London"},
                    },
                    {
                        "id": "call_3",
                        "name": "get_weather",
                        "args": {"city": "Tokyo"},
                    },
                    {
                        "id": "call_4",
                        "name": "get_weather",
                        "args": {"city": "Sydney"},
                    },
                    {
                        "id": "call_5",
                        "name": "get_weather",
                        "args": {"city": "Dubai"},
                    },
                ],
            ),
            ToolMessage(
                content='{"temperature": 5, "condition": "snowy"}',
                tool_call_id="call_1",
                name="get_weather",
            ),
            ToolMessage(
                content='{"temperature": 10, "condition": "rainy"}',
                tool_call_id="call_2",
                name="get_weather",
            ),
            ToolMessage(
                content='{"temperature": 15, "condition": "cloudy"}',
                tool_call_id="call_3",
                name="get_weather",
            ),
            ToolMessage(
                content='{"temperature": 28, "condition": "sunny"}',
                tool_call_id="call_4",
                name="get_weather",
            ),
            ToolMessage(
                content='{"temperature": 35, "condition": "hot"}',
                tool_call_id="call_5",
                name="get_weather",
            ),
        ]

        response = llm.invoke(messages)

        assert response is not None
        assert response.content
        # Response should be substantive given 5 data points
        assert len(response.content) > 20, (
            f"Response should summarize multiple cities: {response.content}"
        )

    def test_parallel_tool_calls_with_errors(self, model_id: str) -> None:
        """Test handling parallel tool calls where some fail.

        Scenario: Some API calls succeed, others return errors.
        """
        llm = create_llm(model_id)

        messages = [
            HumanMessage(content="Get weather for Paris and stock for INVALID."),
            AIMessage(
                content="Checking both.",
                tool_calls=[
                    {
                        "id": "call_paris",
                        "name": "get_weather",
                        "args": {"city": "Paris"},
                    },
                    {
                        "id": "call_invalid",
                        "name": "get_stock_price",
                        "args": {"ticker": "INVALID"},
                    },
                ],
            ),
            ToolMessage(
                content='{"temperature": 14, "condition": "partly cloudy"}',
                tool_call_id="call_paris",
                name="get_weather",
            ),
            ToolMessage(
                content='{"error": "Ticker not found", "code": "NOT_FOUND"}',
                tool_call_id="call_invalid",
                name="get_stock_price",
            ),
        ]

        response = llm.invoke(messages)

        assert response is not None
        assert response.content
        # Should handle the mixed success/error gracefully
        content_lower = response.content.lower()
        # Should at least mention the successful weather data
        assert any(
            term in content_lower
            for term in ["paris", "14", "cloudy", "weather", "error", "not found"]
        ), f"Response should handle mixed results: {response.content}"


@pytest.mark.requires("oci")
@pytest.mark.parametrize("model_id", GEMINI_MODELS + COMPARISON_MODELS)
class TestParallelToolCallingCrossModel:
    """Cross-model tests for parallel tool calling.

    Verify that Gemini handles parallel tool calls the same way as other models.
    """

    def test_parallel_calls_parity(self, model_id: str) -> None:
        """Verify all models handle parallel tool calls similarly."""
        llm = create_llm(model_id)

        messages = [
            HumanMessage(content="Weather in Berlin and price of GOOGL?"),
            AIMessage(
                content="Checking both.",
                tool_calls=[
                    {
                        "id": "call_berlin",
                        "name": "get_weather",
                        "args": {"city": "Berlin"},
                    },
                    {
                        "id": "call_googl",
                        "name": "get_stock_price",
                        "args": {"ticker": "GOOGL"},
                    },
                ],
            ),
            ToolMessage(
                content='{"temperature": 7, "condition": "overcast"}',
                tool_call_id="call_berlin",
                name="get_weather",
            ),
            ToolMessage(
                content='{"price": 175.25, "change": "-1.50"}',
                tool_call_id="call_googl",
                name="get_stock_price",
            ),
        ]

        response = llm.invoke(messages)

        assert response is not None
        assert response.content
        assert len(response.content) > 10, (
            f"All models should provide substantive response: {response.content}"
        )
