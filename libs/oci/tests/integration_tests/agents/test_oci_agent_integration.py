# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Integration tests for OCIGenAIAgent.

## Overview

These tests demonstrate the full capabilities of OCIGenAIAgent:
- Immutable state management with full audit trail
- Typed event streaming for real-time progress tracking
- Reflexion (confidence tracking + loop detection)
- 5 termination conditions
- LangChain Runnable interface for LCEL composability
- LangGraph node compatibility

## Prerequisites

1. **OCI Authentication**: Set up OCI authentication with security token:
   ```bash
   oci session authenticate
   ```

2. **Environment Variables**: Export the following:
   ```bash
   export OCI_REGION="us-chicago-1"  # or your region
   export OCI_COMPARTMENT_ID="ocid1.compartment.oc1..your-compartment-id"
   ```

3. **OCI Config**: Ensure `~/.oci/config` exists with DEFAULT profile

## Running the Tests

Run all OCIGenAIAgent integration tests:
```bash
cd libs/oci
python -m pytest tests/integration_tests/agents/test_oci_agent_integration.py -v
```

Run specific test:
```bash
pytest tests/integration_tests/agents/test_oci_agent_integration.py \
  ::TestOCIGenAIAgentIntegration::test_basic_invoke -v
```
"""

import os
import time

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from langchain_oci import (
    AgentResult,
    OCIGenAIAgent,
    ReasoningStep,
    ReflectEvent,
    TerminateEvent,
    ThinkEvent,
    ToolCompleteEvent,
    ToolExecution,
    ToolStartEvent,
)
from langchain_oci.agents.oci_agent.termination import TerminationReason

# =============================================================================
# Test Tools
# =============================================================================


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: Name of the city to get weather for.

    Returns:
        Weather description for the city.
    """
    weather_data = {
        "chicago": "72F and sunny in Chicago",
        "new york": "65F and cloudy in New York",
        "san francisco": "58F and foggy in San Francisco",
        "paris": "68F and partly cloudy in Paris",
        "london": "55F and rainy in London",
    }
    return weather_data.get(city.lower(), f"Weather data not available for {city}")


@tool
def calculate(expression: str) -> str:
    """Evaluate a simple math expression.

    Args:
        expression: A mathematical expression (basic arithmetic only).

    Returns:
        The result of the calculation.
    """
    allowed_chars = set("0123456789+-*/.(). ")
    if not all(c in allowed_chars for c in expression):
        return "Error: Only basic arithmetic operations are supported"
    try:
        result = eval(expression, {"__builtins__": {}}, {})  # noqa: S307
        return f"The result is {result}"
    except Exception as e:
        return f"Error calculating: {e}"


@tool
def search_database(query: str) -> str:
    """Search a database for information.

    Args:
        query: Search query string.

    Returns:
        Search results.
    """
    # Simulated database
    data = {
        "capital france": "Paris is the capital of France.",
        "capital japan": "Tokyo is the capital of Japan.",
        "python creator": "Python was created by Guido van Rossum.",
        "langchain": "LangChain is a framework for developing LLM applications.",
    }
    query_lower = query.lower()
    for key, value in data.items():
        if key in query_lower or all(word in query_lower for word in key.split()):
            return value
    return f"No results found for: {query}"


@tool
def submit_answer(answer: str) -> str:
    """Submit the final answer.

    Call this when you have found the answer and are ready to submit.

    Args:
        answer: The final answer to submit.

    Returns:
        Confirmation of submission.
    """
    return f"SUBMITTED: {answer}"


@tool
def slow_tool(seconds: float) -> str:
    """A slow tool for testing timing.

    Args:
        seconds: Number of seconds to wait.

    Returns:
        Completion message with timing.
    """
    time.sleep(min(seconds, 2.0))  # Cap at 2 seconds for safety
    return f"Completed after {seconds} seconds"


@tool
def failing_tool(should_fail: bool = True) -> str:
    """A tool that can fail on demand.

    Args:
        should_fail: Whether the tool should raise an error.

    Returns:
        Success message if not failing.

    Raises:
        RuntimeError: If should_fail is True.
    """
    if should_fail:
        raise RuntimeError("This tool intentionally failed!")
    return "Tool executed successfully"


def skip_if_no_oci_credentials() -> bool:
    """Check if OCI credentials are available."""
    compartment_id = os.environ.get("OCI_COMPARTMENT_ID")
    return compartment_id is None


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.requires("oci")
@pytest.mark.skipif(
    skip_if_no_oci_credentials(),
    reason="OCI credentials not available (OCI_COMPARTMENT_ID not set)",
)
class TestOCIGenAIAgentIntegration:
    """Integration tests for OCIGenAIAgent.

    These tests require OCI credentials and demonstrate the full
    capabilities of the OCIGenAIAgent agentic loop.
    """

    @pytest.fixture
    def compartment_id(self) -> str:
        """Get compartment ID from environment."""
        return os.environ.get("OCI_COMPARTMENT_ID", "")

    @pytest.fixture
    def service_endpoint(self) -> str:
        """Get service endpoint from environment."""
        region = os.environ.get("OCI_REGION", "us-chicago-1")
        return f"https://inference.generativeai.{region}.oci.oraclecloud.com"

    @pytest.fixture
    def auth_type(self) -> str:
        """Get auth type from environment."""
        return os.environ.get("OCI_AUTH_TYPE", "SECURITY_TOKEN")

    @pytest.fixture
    def auth_profile(self) -> str:
        """Get auth profile from environment."""
        return os.environ.get("OCI_AUTH_PROFILE", "DEFAULT")

    @pytest.fixture
    def model_id(self) -> str:
        """Get model ID from environment."""
        return os.environ.get("OCI_MODEL_ID", "meta.llama-4-scout-17b-16e-instruct")

    @pytest.fixture
    def basic_agent(
        self,
        model_id: str,
        compartment_id: str,
        service_endpoint: str,
        auth_type: str,
        auth_profile: str,
    ) -> OCIGenAIAgent:
        """Create a basic agent for testing."""
        return OCIGenAIAgent(
            model_id=model_id,
            tools=[get_weather, calculate, search_database],
            compartment_id=compartment_id,
            service_endpoint=service_endpoint,
            auth_type=auth_type,
            auth_profile=auth_profile,
            max_iterations=5,
            enable_reflexion=True,
        )

    # =========================================================================
    # Basic Functionality Tests
    # =========================================================================

    def test_basic_invoke(self, basic_agent: OCIGenAIAgent):
        """Test basic agent invocation with a simple query.

        Demonstrates:
        - Simple invoke() call
        - AgentResult structure
        - Natural completion (no_tools termination)
        """
        result = basic_agent.invoke("What is 2 + 2?")

        assert isinstance(result, AgentResult)
        assert result.final_answer, "Should have a final answer"
        assert result.termination_reason in [
            TerminationReason.NO_TOOLS,
            TerminationReason.CONFIDENCE_MET,
            TerminationReason.MAX_ITERATIONS,
        ]
        assert result.total_iterations >= 1

    def test_single_tool_invocation(self, basic_agent: OCIGenAIAgent):
        """Test agent with a single tool call.

        Demonstrates:
        - Tool execution
        - Tool results in reasoning steps
        - Audit trail
        """
        result = basic_agent.invoke("What's the weather in Chicago?")

        assert result.total_tool_calls >= 1, "Should have called at least one tool"

        # Check reasoning steps contain tool executions
        tool_calls_found = sum(
            len(step.tool_executions) for step in result.reasoning_steps
        )
        assert tool_calls_found >= 1, "Tool calls should be recorded in reasoning steps"

    def test_multi_tool_invocation(self, basic_agent: OCIGenAIAgent):
        """Test agent with multiple tool calls.

        Demonstrates:
        - Multiple tool execution in sequence
        - Complex reasoning
        """
        result = basic_agent.invoke(
            "What's the weather in Chicago and also calculate 25 * 4?"
        )

        assert result.total_tool_calls >= 2, "Should call multiple tools"
        assert "100" in result.final_answer or "weather" in result.final_answer.lower()

    def test_dict_input(self, basic_agent: OCIGenAIAgent):
        """Test agent with dictionary input.

        Demonstrates:
        - Multiple input formats
        - LangGraph compatibility
        """
        result = basic_agent.invoke({"input": "What is the capital of France?"})

        assert isinstance(result, AgentResult)
        assert result.final_answer

    def test_messages_input(self, basic_agent: OCIGenAIAgent):
        """Test agent with messages input.

        Demonstrates:
        - LangChain message compatibility
        - Conversation history support
        """
        result = basic_agent.invoke(
            {
                "messages": [
                    HumanMessage(content="What's 15 + 27?"),
                ]
            }
        )

        assert isinstance(result, AgentResult)
        assert "42" in result.final_answer

    # =========================================================================
    # Streaming Tests
    # =========================================================================

    def test_stream_events(self, basic_agent: OCIGenAIAgent):
        """Test streaming typed events.

        Demonstrates:
        - Typed event streaming
        - All 5 event types
        - Real-time progress tracking
        """
        events = list(basic_agent.stream("What's the weather in Paris?"))

        # Should have at least think and terminate events
        event_types = {type(e).__name__ for e in events}
        assert "ThinkEvent" in event_types, "Should emit ThinkEvent"
        assert "TerminateEvent" in event_types, "Should emit TerminateEvent"

        # Check event structure
        for event in events:
            if isinstance(event, ThinkEvent):
                assert hasattr(event, "iteration")
                assert hasattr(event, "thought")
                assert hasattr(event, "tool_calls_planned")
            elif isinstance(event, ToolStartEvent):
                assert hasattr(event, "tool_name")
                assert hasattr(event, "arguments")
            elif isinstance(event, ToolCompleteEvent):
                assert hasattr(event, "tool_name")
                assert hasattr(event, "result")
                assert hasattr(event, "duration_ms")
            elif isinstance(event, ReflectEvent):
                assert hasattr(event, "confidence")
                assert hasattr(event, "assessment")
            elif isinstance(event, TerminateEvent):
                assert hasattr(event, "reason")
                assert hasattr(event, "final_answer")

    def test_stream_with_tool_events(self, basic_agent: OCIGenAIAgent):
        """Test streaming includes tool start/complete events.

        Demonstrates:
        - ToolStartEvent before execution
        - ToolCompleteEvent after execution
        - Timing information
        """
        events = list(basic_agent.stream("Calculate 10 * 5"))

        tool_starts = [e for e in events if isinstance(e, ToolStartEvent)]
        tool_completes = [e for e in events if isinstance(e, ToolCompleteEvent)]

        # If tools were called, should have matching start/complete
        if tool_starts:
            assert len(tool_starts) == len(tool_completes)
            for complete in tool_completes:
                assert complete.duration_ms >= 0, "Should have timing info"

    def test_stream_reflect_events(
        self,
        model_id: str,
        compartment_id: str,
        service_endpoint: str,
        auth_type: str,
        auth_profile: str,
    ):
        """Test streaming includes reflexion events.

        Demonstrates:
        - ReflectEvent after tool execution
        - Confidence tracking
        - Progress assessment
        """
        agent = OCIGenAIAgent(
            model_id=model_id,
            tools=[get_weather, calculate],
            compartment_id=compartment_id,
            service_endpoint=service_endpoint,
            auth_type=auth_type,
            auth_profile=auth_profile,
            enable_reflexion=True,
            max_iterations=5,
        )

        events = list(agent.stream("Get weather in Chicago and calculate 5 * 5"))

        reflect_events = [e for e in events if isinstance(e, ReflectEvent)]

        # If tools were called and reflexion is enabled, should have reflect events
        tool_events = [e for e in events if isinstance(e, ToolCompleteEvent)]
        if tool_events:
            assert len(reflect_events) >= 1, "Should have ReflectEvent after tools"
            for reflect in reflect_events:
                assert 0 <= reflect.confidence <= 1
                assert reflect.assessment in [
                    "on_track",
                    "stuck",
                    "new_findings",
                    "loop_detected",
                ]

    # =========================================================================
    # Termination Condition Tests
    # =========================================================================

    def test_no_tools_termination(
        self,
        model_id: str,
        compartment_id: str,
        service_endpoint: str,
        auth_type: str,
        auth_profile: str,
    ):
        """Test natural completion without tools.

        Demonstrates:
        - no_tools termination condition
        - Agent knows when no tools needed
        """
        agent = OCIGenAIAgent(
            model_id=model_id,
            tools=[get_weather],
            compartment_id=compartment_id,
            service_endpoint=service_endpoint,
            auth_type=auth_type,
            auth_profile=auth_profile,
            max_iterations=5,
        )

        # Simple question that doesn't need tools
        result = agent.invoke("What is 2 + 2?")

        # Could be no_tools or max_iterations depending on model behavior
        assert result.termination_reason in [
            TerminationReason.NO_TOOLS,
            TerminationReason.MAX_ITERATIONS,
            TerminationReason.CONFIDENCE_MET,
        ]

    def test_max_iterations_termination(
        self,
        model_id: str,
        compartment_id: str,
        service_endpoint: str,
        auth_type: str,
        auth_profile: str,
    ):
        """Test max iterations termination.

        Demonstrates:
        - max_iterations termination condition
        - Agent stops at limit
        """
        agent = OCIGenAIAgent(
            model_id=model_id,
            tools=[get_weather, calculate, search_database],
            compartment_id=compartment_id,
            service_endpoint=service_endpoint,
            auth_type=auth_type,
            auth_profile=auth_profile,
            max_iterations=2,  # Low limit
        )

        result = agent.invoke(
            "Get weather for Chicago, New York, San Francisco, Paris, and London"
        )

        assert result.total_iterations <= 2, "Should respect max_iterations"
        if result.total_iterations == 2:
            assert result.termination_reason == TerminationReason.MAX_ITERATIONS

    def test_terminal_tool_termination(
        self,
        model_id: str,
        compartment_id: str,
        service_endpoint: str,
        auth_type: str,
        auth_profile: str,
    ):
        """Test terminal tool termination.

        Demonstrates:
        - terminal_tool termination condition
        - Explicit completion signal
        """
        agent = OCIGenAIAgent(
            model_id=model_id,
            tools=[search_database, submit_answer],
            compartment_id=compartment_id,
            service_endpoint=service_endpoint,
            auth_type=auth_type,
            auth_profile=auth_profile,
            terminal_tools=["submit_answer"],
            max_iterations=5,
        )

        result = agent.invoke(
            "Find the capital of France and submit the answer using submit_answer."
        )

        # Should terminate due to terminal tool
        if "SUBMITTED" in result.final_answer:
            assert result.termination_reason == TerminationReason.TERMINAL_TOOL

    def test_confidence_threshold_termination(
        self,
        model_id: str,
        compartment_id: str,
        service_endpoint: str,
        auth_type: str,
        auth_profile: str,
    ):
        """Test confidence threshold termination.

        Demonstrates:
        - confidence_met termination condition
        - Reflexion builds confidence
        """
        agent = OCIGenAIAgent(
            model_id=model_id,
            tools=[get_weather],
            compartment_id=compartment_id,
            service_endpoint=service_endpoint,
            auth_type=auth_type,
            auth_profile=auth_profile,
            enable_reflexion=True,
            confidence_threshold=0.3,  # Low threshold for easier testing
            max_iterations=10,
        )

        result = agent.invoke("What's the weather in Chicago?")

        # With successful tool calls, confidence should build
        if result.termination_reason == TerminationReason.CONFIDENCE_MET:
            assert result.confidence >= 0.3

    # =========================================================================
    # Reflexion Tests
    # =========================================================================

    def test_reflexion_confidence_tracking(
        self,
        model_id: str,
        compartment_id: str,
        service_endpoint: str,
        auth_type: str,
        auth_profile: str,
    ):
        """Test confidence tracking through reflexion.

        Demonstrates:
        - Confidence increases with successful tools
        - Confidence recorded in reasoning steps
        """
        agent = OCIGenAIAgent(
            model_id=model_id,
            tools=[get_weather, calculate],
            compartment_id=compartment_id,
            service_endpoint=service_endpoint,
            auth_type=auth_type,
            auth_profile=auth_profile,
            enable_reflexion=True,
            max_iterations=5,
        )

        result = agent.invoke("Get weather for Chicago and calculate 10 + 20")

        if result.reasoning_steps:
            # Confidence should be tracked in each step
            for step in result.reasoning_steps:
                assert 0 <= step.confidence <= 1
                assert step.assessment in [
                    "on_track",
                    "stuck",
                    "new_findings",
                    "loop_detected",
                ]

    def test_reflexion_disabled(
        self,
        model_id: str,
        compartment_id: str,
        service_endpoint: str,
        auth_type: str,
        auth_profile: str,
    ):
        """Test agent without reflexion.

        Demonstrates:
        - enable_reflexion=False disables self-evaluation
        - Simpler execution flow
        """
        agent = OCIGenAIAgent(
            model_id=model_id,
            tools=[get_weather],
            compartment_id=compartment_id,
            service_endpoint=service_endpoint,
            auth_type=auth_type,
            auth_profile=auth_profile,
            enable_reflexion=False,
            max_iterations=5,
        )

        events = list(agent.stream("What's the weather in Chicago?"))

        # Should not have ReflectEvent when disabled
        reflect_events = [e for e in events if isinstance(e, ReflectEvent)]
        assert len(reflect_events) == 0, "Should not emit ReflectEvent when disabled"

    # =========================================================================
    # State and Audit Trail Tests
    # =========================================================================

    def test_reasoning_steps_audit_trail(self, basic_agent: OCIGenAIAgent):
        """Test reasoning steps provide full audit trail.

        Demonstrates:
        - ReasoningStep captures each iteration
        - Tool executions recorded
        - Thoughts preserved
        """
        result = basic_agent.invoke("What's the weather in Chicago?")

        assert isinstance(result.reasoning_steps, list)

        for step in result.reasoning_steps:
            assert isinstance(step, ReasoningStep)
            assert isinstance(step.iteration, int)
            assert isinstance(step.tool_executions, tuple)

            # Each tool execution should have details
            for execution in step.tool_executions:
                assert isinstance(execution, ToolExecution)
                assert execution.tool_name
                assert execution.tool_call_id
                assert isinstance(execution.success, bool)
                assert execution.duration_ms >= 0

    def test_message_history(self, basic_agent: OCIGenAIAgent):
        """Test message history is preserved.

        Demonstrates:
        - Full conversation in result.messages
        - Includes tool messages
        """
        result = basic_agent.invoke("What's the weather in New York?")

        assert len(result.messages) >= 1, "Should have messages"

        # Check message types
        message_types = {type(m).__name__ for m in result.messages}
        assert "HumanMessage" in message_types or "AIMessage" in message_types

    # =========================================================================
    # Error Handling Tests
    # =========================================================================

    def test_tool_error_handling(
        self,
        model_id: str,
        compartment_id: str,
        service_endpoint: str,
        auth_type: str,
        auth_profile: str,
    ):
        """Test agent handles tool errors gracefully.

        Demonstrates:
        - Tool errors don't crash agent
        - Error recorded in execution
        - Agent can continue or report
        """
        agent = OCIGenAIAgent(
            model_id=model_id,
            tools=[failing_tool, get_weather],
            compartment_id=compartment_id,
            service_endpoint=service_endpoint,
            auth_type=auth_type,
            auth_profile=auth_profile,
            max_iterations=3,
        )

        # This may or may not call the failing tool
        result = agent.invoke("Try to use the failing_tool and then get weather")

        # Agent should complete without crashing
        assert isinstance(result, AgentResult)
        assert result.termination_reason

    # =========================================================================
    # LCEL Chain Tests
    # =========================================================================

    def test_lcel_chain_composition(self, basic_agent: OCIGenAIAgent):
        """Test OCIGenAIAgent in LCEL chain.

        Demonstrates:
        - Runnable interface
        - Chain composition
        - Output transformation
        """
        from langchain_core.runnables import RunnablePassthrough

        # Create a chain that extracts just the answer
        chain = RunnablePassthrough() | basic_agent | (lambda r: r.final_answer)

        answer = chain.invoke("What is 5 * 5?")

        assert isinstance(answer, str)
        # The answer should contain 25 or be a reasonable response
        assert answer, "Should have an answer"

    # =========================================================================
    # Custom Configuration Tests
    # =========================================================================

    def test_custom_system_prompt(
        self,
        model_id: str,
        compartment_id: str,
        service_endpoint: str,
        auth_type: str,
        auth_profile: str,
    ):
        """Test agent with custom system prompt.

        Demonstrates:
        - system_prompt customization
        - Behavioral modification
        """
        agent = OCIGenAIAgent(
            model_id=model_id,
            tools=[calculate],
            compartment_id=compartment_id,
            service_endpoint=service_endpoint,
            auth_type=auth_type,
            auth_profile=auth_profile,
            system_prompt="You are a math assistant. Always show your work.",
            max_iterations=3,
        )

        result = agent.invoke("What is 15 * 3?")

        assert isinstance(result, AgentResult)
        # The system prompt should influence the response style

    def test_agent_repr(self, basic_agent: OCIGenAIAgent):
        """Test agent string representation.

        Demonstrates:
        - Useful repr for debugging
        - Shows configuration
        """
        repr_str = repr(basic_agent)

        assert "OCIGenAIAgent" in repr_str
        assert "model_id" in repr_str
        assert "tools" in repr_str


# =============================================================================
# LangGraph Integration Tests (optional - requires langgraph)
# =============================================================================


@pytest.mark.requires("oci", "langgraph")
@pytest.mark.skipif(
    skip_if_no_oci_credentials(),
    reason="OCI credentials not available (OCI_COMPARTMENT_ID not set)",
)
class TestOCIGenAIAgentLangGraphIntegration:
    """Integration tests for OCIGenAIAgent with LangGraph.

    These tests demonstrate using OCIGenAIAgent as a LangGraph node.
    """

    @pytest.fixture
    def compartment_id(self) -> str:
        return os.environ.get("OCI_COMPARTMENT_ID", "")

    @pytest.fixture
    def service_endpoint(self) -> str:
        region = os.environ.get("OCI_REGION", "us-chicago-1")
        return f"https://inference.generativeai.{region}.oci.oraclecloud.com"

    @pytest.fixture
    def auth_type(self) -> str:
        return os.environ.get("OCI_AUTH_TYPE", "SECURITY_TOKEN")

    @pytest.fixture
    def auth_profile(self) -> str:
        return os.environ.get("OCI_AUTH_PROFILE", "DEFAULT")

    @pytest.fixture
    def model_id(self) -> str:
        return os.environ.get("OCI_MODEL_ID", "meta.llama-4-scout-17b-16e-instruct")

    def test_langgraph_node(
        self,
        model_id: str,
        compartment_id: str,
        service_endpoint: str,
        auth_type: str,
        auth_profile: str,
    ):
        """Test OCIGenAIAgent as a LangGraph node.

        Demonstrates:
        - LangGraph StateGraph integration
        - OCIGenAIAgent as processing node
        """
        try:
            from langgraph.graph import END, START, StateGraph
        except ImportError:
            pytest.skip("LangGraph not installed")

        agent = OCIGenAIAgent(
            model_id=model_id,
            tools=[get_weather],
            compartment_id=compartment_id,
            service_endpoint=service_endpoint,
            auth_type=auth_type,
            auth_profile=auth_profile,
            max_iterations=3,
        )

        # Create graph with agent as node
        graph = StateGraph(dict)
        graph.add_node("agent", agent)
        graph.add_edge(START, "agent")
        graph.add_edge("agent", END)

        compiled = graph.compile()

        # Invoke the graph
        result = compiled.invoke({"input": "What's the weather?"})

        assert result is not None


# =============================================================================
# New Feature Integration Tests (Compression, Confidence Signals, Hooks)
# =============================================================================


@pytest.mark.requires("oci")
@pytest.mark.skipif(
    skip_if_no_oci_credentials(),
    reason="OCI credentials not available (OCI_COMPARTMENT_ID not set)",
)
class TestOCIGenAIAgentNewFeatures:
    """Integration tests for new OCIGenAIAgent features.

    Tests for:
    - Message compression
    - Confidence signal detection
    - Hooks system
    - Conversation history
    """

    @pytest.fixture
    def compartment_id(self) -> str:
        return os.environ.get("OCI_COMPARTMENT_ID", "")

    @pytest.fixture
    def service_endpoint(self) -> str:
        region = os.environ.get("OCI_REGION", "us-chicago-1")
        return f"https://inference.generativeai.{region}.oci.oraclecloud.com"

    @pytest.fixture
    def auth_type(self) -> str:
        return os.environ.get("OCI_AUTH_TYPE", "SECURITY_TOKEN")

    @pytest.fixture
    def auth_profile(self) -> str:
        return os.environ.get("OCI_AUTH_PROFILE", "DEFAULT")

    @pytest.fixture
    def model_id(self) -> str:
        return os.environ.get("OCI_MODEL_ID", "meta.llama-4-scout-17b-16e-instruct")

    # =========================================================================
    # Hooks System Tests
    # =========================================================================

    def test_hooks_triggered_during_execution(
        self,
        model_id: str,
        compartment_id: str,
        service_endpoint: str,
        auth_type: str,
        auth_profile: str,
    ):
        """Test hooks are triggered during agent execution.

        Demonstrates:
        - on_tool_start and on_tool_end hooks
        - on_iteration_start and on_iteration_end hooks
        - on_terminate hook
        """
        from langchain_oci.agents.oci_agent.hooks import (
            AgentHooks,
            IterationContext,
            ToolHookContext,
            ToolResultContext,
        )

        # Track hook calls
        hook_calls: dict = {
            "tool_starts": [],
            "tool_ends": [],
            "iteration_starts": [],
            "iteration_ends": [],
            "terminates": [],
        }

        def on_tool_start(ctx: ToolHookContext):
            hook_calls["tool_starts"].append(ctx.tool_name)

        def on_tool_end(ctx: ToolResultContext):
            hook_calls["tool_ends"].append((ctx.tool_name, ctx.success))

        def on_iteration_start(ctx: IterationContext):
            hook_calls["iteration_starts"].append(ctx.iteration)

        def on_iteration_end(ctx: IterationContext):
            hook_calls["iteration_ends"].append(ctx.iteration)

        def on_terminate(reason: str, final_answer: str):
            hook_calls["terminates"].append(reason)

        hooks = AgentHooks(
            on_tool_start=[on_tool_start],
            on_tool_end=[on_tool_end],
            on_iteration_start=[on_iteration_start],
            on_iteration_end=[on_iteration_end],
            on_terminate=[on_terminate],
        )

        agent = OCIGenAIAgent(
            model_id=model_id,
            tools=[calculate],
            compartment_id=compartment_id,
            service_endpoint=service_endpoint,
            auth_type=auth_type,
            auth_profile=auth_profile,
            hooks=hooks,
            max_iterations=3,
        )

        result = agent.invoke("What is 7 * 8?")

        # Verify hooks were triggered
        assert len(hook_calls["iteration_starts"]) >= 1, "Should trigger iteration start"
        assert len(hook_calls["iteration_ends"]) >= 1, "Should trigger iteration end"
        assert len(hook_calls["terminates"]) == 1, "Should trigger terminate once"

        # If tools were called, verify tool hooks
        if result.total_tool_calls > 0:
            assert len(hook_calls["tool_starts"]) > 0, "Should trigger tool start"
            assert len(hook_calls["tool_ends"]) > 0, "Should trigger tool end"
            assert "calculate" in hook_calls["tool_starts"]

    def test_metrics_hooks_collect_data(
        self,
        model_id: str,
        compartment_id: str,
        service_endpoint: str,
        auth_type: str,
        auth_profile: str,
    ):
        """Test metrics hooks collect execution data.

        Demonstrates:
        - create_metrics_hooks factory function
        - Metrics collection during execution
        """
        from langchain_oci.agents.oci_agent.hooks import create_metrics_hooks

        hooks, metrics = create_metrics_hooks()

        agent = OCIGenAIAgent(
            model_id=model_id,
            tools=[calculate, get_weather],
            compartment_id=compartment_id,
            service_endpoint=service_endpoint,
            auth_type=auth_type,
            auth_profile=auth_profile,
            hooks=hooks,
            max_iterations=3,
        )

        result = agent.invoke("Calculate 15 + 25")

        # Verify metrics were collected
        assert metrics["iterations"] >= 1, "Should track iterations"
        assert metrics["termination_reason"] is not None, "Should record termination"

        if result.total_tool_calls > 0:
            assert metrics["total_tool_calls"] > 0, "Should count tool calls"
            assert len(metrics["tool_durations_ms"]) > 0, "Should track durations"

    # =========================================================================
    # Message Compression Tests
    # =========================================================================

    def test_compression_enabled(
        self,
        model_id: str,
        compartment_id: str,
        service_endpoint: str,
        auth_type: str,
        auth_profile: str,
    ):
        """Test message compression is enabled by default.

        Demonstrates:
        - enable_compression=True by default
        - Compression config accessible
        """
        agent = OCIGenAIAgent(
            model_id=model_id,
            tools=[get_weather],
            compartment_id=compartment_id,
            service_endpoint=service_endpoint,
            auth_type=auth_type,
            auth_profile=auth_profile,
            enable_compression=True,
            max_messages=20,
        )

        assert agent.enable_compression is True
        assert agent._compression_config.max_messages == 20

        # Execute a query
        result = agent.invoke("What's the weather in Chicago?")
        assert isinstance(result, AgentResult)

    def test_compression_with_long_conversation(
        self,
        model_id: str,
        compartment_id: str,
        service_endpoint: str,
        auth_type: str,
        auth_profile: str,
    ):
        """Test compression handles long conversations.

        Demonstrates:
        - Multiple message_history entries
        - Compression prevents context overflow
        """
        agent = OCIGenAIAgent(
            model_id=model_id,
            tools=[],  # No tools to avoid tool_call_id issues with history
            compartment_id=compartment_id,
            service_endpoint=service_endpoint,
            auth_type=auth_type,
            auth_profile=auth_profile,
            enable_compression=True,
            max_messages=10,  # Low limit to trigger compression
            max_iterations=3,
        )

        # Build up message history (simple user/assistant pairs)
        history = [
            {"role": "user", "content": "Hi, I need help with math."},
            {"role": "assistant", "content": "I can help with math!"},
            {"role": "user", "content": "What is 2 + 2?"},
            {"role": "assistant", "content": "2 + 2 = 4"},
            {"role": "user", "content": "And 3 + 3?"},
            {"role": "assistant", "content": "3 + 3 = 6"},
        ]

        result = agent.invoke("What's 20 + 30?", message_history=history)

        assert isinstance(result, AgentResult)
        assert result.final_answer

    # =========================================================================
    # Confidence Signal Detection Tests
    # =========================================================================

    def test_confidence_signals_detection(
        self,
        model_id: str,
        compartment_id: str,
        service_endpoint: str,
        auth_type: str,
        auth_profile: str,
    ):
        """Test confidence signals are detected in responses.

        Demonstrates:
        - enable_confidence_signals=True
        - Confidence accumulates during execution
        """
        agent = OCIGenAIAgent(
            model_id=model_id,
            tools=[search_database],
            compartment_id=compartment_id,
            service_endpoint=service_endpoint,
            auth_type=auth_type,
            auth_profile=auth_profile,
            enable_confidence_signals=True,
            enable_reflexion=True,
            max_iterations=5,
        )

        result = agent.invoke("What is the capital of France?")

        # Confidence should be tracked
        assert result.confidence >= 0.0
        assert result.confidence <= 1.0

    def test_early_exit_on_high_confidence(
        self,
        model_id: str,
        compartment_id: str,
        service_endpoint: str,
        auth_type: str,
        auth_profile: str,
    ):
        """Test agent can exit early when confidence is high.

        Demonstrates:
        - min_iterations_for_early_exit setting
        - Early exit saves tokens
        """
        agent = OCIGenAIAgent(
            model_id=model_id,
            tools=[search_database],
            compartment_id=compartment_id,
            service_endpoint=service_endpoint,
            auth_type=auth_type,
            auth_profile=auth_profile,
            enable_confidence_signals=True,
            min_iterations_for_early_exit=1,  # Allow early exit after 1 iteration
            confidence_threshold=0.3,  # Low threshold for testing
            max_iterations=10,
        )

        result = agent.invoke("What is the capital of Japan?")

        # Agent should complete (possibly with early exit)
        assert isinstance(result, AgentResult)
        assert result.termination_reason is not None

    # =========================================================================
    # Conversation History Tests
    # =========================================================================

    def test_conversation_history_basic(
        self,
        model_id: str,
        compartment_id: str,
        service_endpoint: str,
        auth_type: str,
        auth_profile: str,
    ):
        """Test basic conversation history support.

        Demonstrates:
        - message_history parameter
        - Context from previous turns
        """
        agent = OCIGenAIAgent(
            model_id=model_id,
            tools=[calculate],
            compartment_id=compartment_id,
            service_endpoint=service_endpoint,
            auth_type=auth_type,
            auth_profile=auth_profile,
            max_iterations=3,
        )

        # First turn
        result1 = agent.invoke("What is 10 + 5?")

        # Second turn with history (convert messages to dicts)
        history = [
            {"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content}
            for m in result1.messages if hasattr(m, 'content') and isinstance(m.content, str)
        ]
        result2 = agent.invoke("Now double that result", message_history=history)

        assert isinstance(result2, AgentResult)
        # The agent should understand "that result" from context

    def test_conversation_history_multi_turn(
        self,
        model_id: str,
        compartment_id: str,
        service_endpoint: str,
        auth_type: str,
        auth_profile: str,
    ):
        """Test multi-turn conversation with accumulating history.

        Demonstrates:
        - Multiple conversation turns
        - History accumulation
        """
        agent = OCIGenAIAgent(
            model_id=model_id,
            tools=[get_weather, calculate],
            compartment_id=compartment_id,
            service_endpoint=service_endpoint,
            auth_type=auth_type,
            auth_profile=auth_profile,
            max_iterations=3,
        )

        def msgs_to_history(messages):
            """Convert Message objects to history dicts."""
            return [
                {"role": "user" if isinstance(m, HumanMessage) else "assistant",
                 "content": m.content if isinstance(m.content, str) else str(m.content)}
                for m in messages if hasattr(m, 'content')
            ]

        # Turn 1
        result1 = agent.invoke("What's the weather in Chicago?")
        history = msgs_to_history(result1.messages)

        # Turn 2 - builds on turn 1
        result2 = agent.invoke(
            "And what about New York?",
            message_history=history,
        )
        history.extend(msgs_to_history(result2.messages))

        # Turn 3 - uses context from both turns
        result3 = agent.invoke(
            "Calculate the difference between those two temperatures",
            message_history=history,
        )

        assert isinstance(result3, AgentResult)

    # =========================================================================
    # Combined Features Tests
    # =========================================================================

    def test_all_features_together(
        self,
        model_id: str,
        compartment_id: str,
        service_endpoint: str,
        auth_type: str,
        auth_profile: str,
    ):
        """Test all new features working together.

        Demonstrates:
        - Hooks + Compression + Confidence Signals
        - Full feature integration
        """
        from langchain_oci.agents.oci_agent.hooks import create_metrics_hooks

        hooks, metrics = create_metrics_hooks()

        agent = OCIGenAIAgent(
            model_id=model_id,
            tools=[get_weather, calculate, search_database],
            compartment_id=compartment_id,
            service_endpoint=service_endpoint,
            auth_type=auth_type,
            auth_profile=auth_profile,
            # Hooks
            hooks=hooks,
            # Compression
            enable_compression=True,
            max_messages=15,
            # Confidence signals
            enable_confidence_signals=True,
            min_iterations_for_early_exit=2,
            # Reflexion
            enable_reflexion=True,
            max_iterations=5,
        )

        # Build some history
        history = [
            {"role": "user", "content": "I need help with multiple things today."},
        ]

        result = agent.invoke(
            "What's the weather in Paris and calculate 12 * 12?",
            message_history=history,
        )

        # Verify all features worked
        assert isinstance(result, AgentResult)
        assert result.final_answer

        # Metrics should be collected
        assert metrics["iterations"] >= 1
        assert metrics["termination_reason"] is not None

        # Confidence should be tracked
        assert 0.0 <= result.confidence <= 1.0
