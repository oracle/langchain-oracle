# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Unit tests for OCIGenAIAgent."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from langchain_oci.agents.oci_agent.events import (
    ReflectEvent,
    TerminateEvent,
    ThinkEvent,
    ToolCompleteEvent,
    ToolStartEvent,
)
from langchain_oci.agents.oci_agent.reflexion import (
    AssessmentCategory,
    Reflector,
    assess_confidence,
    assess_progress,
    detect_loop,
)
from langchain_oci.agents.oci_agent.state import (
    AgentResult,
    AgentState,
    ReasoningStep,
    ToolExecution,
)
from langchain_oci.agents.oci_agent.termination import (
    TerminationReason,
    check_confidence_threshold,
    check_max_iterations,
    check_no_tools,
    check_terminal_tool,
    check_termination,
    check_tool_loop,
    get_termination_description,
)


class TestToolExecution:
    """Tests for ToolExecution model."""

    def test_creation(self):
        """Test basic creation."""
        execution = ToolExecution(
            tool_name="search",
            tool_call_id="tc_123",
            arguments={"query": "test"},
            result="found results",
            success=True,
            duration_ms=100.5,
        )
        assert execution.tool_name == "search"
        assert execution.tool_call_id == "tc_123"
        assert execution.arguments == {"query": "test"}
        assert execution.result == "found results"
        assert execution.success is True
        assert execution.duration_ms == 100.5

    def test_frozen(self):
        """Test that model is frozen."""
        execution = ToolExecution(
            tool_name="search",
            tool_call_id="tc_123",
        )
        with pytest.raises(Exception):
            execution.tool_name = "other"

    def test_failure_execution(self):
        """Test failed execution."""
        execution = ToolExecution(
            tool_name="search",
            tool_call_id="tc_123",
            success=False,
            error="Connection failed",
        )
        assert execution.success is False
        assert execution.error == "Connection failed"


class TestReasoningStep:
    """Tests for ReasoningStep model."""

    def test_creation(self):
        """Test basic creation."""
        step = ReasoningStep(
            iteration=0,
            thought="Let me search for information.",
            confidence=0.3,
            assessment="on_track",
        )
        assert step.iteration == 0
        assert step.thought == "Let me search for information."
        assert step.confidence == 0.3
        assert step.assessment == "on_track"

    def test_with_tool_executions(self):
        """Test step with tool executions."""
        execution = ToolExecution(
            tool_name="search",
            tool_call_id="tc_1",
            result="data",
        )
        step = ReasoningStep(
            iteration=1,
            tool_executions=(execution,),
            confidence=0.5,
        )
        assert len(step.tool_executions) == 1
        assert step.tool_executions[0].tool_name == "search"


class TestAgentState:
    """Tests for AgentState model."""

    def test_creation(self):
        """Test basic creation."""
        state = AgentState()
        assert state.messages == ()
        assert state.reasoning_steps == ()
        assert state.iteration == 0
        assert state.confidence == 0.0
        assert state.tool_history == ()

    def test_with_message(self):
        """Test adding a message."""
        state = AgentState()
        msg = HumanMessage(content="Hello")
        new_state = state.with_message(msg)

        # Original unchanged
        assert len(state.messages) == 0
        # New state has message
        assert len(new_state.messages) == 1
        assert new_state.messages[0].content == "Hello"

    def test_with_messages(self):
        """Test adding multiple messages."""
        state = AgentState()
        msgs = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there!"),
        ]
        new_state = state.with_messages(msgs)
        assert len(new_state.messages) == 2

    def test_with_reasoning_step(self):
        """Test adding a reasoning step."""
        state = AgentState()
        step = ReasoningStep(iteration=0, thought="Thinking...")
        new_state = state.with_reasoning_step(step)

        assert len(state.reasoning_steps) == 0
        assert len(new_state.reasoning_steps) == 1
        assert new_state.reasoning_steps[0].thought == "Thinking..."

    def test_with_tool_call(self):
        """Test recording tool call."""
        state = AgentState()
        new_state = state.with_tool_call("search")
        assert state.tool_history == ()
        assert new_state.tool_history == ("search",)

    def test_with_confidence(self):
        """Test setting confidence."""
        state = AgentState()
        new_state = state.with_confidence(0.75)
        assert new_state.confidence == 0.75

    def test_with_confidence_clamping(self):
        """Test confidence clamping."""
        state = AgentState()
        assert state.with_confidence(-0.5).confidence == 0.0
        assert state.with_confidence(1.5).confidence == 1.0

    def test_adjust_confidence(self):
        """Test adjusting confidence."""
        state = AgentState(confidence=0.5)
        new_state = state.adjust_confidence(0.2)
        # With diminishing returns: 0.5 + 0.2 * (1 - 0.5) = 0.6
        assert new_state.confidence == 0.6

    def test_adjust_confidence_without_diminishing(self):
        """Test adjusting confidence without diminishing returns."""
        state = AgentState(confidence=0.5)
        new_state = state.adjust_confidence(0.2, diminishing=False)
        assert new_state.confidence == 0.7

    def test_increment_iteration(self):
        """Test incrementing iteration."""
        state = AgentState(iteration=5)
        new_state = state.increment_iteration()
        assert state.iteration == 5
        assert new_state.iteration == 6

    def test_with_metadata(self):
        """Test adding metadata."""
        state = AgentState()
        new_state = state.with_metadata("key", "value")
        assert state.metadata == {}
        assert new_state.metadata == {"key": "value"}

    def test_immutability(self):
        """Test that state is immutable."""
        state = AgentState()
        with pytest.raises(Exception):
            state.iteration = 5


class TestAgentResult:
    """Tests for AgentResult model."""

    def test_creation(self):
        """Test basic creation."""
        result = AgentResult(
            messages=[HumanMessage(content="Hi")],
            final_answer="Hello!",
            termination_reason="no_tools",
            reasoning_steps=[],
            total_iterations=1,
            total_tool_calls=0,
        )
        assert result.final_answer == "Hello!"
        assert result.termination_reason == "no_tools"

    def test_from_state(self):
        """Test creating result from state."""
        execution = ToolExecution(tool_name="search", tool_call_id="tc_1")
        step = ReasoningStep(
            iteration=0,
            tool_executions=(execution,),
        )
        state = AgentState(
            messages=(
                HumanMessage(content="Hi"),
                AIMessage(content="Hello!"),
            ),
            reasoning_steps=(step,),
            iteration=1,
            confidence=0.8,
        )
        result = AgentResult.from_state(
            state,
            final_answer="Hello!",
            termination_reason="confidence_met",
        )
        assert result.final_answer == "Hello!"
        assert result.termination_reason == "confidence_met"
        assert result.total_iterations == 1
        assert result.total_tool_calls == 1
        assert result.confidence == 0.8


class TestEvents:
    """Tests for event models."""

    def test_think_event(self):
        """Test ThinkEvent."""
        event = ThinkEvent(
            iteration=0,
            thought="Let me search...",
            tool_calls_planned=1,
        )
        assert event.event_type == "think"
        assert event.iteration == 0
        assert event.thought == "Let me search..."
        assert event.tool_calls_planned == 1

    def test_tool_start_event(self):
        """Test ToolStartEvent."""
        event = ToolStartEvent(
            iteration=0,
            tool_name="search",
            tool_call_id="tc_1",
            arguments={"query": "test"},
        )
        assert event.event_type == "tool_start"
        assert event.tool_name == "search"

    def test_tool_complete_event(self):
        """Test ToolCompleteEvent."""
        event = ToolCompleteEvent(
            iteration=0,
            tool_name="search",
            tool_call_id="tc_1",
            result="found data",
            success=True,
            duration_ms=50.0,
        )
        assert event.event_type == "tool_complete"
        assert event.success is True
        assert event.result == "found data"

    def test_reflect_event(self):
        """Test ReflectEvent."""
        event = ReflectEvent(
            iteration=0,
            confidence=0.5,
            confidence_delta=0.1,
            assessment="on_track",
            loop_detected=False,
        )
        assert event.event_type == "reflect"
        assert event.confidence == 0.5

    def test_terminate_event(self):
        """Test TerminateEvent."""
        event = TerminateEvent(
            reason="no_tools",
            final_answer="The answer is 42.",
            total_iterations=3,
            total_tool_calls=5,
            confidence=0.9,
        )
        assert event.event_type == "terminate"
        assert event.reason == "no_tools"


class TestTermination:
    """Tests for termination conditions."""

    def test_check_max_iterations(self):
        """Test max iterations check."""
        state = AgentState(iteration=10)
        assert check_max_iterations(state, 10) == TerminationReason.MAX_ITERATIONS
        assert check_max_iterations(state, 11) is None

    def test_check_confidence_threshold(self):
        """Test confidence threshold check."""
        state = AgentState(confidence=0.8)
        assert (
            check_confidence_threshold(state, 0.8) == TerminationReason.CONFIDENCE_MET
        )
        assert check_confidence_threshold(state, 0.9) is None

    def test_check_terminal_tool(self):
        """Test terminal tool check."""
        terminal_tools = {"done", "submit"}
        assert (
            check_terminal_tool(["search", "done"], terminal_tools)
            == TerminationReason.TERMINAL_TOOL
        )
        assert check_terminal_tool(["search"], terminal_tools) is None

    def test_check_no_tools(self):
        """Test no tools check."""
        assert check_no_tools(0) == TerminationReason.NO_TOOLS
        assert check_no_tools(1) is None

    def test_check_tool_loop_single(self):
        """Test single tool loop detection."""
        state = AgentState(tool_history=("search", "search", "search"))
        assert check_tool_loop(state, 3) == TerminationReason.TOOL_LOOP

    def test_check_tool_loop_alternating(self):
        """Test alternating tool loop detection."""
        state = AgentState(tool_history=("search", "fetch", "search", "fetch"))
        assert check_tool_loop(state, 4) == TerminationReason.TOOL_LOOP

    def test_check_tool_loop_no_loop(self):
        """Test no loop detected."""
        state = AgentState(tool_history=("search", "fetch", "process"))
        assert check_tool_loop(state, 3) is None

    def test_check_termination_priority(self):
        """Test termination check priority."""
        # Max iterations takes priority
        state = AgentState(
            iteration=10,
            confidence=0.9,
            tool_history=("search", "search", "search"),
        )
        assert (
            check_termination(
                state,
                max_iterations=10,
                confidence_threshold=0.8,
                terminal_tools=set(),
            )
            == TerminationReason.MAX_ITERATIONS
        )

    def test_get_termination_description(self):
        """Test termination descriptions."""
        assert "Maximum" in get_termination_description(
            TerminationReason.MAX_ITERATIONS
        )
        assert "Confidence" in get_termination_description(
            TerminationReason.CONFIDENCE_MET
        )


class TestReflexion:
    """Tests for reflexion functionality."""

    def test_reflector_creation(self):
        """Test Reflector creation."""
        reflector = Reflector(
            loop_threshold=3,
            success_weight=0.15,
            error_penalty=0.2,
        )
        assert reflector.loop_threshold == 3
        assert reflector.success_weight == 0.15

    def test_reflect_no_executions(self):
        """Test reflection with no executions."""
        reflector = Reflector()
        state = AgentState()
        result = reflector.reflect(state, [])
        assert result.assessment == AssessmentCategory.ON_TRACK
        assert result.confidence_delta == 0.0

    def test_reflect_successful_execution(self):
        """Test reflection with successful execution."""
        reflector = Reflector()
        state = AgentState()
        executions = [
            ToolExecution(
                tool_name="search",
                tool_call_id="tc_1",
                result="found data",
                success=True,
            )
        ]
        result = reflector.reflect(state, executions)
        assert result.confidence_delta > 0
        assert result.assessment in (
            AssessmentCategory.ON_TRACK,
            AssessmentCategory.NEW_FINDINGS,
        )

    def test_reflect_failed_execution(self):
        """Test reflection with failed execution."""
        reflector = Reflector()
        state = AgentState()
        executions = [
            ToolExecution(
                tool_name="search",
                tool_call_id="tc_1",
                success=False,
                error="Failed",
            )
        ]
        result = reflector.reflect(state, executions)
        assert result.confidence_delta < 0
        assert result.assessment == AssessmentCategory.STUCK

    def test_reflect_loop_detected(self):
        """Test reflection detects loop."""
        reflector = Reflector(loop_threshold=3)
        state = AgentState(tool_history=("search", "search", "search"))
        result = reflector.reflect(state, [])
        assert result.assessment == AssessmentCategory.LOOP_DETECTED
        assert result.loop_pattern is not None
        assert result.confidence_delta < 0

    def test_reflect_new_findings(self):
        """Test reflection with new findings."""
        reflector = Reflector()
        state = AgentState()
        # Create execution with substantial content
        executions = [
            ToolExecution(
                tool_name="search",
                tool_call_id="tc_1",
                result="x" * 150,  # > 100 chars triggers new_findings
                success=True,
            )
        ]
        result = reflector.reflect(state, executions)
        assert result.assessment == AssessmentCategory.NEW_FINDINGS
        assert result.findings_summary is not None

    def test_assess_confidence(self):
        """Test convenience confidence assessment."""
        state = AgentState(confidence=0.3)
        executions = [
            ToolExecution(
                tool_name="search",
                tool_call_id="tc_1",
                success=True,
            )
        ]
        new_confidence = assess_confidence(state, executions)
        assert new_confidence > 0.3

    def test_detect_loop(self):
        """Test convenience loop detection."""
        state_with_loop = AgentState(tool_history=("a", "a", "a"))
        state_without_loop = AgentState(tool_history=("a", "b", "c"))
        assert detect_loop(state_with_loop, threshold=3) is True
        assert detect_loop(state_without_loop, threshold=3) is False

    def test_assess_progress(self):
        """Test convenience progress assessment."""
        state = AgentState()
        executions = [
            ToolExecution(tool_name="search", tool_call_id="tc_1", success=True)
        ]
        assessment = assess_progress(state, executions)
        assert assessment in (
            AssessmentCategory.ON_TRACK,
            AssessmentCategory.NEW_FINDINGS,
        )


class TestOCIGenAIAgentInitialization:
    """Tests for OCIGenAIAgent initialization (mocked)."""

    @patch("langchain_oci.agents.oci_agent.agent.ChatOCIGenAI")
    def test_basic_initialization(self, mock_chat):
        """Test basic agent initialization."""
        from langchain_core.tools import tool

        @tool
        def dummy_tool(x: str) -> str:
            """Dummy tool."""
            return x

        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_chat.return_value = mock_llm

        from langchain_oci.agents.oci_agent import OCIGenAIAgent

        agent = OCIGenAIAgent(
            model_id="test-model",
            tools=[dummy_tool],
            compartment_id="test-compartment",
        )

        assert agent.model_id == "test-model"
        assert agent.max_iterations == 10
        assert agent.enable_reflexion is True
        assert "dummy_tool" in agent._tools

    @patch("langchain_oci.agents.oci_agent.agent.ChatOCIGenAI")
    def test_custom_configuration(self, mock_chat):
        """Test agent with custom configuration."""
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_chat.return_value = mock_llm

        from langchain_oci.agents.oci_agent import OCIGenAIAgent

        agent = OCIGenAIAgent(
            model_id="test-model",
            tools=[],
            max_iterations=5,
            confidence_threshold=0.9,
            enable_reflexion=False,
            terminal_tools=["complete"],
        )

        assert agent.max_iterations == 5
        assert agent.confidence_threshold == 0.9
        assert agent.enable_reflexion is False
        assert "complete" in agent.terminal_tools

    @patch("langchain_oci.agents.oci_agent.agent.ChatOCIGenAI")
    def test_repr(self, mock_chat):
        """Test agent string representation."""
        from langchain_core.tools import tool

        @tool
        def search(query: str) -> str:
            """Search."""
            return query

        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_chat.return_value = mock_llm

        from langchain_oci.agents.oci_agent import OCIGenAIAgent

        agent = OCIGenAIAgent(model_id="test-model", tools=[search])
        repr_str = repr(agent)
        assert "OCIGenAIAgent" in repr_str
        assert "test-model" in repr_str
        assert "search" in repr_str


class TestCompression:
    """Tests for message compression functionality."""

    def test_compression_strategy_enum(self):
        """Test CompressionStrategy enum values."""
        from langchain_oci.agents.oci_agent.compression import CompressionStrategy

        assert CompressionStrategy.NONE.value == "none"
        assert CompressionStrategy.FIXED_WINDOW.value == "fixed_window"
        assert CompressionStrategy.SMART_TRIM.value == "smart_trim"

    def test_compression_config_defaults(self):
        """Test CompressionConfig default values."""
        from langchain_oci.agents.oci_agent.compression import (
            CompressionConfig,
            CompressionStrategy,
        )

        config = CompressionConfig()
        assert config.strategy == CompressionStrategy.SMART_TRIM
        assert config.max_messages == 20
        assert config.preserve_system is True

    def test_compression_config_frozen(self):
        """Test CompressionConfig is frozen."""
        from langchain_oci.agents.oci_agent.compression import CompressionConfig

        config = CompressionConfig()
        with pytest.raises(Exception):
            config.max_messages = 50

    def test_compression_result(self):
        """Test CompressionResult creation."""
        from langchain_oci.agents.oci_agent.compression import CompressionResult

        result = CompressionResult(
            messages=(HumanMessage(content="Hi"),),
            dropped_count=5,
        )
        assert len(result.messages) == 1
        assert result.dropped_count == 5

    def test_compress_no_compression(self):
        """Test compression with NONE strategy."""
        from langchain_oci.agents.oci_agent.compression import (
            CompressionConfig,
            CompressionStrategy,
            compress_messages,
        )

        messages = tuple(HumanMessage(content=f"Msg {i}") for i in range(30))
        config = CompressionConfig(strategy=CompressionStrategy.NONE)
        result = compress_messages(messages, config)

        assert len(result.messages) == 30
        assert result.dropped_count == 0

    def test_compress_under_limit(self):
        """Test compression when under message limit."""
        from langchain_oci.agents.oci_agent.compression import (
            CompressionConfig,
            compress_messages,
        )

        messages = tuple(HumanMessage(content=f"Msg {i}") for i in range(5))
        config = CompressionConfig(max_messages=10)
        result = compress_messages(messages, config)

        assert len(result.messages) == 5
        assert result.dropped_count == 0

    def test_fixed_window_compression(self):
        """Test fixed window compression."""
        from langchain_oci.agents.oci_agent.compression import (
            CompressionConfig,
            CompressionStrategy,
            compress_messages,
        )

        messages = (
            SystemMessage(content="System"),
            HumanMessage(content="Old 1"),
            HumanMessage(content="Old 2"),
            HumanMessage(content="Recent 1"),
            HumanMessage(content="Recent 2"),
        )
        config = CompressionConfig(
            strategy=CompressionStrategy.FIXED_WINDOW,
            max_messages=3,
        )
        result = compress_messages(messages, config)

        # Should keep system + 2 most recent
        assert len(result.messages) == 3
        assert result.dropped_count == 2
        assert result.messages[0].content == "System"
        assert result.messages[1].content == "Recent 1"
        assert result.messages[2].content == "Recent 2"

    def test_smart_trim_preserves_tool_pairs(self):
        """Test smart trim keeps AI+Tool message pairs together."""
        from langchain_oci.agents.oci_agent.compression import (
            CompressionConfig,
            CompressionStrategy,
            compress_messages,
        )

        # AI message with tool calls + tool message
        ai_with_tools = AIMessage(
            content="Using tool",
            tool_calls=[{"name": "search", "args": {}, "id": "tc_1"}],
        )
        tool_result = ToolMessage(content="Result", tool_call_id="tc_1")

        messages = (
            SystemMessage(content="System"),
            HumanMessage(content="Old question"),
            AIMessage(content="Old answer"),
            HumanMessage(content="New question"),
            ai_with_tools,
            tool_result,
        )
        config = CompressionConfig(
            strategy=CompressionStrategy.SMART_TRIM,
            max_messages=4,
        )
        result = compress_messages(messages, config)

        # Should keep system + most recent group (AI+Tool pair)
        assert result.dropped_count > 0
        # AI with tool_calls and its ToolMessage must stay together
        tool_call_found = False
        tool_result_found = False
        for msg in result.messages:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                tool_call_found = True
            if isinstance(msg, ToolMessage):
                tool_result_found = True
        assert tool_call_found == tool_result_found  # Either both or neither

    def test_smart_trim_preserves_system(self):
        """Test smart trim always preserves system messages."""
        from langchain_oci.agents.oci_agent.compression import (
            CompressionConfig,
            CompressionStrategy,
            compress_messages,
        )

        messages = (
            SystemMessage(content="Important system prompt"),
            *tuple(HumanMessage(content=f"Msg {i}") for i in range(25)),
        )
        config = CompressionConfig(
            strategy=CompressionStrategy.SMART_TRIM,
            max_messages=10,
        )
        result = compress_messages(messages, config)

        # System message should be first
        assert isinstance(result.messages[0], SystemMessage)
        assert result.messages[0].content == "Important system prompt"


class TestConfidenceSignals:
    """Tests for confidence signal detection."""

    def test_signal_type_enum(self):
        """Test SignalType enum values."""
        from langchain_oci.agents.oci_agent.confidence import SignalType

        assert SignalType.EXPLICIT_CONFIDENCE.value == "explicit_confidence"
        assert SignalType.VERIFICATION.value == "verification"
        assert SignalType.ROOT_CAUSE.value == "root_cause"
        assert SignalType.TASK_COMPLETE.value == "task_complete"

    def test_confidence_signal_creation(self):
        """Test ConfidenceSignal creation."""
        from langchain_oci.agents.oci_agent.confidence import (
            ConfidenceSignal,
            SignalType,
        )

        signal = ConfidenceSignal(
            signal_type=SignalType.VERIFICATION,
            text_match="verified",
            weight=0.15,
            iteration=2,
        )
        assert signal.signal_type == SignalType.VERIFICATION
        assert signal.text_match == "verified"
        assert signal.weight == 0.15
        assert signal.iteration == 2

    def test_confidence_signal_frozen(self):
        """Test ConfidenceSignal is frozen."""
        from langchain_oci.agents.oci_agent.confidence import (
            ConfidenceSignal,
            SignalType,
        )

        signal = ConfidenceSignal(
            signal_type=SignalType.VERIFICATION,
            text_match="verified",
            weight=0.15,
            iteration=1,
        )
        with pytest.raises(Exception):
            signal.weight = 0.5

    def test_detect_no_signals(self):
        """Test detection returns empty for neutral text."""
        from langchain_oci.agents.oci_agent.confidence import detect_confidence_signals

        signals = detect_confidence_signals("Here is some neutral text.", 1)
        assert len(signals) == 0

    def test_detect_verification_signal(self):
        """Test detection of verification signals."""
        from langchain_oci.agents.oci_agent.confidence import (
            SignalType,
            detect_confidence_signals,
        )

        signals = detect_confidence_signals(
            "I have verified that the data is correct.", 1
        )
        assert any(s.signal_type == SignalType.VERIFICATION for s in signals)
        assert any(s.text_match == "verified" for s in signals)

    def test_detect_explicit_confidence(self):
        """Test detection of explicit confidence markers."""
        from langchain_oci.agents.oci_agent.confidence import (
            SignalType,
            detect_confidence_signals,
        )

        signals = detect_confidence_signals(
            "I am confident that this is the correct answer.", 1
        )
        assert any(s.signal_type == SignalType.EXPLICIT_CONFIDENCE for s in signals)

    def test_detect_root_cause(self):
        """Test detection of root cause signals."""
        from langchain_oci.agents.oci_agent.confidence import (
            SignalType,
            detect_confidence_signals,
        )

        signals = detect_confidence_signals("The root cause is a memory leak.", 2)
        assert any(s.signal_type == SignalType.ROOT_CAUSE for s in signals)

    def test_detect_task_complete(self):
        """Test detection of task completion signals."""
        from langchain_oci.agents.oci_agent.confidence import (
            SignalType,
            detect_confidence_signals,
        )

        signals = detect_confidence_signals(
            "I have successfully completed the task.", 3
        )
        assert any(s.signal_type == SignalType.TASK_COMPLETE for s in signals)

    def test_detect_multiple_signals(self):
        """Test detection of multiple signals in one message."""
        from langchain_oci.agents.oci_agent.confidence import detect_confidence_signals

        text = "I am confident that this is verified and the root cause is found."
        signals = detect_confidence_signals(text, 1)
        assert len(signals) >= 2

    def test_case_insensitive_detection(self):
        """Test detection is case insensitive."""
        from langchain_oci.agents.oci_agent.confidence import detect_confidence_signals

        signals = detect_confidence_signals("I AM CONFIDENT that this works.", 1)
        assert len(signals) > 0

    def test_compute_accumulated_confidence_empty(self):
        """Test accumulated confidence with no signals."""
        from langchain_oci.agents.oci_agent.confidence import (
            compute_accumulated_confidence,
        )

        confidence = compute_accumulated_confidence([], base_confidence=0.5)
        assert confidence == 0.5

    def test_compute_accumulated_confidence_single(self):
        """Test accumulated confidence with one signal."""
        from langchain_oci.agents.oci_agent.confidence import (
            ConfidenceSignal,
            SignalType,
            compute_accumulated_confidence,
        )

        signals = [
            ConfidenceSignal(
                signal_type=SignalType.VERIFICATION,
                text_match="verified",
                weight=0.15,
                iteration=1,
            )
        ]
        confidence = compute_accumulated_confidence(signals, base_confidence=0.5)
        assert confidence == 0.65  # 0.5 + 0.15

    def test_compute_accumulated_confidence_diminishing(self):
        """Test diminishing returns for same signal type."""
        from langchain_oci.agents.oci_agent.confidence import (
            ConfidenceSignal,
            SignalType,
            compute_accumulated_confidence,
        )

        # Two signals of same type
        signals = [
            ConfidenceSignal(
                signal_type=SignalType.VERIFICATION,
                text_match="verified",
                weight=0.20,
                iteration=1,
            ),
            ConfidenceSignal(
                signal_type=SignalType.VERIFICATION,
                text_match="confirmed",
                weight=0.15,
                iteration=2,
            ),
        ]
        confidence = compute_accumulated_confidence(signals, base_confidence=0.5)
        # First: 0.20, Second: 0.15 * 0.5 = 0.075
        # Total: 0.5 + 0.20 + 0.075 = 0.775
        assert confidence == pytest.approx(0.775)

    def test_compute_accumulated_capped_at_one(self):
        """Test confidence is capped at 1.0."""
        from langchain_oci.agents.oci_agent.confidence import (
            ConfidenceSignal,
            SignalType,
            compute_accumulated_confidence,
        )

        signals = [
            ConfidenceSignal(
                signal_type=SignalType.TASK_COMPLETE,
                text_match="successfully completed",
                weight=0.25,
                iteration=i,
            )
            for i in range(10)
        ]
        confidence = compute_accumulated_confidence(signals, base_confidence=0.9)
        assert confidence == 1.0

    def test_should_early_exit_below_threshold(self):
        """Test early exit returns False when below threshold."""
        from langchain_oci.agents.oci_agent.confidence import should_early_exit

        assert should_early_exit(0.8, iteration=3, threshold=0.9) is False

    def test_should_early_exit_below_min_iterations(self):
        """Test early exit returns False when below min iterations."""
        from langchain_oci.agents.oci_agent.confidence import should_early_exit

        assert should_early_exit(0.95, iteration=1, min_iterations=2) is False

    def test_should_early_exit_conditions_met(self):
        """Test early exit returns True when both conditions met."""
        from langchain_oci.agents.oci_agent.confidence import should_early_exit

        assert should_early_exit(0.95, iteration=3, min_iterations=2, threshold=0.9)


class TestHooks:
    """Tests for agent hooks system."""

    def test_tool_hook_context(self):
        """Test ToolHookContext creation."""
        from langchain_oci.agents.oci_agent.hooks import ToolHookContext

        ctx = ToolHookContext(
            tool_name="search",
            tool_call_id="tc_123",
            arguments={"query": "test"},
            iteration=5,
        )
        assert ctx.tool_name == "search"
        assert ctx.tool_call_id == "tc_123"
        assert ctx.arguments == {"query": "test"}
        assert ctx.iteration == 5

    def test_tool_hook_context_frozen(self):
        """Test ToolHookContext is frozen."""
        from langchain_oci.agents.oci_agent.hooks import ToolHookContext

        ctx = ToolHookContext(
            tool_name="search",
            tool_call_id="tc_123",
            arguments={},
            iteration=1,
        )
        with pytest.raises(Exception):
            ctx.tool_name = "other"

    def test_tool_result_context(self):
        """Test ToolResultContext creation."""
        from langchain_oci.agents.oci_agent.hooks import ToolResultContext

        ctx = ToolResultContext(
            tool_name="search",
            tool_call_id="tc_123",
            arguments={"query": "test"},
            result="found 10 items",
            success=True,
            error=None,
            duration_ms=45.2,
            iteration=3,
        )
        assert ctx.tool_name == "search"
        assert ctx.result == "found 10 items"
        assert ctx.success is True
        assert ctx.error is None
        assert ctx.duration_ms == 45.2

    def test_tool_result_context_with_error(self):
        """Test ToolResultContext with error."""
        from langchain_oci.agents.oci_agent.hooks import ToolResultContext

        ctx = ToolResultContext(
            tool_name="api_call",
            tool_call_id="tc_456",
            arguments={"url": "http://example.com"},
            result="",
            success=False,
            error="Connection timeout",
            duration_ms=5000.0,
            iteration=2,
        )
        assert ctx.success is False
        assert ctx.error == "Connection timeout"

    def test_iteration_context(self):
        """Test IterationContext creation."""
        from langchain_oci.agents.oci_agent.hooks import IterationContext

        ctx = IterationContext(
            iteration=4,
            confidence=0.75,
            tool_count=8,
        )
        assert ctx.iteration == 4
        assert ctx.confidence == 0.75
        assert ctx.tool_count == 8

    def test_agent_hooks_creation(self):
        """Test AgentHooks creation with no hooks."""
        from langchain_oci.agents.oci_agent.hooks import AgentHooks

        hooks = AgentHooks()
        assert hooks.on_tool_start is None
        assert hooks.on_tool_end is None
        assert hooks.on_iteration_start is None
        assert hooks.on_iteration_end is None
        assert hooks.on_terminate is None

    def test_agent_hooks_with_callbacks(self):
        """Test AgentHooks with callbacks."""
        from langchain_oci.agents.oci_agent.hooks import AgentHooks, ToolHookContext

        calls = []

        def on_start(ctx: ToolHookContext):
            calls.append(("start", ctx.tool_name))

        hooks = AgentHooks(on_tool_start=[on_start])
        assert hooks.on_tool_start is not None
        assert len(hooks.on_tool_start) == 1

    def test_trigger_tool_start(self):
        """Test triggering tool start hooks."""
        from langchain_oci.agents.oci_agent.hooks import (
            AgentHooks,
            ToolHookContext,
        )

        calls = []

        def on_start(ctx: ToolHookContext):
            calls.append(ctx.tool_name)

        hooks = AgentHooks(on_tool_start=[on_start])
        ctx = ToolHookContext(
            tool_name="search",
            tool_call_id="tc_1",
            arguments={},
            iteration=1,
        )
        hooks.trigger_tool_start(ctx)

        assert calls == ["search"]

    def test_trigger_tool_end(self):
        """Test triggering tool end hooks."""
        from langchain_oci.agents.oci_agent.hooks import (
            AgentHooks,
            ToolResultContext,
        )

        calls = []

        def on_end(ctx: ToolResultContext):
            calls.append((ctx.tool_name, ctx.success))

        hooks = AgentHooks(on_tool_end=[on_end])
        ctx = ToolResultContext(
            tool_name="fetch",
            tool_call_id="tc_2",
            arguments={},
            result="data",
            success=True,
            error=None,
            duration_ms=100.0,
            iteration=1,
        )
        hooks.trigger_tool_end(ctx)

        assert calls == [("fetch", True)]

    def test_trigger_iteration_hooks(self):
        """Test triggering iteration hooks."""
        from langchain_oci.agents.oci_agent.hooks import (
            AgentHooks,
            IterationContext,
        )

        start_calls = []
        end_calls = []

        def on_iter_start(ctx: IterationContext):
            start_calls.append(ctx.iteration)

        def on_iter_end(ctx: IterationContext):
            end_calls.append(ctx.iteration)

        hooks = AgentHooks(
            on_iteration_start=[on_iter_start],
            on_iteration_end=[on_iter_end],
        )
        ctx = IterationContext(iteration=3, confidence=0.6, tool_count=5)

        hooks.trigger_iteration_start(ctx)
        hooks.trigger_iteration_end(ctx)

        assert start_calls == [3]
        assert end_calls == [3]

    def test_trigger_terminate(self):
        """Test triggering terminate hooks."""
        from langchain_oci.agents.oci_agent.hooks import AgentHooks

        calls = []

        def on_terminate(reason: str, final_answer: str):
            calls.append((reason, final_answer))

        hooks = AgentHooks(on_terminate=[on_terminate])
        hooks.trigger_terminate("confidence_met", "The answer is 42.")

        assert calls == [("confidence_met", "The answer is 42.")]

    def test_multiple_hooks_triggered(self):
        """Test multiple hooks of same type are all triggered."""
        from langchain_oci.agents.oci_agent.hooks import AgentHooks, ToolHookContext

        calls = []

        def hook1(ctx: ToolHookContext):
            calls.append("hook1")

        def hook2(ctx: ToolHookContext):
            calls.append("hook2")

        hooks = AgentHooks(on_tool_start=[hook1, hook2])
        ctx = ToolHookContext(
            tool_name="test",
            tool_call_id="tc_1",
            arguments={},
            iteration=1,
        )
        hooks.trigger_tool_start(ctx)

        assert calls == ["hook1", "hook2"]

    def test_hook_exception_does_not_break_execution(self):
        """Test that hook exceptions don't break agent execution."""
        from langchain_oci.agents.oci_agent.hooks import AgentHooks, ToolHookContext

        calls = []

        def failing_hook(ctx: ToolHookContext):
            raise ValueError("Hook failed")

        def working_hook(ctx: ToolHookContext):
            calls.append("working")

        hooks = AgentHooks(on_tool_start=[failing_hook, working_hook])
        ctx = ToolHookContext(
            tool_name="test",
            tool_call_id="tc_1",
            arguments={},
            iteration=1,
        )

        # Should not raise
        hooks.trigger_tool_start(ctx)
        # Working hook should still be called
        assert calls == ["working"]

    def test_trigger_with_no_hooks(self):
        """Test trigger methods work when no hooks are set."""
        from langchain_oci.agents.oci_agent.hooks import (
            AgentHooks,
            IterationContext,
            ToolHookContext,
            ToolResultContext,
        )

        hooks = AgentHooks()

        # Should not raise
        hooks.trigger_tool_start(
            ToolHookContext(tool_name="t", tool_call_id="tc", arguments={}, iteration=1)
        )
        hooks.trigger_tool_end(
            ToolResultContext(
                tool_name="t",
                tool_call_id="tc",
                arguments={},
                result="r",
                success=True,
                error=None,
                duration_ms=1.0,
                iteration=1,
            )
        )
        hooks.trigger_iteration_start(
            IterationContext(iteration=1, confidence=0.5, tool_count=0)
        )
        hooks.trigger_iteration_end(
            IterationContext(iteration=1, confidence=0.5, tool_count=0)
        )
        hooks.trigger_terminate("done", "answer")

    def test_create_logging_hooks(self):
        """Test create_logging_hooks returns valid hooks."""
        from langchain_oci.agents.oci_agent.hooks import create_logging_hooks

        hooks = create_logging_hooks()

        assert hooks.on_tool_start is not None
        assert len(hooks.on_tool_start) == 1
        assert hooks.on_tool_end is not None
        assert len(hooks.on_tool_end) == 1
        assert hooks.on_iteration_start is not None
        assert hooks.on_iteration_end is not None
        assert hooks.on_terminate is not None

    def test_create_metrics_hooks(self):
        """Test create_metrics_hooks returns hooks and metrics dict."""
        from langchain_oci.agents.oci_agent.hooks import (
            create_metrics_hooks,
        )

        hooks, metrics = create_metrics_hooks()

        assert hooks.on_tool_end is not None
        assert hooks.on_iteration_end is not None
        assert hooks.on_terminate is not None

        assert metrics["total_tool_calls"] == 0
        assert metrics["tool_durations_ms"] == []
        assert metrics["tool_errors"] == 0
        assert metrics["iterations"] == 0
        assert metrics["termination_reason"] is None

    def test_metrics_hooks_track_tool_calls(self):
        """Test metrics hooks track tool execution."""
        from langchain_oci.agents.oci_agent.hooks import (
            ToolResultContext,
            create_metrics_hooks,
        )

        hooks, metrics = create_metrics_hooks()

        # Simulate tool completion
        ctx = ToolResultContext(
            tool_name="search",
            tool_call_id="tc_1",
            arguments={},
            result="data",
            success=True,
            error=None,
            duration_ms=150.0,
            iteration=1,
        )
        hooks.trigger_tool_end(ctx)

        assert metrics["total_tool_calls"] == 1
        assert metrics["tool_durations_ms"] == [150.0]
        assert metrics["tool_errors"] == 0

    def test_metrics_hooks_track_errors(self):
        """Test metrics hooks track tool errors."""
        from langchain_oci.agents.oci_agent.hooks import (
            ToolResultContext,
            create_metrics_hooks,
        )

        hooks, metrics = create_metrics_hooks()

        ctx = ToolResultContext(
            tool_name="api",
            tool_call_id="tc_1",
            arguments={},
            result="",
            success=False,
            error="Failed",
            duration_ms=50.0,
            iteration=1,
        )
        hooks.trigger_tool_end(ctx)

        assert metrics["total_tool_calls"] == 1
        assert metrics["tool_errors"] == 1


# =============================================================================
# Checkpoint Tests
# =============================================================================


class TestCheckpoint:
    """Tests for Checkpoint model."""

    def test_checkpoint_creation(self):
        """Test creating a checkpoint."""
        from langchain_oci.agents.oci_agent.checkpoint import Checkpoint

        messages = [HumanMessage(content="Hello"), AIMessage(content="Hi there")]
        checkpoint = Checkpoint.create(
            thread_id="thread-1",
            iteration=2,
            messages=messages,
            metadata={"extra": "data"},
        )

        assert checkpoint.thread_id == "thread-1"
        assert checkpoint.iteration == 2
        assert len(checkpoint.messages) == 2
        assert checkpoint.metadata["extra"] == "data"
        assert "created_at" in checkpoint.metadata
        assert checkpoint.id.startswith("ckpt_")

    def test_checkpoint_get_messages(self):
        """Test deserializing messages from checkpoint."""
        from langchain_oci.agents.oci_agent.checkpoint import Checkpoint

        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi"),
            ToolMessage(content="result", tool_call_id="tc_1"),
        ]
        checkpoint = Checkpoint.create(thread_id="t1", iteration=1, messages=messages)

        restored = checkpoint.get_messages()
        assert len(restored) == 3
        assert isinstance(restored[0], HumanMessage)
        assert isinstance(restored[1], AIMessage)
        assert isinstance(restored[2], ToolMessage)
        assert restored[0].content == "Hello"

    def test_checkpoint_to_dict_from_dict(self):
        """Test checkpoint serialization round-trip."""
        from langchain_oci.agents.oci_agent.checkpoint import Checkpoint

        messages = [HumanMessage(content="Test")]
        original = Checkpoint.create(thread_id="t1", iteration=1, messages=messages)

        data = original.to_dict()
        restored = Checkpoint.from_dict(data)

        assert restored.id == original.id
        assert restored.thread_id == original.thread_id
        assert restored.iteration == original.iteration
        assert restored.messages == original.messages

    def test_checkpoint_is_frozen(self):
        """Test that checkpoint is immutable."""
        from langchain_oci.agents.oci_agent.checkpoint import Checkpoint

        checkpoint = Checkpoint.create(thread_id="t1", iteration=1, messages=[])

        with pytest.raises(Exception):  # frozen dataclass
            checkpoint.thread_id = "new"


class TestMemoryCheckpointer:
    """Tests for MemoryCheckpointer."""

    def test_put_and_get(self):
        """Test saving and retrieving checkpoints."""
        from langchain_oci.agents.oci_agent.checkpoint import (
            Checkpoint,
            MemoryCheckpointer,
        )

        checkpointer = MemoryCheckpointer()
        checkpoint = Checkpoint.create(
            thread_id="t1", iteration=1, messages=[HumanMessage(content="Hi")]
        )

        checkpointer.put(checkpoint)
        retrieved = checkpointer.get("t1")

        assert retrieved is not None
        assert retrieved.id == checkpoint.id
        assert retrieved.thread_id == "t1"

    def test_get_returns_latest(self):
        """Test that get returns the most recent checkpoint."""
        from langchain_oci.agents.oci_agent.checkpoint import (
            Checkpoint,
            MemoryCheckpointer,
        )

        checkpointer = MemoryCheckpointer()

        # Add two checkpoints
        ckpt1 = Checkpoint.create(thread_id="t1", iteration=1, messages=[])
        ckpt2 = Checkpoint.create(thread_id="t1", iteration=2, messages=[])

        checkpointer.put(ckpt1)
        checkpointer.put(ckpt2)

        latest = checkpointer.get("t1")
        assert latest.id == ckpt2.id
        assert latest.iteration == 2

    def test_get_nonexistent_thread(self):
        """Test getting checkpoint for nonexistent thread."""
        from langchain_oci.agents.oci_agent.checkpoint import MemoryCheckpointer

        checkpointer = MemoryCheckpointer()
        result = checkpointer.get("nonexistent")
        assert result is None

    def test_list_checkpoints(self):
        """Test listing all checkpoints for a thread."""
        from langchain_oci.agents.oci_agent.checkpoint import (
            Checkpoint,
            MemoryCheckpointer,
        )

        checkpointer = MemoryCheckpointer()

        for i in range(3):
            ckpt = Checkpoint.create(thread_id="t1", iteration=i, messages=[])
            checkpointer.put(ckpt)

        checkpoints = list(checkpointer.list("t1"))
        assert len(checkpoints) == 3
        assert [c.iteration for c in checkpoints] == [0, 1, 2]

    def test_delete_thread(self):
        """Test deleting all checkpoints for a thread."""
        from langchain_oci.agents.oci_agent.checkpoint import (
            Checkpoint,
            MemoryCheckpointer,
        )

        checkpointer = MemoryCheckpointer()

        for i in range(3):
            ckpt = Checkpoint.create(thread_id="t1", iteration=i, messages=[])
            checkpointer.put(ckpt)

        deleted = checkpointer.delete("t1")
        assert deleted == 3
        assert checkpointer.get("t1") is None

    def test_get_by_id(self):
        """Test getting checkpoint by ID."""
        from langchain_oci.agents.oci_agent.checkpoint import (
            Checkpoint,
            MemoryCheckpointer,
        )

        checkpointer = MemoryCheckpointer()
        checkpoint = Checkpoint.create(thread_id="t1", iteration=1, messages=[])
        checkpointer.put(checkpoint)

        retrieved = checkpointer.get_by_id(checkpoint.id)
        assert retrieved is not None
        assert retrieved.id == checkpoint.id

    def test_checkpoint_count(self):
        """Test counting checkpoints."""
        from langchain_oci.agents.oci_agent.checkpoint import (
            Checkpoint,
            MemoryCheckpointer,
        )

        checkpointer = MemoryCheckpointer()
        assert checkpointer.checkpoint_count == 0

        for thread in ["t1", "t2"]:
            for i in range(2):
                ckpt = Checkpoint.create(thread_id=thread, iteration=i, messages=[])
                checkpointer.put(ckpt)

        assert checkpointer.checkpoint_count == 4
        assert checkpointer.thread_count == 2

    def test_clear(self):
        """Test clearing all checkpoints."""
        from langchain_oci.agents.oci_agent.checkpoint import (
            Checkpoint,
            MemoryCheckpointer,
        )

        checkpointer = MemoryCheckpointer()
        ckpt = Checkpoint.create(thread_id="t1", iteration=1, messages=[])
        checkpointer.put(ckpt)

        checkpointer.clear()
        assert checkpointer.checkpoint_count == 0
        assert checkpointer.thread_count == 0


class TestLangGraphCheckpointerAdapter:
    """Tests for LangGraph checkpointer adapter."""

    def test_wrap_native_checkpointer(self):
        """Test wrapping a native checkpointer (no-op)."""
        from langchain_oci.agents.oci_agent.checkpoint import (
            MemoryCheckpointer,
            wrap_checkpointer,
        )

        native = MemoryCheckpointer()
        wrapped = wrap_checkpointer(native)
        assert wrapped is native

    def test_wrap_langgraph_checkpointer(self):
        """Test wrapping a LangGraph checkpointer."""
        from langchain_oci.agents.oci_agent.checkpoint import (
            LangGraphCheckpointerAdapter,
            wrap_checkpointer,
        )

        # Create a mock LangGraph checkpointer
        mock_lg = MagicMock()
        mock_lg.get = MagicMock(return_value=None)
        mock_lg.put = MagicMock()

        wrapped = wrap_checkpointer(mock_lg)
        assert isinstance(wrapped, LangGraphCheckpointerAdapter)

    def test_wrap_invalid_type(self):
        """Test wrapping an invalid type raises error."""
        from langchain_oci.agents.oci_agent.checkpoint import wrap_checkpointer

        with pytest.raises(TypeError):
            wrap_checkpointer("not a checkpointer")

    def test_adapter_get_returns_none_for_empty(self):
        """Test adapter returns None for empty thread."""
        from langchain_oci.agents.oci_agent.checkpoint import (
            LangGraphCheckpointerAdapter,
        )

        mock_lg = MagicMock()
        mock_lg.get = MagicMock(return_value=None)

        adapter = LangGraphCheckpointerAdapter(mock_lg)
        result = adapter.get("thread-1")

        assert result is None
        mock_lg.get.assert_called_once()


# =============================================================================
# Agent Method Tests (async, batch, LangGraph node)
# =============================================================================


class TestAgentAsyncMethods:
    """Tests for async agent methods."""

    def test_ainvoke_exists(self):
        """Test ainvoke method exists on agent."""
        from langchain_oci.agents.oci_agent.agent import OCIGenAIAgent

        assert hasattr(OCIGenAIAgent, "ainvoke")
        assert callable(getattr(OCIGenAIAgent, "ainvoke"))

    def test_astream_exists(self):
        """Test astream method exists on agent."""
        from langchain_oci.agents.oci_agent.agent import OCIGenAIAgent

        assert hasattr(OCIGenAIAgent, "astream")
        assert callable(getattr(OCIGenAIAgent, "astream"))

    def test_abatch_exists(self):
        """Test abatch method exists on agent."""
        from langchain_oci.agents.oci_agent.agent import OCIGenAIAgent

        assert hasattr(OCIGenAIAgent, "abatch")
        assert callable(getattr(OCIGenAIAgent, "abatch"))


class TestAgentBatchMethod:
    """Tests for batch method."""

    def test_batch_exists(self):
        """Test batch method exists on agent."""
        from langchain_oci.agents.oci_agent.agent import OCIGenAIAgent

        assert hasattr(OCIGenAIAgent, "batch")
        assert callable(getattr(OCIGenAIAgent, "batch"))


class TestAgentLangGraphNode:
    """Tests for LangGraph node compatibility."""

    def test_as_node_exists(self):
        """Test as_node method exists on agent."""
        from langchain_oci.agents.oci_agent.agent import OCIGenAIAgent

        assert hasattr(OCIGenAIAgent, "as_node")
        assert callable(getattr(OCIGenAIAgent, "as_node"))

    def test_call_exists(self):
        """Test __call__ method exists on agent."""
        from langchain_oci.agents.oci_agent.agent import OCIGenAIAgent

        assert hasattr(OCIGenAIAgent, "__call__")
        assert callable(getattr(OCIGenAIAgent, "__call__"))

    def test_as_node_returns_callable(self):
        """Test as_node returns a callable."""
        from langchain_oci.agents.oci_agent.agent import OCIGenAIAgent

        # Create mock agent
        with patch.object(OCIGenAIAgent, "__init__", lambda x: None):
            agent = OCIGenAIAgent.__new__(OCIGenAIAgent)
            agent._tools = {}
            agent.model_id = "test"
            agent.max_iterations = 5
            agent.enable_reflexion = True

            # Mock invoke
            agent.invoke = MagicMock(
                return_value=AgentResult(
                    messages=[],
                    final_answer="test answer",
                    termination_reason="no_tools",
                    reasoning_steps=[],
                    total_iterations=1,
                    total_tool_calls=0,
                )
            )

            node_fn = agent.as_node()
            assert callable(node_fn)

            # Test node function returns dict
            result = node_fn({"input": "test"})
            assert isinstance(result, dict)
            assert "final_answer" in result
            assert "messages" in result


class TestAsyncToolExecution:
    """Tests for async tool execution functionality."""

    def test_aexecute_tool_exists(self):
        """Test _aexecute_tool method exists on agent."""
        from langchain_oci.agents.oci_agent.agent import OCIGenAIAgent

        assert hasattr(OCIGenAIAgent, "_aexecute_tool")

    @pytest.mark.asyncio
    async def test_aexecute_tool_with_sync_tool(self):
        """Test async tool execution falls back to sync for non-async tools."""
        from langchain_core.tools import tool

        from langchain_oci.agents.oci_agent.agent import OCIGenAIAgent

        @tool
        def sync_tool(x: str) -> str:
            """A sync tool."""
            return f"result: {x}"

        # Create mock agent
        with patch.object(OCIGenAIAgent, "__init__", lambda self: None):
            agent = OCIGenAIAgent.__new__(OCIGenAIAgent)
            agent._tools = {"sync_tool": sync_tool}

            # Execute tool async
            execution = await agent._aexecute_tool("sync_tool", "tc_123", {"x": "test"})

            assert execution.success is True
            assert execution.result == "result: test"
            assert execution.tool_name == "sync_tool"
            assert execution.tool_call_id == "tc_123"
            assert execution.duration_ms > 0

    @pytest.mark.asyncio
    async def test_aexecute_tool_with_async_tool(self):
        """Test async tool execution uses native ainvoke when available."""
        import asyncio

        from langchain_oci.agents.oci_agent.agent import OCIGenAIAgent

        # Create a mock tool with ainvoke
        async_tool = MagicMock()
        async_tool.name = "async_tool"

        async def mock_ainvoke(args):
            await asyncio.sleep(0.001)  # Simulate async work
            return f"async result: {args.get('x', '')}"

        async_tool.ainvoke = mock_ainvoke

        # Create mock agent
        with patch.object(OCIGenAIAgent, "__init__", lambda self: None):
            agent = OCIGenAIAgent.__new__(OCIGenAIAgent)
            agent._tools = {"async_tool": async_tool}

            # Execute tool async
            execution = await agent._aexecute_tool(
                "async_tool", "tc_456", {"x": "test"}
            )

            assert execution.success is True
            assert execution.result == "async result: test"
            assert execution.tool_name == "async_tool"

    @pytest.mark.asyncio
    async def test_aexecute_tool_unknown_tool(self):
        """Test async tool execution handles unknown tools."""
        from langchain_oci.agents.oci_agent.agent import OCIGenAIAgent

        with patch.object(OCIGenAIAgent, "__init__", lambda self: None):
            agent = OCIGenAIAgent.__new__(OCIGenAIAgent)
            agent._tools = {}

            execution = await agent._aexecute_tool(
                "unknown_tool", "tc_789", {"x": "test"}
            )

            assert execution.success is False
            assert "Unknown tool" in execution.error

    @pytest.mark.asyncio
    async def test_aexecute_tool_handles_error(self):
        """Test async tool execution handles errors gracefully."""
        from langchain_oci.agents.oci_agent.agent import OCIGenAIAgent

        # Create a tool that raises an error
        error_tool = MagicMock()
        error_tool.name = "error_tool"
        error_tool.invoke = MagicMock(side_effect=RuntimeError("Tool failed"))

        with patch.object(OCIGenAIAgent, "__init__", lambda self: None):
            agent = OCIGenAIAgent.__new__(OCIGenAIAgent)
            agent._tools = {"error_tool": error_tool}

            execution = await agent._aexecute_tool("error_tool", "tc_error", {})

            assert execution.success is False
            assert "Tool failed" in execution.error
            assert execution.duration_ms >= 0
