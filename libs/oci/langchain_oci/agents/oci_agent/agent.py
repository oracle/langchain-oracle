# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""OCIGenAIAgent - Sophisticated agentic loop for OCI Generative AI.

Provides a full agentic loop with:
- Immutable state management
- Typed event streaming
- Reflexion (confidence tracking + loop detection)
- Confidence signal detection (heuristic early exit)
- Message compression (context management)
- Hooks system (pre/post callbacks)
- Conversation history support
- 5 termination conditions
- LangChain Runnable interface for LCEL composability
- LangGraph node compatibility

Example:
    from langchain_oci import OCIGenAIAgent
    from langchain_core.tools import tool

    @tool
    def search(query: str) -> str:
        '''Search for information.'''
        return f"Results for: {query}"

    agent = OCIGenAIAgent(
        model_id="meta.llama-4-scout-17b-16e-instruct",
        tools=[search],
        compartment_id="ocid1.compartment...",
    )

    # Standalone usage
    result = agent.invoke("What is the capital of France?")
    print(result.final_answer)

    # Stream typed events
    for event in agent.stream("What's the weather?"):
        if isinstance(event, ThinkEvent):
            print(f"Thinking: {event.thought}")
        elif isinstance(event, ToolCompleteEvent):
            print(f"Tool result: {event.result}")

    # Use as LangGraph node
    from langgraph.graph import StateGraph, START, END
    graph = StateGraph(dict)
    graph.add_node("agent", agent)
    graph.add_edge(START, "agent")
    graph.add_edge("agent", END)
"""

from __future__ import annotations

import time
import uuid
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterator,
    Sequence,
    Union,
)

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool
from langchain_core.tools import tool as tool_decorator

from langchain_oci.agents.oci_agent.compression import (
    CompressionConfig,
    CompressionStrategy,
    compress_messages,
)
from langchain_oci.agents.oci_agent.confidence import (
    ConfidenceSignal,
    compute_accumulated_confidence,
    detect_confidence_signals,
    should_early_exit,
)
from langchain_oci.agents.oci_agent.events import (
    AgentEvent,
    ReflectEvent,
    TerminateEvent,
    ThinkEvent,
    ToolCompleteEvent,
    ToolStartEvent,
)
from langchain_oci.agents.oci_agent.hooks import (
    AgentHooks,
    IterationContext,
    ToolHookContext,
    ToolResultContext,
)
from langchain_oci.agents.oci_agent.reflexion import (
    AssessmentCategory,
    Reflector,
)
from langchain_oci.agents.oci_agent.state import (
    AgentResult,
    AgentState,
    ReasoningStep,
    ToolExecution,
)
from langchain_oci.agents.oci_agent.termination import (
    TerminationReason,
    check_termination,
)
from langchain_oci.chat_models.oci_generative_ai import ChatOCIGenAI
from langchain_oci.common.auth import OCIAuthType

if TYPE_CHECKING:
    pass


class OCIGenAIAgent(Runnable[dict, AgentResult]):
    """OCI Generative AI Agent with reflexion and typed event streaming.

    Implements LangChain Runnable for LCEL composability.
    Can be used as a LangGraph node.

    Attributes:
        model_id: OCI model identifier.
        tools: Dictionary of available tools.
        system_prompt: System message for the agent.
        max_iterations: Maximum iterations before termination.
        confidence_threshold: Confidence level to consider task complete.
        terminal_tools: Tool names that signal explicit completion.
        enable_reflexion: Whether to enable self-evaluation.
        enable_confidence_signals: Whether to detect heuristic confidence patterns.
        enable_compression: Whether to compress messages to prevent overflow.
        hooks: Lifecycle callbacks for tool/iteration events.
        loop_threshold: Repeated tool calls to consider a loop.
    """

    def __init__(
        self,
        model_id: str,
        tools: Sequence[Union[BaseTool, Callable[..., Any]]],
        *,
        # OCI auth parameters
        compartment_id: str | None = None,
        service_endpoint: str | None = None,
        auth_type: Union[str, OCIAuthType] = OCIAuthType.API_KEY,
        auth_profile: str = "DEFAULT",
        auth_file_location: str = "~/.oci/config",
        # Agent configuration
        system_prompt: str | None = None,
        max_iterations: int = 10,
        confidence_threshold: float = 0.8,
        terminal_tools: list[str] | None = None,
        enable_reflexion: bool = True,
        loop_threshold: int = 3,
        # New features
        enable_confidence_signals: bool = True,
        min_iterations_for_early_exit: int = 2,
        enable_compression: bool = True,
        max_messages: int = 20,
        hooks: AgentHooks | None = None,
        # Model parameters
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        """Initialize OCIGenAIAgent.

        Args:
            model_id: OCI model identifier
                (e.g., "meta.llama-4-scout-17b-16e-instruct").
            tools: Tools available to the agent.
            compartment_id: OCI compartment OCID.
            service_endpoint: OCI service endpoint URL.
            auth_type: Authentication type (API_KEY, SECURITY_TOKEN, etc.).
            auth_profile: OCI config profile name.
            auth_file_location: Path to OCI config file.
            system_prompt: System message for the agent.
            max_iterations: Maximum iterations before forced termination.
            confidence_threshold: Confidence level to consider task complete.
            terminal_tools: Tool names that signal explicit completion.
            enable_reflexion: Whether to enable self-evaluation.
            loop_threshold: Repeated tool calls to consider a loop.
            enable_confidence_signals: Detect heuristic confidence patterns.
            min_iterations_for_early_exit: Min iterations before early exit.
            enable_compression: Compress messages to prevent context overflow.
            max_messages: Maximum messages before compression.
            hooks: Lifecycle callbacks for tool/iteration events.
            temperature: Model temperature.
            max_tokens: Maximum tokens to generate.
        """
        self.model_id = model_id
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.confidence_threshold = confidence_threshold
        self.terminal_tools = set(terminal_tools or ["done", "submit", "finish"])
        self.enable_reflexion = enable_reflexion
        self.loop_threshold = loop_threshold

        # New features
        self.enable_confidence_signals = enable_confidence_signals
        self.min_iterations_for_early_exit = min_iterations_for_early_exit
        self.enable_compression = enable_compression
        self._compression_config = CompressionConfig(
            strategy=CompressionStrategy.SMART_TRIM if enable_compression
            else CompressionStrategy.NONE,
            max_messages=max_messages,
        )
        self._hooks = hooks or AgentHooks()
        self._confidence_signals: list[ConfidenceSignal] = []

        # Build model kwargs
        model_kwargs: dict[str, Any] = {}
        if temperature is not None:
            model_kwargs["temperature"] = temperature
        if max_tokens is not None:
            model_kwargs["max_tokens"] = max_tokens

        # Create ChatOCIGenAI instance
        self._llm = ChatOCIGenAI(
            model_id=model_id,
            compartment_id=compartment_id,
            service_endpoint=service_endpoint,
            auth_type=auth_type,
            auth_profile=auth_profile,
            auth_file_location=auth_file_location,
            model_kwargs=model_kwargs if model_kwargs else None,
        )

        # Convert and store tools
        self._tools = self._convert_tools(tools)
        self._llm_with_tools = self._llm.bind_tools(list(self._tools.values()))

        # Create reflector
        self._reflector = Reflector(
            loop_threshold=loop_threshold,
        )

    def _convert_tools(
        self,
        tools: Sequence[Union[BaseTool, Callable[..., Any]]],
    ) -> dict[str, BaseTool]:
        """Convert tools to BaseTool instances.

        Args:
            tools: Sequence of tools (BaseTool or callable).

        Returns:
            Dictionary mapping tool names to BaseTool instances.
        """
        result: dict[str, BaseTool] = {}
        for t in tools:
            if isinstance(t, BaseTool):
                result[t.name] = t
            elif callable(t):
                # Convert callable to tool using decorator
                converted = tool_decorator(t)
                result[converted.name] = converted
            else:
                raise TypeError(f"Tool must be BaseTool or callable, got {type(t)}")
        return result

    def _init_state(
        self,
        input_data: Union[str, dict],
        message_history: list[dict] | None = None,
    ) -> AgentState:
        """Initialize agent state from input.

        Args:
            input_data: User query string or dict with messages.
            message_history: Optional conversation history as list of
                {"role": "user"|"assistant", "content": "..."} dicts.

        Returns:
            Initial AgentState.
        """
        messages: list[BaseMessage] = []

        # Add system prompt if configured
        if self.system_prompt:
            messages.append(SystemMessage(content=self.system_prompt))

        # Add conversation history if provided
        if message_history:
            for msg in message_history:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role in ("human", "user"):
                    messages.append(HumanMessage(content=content))
                elif role in ("ai", "assistant"):
                    messages.append(AIMessage(content=content))
                elif role == "system":
                    messages.append(SystemMessage(content=content))

        # Handle input format
        if isinstance(input_data, str):
            messages.append(HumanMessage(content=input_data))
        elif isinstance(input_data, dict):
            if "messages" in input_data:
                # Extract messages from dict
                for msg in input_data["messages"]:
                    if isinstance(msg, BaseMessage):
                        messages.append(msg)
                    elif isinstance(msg, dict):
                        # Convert dict to message
                        role = msg.get("role", "human")
                        content = msg.get("content", "")
                        if role in ("human", "user"):
                            messages.append(HumanMessage(content=content))
                        elif role in ("ai", "assistant"):
                            messages.append(AIMessage(content=content))
                        elif role == "system":
                            messages.append(SystemMessage(content=content))
            elif "input" in input_data:
                messages.append(HumanMessage(content=input_data["input"]))
            elif "query" in input_data:
                messages.append(HumanMessage(content=input_data["query"]))
            else:
                # Assume the dict itself is the query
                messages.append(HumanMessage(content=str(input_data)))
        else:
            raise TypeError(f"Input must be str or dict, got {type(input_data)}")

        # Reset confidence signals for new conversation
        self._confidence_signals = []

        return AgentState(messages=tuple(messages))

    def _execute_tool(
        self,
        tool_name: str,
        tool_call_id: str,
        arguments: dict[str, Any],
    ) -> ToolExecution:
        """Execute a single tool call.

        Args:
            tool_name: Name of the tool to execute.
            tool_call_id: Unique identifier for this call.
            arguments: Arguments to pass to the tool.

        Returns:
            ToolExecution record.
        """
        start_time = time.perf_counter()

        tool = self._tools.get(tool_name)
        if tool is None:
            return ToolExecution(
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                arguments=arguments,
                result="",
                success=False,
                error=f"Unknown tool: {tool_name}",
                duration_ms=0.0,
            )

        try:
            result = tool.invoke(arguments)
            duration_ms = (time.perf_counter() - start_time) * 1000

            return ToolExecution(
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                arguments=arguments,
                result=str(result),
                success=True,
                duration_ms=duration_ms,
            )
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            return ToolExecution(
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                arguments=arguments,
                result="",
                success=False,
                error=str(e),
                duration_ms=duration_ms,
            )

    def _get_final_answer(self, state: AgentState) -> str:
        """Extract final answer from state.

        Args:
            state: Current agent state.

        Returns:
            Final answer string.
        """
        # Look for the last AI message content
        for msg in reversed(state.messages):
            if isinstance(msg, AIMessage) and msg.content:
                return str(msg.content)
        return ""

    # Runnable interface

    @property
    def InputType(self) -> type:
        """Input type for LangGraph compatibility."""
        return dict

    @property
    def OutputType(self) -> type:
        """Output type for LangGraph compatibility."""
        return AgentResult

    def invoke(
        self,
        input_data: Union[str, dict],
        config: RunnableConfig | None = None,
        *,
        message_history: list[dict] | None = None,
    ) -> AgentResult:
        """Run agent to completion.

        Args:
            input_data: User query string or dict with messages.
            config: Optional runnable configuration.
            message_history: Optional conversation history as list of
                {"role": "user"|"assistant", "content": "..."} dicts.

        Returns:
            AgentResult with final answer and execution details.
        """
        # Consume the stream to get the final result
        final_event = None
        for event in self.stream(input_data, config, message_history=message_history):
            if isinstance(event, TerminateEvent):
                final_event = event
                break

        if final_event is None:
            # Should not happen, but handle gracefully
            return AgentResult(
                messages=[],
                final_answer="",
                termination_reason="unknown",
                reasoning_steps=[],
                total_iterations=0,
                total_tool_calls=0,
            )

        # We need to reconstruct the result from the stream
        # For invoke, we'll run the full loop and build the result directly
        state = self._init_state(input_data, message_history)
        termination_reason = None

        while True:
            # Check termination before iteration
            reason = check_termination(
                state,
                max_iterations=self.max_iterations,
                confidence_threshold=self.confidence_threshold,
                terminal_tools=self.terminal_tools,
                loop_threshold=self.loop_threshold,
                check_confidence=self.enable_reflexion,
            )
            if reason:
                termination_reason = reason
                break

            # Call the model
            response = self._llm_with_tools.invoke(list(state.messages), config)
            state = state.with_message(response)

            # Get tool calls from response
            tool_calls = getattr(response, "tool_calls", []) or []

            # Check no_tools termination
            if not tool_calls:
                termination_reason = TerminationReason.NO_TOOLS
                break

            # Execute tools
            tool_executions: list[ToolExecution] = []
            last_tool_calls: list[str] = []

            for tool_call in tool_calls:
                tool_name = tool_call.get("name", "")
                tool_call_id = tool_call.get("id", str(uuid.uuid4()))
                arguments = tool_call.get("args", {})

                execution = self._execute_tool(tool_name, tool_call_id, arguments)
                tool_executions.append(execution)
                last_tool_calls.append(tool_name)

                # Record tool in history
                state = state.with_tool_call(tool_name)

                # Add tool result as message
                content = (
                    execution.result
                    if execution.success
                    else f"Error: {execution.error}"
                )
                tool_message = ToolMessage(
                    content=content,
                    tool_call_id=tool_call_id,
                )
                state = state.with_message(tool_message)

            # Check terminal tool
            reason = check_termination(
                state,
                max_iterations=self.max_iterations,
                confidence_threshold=self.confidence_threshold,
                terminal_tools=self.terminal_tools,
                last_tool_calls=last_tool_calls,
                loop_threshold=self.loop_threshold,
                check_confidence=False,  # Will check after reflexion
            )
            if reason:
                termination_reason = reason
                break

            # Reflexion
            if self.enable_reflexion:
                reflection = self._reflector.reflect(state, tool_executions)
                state = state.adjust_confidence(reflection.confidence_delta)

                if reflection.assessment == AssessmentCategory.LOOP_DETECTED:
                    termination_reason = TerminationReason.TOOL_LOOP
                    break

                # Check confidence threshold after reflexion
                if state.confidence >= self.confidence_threshold:
                    termination_reason = TerminationReason.CONFIDENCE_MET
                    break

            # Record reasoning step
            assessment = (
                reflection.assessment if self.enable_reflexion else "on_track"
            )
            step = ReasoningStep(
                iteration=state.iteration,
                thought=str(response.content) if response.content else "",
                tool_executions=tuple(tool_executions),
                confidence=state.confidence,
                assessment=assessment,
            )
            state = state.with_reasoning_step(step)
            state = state.increment_iteration()

        return AgentResult.from_state(
            state,
            final_answer=self._get_final_answer(state),
            termination_reason=termination_reason or "unknown",
        )

    def stream(
        self,
        input_data: Union[str, dict],
        config: RunnableConfig | None = None,
        *,
        message_history: list[dict] | None = None,
    ) -> Iterator[AgentEvent]:
        """Stream agent events during execution.

        Yields typed events as the agent processes the query.

        Args:
            input_data: User query string or dict with messages.
            config: Optional runnable configuration.
            message_history: Optional conversation history as list of
                {"role": "user"|"assistant", "content": "..."} dicts.

        Yields:
            AgentEvent instances (ThinkEvent, ToolStartEvent, etc.).
        """
        state = self._init_state(input_data, message_history)
        termination_reason = None
        total_tool_calls = 0

        while True:
            # Trigger iteration start hook
            self._hooks.trigger_iteration_start(
                IterationContext(
                    iteration=state.iteration,
                    confidence=state.confidence,
                    tool_count=total_tool_calls,
                )
            )

            # Check termination before iteration
            reason = check_termination(
                state,
                max_iterations=self.max_iterations,
                confidence_threshold=self.confidence_threshold,
                terminal_tools=self.terminal_tools,
                loop_threshold=self.loop_threshold,
                check_confidence=self.enable_reflexion,
            )
            if reason:
                termination_reason = reason
                break

            # Apply message compression if enabled
            if self.enable_compression:
                compression_result = compress_messages(
                    state.messages, self._compression_config
                )
                if compression_result.dropped_count > 0:
                    state = state.model_copy(
                        update={"messages": compression_result.messages}
                    )

            # Call the model
            response = self._llm_with_tools.invoke(list(state.messages), config)
            state = state.with_message(response)

            # Get tool calls from response
            tool_calls = getattr(response, "tool_calls", []) or []

            # Detect confidence signals in response
            if self.enable_confidence_signals and response.content:
                new_signals = detect_confidence_signals(
                    str(response.content), state.iteration
                )
                self._confidence_signals.extend(new_signals)

                # Check for early exit based on confidence signals
                accumulated = compute_accumulated_confidence(self._confidence_signals)
                if should_early_exit(
                    accumulated,
                    state.iteration,
                    min_iterations=self.min_iterations_for_early_exit,
                ):
                    state = state.with_confidence(accumulated)
                    termination_reason = TerminationReason.CONFIDENCE_MET
                    # Emit think event before breaking
                    yield ThinkEvent(
                        iteration=state.iteration,
                        thought=str(response.content) if response.content else "",
                        tool_calls_planned=len(tool_calls),
                    )
                    break

            # Emit think event
            yield ThinkEvent(
                iteration=state.iteration,
                thought=str(response.content) if response.content else "",
                tool_calls_planned=len(tool_calls),
            )

            # Check no_tools termination
            if not tool_calls:
                termination_reason = TerminationReason.NO_TOOLS
                break

            # Execute tools
            tool_executions: list[ToolExecution] = []
            last_tool_calls: list[str] = []

            for tool_call in tool_calls:
                tool_name = tool_call.get("name", "")
                tool_call_id = tool_call.get("id", str(uuid.uuid4()))
                arguments = tool_call.get("args", {})

                # Trigger tool start hook
                hook_ctx = ToolHookContext(
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                    arguments=arguments,
                    iteration=state.iteration,
                )
                self._hooks.trigger_tool_start(hook_ctx)

                # Emit tool start event
                yield ToolStartEvent(
                    iteration=state.iteration,
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                    arguments=arguments,
                )

                execution = self._execute_tool(tool_name, tool_call_id, arguments)
                tool_executions.append(execution)
                last_tool_calls.append(tool_name)
                total_tool_calls += 1

                # Trigger tool end hook
                result_ctx = ToolResultContext(
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                    arguments=arguments,
                    result=execution.result,
                    success=execution.success,
                    error=execution.error,
                    duration_ms=execution.duration_ms,
                    iteration=state.iteration,
                )
                self._hooks.trigger_tool_end(result_ctx)

                # Emit tool complete event
                yield ToolCompleteEvent(
                    iteration=state.iteration,
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                    result=execution.result,
                    success=execution.success,
                    error=execution.error,
                    duration_ms=execution.duration_ms,
                )

                # Record tool in history
                state = state.with_tool_call(tool_name)

                # Add tool result as message
                content = (
                    execution.result
                    if execution.success
                    else f"Error: {execution.error}"
                )
                tool_message = ToolMessage(
                    content=content,
                    tool_call_id=tool_call_id,
                )
                state = state.with_message(tool_message)

            # Check terminal tool
            reason = check_termination(
                state,
                max_iterations=self.max_iterations,
                confidence_threshold=self.confidence_threshold,
                terminal_tools=self.terminal_tools,
                last_tool_calls=last_tool_calls,
                loop_threshold=self.loop_threshold,
                check_confidence=False,  # Will check after reflexion
            )
            if reason:
                termination_reason = reason
                break

            # Reflexion
            reflection_assessment = "on_track"
            confidence_delta = 0.0

            if self.enable_reflexion:
                reflection = self._reflector.reflect(state, tool_executions)
                state = state.adjust_confidence(reflection.confidence_delta)
                reflection_assessment = reflection.assessment
                confidence_delta = reflection.confidence_delta

                # Emit reflect event
                is_loop = (
                    reflection.assessment == AssessmentCategory.LOOP_DETECTED
                )
                yield ReflectEvent(
                    iteration=state.iteration,
                    confidence=state.confidence,
                    confidence_delta=confidence_delta,
                    assessment=reflection_assessment,
                    loop_detected=is_loop,
                    guidance=reflection.guidance,
                )

                if reflection.assessment == AssessmentCategory.LOOP_DETECTED:
                    termination_reason = TerminationReason.TOOL_LOOP
                    break

                # Check confidence threshold after reflexion
                if state.confidence >= self.confidence_threshold:
                    termination_reason = TerminationReason.CONFIDENCE_MET
                    break

            # Record reasoning step
            step = ReasoningStep(
                iteration=state.iteration,
                thought=str(response.content) if response.content else "",
                tool_executions=tuple(tool_executions),
                confidence=state.confidence,
                assessment=reflection_assessment,
            )
            state = state.with_reasoning_step(step)
            state = state.increment_iteration()

            # Trigger iteration end hook
            self._hooks.trigger_iteration_end(
                IterationContext(
                    iteration=state.iteration,
                    confidence=state.confidence,
                    tool_count=total_tool_calls,
                )
            )

        # Get final answer
        final_answer = self._get_final_answer(state)

        # Trigger terminate hook
        self._hooks.trigger_terminate(
            termination_reason or "unknown",
            final_answer,
        )

        # Emit terminate event
        yield TerminateEvent(
            reason=termination_reason or "unknown",
            final_answer=final_answer,
            total_iterations=state.iteration,
            total_tool_calls=total_tool_calls,
            confidence=state.confidence,
        )

    def __repr__(self) -> str:
        """String representation."""
        tool_names = list(self._tools.keys())
        return (
            f"OCIGenAIAgent(model_id={self.model_id!r}, "
            f"tools={tool_names}, "
            f"max_iterations={self.max_iterations}, "
            f"enable_reflexion={self.enable_reflexion})"
        )
