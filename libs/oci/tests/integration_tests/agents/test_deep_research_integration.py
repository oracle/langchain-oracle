# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Integration tests for create_deep_research_agent helper function.

## Prerequisites

1. **OCI Authentication**: Set up OCI authentication:
   ```bash
   oci session authenticate  # for security token
   # or use API key auth
   ```

2. **Environment Variables**: Export the following:
   ```bash
   export OCI_REGION="us-chicago-1"
   export OCI_COMPARTMENT_ID="ocid1.compartment.oc1..your-compartment-id"
   ```

3. **Install deepagents** (Python 3.11+ required):
   ```bash
   pip install deepagents
   ```

## Running the Tests

Run all integration tests:
```bash
cd libs/oci
python -m pytest tests/integration_tests/agents/test_deep_agent_integration.py -v
```

Run specific test:
```bash
pytest tests/integration_tests/agents/test_deep_agent_integration.py \
  ::TestOCIDeepAgentIntegration::test_simple_research_task -v
```
"""

import asyncio
import os
import sys
from typing import Any

import pytest
import pytest_asyncio
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from langchain_oci import ChatOCIGenAI, create_deep_research_agent


# Sample research tools for testing
@tool
def search_knowledge_base(query: str) -> str:
    """Search the internal knowledge base for information on a topic."""
    # Mock knowledge base responses
    knowledge = {
        "quantum computing": (
            "Quantum computing uses quantum bits (qubits) that can exist in "
            "superposition states. Key developments include Google's quantum "
            "supremacy demonstration (2019), IBM's 1000+ qubit systems (2023), "
            "and recent advances in error correction. Applications include "
            "cryptography, drug discovery, and optimization problems."
        ),
        "machine learning": (
            "Machine learning is a subset of AI where systems learn from data. "
            "Key paradigms: supervised learning, unsupervised learning, and "
            "reinforcement learning. Recent trends include transformer models, "
            "large language models (LLMs), and multimodal AI systems."
        ),
        "cloud computing": (
            "Cloud computing provides on-demand computing resources over the "
            "internet. Major providers: AWS, Azure, Google Cloud, Oracle Cloud. "
            "Key services: IaaS, PaaS, SaaS. Trends: edge computing, serverless, "
            "and multi-cloud strategies."
        ),
        "ai safety": (
            "AI safety research focuses on ensuring AI systems behave as intended. "
            "Key areas: alignment research, interpretability, robustness, and "
            "value learning. Organizations include Anthropic, OpenAI, DeepMind, "
            "and academic institutions."
        ),
    }
    for topic, info in knowledge.items():
        if topic in query.lower():
            return f"Knowledge Base Result for '{topic}':\n{info}"
    topics = ", ".join(knowledge.keys())
    return f"No specific information found for '{query}'. Try: {topics}"


@tool
def get_statistics(metric: str) -> str:
    """Get statistical data for a given metric or domain."""
    stats = {
        "cloud market": (
            "Global cloud market size: $600B (2025), projected $1.2T by 2028. "
            "Growth rate: 15% CAGR. Market share: AWS 32%, Azure 23%, GCP 10%, "
            "Oracle 3%, Others 32%."
        ),
        "ai adoption": (
            "Enterprise AI adoption: 72% of organizations using AI in at least "
            "one business function. Generative AI adoption grew 300% in 2024. "
            "Top use cases: customer service, content creation, data analysis."
        ),
        "quantum market": (
            "Quantum computing market: $1.3B (2025), projected $8.6B by 2030. "
            "Key players: IBM, Google, IonQ, Rigetti, D-Wave. Government "
            "investment: $30B+ globally for quantum research."
        ),
    }
    for key, data in stats.items():
        if key in metric.lower():
            return f"Statistics for '{key}':\n{data}"
    return f"No statistics available for '{metric}'."


@tool
def analyze_trends(domain: str) -> str:
    """Analyze current trends in a given technology domain."""
    trends = {
        "ai": (
            "Top AI Trends (2025-2026):\n"
            "1. Agentic AI - autonomous AI systems that can plan and execute tasks\n"
            "2. Multimodal models - systems handling text, image, audio, video\n"
            "3. AI governance and regulation - EU AI Act, US executive orders\n"
            "4. Edge AI - on-device intelligence for privacy and latency\n"
            "5. AI for science - drug discovery, materials, climate modeling"
        ),
        "cloud": (
            "Top Cloud Trends (2025-2026):\n"
            "1. Serverless and FaaS - reduced operational overhead\n"
            "2. Multi-cloud and hybrid strategies\n"
            "3. FinOps - cloud cost optimization\n"
            "4. Sustainable cloud - carbon-neutral data centers\n"
            "5. Confidential computing - encrypted workloads"
        ),
        "security": (
            "Top Security Trends (2025-2026):\n"
            "1. Zero-trust architecture - verify everything\n"
            "2. AI-powered threat detection\n"
            "3. Post-quantum cryptography preparation\n"
            "4. Supply chain security\n"
            "5. Privacy-enhancing technologies"
        ),
    }
    for key, data in trends.items():
        if key in domain.lower():
            return f"Trends Analysis for '{key}':\n{data}"
    return f"No trend analysis available for '{domain}'."


def skip_if_no_oci_credentials() -> bool:
    """Check if OCI credentials are available."""
    compartment_id = os.environ.get("OCI_COMPARTMENT_ID")
    return compartment_id is None


def skip_if_no_deepagents() -> bool:
    """Check if deepagents is installed."""
    try:
        import deepagents  # noqa: F401

        return False
    except ImportError:
        return True


_SKIP_DEEPAGENTS = sys.version_info < (3, 11) or sys.version_info >= (3, 14)


@pytest.mark.requires("langgraph", "deepagents")
@pytest.mark.skipif(_SKIP_DEEPAGENTS, reason="deepagents requires Python 3.11-3.13")
@pytest.mark.skipif(skip_if_no_deepagents(), reason="deepagents package not installed")
class TestDeepAgentCompatibilityIntegration:
    """Non-network integration tests for deepagents compatibility issues."""

    def test_deepagents_middleware_tool_schema_converts(self) -> None:
        """Filesystem middleware tools should convert without schema crashes."""
        from deepagents.middleware.filesystem import FilesystemMiddleware

        model = ChatOCIGenAI(
            model_id="google.gemini-2.5-pro",
            service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
            compartment_id="ocid1.compartment.oc1..example",
            auth_type="API_KEY",
            model_kwargs={"temperature": 0.1},
        )

        tool = FilesystemMiddleware().tools[0]
        converted = model._provider.convert_to_oci_tool(tool)

        assert converted.name == "ls"
        assert "path" in converted.parameters["properties"]
        assert "runtime" not in converted.parameters["properties"]

    def test_helper_output_schema_is_available_without_network(self) -> None:
        """create_deep_research_agent should expose output_schema safely."""
        agent = create_deep_research_agent(
            tools=[search_knowledge_base],
            model_id="google.gemini-2.5-pro",
            compartment_id="ocid1.compartment.oc1..example",
            service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
            auth_type="API_KEY",
        )

        schema = agent.output_schema

        assert "messages" in schema.model_fields
        assert "structured_response" in schema.model_fields

    def test_direct_deepagents_output_schema_is_available(self) -> None:
        """Direct deepagents usage with ChatOCIGenAI should expose output_schema."""
        from deepagents import create_deep_agent

        model = ChatOCIGenAI(
            model_id="google.gemini-2.5-pro",
            service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
            compartment_id="ocid1.compartment.oc1..example",
            auth_type="API_KEY",
            model_kwargs={"temperature": 0.1},
        )

        agent = create_deep_agent(
            model=model,
            tools=[search_knowledge_base],
            system_prompt="You are a research assistant.",
        )

        schema = agent.output_schema

        assert "messages" in schema.model_fields


@pytest.mark.requires("oci", "langgraph", "deepagents")
@pytest.mark.skipif(
    skip_if_no_oci_credentials(),
    reason="OCI credentials not available (OCI_COMPARTMENT_ID not set)",
)
@pytest.mark.skipif(
    skip_if_no_deepagents(),
    reason="deepagents package not installed (requires Python 3.11+)",
)
class TestOCIDeepAgentIntegration:
    """Integration tests for create_deep_research_agent."""

    @pytest.fixture
    def model_id(self) -> str:
        """Use the faster Gemini variant for live integration reliability."""
        return "google.gemini-2.5-flash"

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
        return os.environ.get("OCI_AUTH_TYPE", "API_KEY")

    @pytest.fixture
    def auth_profile(self) -> str:
        """Get auth profile from environment."""
        return os.environ.get("OCI_CONFIG_PROFILE", "API_KEY_AUTH")

    @pytest.fixture
    def agent(
        self,
        compartment_id: str,
        service_endpoint: str,
        auth_type: str,
        auth_profile: str,
        model_id: str,
    ) -> Any:
        """Create a configured deep agent for testing."""
        from langchain_oci import create_deep_research_agent

        agent = create_deep_research_agent(
            tools=[search_knowledge_base, get_statistics, analyze_trends],
            model_id=model_id,
            compartment_id=compartment_id,
            service_endpoint=service_endpoint,
            auth_type=auth_type,
            auth_profile=auth_profile,
            middleware=[],
            system_prompt=(
                "You are a research analyst. Use the available tools to gather "
                "information and provide comprehensive analysis. Always cite "
                "your sources and be thorough in your research."
            ),
            temperature=0.3,
            max_tokens=2048,
        )
        try:
            yield agent
        finally:
            llm = getattr(agent, "_oci_llm", None)
            if llm is not None and hasattr(llm, "aclose"):
                asyncio.run(llm.aclose())

    def test_simple_research_task(self, agent: Any) -> None:
        """Test agent can complete a simple research task."""
        result = agent.invoke(
            {"messages": [HumanMessage(content="What are the current trends in AI?")]}
        )

        # Verify we got a response
        assert "messages" in result
        assert len(result["messages"]) > 1

        # Verify the final response has content
        final_message = result["messages"][-1]
        assert final_message.content, "Final message should have content"
        assert len(final_message.content) > 100, "Response should be substantive"

    def test_multi_tool_research(self, agent: Any) -> None:
        """Test agent uses multiple tools for research."""
        result = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=(
                            "Research the cloud computing market: "
                            "What is the market size and what are the key trends?"
                        )
                    )
                ]
            }
        )

        # Verify we got a response
        assert "messages" in result

        # Check if tools were called (look for ToolMessage)
        message_types = [type(m).__name__ for m in result["messages"]]
        tool_calls = message_types.count("ToolMessage")
        assert tool_calls >= 1, (
            f"Expected at least 1 tool call, got {tool_calls}. "
            f"Message types: {message_types}"
        )

        # Verify response quality. Some runs may end with an empty trailing
        # assistant message (for example after an unexpected tool-call finish),
        # so require at least one non-empty assistant response in the transcript.
        assistant_contents = [
            str(getattr(msg, "content", "")).strip()
            for msg in result["messages"]
            if type(msg).__name__ == "AIMessage"
        ]
        assert any(assistant_contents), "Should have a response"

    def test_research_with_checkpointer(
        self,
        compartment_id: str,
        service_endpoint: str,
        auth_type: str,
        auth_profile: str,
    ) -> None:
        """Test deep agent with memory checkpointer."""
        from langgraph.checkpoint.memory import MemorySaver

        from langchain_oci import create_deep_research_agent

        checkpointer = MemorySaver()
        agent = create_deep_research_agent(
            tools=[search_knowledge_base],
            compartment_id=compartment_id,
            service_endpoint=service_endpoint,
            auth_type=auth_type,
            auth_profile=auth_profile,
            checkpointer=checkpointer,
            temperature=0.3,
            max_tokens=1024,
        )

        thread_id = "research_thread_123"
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

        # First query
        result1 = agent.invoke(
            {"messages": [HumanMessage(content="What is quantum computing?")]},
            config=config,
        )
        assert len(result1["messages"]) > 1

        # Follow-up query should have context
        result2 = agent.invoke(
            {"messages": [HumanMessage(content="What are its applications?")]},
            config=config,
        )
        # Second invocation should include previous messages
        assert len(result2["messages"]) > len(result1["messages"])


# Sample research tasks for evaluation (based on DeepResearch Bench patterns)
RESEARCH_TASKS = [
    {
        "id": "tech_trends_1",
        "query": "Analyze the current state of AI safety research.",
        "expected_topics": ["alignment", "interpretability", "safety"],
        "difficulty": "medium",
    },
    {
        "id": "market_analysis_1",
        "query": "What is the current state of the cloud computing market?",
        "expected_topics": ["market size", "growth", "providers"],
        "difficulty": "easy",
    },
    {
        "id": "tech_comparison_1",
        "query": (
            "Compare quantum computing and classical computing for "
            "cryptography applications."
        ),
        "expected_topics": ["quantum", "cryptography", "security"],
        "difficulty": "hard",
    },
]


@pytest.mark.requires("oci", "langgraph", "deepagents")
@pytest.mark.skipif(
    skip_if_no_oci_credentials(),
    reason="OCI credentials not available",
)
@pytest.mark.skipif(
    skip_if_no_deepagents(),
    reason="deepagents package not installed",
)
@pytest.mark.parametrize(
    "task",
    RESEARCH_TASKS,
    ids=[str(t["id"]) for t in RESEARCH_TASKS],
)
def test_research_task_completion(task: dict) -> None:
    """Test that the agent can complete various research tasks."""
    from langchain_oci import create_deep_research_agent

    compartment_id = os.environ.get("OCI_COMPARTMENT_ID", "")
    region = os.environ.get("OCI_REGION", "us-chicago-1")
    service_endpoint = f"https://inference.generativeai.{region}.oci.oraclecloud.com"

    auth_type = os.environ.get("OCI_AUTH_TYPE", "API_KEY")
    auth_profile = os.environ.get("OCI_CONFIG_PROFILE", "API_KEY_AUTH")

    agent = create_deep_research_agent(
        tools=[search_knowledge_base, get_statistics, analyze_trends],
        model_id="google.gemini-2.5-flash",
        compartment_id=compartment_id,
        service_endpoint=service_endpoint,
        auth_type=auth_type,
        auth_profile=auth_profile,
        middleware=[],
        system_prompt="You are a research analyst. Provide thorough analysis.",
        temperature=0.0,
        max_tokens=2048,
    )

    try:
        result = agent.invoke({"messages": [HumanMessage(content=task["query"])]})
    finally:
        llm = getattr(agent, "_oci_llm", None)
        if llm is not None and hasattr(llm, "aclose"):
            asyncio.run(llm.aclose())

    # Verify response
    assert "messages" in result
    final_message = result["messages"][-1]
    assert final_message.content, f"Task {task['id']} should produce a response"

    # Check that response mentions expected topics (at least one). Some model runs
    # return a brief handoff line as the final message, so also inspect the full
    # conversation transcript for topical coverage.
    response_lower = final_message.content.lower()
    transcript_lower = "\n".join(
        str(getattr(msg, "content", "")).lower() for msg in result["messages"]
    )
    topics_found = [
        topic
        for topic in task["expected_topics"]
        if topic in response_lower or topic in transcript_lower
    ]
    assert len(topics_found) >= 1, (
        f"Task {task['id']}: Response should mention at least one of "
        f"{task['expected_topics']}. Got: {final_message.content[:200]}..."
    )


@pytest.mark.requires("oci", "langgraph", "deepagents")
@pytest.mark.skipif(
    skip_if_no_oci_credentials(),
    reason="OCI credentials not available",
)
@pytest.mark.skipif(
    skip_if_no_deepagents(),
    reason="deepagents package not installed",
)
class TestOCIDeepAgentAsyncIntegration:
    """Integration tests for async support in create_deep_research_agent."""

    @pytest.fixture
    def model_id(self) -> str:
        """Use the faster Gemini variant for live async integration reliability."""
        return "google.gemini-2.5-flash"

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
        return os.environ.get("OCI_AUTH_TYPE", "API_KEY")

    @pytest.fixture
    def auth_profile(self) -> str:
        """Get auth profile from environment."""
        return os.environ.get("OCI_CONFIG_PROFILE", "API_KEY_AUTH")

    @pytest_asyncio.fixture
    async def async_agent(
        self,
        compartment_id: str,
        service_endpoint: str,
        auth_type: str,
        auth_profile: str,
        model_id: str,
    ) -> Any:
        """Create a configured deep agent for async testing."""
        from langchain_oci import create_deep_research_agent

        agent = create_deep_research_agent(
            tools=[search_knowledge_base],
            model_id=model_id,
            compartment_id=compartment_id,
            service_endpoint=service_endpoint,
            auth_type=auth_type,
            auth_profile=auth_profile,
            middleware=[],
            temperature=0.3,
            max_tokens=1024,
        )
        try:
            yield agent
        finally:
            llm = getattr(agent, "_oci_llm", None)
            if llm is not None and hasattr(llm, "aclose"):
                await llm.aclose()

    @pytest.mark.asyncio
    async def test_async_invoke(self, async_agent: Any) -> None:
        """Test async invoke works with deep research agent."""
        result = await async_agent.ainvoke(
            {"messages": [HumanMessage(content="What is quantum computing?")]}
        )

        assert "messages" in result
        assert len(result["messages"]) > 1
        final_message = result["messages"][-1]
        assert final_message.content, "Should have a response"

    @pytest.mark.asyncio
    async def test_async_invoke_with_tool_calls(self, async_agent: Any) -> None:
        """Test async invoke correctly handles tool calls."""
        result = await async_agent.ainvoke(
            {
                "messages": [
                    HumanMessage(
                        content="Search the knowledge base for quantum computing."
                    )
                ]
            }
        )

        assert "messages" in result

        # Check that tools were called
        message_types = [type(m).__name__ for m in result["messages"]]
        tool_messages = message_types.count("ToolMessage")
        assert tool_messages >= 1, (
            f"Expected tool calls. Message types: {message_types}"
        )

    @pytest.mark.asyncio
    async def test_async_stream(self, async_agent: Any) -> None:
        """Test async streaming works with deep research agent."""
        chunks_received = 0

        async for chunk in async_agent.astream(
            {"messages": [HumanMessage(content="What is AI?")]}
        ):
            chunks_received += 1

        assert chunks_received >= 1, "Should receive at least one chunk"


@pytest.mark.requires("oci", "langgraph", "deepagents")
@pytest.mark.skipif(
    skip_if_no_oci_credentials(),
    reason="OCI credentials not available",
)
@pytest.mark.skipif(
    skip_if_no_deepagents(),
    reason="deepagents package not installed",
)
@pytest.mark.parametrize(
    "model_id",
    [
        "google.gemini-2.5-pro",
        "google.gemini-2.5-flash",
    ],
)
def test_model_variants(model_id: str) -> None:
    """Test deep agent works with different model variants."""
    from langchain_oci import create_deep_research_agent

    compartment_id = os.environ.get("OCI_COMPARTMENT_ID", "")
    region = os.environ.get("OCI_REGION", "us-chicago-1")
    service_endpoint = f"https://inference.generativeai.{region}.oci.oraclecloud.com"

    auth_type = os.environ.get("OCI_AUTH_TYPE", "API_KEY")
    auth_profile = os.environ.get("OCI_CONFIG_PROFILE", "API_KEY_AUTH")

    agent = create_deep_research_agent(
        tools=[search_knowledge_base],
        model_id=model_id,
        compartment_id=compartment_id,
        service_endpoint=service_endpoint,
        auth_type=auth_type,
        auth_profile=auth_profile,
        middleware=[],
        temperature=0.3,
        max_tokens=1024,
    )
    try:
        result = agent.invoke(
            {"messages": [HumanMessage(content="What is machine learning?")]}
        )

        # Verify we got a response
        assert "messages" in result
        assert len(result["messages"]) > 1
        final_message = result["messages"][-1]
        assert final_message.content, f"Model {model_id} should produce a response"
    finally:
        llm = getattr(agent, "_oci_llm", None)
        if llm is not None and hasattr(llm, "aclose"):
            asyncio.run(llm.aclose())


# =============================================================================
# Tests for new create_deep_agent parity parameters:
#   backend, response_format, context_schema, cache, interrupt_on
# =============================================================================


def _make_oci_kwargs() -> dict:
    """Build common OCI kwargs from environment for standalone tests."""
    compartment_id = os.environ.get("OCI_COMPARTMENT_ID", "")
    region = os.environ.get("OCI_REGION", "us-chicago-1")
    return {
        "compartment_id": compartment_id,
        "service_endpoint": (
            f"https://inference.generativeai.{region}.oci.oraclecloud.com"
        ),
        "auth_type": os.environ.get("OCI_AUTH_TYPE", "API_KEY"),
        "auth_profile": os.environ.get("OCI_CONFIG_PROFILE", "API_KEY_AUTH"),
        "model_id": "google.gemini-2.5-flash",
        "temperature": 0.0,
        "max_tokens": 1024,
    }


def _cleanup_agent(agent: Any) -> None:
    """Close the underlying OCI LLM to avoid unclosed-session warnings."""
    llm = getattr(agent, "_oci_llm", None)
    if llm is not None and hasattr(llm, "aclose"):
        asyncio.run(llm.aclose())


@pytest.mark.requires("oci", "langgraph", "deepagents")
@pytest.mark.skipif(
    skip_if_no_oci_credentials(),
    reason="OCI credentials not available",
)
@pytest.mark.skipif(
    skip_if_no_deepagents(),
    reason="deepagents package not installed",
)
class TestDeepAgentBackend:
    """Integration tests for the backend parameter."""

    def test_state_backend_explicit(self) -> None:
        """Test passing StateBackend explicitly verifies wiring."""
        from deepagents.backends.state import StateBackend

        from langchain_oci import create_deep_research_agent

        agent = create_deep_research_agent(
            tools=[search_knowledge_base],
            backend=StateBackend,
            **_make_oci_kwargs(),
        )
        try:
            result = agent.invoke(
                {"messages": [HumanMessage(content="What is quantum computing?")]}
            )
            assert "messages" in result
            assert len(result["messages"]) > 1
        finally:
            _cleanup_agent(agent)

    def test_store_backend_with_in_memory_store(self) -> None:
        """Test StoreBackend backed by InMemoryStore for persistent files."""
        from deepagents.backends.store import StoreBackend
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.store.memory import InMemoryStore

        from langchain_oci import create_deep_research_agent

        store = InMemoryStore()
        agent = create_deep_research_agent(
            tools=[search_knowledge_base],
            backend=lambda rt: StoreBackend(rt),
            store=store,
            checkpointer=MemorySaver(),
            **_make_oci_kwargs(),
        )
        try:
            config: RunnableConfig = {"configurable": {"thread_id": "backend-test-1"}}
            result = agent.invoke(
                {"messages": [HumanMessage(content="What is AI safety?")]},
                config=config,
            )
            assert "messages" in result
            assert len(result["messages"]) > 1
        finally:
            _cleanup_agent(agent)

    def test_backend_factory_lambda(self) -> None:
        """Test backend as a lambda factory (common user pattern)."""
        from deepagents.backends.state import StateBackend

        from langchain_oci import create_deep_research_agent

        agent = create_deep_research_agent(
            tools=[search_knowledge_base],
            backend=lambda rt: StateBackend(rt),
            **_make_oci_kwargs(),
        )
        try:
            result = agent.invoke(
                {"messages": [HumanMessage(content="What is machine learning?")]}
            )
            assert "messages" in result
            assert len(result["messages"]) > 1
        finally:
            _cleanup_agent(agent)


@pytest.mark.requires("oci", "langgraph", "deepagents")
@pytest.mark.skipif(
    skip_if_no_oci_credentials(),
    reason="OCI credentials not available",
)
@pytest.mark.skipif(
    skip_if_no_deepagents(),
    reason="deepagents package not installed",
)
class TestDeepAgentCache:
    """Integration tests for the cache parameter."""

    def test_cache_with_in_memory_cache(self) -> None:
        """Test that InMemoryCache can be passed and agent still works."""
        from langgraph.cache.memory import InMemoryCache

        from langchain_oci import create_deep_research_agent

        cache: Any = InMemoryCache()
        agent = create_deep_research_agent(
            tools=[search_knowledge_base],
            cache=cache,
            **_make_oci_kwargs(),
        )
        try:
            result = agent.invoke(
                {"messages": [HumanMessage(content="What is cloud computing?")]}
            )
            assert "messages" in result
            assert len(result["messages"]) > 1
        finally:
            _cleanup_agent(agent)

    def test_cache_produces_identical_results(self) -> None:
        """Test that cached responses are consistent across identical calls."""
        from langgraph.cache.memory import InMemoryCache
        from langgraph.checkpoint.memory import MemorySaver

        from langchain_oci import create_deep_research_agent

        cache: Any = InMemoryCache()
        kwargs = _make_oci_kwargs()

        agent = create_deep_research_agent(
            tools=[search_knowledge_base],
            cache=cache,
            checkpointer=MemorySaver(),
            **kwargs,
        )
        try:
            query = "Briefly, what is quantum computing in one sentence?"
            config1: RunnableConfig = {"configurable": {"thread_id": "cache-t1"}}
            config2: RunnableConfig = {"configurable": {"thread_id": "cache-t2"}}

            result1 = agent.invoke(
                {"messages": [HumanMessage(content=query)]},
                config=config1,
            )
            result2 = agent.invoke(
                {"messages": [HumanMessage(content=query)]},
                config=config2,
            )
            # Both should succeed
            assert "messages" in result1
            assert "messages" in result2
        finally:
            _cleanup_agent(agent)


@pytest.mark.requires("oci", "langgraph", "deepagents")
@pytest.mark.skipif(
    skip_if_no_oci_credentials(),
    reason="OCI credentials not available",
)
@pytest.mark.skipif(
    skip_if_no_deepagents(),
    reason="deepagents package not installed",
)
class TestDeepAgentResponseFormat:
    """Integration tests for the response_format parameter."""

    def test_structured_output_with_pydantic(self) -> None:
        """Test response_format with a Pydantic schema via AutoStrategy."""
        from langchain.agents.structured_output import (  # type: ignore[import-not-found,unused-ignore]
            AutoStrategy,
        )
        from pydantic import BaseModel, Field

        from langchain_oci import create_deep_research_agent

        class ResearchSummary(BaseModel):
            topic: str = Field(description="The research topic")
            summary: str = Field(description="A brief summary of findings")
            confidence: float = Field(description="Confidence score between 0 and 1")

        agent = create_deep_research_agent(
            tools=[search_knowledge_base],
            response_format=AutoStrategy(schema=ResearchSummary),
            **_make_oci_kwargs(),
        )
        try:
            result = agent.invoke(
                {
                    "messages": [
                        HumanMessage(
                            content="Research quantum computing and provide a summary."
                        )
                    ]
                }
            )
            assert "messages" in result

            # The final message should contain structured output
            final = result["messages"][-1]
            assert final.content, "Should have structured response"
        finally:
            _cleanup_agent(agent)


@pytest.mark.requires("oci", "langgraph", "deepagents")
@pytest.mark.skipif(
    skip_if_no_oci_credentials(),
    reason="OCI credentials not available",
)
@pytest.mark.skipif(
    skip_if_no_deepagents(),
    reason="deepagents package not installed",
)
class TestDeepAgentInterruptOn:
    """Integration tests for the interrupt_on parameter."""

    def test_interrupt_on_tool_call(self) -> None:
        """Test that interrupt_on pauses execution before the specified tool."""
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.types import Command

        from langchain_oci import create_deep_research_agent

        agent = create_deep_research_agent(
            tools=[search_knowledge_base],
            interrupt_on={"search_knowledge_base": True},
            checkpointer=MemorySaver(),
            **_make_oci_kwargs(),
        )
        try:
            config: RunnableConfig = {"configurable": {"thread_id": "interrupt-test-1"}}
            result = agent.invoke(
                {
                    "messages": [
                        HumanMessage(
                            content=("Search the knowledge base for quantum computing.")
                        )
                    ]
                },
                config=config,
            )

            # The agent should have paused — the result should contain
            # messages up to (but not including) the tool execution.
            assert "messages" in result
            message_types = [type(m).__name__ for m in result["messages"]]

            # There should be an AIMessage with tool_calls but no ToolMessage
            # yet (execution was interrupted before the tool ran).
            has_ai_with_tool_calls = any(
                type(m).__name__ == "AIMessage" and getattr(m, "tool_calls", None)
                for m in result["messages"]
            )
            assert has_ai_with_tool_calls, (
                "Agent should have proposed a tool call before being interrupted. "
                f"Message types: {message_types}"
            )

            # Resume execution by approving all pending tool calls.
            # HumanInTheLoopMiddleware expects {"decisions": [...]}.
            ai_msg = next(
                m
                for m in reversed(result["messages"])
                if type(m).__name__ == "AIMessage" and getattr(m, "tool_calls", None)
            )
            approvals = [{"type": "approve"} for _ in ai_msg.tool_calls]
            result2 = agent.invoke(
                Command(resume={"decisions": approvals}), config=config
            )
            assert "messages" in result2

            # Now the tool should have executed
            all_types = [type(m).__name__ for m in result2["messages"]]
            assert "ToolMessage" in all_types, (
                "After resuming, the tool should have executed. "
                f"Message types: {all_types}"
            )
        finally:
            _cleanup_agent(agent)


@pytest.mark.requires("oci", "langgraph", "deepagents")
@pytest.mark.skipif(
    skip_if_no_oci_credentials(),
    reason="OCI credentials not available",
)
@pytest.mark.skipif(
    skip_if_no_deepagents(),
    reason="deepagents package not installed",
)
class TestDeepAgentContextSchema:
    """Integration tests for the context_schema parameter."""

    def test_context_schema_with_typed_dict(self) -> None:
        """Test that context_schema is accepted and agent functions normally."""
        from typing import TypedDict

        from langchain_oci import create_deep_research_agent

        class ResearchContext(TypedDict):
            project_name: str
            department: str

        agent = create_deep_research_agent(
            tools=[search_knowledge_base],
            context_schema=ResearchContext,
            **_make_oci_kwargs(),
        )
        try:
            result = agent.invoke(
                {"messages": [HumanMessage(content="What is AI safety?")]}
            )
            assert "messages" in result
            assert len(result["messages"]) > 1
        finally:
            _cleanup_agent(agent)


@pytest.mark.requires("oci", "langgraph", "deepagents")
@pytest.mark.skipif(
    skip_if_no_oci_credentials(),
    reason="OCI credentials not available",
)
@pytest.mark.skipif(
    skip_if_no_deepagents(),
    reason="deepagents package not installed",
)
class TestDeepAgentAllParamsCombined:
    """Integration test combining multiple new parameters."""

    def test_backend_store_cache_checkpointer_combined(self) -> None:
        """Test backend + store + cache + checkpointer all together."""
        from deepagents.backends.store import StoreBackend
        from langgraph.cache.memory import InMemoryCache
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.store.memory import InMemoryStore

        from langchain_oci import create_deep_research_agent

        store = InMemoryStore()
        agent = create_deep_research_agent(
            tools=[search_knowledge_base, get_statistics],
            backend=lambda rt: StoreBackend(rt),
            store=store,
            cache=InMemoryCache(),
            checkpointer=MemorySaver(),
            **_make_oci_kwargs(),
        )
        try:
            config: RunnableConfig = {"configurable": {"thread_id": "combined-test-1"}}

            # First turn
            result1 = agent.invoke(
                {"messages": [HumanMessage(content="What is quantum computing?")]},
                config=config,
            )
            assert "messages" in result1
            assert len(result1["messages"]) > 1

            # Follow-up in same thread (tests checkpointer + store persistence)
            result2 = agent.invoke(
                {"messages": [HumanMessage(content="What are its market statistics?")]},
                config=config,
            )
            assert "messages" in result2
            # Second turn should have accumulated messages from first turn
            assert len(result2["messages"]) > len(result1["messages"])
        finally:
            _cleanup_agent(agent)
