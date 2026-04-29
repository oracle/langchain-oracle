# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Integration tests for agents with datastores."""

import sys

import pytest
from langchain_core.messages import HumanMessage

from .conftest import (
    adb_is_reachable,
    create_adb_store,
    create_embedding_model,
    create_opensearch_store,
    get_adb_config,
    get_oci_config,
    get_opensearch_config,
    opensearch_is_reachable,
)

# deepagents requires Python 3.11+; Python 3.14 not yet tested
_SKIP_DEEPAGENTS = sys.version_info < (3, 11) or sys.version_info >= (3, 14)


@pytest.mark.requires("oci", "opensearchpy", "langgraph")
@pytest.mark.skipif(
    not opensearch_is_reachable() or not get_oci_config().get("chat_model"),
    reason="OpenSearch or OCI chat model not configured",
)
class TestReactAgentWithOpenSearch:
    """Test ReAct agent with real OpenSearch datastore."""

    @pytest.fixture
    def embedding_model(self):
        """Create embedding model matching OpenSearch index dimensions."""
        config = get_opensearch_config()
        return create_embedding_model(config["embedding_model"])

    @pytest.fixture
    def stores(self) -> dict:
        """Create OpenSearch stores for testing."""
        return {"diagnostics": create_opensearch_store()}

    def test_agent_uses_search_tool(self, stores, embedding_model) -> None:
        """Test that agent can use search tool to answer questions."""
        from langchain_oci import create_datastore_tools, create_oci_agent

        oci_config = get_oci_config()

        tools = create_datastore_tools(
            stores=stores,
            embedding_model=embedding_model,
            top_k=3,
        )

        agent = create_oci_agent(
            model_id=oci_config["chat_model"],
            tools=tools,
            compartment_id=oci_config["compartment_id"],
            service_endpoint=oci_config["service_endpoint"],
            auth_type=oci_config["auth_type"],
            auth_profile=oci_config["auth_profile"],
            system_prompt="You are an assistant with access to search tools.",
            temperature=0.3,
            max_tokens=1024,
        )

        question = "What information can you find about the available data?"
        result = agent.invoke({"messages": [HumanMessage(content=question)]})

        assert "messages" in result
        assert len(result["messages"]) > 1


@pytest.mark.requires("oci", "opensearchpy", "langgraph", "deepagents")
@pytest.mark.skipif(_SKIP_DEEPAGENTS, reason="deepagents requires Python 3.11-3.13")
@pytest.mark.skipif(
    not opensearch_is_reachable(), reason="OpenSearch not configured or reachable"
)
class TestDeepResearchAgentWithOpenSearch:
    """Test Deep Research agent with real OpenSearch datastore."""

    @pytest.fixture
    def embedding_model(self):
        """Create embedding model matching OpenSearch index dimensions."""
        config = get_opensearch_config()
        return create_embedding_model(config["embedding_model"])

    @pytest.fixture
    def stores(self) -> dict:
        """Create OpenSearch stores for testing."""
        return {"diagnostics": create_opensearch_store()}

    def test_deep_agent_with_real_opensearch(self, stores, embedding_model) -> None:
        """Test deep research agent with real OpenSearch."""
        try:
            import deepagents  # noqa: F401
        except ImportError:
            pytest.skip("deepagents not installed")

        from langchain_oci import create_deep_research_agent

        oci_config = get_oci_config()
        model_id = oci_config.get("deep_research_model") or "google.gemini-2.5-pro"

        agent = create_deep_research_agent(
            datastores=stores,
            embedding_model=embedding_model,
            model_id=model_id,
            compartment_id=oci_config["compartment_id"],
            service_endpoint=oci_config["service_endpoint"],
            auth_type=oci_config["auth_type"],
            auth_profile=oci_config["auth_profile"],
            temperature=0.3,
            middleware=[],
        )

        question = "What information is available in the datastore?"
        result = agent.invoke({"messages": [HumanMessage(content=question)]})

        assert "messages" in result
        assert len(result["messages"]) > 1


@pytest.mark.requires("oci", "oracledb", "langgraph", "deepagents")
@pytest.mark.skipif(_SKIP_DEEPAGENTS, reason="deepagents requires Python 3.11-3.13")
@pytest.mark.skipif(not adb_is_reachable(), reason="ADB not configured or reachable")
class TestDeepResearchAgentWithADB:
    """Test Deep Research agent with real ADB datastore."""

    @pytest.fixture
    def embedding_model(self):
        """Create embedding model matching ADB dimensions."""
        config = get_adb_config()
        return create_embedding_model(config["embedding_model"])

    @pytest.fixture
    def stores(self) -> dict:
        """Create ADB stores for testing."""
        return {"research": create_adb_store()}

    def test_deep_agent_with_real_adb(self, stores, embedding_model) -> None:
        """Test deep research agent with real ADB."""
        try:
            import deepagents  # noqa: F401
        except ImportError:
            pytest.skip("deepagents not installed")

        from langchain_oci import create_deep_research_agent

        oci_config = get_oci_config()
        model_id = oci_config.get("deep_research_model") or "google.gemini-2.5-pro"

        agent = create_deep_research_agent(
            datastores=stores,
            embedding_model=embedding_model,
            model_id=model_id,
            compartment_id=oci_config["compartment_id"],
            service_endpoint=oci_config["service_endpoint"],
            auth_type=oci_config["auth_type"],
            auth_profile=oci_config["auth_profile"],
            temperature=0.3,
            middleware=[],
        )

        question = "What information is available in the datastore?"
        result = agent.invoke({"messages": [HumanMessage(content=question)]})

        assert "messages" in result
        assert len(result["messages"]) > 1


@pytest.mark.requires("oci", "opensearchpy", "oracledb", "langgraph")
@pytest.mark.skipif(
    not opensearch_is_reachable() or not adb_is_reachable(),
    reason="OpenSearch or ADB not configured or reachable",
)
class TestMultiStoreRouting:
    """Test store routing with multiple datastores."""

    def test_routing_based_on_datastore_descriptions(self) -> None:
        """Test that queries are routed based on datastore descriptions."""
        from langchain_oci.datastores.tools import StoreSelector

        opensearch_store = create_opensearch_store()
        adb_store = create_adb_store()

        opensearch_store.datastore_description = "diagnostic patterns, troubleshooting"
        adb_store.datastore_description = "legal contracts, agreements"

        opensearch_config = get_opensearch_config()
        embedding_model = create_embedding_model(opensearch_config["embedding_model"])

        stores = {"diagnostics": opensearch_store, "legal": adb_store}

        selector = StoreSelector(
            stores=stores,
            embedding_model=embedding_model,
            default_store="diagnostics",
        )

        routed = selector.route("database connection errors")
        assert routed in stores

    def test_tool_descriptions_include_datastore_descriptions(self) -> None:
        """Verify tool descriptions include store datastore descriptions."""
        from langchain_oci import create_datastore_tools

        opensearch_config = get_opensearch_config()

        opensearch_store = create_opensearch_store()
        opensearch_store.datastore_description = "diagnostic patterns"

        adb_store = create_adb_store()
        adb_store.datastore_description = "legal contracts"

        stores = {"diagnostics": opensearch_store, "legal": adb_store}
        embedding_model = create_embedding_model(opensearch_config["embedding_model"])

        tools = create_datastore_tools(
            stores=stores,
            embedding_model=embedding_model,
        )

        search_tool = next(t for t in tools if t.name == "search")
        desc = search_tool.description.lower()
        assert "diagnostics" in search_tool.description or "diagnostic" in desc


@pytest.mark.requires("oci", "langgraph", "deepagents")
@pytest.mark.skipif(_SKIP_DEEPAGENTS, reason="deepagents requires Python 3.11-3.13")
class TestResearchDocumentGeneration:
    """Test generating research documents with deep research agent."""

    def test_generate_research_document(self) -> None:
        """Generate a research document using available datastores."""
        try:
            import deepagents  # noqa: F401
        except ImportError:
            pytest.skip("deepagents not installed")

        from langchain_oci import create_deep_research_agent

        oci_config = get_oci_config()
        model_id = oci_config.get("deep_research_model") or "google.gemini-2.5-pro"

        stores: dict = {}
        embedding_model = None

        if opensearch_is_reachable():
            config = get_opensearch_config()
            store = create_opensearch_store()
            store.datastore_description = config.get("hint", "")
            stores["opensearch"] = store
            embedding_model = create_embedding_model(config["embedding_model"])

        if adb_is_reachable():
            config = get_adb_config()
            store = create_adb_store()
            store.datastore_description = config.get("hint", "")
            stores["adb"] = store
            if embedding_model is None:
                embedding_model = create_embedding_model(config["embedding_model"])

        if not stores:
            pytest.skip("No datastores configured or reachable")

        agent = create_deep_research_agent(
            datastores=stores,
            embedding_model=embedding_model,
            model_id=model_id,
            compartment_id=oci_config["compartment_id"],
            service_endpoint=oci_config["service_endpoint"],
            auth_type=oci_config["auth_type"],
            auth_profile=oci_config["auth_profile"],
            temperature=0.4,
            max_tokens=4096,
            default_store=list(stores.keys())[0],
            middleware=[],
        )

        prompt = "Explore the knowledge base and summarize the main topics found."
        result = agent.invoke({"messages": [HumanMessage(content=prompt)]})

        assert "messages" in result
        assert len(result["messages"]) > 1

        # Verify tools were used
        tool_calls = []
        for msg in result["messages"]:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                tool_calls.extend([tc["name"] for tc in msg.tool_calls])

        assert len(tool_calls) > 0
