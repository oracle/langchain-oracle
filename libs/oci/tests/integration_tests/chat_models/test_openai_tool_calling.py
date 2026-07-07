# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Integration tests for OpenAI GPT streamed parallel tool calling (issue #253).

GPT models stream parallel tool calls sequentially at ``toolCalls[0]``:
each call opens with a chunk carrying id/name and empty arguments, then
id-less argument fragments follow. These tests verify that every tool
call in a multi-tool assistant turn reconstructs with its own complete,
valid arguments — on both the sync and async streaming paths.

## Environment variables

- ``OCI_COMPARTMENT_ID`` — required; tests skip when unset.
- ``OCI_REGION`` — region for the GenAI endpoint (default ``us-chicago-1``).
- ``OCI_AUTH_TYPE`` / ``OCI_CONFIG_PROFILE`` — auth configuration
  (defaults: ``API_KEY`` / ``API_KEY_AUTH``).
- ``OCI_GPT_PARALLEL_MODELS`` — comma-separated GPT model ids to
  exercise (default ``openai.gpt-4.1``, the model from issue #253).
"""

import os
from typing import Callable, Dict, Optional

import pytest
from langchain_core.messages import AIMessageChunk, BaseMessageChunk
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from langchain_oci.chat_models import ChatOCIGenAI

from .conftest import _env_list

GPT_PARALLEL_MODELS = _env_list("OCI_GPT_PARALLEL_MODELS", ["openai.gpt-4.1"])

pytestmark = pytest.mark.skipif(
    not os.environ.get("OCI_COMPARTMENT_ID"),
    reason="OCI_COMPARTMENT_ID not set",
)


# --------------- tools ---------------


class MarketResearchInput(BaseModel):
    idea: str = Field(description="The product idea to research")
    market_segment: str = Field(description="Target market segment")


class CustomerSignalInput(BaseModel):
    idea: str = Field(description="The product idea to analyze")
    audience: str = Field(description="Target audience")


class RoadmapRiskInput(BaseModel):
    product_area: str = Field(description="Product area to assess")
    risk_focus: str = Field(description="Primary risk dimension")


def _market_research(idea: str, market_segment: str) -> str:
    return f"Market research for {idea} in {market_segment}: TAM $1B."


def _customer_signal(idea: str, audience: str) -> str:
    return f"Customer signal for {idea} among {audience}: strong interest."


def _roadmap_risk(product_area: str, risk_focus: str) -> str:
    return f"Roadmap risk for {product_area} on {risk_focus}: moderate."


market_research_tool = StructuredTool.from_function(
    func=_market_research,
    name="market_research_tool",
    description="Research the market for a product idea.",
    args_schema=MarketResearchInput,
)

customer_signal_tool = StructuredTool.from_function(
    func=_customer_signal,
    name="customer_signal_tool",
    description="Analyze customer signals for a product idea.",
    args_schema=CustomerSignalInput,
)

roadmap_risk_tool = StructuredTool.from_function(
    func=_roadmap_risk,
    name="roadmap_risk_tool",
    description="Assess roadmap risk for a product area.",
    args_schema=RoadmapRiskInput,
)

ALL_TOOLS = [market_research_tool, customer_signal_tool, roadmap_risk_tool]

# Typed as constructors rather than type[BaseModel]: the pydantic mypy
# plugin synthesizes per-class __init__ signatures, making the class
# objects mutually incompatible as types.
ARGS_SCHEMAS: Dict[str, Callable[..., BaseModel]] = {
    "market_research_tool": MarketResearchInput,
    "customer_signal_tool": CustomerSignalInput,
    "roadmap_risk_tool": RoadmapRiskInput,
}

PARALLEL_PROMPT = (
    "Call all three tools before responding:\n"
    "- market_research_tool for the idea 'AI meal planner' in the "
    "'busy professionals' market segment\n"
    "- customer_signal_tool for the idea 'AI meal planner' with the "
    "'working parents' audience\n"
    "- roadmap_risk_tool for the 'mobile app' product area with a "
    "'churn' risk focus"
)


# --------------- helpers ---------------


def _make_llm(model_id: str) -> ChatOCIGenAI:
    region = os.getenv("OCI_REGION", "us-chicago-1")
    endpoint = f"https://inference.generativeai.{region}.oci.oraclecloud.com"
    return ChatOCIGenAI(
        model_id=model_id,
        service_endpoint=endpoint,
        compartment_id=os.environ["OCI_COMPARTMENT_ID"],
        model_kwargs={"max_tokens": 1024},
        auth_type=os.environ.get("OCI_AUTH_TYPE", "API_KEY"),
        auth_profile=os.environ.get("OCI_CONFIG_PROFILE", "API_KEY_AUTH"),
        auth_file_location=os.path.expanduser("~/.oci/config"),
    )


def _assert_parallel_tool_calls_reconstructed(
    merged: Optional[BaseMessageChunk], model_id: str
) -> None:
    """Assert every streamed tool call carries its own complete arguments."""
    assert merged is not None, f"{model_id}: stream produced no chunks"
    assert isinstance(merged, AIMessageChunk)

    assert not merged.invalid_tool_calls, (
        f"{model_id}: malformed tool calls reconstructed from stream: "
        f"{merged.invalid_tool_calls}"
    )

    if len(merged.tool_calls) < 2:
        pytest.skip(
            f"{model_id}: model made {len(merged.tool_calls)} tool call(s); "
            "need 2+ in one turn to exercise parallel reconstruction"
        )

    for tc in merged.tool_calls:
        assert tc["name"] in ARGS_SCHEMAS, f"{model_id}: unknown tool {tc['name']}"
        assert tc["args"], (
            f"{model_id}: {tc['name']} reconstructed with empty args "
            "(fragments were routed to another tool call)"
        )
        # Args must validate against the tool's schema — catches both
        # empty {} and cross-contaminated argument fragments.
        ARGS_SCHEMAS[tc["name"]](**tc["args"])

    ids = [tc["id"] for tc in merged.tool_calls]
    assert len(ids) == len(set(ids)), f"{model_id}: duplicate tool call ids: {ids}"


# --------------- tests ---------------


@pytest.mark.parametrize("model_id", GPT_PARALLEL_MODELS)
def test_gpt_streamed_parallel_tool_calls(model_id: str) -> None:
    """Streamed parallel tool calls each reconstruct with their own args."""
    llm = _make_llm(model_id).bind_tools(ALL_TOOLS)

    merged: Optional[BaseMessageChunk] = None
    for chunk in llm.stream(PARALLEL_PROMPT):
        assert isinstance(chunk, AIMessageChunk)
        merged = chunk if merged is None else merged + chunk

    _assert_parallel_tool_calls_reconstructed(merged, model_id)


@pytest.mark.parametrize("model_id", GPT_PARALLEL_MODELS)
async def test_gpt_streamed_parallel_tool_calls_async(model_id: str) -> None:
    """Async streaming path reconstructs parallel tool calls identically."""
    llm = _make_llm(model_id).bind_tools(ALL_TOOLS)

    merged: Optional[BaseMessageChunk] = None
    async for chunk in llm.astream(PARALLEL_PROMPT):
        assert isinstance(chunk, AIMessageChunk)
        merged = chunk if merged is None else merged + chunk

    _assert_parallel_tool_calls_reconstructed(merged, model_id)
