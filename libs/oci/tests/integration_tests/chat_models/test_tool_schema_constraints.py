# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Integration tests for tool schema constraint preservation.

Verifies that enum, min/max, and other JSON Schema constraints are
passed through to the model and respected in tool call arguments.

Prerequisites: same as test_tool_calling.py (OCI_COMPARTMENT_ID, auth).
"""

import os
from typing import Optional

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from langchain_oci.chat_models import ChatOCIGenAI


def _create_chat_model(model_id: str) -> ChatOCIGenAI:
    compartment_id = os.environ.get("OCI_COMPARTMENT_ID")
    if not compartment_id:
        pytest.skip("OCI_COMPARTMENT_ID not set")

    region = os.getenv("OCI_REGION", "us-chicago-1")
    endpoint = f"https://inference.generativeai.{region}.oci.oraclecloud.com"
    return ChatOCIGenAI(
        model_id=model_id,
        service_endpoint=endpoint,
        compartment_id=compartment_id,
        model_kwargs={"temperature": 0, "max_tokens": 512},
        auth_type=os.environ.get("OCI_AUTH_TYPE", "API_KEY"),
        auth_profile=os.environ.get("OCI_CONFIG_PROFILE", "DEFAULT"),
        auth_file_location=os.path.expanduser("~/.oci/config"),
    )


class QueryMetricsInput(BaseModel):
    metric_name: str = Field(
        description="Infrastructure metric to query",
        json_schema_extra={"enum": ["cpu", "memory", "disk", "network"]},
    )
    duration_hours: int = Field(
        description="Time window in hours",
        ge=1,
        le=168,
        default=24,
    )
    output_format: Optional[str] = Field(
        description="Output format",
        default="json",
        json_schema_extra={"enum": ["json", "csv", "table"]},
    )


@tool(args_schema=QueryMetricsInput)
def query_metrics(
    metric_name: str,
    duration_hours: int = 24,
    output_format: Optional[str] = "json",
) -> str:
    """Query infrastructure metrics for monitoring dashboards."""
    return f"{metric_name} data for last {duration_hours}h in {output_format} format"


VALID_METRICS = {"cpu", "memory", "disk", "network"}
VALID_FORMATS = {"json", "csv", "table"}


@pytest.mark.requires("oci")
@pytest.mark.parametrize(
    "model_id",
    [
        "meta.llama-4-scout-17b-16e-instruct",
        "cohere.command-a-03-2025",
        "google.gemini-2.5-flash",
    ],
)
def test_enum_constraints_respected(model_id: str):
    """Model should pick metric_name from the enum values, not hallucinate."""
    llm = _create_chat_model(model_id)
    model_with_tools = llm.bind_tools([query_metrics])

    response = model_with_tools.invoke(
        [
            SystemMessage(content="Use the query_metrics tool to answer."),
            HumanMessage(content="Show me the CPU usage for the last 6 hours."),
        ]
    )

    assert isinstance(response, AIMessage)
    assert response.tool_calls, "Model should have made a tool call"

    tc = response.tool_calls[0]
    assert tc["name"] == "query_metrics"

    args = tc["args"]
    assert args["metric_name"] in VALID_METRICS, (
        f"metric_name '{args['metric_name']}' not in enum {VALID_METRICS}"
    )
    assert args["metric_name"] == "cpu", "Should have picked 'cpu' from enum"


@pytest.mark.requires("oci")
@pytest.mark.parametrize(
    "model_id",
    [
        "meta.llama-4-scout-17b-16e-instruct",
        "cohere.command-a-03-2025",
        "google.gemini-2.5-flash",
    ],
)
def test_numeric_range_constraints(model_id: str):
    """duration_hours should be within min=1, max=168."""
    llm = _create_chat_model(model_id)
    model_with_tools = llm.bind_tools([query_metrics])

    response = model_with_tools.invoke(
        [
            SystemMessage(content="Use the query_metrics tool to answer."),
            HumanMessage(content="Get memory metrics for the past 48 hours in CSV."),
        ]
    )

    assert isinstance(response, AIMessage)
    assert response.tool_calls, "Model should have made a tool call"

    args = response.tool_calls[0]["args"]
    assert args["metric_name"] in VALID_METRICS
    if "duration_hours" in args:
        assert 1 <= args["duration_hours"] <= 168, (
            f"duration_hours {args['duration_hours']} outside range [1, 168]"
        )
    if "output_format" in args:
        assert args["output_format"] in VALID_FORMATS, (
            f"output_format '{args['output_format']}' not in {VALID_FORMATS}"
        )


@pytest.mark.requires("oci")
@pytest.mark.parametrize(
    "model_id",
    [
        "meta.llama-4-scout-17b-16e-instruct",
        "cohere.command-a-03-2025",
        "google.gemini-2.5-flash",
    ],
)
def test_schema_sent_to_oci_has_constraints(model_id: str):
    """Verify the converted OCI tool definition actually contains constraints."""
    llm = _create_chat_model(model_id)

    oci_tool = llm._provider.convert_to_oci_tool(query_metrics)

    # GenericProvider returns FunctionDefinition with .parameters dict
    # CohereProvider returns CohereTool with .parameter_definitions dict
    if hasattr(oci_tool, "parameters"):
        # Generic provider path
        params = oci_tool.parameters
        props = params["properties"]
        assert "enum" in props["metric_name"], (
            "enum missing from metric_name — constraints not passed through"
        )
        assert props["metric_name"]["enum"] == ["cpu", "memory", "disk", "network"]
        assert props["duration_hours"].get("minimum") == 1
        assert props["duration_hours"].get("maximum") == 168
    else:
        # Cohere provider path — constraints embedded in description
        param_defs = oci_tool.parameter_definitions
        desc = param_defs["metric_name"].description
        assert "cpu" in desc and "memory" in desc, (
            f"Enum values missing from description: {desc}"
        )
