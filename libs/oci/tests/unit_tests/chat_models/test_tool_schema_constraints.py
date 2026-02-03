# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Unit tests for tool schema conversion — all 14 scenarios.

Covers nested schemas, anyOf resolution, json_schema_extra constraints,
Pydantic-native constraints, Python Enums, and backward compatibility
for both GenericProvider and CohereProvider.

See: https://github.com/oracle/langchain-oracle/issues/103
     https://github.com/oracle/langchain-oracle/pull/109#issuecomment-3837468732
"""

from enum import Enum
from typing import List, Optional

import pytest
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from langchain_oci.chat_models.providers.cohere import CohereProvider
from langchain_oci.chat_models.providers.generic import GenericProvider

# ---------------------------------------------------------------------------
# Test tool definitions
# ---------------------------------------------------------------------------


# 01: enum via json_schema_extra
class EnumInput(BaseModel):
    metric_name: str = Field(
        description="Metric to query",
        json_schema_extra={"enum": ["cpu", "memory", "disk", "network"]},
    )


@tool(args_schema=EnumInput)
def enum_tool(metric_name: str) -> str:
    """Query a metric."""
    return metric_name


# 02: numeric range (ge/le -> minimum/maximum)
class RangeInput(BaseModel):
    duration_hours: int = Field(
        description="Time window in hours", ge=1, le=168, default=24
    )


@tool(args_schema=RangeInput)
def range_tool(duration_hours: int = 24) -> str:
    """Get data for a time range."""
    return str(duration_hours)


# 03: Optional[str] + enum (anyOf + json_schema_extra)
class OptionalEnumInput(BaseModel):
    output_format: Optional[str] = Field(
        description="Output format",
        default="json",
        json_schema_extra={"enum": ["json", "csv", "table"]},
    )


@tool(args_schema=OptionalEnumInput)
def optional_enum_tool(output_format: Optional[str] = "json") -> str:
    """Format output."""
    return output_format or "json"


# 04: Optional[int] (anyOf -> integer, not "any")
class OptionalIntInput(BaseModel):
    count: Optional[int] = Field(description="Number of results", default=None)


@tool(args_schema=OptionalIntInput)
def optional_int_tool(count: Optional[int] = None) -> str:
    """Count results."""
    return str(count)


# 05: pattern (Pydantic-native)
class PatternInput(BaseModel):
    email: str = Field(description="Email address", pattern=r"^[\w.]+@[\w.]+$")


@tool(args_schema=PatternInput)
def pattern_tool(email: str) -> str:
    """Validate email."""
    return email


# 06: format via json_schema_extra
class FormatInput(BaseModel):
    timestamp: str = Field(
        description="A timestamp",
        json_schema_extra={"format": "date-time"},
    )


@tool(args_schema=FormatInput)
def format_tool(timestamp: str) -> str:
    """Process timestamp."""
    return timestamp


# 07: string length (Pydantic-native)
class LengthInput(BaseModel):
    name: str = Field(description="User name", min_length=1, max_length=100)


@tool(args_schema=LengthInput)
def length_tool(name: str) -> str:
    """Process name."""
    return name


# 08: array items with enum (json_schema_extra)
class ArrayInput(BaseModel):
    tags: List[str] = Field(
        description="Tags",
        min_length=1,
        max_length=5,
        json_schema_extra={
            "items": {"type": "string", "enum": ["a", "b", "c"]},
        },
    )


@tool(args_schema=ArrayInput)
def array_tool(tags: List[str]) -> str:
    """Process tags."""
    return ",".join(tags)


# 09: nested object ($defs + $ref)
class FilterParams(BaseModel):
    status: str = Field(
        description="Status filter",
        json_schema_extra={"enum": ["active", "inactive"]},
    )
    count: int = Field(description="Max results", ge=0)


class FilterInput(BaseModel):
    filter: FilterParams = Field(description="Filter parameters")


@tool(args_schema=FilterInput)
def nested_tool(filter: dict) -> str:
    """Apply filter."""
    return str(filter)


# 10: combined MCP scenario
class FullMCPInput(BaseModel):
    metric_name: str = Field(
        description="Infrastructure metric to query",
        json_schema_extra={"enum": ["cpu", "memory", "disk", "network"]},
    )
    duration_hours: int = Field(
        description="Time window in hours", ge=1, le=168, default=24
    )
    output_format: Optional[str] = Field(
        description="Output format",
        default="json",
        json_schema_extra={"enum": ["json", "csv", "table"]},
    )


@tool(args_schema=FullMCPInput)
def full_mcp_tool(
    metric_name: str,
    duration_hours: int = 24,
    output_format: Optional[str] = "json",
) -> str:
    """Query infrastructure metrics for monitoring dashboards."""
    return f"{metric_name} {duration_hours}h {output_format}"


# 11: Python Enum ($defs + $ref)
class MetricNameEnum(str, Enum):
    cpu = "cpu"
    memory = "memory"
    disk = "disk"


class NativeEnumInput(BaseModel):
    metric: MetricNameEnum = Field(description="Which metric")


@tool(args_schema=NativeEnumInput)
def native_enum_tool(metric: MetricNameEnum) -> str:
    """Query with native enum."""
    return metric.value


# 12: exclusiveMinimum/exclusiveMaximum (gt/lt)
class ExclusiveRangeInput(BaseModel):
    score: float = Field(description="A score", gt=0, lt=1.0)


@tool(args_schema=ExclusiveRangeInput)
def exclusive_range_tool(score: float) -> str:
    """Process score."""
    return str(score)


# 13: multiple Optional fields
class MultiOptionalInput(BaseModel):
    name: Optional[str] = Field(description="Name", default=None)
    age: Optional[int] = Field(description="Age", default=None)
    active: Optional[bool] = Field(description="Active", default=None)


@tool(args_schema=MultiOptionalInput)
def multi_optional_tool(
    name: Optional[str] = None,
    age: Optional[int] = None,
    active: Optional[bool] = None,
) -> str:
    """Process optional fields."""
    return f"{name} {age} {active}"


# 14: const via json_schema_extra
class ConstInput(BaseModel):
    version: str = Field(
        description="API version",
        json_schema_extra={"const": "v1"},
    )


@tool(args_schema=ConstInput)
def const_tool(version: str) -> str:
    """Use const version."""
    return version


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _props(tool_obj):
    """Run convert_to_oci_tool and return the properties dict."""
    provider = GenericProvider()
    result = provider.convert_to_oci_tool(tool_obj)
    return result.parameters.get("properties", {})  # type: ignore[attr-defined]


def _result(tool_obj):
    """Run convert_to_oci_tool and return the full result."""
    provider = GenericProvider()
    return provider.convert_to_oci_tool(tool_obj)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.requires("oci")
def test_01_enum_extra():
    """json_schema_extra enum values must be preserved."""
    p = _props(enum_tool)
    assert p["metric_name"]["enum"] == ["cpu", "memory", "disk", "network"]


@pytest.mark.requires("oci")
def test_02_range_native():
    """Pydantic ge/le must produce minimum/maximum."""
    p = _props(range_tool)
    assert p["duration_hours"]["minimum"] == 1
    assert p["duration_hours"]["maximum"] == 168


@pytest.mark.requires("oci")
def test_03_optional_enum_extra():
    """Optional[str] + json_schema_extra enum: anyOf resolved, enum kept."""
    p = _props(optional_enum_tool)
    assert p["output_format"]["type"] == "string"
    assert p["output_format"]["enum"] == ["json", "csv", "table"]


@pytest.mark.requires("oci")
def test_04_optional_int_anyof():
    """Optional[int] must resolve to type=integer, not type=any."""
    p = _props(optional_int_tool)
    assert p["count"]["type"] == "integer"


@pytest.mark.requires("oci")
def test_05_pattern_native():
    """Pydantic pattern constraint must be preserved."""
    p = _props(pattern_tool)
    assert p["email"]["pattern"] == r"^[\w.]+@[\w.]+$"


@pytest.mark.requires("oci")
def test_06_format_extra():
    """json_schema_extra format must be preserved."""
    p = _props(format_tool)
    assert p["timestamp"]["format"] == "date-time"


@pytest.mark.requires("oci")
def test_07_length_native():
    """Pydantic min_length/max_length must be preserved."""
    p = _props(length_tool)
    assert p["name"]["minLength"] == 1
    assert p["name"]["maxLength"] == 100


@pytest.mark.requires("oci")
def test_08_array_item_extra():
    """Array items with json_schema_extra enum must be preserved."""
    p = _props(array_tool)
    assert p["tags"]["type"] == "array"
    assert p["tags"]["items"]["enum"] == ["a", "b", "c"]


@pytest.mark.requires("oci")
def test_09_nested_ref():
    """Nested BaseModel ($defs/$ref) must be fully resolved and inlined."""
    p = _props(nested_tool)
    assert p["filter"]["type"] == "object"
    nested = p["filter"]["properties"]
    assert "status" in nested
    assert "count" in nested
    assert nested["status"]["enum"] == ["active", "inactive"]
    assert nested["count"]["minimum"] == 0
    # no $ref or $defs should remain anywhere
    assert "$ref" not in str(p)
    assert "$defs" not in str(p)


@pytest.mark.requires("oci")
def test_10_full_mcp_combo():
    """Combined: enum (extra) + range (native) + optional enum (extra+anyOf)."""
    p = _props(full_mcp_tool)
    assert p["metric_name"]["enum"] == ["cpu", "memory", "disk", "network"]
    assert p["duration_hours"]["minimum"] == 1
    assert p["duration_hours"]["maximum"] == 168
    assert p["output_format"]["type"] == "string"
    assert p["output_format"]["enum"] == ["json", "csv", "table"]


@pytest.mark.requires("oci")
def test_11_native_enum_ref():
    """Python Enum class ($defs/$ref) must resolve to type+enum."""
    p = _props(native_enum_tool)
    assert p["metric"]["type"] == "string"
    assert p["metric"]["enum"] == ["cpu", "memory", "disk"]
    assert "$ref" not in str(p)


@pytest.mark.requires("oci")
def test_12_exclusive_native():
    """Pydantic gt/lt must produce exclusiveMinimum/exclusiveMaximum."""
    p = _props(exclusive_range_tool)
    assert p["score"]["exclusiveMinimum"] == 0
    assert p["score"]["exclusiveMaximum"] == 1.0


@pytest.mark.requires("oci")
def test_13_multi_optional():
    """Multiple Optional[T] fields must all resolve to correct types."""
    p = _props(multi_optional_tool)
    assert p["name"]["type"] == "string"
    assert p["age"]["type"] == "integer"
    assert p["active"]["type"] == "boolean"
    assert "anyOf" not in str(p)


@pytest.mark.requires("oci")
def test_14_const_extra():
    """json_schema_extra const must be preserved."""
    p = _props(const_tool)
    assert p["version"]["const"] == "v1"


# ---------------------------------------------------------------------------
# Defensive checks
# ---------------------------------------------------------------------------


@pytest.mark.requires("oci")
def test_no_type_any_anywhere():
    """type: 'any' must never appear (breaks Gemini)."""
    for tool_obj in [
        enum_tool,
        range_tool,
        optional_enum_tool,
        optional_int_tool,
        pattern_tool,
        format_tool,
        length_tool,
        array_tool,
        nested_tool,
        full_mcp_tool,
        native_enum_tool,
        exclusive_range_tool,
        multi_optional_tool,
        const_tool,
    ]:
        r = _result(tool_obj)
        s = str(r.parameters)
        assert "'type': 'any'" not in s, f"{tool_obj.name} has type: 'any'"
        assert '"type": "any"' not in s, f'{tool_obj.name} has type: "any"'


@pytest.mark.requires("oci")
def test_no_refs_anywhere():
    """$ref and $defs must never remain (OCI doesn't support them)."""
    for tool_obj in [
        nested_tool,
        native_enum_tool,
        full_mcp_tool,
    ]:
        r = _result(tool_obj)
        s = str(r.parameters)
        assert "$ref" not in s, f"{tool_obj.name} has $ref"
        assert "$defs" not in s, f"{tool_obj.name} has $defs"


@pytest.mark.requires("oci")
def test_required_fields():
    """Required fields must be correct (required=no default, optional=has default)."""
    r = _result(full_mcp_tool)
    req = r.parameters.get("required", [])
    assert "metric_name" in req
    assert "duration_hours" not in req  # has default=24
    assert "output_format" not in req  # has default="json"


@pytest.mark.requires("oci")
def test_descriptions_preserved():
    """Field descriptions must survive conversion."""
    p = _props(nested_tool)
    nested = p["filter"]["properties"]
    assert nested["status"]["description"] == "Status filter"
    assert nested["count"]["description"] == "Max results"


@pytest.mark.requires("oci")
def test_backward_compat_simple_tool():
    """Simple tools without nested schemas or constraints still work."""

    class SimpleTool(BaseTool):
        name: str = "simple_tool"
        description: str = "A simple tool"

        def _run(self, query: str, count: int = 10) -> str:
            return f"Processed {query}"

    provider = GenericProvider()
    result = provider.convert_to_oci_tool(SimpleTool())

    assert result.name == "simple_tool"  # type: ignore[attr-defined]
    p = result.parameters["properties"]  # type: ignore[attr-defined]
    assert p["query"]["type"] == "string"
    assert p["count"]["type"] == "integer"


# ---------------------------------------------------------------------------
# CohereProvider tests
# ---------------------------------------------------------------------------


def _cohere_params(tool_obj):
    """Run CohereProvider.convert_to_oci_tool and return parameter_definitions."""
    provider = CohereProvider()
    result = provider.convert_to_oci_tool(tool_obj)
    return result.parameter_definitions  # type: ignore[attr-defined]


@pytest.mark.requires("oci")
def test_cohere_optional_types_resolved():
    """CohereProvider: Optional[T] fields must resolve to correct types, not 'any'."""
    params = _cohere_params(multi_optional_tool)
    assert params["name"].type == "str"
    assert params["age"].type == "int"
    assert params["active"].type == "bool"


@pytest.mark.requires("oci")
def test_cohere_enum_in_description():
    """CohereProvider: enum values must be embedded in description."""
    params = _cohere_params(enum_tool)
    desc = params["metric_name"].description
    assert "cpu" in desc
    assert "memory" in desc
    assert "disk" in desc
    assert "network" in desc


@pytest.mark.requires("oci")
def test_cohere_range_in_description():
    """CohereProvider: range constraints must be embedded in description."""
    params = _cohere_params(range_tool)
    desc = params["duration_hours"].description
    assert "min=1" in desc
    assert "max=168" in desc


@pytest.mark.requires("oci")
def test_cohere_optional_enum_resolved():
    """CohereProvider: Optional[str] + enum must resolve type and embed enum."""
    params = _cohere_params(optional_enum_tool)
    assert params["output_format"].type == "str"
    desc = params["output_format"].description
    assert "json" in desc
    assert "csv" in desc
    assert "table" in desc


@pytest.mark.requires("oci")
def test_cohere_nested_ref_resolved():
    """CohereProvider: nested $ref/$defs must be resolved."""
    params = _cohere_params(nested_tool)
    # Nested models become type "Dict" in Cohere
    assert params["filter"].type == "Dict"


@pytest.mark.requires("oci")
def test_cohere_native_enum_resolved():
    """CohereProvider: Python Enum ($defs/$ref) resolves to str with enum."""
    params = _cohere_params(native_enum_tool)
    assert params["metric"].type == "str"
    desc = params["metric"].description
    assert "cpu" in desc
    assert "memory" in desc
    assert "disk" in desc


@pytest.mark.requires("oci")
def test_cohere_no_type_any():
    """CohereProvider: type must never be 'any' for any tool."""
    for tool_obj in [
        enum_tool,
        range_tool,
        optional_enum_tool,
        optional_int_tool,
        pattern_tool,
        format_tool,
        length_tool,
        nested_tool,
        full_mcp_tool,
        native_enum_tool,
        exclusive_range_tool,
        multi_optional_tool,
        const_tool,
    ]:
        params = _cohere_params(tool_obj)
        for p_name, p_def in params.items():
            assert p_def.type != "any", f"{tool_obj.name}.{p_name} has type='any'"


@pytest.mark.requires("oci")
def test_cohere_full_mcp_combo():
    """CohereProvider: combined enum + range + optional enum all work."""
    params = _cohere_params(full_mcp_tool)
    # metric_name: enum in desc, type=str
    assert params["metric_name"].type == "str"
    assert "cpu" in params["metric_name"].description
    # duration_hours: range in desc, type=int
    assert params["duration_hours"].type == "int"
    assert "min=1" in params["duration_hours"].description
    # output_format: resolved Optional, enum in desc
    assert params["output_format"].type == "str"
    assert "json" in params["output_format"].description
