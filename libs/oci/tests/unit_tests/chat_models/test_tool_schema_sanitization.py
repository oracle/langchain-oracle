# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Unit tests for GenericProvider._sanitize_tool_property()."""

from langchain_oci.chat_models.providers.generic import GenericProvider


class TestSanitizeToolProperty:
    """Tests for _sanitize_tool_property schema preservation."""

    def test_simple_type_and_description(self):
        result = GenericProvider._sanitize_tool_property(
            {"type": "string", "description": "A query"}
        )
        assert result == {"type": "string", "description": "A query"}

    def test_missing_description_gets_default(self):
        result = GenericProvider._sanitize_tool_property({"type": "integer"})
        assert result["description"] == ""
        assert result["type"] == "integer"

    def test_missing_type_defaults_to_string(self):
        result = GenericProvider._sanitize_tool_property({"description": "something"})
        assert result["type"] == "string"

    def test_enum_preserved(self):
        result = GenericProvider._sanitize_tool_property(
            {
                "type": "string",
                "description": "Color choice",
                "enum": ["red", "green", "blue"],
            }
        )
        assert result["enum"] == ["red", "green", "blue"]
        assert result["type"] == "string"

    def test_format_preserved(self):
        result = GenericProvider._sanitize_tool_property(
            {
                "type": "string",
                "description": "A date",
                "format": "date-time",
            }
        )
        assert result["format"] == "date-time"

    def test_minimum_maximum_preserved(self):
        result = GenericProvider._sanitize_tool_property(
            {
                "type": "integer",
                "description": "Age",
                "minimum": 0,
                "maximum": 150,
            }
        )
        assert result["minimum"] == 0
        assert result["maximum"] == 150

    def test_exclusive_min_max_preserved(self):
        result = GenericProvider._sanitize_tool_property(
            {
                "type": "number",
                "description": "Score",
                "exclusiveMinimum": 0,
                "exclusiveMaximum": 1.0,
            }
        )
        assert result["exclusiveMinimum"] == 0
        assert result["exclusiveMaximum"] == 1.0

    def test_pattern_preserved(self):
        result = GenericProvider._sanitize_tool_property(
            {
                "type": "string",
                "description": "Email",
                "pattern": r"^[\w.]+@[\w.]+$",
            }
        )
        assert result["pattern"] == r"^[\w.]+@[\w.]+$"

    def test_min_max_length_preserved(self):
        result = GenericProvider._sanitize_tool_property(
            {
                "type": "string",
                "description": "Name",
                "minLength": 1,
                "maxLength": 100,
            }
        )
        assert result["minLength"] == 1
        assert result["maxLength"] == 100

    def test_examples_preserved(self):
        result = GenericProvider._sanitize_tool_property(
            {
                "type": "string",
                "description": "City",
                "examples": ["New York", "London"],
            }
        )
        assert result["examples"] == ["New York", "London"]

    def test_const_preserved(self):
        result = GenericProvider._sanitize_tool_property(
            {
                "type": "string",
                "description": "Version",
                "const": "v1",
            }
        )
        assert result["const"] == "v1"

    def test_default_preserved(self):
        result = GenericProvider._sanitize_tool_property(
            {
                "type": "integer",
                "description": "Limit",
                "default": 10,
            }
        )
        assert result["default"] == 10

    def test_items_preserved_and_sanitized(self):
        result = GenericProvider._sanitize_tool_property(
            {
                "type": "array",
                "description": "Tags",
                "items": {
                    "type": "string",
                    "enum": ["a", "b", "c"],
                },
                "minItems": 1,
                "maxItems": 5,
            }
        )
        assert result["type"] == "array"
        assert result["items"]["type"] == "string"
        assert result["items"]["enum"] == ["a", "b", "c"]
        assert result["minItems"] == 1
        assert result["maxItems"] == 5

    def test_nested_properties_sanitized(self):
        result = GenericProvider._sanitize_tool_property(
            {
                "type": "object",
                "description": "Filter",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["active", "inactive"],
                    },
                    "count": {
                        "type": "integer",
                        "minimum": 0,
                    },
                },
                "required": ["status"],
            }
        )
        assert result["type"] == "object"
        assert result["properties"]["status"]["enum"] == ["active", "inactive"]
        assert result["properties"]["count"]["minimum"] == 0
        assert result["required"] == ["status"]

    def test_anyof_resolved_with_enum(self):
        """anyOf (Optional[T]) is resolved and enum is preserved."""
        result = GenericProvider._sanitize_tool_property(
            {
                "anyOf": [
                    {"type": "string", "enum": ["a", "b"]},
                    {"type": "null"},
                ],
                "description": "Optional enum field",
            }
        )
        assert result["type"] == "string"
        assert result["enum"] == ["a", "b"]
        assert result["description"] == "Optional enum field"

    def test_anyof_resolved_preserves_top_level_metadata(self):
        result = GenericProvider._sanitize_tool_property(
            {
                "anyOf": [
                    {"type": "integer"},
                    {"type": "null"},
                ],
                "description": "Optional count",
                "default": 5,
                "minimum": 0,
            }
        )
        assert result["type"] == "integer"
        assert result["description"] == "Optional count"
        assert result["default"] == 5
        assert result["minimum"] == 0

    def test_anyof_all_null_falls_back_to_string(self):
        result = GenericProvider._sanitize_tool_property(
            {
                "anyOf": [{"type": "null"}],
            }
        )
        assert result["type"] == "string"

    def test_unknown_keys_stripped(self):
        result = GenericProvider._sanitize_tool_property(
            {
                "type": "string",
                "description": "test",
                "$ref": "#/defs/Foo",
                "x-custom": "ignored",
                "allOf": [{"type": "string"}],
            }
        )
        assert "$ref" not in result
        assert "x-custom" not in result
        assert "allOf" not in result
        assert result["type"] == "string"

    def test_unique_items_preserved(self):
        result = GenericProvider._sanitize_tool_property(
            {
                "type": "array",
                "description": "Unique tags",
                "items": {"type": "string"},
                "uniqueItems": True,
            }
        )
        assert result["uniqueItems"] is True

    def test_additional_properties_preserved(self):
        result = GenericProvider._sanitize_tool_property(
            {
                "type": "object",
                "description": "Config",
                "additionalProperties": False,
            }
        )
        assert result["additionalProperties"] is False


class TestSanitizeToolPropertyRoundTrip:
    """Simulating MCP tool schemas through the sanitization pipeline."""

    def test_mcp_enum_survives_pipeline(self):
        mcp_property = {
            "type": "string",
            "description": "Log level",
            "enum": ["debug", "info", "warn", "error"],
        }
        result = GenericProvider._sanitize_tool_property(mcp_property)
        assert result["enum"] == ["debug", "info", "warn", "error"]

    def test_mcp_format_survives_pipeline(self):
        mcp_property = {
            "type": "string",
            "description": "Timestamp",
            "format": "date-time",
        }
        result = GenericProvider._sanitize_tool_property(mcp_property)
        assert result["format"] == "date-time"

    def test_complex_mcp_schema(self):
        mcp_property = {
            "type": "object",
            "description": "Query parameters",
            "properties": {
                "metric_name": {
                    "type": "string",
                    "description": "Metric to query",
                    "enum": ["cpu", "memory", "disk"],
                },
                "duration_hours": {
                    "type": "integer",
                    "description": "Time window",
                    "minimum": 1,
                    "maximum": 168,
                    "default": 24,
                },
                "format": {
                    "type": "string",
                    "description": "Output format",
                    "pattern": "^(json|csv|table)$",
                },
            },
            "required": ["metric_name"],
        }
        result = GenericProvider._sanitize_tool_property(mcp_property)

        assert result["properties"]["metric_name"]["enum"] == [
            "cpu",
            "memory",
            "disk",
        ]
        assert result["properties"]["duration_hours"]["minimum"] == 1
        assert result["properties"]["duration_hours"]["maximum"] == 168
        assert result["properties"]["duration_hours"]["default"] == 24
        assert result["properties"]["format"]["pattern"] == "^(json|csv|table)$"
        assert result["required"] == ["metric_name"]
