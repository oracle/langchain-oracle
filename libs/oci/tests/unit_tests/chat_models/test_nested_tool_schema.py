# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Unit tests for GenericProvider tool schema conversion.

Tests the fix for nested tool schemas and anyOf resolution.
"""

from typing import List, Optional

import pytest
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_oci.chat_models.providers.generic import GenericProvider


class FileInput(BaseModel):
    """Test model for nested schema."""

    name: str = Field(description="File name")
    folder_id: int = Field(description="Folder ID")


class FileInputWithOptional(BaseModel):
    """Test model with Optional fields."""

    name: str = Field(description="File name")
    description: Optional[str] = Field(default=None, description="Optional description")


class CreateFilesTool(BaseTool):
    """Tool with nested List[BaseModel] schema."""

    name: str = "create_files"
    description: str = "Create multiple files"

    def _run(self, files: List[FileInput]) -> str:
        return f"Created {len(files)} files"


class CreateFilesWithOptionalTool(BaseTool):
    """Tool with Optional fields."""

    name: str = "create_files_optional"
    description: str = "Create files with optional metadata"

    def _run(self, files: List[FileInputWithOptional]) -> str:
        return f"Created {len(files)} files"


@pytest.mark.requires("oci")
def test_nested_list_schema_resolved():
    """Test that List[BaseModel] schemas are fully resolved without $ref."""
    provider = GenericProvider()
    tool = CreateFilesTool()

    result = provider.convert_to_oci_tool(tool)

    # Verify basic structure
    assert result.name == "create_files"  # type: ignore[attr-defined]
    properties = result.parameters.get("properties", {})  # type: ignore[attr-defined]
    assert "files" in properties

    # Verify array with nested object schema
    files_schema = properties["files"]
    assert files_schema["type"] == "array"
    items_schema = files_schema["items"]
    assert items_schema["type"] == "object"

    # Verify nested properties are inlined (no $ref)
    nested_props = items_schema["properties"]
    assert "name" in nested_props
    assert "folder_id" in nested_props
    assert nested_props["name"]["type"] == "string"
    assert nested_props["folder_id"]["type"] == "integer"

    # Verify no $ref or $defs remain
    result_str = str(result.parameters)  # type: ignore[attr-defined]
    assert "$ref" not in result_str
    assert "$defs" not in result_str


@pytest.mark.requires("oci")
def test_anyof_patterns_resolved():
    """Test that anyOf patterns from Optional fields are resolved."""
    provider = GenericProvider()
    tool = CreateFilesWithOptionalTool()

    result = provider.convert_to_oci_tool(tool)

    # Get nested properties
    items_schema = result.parameters["properties"]["files"]["items"]  # type: ignore[attr-defined]
    nested_props = items_schema["properties"]

    # Verify Optional field exists and has clean type
    assert "description" in nested_props
    desc_field = nested_props["description"]
    assert desc_field["type"] == "string"
    assert desc_field.get("default") is None

    # Verify no anyOf remains
    result_str = str(result.parameters)  # type: ignore[attr-defined]
    assert "anyOf" not in result_str


@pytest.mark.requires("oci")
def test_no_type_any_in_output():
    """Test that type: 'any' never appears in output (breaks Gemini)."""
    provider = GenericProvider()
    tool = CreateFilesWithOptionalTool()

    result = provider.convert_to_oci_tool(tool)

    # Verify no type: "any" in output
    result_str = str(result.parameters)  # type: ignore[attr-defined]
    assert '"type": "any"' not in result_str
    assert "'type': 'any'" not in result_str


@pytest.mark.requires("oci")
def test_required_fields_correct():
    """Test that required fields are correctly identified."""
    provider = GenericProvider()
    tool = CreateFilesWithOptionalTool()

    result = provider.convert_to_oci_tool(tool)

    # Check top-level required
    assert "files" in result.parameters.get("required", [])  # type: ignore[attr-defined]

    # Check nested required
    items_schema = result.parameters["properties"]["files"]["items"]  # type: ignore[attr-defined]
    required = items_schema.get("required", [])
    assert "name" in required
    assert "description" not in required  # Optional field


@pytest.mark.requires("oci")
def test_descriptions_preserved():
    """Test that field descriptions are preserved."""
    provider = GenericProvider()
    tool = CreateFilesTool()

    result = provider.convert_to_oci_tool(tool)

    # Get nested properties
    items_schema = result.parameters["properties"]["files"]["items"]  # type: ignore[attr-defined]
    nested_props = items_schema["properties"]

    # Verify descriptions
    assert nested_props["name"]["description"] == "File name"
    assert nested_props["folder_id"]["description"] == "Folder ID"


@pytest.mark.requires("oci")
def test_backward_compatibility_simple_tool():
    """Test that simple tools without nested schemas still work."""

    class SimpleTool(BaseTool):
        name: str = "simple_tool"
        description: str = "A simple tool"

        def _run(self, query: str, count: int = 10) -> str:
            return f"Processed {query}"

    provider = GenericProvider()
    tool = SimpleTool()

    result = provider.convert_to_oci_tool(tool)

    assert result.name == "simple_tool"  # type: ignore[attr-defined]
    properties = result.parameters["properties"]  # type: ignore[attr-defined]
    assert "query" in properties
    assert "count" in properties
    assert properties["query"]["type"] == "string"
    assert properties["count"]["type"] == "integer"
