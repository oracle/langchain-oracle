# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Shared utility functions for langchain-oci."""

import json
import re
import uuid
from typing import Any, Dict

from langchain_core.messages import ToolCall
from pydantic import BaseModel


class OCIUtils:
    """Utility functions for OCI Generative AI integration."""

    @staticmethod
    def is_pydantic_class(obj: Any) -> bool:
        """Check if an object is a Pydantic BaseModel subclass."""
        return isinstance(obj, type) and issubclass(obj, BaseModel)

    @staticmethod
    def remove_signature_from_tool_description(name: str, description: str) -> str:
        """
        Remove the tool signature and Args section from a tool description.

        The signature is typically prefixed to the description and followed
        by an Args section.
        """
        description = re.sub(rf"^{name}\(.*?\) -(?:> \w+? -)? ", "", description)
        description = re.sub(r"(?s)(?:\n?\n\s*?)?Args:.*$", "", description)
        return description

    @staticmethod
    def convert_oci_tool_call_to_langchain(tool_call: Any) -> ToolCall:
        """Convert an OCI tool call to a LangChain ToolCall.

        Handles both GenericProvider (uses 'arguments' as JSON string) and
        CohereProvider (uses 'parameters' as dict) tool call formats.
        """
        # Determine if this is a Generic or Cohere tool call
        has_arguments = "arguments" in getattr(tool_call, "attribute_map", {})

        if has_arguments:
            # Generic provider: arguments is a JSON string
            parsed = json.loads(tool_call.arguments)

            # If the parsed result is a string, it means the JSON was escaped
            if isinstance(parsed, str):
                try:
                    parsed = json.loads(parsed)
                except json.JSONDecodeError:
                    pass
        else:
            # Cohere provider: parameters is already a dict
            parsed = tool_call.parameters

        # Get or generate tool call ID
        if "id" in getattr(tool_call, "attribute_map", {}) and tool_call.id:
            tool_id = tool_call.id
        else:
            tool_id = uuid.uuid4().hex

        return ToolCall(
            name=tool_call.name,
            args=parsed,
            id=tool_id,
        )

    @staticmethod
    def resolve_schema_refs(schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        OCI Generative AI doesn't support $ref and $defs, so we inline all references.
        """
        defs = schema.get("$defs", {})  # OCI Generative AI doesn't support $defs

        def resolve(obj: Any) -> Any:
            if isinstance(obj, dict):
                if "$ref" in obj:
                    ref = obj["$ref"]
                    if ref.startswith("#/$defs/"):
                        key = ref.split("/")[-1]
                        return resolve(defs.get(key, obj))
                    return obj  # Cannot resolve $ref, return unchanged
                return {k: resolve(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [resolve(item) for item in obj]
            return obj

        resolved = resolve(schema)
        if isinstance(resolved, dict):
            resolved.pop("$defs", None)
        return resolved


# Prefix for custom endpoint OCIDs
CUSTOM_ENDPOINT_PREFIX = "ocid1.generativeaiendpoint"

# Mapping of JSON schema types to Python types
JSON_TO_PYTHON_TYPES = {
    "string": "str",
    "number": "float",
    "boolean": "bool",
    "integer": "int",
    "array": "List",
    "object": "Dict",
    "any": "any",
}
