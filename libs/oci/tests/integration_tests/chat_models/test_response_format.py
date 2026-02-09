# Copyright (c) 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Integration tests for response_format feature with OCI Generative AI chat models.

These tests verify that the response_format parameter works correctly with real
OCI Generative AI API calls for both JSON mode and JSON schema mode.

## Prerequisites

1. **OCI Authentication**: Set up OCI authentication with security token:
   ```bash
   oci session authenticate
   ```

2. **Environment Variables**: Export the following:
   ```bash
   export OCI_COMPARTMENT_ID="ocid1.compartment.oc1..your-compartment-id"
   export OCI_REGION="us-chicago-1"  # Optional, defaults to us-chicago-1
   export OCI_AUTH_TYPE="SECURITY_TOKEN"  # Optional, defaults to SECURITY_TOKEN
   export OCI_CONFIG_PROFILE="DEFAULT"  # Optional, defaults to DEFAULT
   ```

3. **OCI Config**: Ensure `~/.oci/config` exists with DEFAULT profile

## Running the Tests

Run all integration tests:
```bash
cd libs/oci
pytest tests/integration_tests/chat_models/test_response_format.py -v
```

Run specific test:
```bash
pytest tests/integration_tests/chat_models/\
test_response_format.py::test_json_mode_meta_llama -v
```

## What These Tests Verify

1. **JSON Mode**: Models return valid JSON when using {"type": "JSON_OBJECT"}
2. **JSON Schema Mode**: Models follow specific JSON schemas when provided
3. **Multi-Vendor**: Works for both Meta Llama and Cohere models
4. **Structured Output**: with_structured_output integration works end-to-end
"""

import json
import os

import pytest
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from langchain_oci.chat_models import ChatOCIGenAI


def create_chat_model(model_id: str, response_format=None, **kwargs):
    """Create a ChatOCIGenAI instance for testing."""
    compartment_id = os.environ.get("OCI_COMPARTMENT_ID")
    if not compartment_id:
        pytest.skip("OCI_COMPARTMENT_ID not set")

    region = os.getenv("OCI_REGION", "us-chicago-1")
    endpoint = f"https://inference.generativeai.{region}.oci.oraclecloud.com"

    model_kwargs = {"temperature": 0.1, "max_tokens": 512}
    if response_format:
        model_kwargs["response_format"] = response_format

    return ChatOCIGenAI(
        model_id=model_id,
        service_endpoint=endpoint,
        compartment_id=compartment_id,
        model_kwargs=model_kwargs,
        auth_type=os.environ.get("OCI_AUTH_TYPE", "SECURITY_TOKEN"),
        auth_profile=os.environ.get("OCI_CONFIG_PROFILE", "DEFAULT"),
        auth_file_location=os.path.expanduser("~/.oci/config"),
        **kwargs,
    )


@pytest.mark.requires("oci")
@pytest.mark.parametrize(
    "model_id",
    [
        "meta.llama-3.3-70b-instruct",
        "cohere.command-r-plus-08-2024",
    ],
)
def test_json_mode_basic(model_id: str):
    """Test basic JSON mode with response_format parameter.

    This test verifies that when response_format={"type": "JSON_OBJECT"} is set,
    the model returns valid JSON output.
    """
    llm = create_chat_model(model_id)
    llm_with_json = llm.bind(response_format={"type": "JSON_OBJECT"})

    response = llm_with_json.invoke(
        [
            HumanMessage(
                content="List three colors in JSON format with a 'colors' array."
            )
        ]
    )

    # Verify response is valid JSON
    try:
        parsed = json.loads(response.content)
        assert isinstance(parsed, dict), "Response should be a JSON object"
        assert "colors" in parsed or "colour" in parsed, "Should contain colors array"
    except json.JSONDecodeError as e:
        pytest.fail(f"Response is not valid JSON: {e}\nContent: {response.content}")


@pytest.mark.requires("oci")
def test_json_mode_meta_llama():
    """Test JSON mode specifically with Meta Llama models."""
    model_id = "meta.llama-3.3-70b-instruct"
    llm = create_chat_model(model_id, response_format={"type": "JSON_OBJECT"})

    response = llm.invoke(
        [
            HumanMessage(
                content=(
                    "Create a JSON object with a person's name and age. "
                    "Name: Alice, Age: 30"
                )
            )
        ]
    )

    # Verify valid JSON
    try:
        parsed = json.loads(response.content)
        assert isinstance(parsed, dict)
        # Check for common variations in key names
        has_name = any(
            k.lower() in ["name", "person", "alice"] for k in str(parsed).lower()
        )
        has_age = "30" in str(parsed) or "age" in str(parsed).lower()
        assert has_name or has_age, f"Should contain person info: {parsed}"
    except json.JSONDecodeError as e:
        pytest.fail(f"Meta Llama JSON mode failed: {e}\nContent: {response.content}")


@pytest.mark.requires("oci")
def test_json_mode_cohere():
    """Test JSON mode specifically with Cohere models."""
    model_id = "cohere.command-r-plus-08-2024"
    llm = create_chat_model(model_id, response_format={"type": "JSON_OBJECT"})

    response = llm.invoke(
        [
            HumanMessage(
                content=(
                    "Generate a JSON object with a book title and author. "
                    "Use 'title' and 'author' as keys."
                )
            )
        ]
    )

    # Verify valid JSON
    try:
        parsed = json.loads(response.content)
        assert isinstance(parsed, dict)
        # Cohere should follow instructions closely
        assert len(parsed) >= 1, f"Should have at least one key: {parsed}"
    except json.JSONDecodeError as e:
        pytest.fail(f"Cohere JSON mode failed: {e}\nContent: {response.content}")


@pytest.mark.requires("oci")
@pytest.mark.parametrize(
    "model_id",
    [
        "meta.llama-3.3-70b-instruct",
        "cohere.command-r-plus-08-2024",
    ],
)
def test_with_structured_output_json_mode(model_id: str):
    """Test with_structured_output using json_mode method.

    This verifies the integration between response_format and LangChain's
    structured output feature using JSON mode.
    """

    class Person(BaseModel):
        """A person with name and age."""

        name: str = Field(description="The person's name")
        age: int = Field(description="The person's age")

    llm = create_chat_model(model_id)
    structured_llm = llm.with_structured_output(Person, method="json_mode")

    result = structured_llm.invoke(
        "Tell me about a person named Bob who is 25 years old."
    )

    # Verify we got a Person object
    assert isinstance(result, Person), (
        f"Should return Person object, got {type(result)}"
    )
    assert hasattr(result, "name"), "Should have name attribute"
    assert hasattr(result, "age"), "Should have age attribute"

    # Verify the content is reasonable (some models might not follow exactly)
    # Just check that we got some data
    assert result.name, "Name should not be empty"
    assert result.age > 0, "Age should be positive"


@pytest.mark.requires("oci")
@pytest.mark.parametrize(
    "model_id",
    [
        "meta.llama-3.3-70b-instruct",
        "cohere.command-r-08-2024",
    ],
)
def test_with_structured_output_json_schema(model_id: str):
    """Test with_structured_output using json_schema method.

    This verifies that JSON schema mode works with the OCI API and properly
    constrains the output to match the provided schema.

    Supports both Generic models (Meta, Mistral) and Cohere models.
    """

    class Product(BaseModel):
        """A product with details."""

        product_name: str = Field(description="Name of the product")
        price: float = Field(description="Price in USD")
        in_stock: bool = Field(description="Whether the product is in stock")

    llm = create_chat_model(model_id)
    structured_llm = llm.with_structured_output(Product, method="json_schema")

    result = structured_llm.invoke(
        "Create a product: Laptop, $999.99, available in stock"
    )

    # Verify we got a Product object with correct types
    assert isinstance(result, Product), (
        f"Should return Product object, got {type(result)}"
    )
    assert isinstance(result.product_name, str), "product_name should be string"
    assert isinstance(result.price, (int, float)), "price should be numeric"
    assert isinstance(result.in_stock, bool), "in_stock should be boolean"

    # Verify reasonable values
    assert result.product_name, "product_name should not be empty"
    assert result.price > 0, "price should be positive"


@pytest.mark.requires("oci")
def test_response_format_via_model_kwargs():
    """Test that response_format works when passed via model_kwargs.

    This tests an alternative way to set response_format at initialization time.
    """
    compartment_id = os.environ.get("OCI_COMPARTMENT_ID")
    if not compartment_id:
        pytest.skip("OCI_COMPARTMENT_ID not set")

    model_id = "meta.llama-3.3-70b-instruct"
    region = os.getenv("OCI_REGION", "us-chicago-1")
    endpoint = f"https://inference.generativeai.{region}.oci.oraclecloud.com"

    llm = ChatOCIGenAI(
        model_id=model_id,
        service_endpoint=endpoint,
        compartment_id=compartment_id,
        model_kwargs={
            "temperature": 0.1,
            "max_tokens": 512,
            "response_format": {"type": "JSON_OBJECT"},
        },
        auth_type=os.environ.get("OCI_AUTH_TYPE", "SECURITY_TOKEN"),
        auth_profile=os.environ.get("OCI_CONFIG_PROFILE", "DEFAULT"),
        auth_file_location=os.path.expanduser("~/.oci/config"),
    )

    response = llm.invoke(
        [HumanMessage(content="Create a JSON with a list of two fruits.")]
    )

    # Verify valid JSON
    try:
        parsed = json.loads(response.content)  # type: ignore
        assert isinstance(parsed, dict), "Response should be a JSON object"
    except json.JSONDecodeError as e:
        pytest.fail(
            f"model_kwargs response_format failed: {e}\nContent: {response.content}"
        )


@pytest.mark.requires("oci")
def test_json_mode_complex_nested_structure():
    """Test JSON mode with a more complex nested structure request."""
    model_id = "cohere.command-r-plus-08-2024"
    llm = create_chat_model(model_id, response_format={"type": "JSON_OBJECT"})

    response = llm.invoke(
        [
            HumanMessage(
                content="""Create a JSON object representing a company with:
        - name: "TechCorp"
        - employees: array of 2 employees, each with name and role
        - founded: 2020"""
            )
        ]
    )

    # Verify valid JSON with nested structure
    try:
        parsed = json.loads(response.content)
        assert isinstance(parsed, dict), "Response should be a JSON object"

        # Check for reasonable structure (flexible since models vary)
        assert len(parsed) >= 1, "Should have at least one top-level key"

        # Try to verify it has some nested structure
        has_nested = any(isinstance(v, (dict, list)) for v in parsed.values())
        assert has_nested or len(str(parsed)) > 50, (
            "Should have some nested structure or substantial content"
        )

    except json.JSONDecodeError as e:
        pytest.fail(f"Complex JSON failed: {e}\nContent: {response.content}")


@pytest.mark.requires("oci")
def test_response_format_class_level():
    """Test response_format set at class initialization level."""
    model_id = "meta.llama-3.3-70b-instruct"
    llm = create_chat_model(model_id, response_format={"type": "JSON_OBJECT"})

    # Should work without bind()
    response = llm.invoke(
        [HumanMessage(content="Return JSON with a single key 'status' set to 'ok'")]
    )

    # Verify valid JSON
    try:
        parsed = json.loads(response.content)
        assert isinstance(parsed, dict), "Response should be a JSON object"
    except json.JSONDecodeError as e:
        pytest.fail(
            f"Class-level response_format failed: {e}\nContent: {response.content}"
        )
