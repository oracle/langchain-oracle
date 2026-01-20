# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
# ruff: noqa: T201

"""
Example: Creating a ReAct Agent with OCI Generative AI

This example demonstrates how to create a tool-using agent
with OCI's Llama models using the create_oci_react_agent helper.

## Prerequisites

1. Install dependencies:
   ```bash
   pip install langchain-oci langgraph
   ```

2. Set up OCI authentication:
   ```bash
   oci session authenticate
   ```

3. Set environment variables:
   ```bash
   export OCI_COMPARTMENT_ID="ocid1.compartment.oc1..your-compartment-id"
   export OCI_REGION="us-chicago-1"  # or your region
   ```

## Running the Example

```bash
cd libs/oci
python examples/agents/react_agent_example.py
```
"""

import os

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from langchain_oci import create_oci_react_agent


# Define some tools
@tool
def search_products(query: str) -> str:
    """Search the product database for items matching the query."""
    # In a real application, this would query a database
    products = {
        "laptop": ["MacBook Pro - $1999", "Dell XPS 15 - $1499", "ThinkPad X1"],
        "phone": ["iPhone 15 - $999", "Galaxy S24 - $899", "Pixel 8 - $699"],
        "tablet": ["iPad Pro - $799", "Galaxy Tab S9 - $649", "Surface Pro"],
    }
    for category, items in products.items():
        if category in query.lower():
            return f"Found products: {', '.join(items)}"
    return f"No products found matching '{query}'. Try: laptop, phone, or tablet"


@tool
def get_product_details(product_name: str) -> str:
    """Get detailed information about a specific product."""
    details = {
        "macbook pro": "MacBook Pro: 14-inch, M3 chip, 16GB RAM. Rating: 4.8/5.",
        "iphone 15": "iPhone 15: 6.1-inch display, A16 chip. Rating: 4.7/5.",
        "ipad pro": "iPad Pro: 11-inch, M2 chip, 256GB. Rating: 4.9/5.",
    }
    key = product_name.lower()
    for name, detail in details.items():
        if name in key or key in name:
            return detail
    return f"Product details not found for '{product_name}'"


@tool
def check_inventory(product_name: str) -> str:
    """Check inventory status for a product."""
    # Simulated inventory check
    return f"{product_name}: 15 units in warehouse, 3 in local store."


def main() -> None:
    """Run the example agent."""
    # Get configuration from environment
    compartment_id = os.environ.get("OCI_COMPARTMENT_ID")
    if not compartment_id:
        raise ValueError("Please set OCI_COMPARTMENT_ID environment variable")

    region = os.environ.get("OCI_REGION", "us-chicago-1")
    service_endpoint = f"https://inference.generativeai.{region}.oci.oraclecloud.com"

    print("Creating OCI ReAct Agent...")
    print("  Model: meta.llama-4-scout-17b-16e-instruct")
    print(f"  Region: {region}")
    print("  Tools: search_products, get_product_details, check_inventory")
    print()

    # Create the agent - just a few lines!
    agent = create_oci_react_agent(
        model_id="meta.llama-4-scout-17b-16e-instruct",
        tools=[search_products, get_product_details, check_inventory],
        compartment_id=compartment_id,
        service_endpoint=service_endpoint,
        auth_type="SECURITY_TOKEN",
        system_prompt="""You are a helpful shopping assistant for an electronics store.
Help users find products, get details, and check inventory.
Be concise and helpful in your responses.""",
        temperature=0.3,
        max_tokens=1024,
    )

    # Example conversation
    queries = [
        "I'm looking for laptops. What do you have?",
        "Tell me more about the MacBook Pro",
        "Is it in stock?",
    ]

    print("=" * 60)
    print("Starting conversation with the agent")
    print("=" * 60)

    for query in queries:
        print(f"\nUser: {query}")
        print("-" * 40)

        result = agent.invoke({"messages": [HumanMessage(content=query)]})

        # Print the final response
        final_message = result["messages"][-1]
        print(f"Assistant: {final_message.content}")

        # Show tool calls if any were made
        tool_messages = [
            m for m in result["messages"] if type(m).__name__ == "ToolMessage"
        ]
        if tool_messages:
            print(f"  (Used {len(tool_messages)} tool(s) to answer)")

    print()
    print("=" * 60)
    print("Example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
