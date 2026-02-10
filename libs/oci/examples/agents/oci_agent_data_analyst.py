# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Data Analyst Agent Example.

Demonstrates OCIGenAIAgent as a data analysis assistant that can:
- Query sales data
- Calculate statistics
- Generate insights
- Create reports

Shows multi-tool execution and reasoning steps inspection.
"""
# ruff: noqa: T201

import os

from langchain_core.tools import tool

from langchain_oci import OCIGenAIAgent

# Simulated sales data
SALES_DATA = {
    "2026-01": {"revenue": 125000, "orders": 450, "returns": 12},
    "2026-02": {"revenue": 142000, "orders": 520, "returns": 8},
    "2025-12": {"revenue": 189000, "orders": 680, "returns": 15},
    "2025-11": {"revenue": 134000, "orders": 490, "returns": 11},
}

PRODUCTS_SALES = {
    "laptop": {"units": 120, "revenue": 119880},
    "headphones": {"units": 340, "revenue": 50966},
    "keyboard": {"units": 210, "revenue": 16800},
    "monitor": {"units": 85, "revenue": 29665},
}


@tool
def query_monthly_sales(month: str) -> str:
    """Query sales data for a specific month.

    Args:
        month: Month in YYYY-MM format (e.g., 2026-01).

    Returns:
        Sales data for the month.
    """
    data = SALES_DATA.get(month)
    if not data:
        available = ", ".join(sorted(SALES_DATA.keys()))
        return f"No data for {month}. Available months: {available}"

    return (
        f"Sales for {month}: "
        f"Revenue: ${data['revenue']:,}, "
        f"Orders: {data['orders']}, "
        f"Returns: {data['returns']} ({data['returns'] / data['orders'] * 100:.1f}% return rate)"
    )


@tool
def query_product_performance(product: str) -> str:
    """Query sales performance for a specific product.

    Args:
        product: Product name (e.g., laptop, headphones).

    Returns:
        Product sales performance.
    """
    data = PRODUCTS_SALES.get(product.lower())
    if not data:
        available = ", ".join(PRODUCTS_SALES.keys())
        return f"No data for '{product}'. Available products: {available}"

    avg_price = data["revenue"] / data["units"]
    return (
        f"{product.title()} performance: "
        f"Units sold: {data['units']}, "
        f"Revenue: ${data['revenue']:,}, "
        f"Avg price: ${avg_price:.2f}"
    )


@tool
def calculate_metric(expression: str) -> str:
    """Calculate a mathematical expression or metric.

    Args:
        expression: A mathematical expression (e.g., "125000 + 142000").

    Returns:
        The calculation result.
    """
    try:
        allowed = set("0123456789+-*/.(). ")
        if not all(c in allowed for c in expression):
            return "Error: Invalid characters in expression"
        result = eval(expression)  # noqa: S307
        if isinstance(result, float):
            return f"Result: {result:,.2f}"
        return f"Result: {result:,}"
    except Exception as e:
        return f"Calculation error: {e}"


@tool
def compare_periods(period1: str, period2: str) -> str:
    """Compare sales metrics between two periods.

    Args:
        period1: First month (YYYY-MM format).
        period2: Second month (YYYY-MM format).

    Returns:
        Comparison analysis.
    """
    data1 = SALES_DATA.get(period1)
    data2 = SALES_DATA.get(period2)

    if not data1 or not data2:
        return f"Cannot compare: missing data for {period1 if not data1 else period2}"

    rev_change = ((data2["revenue"] - data1["revenue"]) / data1["revenue"]) * 100
    order_change = ((data2["orders"] - data1["orders"]) / data1["orders"]) * 100

    rev_dir = "increased" if rev_change > 0 else "decreased"
    order_dir = "increased" if order_change > 0 else "decreased"

    return (
        f"Comparison {period1} vs {period2}: "
        f"Revenue {rev_dir} by {abs(rev_change):.1f}% "
        f"(${data1['revenue']:,} → ${data2['revenue']:,}). "
        f"Orders {order_dir} by {abs(order_change):.1f}% "
        f"({data1['orders']} → {data2['orders']})."
    )


@tool
def generate_insight(data_points: str) -> str:
    """Generate a business insight based on data points.

    Args:
        data_points: Comma-separated key findings to analyze.

    Returns:
        Business insight and recommendation.
    """
    return (
        f"Insight based on: {data_points}. "
        "Recommendation: Focus on high-performing products and "
        "investigate the cause of return rate variations across months."
    )


def main():
    print("=" * 70)
    print("Data Analyst Agent Demo")
    print("=" * 70)

    agent = OCIGenAIAgent(
        model_id="meta.llama-4-scout-17b-16e-instruct",
        tools=[
            query_monthly_sales,
            query_product_performance,
            calculate_metric,
            compare_periods,
            generate_insight,
        ],
        compartment_id=os.environ.get("OCI_COMPARTMENT_ID"),
        auth_type=os.environ.get("OCI_AUTH_TYPE", "API_KEY"),
        auth_profile=os.environ.get("OCI_AUTH_PROFILE", "DEFAULT"),
        system_prompt="""You are a data analyst assistant.
Help users understand sales data by querying metrics, calculating statistics,
and providing insights. Always support your conclusions with data.""",
        enable_reflexion=True,
        max_iterations=8,
    )

    # Analysis request
    query = (
        "Compare January 2026 with December 2025 sales, "
        "and tell me which product is performing best."
    )

    print(f"\nAnalysis Request: {query}")
    print("─" * 70)

    result = agent.invoke(query)

    print("\n📊 Analysis Complete")
    print(f"{'─' * 70}")
    print(f"Answer: {result.final_answer}")
    print("\n📈 Execution Summary:")
    print(f"   Iterations: {result.total_iterations}")
    print(f"   Tool Calls: {result.total_tool_calls}")
    print(f"   Confidence: {result.confidence:.2f}")

    print("\n🔍 Reasoning Steps:")
    for i, step in enumerate(result.reasoning_steps):
        print(f"\n   Step {i + 1}:")
        for exec in step.tool_executions:
            print(f"   → {exec.tool_name}: {exec.result[:60]}...")


if __name__ == "__main__":
    main()
