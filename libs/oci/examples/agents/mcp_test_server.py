#!/usr/bin/env python
"""Simple MCP server for testing OCIGenAIAgent integration."""

from fastmcp import FastMCP

mcp = FastMCP("TestServer")


@mcp.tool()
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together.

    Args:
        a: First number
        b: Second number

    Returns:
        Sum of a and b
    """
    return a + b


@mcp.tool()
def multiply_numbers(a: int, b: int) -> int:
    """Multiply two numbers.

    Args:
        a: First number
        b: Second number

    Returns:
        Product of a and b
    """
    return a * b


@mcp.tool()
def get_stock_price(symbol: str) -> dict:
    """Get current stock price (mock data).

    Args:
        symbol: Stock ticker symbol (e.g., ORCL, GOOG)

    Returns:
        Stock price information
    """
    prices = {
        "ORCL": {"price": 178.50, "change": 2.3, "currency": "USD"},
        "GOOG": {"price": 175.20, "change": -1.1, "currency": "USD"},
        "AAPL": {"price": 225.80, "change": 0.5, "currency": "USD"},
        "MSFT": {"price": 448.90, "change": 1.8, "currency": "USD"},
    }
    symbol = symbol.upper()
    if symbol in prices:
        return {"symbol": symbol, **prices[symbol]}
    return {"symbol": symbol, "error": "Unknown symbol"}


@mcp.tool()
def search_database(query: str, limit: int = 5) -> list:
    """Search a mock database.

    Args:
        query: Search query
        limit: Maximum results to return

    Returns:
        List of matching records
    """
    # Mock database
    records = [
        {
            "id": 1,
            "name": "Oracle Cloud",
            "type": "cloud",
            "desc": "Enterprise cloud platform",
        },
        {
            "id": 2,
            "name": "MySQL",
            "type": "database",
            "desc": "Popular open-source database",
        },
        {
            "id": 3,
            "name": "Java",
            "type": "language",
            "desc": "Enterprise programming language",
        },
        {
            "id": 4,
            "name": "Python",
            "type": "language",
            "desc": "Versatile scripting language",
        },
        {
            "id": 5,
            "name": "Kubernetes",
            "type": "orchestration",
            "desc": "Container orchestration",
        },
    ]

    query_lower = query.lower()
    matches = [
        r
        for r in records
        if query_lower in r["name"].lower() or query_lower in r["desc"].lower()
    ]
    return matches[:limit]


if __name__ == "__main__":
    mcp.run()
