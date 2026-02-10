# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Customer Support Agent Example.

Demonstrates OCIGenAIAgent as a customer support assistant that can:
- Look up order status
- Check product availability
- Process returns
- Escalate to human support

Shows terminal_tools for explicit task completion.
"""
# ruff: noqa: T201

import os

from langchain_core.tools import tool

from langchain_oci import OCIGenAIAgent, TerminateEvent, ThinkEvent, ToolCompleteEvent


# Simulated database
ORDERS = {
    "ORD-001": {"status": "shipped", "eta": "Feb 12", "item": "Laptop"},
    "ORD-002": {"status": "processing", "eta": "Feb 15", "item": "Headphones"},
    "ORD-003": {"status": "delivered", "eta": None, "item": "Keyboard"},
}

PRODUCTS = {
    "laptop": {"in_stock": True, "quantity": 15, "price": 999.99},
    "headphones": {"in_stock": True, "quantity": 50, "price": 149.99},
    "monitor": {"in_stock": False, "quantity": 0, "price": 349.99},
}


@tool
def lookup_order(order_id: str) -> str:
    """Look up the status of a customer order.

    Args:
        order_id: The order ID (e.g., ORD-001).

    Returns:
        Order status information.
    """
    order = ORDERS.get(order_id.upper())
    if not order:
        return f"Order {order_id} not found. Please verify the order ID."

    status = order["status"]
    item = order["item"]
    if status == "delivered":
        return f"Order {order_id}: {item} has been delivered."
    elif status == "shipped":
        return f"Order {order_id}: {item} is shipped, ETA: {order['eta']}."
    else:
        return f"Order {order_id}: {item} is being processed, ETA: {order['eta']}."


@tool
def check_availability(product_name: str) -> str:
    """Check if a product is available in stock.

    Args:
        product_name: Name of the product to check.

    Returns:
        Availability information.
    """
    product = PRODUCTS.get(product_name.lower())
    if not product:
        return f"Product '{product_name}' not found in catalog."

    if product["in_stock"]:
        return (
            f"{product_name.title()}: In stock ({product['quantity']} available), "
            f"Price: ${product['price']}"
        )
    else:
        return f"{product_name.title()}: Out of stock. Check back later."


@tool
def initiate_return(order_id: str, reason: str) -> str:
    """Initiate a return request for an order.

    Args:
        order_id: The order ID to return.
        reason: Reason for the return.

    Returns:
        Return confirmation.
    """
    order = ORDERS.get(order_id.upper())
    if not order:
        return f"Cannot initiate return: Order {order_id} not found."

    if order["status"] != "delivered":
        return f"Cannot initiate return: Order {order_id} has not been delivered yet."

    return (
        f"Return initiated for {order_id} ({order['item']}). "
        f"Reason: {reason}. "
        f"Return label will be emailed within 24 hours. "
        f"Reference: RET-{order_id[-3:]}"
    )


@tool
def escalate_to_human(issue_summary: str) -> str:
    """Escalate the issue to a human support agent.

    Call this when the customer request cannot be handled automatically.

    Args:
        issue_summary: Brief summary of the customer's issue.

    Returns:
        Escalation confirmation.
    """
    return (
        f"Issue escalated to human support. "
        f"Summary: {issue_summary}. "
        f"A support agent will contact the customer within 2 hours. "
        f"Ticket: SUP-2026-0209"
    )


@tool
def resolve_inquiry(resolution: str) -> str:
    """Mark the customer inquiry as resolved.

    Call this when the customer's question has been fully answered.

    Args:
        resolution: Summary of how the inquiry was resolved.

    Returns:
        Resolution confirmation.
    """
    return f"Inquiry resolved: {resolution}"


def main():
    print("=" * 70)
    print("Customer Support Agent Demo")
    print("=" * 70)

    agent = OCIGenAIAgent(
        model_id="meta.llama-4-scout-17b-16e-instruct",
        tools=[
            lookup_order,
            check_availability,
            initiate_return,
            escalate_to_human,
            resolve_inquiry,
        ],
        compartment_id=os.environ.get("OCI_COMPARTMENT_ID"),
        auth_type=os.environ.get("OCI_AUTH_TYPE", "API_KEY"),
        auth_profile=os.environ.get("OCI_AUTH_PROFILE", "DEFAULT"),
        system_prompt="""You are a helpful customer support agent.
Use the available tools to help customers with their inquiries.
Always be polite and professional.
When you have fully answered the customer's question, call resolve_inquiry.
If you cannot help, escalate to human support.""",
        terminal_tools=["resolve_inquiry", "escalate_to_human"],
        max_iterations=5,
    )

    # Example inquiries
    inquiries = [
        "What's the status of my order ORD-001?",
        "Is the monitor available? I want to buy one.",
        "I want to return my keyboard from order ORD-003, it's defective.",
    ]

    for inquiry in inquiries:
        print(f"\n{'─' * 70}")
        print(f"Customer: {inquiry}")
        print("─" * 70)

        for event in agent.stream(inquiry):
            if isinstance(event, ThinkEvent) and event.tool_calls_planned > 0:
                print(f"[Agent thinking, planning {event.tool_calls_planned} action(s)]")
            elif isinstance(event, ToolCompleteEvent):
                print(f"[{event.tool_name}] {event.result}")
            elif isinstance(event, TerminateEvent):
                print(f"\nAgent: {event.final_answer}")
                print(f"(Terminated: {event.reason})")


if __name__ == "__main__":
    main()
