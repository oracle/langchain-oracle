# Sample 03: Human-in-the-Loop Agent
# Demonstrates pausing agent execution for human approval

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver

from langchain_oci import create_oci_agent
import os

# Configuration - uses environment variables or defaults
COMPARTMENT_ID = os.environ.get(
    "OCI_COMPARTMENT_ID", "ocid1.compartment.oc1..your-compartment-id"
)
SERVICE_ENDPOINT = os.environ.get(
    "OCI_SERVICE_ENDPOINT",
    "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
)
AUTH_PROFILE = os.environ.get("OCI_AUTH_PROFILE", "DEFAULT")


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to a recipient.

    Args:
        to: Email recipient address
        subject: Email subject line
        body: Email body content
    """
    # In production, actually send the email
    return f"Email sent successfully to {to}"


@tool
def delete_file(filename: str) -> str:
    """Delete a file from the system.

    Args:
        filename: Name of the file to delete
    """
    # In production, actually delete the file
    return f"File '{filename}' has been deleted"


def get_user_approval(tool_name: str, args: dict) -> bool:
    """Simulate user approval (in production, use a real UI)."""
    print(f"\n⚠️  Agent wants to execute: {tool_name}")
    print(f"   Arguments: {args}")
    response = input("   Approve? (y/n): ").lower().strip()
    return response == "y"


def main():
    # Create checkpointer (required for human-in-the-loop)
    checkpointer = MemorySaver()

    # Create agent with interrupt_before to pause before tool execution
    agent = create_oci_agent(
        model_id="meta.llama-4-scout-17b-16e-instruct",
        tools=[send_email, delete_file],
        compartment_id=COMPARTMENT_ID,
        service_endpoint=SERVICE_ENDPOINT,
        checkpointer=checkpointer,
        interrupt_before=["tools"],  # Pause before executing any tool
        system_prompt="You are an assistant that can send emails and manage files. "
        "Always confirm actions with the user before proceeding.",
    )

    thread_id = "approval_thread_001"
    config = {"configurable": {"thread_id": thread_id}}

    # User requests an action
    print("User: Send an email to john@example.com about the meeting tomorrow")
    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Send an email to john@example.com saying 'Meeting at 10am'"
                )
            ]
        },
        config=config,
    )

    # Check if agent is waiting for tool execution
    last_message = result["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            approved = get_user_approval(tool_call["name"], tool_call["args"])

            if approved:
                # Continue execution
                print("\n✅ Approved! Continuing execution...")
                result = agent.invoke(None, config=config)
                print(f"\nAgent: {result['messages'][-1].content}")
            else:
                # Reject the action
                print("\n❌ Rejected! Action will not be executed.")
                # In production, you might want to tell the agent it was rejected
    else:
        print(f"\nAgent: {last_message.content}")


if __name__ == "__main__":
    main()
