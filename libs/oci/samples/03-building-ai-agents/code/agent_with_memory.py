# Sample 03: Agent with Memory (Checkpointing)
# Demonstrates how to persist conversation state across invocations

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
def remember_preference(preference: str) -> str:
    """Remember a user's preference.

    Args:
        preference: The preference to remember
    """
    return f"I'll remember that you prefer: {preference}"


@tool
def get_recommendation(category: str) -> str:
    """Get a recommendation based on user's preferences.

    Args:
        category: The category to get a recommendation for (food, music, movies)
    """
    recommendations = {
        "food": (
            "Based on your preferences, I recommend trying the new "
            "Italian restaurant downtown."
        ),
        "music": "You might enjoy the latest album by The Weeknd.",
        "movies": "I recommend watching 'Oppenheimer' - great drama.",
    }
    return recommendations.get(category, f"No recommendations available for {category}")


def main():
    # Create checkpointer for memory persistence
    checkpointer = MemorySaver()

    # Create agent with checkpointing
    agent = create_oci_agent(
        model_id="meta.llama-4-scout-17b-16e-instruct",
        tools=[remember_preference, get_recommendation],
        compartment_id=COMPARTMENT_ID,
        service_endpoint=SERVICE_ENDPOINT,
        auth_profile=AUTH_PROFILE,
        checkpointer=checkpointer,  # Enable memory
        system_prompt="You are a personal assistant that remembers user preferences "
        "and provides personalized recommendations.",
    )

    # Conversation thread ID - same ID = same conversation
    thread_id = "user_alice_123"
    config = {"configurable": {"thread_id": thread_id}}

    # First message - set a preference
    print("Turn 1: Setting preference")
    result1 = agent.invoke(
        {"messages": [HumanMessage(content="I love Italian food and rock.")]},
        config=config,
    )
    print(f"Agent: {result1['messages'][-1].content}\n")

    # Second message - ask for a recommendation
    # The agent remembers the previous context!
    print("Turn 2: Asking for food recommendation")
    result2 = agent.invoke(
        {"messages": [HumanMessage(content="Can you recommend a restaurant?")]},
        config=config,
    )
    print(f"Agent: {result2['messages'][-1].content}\n")

    # Third message - continue the conversation
    print("Turn 3: Asking for music recommendation")
    result3 = agent.invoke(
        {"messages": [HumanMessage(content="What about music?")]},
        config=config,
    )
    print(f"Agent: {result3['messages'][-1].content}\n")

    # Different thread = different conversation
    print("New Thread: Different user")
    result_new = agent.invoke(
        {"messages": [HumanMessage(content="What do I like?")]},
        config={"configurable": {"thread_id": "user_bob_456"}},
    )
    print(f"Agent (no memory of Alice): {result_new['messages'][-1].content}")


if __name__ == "__main__":
    main()
