# Tutorial 08: Conversation Store Example
# Demonstrates persistent conversation memory with ChatOCIOpenAI

# Note: Requires oci-openai package
# pip install oci-openai langchain-openai langchain-oci

# Configuration - replace with your values
COMPARTMENT_ID = "ocid1.compartment.oc1..your-compartment-id"
REGION = "us-chicago-1"
MODEL = "openai.gpt-4.1"
CONVERSATION_STORE_ID = "ocid1.conversationstore.oc1..your-store-id"


def setup_client_with_store():
    """Set up ChatOCIOpenAI with conversation store."""
    from oci_openai import OciSessionAuth

    from langchain_oci import ChatOCIOpenAI

    auth = OciSessionAuth(profile_name="DEFAULT")

    client = ChatOCIOpenAI(
        auth=auth,
        compartment_id=COMPARTMENT_ID,
        region=REGION,
        model=MODEL,
        conversation_store_id=CONVERSATION_STORE_ID,
    )

    return client


def persistent_memory_demo():
    """Demonstrate persistent conversation memory."""
    print("Persistent Memory Demo")
    print("=" * 50)

    client = setup_client_with_store()

    # First message - introduce yourself
    print("User: My name is Alice and I'm a data scientist.")
    response1 = client.invoke("My name is Alice and I'm a data scientist.")
    print(f"Assistant: {response1.content}")

    # Second message - model should remember
    print("\nUser: What is my name and profession?")
    response2 = client.invoke("What is my name and profession?")
    print(f"Assistant: {response2.content}")

    # Third message - continue context
    print("\nUser: What programming languages should I learn?")
    response3 = client.invoke("What programming languages should I learn?")
    print(f"Assistant: {response3.content}")


def multi_session_demo():
    """Demonstrate memory persistence across sessions."""
    print("\nMulti-Session Demo")
    print("=" * 50)

    # Session 1
    print("--- Session 1 ---")
    client1 = setup_client_with_store()

    print("User: Remember this number: 42")
    response1 = client1.invoke("Remember this number: 42")
    print(f"Assistant: {response1.content}")

    # Simulate new session (create new client)
    print("\n--- Session 2 (new client instance) ---")
    client2 = setup_client_with_store()

    print("User: What number did I ask you to remember?")
    response2 = client2.invoke("What number did I ask you to remember?")
    print(f"Assistant: {response2.content}")


def conversation_store_info():
    """Information about conversation stores."""
    print("\nConversation Store Information")
    print("=" * 50)

    info = """
    Conversation Stores in OCI:

    1. Creating a Store:
       oci generative-ai conversation-store create \\
           --compartment-id <compartment-ocid> \\
           --display-name "My Store" \\
           --region us-chicago-1

    2. Listing Stores:
       oci generative-ai conversation-store list \\
           --compartment-id <compartment-ocid> \\
           --region us-chicago-1

    3. Getting Store Details:
       oci generative-ai conversation-store get \\
           --conversation-store-id <store-ocid> \\
           --region us-chicago-1

    4. Deleting a Store:
       oci generative-ai conversation-store delete \\
           --conversation-store-id <store-ocid> \\
           --region us-chicago-1

    Benefits:
    - Persistent memory across sessions
    - Managed by OCI (no external database needed)
    - Secure and compliant with OCI policies
    - Automatic conversation management
    """
    print(info)


if __name__ == "__main__":
    print("Conversation Store Examples")
    print("Note: Requires oci-openai package and a conversation store\n")

    # Show information
    conversation_store_info()

    # Uncomment to run (requires actual OCI setup):
    # persistent_memory_demo()
    # multi_session_demo()

    print("\nExamples are commented out - configure credentials and uncomment to run.")
