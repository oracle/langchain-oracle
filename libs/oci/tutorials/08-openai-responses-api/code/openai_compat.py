# Tutorial 08: OpenAI Compatibility Example
# Demonstrates ChatOCIOpenAI for OpenAI Responses API

from langchain_core.prompts import ChatPromptTemplate

# Note: Requires oci-openai package
# pip install oci-openai langchain-openai langchain-oci

# Configuration - replace with your values
COMPARTMENT_ID = "ocid1.compartment.oc1..your-compartment-id"
REGION = "us-chicago-1"
MODEL = "openai.gpt-4.1"


def setup_client():
    """Set up ChatOCIOpenAI client with session auth."""
    from oci_openai import OciSessionAuth

    from langchain_oci import ChatOCIOpenAI

    # Session auth requires: oci session authenticate --profile-name DEFAULT
    auth = OciSessionAuth(profile_name="DEFAULT")

    client = ChatOCIOpenAI(
        auth=auth,
        compartment_id=COMPARTMENT_ID,
        region=REGION,
        model=MODEL,
    )

    return client


def basic_usage():
    """Basic invocation example."""
    print("Basic Usage")
    print("=" * 50)

    client = setup_client()

    # Simple string message
    response = client.invoke("What is the capital of Japan?")
    print(f"Response: {response.content}")


def message_formats():
    """Different message format examples."""
    print("\nMessage Formats")
    print("=" * 50)

    client = setup_client()

    # Tuple format (role, content)
    messages = [
        ("system", "You are a helpful coding assistant."),
        ("human", "Write a Python function to reverse a string."),
    ]

    response = client.invoke(messages)
    print(f"Response:\n{response.content}")


def prompt_chaining():
    """Using prompt templates with ChatOCIOpenAI."""
    print("\nPrompt Chaining")
    print("=" * 50)

    client = setup_client()

    # Create a prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant that translates "
                "{input_language} to {output_language}. "
                "Only output the translation, nothing else.",
            ),
            ("human", "{input}"),
        ]
    )

    # Create chain
    chain = prompt | client

    # Invoke with variables
    response = chain.invoke(
        {
            "input_language": "English",
            "output_language": "Spanish",
            "input": "Hello, how are you today?",
        }
    )

    print(f"Translation: {response.content}")


def streaming_response():
    """Streaming example."""
    print("\nStreaming Response")
    print("=" * 50)

    client = setup_client()

    print("Response: ", end="")
    for chunk in client.stream("Tell me a short joke about programming."):
        print(chunk.content, end="", flush=True)
    print()


if __name__ == "__main__":
    print("ChatOCIOpenAI Examples")
    print("Note: Requires oci-openai package and valid OCI session\n")

    # Uncomment to run (requires actual OCI setup):
    # basic_usage()
    # message_formats()
    # prompt_chaining()
    # streaming_response()

    print("Examples are commented out - configure credentials and uncomment to run.")
    print("First authenticate: oci session authenticate --profile-name DEFAULT")
