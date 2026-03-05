# Sample 09: Meta Llama Provider Examples
# Demonstrates Meta-specific features: vision, parallel tools, guidance

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from langchain_oci import ChatOCIGenAI, load_image
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


def basic_meta_chat():
    """Basic chat with Meta Llama."""
    print("Basic Meta Llama Chat")
    print("=" * 50)

    llm = ChatOCIGenAI(
        auth_profile=AUTH_PROFILE,
        model_id="meta.llama-3.3-70b-instruct",
        service_endpoint=SERVICE_ENDPOINT,
        compartment_id=COMPARTMENT_ID,
    )

    response = llm.invoke("What are the key features of Python programming?")
    print(response.content)


def vision_with_llama():
    """Vision analysis with Llama 3.2."""
    print("\nVision with Llama 3.2")
    print("=" * 50)

    llm = ChatOCIGenAI(
        auth_profile=AUTH_PROFILE,
        model_id="meta.llama-3.2-90b-vision-instruct",
        service_endpoint=SERVICE_ENDPOINT,
        compartment_id=COMPARTMENT_ID,
    )

    # Check if image exists, otherwise show how it would work
    import os

    if os.path.exists("sample.jpg"):
        message = HumanMessage(
            content=[
                {"type": "text", "text": "Describe this image in detail."},
                load_image("sample.jpg"),
            ]
        )
        response = llm.invoke([message])
        print(response.content)
    else:
        print("To test vision:")
        print("1. Place an image file named 'sample.jpg' in this directory")
        print("2. Run this example again")
        print("\nCode pattern:")
        print("""
        message = HumanMessage(content=[
            {"type": "text", "text": "Describe this image."},
            load_image("sample.jpg"),
        ])
        response = llm.invoke([message])
        """)


@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    # Simulated weather data
    weather_data = {
        "new york": "72F, sunny",
        "london": "58F, cloudy",
        "tokyo": "75F, clear",
    }
    return weather_data.get(city.lower(), f"Weather data unavailable for {city}")


@tool
def get_time(city: str) -> str:
    """Get current time in a city."""
    # Simulated time data
    time_data = {
        "new york": "10:00 AM EST",
        "london": "3:00 PM GMT",
        "tokyo": "12:00 AM JST",
    }
    return time_data.get(city.lower(), f"Time data unavailable for {city}")


def parallel_tool_calls():
    """Parallel tool calling with Llama 4."""
    print("\nParallel Tool Calls (Llama 4)")
    print("=" * 50)

    llm = ChatOCIGenAI(
        auth_profile=AUTH_PROFILE,
        model_id="meta.llama-4-scout-17b-16e-instruct",
        service_endpoint=SERVICE_ENDPOINT,
        compartment_id=COMPARTMENT_ID,
    )

    # Enable parallel tool calls
    llm_with_tools = llm.bind_tools(
        [get_weather, get_time],
        parallel_tool_calls=True,
    )

    print("Query: What's the weather and time in New York and London?")
    query = "What's the weather and time in New York and London?"
    response = llm_with_tools.invoke(query)

    print(f"\nTool calls made: {len(response.tool_calls)}")
    for tc in response.tool_calls:
        print(f"  - {tc['name']}({tc['args']})")

    # Execute tools and get final response
    # (In production, you'd use an agent loop)


def tool_result_guidance():
    """Using tool_result_guidance to improve responses."""
    print("\nTool Result Guidance")
    print("=" * 50)

    # Example configuration with tool_result_guidance:
    # llm = ChatOCIGenAI(
    #     model_id="meta.llama-3.3-70b-instruct",
    #     service_endpoint=SERVICE_ENDPOINT,
    #     compartment_id=COMPARTMENT_ID,
    # )
    # llm_with_tools = llm.bind_tools(
    #     [get_weather],
    #     tool_result_guidance=True,
    #     max_sequential_tool_calls=5,
    # )

    print("With tool_result_guidance=True:")
    print("- Model receives instruction to synthesize tool results")
    print("- Prevents raw JSON output")
    print("- max_sequential_tool_calls prevents infinite loops")


def multi_image_comparison():
    """Compare multiple images with Llama 3.2 Vision."""
    print("\nMulti-Image Comparison")
    print("=" * 50)

    # Example configuration:
    # llm = ChatOCIGenAI(
    #     model_id="meta.llama-3.2-90b-vision-instruct",
    #     service_endpoint=SERVICE_ENDPOINT,
    #     compartment_id=COMPARTMENT_ID,
    # )

    print("Pattern for comparing multiple images:")
    print("""
    message = HumanMessage(content=[
        {"type": "text", "text": "Compare these two images."},
        load_image("image1.jpg"),
        load_image("image2.jpg"),
    ])
    response = llm.invoke([message])
    """)


if __name__ == "__main__":
    print("Meta Llama Provider Examples")
    print("=" * 60)

    # Uncomment to run (requires valid credentials):
    # basic_meta_chat()
    # vision_with_llama()
    # parallel_tool_calls()
    # tool_result_guidance()
    # multi_image_comparison()

    print("\nExamples are commented out - configure credentials and uncomment to run.")
