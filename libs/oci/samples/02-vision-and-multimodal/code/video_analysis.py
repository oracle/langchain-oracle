# Sample 02: Video Analysis with Gemini
# Demonstrates how to analyze video content using Google Gemini

import base64

from langchain_core.messages import HumanMessage

from langchain_oci import ChatOCIGenAI

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


def analyze_video(video_path: str, prompt: str):
    """Analyze a video file with Gemini."""
    llm = ChatOCIGenAI(
    auth_profile=AUTH_PROFILE,
    model_id="google.gemini-2.5-flash",
        service_endpoint=SERVICE_ENDPOINT,
        compartment_id=COMPARTMENT_ID,
        model_kwargs={
            "max_tokens": 2000,  # Videos may need longer responses
        },
    )

    # Load and encode video
    with open(video_path, "rb") as f:
        video_data = base64.b64encode(f.read()).decode("utf-8")

    # Determine mime type from extension
    if video_path.endswith(".mp4"):
        mime_type = "video/mp4"
    elif video_path.endswith(".webm"):
        mime_type = "video/webm"
    elif video_path.endswith(".mov"):
        mime_type = "video/quicktime"
    else:
        mime_type = "video/mp4"  # Default

    # Create message with video using video_url format
    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {
                "type": "video_url",
                "video_url": {"url": f"data:{mime_type};base64,{video_data}"},
            },
        ]
    )

    response = llm.invoke([message])
    return response.content


def describe_video(video_path: str):
    """Get a detailed description of video content."""
    return analyze_video(
        video_path,
        "Describe what's happening in this video. "
        "Include: actions, people/objects, setting, and timeline of events.",
    )


def extract_key_moments(video_path: str):
    """Extract key moments from a video."""
    return analyze_video(
        video_path,
        "Identify and describe the key moments in this video. "
        "For each moment, provide: timestamp (if visible), what happens, "
        "and why it's significant.",
    )


def check_for_safety_issues(video_path: str):
    """Analyze video for safety/compliance issues."""
    return analyze_video(
        video_path,
        "Analyze this video for any safety hazards or compliance issues. "
        "List any concerns found with descriptions.",
    )


if __name__ == "__main__":
    # Example usage (uncomment and provide a video path):
    # video_file = "path/to/your/video.mp4"

    # Describe video
    # description = describe_video(video_file)
    # print(f"Description:\n{description}")

    # Extract key moments
    # moments = extract_key_moments(video_file)
    # print(f"Key Moments:\n{moments}")

    print("Video Analysis Example")
    print("Uncomment the code above and provide a video path to test.")
    print("Note: Video analysis can take longer due to file size.")
