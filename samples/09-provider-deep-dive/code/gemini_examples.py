# Sample 09: Google Gemini Provider Examples
# Demonstrates Gemini-specific features: multimodal (PDF, video, audio)

import base64
import os

from langchain_core.messages import HumanMessage

from langchain_oci import ChatOCIGenAI

# Configuration - uses environment variables or defaults
COMPARTMENT_ID = os.environ.get(
    "OCI_COMPARTMENT_ID", "ocid1.compartment.oc1..your-compartment-id"
)
SERVICE_ENDPOINT = os.environ.get(
    "OCI_SERVICE_ENDPOINT",
    "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
)
AUTH_PROFILE = os.environ.get("OCI_AUTH_PROFILE", "DEFAULT")


def basic_gemini_chat():
    """Basic chat with Gemini."""
    print("Basic Gemini Chat")
    print("=" * 50)

    llm = ChatOCIGenAI(
        auth_profile=AUTH_PROFILE,
        model_id="google.gemini-2.5-flash",
        service_endpoint=SERVICE_ENDPOINT,
        compartment_id=COMPARTMENT_ID,
    )

    response = llm.invoke("Explain machine learning in simple terms.")
    print(response.content)


def pdf_processing():
    """Process PDF documents with Gemini."""
    print("\nPDF Processing with Gemini")
    print("=" * 50)

    llm = ChatOCIGenAI(
        auth_profile=AUTH_PROFILE,
        model_id="google.gemini-2.5-flash",
        service_endpoint=SERVICE_ENDPOINT,
        compartment_id=COMPARTMENT_ID,
    )

    pdf_path = "document.pdf"

    if os.path.exists(pdf_path):
        # Load and encode PDF
        with open(pdf_path, "rb") as f:
            pdf_data = base64.b64encode(f.read()).decode()

        message = HumanMessage(
            content=[
                {"type": "text", "text": "Summarize key points from this document."},
                {"type": "media", "data": pdf_data, "mime_type": "application/pdf"},
            ]
        )

        response = llm.invoke([message])
        print(response.content)
    else:
        print("To test PDF processing:")
        print(f"1. Place a PDF file at: {pdf_path}")
        print("2. Run this example again")
        print("\nCode pattern:")
        print("""
        with open("document.pdf", "rb") as f:
            pdf_data = base64.b64encode(f.read()).decode()

        message = HumanMessage(content=[
            {"type": "text", "text": "Summarize this document."},
            {"type": "media", "data": pdf_data, "mime_type": "application/pdf"},
        ])
        response = llm.invoke([message])
        """)


def video_analysis():
    """Analyze video content with Gemini."""
    print("\nVideo Analysis with Gemini")
    print("=" * 50)

    llm = ChatOCIGenAI(
        auth_profile=AUTH_PROFILE,
        model_id="google.gemini-2.5-flash",
        service_endpoint=SERVICE_ENDPOINT,
        compartment_id=COMPARTMENT_ID,
    )

    video_path = "video.mp4"

    if os.path.exists(video_path):
        with open(video_path, "rb") as f:
            video_data = base64.b64encode(f.read()).decode()

        message = HumanMessage(
            content=[
                {"type": "text", "text": "Describe what happens in this video."},
                {"type": "media", "data": video_data, "mime_type": "video/mp4"},
            ]
        )

        response = llm.invoke([message])
        print(response.content)
    else:
        print("To test video analysis:")
        print(f"1. Place a video file at: {video_path}")
        print("2. Run this example again")
        print("\nFormats: MP4, MPEG, MOV, AVI, FLV, MPG, WEBM, WMV, 3GPP")


def audio_transcription():
    """Transcribe and analyze audio with Gemini."""
    print("\nAudio Transcription with Gemini")
    print("=" * 50)

    llm = ChatOCIGenAI(
        auth_profile=AUTH_PROFILE,
        model_id="google.gemini-2.5-flash",
        service_endpoint=SERVICE_ENDPOINT,
        compartment_id=COMPARTMENT_ID,
    )

    audio_path = "audio.mp3"

    if os.path.exists(audio_path):
        with open(audio_path, "rb") as f:
            audio_data = base64.b64encode(f.read()).decode()

        message = HumanMessage(
            content=[
                {"type": "text", "text": "Transcribe and summarize this audio."},
                {"type": "media", "data": audio_data, "mime_type": "audio/mp3"},
            ]
        )

        response = llm.invoke([message])
        print(response.content)
    else:
        print("To test audio transcription:")
        print(f"1. Place an audio file at: {audio_path}")
        print("2. Run this example again")
        print("\nSupported formats: WAV, MP3, AIFF, AAC, OGG, FLAC")


def gemini_parameters():
    """Gemini-specific parameter handling."""
    print("\nGemini Parameter Handling")
    print("=" * 50)

    # Note: OCI uses max_tokens, not max_output_tokens
    # Example configuration:
    # llm = ChatOCIGenAI(
    #     model_id="google.gemini-2.5-flash",
    #     service_endpoint=SERVICE_ENDPOINT,
    #     compartment_id=COMPARTMENT_ID,
    #     model_kwargs={
    #         "max_tokens": 1024,  # Use max_tokens for OCI
    #         "temperature": 0.7,
    #     },
    # )

    print("Key differences from native Gemini SDK:")
    print("- Use 'max_tokens' instead of 'max_output_tokens'")
    print("- Provider auto-maps max_output_tokens -> max_tokens with warning")
    print("- Both parameters provided? max_tokens takes precedence")


def multimodal_comparison():
    """Compare different media types."""
    print("\nMultimodal Capability Summary")
    print("=" * 50)

    summary = """
    Gemini Multimodal Support:

    | Media Type | MIME Types | Use Case |
    |------------|------------|----------|
    | PDF | application/pdf | Document analysis |
    | Video | video/mp4, video/webm | Video understanding |
    | Audio | audio/mp3, audio/wav | Transcription, analysis |
    | Image | image/jpeg, image/png | Vision (like other models) |

    Unique to Gemini:
    - Only provider supporting PDF, video, audio natively
    - Can combine multiple media types in one request
    - Supports long-form content (e.g., hour-long videos)
    """
    print(summary)


if __name__ == "__main__":
    print("Google Gemini Provider Examples")
    print("=" * 60)

    # Uncomment to run (requires valid credentials):
    # basic_gemini_chat()
    # pdf_processing()
    # video_analysis()
    # audio_transcription()
    # gemini_parameters()
    multimodal_comparison()

    print("\nExamples are commented out - configure credentials and uncomment to run.")
