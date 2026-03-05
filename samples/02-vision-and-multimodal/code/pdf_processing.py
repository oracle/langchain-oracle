# Sample 02: PDF Processing with Gemini
# Demonstrates how to analyze PDF documents using Google Gemini

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


def analyze_pdf(pdf_path: str, prompt: str):
    """Analyze a PDF document with Gemini."""
    # Create Gemini model
    llm = ChatOCIGenAI(
    auth_profile=AUTH_PROFILE,
    model_id="google.gemini-2.5-flash",
        service_endpoint=SERVICE_ENDPOINT,
        compartment_id=COMPARTMENT_ID,
    )

    # Load and encode PDF
    with open(pdf_path, "rb") as f:
        pdf_data = base64.b64encode(f.read()).decode("utf-8")

    # Create message with PDF using document_url format
    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {
                "type": "document_url",
                "document_url": {"url": f"data:application/pdf;base64,{pdf_data}"},
            },
        ]
    )

    # Get response
    response = llm.invoke([message])
    return response.content


def summarize_document(pdf_path: str):
    """Summarize a PDF document."""
    return analyze_pdf(pdf_path, "Summarize this document in 3-5 bullet points.")


def extract_key_data(pdf_path: str):
    """Extract key data from a PDF (e.g., invoice, contract)."""
    return analyze_pdf(
        pdf_path,
        "Extract the following from this document: "
        "1. All dates mentioned "
        "2. All monetary amounts "
        "3. Names of parties involved "
        "Format as a structured list.",
    )


def answer_question(pdf_path: str, question: str):
    """Answer a question about a PDF document."""
    return analyze_pdf(pdf_path, question)


if __name__ == "__main__":
    # Example usage (uncomment and provide a PDF path):
    # pdf_file = "path/to/your/document.pdf"

    # Summarize
    # summary = summarize_document(pdf_file)
    # print(f"Summary:\n{summary}")

    # Extract data
    # data = extract_key_data(pdf_file)
    # print(f"Extracted Data:\n{data}")

    # Ask a question
    # answer = answer_question(pdf_file, "What are the payment terms?")
    # print(f"Answer:\n{answer}")

    print("PDF Processing Example")
    print("Uncomment the code above and provide a PDF path to test.")
