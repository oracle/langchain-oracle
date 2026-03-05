# Sample 05: Document Classification Example
# Demonstrates using structured output for classification tasks

from enum import Enum
from typing import List

from pydantic import BaseModel, Field

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


class DocumentCategory(str, Enum):
    LEGAL = "legal"
    FINANCIAL = "financial"
    TECHNICAL = "technical"
    MARKETING = "marketing"
    HR = "hr"
    OTHER = "other"


class UrgencyLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DocumentClassification(BaseModel):
    """Classification result for a document."""

    category: DocumentCategory = Field(description="Primary document category")
    subcategory: str = Field(description="Specific subcategory within main category")
    urgency: UrgencyLevel = Field(description="How urgent is this document")
    confidence: float = Field(ge=0, le=1, description="Classification confidence 0-1")
    key_topics: List[str] = Field(description="Main topics covered in the document")
    summary: str = Field(max_length=200, description="Brief summary of the document")
    action_required: bool = Field(description="Does this require immediate action")


def classify_document(llm, document_text: str) -> DocumentClassification:
    """Classify a document using structured output."""
    classifier = llm.with_structured_output(DocumentClassification)

    prompt = f"""Classify the following document. Analyze its content to determine
the category, urgency, key topics, and whether action is required.

Document:
---
{document_text}
---

Provide a structured classification."""

    return classifier.invoke(prompt)


def main():
    llm = ChatOCIGenAI(
        auth_profile=AUTH_PROFILE,
        model_id="meta.llama-3.3-70b-instruct",
        service_endpoint=SERVICE_ENDPOINT,
        compartment_id=COMPARTMENT_ID,
    )

    # Test documents
    documents = [
        """
        URGENT: Server Outage Notice
        Our primary database server is experiencing critical failures.
        All teams must immediately halt deployments and await further notice.
        Estimated time to resolution: 4 hours.
        Contact the SRE team for updates.
        """,
        """
        Q4 2025 Financial Report
        Revenue increased by 15% compared to Q3. Operating expenses
        remained stable. Net profit margin improved to 12%.
        Recommend continued investment in R&D and marketing.
        """,
        """
        Employee Handbook Update
        Section 5.3 regarding remote work policy has been updated.
        Employees may now work remotely up to 3 days per week
        with manager approval. Please review and acknowledge.
        """,
    ]

    for i, doc in enumerate(documents, 1):
        print(f"\nDocument {i}")
        print("=" * 50)

        result = classify_document(llm, doc)

        print(f"Category: {result.category.value}")
        print(f"Subcategory: {result.subcategory}")
        print(f"Urgency: {result.urgency.value}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Key Topics: {', '.join(result.key_topics)}")
        print(f"Action Required: {result.action_required}")
        print(f"Summary: {result.summary}")


if __name__ == "__main__":
    main()
