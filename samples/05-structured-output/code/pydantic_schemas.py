# Sample 05: Pydantic Schema Examples
# Demonstrates structured output with various schema patterns

from enum import Enum
from typing import List, Optional

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


# Simple schema
class Contact(BaseModel):
    """A contact with name and email."""

    name: str = Field(description="The person's full name")
    email: str = Field(description="The email address")
    phone: Optional[str] = Field(default=None, description="Phone number if available")


# Nested schema
class Address(BaseModel):
    """A physical address."""

    street: str = Field(description="Street address")
    city: str = Field(description="City name")
    state: str = Field(description="State or province")
    zip_code: str = Field(description="Postal code")


class Company(BaseModel):
    """A company with address."""

    name: str = Field(description="Company name")
    industry: str = Field(description="Industry sector")
    headquarters: Address = Field(description="Main office location")
    employee_count: int = Field(ge=1, description="Number of employees")


# Schema with enum
class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class ReviewAnalysis(BaseModel):
    """Analysis of a product review."""

    summary: str = Field(description="Brief summary")
    sentiment: Sentiment = Field(description="Overall sentiment")
    keywords: List[str] = Field(description="Key topics")
    rating: int = Field(ge=1, le=5, description="Inferred rating 1-5")


def main():
    llm = ChatOCIGenAI(
        auth_profile=AUTH_PROFILE,
        model_id="meta.llama-3.3-70b-instruct",
        service_endpoint=SERVICE_ENDPOINT,
        compartment_id=COMPARTMENT_ID,
    )

    # Test 1: Simple extraction
    print("Test 1: Contact Extraction")
    print("-" * 40)
    contact_llm = llm.with_structured_output(Contact)
    contact = contact_llm.invoke(
        "Extract contact: John Doe, john.doe@example.com, (555) 123-4567"
    )
    print(f"Name: {contact.name}")
    print(f"Email: {contact.email}")
    print(f"Phone: {contact.phone}")

    # Test 2: Nested structure
    print("\nTest 2: Company Extraction")
    print("-" * 40)
    company_llm = llm.with_structured_output(Company)
    company = company_llm.invoke("""
        Extract: TechCorp is a software company located at
        456 Innovation Way, Austin, TX 78701. They employ about 250 people.
    """)
    print(f"Company: {company.name}")
    print(f"Industry: {company.industry}")
    print(f"City: {company.headquarters.city}")
    print(f"Employees: {company.employee_count}")

    # Test 3: With enum
    print("\nTest 3: Review Analysis")
    print("-" * 40)
    review_llm = llm.with_structured_output(ReviewAnalysis)
    review = review_llm.invoke("""
        Analyze: "Terrible experience! The product broke after one week
        and customer service was unhelpful. Do not buy!"
    """)
    print(f"Summary: {review.summary}")
    print(f"Sentiment: {review.sentiment.value}")
    print(f"Keywords: {review.keywords}")
    print(f"Rating: {review.rating}")


if __name__ == "__main__":
    main()
