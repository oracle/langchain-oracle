# Sample 01: Authentication Examples
# Demonstrates the 4 authentication methods for OCI Generative AI

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
MODEL_ID = "meta.llama-3.3-70b-instruct"


def example_api_key():
    """Method 1: API Key Authentication (Default)

    Uses credentials from ~/.oci/config file.
    This is the most common method for local development.
    """
    llm = ChatOCIGenAI(
        model_id=MODEL_ID,
        service_endpoint=SERVICE_ENDPOINT,
        compartment_id=COMPARTMENT_ID,
        auth_type="API_KEY",  # Optional, this is the default
        auth_profile="DEFAULT",  # Optional, uses DEFAULT profile
    )
    return llm.invoke("Hello!")


def example_security_token():
    """Method 2: Security Token (Session-Based)

    First run: oci session authenticate --profile-name MY_PROFILE
    Uses temporary session credentials.
    """
    llm = ChatOCIGenAI(
        model_id=MODEL_ID,
        service_endpoint=SERVICE_ENDPOINT,
        compartment_id=COMPARTMENT_ID,
        auth_type="SECURITY_TOKEN",
        auth_profile="MY_PROFILE",
    )
    return llm.invoke("Hello!")


def example_instance_principal():
    """Method 3: Instance Principal

    For applications running on OCI Compute instances.
    No credentials needed - uses instance metadata.
    """
    llm = ChatOCIGenAI(
        model_id=MODEL_ID,
        service_endpoint=SERVICE_ENDPOINT,
        compartment_id=COMPARTMENT_ID,
        auth_type="INSTANCE_PRINCIPAL",
    )
    return llm.invoke("Hello!")


def example_resource_principal():
    """Method 4: Resource Principal

    For OCI Functions and other OCI resources.
    No credentials needed - uses resource metadata.
    """
    llm = ChatOCIGenAI(
        model_id=MODEL_ID,
        service_endpoint=SERVICE_ENDPOINT,
        compartment_id=COMPARTMENT_ID,
        auth_type="RESOURCE_PRINCIPAL",
    )
    return llm.invoke("Hello!")


if __name__ == "__main__":
    # Try API Key authentication (default)
    print("Testing API Key authentication...")
    response = example_api_key()
    print(f"Response: {response.content}")
