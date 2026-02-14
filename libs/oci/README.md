# langchain-oci

This package contains the LangChain integrations with oci.

## Installation

```bash
pip install -U langchain-oci
```
All integrations in this package assume that you have the credentials setup to connect with oci services.

---

## Quick Start

This repository includes two main integration categories:

- [OCI Generative AI](#oci-generative-ai-examples)
- [OCI Data Science (Model Deployment)](#oci-data-science-model-deployment-examples)


---

## OCI Generative AI Examples

OCI Generative AI supports two types of models:
- **On-Demand Models**: Pre-hosted foundation models.
- **DAC Models**: Models hosted on Dedicated AI Clusters (DAC), including custom models imported from Hugging Face or Object Storage

### 1a. Use a Chat Model (On-Demand)

`ChatOCIGenAI` class exposes chat models from OCI Generative AI.

```python
from langchain_oci import ChatOCIGenAI

# Using a pre-hosted on-demand model
llm = ChatOCIGenAI(
    model_id="MY_MODEL_ID",  # Pre-hosted model ID
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",  # Regional endpoint
    compartment_id="ocid1.compartment.oc1..xxxxx",  # Your compartment OCID
    model_kwargs={"max_tokens": 1024},  # Use max_completion_tokens for OpenAI models
    auth_profile="MY_AUTH_PROFILE",
    is_stream=True,
    auth_type="SECURITY_TOKEN"
)

response = llm.invoke("Sing a ballad of LangChain.")
```

### 1b. Use a Chat Model (Imported Model on DAC)

For models you've imported and deployed on a Dedicated AI Cluster:

```python
from langchain_oci import ChatOCIGenAI

# Using an imported model on Dedicated AI Cluster
llm = ChatOCIGenAI(
    model_id="ocid1.generativeaiendpoint.oc1.us-chicago-1.xxxxx",  # Endpoint OCID from your DAC
    provider="generic",  # Provider type: "cohere", "google", "meta", or "generic"
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",  # Regional endpoint
    compartment_id="ocid1.compartment.oc1..xxxxx",  # Your compartment OCID
    auth_type="SECURITY_TOKEN",  # Authentication type
    auth_profile="MY_AUTH_PROFILE",
    model_kwargs={"temperature": 0.7, "max_tokens": 500},
)

response = llm.invoke("Hello, what is your name?")
```

**Additional Arguments for Imported Models:**
- `model_id`: Use the **endpoint OCID** (starts with `ocid1.generativeaiendpoint`)
- `provider`: Provider type for your model. Available providers:
  - `"cohere"`: For Cohere models (CohereProvider)
  - `"google"`: For Google Gemini models (GeminiProvider) - automatically handles `max_output_tokens` to `max_tokens` parameter mapping
  - `"meta"`: For Meta Llama models (MetaProvider)
  - `"generic"`: Default for other models including OpenAI (GenericProvider)
  If not specified, the provider is auto-detected from the model_id prefix.
- `service_endpoint`: Use regional API endpoint (not the internal cluster URL)


### 2. Use a Completion Model
`OCIGenAI` class exposes LLMs from OCI Generative AI.

```python
from langchain_oci import OCIGenAI

llm = OCIGenAI()
llm.invoke("The meaning of life is")
```

### 3. Use an Embedding Model
`OCIGenAIEmbeddings` class exposes embeddings from OCI Generative AI.

```python
from langchain_oci import OCIGenAIEmbeddings

embeddings = OCIGenAIEmbeddings()
embeddings.embed_query("What is the meaning of life?")
```

### 3b. Use Image Embeddings (Multimodal)
`OCIGenAIEmbeddings` supports image embeddings with multimodal models like `cohere.embed-v4.0`.

```python
from langchain_oci import OCIGenAIEmbeddings

embeddings = OCIGenAIEmbeddings(
    model_id="cohere.embed-v4.0",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.compartment.oc1..xxxxx",
)

# Embed a single image (from file path, bytes, or data URI)
image_vector = embeddings.embed_image("path/to/image.png")

# Embed multiple images in a batch
image_vectors = embeddings.embed_image_batch([
    "path/to/image1.png",
    "path/to/image2.jpg",
    b"\x89PNG...",  # raw bytes
])

# Image and text embeddings share the same vector space for cross-modal retrieval
text_vector = embeddings.embed_query("a photo of a cat")
```

<sub>**Note:** Image embeddings require a multimodal model. Use `IMAGE_EMBEDDING_MODELS` to check supported models.</sub>

### 4. Use Structured Output
`ChatOCIGenAI` supports structured output.

<sub>**Note:** The default method is `function_calling`. If default method returns `None` (e.g., for Google Gemini models using GeminiProvider), try `json_schema` or `json_mode`.</sub>

```python
from langchain_oci import ChatOCIGenAI
from pydantic import BaseModel

class Joke(BaseModel):
    setup: str
    punchline: str

llm = ChatOCIGenAI()
structured_llm = llm.with_structured_output(Joke)
structured_llm.invoke("Tell me a joke about programming")
```

### 5. Use OpenAI Responses API
`ChatOCIOpenAI` supports OpenAI Responses API.

```python
from oci_openai import (
    OciSessionAuth,
)
from langchain_oci import ChatOCIOpenAI
client = ChatOCIOpenAI(
        auth=OciSessionAuth(profile_name="MY_PROFILE_NAME"),
        compartment_id="MY_COMPARTMENT_ID",
        region="us-chicago-1",
        model="openai.gpt-4.1",
        conversation_store_id="MY_CONVERSATION_STORE_ID"
    )
messages = [
        (
            "system",
            "You are a helpful translator. Translate the user sentence to French.",
        ),
        ("human", "I love programming."),
    ]
response = client.invoke(messages)
```
NOTE: By default `store` argument is set to `True` which requires passing `conversation_store_id`. You can set `store` to `False` and not pass `conversation_store_id`.
```python
from oci_openai import (
    OciSessionAuth,
)
from langchain_oci import ChatOCIOpenAI
client = ChatOCIOpenAI(
        auth=OciSessionAuth(profile_name="MY_PROFILE_NAME"),
        compartment_id="MY_COMPARTMENT_ID",
        region="us-chicago-1",
        model="openai.gpt-4.1",
        store=False
    )
messages = [
        (
            "system",
            "You are a helpful translator. Translate the user sentence to French.",
        ),
        ("human", "I love programming."),
    ]
response = client.invoke(messages)
```

### 6. Use Parallel Tool Calling (Meta/Llama 4+ models only)
Enable parallel tool calling to execute multiple tools simultaneously, improving performance for multi-tool workflows.

```python
from langchain_oci import ChatOCIGenAI

llm = ChatOCIGenAI(
    model_id="meta.llama-4-maverick-17b-128e-instruct-fp8",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="MY_COMPARTMENT_ID",
)

# Enable parallel tool calling in bind_tools
llm_with_tools = llm.bind_tools(
    [get_weather, calculate_tip, get_population],
    parallel_tool_calls=True  # Tools can execute simultaneously
)
```

<sub>**Note:** Parallel tool calling is only supported for Llama 4+ models. Llama 3.x (including 3.3) and Cohere models will raise an error if this parameter is used.</sub>


## OCI Data Science Model Deployment Examples

### 1. Use a Chat Model

You may instantiate the OCI Data Science model with the generic `ChatOCIModelDeployment` or framework specific class like `ChatOCIModelDeploymentVLLM`.

```python
from langchain_oci.chat_models import ChatOCIModelDeployment, ChatOCIModelDeploymentVLLM

# Create an instance of OCI Model Deployment Endpoint
# Replace the endpoint uri with your own
endpoint = "https://modeldeployment.<region>.oci.customer-oci.com/<ocid>/predict"

messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    ("human", "I love programming."),
]

chat = ChatOCIModelDeployment(
    endpoint=endpoint,
    streaming=True,
    max_retries=1,
    model_kwargs={
        "temperature": 0.2,
        "max_tokens": 512,
    },  # other model params...
    default_headers={
        "route": "/v1/chat/completions",
        # other request headers ...
    },
)
chat.invoke(messages)

chat_vllm = ChatOCIModelDeploymentVLLM(endpoint=endpoint)
chat_vllm.invoke(messages)
```

### 2. Use a Completion Model
You may instantiate the OCI Data Science model with `OCIModelDeploymentLLM` or `OCIModelDeploymentVLLM`.

```python
from langchain_oci.llms import OCIModelDeploymentLLM, OCIModelDeploymentVLLM

# Create an instance of OCI Model Deployment Endpoint
# Replace the endpoint uri and model name with your own
endpoint = "https://modeldeployment.<region>.oci.customer-oci.com/<ocid>/predict"

llm = OCIModelDeploymentLLM(
    endpoint=endpoint,
    model="odsc-llm",
)
llm.invoke("Who is the first president of United States?")

vllm = OCIModelDeploymentVLLM(
    endpoint=endpoint,
)
vllm.invoke("Who is the first president of United States?")
```

### 3. Use an Embedding Model
You may instantiate the OCI Data Science model with the `OCIModelDeploymentEndpointEmbeddings`.

```python
from langchain_oci.embeddings import OCIModelDeploymentEndpointEmbeddings

# Create an instance of OCI Model Deployment Endpoint
# Replace the endpoint uri with your own
endpoint = "https://modeldeployment.<region>.oci.customer-oci.com/<ocid>/predict"

embeddings = OCIModelDeploymentEndpointEmbeddings(
    endpoint=endpoint,
)

query = "Hello World!"
embeddings.embed_query(query)

documents = ["This is a sample document", "and here is another one"]
embeddings.embed_documents(documents)
```
