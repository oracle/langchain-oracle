# langchain-oci

This package contains the LangChain integrations with oci.

## Installation

```bash
# Base installation
pip install -U langchain-oci

# With deep research support
pip install -U langchain-oci[deep-research]

# With ADB datastore support (requires Oracle Database)
pip install -U langchain-oci langchain-oracledb
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


### 1c. Multimodal Content (Vision, PDF, Video, Audio)

`ChatOCIGenAI` supports multimodal content types including images, PDFs, video, and audio. Support varies by model:

| Model Family | Images | PDF | Video | Audio |
|--------------|--------|-----|-------|-------|
| **Google Gemini** | ✓ | ✓ | ✓ | ✓ |
| **Meta Llama Vision** | ✓ | - | - | - |
| **Cohere Vision** | ✓ | - | - | - |
| **OpenAI GPT-5.x** | ✓ | - | - | - |

<sub>**Note:** Other models may have limited or no multimodal support. Check your model's documentation.</sub>

#### Image Analysis

```python
import base64
from langchain_core.messages import HumanMessage
from langchain_oci import ChatOCIGenAI

llm = ChatOCIGenAI(
    model_id="meta.llama-3.2-90b-vision-instruct",  # Any vision model
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="MY_COMPARTMENT_ID",
)

with open("image.png", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode("utf-8")

message = HumanMessage(content=[
    {"type": "text", "text": "Describe this image"},
    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
])
response = llm.invoke([message])
```

#### PDF Document Analysis

```python
import base64
from langchain_core.messages import HumanMessage
from langchain_oci import ChatOCIGenAI

llm = ChatOCIGenAI(
    model_id="google.gemini-2.5-flash",  # Gemini supports PDF
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="MY_COMPARTMENT_ID",
)

with open("document.pdf", "rb") as f:
    pdf_b64 = base64.b64encode(f.read()).decode("utf-8")

message = HumanMessage(content=[
    {"type": "text", "text": "Summarize this PDF document"},
    {"type": "document_url", "document_url": {"url": f"data:application/pdf;base64,{pdf_b64}"}},
])
response = llm.invoke([message])
```

#### Video Analysis

```python
import base64
from langchain_core.messages import HumanMessage
from langchain_oci import ChatOCIGenAI

llm = ChatOCIGenAI(
    model_id="google.gemini-2.5-flash",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="MY_COMPARTMENT_ID",
)

with open("video.mp4", "rb") as f:
    video_b64 = base64.b64encode(f.read()).decode("utf-8")

message = HumanMessage(content=[
    {"type": "text", "text": "What happens in this video?"},
    {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{video_b64}"}},
])
response = llm.invoke([message])
```

#### Audio Analysis

```python
import base64
from langchain_core.messages import HumanMessage
from langchain_oci import ChatOCIGenAI

llm = ChatOCIGenAI(
    model_id="google.gemini-2.5-flash",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="MY_COMPARTMENT_ID",
)

with open("audio.wav", "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode("utf-8")

message = HumanMessage(content=[
    {"type": "text", "text": "Transcribe this audio"},
    {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{audio_b64}"}},
])
response = llm.invoke([message])
```

<sub>**Note:** Document, video, and audio content requires a multimodal-capable model. Check your model's documentation for supported content types.</sub>


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


## Deep Research + Datastores (Integration Points)

The deep-research integration in `langchain-oci` is built around datastore adapters (`ADB`, `OpenSearch`) and auto-generated tools (`stats`, `search`, `keyword_search`, `get_document`).

Full deep-research guide with embedded examples:
`langchain_oci/agents/deep_research/README.md`

Why this is in scope for this SDK:
- it extends OCI-LangChain integration primitives already exported by the package
- it is shipped as an optional extra (`deep-research`) rather than mandatory core
- it reuses existing model/embedding/datastore integrations already in `langchain-oci`

PR rationale (review-ready):
This belongs in `langchain-oracle/libs/oci` because it is integration code that
connects OCI services (GenAI, ADB, OpenSearch, Object Storage) to LangChain
agent abstractions. It does not add product-specific business logic; it exposes
reusable SDK primitives (`create_deep_research_agent`, datastore adapters, tool
factories), keeps heavy dependencies optional (`deep-research` extra), and is
covered by unit/integration tests under `libs/oci/tests`. In short: this is an
SDK integration layer feature, which is exactly this repository's purpose.

### Data Loading Path for ADB

For the deep-research examples in this repo, the ADB vector table is populated through these scripts:

1. `scripts/upload_research_datasets.py`: downloads MedMCQA, PubMedQA, and CUAD from Hugging Face and uploads JSON files to OCI Object Storage buckets.
2. `scripts/upload_large_datasets.py` (optional): uploads larger corpora (Wikipedia, C4, ArXiv) to OCI Object Storage.
3. `scripts/vectorize_datasets.py`: reads bucket objects, generates embeddings, and writes rows into ADB table `VECTOR_DOCUMENTS`.

### Embedding Model Used

- Default datastore embedding model in `create_datastore_tools(...)`: `cohere.embed-v4.0` via `OCIGenAIEmbeddings`.
- Same model is used in `scripts/vectorize_datasets.py` by default.
- You can override by passing `embedding_model=...` to `create_deep_research_agent(...)` or `create_datastore_tools(...)`.

### Search Implementation (ADB)

**The `ADB` class is a wrapper/adapter, NOT a replacement for OracleVS:**

- `ADB` internally uses `OracleVS` from `langchain-oracledb` for all vector operations
- `ADB` internally uses `OracleTextSplitter` from `langchain-oracledb` for document chunking
- `ADB` internally uses `OracleTextSearchRetriever` from `langchain-oracledb` for keyword search

**Document Chunking (enabled by default):**

```python
store = ADB(
    dsn="mydb_low",
    user="ADMIN",
    password="***",
    chunk_on_write=True,  # Default: True
    chunking_params={
        "split": "sentence",  # Split by sentences
        "max": 20,            # Max 20 sentences per chunk
        "normalize": "all",   # Normalize text
    }
)
```

When `chunk_on_write=True` (default), documents are automatically split into chunks using Oracle's native text splitter. Each chunk:
- Gets its own embedding
- Is stored as a separate row in the vector table
- Enables precise retrieval of relevant document sections

This is optimal for large documents (e.g., 800-page legal documents) - each page/section becomes a searchable chunk.

**Semantic Retrieval:**

Semantic retrieval is executed through datastore tools, especially `SearchTool`, which:
- routes to the best datastore
- calls `store.search_documents_with_scores(...)` on that datastore
- relies on the datastore adapter's LangChain-compatible `vectorstore` implementation

For `ADB`, the adapter exposes OracleVS through the standard datastore contract and
uses the configured query-time embedding model under the hood.

You do not need to implement an extra ADB-specific search tool class for normal usage.

### What You Need to Provide

**At Ingestion Time (one-time setup):**

1. **Load documents into ADB** using the repository scripts:
   - `scripts/upload_research_datasets.py` - Download datasets from Hugging Face
   - `scripts/vectorize_datasets.py` - Generate embeddings and populate ADB

   Or use your own ingestion pipeline that populates the OracleVS table schema (`id/embedding/text/metadata`).

2. **Ensure consistent embedding model**: Use the same embedding model for both ingestion and runtime queries (default: `cohere.embed-v4.0`).

**At Runtime (every query):**

1. **ADB connection config**: `dsn`, `user`, `password`, optional `wallet_location`
2. **OCI GenAI config**: `compartment_id`, `service_endpoint`, `auth_type`, `auth_profile`
3. **Your research prompt/query**: The question you want the agent to answer
4. **Optional diagnostics**: `OCI_AGENT_LOG_LEVEL=DEBUG` for datastore-level logs

**You do NOT need to:**
- ❌ Implement a custom ADB search class (handled by `ADB` adapter)
- ❌ Manually create search tools (auto-generated from `datastores=...`)
- ❌ Handle chunking manually (automatic with `chunk_on_write=True`)
- ❌ Import from `langchain-oracledb` directly (handled internally by `ADB`)

### What `datastore_description` Does

`datastore_description` is not indexed document content. It is store-level metadata used by the
SDK to route queries across multiple datastores.

- Each store description is embedded once at initialization.
- Query embeddings are compared against description embeddings using cosine similarity.
- The best-scoring store is selected for search tools.

Example:
- Store A description: "SRE incidents, runbooks, diagnostics"
- Store B description: "legal contracts, clauses, compliance"
- Query "timeout troubleshooting" routes toward Store A.

### Minimal ADB Integration Example

**Prerequisites:** `pip install langchain-oci langchain-oracledb`

```python
from langchain_core.messages import HumanMessage
from langchain_oci import OCIGenAIEmbeddings
from langchain_oci import create_deep_research_agent
from langchain_oci.datastores import ADB

# ADB datastore requires langchain-oracledb
store = ADB(
    dsn="mydb_low",
    user="ADMIN",
    password="***",
    wallet_location="~/.oracle-wallet/mydb",  # optional
    table_name="VECTOR_DOCUMENTS",
    datastore_description="medical QA, legal clauses, web docs",
)

embedding_model = OCIGenAIEmbeddings(
    model_id="cohere.embed-v4.0",
    compartment_id="ocid1.compartment...",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    auth_type="API_KEY",
)

agent = create_deep_research_agent(
    datastores={"research": store},
    embedding_model=embedding_model,  # explicit, same as default
    model_id="google.gemini-2.5-pro",
    compartment_id="ocid1.compartment...",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    auth_type="API_KEY",
)

result = agent.invoke(
    {"messages": [HumanMessage(content="Summarize evidence for leukocytosis treatment")]}
)
print(result["messages"][-1].content)
```


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
